"""Training loop for the 1D MobileViT EGM classifier (design doc Section 12).

Recipe (single-activation v1):

- Loss: BCE-with-logits (single-logit head), with ``pos_weight`` to counter
  the typical fibrotic/healthy class imbalance. The multi-class path uses
  cross-entropy instead.
- Optimizer: AdamW, lr 3e-4, weight_decay 0.05, betas (0.9, 0.999).
- Schedule: linear warmup over 5% of steps, then cosine decay to 0.
- Stochastic depth + dropout live inside the model; augmentation (gain,
  time-shift, baked-in noise) lives in :class:`TraceTransform` upstream.

Validation uses AUROC as the checkpoint-selection metric (threshold-free,
robust under class imbalance) and additionally reports accuracy, F1, and
ECE each epoch.

Per-epoch state (loss, lr, val metrics, reliability bins) is materialized
as the Pydantic :class:`EpochRecord` from ``myocard-egm-data.records``,
so the on-disk run.json shape is the cross-package contract (no internal
schema gymnastics).
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from myocard_egm_classifier.data.datasets import LoaderBundle
from myocard_egm_classifier.inference_helpers import collect_logits
from myocard_egm_classifier.metrics import binary_metrics
from myocard_egm_classifier.training.reporting import EpochRecord, make_epoch_record


@dataclass
class TrainConfig:
    """Hyperparameters for a training run (design doc Section 12 defaults)."""

    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_frac: float = 0.05  # linear warmup over 5% of total steps
    min_lr: float = 0.0  # cosine decays to 0
    grad_clip_norm: float | None = 1.0
    binary: bool = True  # single-logit BCE vs multi-class CE
    pos_weight: float | None = None  # BCE positive-class weight
    amp: bool = False  # mixed precision (set True on GPU)
    amp_dtype: str = "bf16"  # "bf16" (Ampere+) or "fp16"
    log_every_n_steps: int = 20
    select_metric: str = "auroc"  # val metric to checkpoint on (maximize)


def _amp_dtype_from_str(name: str) -> torch.dtype:
    """Translate a config string to the torch autocast dtype."""
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def cosine_warmup_lr(
    step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float = 0.0,
) -> float:
    """Compute the LR for ``step`` under linear warmup then cosine decay to ``min_lr``."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Apply ``lr`` to every parameter group of ``optimizer`` in place."""
    for group in optimizer.param_groups:
        group["lr"] = lr


def _make_loss(config: TrainConfig, device: torch.device) -> nn.Module:
    """Build the loss function for the configured task (BCE vs CE)."""
    if config.binary:
        pw = (
            torch.tensor([config.pos_weight], dtype=torch.float32, device=device)
            if config.pos_weight is not None
            else None
        )
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    return nn.CrossEntropyLoss()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainConfig,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    total_epochs: int,
    global_step: int,
    total_steps: int,
    warmup_steps: int,
) -> tuple[float, float, int]:
    """Train one epoch; returns ``(avg_loss, last_lr, new_global_step)``.

    Advances the dataset's epoch-aware RNG stream via ``set_epoch`` if
    the underlying ``EGMTraceDataset`` supports it (the egm-data dataset
    does), so augmentation seeds vary per epoch.
    """
    model.train()
    # Advance augmentation RNG stream for this epoch (if supported).
    ds = loader.dataset
    if hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)

    running_loss = 0.0
    n_batches = 0
    lr = config.lr
    amp_dtype = _amp_dtype_from_str(config.amp_dtype)
    bar = tqdm(loader, desc=f"epoch {epoch + 1}/{total_epochs}", leave=False)
    for signals, targets in bar:
        signals = signals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        lr = cosine_warmup_lr(global_step, total_steps, warmup_steps, config.lr, config.min_lr)
        _set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=config.amp):
            logits = model(signals)  # [B, num_outputs]
            loss = loss_fn(logits, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()

        running_loss += float(loss.detach().item())
        n_batches += 1
        global_step += 1
        if global_step % config.log_every_n_steps == 0:
            bar.set_postfix(loss=f"{running_loss / n_batches:.3f}", lr=f"{lr:.2e}")

    return running_loss / max(1, n_batches), lr, global_step


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainConfig,
) -> tuple[float, dict[str, Any]]:
    """Evaluate over a loader; returns ``(avg_loss, metrics_dict)``."""
    model.eval()
    logits, labels = collect_logits(model, loader, device)
    if logits.shape[0] == 0:
        return float("nan"), {}
    logits_t = torch.from_numpy(logits).to(device)
    if config.binary:
        targets_t = torch.from_numpy(labels.astype(np.float32)).reshape(-1, 1).to(device)
    else:
        targets_t = torch.from_numpy(labels.astype(np.int64)).to(device)
    loss = float(loss_fn(logits_t, targets_t).item())
    metrics = binary_metrics(logits, labels) if config.binary else {}
    return loss, metrics


def _fmt_metrics(loss: float, m: dict[str, Any]) -> str:
    """One-line human-readable summary of (loss, metrics) for the per-epoch log."""
    if not m:
        return f"loss={loss:.3f}"
    return (
        f"loss={loss:.3f} auroc={m['auroc']:.3f} acc={m['accuracy']:.3f} "
        f"f1={m['f1']:.3f} ece={m['ece']:.3f}"
    )


def train(
    model: nn.Module,
    loaders: LoaderBundle,
    device: torch.device,
    config: TrainConfig,
    checkpoint_dir: Path | None = None,
    model_meta: dict[str, Any] | None = None,
    training_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full training loop; returns a history dict and writes ``best.pt``.

    ``model_meta`` (architecture args + data info) is embedded in the
    checkpoint so eval/export can rebuild the model without re-deriving
    it. ``training_provenance`` (run_id, produced_model_id,
    trained_on_bank_id, run_name) is embedded too, so export + eval can
    read the stable cross-artifact ids without a sibling run.json.
    ``history["epochs"]`` is a list of Pydantic :class:`EpochRecord`
    instances ready for :func:`reporting.write_run`.
    """
    model.to(device)
    loss_fn = _make_loss(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    use_scaler = config.amp and config.amp_dtype == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    steps_per_epoch = len(loaders.train)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = max(1, round(total_steps * config.warmup_frac))

    history: dict[str, list[Any]] = {
        "train_loss": [],
        "val_loss": [],
        "val_metrics": [],
        "epochs": [],
    }
    best_score = -math.inf
    global_step = 0

    print(
        f"Training {config.epochs} epochs ({steps_per_epoch} steps/epoch, "
        f"{total_steps} total). lr={config.lr:.1e} wd={config.weight_decay} "
        f"warmup={warmup_steps} steps, select on val {config.select_metric}."
    )

    for epoch in range(config.epochs):
        t0 = time.time()
        train_loss, last_lr, global_step = train_one_epoch(
            model,
            loaders.train,
            optimizer,
            loss_fn,
            device,
            config,
            scaler,
            epoch,
            config.epochs,
            global_step,
            total_steps,
            warmup_steps,
        )
        val_loss, val_metrics = evaluate(model, loaders.val, loss_fn, device, config)
        elapsed = time.time() - t0

        record: EpochRecord = make_epoch_record(
            epoch=epoch + 1,
            lr=last_lr,
            train_loss=train_loss,
            val_loss=val_loss,
            epoch_seconds=elapsed,
            val_metrics=val_metrics,
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)
        history["epochs"].append(record)

        print(
            f"[epoch {epoch + 1:3d}/{config.epochs}] "
            f"train loss={train_loss:.3f}  val {_fmt_metrics(val_loss, val_metrics)}  "
            f"({elapsed:.1f}s)"
        )

        score = val_metrics.get(config.select_metric, -val_loss) if val_metrics else -val_loss
        if checkpoint_dir is not None and score is not None and score > best_score:
            best_score = score
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_metrics": {k: v for k, v in val_metrics.items() if k != "reliability"},
                "train_config": asdict(config),
                "model_meta": model_meta or {},
                "training_provenance": training_provenance or {},
            }
            torch.save(ckpt, checkpoint_dir / "best.pt")
            print(f"  -> new best ({config.select_metric}={best_score:.3f}), saved best.pt")

    return history
