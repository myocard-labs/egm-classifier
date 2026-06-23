"""CLI: train the 1D MobileViT EGM classifier from a YAML config.

Wired as the ``egm-class-train`` console script.

Usage
-----
::

    egm-class-train CONFIG.yaml [--bank PATH] [--epochs N] [--width F]
                                [--batch-size N] [--checkpoint-dir PATH]
                                [--device cuda|cpu] [--seed N] [--no-augment]

Examples
--------
::

    egm-class-train examples/v1_baseline.yaml
    egm-class-train examples/v1_small.yaml --epochs 5 --device cpu
    egm-class-train examples/v1_baseline.yaml --width 0.5 --bank banks/clean_v1.h5

The CLI consumes a ClassifierBank HDF5 produced by either the synthetic
producer (``synthetic-egm-pipeline``) or the IAFDB producer
(``iafdb-pipeline``). Label policy was applied at conversion time — this
CLI does not re-threshold. At end-of-run, the trainer writes the best
checkpoint plus the typed ``run.json`` + ``metrics.csv`` interchange
files (owned by ``myocard-egm-data``'s records layer) into
``output.checkpoint_dir`` so the viewer / paper-figure notebooks can
re-render the run without any additional metadata.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from myocard_egm_data.banks import load_classifier_bank
from myocard_egm_data.datasets import build_dataloaders

from myocard_egm_classifier.cli._common import (
    ConfigError,
    load_yaml,
    select_device,
    set_seed,
)
from myocard_egm_classifier.cli._train_config import (
    TrainCLIOverrides,
    TrainExperimentConfig,
    apply_train_overrides,
    build_model_from_config,
    build_train_config,
    experiment_config_to_dict,
    loader_kwargs_from_config,
    model_meta_from_config,
    to_train_runtime_config,
)
from myocard_egm_classifier.models import count_parameters
from myocard_egm_classifier.training import evaluate, train, write_run
from myocard_egm_classifier.training.train import _make_loss


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egm-class-train",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=Path, help="Path to the experiment YAML config.")
    p.add_argument("--bank", type=Path, default=None, help="Override data.bank.")
    p.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    p.add_argument("--width", type=float, default=None, help="Override model.width_multiplier.")
    p.add_argument("--batch-size", type=int, default=None, help="Override data.batch_size.")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Override output.checkpoint_dir.",
    )
    p.add_argument("--device", default=None, help='Override device ("cuda", "cpu").')
    p.add_argument("--seed", type=int, default=None, help="Override train.seed.")
    p.add_argument("--no-augment", action="store_true", help="Disable train-split augmentation.")
    return p


def _overrides_from_args(args: argparse.Namespace) -> TrainCLIOverrides:
    return TrainCLIOverrides(
        bank=args.bank,
        epochs=args.epochs,
        width=args.width,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        no_augment=args.no_augment,
    )


def _git_sha() -> str | None:
    """Best-effort: read ``git rev-parse --short HEAD`` for the run record.

    Returns ``None`` outside a git checkout or if git isn't installed —
    the trainer still runs; the run.json just won't carry a SHA.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _run_meta(
    *,
    cfg: TrainExperimentConfig,
    device_str: str,
    n_params: int,
    started_utc: str,
    ended_utc: str,
    bundle_info: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the ``run`` block for the TrainingRunRecord.

    Stamps the run identity (UUID, host, git SHA, model version, wall
    times) plus the data-side info dict from ``build_dataloaders`` so the
    on-disk record carries the full picture without the viewer needing
    a sibling sidecar.
    """
    assert cfg.data.bank is not None  # validated upstream by build_train_config
    return {
        "run_id": str(uuid.uuid4()),
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "model_version": "egm_classifier_phase1_mobilevit_1d",
        "training_started_utc": started_utc,
        "training_ended_utc": ended_utc,
        "device": device_str,
        "n_params": n_params,
        "select_metric": cfg.train.select_metric,
        "task": "binary" if cfg.model.is_binary else "multiclass",
        "bank_path": str(cfg.data.bank),
        "description": cfg.output.description,
        **bundle_info,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        doc = load_yaml(args.config)
        cfg = apply_train_overrides(build_train_config(doc), _overrides_from_args(args))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    set_seed(cfg.train.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")

    # Load the ClassifierBank from disk; egm-data has already enforced
    # that every trace has label_truth + patient_id by the time the
    # bank was written.
    assert cfg.data.bank is not None  # build_train_config rejects None
    try:
        bank = load_classifier_bank(cfg.data.bank)
    except FileNotFoundError as exc:
        print(f"ERROR reading bank: {exc}", file=sys.stderr)
        return 1

    try:
        bundle = build_dataloaders(bank, **loader_kwargs_from_config(cfg))
    except ValueError as exc:
        # build_dataloaders raises ValueError for empty banks / split mismatches;
        # surface the message verbatim so the user sees the actual problem.
        print(f"Failed to set up data loaders: {exc}", file=sys.stderr)
        return 1

    info = bundle.info
    print(
        f"Bank: {cfg.data.bank}\n"
        f"  {info['n_traces']} traces / {info['n_patients']} patients, "
        f"T={info['input_length']} (bank n_samples={info['n_samples']})\n"
        f"  split traces (train/val/test): {info['split_sizes']}\n"
        f"  split patients                : {info['split_patients']}\n"
        f"  train class counts: {info['train_class_counts']}  "
        f"pos_weight={info['pos_weight']:.3f}"
    )

    model = build_model_from_config(cfg.model)
    n_params = count_parameters(model)
    print(
        f"Model: 1D MobileViT (width={cfg.model.width_multiplier}, "
        f"num_classes={cfg.model.num_classes}) — {n_params / 1e6:.3f}M params"
    )

    started_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()
    runtime_cfg = to_train_runtime_config(cfg, pos_weight=info["pos_weight"])
    history = train(
        model,
        bundle,
        device,
        runtime_cfg,
        checkpoint_dir=cfg.output.checkpoint_dir,
        model_meta=model_meta_from_config(cfg.model),
    )

    # Final held-out test metrics (best checkpoint would be re-loaded for
    # a publication number; here we report the end-of-training model).
    loss_fn = _make_loss(runtime_cfg, device)
    test_loss, test_metrics = evaluate(model, bundle.test, loss_fn, device, runtime_cfg)
    if test_metrics:
        print(
            f"\nTest: loss={test_loss:.3f} auroc={test_metrics['auroc']:.3f} "
            f"acc={test_metrics['accuracy']:.3f} f1={test_metrics['f1']:.3f} "
            f"ece={test_metrics['ece']:.3f}"
        )
    ended_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()

    run_meta = _run_meta(
        cfg=cfg,
        device_str=str(device),
        n_params=n_params,
        started_utc=started_utc,
        ended_utc=ended_utc,
        bundle_info=info,
    )
    csv_path, json_path = write_run(
        cfg.output.checkpoint_dir,
        config=experiment_config_to_dict(cfg),
        run_meta=run_meta,
        epoch_records=history["epochs"],
        select_metric=cfg.train.select_metric,
        test_loss=test_loss if test_metrics else None,
        test_metrics=test_metrics or None,
    )
    print(f"Wrote {csv_path.name} + {json_path.name} to {cfg.output.checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
