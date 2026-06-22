"""End-to-end smoke for the training loop.

One epoch over the tiny synthetic ClassifierBank fixture on CPU. The
test is deliberately small (20 patients x 4 traces, T=64) so it runs
in a couple seconds locally and in CI; it isn't a learning-quality
check, it's a "every code path connects to the next one" check.

What's exercised:

- ``build_dataloaders`` (egm-data) accepts the tiny bank with the
  patient-aware splitter producing nonempty train/val/test splits.
- ``train`` runs one epoch, advances the optimizer, populates a
  per-epoch ``EpochRecord`` (typed), and saves a checkpoint when a
  ``checkpoint_dir`` is supplied with ``model_meta`` embedded.
- ``evaluate`` over the test split returns a sane metrics dict.
"""

from __future__ import annotations

from pathlib import Path

import torch
from myocard_egm_data.banks import ClassifierBank
from myocard_egm_data.datasets import LoaderBundle, build_dataloaders

from myocard_egm_classifier.models import MobileViT1D, default_v1_blocks
from myocard_egm_classifier.training import (
    EpochRecord,
    TrainConfig,
    evaluate,
    train,
)
from myocard_egm_classifier.training.train import _make_loss


def _build_bundle(bank: ClassifierBank) -> LoaderBundle:
    """Build a 64-sample / batch=8 / no-aug bundle from the fixture bank.

    [0.6, 0.2, 0.2] over the 20-patient fixture lands ~12/4/4 patients
    per split, every split has both classes, and val/test each yield
    multiple batches — enough for the 1-epoch smoke to produce finite
    val loss + a populated metrics dict.
    """
    return build_dataloaders(
        bank,
        input_length=64,
        batch_size=8,
        num_workers=0,
        binary=True,
        znorm=True,
        znorm_eps=1e-7,
        augment_train=False,
        max_gain=0.0,
        max_shift_frac=0.0,
        split_fractions=(0.6, 0.2, 0.2),
        split_seed=0,
        dataset_seed=0,
        pin_memory=False,
    )


def _build_small_model() -> MobileViT1D:
    """Smallest sensible v1 model (width=0.5) — keeps the 1-epoch smoke fast."""
    return MobileViT1D(
        blocks=default_v1_blocks(num_outputs=1),
        width_multiplier=0.5,
        in_channels=1,
        stochastic_depth=0.0,
    )


def test_one_epoch_smoke(tiny_classifier_bank: ClassifierBank, tmp_path: Path) -> None:
    """Train one epoch, write a checkpoint, evaluate the test split."""
    bundle = _build_bundle(tiny_classifier_bank)
    assert bundle.info["split_sizes"]["train"] > 0
    assert bundle.info["split_sizes"]["val"] > 0
    assert bundle.info["split_sizes"]["test"] > 0

    model = _build_small_model()
    device = torch.device("cpu")
    config = TrainConfig(
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        warmup_frac=0.0,
        grad_clip_norm=1.0,
        binary=True,
        pos_weight=bundle.info["pos_weight"],
        amp=False,
        log_every_n_steps=1000,  # silence the per-step bar in tests
    )

    ckpt_dir = tmp_path / "ckpt"
    history = train(
        model,
        bundle,
        device,
        config,
        checkpoint_dir=ckpt_dir,
        model_meta={"arch": "mobilevit_1d_v1", "width_multiplier": 0.5},
    )

    # History has one EpochRecord, and it's the typed model.
    assert len(history["epochs"]) == 1
    rec: EpochRecord = history["epochs"][0]
    assert rec.epoch == 1
    assert rec.val_loss is not None

    # Checkpoint landed with the model_meta the trainer was handed.
    ckpt_path = ckpt_dir / "best.pt"
    assert ckpt_path.is_file()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["model_meta"]["arch"] == "mobilevit_1d_v1"

    # Eval over the test split runs and returns the metrics bundle.
    loss_fn = _make_loss(config, device)
    test_loss, test_metrics = evaluate(model, bundle.test, loss_fn, device, config)
    assert isinstance(test_loss, float)
    assert "auroc" in test_metrics
    assert "reliability" in test_metrics
