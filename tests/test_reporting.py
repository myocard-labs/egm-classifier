"""Round-trip tests for the reporting layer.

reporting.write_run is the one-shot end-of-run orchestrator that builds
a typed TrainingRunRecord via egm-data, then writes both metrics.csv
and run.json. The contract this test pins:

- Both files actually land where requested.
- The on-disk run.json deserializes into a typed TrainingRunRecord via
  egm-data's load_training_run_record (proves the schema is honored).
- The per-epoch records, the BestEpoch selection, and the optional
  test block all survive the round trip with the values we wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from myocard_egm_data.records import (
    load_training_metrics,
    load_training_run_record,
)

from myocard_egm_classifier.metrics import binary_metrics
from myocard_egm_classifier.training import make_epoch_record, write_run


def _val_metrics_with(auroc: float) -> dict[str, Any]:
    """Build a minimal val_metrics dict with a controlled AUROC value.

    AUROC is rank-based: scaling the same logits doesn't change it.
    So instead of trying to "tune" AUROC via :func:`binary_metrics`,
    we construct the dict by hand. The reporting layer doesn't care
    whether the values are realistic — it just picks the max for
    ``BestEpoch``. Confusion + reliability come from a single shared
    binary_metrics call against perfect predictions, then we override
    just the AUROC scalar.
    """
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    perfect_logits = np.where(labels == 1, 10.0, -10.0).astype(np.float32)
    base = binary_metrics(perfect_logits, labels, n_bins=5)
    base["auroc"] = auroc
    return base


def test_write_run_round_trips(tmp_path: Path) -> None:
    """write_run + load_training_run_record reproduce the records we wrote."""
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    # Three epochs, AUROC rising then falling; epoch 2 should win BestEpoch.
    aurocs = [0.7, 0.9, 0.8]
    records = [
        make_epoch_record(
            epoch=i + 1,
            lr=1e-3,
            train_loss=0.5 - i * 0.1,
            val_loss=0.4 - i * 0.1,
            epoch_seconds=2.0,
            val_metrics=_val_metrics_with(auroc=a),
        )
        for i, a in enumerate(aurocs)
    ]

    csv_path, json_path = write_run(
        out_dir,
        config={"model": {"width_multiplier": 1.0}, "data": {"bank": "fake.h5"}},
        run_meta={"run_id": "test-run", "device": "cpu", "n_params": 12345},
        epoch_records=records,
        select_metric="auroc",
    )

    assert csv_path.exists()
    assert json_path.exists()

    # JSON round-trip via the typed loader.
    record = load_training_run_record(json_path)
    assert len(record.epochs) == 3
    assert record.epochs[1].epoch == 2
    assert record.best.epoch == 2  # epoch with highest AUROC
    assert record.best.metric == "auroc"
    assert record.best.value == pytest.approx(0.9)
    assert record.test is None  # not provided

    # CSV round-trip via the typed loader.
    rows = load_training_metrics(csv_path)
    assert len(rows) == 3
    assert [r.epoch for r in rows] == [1, 2, 3]


def test_write_run_includes_test_block(tmp_path: Path) -> None:
    """Passing test_loss + test_metrics populates the HeldOutTest block.

    Exercises :func:`write_run`'s scrub of non-scalar entries (the
    nested ``confusion`` dict from :func:`binary_metrics`) before the
    test_metrics dict is handed to ``build_training_run_record`` — the
    contracts schema for ``HeldOutTest.metrics`` only permits scalar
    values, while ``EpochRecord.val_metrics`` accepts the nested
    confusion block. The scrub is a temporary workaround for the
    asymmetry; the contracts fix lives upstream.
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir()

    records = [
        make_epoch_record(
            epoch=1,
            lr=1e-3,
            train_loss=0.5,
            val_loss=0.4,
            epoch_seconds=1.5,
            val_metrics=_val_metrics_with(auroc=0.85),
        )
    ]
    # binary_metrics-shaped test_metrics, including the nested
    # ``confusion`` block: write_run should scrub it before passing
    # downstream.
    test_metrics = _val_metrics_with(auroc=0.78)
    assert "confusion" in test_metrics  # invariant of binary_metrics

    _, json_path = write_run(
        out_dir,
        config={"model": {}, "data": {}},
        run_meta={"run_id": "test-run-with-test"},
        epoch_records=records,
        select_metric="auroc",
        test_loss=0.42,
        test_metrics=test_metrics,
    )

    record = load_training_run_record(json_path)
    test_block = record.test
    assert test_block is not None
    assert test_block.loss == pytest.approx(0.42)
    assert test_block.metrics is not None
    # The scalar metrics survived; nested + reliability got stripped.
    assert "auroc" in test_block.metrics
    assert test_block.metrics["auroc"] == pytest.approx(0.78)
    assert "confusion" not in test_block.metrics
    assert "reliability" not in test_block.metrics
    # The reliability list lives in its own slot.
    assert test_block.reliability is not None
    assert len(test_block.reliability) == 5
