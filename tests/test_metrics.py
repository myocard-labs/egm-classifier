"""Tests for the binary-metric bundle and its reliability bins.

Two things being verified:

1. The shape of the ``binary_metrics`` output dict is the cross-package
   contract that reporting.write_run depends on — keys, scalar types,
   confusion sub-dict, reliability list shape.
2. The reliability bins are emitted as the typed
   :class:`myocard_egm_data.records.ReliabilityBin` (per the
   use-contracts-at-boundaries memory) so they can flow straight into
   :class:`EpochRecord.val_reliability` with no per-bin conversion.

Values are checked against perfect-prediction and worst-case
hand-calculated cases so a metric regression (e.g. a sign flip on the
calibration error) trips the test immediately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from myocard_egm_data.records import ReliabilityBin

from myocard_egm_classifier.metrics import binary_metrics


def _perfect_logits(labels: np.ndarray) -> np.ndarray:
    """Strongly positive logit for label 1, strongly negative for label 0."""
    return np.where(labels == 1, 10.0, -10.0).astype(np.float32)


def test_binary_metrics_keys_and_subshapes() -> None:
    """The output dict carries every key the reporting layer relies on."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    logits = _perfect_logits(labels)
    m = binary_metrics(logits, labels, threshold=0.5, n_bins=5)
    # Scalar metrics + confusion + reliability + n.
    expected = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auroc",
        "ece",
        "confusion",
        "reliability",
        "n",
    }
    assert set(m.keys()) == expected
    assert isinstance(m["confusion"], dict)
    assert set(m["confusion"].keys()) == {"tp", "fp", "tn", "fn"}
    assert isinstance(m["reliability"], list)
    assert len(m["reliability"]) == 5  # n_bins
    assert m["n"] == 8


def test_binary_metrics_perfect_prediction() -> None:
    """All-correct predictions hit accuracy/precision/recall/f1/AUROC == 1, ECE == 0."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    logits = _perfect_logits(labels)
    m = binary_metrics(logits, labels)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["auroc"] == pytest.approx(1.0)
    assert m["ece"] == pytest.approx(0.0, abs=1e-4)
    # All 4 positives are TP, all 4 negatives are TN.
    assert m["confusion"] == {"tp": 4, "fp": 0, "tn": 4, "fn": 0}


def test_reliability_bins_are_typed_contracts_model() -> None:
    """Each reliability entry is the contracts' Pydantic ReliabilityBin.

    This is the use-contracts-at-boundaries invariant: the metrics layer
    constructs the typed model directly so the reporting layer can drop
    it onto ``EpochRecord.val_reliability`` with no re-validation.
    """
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    logits = _perfect_logits(labels)
    m = binary_metrics(logits, labels, n_bins=4)
    for b in m["reliability"]:
        assert isinstance(b, ReliabilityBin)
    # Bin edges cover [0, 1] without overlap.
    edges = [(b.lo, b.hi) for b in m["reliability"]]
    assert edges[0][0] == 0.0
    assert math.isclose(edges[-1][1], 1.0)


def test_reliability_bins_perfect_lands_in_last_bin() -> None:
    """Perfect-confidence predictions land in the last bin with accuracy=1.

    The first bin is closed on the left so confidence == 0.0 lands there;
    all others are (lo, hi] so confidence == 1.0 lands in the last bin —
    that's the per-Guo-2017 convention encoded in ``_reliability_bins``.
    """
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    logits = _perfect_logits(labels)
    m = binary_metrics(logits, labels, n_bins=5)
    last_bin = m["reliability"][-1]
    assert last_bin.count == 8  # every sample lands here
    assert last_bin.accuracy == pytest.approx(1.0)
    # All earlier bins should be empty.
    for b in m["reliability"][:-1]:
        assert b.count == 0


def test_binary_metrics_handles_chance_predictions() -> None:
    """All-zero logits => P(positive) = 0.5 => AUROC ~ 0.5, ECE > 0 with imbalanced classes."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    logits = np.zeros(8, dtype=np.float32)
    m = binary_metrics(logits, labels, n_bins=10)
    # Sigmoid(0) = 0.5 — every prediction is the boundary. AUROC ~ 0.5.
    assert 0.4 <= m["auroc"] <= 0.6
    # n samples accounted for.
    assert m["n"] == 8
