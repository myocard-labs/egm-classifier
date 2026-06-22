"""Classification + calibration metrics, computed via torchmetrics.

The single public function is :func:`binary_metrics` — it accepts raw
single-logit model outputs and ground-truth labels (numpy arrays from
:func:`collect_logits`) and returns a dict of scalar metrics + the
reliability bins for the calibration diagram. The shape of that dict
is the cross-package contract that the trainer's per-epoch state and
the reporting layer's :class:`EpochRecord` are built around.

We use ``torchmetrics`` primitives rather than hand-rolling sigmoid +
rank-based AUROC + ECE: the implementations are well-tested,
hardware-aware, and shrink the metrics module by an order of magnitude.
The trade-off is one additional runtime dep, which we already would
have pulled in transitively the moment we wanted GPU-batched
evaluation anyway.

All metrics are computed CPU-side after the trainer pulls logits via
:func:`collect_logits` (which does its own ``.cpu().numpy()``).
:func:`binary_metrics` then converts back to torch tensors at the
boundary so it can hand them to torchmetrics.

Reliability bins are emitted as instances of the contracts'
:class:`myocard_egm_data.records.ReliabilityBin` Pydantic model
(re-exported from :mod:`myocard_egm_contracts`). egm-classifier already
depends on egm-data + egm-contracts at runtime, so the metrics layer
constructs the typed model directly rather than handing the reporting
layer a dict that would need round-trip validation.

Reference: Guo C et al. "On Calibration of Modern Neural Networks." ICML
2017. arxiv 1706.04599 (reliability diagrams + ECE).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from myocard_egm_data.records import ReliabilityBin
from numpy.typing import NDArray
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryStatScores,
)


def binary_metrics(
    logits: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute the full binary metric bundle from single-logit outputs.

    Returns a dict with the well-known scalar keys
    (``accuracy``/``precision``/``recall``/``f1``/``auroc``/``ece``),
    the confusion-count subdict, the reliability table, and the sample
    count. The trainer's reporting path projects the scalars into the
    Pydantic ``EpochRecord.val_metrics`` field; the reliability list
    fills ``EpochRecord.val_reliability`` after a per-bin conversion.

    Parameters
    ----------
    logits
        ``[N]`` or ``[N, 1]`` single-logit outputs (pre-sigmoid). Will
        be flattened to ``[N]`` and converted to a torch tensor on CPU.
    labels
        ``[N]`` int ground-truth labels (0 = negative, 1 = positive).
    threshold
        Probability threshold for hard-decision metrics (default 0.5).
    n_bins
        Number of equal-width bins for the reliability diagram + ECE
        (default 10).
    """
    logits_t = torch.as_tensor(np.asarray(logits).reshape(-1), dtype=torch.float32)
    labels_t = torch.as_tensor(np.asarray(labels).reshape(-1), dtype=torch.int64)
    n = int(labels_t.numel())

    # torchmetrics' binary primitives all accept raw logits (preds=logits,
    # target=labels) and apply sigmoid internally. Each one is a stateful
    # nn.Module but in functional usage we just instantiate-and-call.
    accuracy = float(BinaryAccuracy(threshold=threshold)(logits_t, labels_t).item())
    precision = float(BinaryPrecision(threshold=threshold)(logits_t, labels_t).item())
    recall = float(BinaryRecall(threshold=threshold)(logits_t, labels_t).item())
    f1 = float(BinaryF1Score(threshold=threshold)(logits_t, labels_t).item())
    auroc = float(BinaryAUROC()(logits_t, labels_t).item())
    ece = float(BinaryCalibrationError(n_bins=n_bins, norm="l1")(logits_t, labels_t).item())

    # Confusion matrix counts (tp / fp / tn / fn) — BinaryStatScores returns
    # a length-5 tensor [tp, fp, tn, fn, support] at the given threshold.
    stats = BinaryStatScores(threshold=threshold)(logits_t, labels_t).tolist()
    tp, fp, tn, fn = int(stats[0]), int(stats[1]), int(stats[2]), int(stats[3])

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "ece": ece,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "reliability": _reliability_bins(logits_t, labels_t, threshold, n_bins),
        "n": n,
    }


# ---------------------------------------------------------------------------
# Reliability bins
# ---------------------------------------------------------------------------


def _reliability_bins(
    logits_t: torch.Tensor,
    labels_t: torch.Tensor,
    threshold: float,
    n_bins: int,
) -> list[ReliabilityBin]:
    """Build the reliability diagram bins for the calibration plot.

    torchmetrics' ``BinaryCalibrationError`` collapses to a scalar; for
    the per-bin (lo, hi, count, confidence, accuracy) view that the
    viewer + the training_run_record schema both want, we walk the
    bins ourselves. Confidence is the model's stated confidence in
    its predicted class (= ``prob`` when predicted=1, ``1 - prob``
    when predicted=0), per Guo 2017 §2.

    Returns the contracts' typed :class:`ReliabilityBin` instances
    directly. egm-data's :func:`make_epoch_record` then drops these
    straight onto :attr:`EpochRecord.val_reliability` without
    re-validation.
    """
    probs = torch.sigmoid(logits_t).cpu().numpy()
    preds = (probs >= threshold).astype(np.int64)
    confidence = np.where(preds == 1, probs, 1.0 - probs)
    correct = (preds == labels_t.cpu().numpy()).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        # First bin is closed on the left so confidence == 0.0 lands somewhere;
        # all others are (lo, hi] so confidence == 1.0 lands in the last bin.
        in_bin = (
            (confidence >= lo) & (confidence <= hi)
            if b == 0
            else (confidence > lo) & (confidence <= hi)
        )
        count = int(in_bin.sum())
        if count == 0:
            bins.append(ReliabilityBin(lo=lo, hi=hi, count=0, confidence=0.0, accuracy=0.0))
            continue
        bins.append(
            ReliabilityBin(
                lo=lo,
                hi=hi,
                count=count,
                confidence=float(confidence[in_bin].mean()),
                accuracy=float(correct[in_bin].mean()),
            )
        )
    return bins
