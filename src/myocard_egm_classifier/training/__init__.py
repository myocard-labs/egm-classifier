"""Training loop + reporting orchestration.

This subpackage owns only the training-loop math (optimizer step,
cosine warmup schedule, per-epoch evaluate-on-val) and the end-of-run
reporting orchestration. Two helpers used by training but **also** by
the eval CLI live at the package top level:

- :func:`myocard_egm_classifier.metrics.binary_metrics` — the
  metric bundle computed from logits + labels.
- :func:`myocard_egm_classifier.inference_helpers.collect_logits` —
  runs a model over a DataLoader and returns logits + labels.

Both are re-exported here for back-compat with code that historically
imported them from ``training``; new call sites should import from
the top-level modules directly.

Public API (training-specific):

- :class:`TrainConfig` — hyperparameters for one run.
- :func:`train`, :func:`train_one_epoch` — training loop entry points.
- :func:`evaluate` — per-epoch eval (uses ``collect_logits`` +
  ``binary_metrics`` internally; also computes the BCE/CE loss).
- :func:`cosine_warmup_lr` — the scheduler primitive.
- :class:`EpochRecord` — re-exported from
  ``myocard-egm-data.records``; the typed per-epoch record the
  trainer produces and the run-record writer consumes.
- :func:`make_epoch_record` — re-exported from
  ``myocard-egm-data.records``; translates trainer state into a typed
  :class:`EpochRecord`.
- :func:`write_run` — one-shot orchestration of ``metrics.csv`` +
  ``run.json``, called by the train CLI at end-of-run.
"""

from __future__ import annotations

# Re-exports of top-level helpers (used by both training + eval) for
# back-compat with code that historically imported them from training.
from myocard_egm_classifier.inference_helpers import collect_logits
from myocard_egm_classifier.metrics import binary_metrics
from myocard_egm_classifier.training.reporting import (
    EpochRecord,
    make_epoch_record,
    write_run,
)
from myocard_egm_classifier.training.train import (
    TrainConfig,
    cosine_warmup_lr,
    evaluate,
    train,
    train_one_epoch,
)

__all__ = [
    "EpochRecord",
    "TrainConfig",
    "binary_metrics",
    "collect_logits",
    "cosine_warmup_lr",
    "evaluate",
    "make_epoch_record",
    "train",
    "train_one_epoch",
    "write_run",
]
