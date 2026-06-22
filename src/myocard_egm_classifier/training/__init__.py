"""Training loop + metrics + reporting.

The metrics layer wraps ``torchmetrics`` primitives (BinaryAUROC,
BinaryAccuracy, BinaryCalibrationError, ...) into the shared
``val_metrics`` dict shape the trainer + reporting layer expect.
Reliability bins are emitted as typed
:class:`myocard_egm_data.records.ReliabilityBin` instances (the
contracts' Pydantic model, re-exported via egm-data) so the boundary
into :func:`make_epoch_record` is dict-free.

The reporting layer delegates ``run.json`` + ``metrics.csv`` writing
to ``myocard-egm-data.records``; this subpackage owns only the
training-time math plus the orchestration that calls the typed
writers at end-of-run.

Public API
----------
- :class:`TrainConfig` — hyperparameters for one run.
- :func:`train`, :func:`train_one_epoch` — training loop entrypoints.
- :func:`evaluate`, :func:`collect_logits` — eval-mode helpers, reused
  by the eval CLI.
- :func:`cosine_warmup_lr` — the scheduler primitive.
- :func:`binary_metrics` — metrics computed via torchmetrics; returned
  in the dict shape the trainer and the run-record writers both
  consume.
- :class:`EpochRecord` — re-exported from
  ``myocard-egm-data.records`` (which re-exports it from
  ``myocard-egm-contracts``); the typed per-epoch record the trainer
  produces and the run-record writer consumes.
- :func:`make_epoch_record` — re-exported from
  ``myocard-egm-data.records``; translates trainer state into a typed
  :class:`EpochRecord`.
- :func:`write_run` — one-shot orchestration of ``metrics.csv`` +
  ``run.json``, called by the train CLI at end-of-run.
"""

from __future__ import annotations

from myocard_egm_classifier.training.metrics import binary_metrics
from myocard_egm_classifier.training.reporting import (
    EpochRecord,
    make_epoch_record,
    write_run,
)
from myocard_egm_classifier.training.train import (
    TrainConfig,
    collect_logits,
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
