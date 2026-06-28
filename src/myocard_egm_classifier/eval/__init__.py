"""Evaluation against labeled ClassifierBank inputs.

PR B ships only the inference + prediction-stamping path. The CLI
(``egm-class-eval`` in :mod:`myocard_egm_classifier.cli.eval_cmd`)
loads a trained checkpoint, runs inference over every trace of a
ClassifierBank, writes a sibling ``<stem>_pred.cbank.h5`` with
:class:`ClassifierPrediction` populated on each trace, and (for a
labeled bank) prints scalar metrics to stdout. No calibration, no
metrics file, no new schemas — see ``project/architecture.md`` for the
design rationale.

Module layout
-------------

- :mod:`.dataset` — :func:`build_eval_dataset` (sequential, no-augment
  :class:`EGMTraceDataset` over every trace). The logit-collection
  step itself is the top-level
  :func:`myocard_egm_classifier.inference_helpers.collect_logits`
  (shared with the per-epoch evaluator inside training).
- :mod:`.predictions` — :func:`populate_predictions` (stamps
  :class:`ClassifierPrediction` onto every trace from the model's
  logits) and :func:`default_predictions_bank_path` (derives the
  sibling ``_pred.cbank.h5`` output path).

The CLI module (``cli/eval_cmd.py``) is intentionally thin: argparse,
config loading, and the top-level orchestration that wires these
helpers together.

Both labeled and unlabeled banks are supported. A fully-labeled bank
yields a scored eval (an ``lpred_`` predictions bank + the full metric
suite); an unlabeled bank yields a label-free diagnostic (a ``upred_``
predictions bank, metrics skipped) — the IAFDB shape, useful for
qualitative inspection in egm-viewer. Interpreting *metrics* on
IAFDB-derived data is still off the table (no substrate ground truth),
so the CLI skips metrics on unlabeled input rather than fabricating
them — see the ``project_iafdb_eval_catch22`` memory for context.

Future additions (calibration application at eval time, per-cohort
slicing, multi-class heads) land here as new modules.
"""

from __future__ import annotations

from myocard_egm_classifier.eval.dataset import build_eval_dataset
from myocard_egm_classifier.eval.predictions import (
    PREDICTION_MODEL_ID_KEY,
    default_predictions_bank_path,
    populate_predictions,
    stamp_predictions_model_id,
)

__all__ = [
    "PREDICTION_MODEL_ID_KEY",
    "build_eval_dataset",
    "default_predictions_bank_path",
    "populate_predictions",
    "stamp_predictions_model_id",
]
