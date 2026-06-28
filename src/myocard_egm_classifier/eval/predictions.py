"""Predictions-bank concerns for ``egm-class-eval``.

Two pieces live here:

- :func:`populate_predictions` mutates every trace on a
  :class:`ClassifierBank` to carry a populated
  :class:`ClassifierPrediction` derived from the per-trace logits
  produced by :func:`myocard_egm_classifier.inference_helpers.collect_logits`.
- :func:`default_predictions_bank_path` derives the sibling
  ``<stem>_pred.cbank.h5`` path next to an input bank.

Both deal with the output-bank concept, so they cluster here instead
of being split between ``cli/`` and ``eval/``. The actual on-disk
write is :func:`myocard_egm_data.banks.write_classifier_bank`, called
from :mod:`myocard_egm_classifier.cli.eval_cmd` after the
predictions are populated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from myocard_egm_data.banks import ClassifierBank, ClassifierPrediction
from numpy.typing import NDArray


def populate_predictions(
    bank: ClassifierBank,
    logits: NDArray[np.float32],
    threshold: float,
) -> None:
    """Stamp :class:`ClassifierPrediction` onto every trace of ``bank``.

    For the single-logit binary head (``num_classes == 1``):

    - ``label_pred`` = 1 iff ``sigmoid(logit) >= threshold``.
    - ``label_prob`` = the probability of the predicted class,
      i.e. ``sigmoid(logit)`` when predicted=1, else ``1 - sigmoid``.
    - ``pred_logits`` = ``{0: -logit, 1: logit}`` — the negative-class
      logit is recovered as ``-logit`` since the model emits one head
      and ``sigmoid(-z) = 1 - sigmoid(z)``.

    Mutates ``bank.traces[i].prediction`` for every trace in place.
    The caller is responsible for ensuring ``logits`` row order matches
    the bank's storage order (the eval CLI builds the loader with
    ``shuffle=False`` to guarantee this).
    """
    n = bank.n_traces
    if logits.shape[0] != n:
        raise RuntimeError(f"logits/bank row mismatch: logits[0]={logits.shape[0]} vs n_traces={n}")
    flat_logits = logits.reshape(-1).astype(np.float64)
    probs_pos = 1.0 / (1.0 + np.exp(-flat_logits))
    for i, trace in enumerate(bank.traces):
        pos_logit = float(flat_logits[i])
        pos_prob = float(probs_pos[i])
        pred = 1 if pos_prob >= threshold else 0
        prob_of_pred = pos_prob if pred == 1 else (1.0 - pos_prob)
        trace.prediction = ClassifierPrediction(
            label_pred=pred,
            label_prob=prob_of_pred,
            pred_logits={0: -pos_logit, 1: pos_logit},
        )


PREDICTION_MODEL_ID_KEY = "produced_by_model_id"
"""``trace_metadata`` key under which eval records the producing model's id.

Short-term home for the model->predictions cross-artifact link: the
predictions :class:`ClassifierBank` has no dedicated ``model_id`` field
yet, and ``ClassifierBank.id`` is the bank's *own* id, so the producing
model's id rides in each trace's generic ``trace_metadata`` dict. The
long-term home is undecided — a ``ClassifierPrediction.model_id`` field
or a bank-level metadata dict are both candidates. Kept as one named
constant so the eventual migration is a single-point find."""


def stamp_predictions_model_id(bank: ClassifierBank, model_id: str) -> None:
    """Record the producing model's id on every trace of ``bank``.

    Writes ``model_id`` into each trace's generic ``trace_metadata`` dict
    under :data:`PREDICTION_MODEL_ID_KEY`. The same model produces every
    prediction in one eval pass, so the value is identical across traces;
    storing it per-trace is the cost of using the only generic provenance
    slot the bank carries today. Idempotent (re-stamping overwrites the
    key). Mutates ``bank`` in place.
    """
    for trace in bank.traces:
        trace.trace_metadata[PREDICTION_MODEL_ID_KEY] = model_id


def default_predictions_bank_path(input_bank: Path) -> Path:
    """Derive the sibling ``_pred.cbank.h5`` path from the input bank.

    Convention: ``<stem>_pred.cbank.h5`` next to the input. Strips the
    ``.cbank.h5`` suffix if present (else ``.h5``, else the bare stem)
    before appending ``_pred.cbank.h5``.
    """
    name = input_bank.name
    if name.endswith(".cbank.h5"):
        stem = name[: -len(".cbank.h5")]
    elif name.endswith(".h5"):
        stem = name[: -len(".h5")]
    else:
        stem = input_bank.stem
    return input_bank.with_name(f"{stem}_pred.cbank.h5")
