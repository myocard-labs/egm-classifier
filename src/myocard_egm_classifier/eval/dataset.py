"""Sequential :class:`EGMTraceDataset` construction for ``egm-class-eval``.

The single public function — :func:`build_eval_dataset` — wraps a
:class:`ClassifierBank` in a sequential
:class:`myocard_egm_data.datasets.EGMTraceDataset`: no patient-aware
split, no augmentation, bank-order iteration. The bank-order
guarantee is load-bearing — the CLI then feeds this dataset to the
top-level :func:`myocard_egm_classifier.inference_helpers.collect_logits`
and uses the returned row order to stamp predictions back onto each
trace in :mod:`myocard_egm_classifier.eval.predictions`.

Kept in its own module (separate from
:mod:`myocard_egm_classifier.inference_helpers`, which is the
training/eval-shared model-runner helper) because dataset
construction is an eval-only concern — training builds its loaders
from :func:`myocard_egm_data.datasets.build_dataloaders`, which does
the patient-aware split + augmentation wiring this function
deliberately bypasses.
"""

from __future__ import annotations

import numpy as np
import torch
from myocard_egm_data.augmentation import TraceTransform
from myocard_egm_data.banks import ClassifierBank
from myocard_egm_data.datasets import EGMTraceDataset

from myocard_egm_classifier.constants import DEFAULT_SEED


def build_eval_dataset(
    bank: ClassifierBank,
    input_length: int,
    znorm: bool,
    znorm_eps: float,
) -> EGMTraceDataset:
    """Sequential, no-augment :class:`EGMTraceDataset` over every trace.

    Indices are ``arange(n_traces)`` so iteration order matches the
    bank's storage order — required for the per-trace prediction
    stamping pass in
    :func:`myocard_egm_classifier.eval.predictions.populate_predictions`.
    The seed is fixed to :data:`DEFAULT_SEED` since augmentation is
    disabled and there's no RNG-sensitive behavior left to control.
    """
    signal = bank.signal_array()
    # EGMTraceDataset needs a labels array, but the eval CLI discards the
    # labels collect_logits returns — predictions are stamped from logits
    # alone. A fully-labeled bank uses its real labels; an unlabeled bank
    # (the IAFDB shape, where label_truth_array() would raise) gets a zero
    # placeholder that never reaches any output.
    if all(t.label_truth is not None for t in bank.traces):
        labels = bank.label_truth_array()
    else:
        labels = np.zeros(bank.n_traces, dtype=np.int64)
    transform = TraceTransform(
        input_length=input_length,
        znorm=znorm,
        znorm_eps=znorm_eps,
        augment=False,
        max_gain=0.0,
        max_shift_frac=0.0,
    )
    return EGMTraceDataset(
        signal=signal,
        labels=labels,
        indices=np.arange(bank.n_traces),
        transform=transform,
        label_dtype=torch.float32,
        seed=DEFAULT_SEED,
    )
