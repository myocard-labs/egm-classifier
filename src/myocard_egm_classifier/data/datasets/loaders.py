"""Tie ClassifierBank + splits + datasets into ``DataLoader``s.

``build_dataloaders`` is the one call training/eval code makes: hand
it a :class:`ClassifierBank` and the data knobs, and get back
train/val/test loaders plus an ``info`` dict.

Conventions the ClassifierBank must satisfy:

- Every trace's ``label_truth`` must be populated (we're training on
  these; an unlabeled trace is a bug).
- Every trace's ``trace_metadata`` must carry a ``patient_id`` key (the
  patient-aware splitter reads it). The converters in
  :mod:`~myocard_egm_data.banks.converters` populate this.

This module's job is gluing — the upstream policy decisions
(label semantics, label_fn, source bank choice) happen at conversion
time, not here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from myocard_egm_data.banks import ClassifierBank
from torch.utils.data import DataLoader

from ..augmentation import TraceTransform
from ..splits import PatientStratificationStrategy, patient_aware_split
from .trace_dataset import EGMTraceDataset

_TrainBatch = tuple[torch.Tensor, torch.Tensor]


@dataclass
class LoaderBundle:
    """The loaders plus the metadata the trainer needs."""

    train: DataLoader[_TrainBatch]
    val: DataLoader[_TrainBatch]
    test: DataLoader[_TrainBatch]
    info: dict[str, Any]


def _class_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


def build_dataloaders(
    bank: ClassifierBank,
    *,
    input_length: int,
    batch_size: int,
    num_workers: int,
    binary: bool,
    znorm: bool,
    znorm_eps: float,
    augment_train: bool,
    max_gain: float,
    max_shift_frac: float,
    split_fractions: tuple[float, float, float],
    split_seed: int,
    dataset_seed: int,
    pin_memory: bool,
    split_strategy: PatientStratificationStrategy | None = None,
) -> LoaderBundle:
    """Build train/val/test loaders with a patient-aware split.

    No kwarg has a default. The consumer (typically egm-classifier) owns
    the full training-config policy and passes every value explicitly.

    Parameters
    ----------
    bank
        Source data. All traces must have a ``label_truth`` and a
        ``patient_id`` in ``trace_metadata``.
    binary
        Single-logit BCE labels (float ``[1]``) when True; long labels
        for the multi-class head when False.
    augment_train
        Apply gain + time-shift augmentation to the train split only.
    split_strategy
        Per-patient stratification strategy passed to
        :func:`~myocard_egm_classifier.data.splits.patient_aware_split`. ``None``
        defaults to the upstream default (``AnyPositiveStrategy``), which
        works for both global-density and local-density banks. Pass
        ``BinnedDensityStrategy(n_bins=...)`` for finer-grained
        stratification under heavy local-density skew.

    Returns
    -------
    LoaderBundle with ``.train``/``.val``/``.test`` loaders and an
    ``.info`` dict carrying split sizes, per-split class counts,
    pos_weight, and the bank's schema version.

    Warnings
    --------
    On a binary task, emits a :class:`UserWarning` when the val or test
    split lands single-class. AUROC and the related rank metrics are
    undefined in that case (torchmetrics returns zero and prints its own
    warning); the message here points the user at split_fractions and
    ``split_strategy`` as the typical fixes.
    """
    if bank.n_traces == 0:
        raise ValueError("build_dataloaders: ClassifierBank is empty.")

    # Pull the columnar views the splitter and dataset need.
    signal = bank.signal_array()  # [N, T] float32
    labels = bank.label_truth_array()  # [N] int64
    patient_id = bank.patient_id_array()  # [N] str

    # patient_aware_split needs an integer group key; factorize patient_id.
    _, group_id = np.unique(patient_id, return_inverse=True)
    group_id = group_id.astype(np.int64)

    split = patient_aware_split(
        group_id,
        labels,
        fractions=split_fractions,
        seed=split_seed,
        strategy=split_strategy,
    )

    label_dtype = torch.float32 if binary else torch.long
    train_tf = TraceTransform(
        input_length=input_length,
        znorm=znorm,
        znorm_eps=znorm_eps,
        augment=augment_train,
        max_gain=max_gain,
        max_shift_frac=max_shift_frac,
    )
    eval_tf = TraceTransform(
        input_length=input_length,
        znorm=znorm,
        znorm_eps=znorm_eps,
        augment=False,
        max_gain=max_gain,
        max_shift_frac=max_shift_frac,
    )

    def make_ds(indices: np.ndarray, transform: TraceTransform) -> EGMTraceDataset:
        return EGMTraceDataset(
            signal=signal,
            labels=labels,
            indices=indices,
            transform=transform,
            label_dtype=label_dtype,
            seed=dataset_seed,
        )

    train_ds = make_ds(split.train, train_tf)
    val_ds = make_ds(split.val, eval_tf)
    test_ds = make_ds(split.test, eval_tf)

    # Explicit kwargs (rather than a packed **common dict) keep the
    # DataLoader call typed; a Mapping with mixed bool/int values gets
    # inferred as dict[str, int] and trips mypy on pin_memory: bool.
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    # BCE pos_weight = (#neg / #pos) on the train split, to counter the
    # typical fibrosis-class imbalance. 1.0 if a class is absent (degenerate
    # small bank).
    train_counts = _class_counts(labels[split.train])
    val_counts = _class_counts(labels[split.val])
    test_counts = _class_counts(labels[split.test])
    n_pos = train_counts.get(1, 0)
    n_neg = train_counts.get(0, 0)
    pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    # Single-class val / test on a binary task makes AUROC undefined and
    # the related rank metrics meaningless. Warn loudly so the user can
    # spot the problem before sinking compute into a useless run. We
    # don't raise — the user may still want to inspect train-side
    # behavior on a known-degenerate split — but the message tells them
    # what's wrong and how to fix it.
    if binary:
        _warn_if_single_class("val", val_counts)
        _warn_if_single_class("test", test_counts)

    info: dict[str, Any] = {
        "bank_schema_version": bank.schema_version,
        "n_traces": bank.n_traces,
        "n_samples": bank.n_samples_first,
        "input_length": input_length,
        "n_patients": int(np.unique(group_id).size),
        "split_sizes": {
            "train": int(split.train.size),
            "val": int(split.val.size),
            "test": int(split.test.size),
        },
        "split_patients": {
            "train": int(np.unique(group_id[split.train]).size),
            "val": int(np.unique(group_id[split.val]).size),
            "test": int(np.unique(group_id[split.test]).size),
        },
        "train_class_counts": train_counts,
        "val_class_counts": val_counts,
        "test_class_counts": test_counts,
        "pos_weight": float(pos_weight),
        "binary": binary,
    }
    return LoaderBundle(train=train_loader, val=val_loader, test=test_loader, info=info)


def _warn_if_single_class(split_name: str, counts: dict[int, int]) -> None:
    """Emit a UserWarning when a binary split has zero of either class.

    Empty splits are skipped (size==0 means no warning to issue — that's
    handled by the loaders' own emptiness checks). A non-empty split
    with only one class present triggers the warning; the message
    includes the actual counts and points the user at the typical fixes.
    """
    if not counts or sum(counts.values()) == 0:
        return
    classes_present = {c for c, n in counts.items() if n > 0}
    if len(classes_present) >= 2:
        return
    warnings.warn(
        f"build_dataloaders: {split_name} split is single-class "
        f"(counts={counts}). AUROC and related rank metrics will be "
        "undefined (torchmetrics will return zero). Increase the bank "
        "size, raise the val/test fraction in split_fractions, or "
        "switch split_strategy (e.g. BinnedDensityStrategy) for a "
        "finer-grained patient stratification.",
        UserWarning,
        stacklevel=2,
    )
