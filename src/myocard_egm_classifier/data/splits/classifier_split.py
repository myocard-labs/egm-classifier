"""Run :func:`patient_aware_split` against a ClassifierBank and (optionally)
write the assigned split name back onto each :class:`ClassifierTrace`.

Two ways to use this:

- :func:`split_classifier_bank` is the high-level helper: pulls
  patient_id + label_truth out of the ClassifierBank, runs the split,
  and writes ``"train"`` / ``"val"`` / ``"test"`` (or whatever names
  you choose) back to every trace's ``split`` field. Returns the
  :class:`SplitIndices` if you want them.
- :func:`apply_split_indices` just writes a pre-computed
  :class:`SplitIndices` back onto the bank — useful if you've called
  the lower-level splitter yourself.

The convention: every trace ends up with ``split`` set to a string
chosen by the caller (or one of the defaults), so a downstream
consumer can serialize the ClassifierBank and the split decision is
captured on disk alongside the data.
"""

from __future__ import annotations

import numpy as np
from myocard_egm_data.banks import ClassifierBank

from .patient_aware import SplitIndices, patient_aware_split
from .strategies import PatientStratificationStrategy


def split_classifier_bank(
    bank: ClassifierBank,
    *,
    fractions: tuple[float, float, float],
    seed: int,
    split_names: tuple[str, str, str] = ("train", "val", "test"),
    strategy: PatientStratificationStrategy | None = None,
) -> SplitIndices:
    """Run a patient-aware split and tag every trace's ``split`` field.

    Reads ``patient_id`` from each trace's ``trace_metadata`` and
    ``label_truth`` from the trace itself (raises if any trace is
    unlabeled — the splitter stratifies by label).

    Mutates every trace's ``ClassifierTrace.split`` to the
    corresponding name from ``split_names``. Also returns the
    :class:`SplitIndices` so the caller can use the row indices
    directly without iterating the bank.

    ``strategy`` chooses how patients are bucketed for stratification;
    defaults to :class:`AnyPositiveStrategy` when ``None``. See
    :mod:`myocard_egm_classifier.data.splits.strategies` for the trade-offs.
    """
    patient_id = bank.patient_id_array()
    labels = bank.label_truth_array()

    # patient_aware_split keys on an integer group id; factorize the
    # string patient_ids into ints first.
    _, group_id = np.unique(patient_id, return_inverse=True)
    group_id = group_id.astype(np.int64)

    split = patient_aware_split(group_id, labels, fractions=fractions, seed=seed, strategy=strategy)
    apply_split_indices(bank, split, split_names=split_names)
    return split


def apply_split_indices(
    bank: ClassifierBank,
    split: SplitIndices,
    *,
    split_names: tuple[str, str, str] = ("train", "val", "test"),
) -> None:
    """Write split-name strings onto each trace's ``split`` field.

    Helpful if you already computed a :class:`SplitIndices` (e.g. via
    :func:`~myocard_egm_classifier.data.splits.patient_aware_split`) and want to
    materialize the assignment back onto the ClassifierBank so the
    decision is preserved on serialization.
    """
    train_name, val_name, test_name = split_names
    assignment: list[str | None] = [None] * bank.n_traces
    for i in split.train.tolist():
        assignment[int(i)] = train_name
    for i in split.val.tolist():
        assignment[int(i)] = val_name
    for i in split.test.tolist():
        assignment[int(i)] = test_name
    for trace, name in zip(bank.traces, assignment, strict=True):
        trace.split = name
