"""Patient-aware train/val/test splitting.

The cardinal rule: every trace from one ``simulation_id`` (the patient
analog — one simulator substrate realization, or one IAFDB recording for
the real-data bank) lands entirely in one split. Random per-trace
splitting lets the model see the same fibrosis-pattern realization in
train and val, which inflates val metrics — a failure mode that has
burned multiple cardiology-ML papers. Splitting by patient closes that
leak.

We additionally *stratify* by a per-patient key chosen by a
:class:`PatientStratificationStrategy` so the (typically imbalanced)
class ratio is preserved across the three splits. Without
stratification, a small dataset can land all healthy patients in one
split by chance.

The strategy is pluggable: callers pick whichever stratification key
makes sense for their bank's labeling scheme. See
:mod:`myocard_egm_classifier.data.splits.strategies` for the shipped strategies
and instructions on adding new ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .strategies import AnyPositiveStrategy, PatientStratificationStrategy


@dataclass(frozen=True)
class SplitIndices:
    """Row indices into the bank arrays for each split."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _allocate(n_patients: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Turn fractions into integer counts that sum to ``n_patients``.

    Train takes the rounded share; val and test split the remainder with
    test getting the leftover, so tiny patient counts still place at least
    what's available rather than silently dropping rows.
    """
    f_train, f_val, _ = fractions
    n_train = round(n_patients * f_train)
    n_val = round(n_patients * f_val)
    n_train = min(n_train, n_patients)
    n_val = min(n_val, n_patients - n_train)
    n_test = n_patients - n_train - n_val
    return n_train, n_val, n_test


def patient_aware_split(
    simulation_id: np.ndarray,
    labels: np.ndarray,
    fractions: tuple[float, float, float],
    seed: int,
    strategy: PatientStratificationStrategy | None = None,
) -> SplitIndices:
    """Split trace rows into train/val/test by patient, stratified by ``strategy``.

    Parameters
    ----------
    simulation_id
        ``[N]`` patient id per trace (integer-factorized upstream).
    labels
        ``[N]`` integer label per trace. The strategy decides how to
        derive a per-patient stratification key from this.
    fractions
        ``(train, val, test)`` patient fractions; must sum to ~1.
    seed
        RNG seed for the patient shuffle (reproducible splits).
    strategy
        How to bucket patients for stratification. ``None`` defaults to
        :class:`~myocard_egm_classifier.data.splits.strategies.AnyPositiveStrategy`
        — one bit per patient (1 if any trace is positive, else 0),
        which recovers the original behavior on global-density banks.

    Returns
    -------
    SplitIndices of row indices into the original arrays.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {fractions} (sum {sum(fractions)}).")
    if strategy is None:
        strategy = AnyPositiveStrategy()
    rng = np.random.default_rng(seed)

    patients = np.unique(simulation_id)
    by_stratum: dict[int, list[int]] = {}
    for pid in patients:
        key = strategy.stratum(simulation_id, labels, int(pid))
        by_stratum.setdefault(key, []).append(int(pid))

    train_pids: list[int] = []
    val_pids: list[int] = []
    test_pids: list[int] = []
    for key in sorted(by_stratum):
        group = np.array(by_stratum[key])
        rng.shuffle(group)
        n_train, n_val, _n_test = _allocate(group.size, fractions)
        train_pids.extend(group[:n_train].tolist())
        val_pids.extend(group[n_train : n_train + n_val].tolist())
        test_pids.extend(group[n_train + n_val :].tolist())

    train_set, val_set, test_set = set(train_pids), set(val_pids), set(test_pids)
    assert not (train_set & val_set), "patient leak: train ∩ val"
    assert not (train_set & test_set), "patient leak: train ∩ test"
    assert not (val_set & test_set), "patient leak: val ∩ test"

    def rows_for(pid_set: set[int]) -> np.ndarray:
        mask = np.isin(simulation_id, list(pid_set))
        return np.nonzero(mask)[0]

    return SplitIndices(
        train=rows_for(train_set),
        val=rows_for(val_set),
        test=rows_for(test_set),
    )
