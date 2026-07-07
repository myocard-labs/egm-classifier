"""Tests for patient-aware splitting + stratification strategies.

Two layers covered:

1. The cardinal patient-aware invariant (every trace from one patient
   lands in exactly one split) holds across all strategies and seeds.
2. The strategies themselves: AnyPositive on a global-density-style
   bank recovers the original per-patient-label behavior; on a mixed-
   label bank it stratifies presence-vs-absence. BinnedDensity slices
   the fraction-positive axis as advertised.
"""

from __future__ import annotations

import numpy as np

from myocard_egm_classifier.data.splits import (
    AnyPositiveStrategy,
    BinnedDensityStrategy,
    patient_aware_split,
)


def test_no_patient_leak_between_splits() -> None:
    """The cardinal rule: every trace from one patient lands entirely in
    one split. Build a 10-patient dataset, split it, assert the three
    patient sets are disjoint and their union is the full patient set.
    A failure here means the patient-aware guarantee is broken (which
    silently inflates val/test metrics — the exact failure mode this
    module exists to prevent)."""
    # 10 patients * 5 traces each. 4 fibrotic patients, 6 healthy.
    sim_id = np.repeat(np.arange(10), 5).astype(np.int64)
    fibrotic_patients = {0, 1, 2, 3}
    labels = np.array([1 if sid in fibrotic_patients else 0 for sid in sim_id], dtype=np.int64)

    split = patient_aware_split(sim_id, labels, fractions=(0.6, 0.2, 0.2), seed=42)

    train_pids = set(sim_id[split.train].tolist())
    val_pids = set(sim_id[split.val].tolist())
    test_pids = set(sim_id[split.test].tolist())

    assert not (train_pids & val_pids)
    assert not (train_pids & test_pids)
    assert not (val_pids & test_pids)
    # Every patient ends up in exactly one split.
    assert train_pids | val_pids | test_pids == set(range(10))


def test_split_is_reproducible() -> None:
    """Same input, same seed -> identical row indices in each split.
    Reproducibility matters for re-running experiments and for the
    "compare two runs" case where any non-determinism in the split would
    invalidate the comparison."""
    sim_id = np.repeat(np.arange(8), 4).astype(np.int64)
    labels = np.zeros_like(sim_id)
    labels[sim_id >= 4] = 1

    a = patient_aware_split(sim_id, labels, fractions=(0.5, 0.25, 0.25), seed=7)
    b = patient_aware_split(sim_id, labels, fractions=(0.5, 0.25, 0.25), seed=7)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.val, b.val)
    assert np.array_equal(a.test, b.test)


def test_default_strategy_handles_global_density_labels() -> None:
    """Default (AnyPositive) on a single-class-per-patient bank: every
    patient's stratum equals its label, so the splitter behaves exactly
    like the original per-patient-label stratifier. Each split should
    contain proportional shares of both classes."""
    # 8 patients * 4 traces, half healthy / half fibrotic.
    sim_id = np.repeat(np.arange(8), 4).astype(np.int64)
    labels = np.where(sim_id < 4, 0, 1).astype(np.int64)

    split = patient_aware_split(sim_id, labels, fractions=(0.5, 0.25, 0.25), seed=0)

    # With 4-of-each class and (0.5, 0.25, 0.25): train gets 2 of each
    # class, val gets 1 of each, test gets 1 of each.
    for indices in (split.train, split.val, split.test):
        present = set(labels[indices].tolist())
        assert present == {0, 1}, f"missing a class in split: present={present}"


def test_any_positive_strategy_handles_mixed_labels() -> None:
    """Local-density labels: each patient has both classes. Plain-pre-
    strategy splitter raised ValueError; AnyPositive does not."""
    # 6 patients with mixed labels (3 traces each, alternating).
    sim_id = np.repeat(np.arange(6), 3).astype(np.int64)
    labels = np.tile([0, 1, 0], 6).astype(np.int64)  # every patient has both
    strategy = AnyPositiveStrategy()
    split = patient_aware_split(
        sim_id, labels, fractions=(0.5, 0.25, 0.25), seed=0, strategy=strategy
    )
    # Every patient lands in exactly one split (the patient-aware rule).
    train_pids = set(sim_id[split.train].tolist())
    val_pids = set(sim_id[split.val].tolist())
    test_pids = set(sim_id[split.test].tolist())
    assert train_pids | val_pids | test_pids == set(range(6))
    assert not (train_pids & val_pids)
    assert not (train_pids & test_pids)
    assert not (val_pids & test_pids)


def test_any_positive_stratum_matches_label_on_pure_patients() -> None:
    """AnyPositive.stratum on a single-class patient = the class itself."""
    sim_id = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    s = AnyPositiveStrategy()
    assert s.stratum(sim_id, labels, 0) == 0
    assert s.stratum(sim_id, labels, 1) == 1


def test_any_positive_stratum_flips_on_first_positive() -> None:
    """AnyPositive returns 1 the moment any trace of a patient is positive."""
    sim_id = np.array([0, 0, 0, 0], dtype=np.int64)
    labels_all_zero = np.array([0, 0, 0, 0], dtype=np.int64)
    labels_one_pos = np.array([0, 1, 0, 0], dtype=np.int64)
    s = AnyPositiveStrategy()
    assert s.stratum(sim_id, labels_all_zero, 0) == 0
    assert s.stratum(sim_id, labels_one_pos, 0) == 1


def test_binned_density_strategy_buckets_by_rate() -> None:
    """BinnedDensity(n_bins=3) over rates 0.0 / 0.5 / 1.0 maps to bins
    0 / 1 / 2 respectively (right edge closed so 1.0 lands in last bin)."""
    sim_id = np.repeat(np.arange(3), 4).astype(np.int64)
    # patient 0: rate 0.0; patient 1: rate 0.5; patient 2: rate 1.0
    labels = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    s = BinnedDensityStrategy(n_bins=3)
    assert s.stratum(sim_id, labels, 0) == 0
    assert s.stratum(sim_id, labels, 1) == 1
    assert s.stratum(sim_id, labels, 2) == 2


def test_binned_density_degenerates_to_any_positive_on_pure_patients() -> None:
    """On a single-class-per-patient bank (rates exactly 0 or 1),
    BinnedDensity collapses to two non-empty bins regardless of n_bins —
    matching AnyPositive's two-bucket behavior in spirit."""
    sim_id = np.array([0, 0, 1, 1], dtype=np.int64)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    s5 = BinnedDensityStrategy(n_bins=5)
    # patient 0: rate 0 -> bin 0; patient 1: rate 1.0 -> bin n_bins-1 = 4
    assert s5.stratum(sim_id, labels, 0) == 0
    assert s5.stratum(sim_id, labels, 1) == 4
