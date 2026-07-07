"""Torch dataset wrapper tests.

The fixture banks are intentionally too small for a meaningful
patient-aware split (the three-way fraction allocation would put zero
patients into val/test for two-patient banks), so we exercise
``EGMTraceDataset`` directly and build a bigger ClassifierBank for the
``build_dataloaders`` smoke test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from myocard_egm_data.banks import (  # noqa: E402
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
    load_synthetic_bank_as_classifier,
)

from myocard_egm_classifier.data.augmentation import TraceTransform  # noqa: E402
from myocard_egm_classifier.data.datasets import EGMTraceDataset, build_dataloaders  # noqa: E402
from myocard_egm_classifier.data.splits import BinnedDensityStrategy  # noqa: E402


def test_dataset_returns_expected_shapes(synthetic_bank_path: Path, n_samples: int) -> None:
    """A single __getitem__ call returns (signal, label) with the right
    torch shapes/dtypes: signal is [1, T] float32 (one channel), label
    is [1] float32 (BCE single-logit head). Sanity-check before the
    model sees a batch."""
    cb = load_synthetic_bank_as_classifier(
        synthetic_bank_path,
        label_fn=lambda b: (
            (np.asarray(b.traces.fibrosis_density) > 0.0).astype(np.int64),
            {0: "healthy", 1: "fibrotic"},
        ),
    )
    signal = cb.signal_array()
    labels = cb.label_truth_array()
    tf = TraceTransform(
        input_length=n_samples,
        znorm=True,
        znorm_eps=1e-6,
        augment=False,
        max_gain=0.0,
        max_shift_frac=0.0,
    )
    ds = EGMTraceDataset(
        signal=signal,
        labels=labels,
        indices=np.arange(cb.n_traces),
        transform=tf,
        label_dtype=torch.float32,
    )
    signal_t, label_t = ds[0]
    assert signal_t.shape == (1, n_samples)
    assert label_t.shape == (1,)
    assert label_t.dtype == torch.float32


def _build_bigger_classifier_bank(n_samples: int, fs_hz: float) -> ClassifierBank:
    """Construct a ClassifierBank with enough patients to actually split.

    8 patients x 4 traces each. 4 healthy, 4 fibrotic. We split at
    (0.5, 0.25, 0.25) so each stratum (per AnyPositive) lands 2/1/1
    patients, giving every split both classes and exercising every
    code path in build_dataloaders.
    """
    rng = np.random.default_rng(0)
    traces: list[ClassifierTrace] = []
    for patient in range(8):
        label = 0 if patient < 4 else 1
        for pair_index in range(4):
            traces.append(
                ClassifierTrace(
                    bank_id="tbank_synthetic_test_2026-06-27",
                    signal=rng.standard_normal(n_samples).astype(np.float32),
                    freq_hz=fs_hz,
                    amp_type="mv",
                    split=None,
                    label_truth=label,
                    prediction=None,
                    trace_metadata={
                        "patient_id": str(patient),
                        "pair_index": pair_index,
                    },
                )
            )
    return ClassifierBank(
        banks=[
            ClassifierBankMetaData(
                bank_id="tbank_synthetic_test_2026-06-27",
                bank_type="synthetic",
                bank_path="test://memory",
                bank_metadata={"simulator": "finitewave"},
            )
        ],
        traces=traces,
        labels={0: "healthy", 1: "fibrotic"},
    )


def test_build_dataloaders_smoke(fs_hz: float, n_samples: int) -> None:
    """End-to-end: ClassifierBank -> patient-aware split -> three
    DataLoaders + an info dict. The test supplies all kwargs explicitly
    (egm-data ships no defaults). Iterate one batch through each loader.
    Catches breakage in the gluing layer that wires
    ClassifierBank/splits/datasets together."""
    bank = _build_bigger_classifier_bank(n_samples=n_samples, fs_hz=fs_hz)
    bundle = build_dataloaders(
        bank,
        input_length=n_samples,
        batch_size=4,
        num_workers=0,
        binary=True,
        znorm=True,
        znorm_eps=1e-6,
        augment_train=False,
        max_gain=0.0,
        max_shift_frac=0.0,
        split_fractions=(0.5, 0.25, 0.25),
        split_seed=0,
        dataset_seed=0,
        pin_memory=False,
    )
    assert bundle.info["n_traces"] == 32
    assert bundle.info["n_patients"] == 8
    sizes = bundle.info["split_sizes"]
    assert sizes["train"] + sizes["val"] + sizes["test"] == 32
    # Per-split class counts are reported (every split here has both
    # classes because the splitter is stratified).
    assert bundle.info["train_class_counts"].keys() == {0, 1}
    assert bundle.info["val_class_counts"].keys() == {0, 1}
    assert bundle.info["test_class_counts"].keys() == {0, 1}
    # Pull a single batch through each loader.
    for loader in (bundle.train, bundle.val, bundle.test):
        for x, y in loader:
            assert x.shape[-1] == n_samples
            assert x.dtype == torch.float32
            assert y.dtype == torch.float32
            break


def _build_mixed_label_bank(n_samples: int, fs_hz: float) -> ClassifierBank:
    """ClassifierBank where each patient has both classes (local-density
    style). 8 patients x 4 traces; within each patient, half the traces
    are positive. Exercises the AnyPositive default strategy against the
    case that broke the pre-v0.3.2 splitter."""
    rng = np.random.default_rng(0)
    traces: list[ClassifierTrace] = []
    for patient in range(8):
        for pair_index in range(4):
            label = pair_index % 2  # within-patient label alternates
            traces.append(
                ClassifierTrace(
                    bank_id="tbank_synthetic_test_2026-06-27",
                    signal=rng.standard_normal(n_samples).astype(np.float32),
                    freq_hz=fs_hz,
                    amp_type="mv",
                    split=None,
                    label_truth=label,
                    prediction=None,
                    trace_metadata={
                        "patient_id": str(patient),
                        "pair_index": pair_index,
                    },
                )
            )
    return ClassifierBank(
        banks=[
            ClassifierBankMetaData(
                bank_id="tbank_synthetic_test_2026-06-27",
                bank_type="synthetic",
                bank_path="test://memory",
                bank_metadata={"simulator": "finitewave"},
            )
        ],
        traces=traces,
        labels={0: "healthy", 1: "fibrotic"},
    )


def test_build_dataloaders_handles_mixed_label_patients(fs_hz: float, n_samples: int) -> None:
    """Local-density-style bank with both classes per patient must build
    cleanly under the default (AnyPositive) strategy. Pre-v0.3.2 this
    raised "simulation_id=X has mixed labels [0, 1]"."""
    bank = _build_mixed_label_bank(n_samples=n_samples, fs_hz=fs_hz)
    bundle = build_dataloaders(
        bank,
        input_length=n_samples,
        batch_size=4,
        num_workers=0,
        binary=True,
        znorm=True,
        znorm_eps=1e-6,
        augment_train=False,
        max_gain=0.0,
        max_shift_frac=0.0,
        split_fractions=(0.75, 0.125, 0.125),
        split_seed=0,
        dataset_seed=0,
        pin_memory=False,
    )
    # 8 patients * 4 traces = 32. Both classes present in train and val.
    assert bundle.info["n_traces"] == 32
    assert bundle.info["train_class_counts"].keys() == {0, 1}
    assert bundle.info["val_class_counts"].keys() == {0, 1}


def test_build_dataloaders_accepts_binned_density_strategy(fs_hz: float, n_samples: int) -> None:
    """Passing an explicit BinnedDensityStrategy works the same as the
    default on this mixed-label bank — every patient has rate=0.5 so
    they all land in the same bin (no per-bin stratification), but the
    splitter still partitions cleanly."""
    bank = _build_mixed_label_bank(n_samples=n_samples, fs_hz=fs_hz)
    bundle = build_dataloaders(
        bank,
        input_length=n_samples,
        batch_size=4,
        num_workers=0,
        binary=True,
        znorm=True,
        znorm_eps=1e-6,
        augment_train=False,
        max_gain=0.0,
        max_shift_frac=0.0,
        split_fractions=(0.75, 0.125, 0.125),
        split_seed=0,
        dataset_seed=0,
        pin_memory=False,
        split_strategy=BinnedDensityStrategy(n_bins=3),
    )
    assert bundle.info["n_traces"] == 32


def test_build_dataloaders_warns_on_single_class_val(fs_hz: float, n_samples: int) -> None:
    """Construct a bank that forces val onto a single-class split, and
    verify the warning fires with the actionable message."""
    # 4 patients: 3 healthy, 1 fibrotic. 0.75/0.25 patient ratio with no
    # test fraction → val ends up with 1 patient. Stratification keeps
    # like-with-like, so val gets the single fibrotic patient OR a
    # single healthy patient — either way it's single-class.
    rng = np.random.default_rng(0)
    traces: list[ClassifierTrace] = []
    for patient in range(4):
        label = 1 if patient == 3 else 0  # 3 healthy, 1 fibrotic
        for _pair_index in range(3):
            traces.append(
                ClassifierTrace(
                    bank_id="tbank_synthetic_test_2026-06-27",
                    signal=rng.standard_normal(n_samples).astype(np.float32),
                    freq_hz=fs_hz,
                    amp_type="mv",
                    split=None,
                    label_truth=label,
                    prediction=None,
                    trace_metadata={"patient_id": str(patient)},
                )
            )
    bank = ClassifierBank(
        banks=[
            ClassifierBankMetaData(
                bank_id="tbank_synthetic_test_2026-06-27",
                bank_type="synthetic",
                bank_path="test://memory",
                bank_metadata={},
            )
        ],
        traces=traces,
        labels={0: "healthy", 1: "fibrotic"},
    )
    with pytest.warns(UserWarning, match="single-class"):
        build_dataloaders(
            bank,
            input_length=n_samples,
            batch_size=2,
            num_workers=0,
            binary=True,
            znorm=True,
            znorm_eps=1e-6,
            augment_train=False,
            max_gain=0.0,
            max_shift_frac=0.0,
            split_fractions=(0.5, 0.25, 0.25),
            split_seed=0,
            dataset_seed=0,
            pin_memory=False,
        )
