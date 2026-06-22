"""Pytest fixtures shared across the egm-classifier test suite.

Two shapes of fixture live here:

- :func:`tiny_classifier_bank` — an in-memory :class:`ClassifierBank`
  with deterministic synthetic signals and labels. Used by the data /
  training / metrics / reporting tests so nothing needs an external
  HDF5 file to round-trip.
- :func:`tiny_bank_path` — the same bank written to a temp HDF5 file
  (via egm-data's writer), for tests that exercise the on-disk path
  (CLI smoke).

Conventions:

- Pure-Python fixtures live here.
- File-backed fixtures use ``tmp_path`` so each test gets its own
  isolated directory.
- No real data files in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from myocard_egm_data.banks import (
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
    write_classifier_bank,
)


def _make_signal(rng: np.random.Generator, n_samples: int, label: int) -> np.ndarray:
    """Generate a deterministic per-label synthetic trace.

    Healthy (label 0) gets a low-frequency sine; fibrotic (label 1) gets
    the same sine plus a high-frequency burst — enough separation that
    a 1-epoch smoke test on a small bank can actually learn something
    above chance.
    """
    t = np.arange(n_samples, dtype=np.float32) / n_samples
    base = 0.4 * np.sin(2.0 * np.pi * 4.0 * t)
    if label == 1:
        base = base + 0.2 * np.sin(2.0 * np.pi * 60.0 * t)
    noise = rng.standard_normal(n_samples).astype(np.float32) * 0.02
    return (base + noise).astype(np.float32)


@pytest.fixture
def tiny_classifier_bank() -> ClassifierBank:
    """A 80-trace, 2-class, 20-patient synthetic ClassifierBank.

    Sized so a [0.6, 0.2, 0.2] patient-aware split lands each of
    train/val/test with multiple patients of both classes — enough
    diversity that the val loss is always finite (a smaller fixture
    can land val on a single-class subset and trip BCE NaN paths in
    1-epoch smoke runs). Signal length 64 keeps the 1D MobileViT
    forward+backward fast on CPU.
    """
    rng = np.random.default_rng(0)
    n_patients = 20
    traces_per_patient = 4
    n_samples = 64
    traces: list[ClassifierTrace] = []
    for p in range(n_patients):
        label = p % 2  # alternate healthy / fibrotic by patient
        for k in range(traces_per_patient):
            sig = _make_signal(rng, n_samples, label)
            traces.append(
                ClassifierTrace(
                    bank_id=0,
                    signal=sig,
                    freq_hz=1000.0,
                    amp_type="mv",
                    split=None,
                    label_truth=label,
                    prediction=None,
                    trace_metadata={
                        "patient_id": f"P{p:02d}",
                        "trace_index": k,
                    },
                )
            )
    bank_md = ClassifierBankMetaData(
        bank_id=0,
        bank_type="synthetic",
        bank_path="<in-memory fixture>",
        bank_metadata={"fixture": "tiny_classifier_bank"},
    )
    return ClassifierBank(
        banks=[bank_md],
        traces=traces,
        labels={0: "healthy", 1: "fibrotic"},
    )


@pytest.fixture
def tiny_bank_path(tmp_path: Path, tiny_classifier_bank: ClassifierBank) -> Path:
    """Write :func:`tiny_classifier_bank` to a temp HDF5 for on-disk tests."""
    out_path = tmp_path / "tiny.h5"
    write_classifier_bank(tiny_classifier_bank, out_path)
    return out_path
