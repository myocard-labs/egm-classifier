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

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from myocard_egm_contracts import synthetic_bank as synthetic_bank_models
from myocard_egm_contracts.schema_info import current_version
from myocard_egm_data.banks import (
    ClassifierBank,
    ClassifierBankMetaData,
    ClassifierTrace,
    write_classifier_bank,
    write_synthetic_bank,
)

# Source-bank stable id for the fixture. egm-data (ClassifierBank >= 0.2)
# replaced the old integer per-source index with a stable cross-artifact
# id string (egm-contracts ArtifactId pattern), validated on construction —
# so traces + the source-bank entry reference the source by this string.
_FIXTURE_BANK_ID = "tbank_fixture_2026-06-27"


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
                    bank_id=_FIXTURE_BANK_ID,
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
        bank_id=_FIXTURE_BANK_ID,
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


# --- ported from egm-data's conftest when the dataset layer moved here (Step 8) ---


@pytest.fixture
def fs_hz() -> float:
    return 1000.0


@pytest.fixture
def trace_duration_ms() -> float:
    return 512.0


@pytest.fixture
def n_samples(fs_hz: float, trace_duration_ms: float) -> int:
    return round(trace_duration_ms * 1e-3 * fs_hz)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def synthetic_bank_path(
    tmp_path: Path, fs_hz: float, trace_duration_ms: float, n_samples: int
) -> Path:
    """Tiny Pydantic SyntheticBank written to HDF5 (2 patients x 3 traces).

    Signals are random — physiological realism isn't needed for the dataset/split tests.
    """
    rng = np.random.default_rng(0)
    n = 6
    signal = [rng.standard_normal(n_samples).astype(np.float32).tolist() for _ in range(n)]
    sim_id = [0, 0, 0, 1, 1, 1]
    pair_index = [0, 1, 2, 0, 1, 2]
    electrode_row = [0, 0, 1, 0, 0, 1]
    densities = [0.0, 0.0, 0.0, 0.3, 0.3, 0.3]

    pyd_bank = synthetic_bank_models.SyntheticBank.model_validate(
        {
            "schema_version": current_version("synthetic_bank"),
            "created_utc": _now_iso(),
            "bank_id": "tbank_synthetic_test_2026-06-27",
            "description": "Synthetic bank test fixture",
            "fs_hz": fs_hz,
            "trace_duration_ms": trace_duration_ms,
            "simulator": "finitewave",
            "cell_model": "aliev_panfilov",
            "patch_size_mm": 40.0,
            "patch_dr_mm": 0.25,
            "ap_time_unit_ms": 12.9,
            "fibrosis_strategy_name": "uniform_random",
            "fibrosis_params": {"density_min": 0.0, "density_max": 0.5},
            "electrode_config": {"grid_rows": 5, "grid_cols": 5, "spacing_mm": 2.0},
            "mixer_config": {"snr_db_min": 10.0, "snr_db_max": 25.0},
            "experiment_config": {"name": "test_fixture"},
            "noise_bank_source": "iafdb_noise_v1.h5",
            "traces": {
                "signal": signal,
                "simulation_id": sim_id,
                "pair_index": pair_index,
                "electrode_row": electrode_row,
                "fibrosis_density": densities,
                "fibrosis_density_realized": densities,
                "electrode_height_mm": [0.5] * n,
                "seed": [i * 100 for i in range(n)],
                "snr_db": [15.0] * n,
                "stim_edge": ["left"] * n,
                "noise_record": ["iaf1_afw"] * n,
                "noise_channel": ["CS12"] * n,
            },
        }
    )
    path = tmp_path / "synthetic_bank.h5"
    write_synthetic_bank(pyd_bank, path)
    return path
