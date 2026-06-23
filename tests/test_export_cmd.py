"""End-to-end smoke for the ``egm-class-export`` CLI.

Builds a fresh untrained MobileViT1D, saves it as a checkpoint (with
the ``model_meta`` block the export CLI rebuilds from), then runs the
export CLI's ``main`` against the synthetic ClassifierBank fixture
from ``conftest.py``. Untrained weights mean the fitted ``T`` is
typically at the bounded-search upper bound — we don't care about
calibration quality, we care that:

1. The CLI returns 0 (no error path).
2. Both ``<name>.onnx`` and ``<name>.model_metadata.json`` land at
   the expected paths under ``output.dir``.
3. The metadata sidecar round-trips through egm-data's loader with
   the right preprocessing / decision / training-provenance shape.
4. The exported ONNX produces logits that match the PyTorch-side
   :class:`CalibratedModel` to within fp32 round-trip tolerance.
5. The CLI flag overrides (``--threshold``, ``--opset``,
   ``--output-name``, ``--skip-calibration``) reach the metadata.
6. The friendly ``ModuleNotFoundError`` -> install-hint path fires
   if torch.onnx pulls in a missing optional dep at export time.
7. Clobber refusal: if the output files already exist, exit non-zero
   without overwriting.
8. Multi-class checkpoint refusal: the v1 export only supports the
   single-logit binary head.

Pairs with ``test_cli_config.py`` (which covers the YAML loader
behavior for ``_export_config.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pytest
import torch
from myocard_egm_data.banks import ClassifierBank, write_classifier_bank
from myocard_egm_data.records import load_egm_class_model_metadata

from myocard_egm_classifier.cli import export_cmd
from myocard_egm_classifier.cli.export_cmd import main as export_main
from myocard_egm_classifier.export import CalibratedModel
from myocard_egm_classifier.models import MobileViT1D, default_v1_blocks

# Sizes chosen to match the conftest fixture: 64-sample traces, 1 channel,
# binary head. The 20-trace bank is too small for meaningful calibration but
# is fine for round-trip / parity / smoke-shape checks.
_INPUT_LEN = 64
_IN_CHANNELS = 1
_NUM_CLASSES = 1


def _make_checkpoint(
    tmp_path: Path,
    *,
    input_length: int = _INPUT_LEN,
    in_channels: int = _IN_CHANNELS,
    num_classes: int = _NUM_CLASSES,
    training_provenance: dict[str, Any] | None = None,
) -> Path:
    """Build a tiny MobileViT1D and save a checkpoint with model_meta.

    Untrained — the export CLI rebuilds the architecture from
    ``model_meta`` and loads ``model_state_dict``; weight quality is
    irrelevant for the end-to-end plumbing checks here.
    """
    model = MobileViT1D(
        blocks=default_v1_blocks(num_outputs=num_classes),
        width_multiplier=0.5,
        in_channels=in_channels,
        stochastic_depth=0.0,
    )
    ckpt_path = tmp_path / "best.pt"
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_loss": 0.5,
        "val_metrics": {"auroc": 0.5},
        "model_meta": {
            "arch": "mobilevit_1d_v1",
            "width_multiplier": 0.5,
            "num_classes": num_classes,
            "in_channels": in_channels,
            "input_length": input_length,
            "stochastic_depth": 0.0,
            "head_expansion_channels": 320,
            "head_dropout": 0.1,
        },
    }
    if training_provenance is not None:
        payload["training_provenance"] = training_provenance
    torch.save(payload, ckpt_path)
    return ckpt_path


def _write_yaml(tmp_path: Path, body: str, name: str = "export.yaml") -> Path:
    """Helper: drop a YAML literal into ``tmp_path/<name>`` and return its path."""
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _write_input_bank(tmp_path: Path, bank: ClassifierBank, name: str = "calib.cbank.h5") -> Path:
    """Write the fixture bank to a temp file under the canonical extension."""
    input_path = tmp_path / name
    write_classifier_bank(bank, input_path)
    return input_path


def _minimal_yaml(
    *, ckpt: Path, out_dir: Path, bank: Path | None = None, name: str = "best"
) -> str:
    """Compose a minimal valid export YAML.

    ``bank=None`` exercises the skip-calibration path; everything else
    is at the package defaults so the test's intent stays narrow.
    """
    calibration_block = (
        f"calibration:\n  bank: {bank}\n  batch_size: 8\n" if bank is not None else ""
    )
    return (
        f"checkpoint: {ckpt}\n"
        f"{calibration_block}"
        "preprocessing:\n"
        "  fs_hz: 1000.0\n"
        "  bandpass_hz: [30.0, 250.0]\n"
        f"output:\n  dir: {out_dir}\n  name: {name}\n"
        "opset: 18\n"
        "decision:\n  threshold: 0.5\n  class_labels: [healthy, fibrotic]\n"
    )


def test_export_writes_onnx_and_metadata(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """Happy path: CLI returns 0; both artifact files land at the expected paths.

    Also confirms the metadata sidecar round-trips through egm-data's
    loader without complaint and that the fitted ``T`` lands under
    ``training_provenance.calibration_temperature``.
    """
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=bank_path))

    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0
    assert (out_dir / "best.onnx").exists()
    assert (out_dir / "best.model_metadata.json").exists()

    meta = load_egm_class_model_metadata(out_dir / "best.model_metadata.json")
    assert meta.preprocessing.expected_fs_hz == 1000.0
    assert meta.preprocessing.expected_trace_samples == _INPUT_LEN
    assert meta.preprocessing.normalization.scheme.value == "zscore"
    assert meta.decision.threshold == 0.5
    assert meta.decision.class_labels == ["healthy", "fibrotic"]
    prov = meta.training_provenance or {}
    assert "calibration_temperature" in prov
    assert prov["calibration_bank_path"] == str(bank_path)


def test_export_skip_calibration_stamps_unity_temperature(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """Omitting ``calibration.bank`` falls back to T = 1.0 (no calibration).

    The metadata sidecar records the unity temperature so audit tooling
    can distinguish "calibrated to 1.0" from "no calibration was run."
    """
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))

    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0
    meta = load_egm_class_model_metadata(out_dir / "best.model_metadata.json")
    prov = meta.training_provenance or {}
    assert prov["calibration_temperature"] == 1.0
    assert prov["calibration_bank_path"] is None


def test_skip_calibration_flag_wins_over_yaml_bank(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """``--skip-calibration`` overrides a bank-present YAML; T = 1.0."""
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=bank_path))

    rc = export_main([str(cfg_path), "--device", "cpu", "--skip-calibration"])
    assert rc == 0
    meta = load_egm_class_model_metadata(out_dir / "best.model_metadata.json")
    prov = meta.training_provenance or {}
    assert prov["calibration_temperature"] == 1.0
    assert prov["calibration_bank_path"] is None


def test_threshold_flag_reaches_metadata(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """``--threshold 0.42`` overrides ``decision.threshold`` in the sidecar."""
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))

    rc = export_main([str(cfg_path), "--device", "cpu", "--threshold", "0.42"])
    assert rc == 0
    meta = load_egm_class_model_metadata(out_dir / "best.model_metadata.json")
    assert meta.decision.threshold == 0.42


def test_output_name_flag_renames_both_artifacts(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """``--output-name foo`` produces ``foo.onnx`` + ``foo.model_metadata.json``."""
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))

    rc = export_main([str(cfg_path), "--device", "cpu", "--output-name", "foo"])
    assert rc == 0
    assert (out_dir / "foo.onnx").exists()
    assert (out_dir / "foo.model_metadata.json").exists()
    assert not (out_dir / "best.onnx").exists()


def test_training_provenance_passes_through_well_known_keys(tmp_path: Path) -> None:
    """``run_id`` / ``git_sha`` etc. on the checkpoint thread through to the sidecar.

    The export CLI doesn't synthesize these — it copies them out of
    ``ckpt["training_provenance"]`` so audit tooling can trace an
    exported artifact back to its training run.
    """
    prov = {"run_id": "abc-123", "git_sha": "deadbeef"}
    ckpt_path = _make_checkpoint(tmp_path, training_provenance=prov)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))

    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0
    meta = load_egm_class_model_metadata(out_dir / "best.model_metadata.json")
    out_prov = meta.training_provenance or {}
    assert out_prov["run_id"] == "abc-123"
    assert out_prov["git_sha"] == "deadbeef"


def test_pytorch_onnx_parity(tmp_path: Path) -> None:
    """The exported ONNX must produce logits matching the PyTorch wrapped model.

    Exports with T = 1.0 (skip-calibration), then runs both the wrapped
    PyTorch model and the ONNX session over the same random batch.
    fp32 round-trip through ONNX serialization typically yields
    ``< 1e-5`` element-wise diff; we use a loose ``1e-4`` bound to allow
    a little headroom across torch/onnxruntime versions.
    """
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))
    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0

    # Re-load the PyTorch model + wrap with the same T = 1.0 the CLI used.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pt_model = MobileViT1D(
        blocks=default_v1_blocks(num_outputs=_NUM_CLASSES),
        width_multiplier=0.5,
        in_channels=_IN_CHANNELS,
        stochastic_depth=0.0,
    )
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()
    wrapped = CalibratedModel(pt_model, temperature=1.0).eval()

    # Run both on the same random batch.
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(4, _IN_CHANNELS, _INPUT_LEN, generator=rng)
    with torch.no_grad():
        pt_out = wrapped(x).numpy()

    sess = ort.InferenceSession(str(out_dir / "best.onnx"), providers=["CPUExecutionProvider"])
    ox_out = sess.run(["logit"], {"signal": x.numpy()})[0]

    diff = float(np.max(np.abs(pt_out - ox_out)))
    assert diff < 1e-4, f"PyTorch vs ONNX max abs diff exceeded tolerance: {diff}"


def test_dynamic_batch_axis(tmp_path: Path) -> None:
    """The exported ONNX accepts any batch size, not just the export-time 1."""
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))
    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0
    sess = ort.InferenceSession(str(out_dir / "best.onnx"), providers=["CPUExecutionProvider"])
    for batch_size in (1, 5, 17):
        out = sess.run(
            ["logit"],
            {"signal": np.random.randn(batch_size, _IN_CHANNELS, _INPUT_LEN).astype(np.float32)},
        )[0]
        assert out.shape == (batch_size, _NUM_CLASSES)


def test_export_refuses_to_clobber_existing_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """If <name>.onnx (or .model_metadata.json) already exists, exit non-zero."""
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    out_dir.mkdir()
    # Pre-create the .onnx so the clobber-check trips.
    (out_dir / "best.onnx").write_bytes(b"")
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))

    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "already exists" in err
    assert "--output-name" in err  # hint about how to fix


def test_export_rejects_multi_class_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The v1 export CLI only supports ``num_classes == 1``; reject others cleanly."""
    ckpt_path = _make_checkpoint(tmp_path, num_classes=3)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))

    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "single-logit binary head" in err
    assert "num_classes == 1" in err


def test_missing_onnx_dep_emits_install_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When ``torch.onnx.export`` transitively can't find an optional dep
    (``onnx``, ``onnxscript``, ...), the CLI catches the ``ModuleNotFoundError``
    and re-emits a friendly install hint pointing at the ``[onnx]`` extra."""
    ckpt_path = _make_checkpoint(tmp_path)
    out_dir = tmp_path / "exports"
    cfg_path = _write_yaml(tmp_path, _minimal_yaml(ckpt=ckpt_path, out_dir=out_dir, bank=None))

    def _raises_module_not_found(*args: Any, **kwargs: Any) -> None:
        raise ModuleNotFoundError("No module named 'onnxscript'")

    monkeypatch.setattr(export_cmd, "export_to_onnx", _raises_module_not_found)

    rc = export_main([str(cfg_path), "--device", "cpu"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "onnxscript" in err
    assert "[onnx]" in err
    assert "pip install" in err
    # The metadata sidecar must NOT have been written if the ONNX export failed.
    assert not (out_dir / "best.model_metadata.json").exists()
