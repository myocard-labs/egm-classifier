"""Build + write the ``egm_class_model_metadata.json`` sidecar.

The metadata sidecar pairs with the exported ONNX file under the
same output directory (e.g. ``best.onnx`` +
``best.model_metadata.json``). It captures everything the deployment
runtime needs that the ONNX graph itself doesn't encode:

- ``model_artifact`` — pointer to the ONNX file plus its sha256 +
  size_bytes for tamper detection.
- ``input`` / ``output`` — tensor names, shapes, dtypes, output
  semantics.
- ``preprocessing`` — expected sample rate, trace length, bandpass
  edges, per-trace normalization scheme. The runtime applies these
  before invoking the model.
- ``decision`` — threshold + class labels.
- ``training_provenance`` — best-effort breadcrumbs (run identifier
  + git SHA if the checkpoint carried them, plus the fitted
  temperature stamped under a producer-specific key).

The schema, Pydantic models, and the on-disk write/load helpers all
live in ``myocard-egm-contracts`` / ``myocard-egm-data``; this
module just composes the per-block dicts from the typed export
config + runtime values and hands them off.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from myocard_egm_data.records import (
    EgmClassModelMetadata,
    build_egm_class_model_metadata,
    write_egm_class_model_metadata,
)

from myocard_egm_classifier.cli._export_config import ExportExperimentConfig


def _sha256_of_file(path: Path) -> str:
    """Return the hex sha256 of ``path``'s contents.

    Streams the file in 1 MiB chunks so this works on multi-hundred-MB
    ONNX exports without loading everything into memory at once.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _training_provenance(
    checkpoint_path: Path,
    checkpoint_dict: dict[str, Any],
    temperature: float,
    calibration_bank: Path | None,
) -> dict[str, Any]:
    """Compose the ``training_provenance`` block.

    The schema's ``training_provenance`` is open-ended
    (``additionalProperties: true``). We populate the well-known keys
    when the checkpoint carries them and stamp two
    producer-specific keys recording the calibration outcome:

    - ``calibration_temperature`` — the fitted ``T`` (or ``1.0`` if
      calibration was skipped). The graph already bakes ``1/T``, but
      recording the value here makes audit + reproducibility easier.
    - ``calibration_bank_path`` — absolute path to the labeled bank
      used for the fit, or ``None`` if calibration was skipped.
    - ``checkpoint_path`` — absolute path to the source checkpoint
      this export was built from. Same audit-trail purpose.
    """
    meta_block: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "calibration_temperature": float(temperature),
        "calibration_bank_path": str(calibration_bank) if calibration_bank else None,
    }
    # Pull well-known keys out of the checkpoint's training_provenance if
    # the trainer stamped them. As of v0.4.0 the trainer stamps the
    # cross-artifact ids (run_id, trained_on_bank_id, run_name) plus
    # git_sha into the checkpoint, so these now thread straight through to
    # the metadata sidecar (closing task #286). 'run_json_path',
    # 'training_bank_path', and 'training_bank_schema_version' stay
    # schema-well-known but unstamped; the guarded copy keeps them
    # forward-compatible. produced_model_id is surfaced as the top-level
    # model_id (see build_metadata), not duplicated into this block.
    src_meta = checkpoint_dict.get("training_provenance", {})
    if isinstance(src_meta, dict):
        for key in (
            "run_id",
            "trained_on_bank_id",
            "run_name",
            "run_json_path",
            "training_bank_path",
            "training_bank_schema_version",
            "git_sha",
        ):
            if key in src_meta:
                meta_block[key] = src_meta[key]
    return meta_block


def build_metadata(
    *,
    cfg: ExportExperimentConfig,
    checkpoint_dict: dict[str, Any],
    onnx_path: Path,
    input_length: int,
    in_channels: int,
    num_outputs: int,
    temperature: float,
) -> EgmClassModelMetadata:
    """Assemble the Pydantic :class:`EgmClassModelMetadata` for the export.

    The metadata's top-level ``model_id`` is sourced from the checkpoint's
    ``training_provenance.produced_model_id`` (the stable id the trainer
    minted for this model); it's ``None`` for legacy checkpoints, which the
    schema permits.

    Parameters
    ----------
    cfg
        The export YAML's typed config — sources for the
        ``preprocessing``, ``decision``, and ``output`` blocks.
    checkpoint_dict
        The full dict returned by :func:`torch.load` on the source
        checkpoint. Used to pull any training-side provenance the
        trainer stamped.
    onnx_path
        Path to the just-written ONNX file. Used to compute
        ``model_artifact.sha256`` and ``model_artifact.size_bytes``,
        and the filename relative to the metadata file's directory.
    input_length, in_channels, num_outputs
        Shape constants derived from the checkpoint's ``model_meta``
        at the call site.
    temperature
        The fitted ``T`` (or ``1.0`` if calibration was skipped) —
        already baked into ``onnx_path``; recorded again under
        ``training_provenance.calibration_temperature`` for audit.
    """
    onnx_sha = _sha256_of_file(onnx_path)
    onnx_size = onnx_path.stat().st_size

    # The model's own stable id is the run's produced_model_id, stamped
    # into the checkpoint by the trainer. None on legacy checkpoints — the
    # schema field is optional, so older exports still validate.
    src_prov = checkpoint_dict.get("training_provenance", {})
    produced_model_id = src_prov.get("produced_model_id") if isinstance(src_prov, dict) else None

    return build_egm_class_model_metadata(
        model_id=produced_model_id,
        model_artifact={
            "filename": onnx_path.name,
            "framework": "onnx",
            "sha256": onnx_sha,
            "size_bytes": onnx_size,
        },
        input_spec={
            "name": "signal",
            "shape": ["?", in_channels, input_length],
            "dtype": "float32",
        },
        output_spec={
            "name": "logit",
            "shape": ["?", num_outputs],
            "dtype": "float32",
            "semantics": "binary_logit",
        },
        preprocessing={
            "expected_fs_hz": cfg.preprocessing.fs_hz,
            "expected_trace_samples": input_length,
            "bandpass_hz": list(cfg.preprocessing.bandpass_hz),
            "normalization": {"scheme": cfg.preprocessing.normalization_scheme},
        },
        decision={
            "threshold": cfg.decision.threshold,
            "class_labels": list(cfg.decision.class_labels),
        },
        training_provenance=_training_provenance(
            checkpoint_path=cfg.checkpoint,
            checkpoint_dict=checkpoint_dict,
            temperature=temperature,
            calibration_bank=cfg.calibration.bank,
        ),
    )


def write_metadata(metadata: EgmClassModelMetadata, *, dir_: Path, base_name: str) -> Path:
    """Write ``metadata`` to ``<dir_>/<base_name>.model_metadata.json``.

    Path convention matches the schema docs example
    (``best.onnx`` + ``best.model_metadata.json`` side by side).
    """
    path = dir_ / f"{base_name}.model_metadata.json"
    write_egm_class_model_metadata(path, metadata)
    return path
