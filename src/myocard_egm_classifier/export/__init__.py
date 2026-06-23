"""ONNX export pipeline for the v1 EGM classifier.

Four functional pieces, orchestrated by ``cli/export_cmd.py``:

- :class:`CalibratedModel` (in :mod:`.graph_wrapper`) — wraps a
  trained classifier with a temperature divisor so the exported ONNX
  emits ``logits / T``. ``T = 1.0`` is identity (no calibration).
- :func:`fit_temperature_from_bank` (in :mod:`.calibration`) — runs
  the model over a labeled :class:`ClassifierBank`, applies the
  configured per-trace normalization to each batch, collects logits,
  and fits ``T`` against the labels via
  :func:`myocard_egm_signal.model.temperature_scaling.fit_temperature`.
- :func:`export_to_onnx` (in :mod:`.onnx_export`) — wraps
  :func:`torch.onnx.export` with the schema-compliant call signature:
  input/output names, dynamic batch axis, opset pinning, eval-mode
  trace.
- :func:`build_metadata` + :func:`write_metadata` (in
  :mod:`.metadata`) — compose + write the
  ``egm_class_model_metadata.json`` sidecar that pairs with the ONNX
  file on disk.

``onnx`` / ``onnxruntime`` are optional-extra dependencies so the
rest of the package can be imported without them.
"""

from __future__ import annotations

from myocard_egm_classifier.export.calibration import fit_temperature_from_bank
from myocard_egm_classifier.export.graph_wrapper import CalibratedModel
from myocard_egm_classifier.export.metadata import build_metadata, write_metadata
from myocard_egm_classifier.export.onnx_export import export_to_onnx

__all__ = [
    "CalibratedModel",
    "build_metadata",
    "export_to_onnx",
    "fit_temperature_from_bank",
    "write_metadata",
]
