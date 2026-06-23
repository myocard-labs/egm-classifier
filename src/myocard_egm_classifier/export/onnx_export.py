"""Emit the ONNX graph for a (calibrated) classifier.

Wraps :func:`torch.onnx.export` with the specific call signature the
v1 EGM classifier needs:

- Input name ``"signal"``, output name ``"logit"`` — match the
  ``egm_class_model_metadata`` schema so the deployment runtime can
  look them up by name.
- Input shape ``[batch, channels, time]``; only the batch axis is
  dynamic. The dynamic batch axis lets the same ONNX serve both
  real-time scoring (batch=1) and offline bulk inference (batch=N)
  without re-export.
- Export under ``torch.onnx.TrainingMode.EVAL`` so BatchNorm /
  Dropout fold to their inference-time formulae (no running-stats
  updates baked into the graph).
- Constant folding on (the default) so the temperature-scaling
  divisor in :class:`CalibratedModel` collapses into a graph
  constant rather than a buffer-fetch at runtime.
- ``dynamo=False`` to use the legacy TorchScript-based exporter.
  PyTorch 2.5+'s default dynamo exporter wants a different
  dynamic-shape API (``dynamic_shapes={"signal": {0: torch.export.Dim(...)}}``
  rather than ``dynamic_axes``) and triggers an opset-downconvert
  pass that currently fails on some Conv1d-related ops. The legacy
  exporter handles our model cleanly; migration to dynamo +
  ``dynamic_shapes`` is tracked as a post-v0.1.0 task.

The exporter is given a model already wrapped by
:class:`CalibratedModel`, so the emitted graph naturally produces
``base_model(x) / T``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch import nn
from torch.jit import TracerWarning  # type: ignore[attr-defined]

# Names pinned to match the egm_class_model_metadata schema's expectations.
INPUT_NAME = "signal"
OUTPUT_NAME = "logit"


def export_to_onnx(
    model: nn.Module,
    *,
    output_path: Path,
    input_length: int,
    in_channels: int,
    opset: int,
    device: torch.device,
) -> Path:
    """Serialize ``model`` to ONNX at ``output_path``; return the resolved path.

    Parameters
    ----------
    model
        Module whose ``forward`` accepts ``[batch, in_channels,
        input_length]`` float32 and returns logits shaped
        ``[batch, num_outputs]``. Typically a :class:`CalibratedModel`
        wrapping the trained classifier so the exported graph emits
        already-calibrated logits.
    output_path
        Where to write the ``.onnx`` file. Parent dir is created if
        needed.
    input_length
        Per-trace length the model expects (from the checkpoint's
        ``model_meta["input_length"]``).
    in_channels
        Channel-axis size (1 for the v1 single-bipolar-trace model).
    opset
        ONNX operator-set version. 17 is the v1 default and covers
        every op MobileViT1D uses.
    device
        Device the model + dummy input live on for the trace pass.
        The exported graph is device-agnostic; this only affects the
        single forward pass `torch.onnx.export` runs internally.

    Returns
    -------
    Path
        ``output_path`` after the file is written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.zeros(1, in_channels, input_length, dtype=torch.float32, device=device)

    # Suppress TorchScript's TracerWarning for the duration of the export.
    # MobileViTBlock.forward currently does an ``if T % p != 0: raise``
    # divisibility check that produces one TracerWarning per block; the
    # check is structural (depends on input_length + stride pattern, both
    # constants here) so the trace is correct, but the warning is noisy.
    # The architectural fix — move the check to MobileViTBlock.__init__ —
    # is tracked separately; until then, suppress the noise here so the
    # CLI output stays scannable. Any other warning category passes
    # through unchanged.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            export_params=True,
            opset_version=opset,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            # Only the batch axis is dynamic. The channel axis is pinned at
            # in_channels (Conv1d API requirement) and the time axis at
            # input_length (the model's expected receptive field).
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            # Use the legacy TorchScript-based exporter. See module
            # docstring for the dynamo-vs-legacy reasoning.
            dynamo=False,
        )
    return output_path
