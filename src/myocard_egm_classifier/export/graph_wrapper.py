"""The :class:`CalibratedModel` wrapper exported to ONNX.

The export CLI builds one of these around the trained classifier with
the fitted temperature ``T`` (or ``T = 1.0`` when calibration is
skipped) and hands it to the ONNX exporter. The exported graph thus
emits ``logit / T`` directly — the deployment runtime applies
``sigmoid`` and the decision threshold to that already-calibrated
logit. No separate temperature parameter has to round-trip through
the metadata sidecar or be re-applied in C++.

Notes
-----
- ``temperature`` is stored as a non-learnable ``torch.Tensor`` buffer
  so it follows the module across devices and serializes inside the
  ONNX graph as a constant. Buffer (not Parameter) is correct because
  no further training happens after export — ``T`` is frozen at the
  value passed at construction time.
- Normalization (z-score, zero2one) is NOT part of this graph. The
  deployment runtime applies the configured normalization scheme
  (per the metadata sidecar's ``preprocessing.normalization.scheme``)
  before invoking the ONNX, exactly matching what the export-time
  calibration loader did before fitting ``T``.
"""

from __future__ import annotations

import torch
from torch import nn


class CalibratedModel(nn.Module):
    """Wrap a trained classifier so its forward returns ``logits / T``.

    The base model's output shape is preserved (typically
    ``[batch, num_classes]``; ``num_classes == 1`` for the v1 binary
    head). Division by ``T`` is elementwise and broadcasts across the
    batch + class axes.

    Parameters
    ----------
    base_model
        The trained ``nn.Module`` whose ``forward(x)`` returns raw
        logits.
    temperature
        Strictly positive scalar. ``1.0`` is identity (no calibration).
    """

    def __init__(self, base_model: nn.Module, temperature: float) -> None:
        super().__init__()
        if not (temperature > 0):
            raise ValueError(f"temperature must be > 0; got {temperature}.")
        self.base_model = base_model
        # Buffer, not Parameter — T is frozen at export time, never trained.
        # Storing as float32 keeps the ONNX graph in the model's working dtype.
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``base_model(x) / T``."""
        logits = self.base_model(x)
        # ``self.temperature`` round-trips through Module.__getattr__, which
        # mypy can only narrow to ``Tensor | Module``; the cast pins the type
        # at the boundary so the division returns ``Tensor`` rather than ``Any``.
        out: torch.Tensor = logits / self.temperature
        return out
