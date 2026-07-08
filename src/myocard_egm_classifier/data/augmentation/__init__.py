"""Per-trace normalize + pad + augment transform.

The transform is pure numpy and has no torch dependency, so it is usable
in notebooks and offline analysis as well as in the PyTorch dataset
wrapper. See :class:`TraceTransform` for the per-trace pipeline.
"""

from __future__ import annotations

from .trace_transform import TraceTransform

__all__ = ["TraceTransform"]
