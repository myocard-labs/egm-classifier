"""Model package: 1D MobileViT and its registry-driven blocks.

Public API
----------
- :class:`MobileViT1D` — the Phase-1 backbone + head.
- :func:`build_mobilevit_1d` — convenience constructor for the default
  v1 architecture.
- :func:`default_v1_blocks` — the canonical block list (override per
  experiment by passing your own list to ``MobileViT1D``).
- :class:`BlockSpec` — one declared block (registry ``type`` + params).
- :func:`count_parameters` — trainable parameter count utility.

The block registry (``register_block``, ``BLOCK_REGISTRY``, etc.) lives
in :mod:`.registry` and is the extension point for new block types —
once added there, the assembler picks them up automatically (design
doc Section 9).

The v1 head is ``num_outputs == 1`` + ``BCEWithLogitsLoss`` (single
logit, ``sigmoid(logit) == P(fibrotic)``); the multi-output softmax
path (``num_outputs >= 2`` + cross-entropy) stays available for the
Phase-2 multi-class severity head. See design doc Section 6 for the
full reconciliation.
"""

from __future__ import annotations

from myocard_egm_classifier.models.mobilevit1d import (
    BlockSpec,
    MobileViT1D,
    build_mobilevit_1d,
    count_parameters,
    default_v1_blocks,
)

__all__ = [
    "BlockSpec",
    "MobileViT1D",
    "build_mobilevit_1d",
    "count_parameters",
    "default_v1_blocks",
]
