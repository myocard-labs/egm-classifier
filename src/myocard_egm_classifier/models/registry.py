"""Block registry — the configuration system's model half (design doc Sec. 9).

The architecture is a *list of blocks*, each declared by a ``type`` string
(``conv_stem_1d`` / ``mv2_1d`` / ``mobilevit_1d`` / ``head_1d``) plus a
parameter dict. A registry maps each ``type`` to (a) a frozen param
dataclass that validates the dict and (b) a builder that turns the params
into an ``nn.Module``. New block types register themselves; the model
assembler never changes.

This is the same shape timm / MMDetection / HF Transformers use, and it
mirrors ``synthetic_egm_pipeline``'s simulator-strategy registry so there
is no cross-package context switch when reading either codebase.

Channel wiring and width scaling
---------------------------------
Builders receive the running ``in_channels`` (wired by the assembler from
the previous block's output) and the global ``width_multiplier``. Every
channel-like field is scaled by the multiplier and rounded to a multiple
of 8 (``make_divisible``) at build time, so one architecture definition
yields any size (design doc Section 8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

from torch import nn

from myocard_egm_classifier.models.blocks import (
    ClassifierHead1d,
    Conv1dBNAct,
    MV2Block1d,
    make_divisible,
)
from myocard_egm_classifier.models.mobilevit_block import MobileViTBlock1d

# A builder takes (params, in_channels, width_multiplier, drop_path) and
# returns (module, out_channels) so the assembler can wire the next block.
Builder = Callable[[Any, int, float, float], tuple[nn.Module, int]]


# ---------------------------------------------------------------------------
# Per-block parameter dataclasses (validate the YAML/template dicts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvStem1dParams:
    """Stem: Conv1d(kernel=7, stride=2) -> BN -> SiLU (design doc Section 3)."""

    out_channels: int
    kernel_size: int = 7
    stride: int = 2


@dataclass(frozen=True)
class MV2BlockParams:
    """MobileNetV2-1D inverted residual (design doc Section 4)."""

    out_channels: int
    stride: int = 1
    expand_ratio: int = 4
    kernel_size: int = 3


@dataclass(frozen=True)
class MobileViT1dParams:
    """MobileViT-1D block (design doc Section 5). Channels preserved."""

    out_channels: int  # the block's operating channel count (in == out)
    transformer_dim: int
    transformer_depth: int
    num_heads: int = 4
    mlp_ratio: float = 2.0
    patch_size: int = 2
    local_kernel_size: int = 3
    fusion_kernel_size: int = 3
    dropout: float = 0.0


@dataclass(frozen=True)
class Head1dParams:
    """Classifier head (design doc Section 6)."""

    expansion_channels: int
    num_outputs: int
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_conv_stem_1d(
    p: ConvStem1dParams, in_channels: int, width: float, drop_path: float
) -> tuple[nn.Module, int]:
    """Build the stem conv (Section 3). ``drop_path`` is unused (no residual)."""
    out = make_divisible(p.out_channels * width)
    module = Conv1dBNAct(in_channels, out, kernel_size=p.kernel_size, stride=p.stride, act=True)
    return module, out


def _build_mv2_1d(
    p: MV2BlockParams, in_channels: int, width: float, drop_path: float
) -> tuple[nn.Module, int]:
    """Build an MV2-1D inverted-residual block (Section 4) with width-scaled channels."""
    out = make_divisible(p.out_channels * width)
    module = MV2Block1d(
        in_channels=in_channels,
        out_channels=out,
        stride=p.stride,
        expand_ratio=p.expand_ratio,
        kernel_size=p.kernel_size,
        drop_path=drop_path,
    )
    return module, out


def _build_mobilevit_1d(
    p: MobileViT1dParams, in_channels: int, width: float, drop_path: float
) -> tuple[nn.Module, int]:
    """Build a MobileViT-1D block (Section 5) and enforce the channel-preserve invariant.

    The MobileViT block preserves channels, so the wired ``in_channels``
    and the scaled declared ``out_channels`` must agree — if they
    don't, the architecture list is mis-ordered (a channel-changing MV2
    block must precede every MobileViT block). We also assert that the
    scaled transformer dim stays divisible by ``num_heads`` after the
    width multiplier is applied.
    """
    # MobileViT block preserves channels, so out == in. We still scale the
    # declared out_channels and assert it matches the wired in_channels, to
    # catch a mis-ordered block list early.
    out = make_divisible(p.out_channels * width)
    if out != in_channels:
        raise ValueError(
            f"mobilevit_1d expects in_channels == out_channels, but wired "
            f"in_channels={in_channels} != scaled out_channels={out}. A "
            f"channel-changing MV2 block must precede each MobileViT block."
        )
    transformer_dim = make_divisible(p.transformer_dim * width)
    if transformer_dim % p.num_heads != 0:
        raise ValueError(
            f"scaled transformer_dim={transformer_dim} not divisible by num_heads={p.num_heads}."
        )
    module = MobileViTBlock1d(
        in_channels=in_channels,
        transformer_dim=transformer_dim,
        transformer_depth=p.transformer_depth,
        num_heads=p.num_heads,
        mlp_ratio=p.mlp_ratio,
        patch_size=p.patch_size,
        local_kernel_size=p.local_kernel_size,
        fusion_kernel_size=p.fusion_kernel_size,
        dropout=p.dropout,
        drop_path=drop_path,
    )
    return module, out


def _build_head_1d(
    p: Head1dParams, in_channels: int, width: float, drop_path: float
) -> tuple[nn.Module, int]:
    """Build the classifier head (Section 6) and return ``num_outputs`` as the wired count."""
    expansion = make_divisible(p.expansion_channels * width)
    module = ClassifierHead1d(
        in_channels=in_channels,
        expansion_channels=expansion,
        num_outputs=p.num_outputs,
        dropout=p.dropout,
    )
    return module, p.num_outputs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# type -> (param dataclass, builder)
BLOCK_REGISTRY: dict[str, tuple[type, Builder]] = {
    "conv_stem_1d": (ConvStem1dParams, _build_conv_stem_1d),
    "mv2_1d": (MV2BlockParams, _build_mv2_1d),
    "mobilevit_1d": (MobileViT1dParams, _build_mobilevit_1d),
    "head_1d": (Head1dParams, _build_head_1d),
}


def register_block(type_name: str, params_cls: type, builder: Builder) -> None:
    """Add a new block type to the registry (extension point, design doc Sec. 9)."""
    if type_name in BLOCK_REGISTRY:
        raise ValueError(f"Block type {type_name!r} already registered.")
    BLOCK_REGISTRY[type_name] = (params_cls, builder)


def parse_block_params(type_name: str, raw: dict[str, Any]) -> Any:
    """Validate a raw param dict into the block's frozen param dataclass."""
    if type_name not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown block type {type_name!r}. Registered: {sorted(BLOCK_REGISTRY)}.")
    params_cls, _ = BLOCK_REGISTRY[type_name]
    allowed = {f.name for f in fields(params_cls)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown params for block {type_name!r}: {sorted(unknown)}.")
    return params_cls(**raw)


def build_block(
    type_name: str, params: Any, in_channels: int, width: float, drop_path: float
) -> tuple[nn.Module, int]:
    """Build one block module; returns (module, out_channels)."""
    _, builder = BLOCK_REGISTRY[type_name]
    return builder(params, in_channels, width, drop_path)
