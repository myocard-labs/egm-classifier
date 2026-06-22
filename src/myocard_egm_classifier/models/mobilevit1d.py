"""Full 1D MobileViT model assembled from a registry-driven block list.

The architecture is the 1D adaptation of MobileViT (Mehta & Rastegari
2022) described in ``project/phase1_design.md``. The canonical block
list lives in :func:`default_v1_blocks` (base channels at
``width_multiplier=1``); :class:`MobileViT1D` iterates that list, looks
each block up in ``models.registry``, scales channels by the width
multiplier, wires ``in_channels`` between blocks, and grades the
stochastic-depth schedule across depth.

Spatial/temporal flow for the default v1 (input T=512)::

    input              [B, 1,   512]
    conv_stem_1d  s2   [B, c0,  256]
    mv2_1d        s1   [B, c1,  256]
    mv2_1d        s2   [B, c2,  128]
    mv2_1d x2     s1   [B, c2,  128]
    mv2_1d        s2   [B, c3,   64]
    mobilevit_1d  p4   [B, c3,   64]
    mv2_1d        s2   [B, c4,   32]
    mobilevit_1d  p2   [B, c4,   32]
    mv2_1d        s2   [B, c5,   16]
    mobilevit_1d  p2   [B, c5,   16]      <- T_final = 16
    head_1d            [B, num_outputs]

Base widths are MobileViT-XS-sized (a conservative middle ground for the
~2k-trace Phase-1 banks); use ``width_multiplier`` 0.5 / 1.0 / 1.5 to scale.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from myocard_egm_classifier.constants import (
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_HEAD_EXPANSION_CHANNELS,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_STOCHASTIC_DEPTH,
    DEFAULT_WIDTH_MULTIPLIER,
)
from myocard_egm_classifier.models.registry import build_block, parse_block_params


@dataclass(frozen=True)
class BlockSpec:
    """One declared block: a registry ``type`` plus its parameter dict."""

    type: str
    params: Mapping[str, Any] = field(default_factory=dict)


# Base channel widths (width_multiplier = 1.0). MobileViT-XS sizing.
_BASE_CHANNELS = (16, 32, 48, 64, 80, 96)  # c0..c5
_BASE_TRANSFORMER_DIMS = (96, 120, 144)  # the 3 MobileViT blocks
_BASE_TRANSFORMER_DEPTHS = (2, 4, 3)
_BASE_PATCH_SIZES = (4, 2, 2)  # p4 first, p2 deeper (Sec. 2)
_MV2_EXPAND_RATIO = 4


def default_v1_blocks(
    num_outputs: int = DEFAULT_NUM_CLASSES,
    head_expansion_channels: int = DEFAULT_HEAD_EXPANSION_CHANNELS,
    head_dropout: float = DEFAULT_HEAD_DROPOUT,
) -> list[BlockSpec]:
    """The canonical Phase-1 architecture as a registry block list.

    Channels here are *base* widths; :class:`MobileViT1D` applies the
    ``width_multiplier`` at build time. ``num_outputs`` is 1 for the v1
    binary head (BCE) or ``n_classes`` for the Phase-2 multi-class head.
    """
    c0, c1, c2, c3, c4, c5 = _BASE_CHANNELS
    td0, td1, td2 = _BASE_TRANSFORMER_DIMS
    d0, d1, d2 = _BASE_TRANSFORMER_DEPTHS
    p0, p1, p2 = _BASE_PATCH_SIZES
    t = _MV2_EXPAND_RATIO

    return [
        BlockSpec("conv_stem_1d", {"out_channels": c0, "kernel_size": 7, "stride": 2}),
        # stage 1
        BlockSpec("mv2_1d", {"out_channels": c1, "stride": 1, "expand_ratio": t}),
        # stage 2
        BlockSpec("mv2_1d", {"out_channels": c2, "stride": 2, "expand_ratio": t}),
        BlockSpec("mv2_1d", {"out_channels": c2, "stride": 1, "expand_ratio": t}),
        BlockSpec("mv2_1d", {"out_channels": c2, "stride": 1, "expand_ratio": t}),
        # stage 3 (first MobileViT block, p=4)
        BlockSpec("mv2_1d", {"out_channels": c3, "stride": 2, "expand_ratio": t}),
        BlockSpec(
            "mobilevit_1d",
            {
                "out_channels": c3,
                "transformer_dim": td0,
                "transformer_depth": d0,
                "patch_size": p0,
            },
        ),
        # stage 4 (p=2)
        BlockSpec("mv2_1d", {"out_channels": c4, "stride": 2, "expand_ratio": t}),
        BlockSpec(
            "mobilevit_1d",
            {
                "out_channels": c4,
                "transformer_dim": td1,
                "transformer_depth": d1,
                "patch_size": p1,
            },
        ),
        # stage 5 (p=2)
        BlockSpec("mv2_1d", {"out_channels": c5, "stride": 2, "expand_ratio": t}),
        BlockSpec(
            "mobilevit_1d",
            {
                "out_channels": c5,
                "transformer_dim": td2,
                "transformer_depth": d2,
                "patch_size": p2,
            },
        ),
        # head
        BlockSpec(
            "head_1d",
            {
                "expansion_channels": head_expansion_channels,
                "num_outputs": num_outputs,
                "dropout": head_dropout,
            },
        ),
    ]


class MobileViT1D(nn.Module):
    """1D MobileViT assembled from a block list via the registry.

    Parameters
    ----------
    blocks
        Sequence of :class:`BlockSpec`. Defaults to :func:`default_v1_blocks`.
    width_multiplier
        Uniform channel scale (design doc Section 8).
    in_channels
        Model input channels (1 for bipolar EGM).
    stochastic_depth
        Max DropPath probability; graded linearly 0 -> this across depth
        (design doc Section 10).
    """

    def __init__(
        self,
        blocks: Sequence[BlockSpec] | None = None,
        width_multiplier: float = DEFAULT_WIDTH_MULTIPLIER,
        in_channels: int = DEFAULT_INPUT_CHANNELS,
        stochastic_depth: float = DEFAULT_STOCHASTIC_DEPTH,
    ) -> None:
        """Walk the block list, build each block via the registry, and wire up channels.

        Per block we (a) compute the per-position drop_path as a linear
        ramp from 0 at the stem to ``stochastic_depth`` at the final
        block; (b) validate the YAML/dict params against the block's
        frozen param dataclass; (c) build the module with the running
        ``in_channels`` and the global ``width_multiplier``. The last
        block in the list is split off as :attr:`head` so callers can
        grab features from :attr:`features` directly (needed for the
        Phase-5 self-supervised pretraining path).
        """
        super().__init__()
        if blocks is None:
            blocks = default_v1_blocks()
        self.block_specs = list(blocks)
        self.width_multiplier = width_multiplier

        modules: list[nn.Module] = []
        running_channels = in_channels
        n_blocks = len(self.block_specs)
        for i, spec in enumerate(self.block_specs):
            # Linear stochastic-depth schedule: 0 at the stem, max at the end.
            drop_path = stochastic_depth * i / max(1, n_blocks - 1)
            params = parse_block_params(spec.type, dict(spec.params))
            module, running_channels = build_block(
                spec.type, params, running_channels, width_multiplier, drop_path
            )
            modules.append(module)

        # Everything except the final head is the feature extractor; the
        # head is the last block. Keep them separate so callers can grab
        # features (Phase 5 self-supervised pretraining will want this).
        self.features = nn.Sequential(*modules[:-1])
        self.head = modules[-1]
        self.num_outputs = running_channels  # head returns num_outputs

        self._init_weights()

    def _init_weights(self) -> None:
        """He init for convs, Xavier for linear, ones/zeros for norms."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, in_channels, T]`` -> ``[B, num_outputs]`` logits (no sigmoid)."""
        x = self.features(x)
        x = self.head(x)
        return x


def build_mobilevit_1d(
    width_multiplier: float = DEFAULT_WIDTH_MULTIPLIER,
    num_outputs: int = DEFAULT_NUM_CLASSES,
    in_channels: int = DEFAULT_INPUT_CHANNELS,
    stochastic_depth: float = DEFAULT_STOCHASTIC_DEPTH,
    head_expansion_channels: int = DEFAULT_HEAD_EXPANSION_CHANNELS,
    head_dropout: float = DEFAULT_HEAD_DROPOUT,
) -> MobileViT1D:
    """Convenience constructor for the default v1 architecture."""
    blocks = default_v1_blocks(
        num_outputs=num_outputs,
        head_expansion_channels=head_expansion_channels,
        head_dropout=head_dropout,
    )
    return MobileViT1D(
        blocks=blocks,
        width_multiplier=width_multiplier,
        in_channels=in_channels,
        stochastic_depth=stochastic_depth,
    )


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
