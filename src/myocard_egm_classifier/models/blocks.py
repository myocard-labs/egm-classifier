"""1D convolutional building blocks for the EGM MobileViT.

The stem and inverted-residual blocks are 1D translations of the 2D
versions in MobileNetV2 (Sandler 2018) and MobileViT (Mehta & Rastegari
2022), plus the stochastic-depth and classifier-head pieces the design
document calls for. Every 2D op is replaced by its 1D counterpart
applied along the *temporal* axis (design doc Section 1): ``Conv2d`` ->
``Conv1d``, ``BatchNorm2d`` -> ``BatchNorm1d``, spatial kernels become
temporal kernels. We name the convs "temporal" at the call sites so the
intent is obvious.

Shapes use the convention ``[B, C, T]`` (batch, channels, time samples).

References
----------
- Sandler M et al. "MobileNetV2: Inverted Residuals and Linear
  Bottlenecks." CVPR 2018. arxiv 1801.04381. (MV2 block, Sec. 3.)
- Huang G et al. "Deep Networks with Stochastic Depth." ECCV 2016.
  arxiv 1603.09382. (DropPath, design doc Section 10.)
"""

from __future__ import annotations

import torch
from torch import nn


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round ``value`` to the nearest multiple of ``divisor`` (design doc Sec. 8).

    Width multipliers produce fractional channel counts; rounding to a
    multiple of 8 keeps convolutions hardware-aligned (and matches the
    MobileNet convention). Never rounds down by more than 10%.
    """
    if min_value is None:
        min_value = divisor
    rounded = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:  # don't drop more than 10% of the channels
        rounded += divisor
    return int(rounded)


class DropPath(nn.Module):
    """Per-sample stochastic depth (design doc Section 10).

    Drops the *residual branch* for a random subset of the batch at train
    time, scaling the survivors by ``1 / keep_prob`` so the expected value
    is unchanged. At eval time it is the identity — which is also why ONNX
    export (done in eval mode) traces it away cleanly.

    Parameters
    ----------
    drop_prob
        Probability of dropping the branch for a given sample. 0.0 = no-op.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Store the per-sample drop probability and validate its range."""
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Drop ``x`` per-sample at train time; identity at eval time.

        At training time: each sample in the batch is kept with
        probability ``1 - drop_prob``; survivors are scaled by
        ``1 / keep_prob`` so the expected activation is unchanged.
        At eval time (or ``drop_prob == 0``): returns ``x`` unchanged.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # One Bernoulli draw per sample, broadcast over the C and T axes.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        """Include ``drop_prob`` in the module's ``repr()`` for debugging."""
        return f"drop_prob={self.drop_prob}"


class Conv1dBNAct(nn.Module):
    """Temporal Conv1d -> BatchNorm1d -> (optional) SiLU.

    The workhorse conv unit. ``act=False`` gives the "linear" conv used at
    the MobileViT channel-projection and MV2 linear-bottleneck (no
    nonlinearity, per the respective papers).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        """Build the bias-free Conv1d, BatchNorm1d, and (optional) SiLU stack.

        Padding is set to ``kernel_size // 2`` so the temporal length is
        preserved when ``stride == 1``. ``groups`` controls depth-wise vs
        standard convs (set ``groups == in_channels`` for depth-wise).
        """
        super().__init__()
        padding = kernel_size // 2  # "same" length when stride == 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Conv1d -> BatchNorm1d -> (SiLU | Identity) to ``[B, C, T]``."""
        # nn.Sequential / individual nn modules return Any in torch's stubs;
        # the actual runtime type is Tensor, so silence strict mypy here.
        out: torch.Tensor = self.act(self.bn(self.conv(x)))
        return out


class MV2Block1d(nn.Module):
    """MobileNetV2-1D inverted residual block (design doc Section 4).

    1x1 expand -> depthwise *temporal* conv -> 1x1 linear projection.
    BatchNorm1d after each conv; SiLU after the first two; the projection
    is linear (no activation — Sandler 2018 Sec. 3.2). The residual is
    added only when ``stride == 1 and in_channels == out_channels``.

    Data flow ``[B, C, T]``::

        in        [B, C_in,        T]
        expand    [B, C_in*t,      T]      (skipped when t == 1)
        depthwise [B, C_in*t,      T/s]    (s = stride, groups = C_in*t)
        project   [B, C_out,       T/s]
        + skip    [B, C_out,       T]      iff stride==1 and C_in==C_out

    Parameters
    ----------
    in_channels, out_channels
        Channel counts (already width-scaled by the caller).
    stride
        Temporal stride of the depthwise conv (1 or 2).
    expand_ratio
        Inverted-residual expansion factor ``t`` (MobileViT uses 2 or 4).
    kernel_size
        Depthwise temporal kernel. 3 by default (design doc Section 4).
    drop_path
        Stochastic-depth probability for this block's residual branch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4,
        kernel_size: int = 3,
        drop_path: float = 0.0,
    ) -> None:
        """Build the expand -> depthwise -> project stack + residual gate.

        The residual is enabled iff ``stride == 1`` and
        ``in_channels == out_channels`` (otherwise the input and output
        shapes don't match and we can't sum them). When the residual is
        live, ``drop_path`` is applied to the block branch; otherwise
        ``self.drop_path`` is ``nn.Identity``.
        """
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        self.use_skip = stride == 1 and in_channels == out_channels
        hidden_channels = in_channels * expand_ratio

        layers: list[nn.Module] = []
        # 1x1 expand (no-op when expand_ratio == 1).
        if expand_ratio != 1:
            layers.append(Conv1dBNAct(in_channels, hidden_channels, kernel_size=1, act=True))
        # Depthwise temporal conv (groups == channels).
        layers.append(
            Conv1dBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_channels,
                act=True,
            )
        )
        # 1x1 linear projection (the linear bottleneck — no activation).
        layers.append(Conv1dBNAct(hidden_channels, out_channels, kernel_size=1, act=False))
        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if self.use_skip else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MV2 stack on ``[B, C_in, T]``; add the residual when shapes match."""
        # nn.Sequential returns Any in torch's stubs; annotate explicitly
        # so strict mypy is happy with the Tensor return type.
        out: torch.Tensor = self.block(x)
        if self.use_skip:
            out = x + self.drop_path(out)
        return out


class ClassifierHead1d(nn.Module):
    """Channel-expansion + global pool + linear classifier (design doc Sec. 6).

    1x1 conv -> BatchNorm1d -> SiLU -> global average pool over T ->
    Dropout -> Linear to ``num_outputs``.

    For the v1 binary task ``num_outputs == 1`` (single logit, BCE). For
    the Phase 2 multi-class severity head, ``num_outputs == n_classes``.
    See the reconciliation note in the ``models`` package docstring.
    """

    def __init__(
        self,
        in_channels: int,
        expansion_channels: int,
        num_outputs: int,
        dropout: float = 0.1,
    ) -> None:
        """Build the channel-expand conv, global average pool, dropout, and linear classifier."""
        super().__init__()
        self.expand = Conv1dBNAct(in_channels, expansion_channels, kernel_size=1, act=True)
        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, C, T] -> [B, C, 1]
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(expansion_channels, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a feature map ``[B, C, T]`` to logits ``[B, num_outputs]`` (no sigmoid)."""
        x = self.expand(x)  # [B, C_exp, T]
        x = self.pool(x).flatten(1)  # [B, C_exp]
        x = self.dropout(x)
        # nn.Linear returns Any in torch's stubs; the runtime type is Tensor.
        logits: torch.Tensor = self.fc(x)  # [B, num_outputs]
        return logits
