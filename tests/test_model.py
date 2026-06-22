"""Architecture sanity checks for the 1D MobileViT model.

We don't test the inner block math here — those tests would belong in
``test_blocks.py`` against the registry — but we do verify the public
end-to-end contract: the default-v1 model takes ``[B, 1, T]`` and
produces ``[B, num_outputs]`` for both the binary and multi-class head
configurations, the width multiplier scales parameter count
monotonically, and ``count_parameters`` returns a sensible nonzero.
"""

from __future__ import annotations

import torch

from myocard_egm_classifier.models import (
    MobileViT1D,
    count_parameters,
    default_v1_blocks,
)


def _build(width: float, num_outputs: int) -> MobileViT1D:
    """Build a default-v1 model at the requested width / output count."""
    return MobileViT1D(
        blocks=default_v1_blocks(num_outputs=num_outputs),
        width_multiplier=width,
        in_channels=1,
        stochastic_depth=0.0,  # deterministic forward for the test
    )


def test_forward_shape_binary() -> None:
    """Single-logit BCE head emits ``[B, 1]`` from a ``[B, 1, T]`` input."""
    model = _build(width=1.0, num_outputs=1).eval()
    x = torch.zeros(3, 1, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (3, 1)
    assert y.dtype == torch.float32


def test_forward_shape_multiclass() -> None:
    """Multi-class softmax head emits ``[B, num_outputs]`` for ``num_outputs >= 2``."""
    model = _build(width=1.0, num_outputs=4).eval()
    x = torch.zeros(2, 1, 512)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 4)


def test_width_multiplier_scales_parameters() -> None:
    """Bigger ``width_multiplier`` => strictly more parameters."""
    small = _build(width=0.5, num_outputs=1)
    base = _build(width=1.0, num_outputs=1)
    large = _build(width=1.5, num_outputs=1)
    n_small = count_parameters(small)
    n_base = count_parameters(base)
    n_large = count_parameters(large)
    assert n_small > 0
    assert n_small < n_base < n_large


def test_forward_handles_short_input() -> None:
    """T=128 still produces ``[B, num_outputs]`` (model is fully convolutional pre-head)."""
    model = _build(width=1.0, num_outputs=1).eval()
    x = torch.randn(1, 1, 128)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1)
