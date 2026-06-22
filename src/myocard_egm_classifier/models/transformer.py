"""Pre-LayerNorm transformer encoder block (token axis = patches).

The standard ViT / MobileViT transformer encoder layer — the transformer
itself never cared about spatial-vs-temporal, it just mixes a sequence
of token vectors, so the 2D-to-1D conversion is free here. The only
addition vs. the textbook recipe is DropPath on the two residual
sub-layers so the network-wide stochastic-depth schedule (design doc
Section 10) reaches inside the MobileViT blocks too.

Layout (pre-LN; what ViT and MobileViT use)::

    x -> LN -> MHSA -> DropPath -> + x -> LN -> MLP -> DropPath -> + x

Shapes for one block (per the notation feedback: explicit shapes)::

    x        [B, N, D]   (B = batch*patch-positions, N = num patches, D = dim)
    LN1(x)   [B, N, D]
    MHSA     [B, N, D]
    MLP      [B, N, D]   (D -> mlp_ratio*D -> D, GELU between)

References
----------
- Vaswani A et al. "Attention Is All You Need." NeurIPS 2017. arxiv 1706.03762.
- Dosovitskiy A et al. "An Image Is Worth 16x16 Words" (ViT). ICLR 2021.
  arxiv 2010.11929. Pre-LN ordering.
- Mehta S, Rastegari M. "MobileViT." ICLR 2022. arxiv 2110.02178.
"""

from __future__ import annotations

import torch
from torch import nn

from myocard_egm_classifier.models.blocks import DropPath


class TransformerBlock(nn.Module):
    """Pre-LN transformer encoder layer with stochastic depth.

    Parameters
    ----------
    dim
        Token embedding dimension D. Must be divisible by ``num_heads``.
    num_heads
        Attention heads H; per-head dim = D / H.
    mlp_ratio
        MLP hidden expansion. MobileViT uses 2.0 (vs 4.0 in ViT/Vaswani).
    dropout
        Dropout after attention and inside the MLP.
    attn_dropout
        Dropout on the attention weights.
    drop_path
        Stochastic-depth probability for both residual sub-layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        """Build the LN + MHSA + MLP sub-layers and their stochastic-depth gates.

        Enforces ``dim % num_heads == 0`` up front so per-head dim is an
        integer. The attention uses ``batch_first=True`` because we feed
        ``[B, N, D]`` from the MobileViT block's unfold.
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,  # we feed [B, N, D]
        )
        self.drop1 = nn.Dropout(dropout)
        self.drop_path1 = DropPath(drop_path)

        hidden_dim = int(dim * mlp_ratio)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # ViT/MobileViT use GELU in the MLP (design doc Sec. 10)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run pre-LN attention + MLP on tokens ``[B, N, D]``; shape preserved."""
        # Self-attention sub-layer (pre-LN + residual).
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.drop_path1(self.drop1(attn_out))
        # MLP sub-layer (pre-LN + residual), applied position-wise.
        x = x + self.drop_path2(self.mlp(self.ln2(x)))
        return x
