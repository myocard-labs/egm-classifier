"""MobileViT-1D block — the headline block (design doc Section 5).

The 1D translation of MobileViT's block (Mehta & Rastegari 2022, Sec. 3.2).
Five stages:

  1. Local rep conv   Conv1d(local_kernel_size) -> BN -> SiLU
  2. Channel expand   1x1 Conv1d -> transformer_dim (no activation)
  3. Unfold + L transformer layers + Fold
  4. Channel project  1x1 Conv1d -> in_channels (-> BN -> SiLU)
  5. Fusion           concat(input, global) -> Conv1d(fusion_kernel_size) -> in

Steps 1+5 are the "local representation learning" (CNN); step 3 is the
"global representation learning" (transformer). The skip-into-fusion (step
5) is what gives MobileViT its locality-preservation property: the input
features flow around the transformer and are recombined with the
attention-mixed features, so no temporal structure is lost.

The 1D unfold (design doc Section 2)
------------------------------------
We group **consecutive** ``p`` timesteps into a patch (matching the 2D
reference's consecutive-pixel patches), then attend across patches at each
intra-patch position. With ``T`` timesteps and patch size ``p``:

    N = T / p  patches,  p  positions-within-patch

    x  [B, d, T]
       reshape T -> (N, p)        [B, d, N, p]   (N slow, p fast = consecutive)
       permute                    [B, p, N, d]
       reshape                    [B*p, N, d]    transformer attends over N

Fold is the exact inverse. ``T`` must be divisible by ``p`` — the caller
chooses input length and strides so this always holds (design doc Sec. 7).

References
----------
- Mehta S, Rastegari M. "MobileViT." ICLR 2022. arxiv 2110.02178. Sec. 3.2.
- Nie Y et al. "A Time Series Is Worth 64 Words" (PatchTST). ICLR 2023.
  arxiv 2211.14730. Patch-based attention on 1D series.
"""

from __future__ import annotations

import torch
from torch import nn

from myocard_egm_classifier.models.blocks import Conv1dBNAct
from myocard_egm_classifier.models.transformer import TransformerBlock


class MobileViTBlock1d(nn.Module):
    """MobileViT-1D block: conv -> unfold -> transformer -> fold -> fuse.

    Parameters
    ----------
    in_channels
        Channel count C of the input feature map.
    transformer_dim
        Transformer model dimension d (the unfolded tensor's channel count
        and each token vector's length). Must be divisible by ``num_heads``.
    transformer_depth
        Number of stacked transformer layers L.
    num_heads
        Attention heads per transformer block.
    mlp_ratio
        Transformer MLP hidden expansion (MobileViT default 2.0).
    patch_size
        ``p`` — number of consecutive timesteps per patch. Design doc Sec. 2:
        4 at the first MobileViT block, 2 at deeper blocks.
    local_kernel_size
        Temporal kernel of the local-representation conv (step 1).
    fusion_kernel_size
        Temporal kernel of the fusion conv (step 5).
    dropout
        Dropout inside the transformer blocks.
    drop_path
        Stochastic-depth probability passed to the transformer sub-layers.
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        transformer_depth: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        patch_size: int = 2,
        local_kernel_size: int = 3,
        fusion_kernel_size: int = 3,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        """Build the five-stage local rep / lift / transformer / project / fuse stack.

        ``patch_size`` is stored on the module so :meth:`_unfold` and
        :meth:`_fold` can use it without re-passing it through forward.
        ``drop_path`` propagates into every transformer sub-layer so the
        network-wide stochastic-depth schedule reaches inside the block.
        """
        super().__init__()
        self.patch_size = patch_size

        # Step 1 + 2: local representation, then lift channels to d.
        self.local_rep = nn.Sequential(
            Conv1dBNAct(in_channels, in_channels, kernel_size=local_kernel_size, act=True),
            # 1x1 to transformer_dim; no activation — next op is the LN.
            Conv1dBNAct(in_channels, transformer_dim, kernel_size=1, act=False),
        )

        # Step 3: L transformer layers over the patch axis.
        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    dim=transformer_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    drop_path=drop_path,
                )
                for _ in range(transformer_depth)
            ]
        )

        # Step 4: project back to C.
        self.project = Conv1dBNAct(transformer_dim, in_channels, kernel_size=1, act=True)

        # Step 5: fuse concat(input, global) -> C.
        self.fuse = Conv1dBNAct(
            2 * in_channels, in_channels, kernel_size=fusion_kernel_size, act=True
        )

    def _unfold(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """[B, d, T] -> [B*p, N, d]. Returns the unfolded tensor and T."""
        B, d, T = x.shape
        p = self.patch_size
        if T % p != 0:
            raise ValueError(
                f"Sequence length T={T} is not divisible by patch size p={p}. "
                "Choose input length / strides so every MobileViT block sees a "
                "divisible T (design doc Section 7)."
            )
        N = T // p
        x = x.reshape(B, d, N, p)  # consecutive p timesteps per patch
        x = x.permute(0, 3, 2, 1)  # [B, p, N, d]
        x = x.reshape(B * p, N, d)  # [B*p, N, d]
        return x, T

    def _fold(self, x: torch.Tensor, B: int, d: int, T: int) -> torch.Tensor:
        """[B*p, N, d] -> [B, d, T]. Inverse of _unfold."""
        p = self.patch_size
        N = T // p
        x = x.reshape(B, p, N, d)  # [B, p, N, d]
        x = x.permute(0, 3, 2, 1)  # [B, d, N, p]
        x = x.reshape(B, d, T)  # [B, d, T]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor ``[B, in_channels, T]``.

        Returns
        -------
        Tensor ``[B, in_channels, T]`` — same shape (downsampling between
        stages is the MV2 blocks' job, not this block's).
        """
        identity = x
        B = x.shape[0]

        # Local rep + channel lift: [B, C, T] -> [B, d, T].
        x = self.local_rep(x)
        d = x.shape[1]

        # Unfold -> transformer -> fold (T preserved).
        x, T = self._unfold(x)
        x = self.transformer(x)
        x = self._fold(x, B, d, T)

        # Project back to C and fuse with the original input.
        x = self.project(x)  # [B, C, T]
        x = torch.cat([identity, x], dim=1)  # [B, 2C, T]
        x = self.fuse(x)  # [B, C, T]
        return x
