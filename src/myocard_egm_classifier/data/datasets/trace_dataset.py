"""Per-trace EGM ``Dataset`` for binary fibrosis classification.

Holds the (already in-memory) signal and label arrays plus the row
indices belonging to this split, and applies a
:class:`~myocard_egm_classifier.data.augmentation.TraceTransform` on access. Returns
``(signal, label)`` where ``signal`` is a ``[1, T]`` float32 tensor
(single channel) and ``label`` is a scalar.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from ..augmentation import TraceTransform


class EGMTraceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Per-trace dataset over a bank's signal + label arrays.

    Parameters
    ----------
    signal
        ``[N, L]`` float array of all traces in the bank.
    labels
        ``[N]`` int label per trace.
    indices
        Row indices selecting this split's traces.
    transform
        The per-trace transform.
    label_dtype
        ``torch.float32`` for the single-logit BCE head (label shape [1]),
        ``torch.long`` for the multi-class cross-entropy head.
    seed
        Base RNG seed; augmentation draws are deterministic given (seed,
        epoch, index) so runs are reproducible. Call :meth:`set_epoch`
        each epoch to reshuffle the augmentation noise.
    """

    def __init__(
        self,
        signal: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        transform: TraceTransform,
        label_dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> None:
        self.signal = signal
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.label_dtype = label_dtype
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation RNG stream (call once per epoch)."""
        self._epoch = epoch

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = int(self.indices[i])
        # Deterministic per (seed, epoch, row) augmentation stream.
        rng = np.random.default_rng((self.seed, self._epoch, row))
        x = self.transform(self.signal[row], rng)
        signal = torch.from_numpy(x).unsqueeze(0)  # [1, T]

        if self.label_dtype == torch.float32:
            label = torch.tensor([float(self.labels[row])], dtype=torch.float32)  # [1]
        else:
            label = torch.tensor(int(self.labels[row]), dtype=self.label_dtype)  # scalar
        return signal, label
