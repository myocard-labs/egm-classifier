"""Inference helper utilities shared by training and eval.

Currently a single function: :func:`collect_logits` — given a model
and a DataLoader, run a forward pass over every batch and return the
per-sample logits + labels as numpy arrays in batch-traversal order.

Lives at the package top level (not under ``training/``) because both
the training loop (per-epoch val + final test) and the eval CLI
consume it. Putting it under ``training/`` would imply training-only
ownership, which isn't true. Kept in its own module (rather than
folded into ``metrics.py``) so the model-runner concern stays
separable as more inference helpers land — e.g. probability
collection in PR C's calibration fit, or per-batch streaming variants.

The function takes ``torch.no_grad`` itself and switches the model to
eval mode internally, so callers don't have to remember either.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader[Any], device: torch.device
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Run ``model`` over every batch in ``loader``; return ``(logits, labels)``.

    Returns
    -------
    logits : ``[N, C]`` float32 — model outputs concatenated in
        batch-traversal order. ``C`` is the model's output dim
        (1 for the single-logit binary head).
    labels : ``[N]`` int64 — ground-truth labels concatenated in the
        same order.

    The loader must be deterministic-order (``shuffle=False`` or a
    seeded shuffle) if the caller plans to re-associate logits with
    a specific row of the source bank — the eval CLI relies on this
    invariant to stamp predictions in lock-step with bank traces.

    Empty loader returns an empty pair of arrays (``logits`` shape
    ``(0, 1)``, ``labels`` shape ``(0,)``); callers should usually
    treat that as a configuration error rather than a valid no-op.
    """
    model.eval()
    all_logits: list[NDArray[np.float32]] = []
    all_labels: list[NDArray[np.int64]] = []
    for signals, targets in loader:
        signals = signals.to(device, non_blocking=True)
        logits = model(signals).float().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(targets.cpu().numpy())
    if not all_logits:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0).reshape(-1).astype(np.int64),
    )
