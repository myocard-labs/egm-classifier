"""Fit the temperature-scaling scalar ``T`` against a labeled bank.

The export CLI's optional calibration step:

1. Build a sequential, no-augment dataset over the calibration bank.
2. Apply the configured per-trace normalization scheme to each batch
   (zscore / zero2one / none) so the inputs match what the deployed
   runtime will see.
3. Forward each batch through the trained model; collect logits.
4. Hand ``(logits, labels)`` to
   :func:`myocard_egm_signal.model.temperature_scaling.fit_temperature`
   for the bounded scalar minimization that yields ``T``.

Normalization lives here (rather than inside :class:`TraceTransform`)
because TraceTransform currently only supports z-score, and the v1
classifier's training side hasn't been wired for zero2one yet. By
applying the normalization step explicitly in the calibration loop
we can support both schemes today; once training/eval grow zero2one
support (post-refactor task), TraceTransform will absorb this logic
and the calibration loop becomes a one-liner.

The model is NOT wrapped with :class:`CalibratedModel` while fitting
``T`` — that wrapping happens at export time, after this function has
returned the fitted ``T``. The fit needs raw logits as input.
"""

from __future__ import annotations

import numpy as np
import torch
from myocard_egm_data.augmentation import TraceTransform
from myocard_egm_data.banks import ClassifierBank
from myocard_egm_data.datasets import EGMTraceDataset
from myocard_egm_signal.model.temperature_scaling import fit_temperature
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from myocard_egm_classifier.constants import DEFAULT_SEED


def _apply_normalization(batch: torch.Tensor, scheme: str, eps: float) -> torch.Tensor:
    """Apply the configured per-trace normalization to a ``[B, C, T]`` batch.

    Parameters
    ----------
    batch
        Float tensor shaped ``[batch, channels, time]``. For the v1
        binary head ``channels == 1`` (the Conv1d axis).
    scheme
        One of ``"zscore"``, ``"zero2one"``, ``"none"``. Validated by
        the caller against the same enum the metadata schema uses.
    eps
        Strictly positive floor on the per-trace divisor (std for
        zscore; max-min for zero2one). Avoids division by zero on
        degenerate flat traces.

    Returns
    -------
    torch.Tensor
        New tensor of the same shape as ``batch`` with the chosen
        scheme applied per trace along the time axis.
    """
    if scheme == "none":
        return batch
    if scheme == "zscore":
        # Per-trace mean/std along the time axis; keepdim for broadcasting.
        mean = batch.mean(dim=-1, keepdim=True)
        std = batch.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (batch - mean) / std
    if scheme == "zero2one":
        lo = batch.amin(dim=-1, keepdim=True)
        hi = batch.amax(dim=-1, keepdim=True)
        denom = (hi - lo).clamp_min(eps)
        return (batch - lo) / denom
    raise ValueError(f"Unknown normalization scheme: {scheme!r}.")


def _build_calibration_dataset(bank: ClassifierBank, input_length: int) -> EGMTraceDataset:
    """Sequential :class:`EGMTraceDataset` over every trace of a bank.

    Uses :class:`TraceTransform` with normalization + augmentation
    DISABLED — we apply normalization ourselves in the calibration
    loop so we can switch on the configured scheme. The TraceTransform
    here just handles pad/crop to ``input_length``.
    """
    signal = bank.signal_array()
    labels = bank.label_truth_array()
    transform = TraceTransform(
        input_length=input_length,
        znorm=False,
        znorm_eps=1.0,  # unused when znorm=False; satisfies the positive-eps invariant
        augment=False,
        max_gain=0.0,
        max_shift_frac=0.0,
    )
    return EGMTraceDataset(
        signal=signal,
        labels=labels,
        indices=np.arange(bank.n_traces),
        transform=transform,
        label_dtype=torch.float32,
        seed=DEFAULT_SEED,
    )


@torch.no_grad()
def fit_temperature_from_bank(
    model: nn.Module,
    bank: ClassifierBank,
    *,
    input_length: int,
    normalization_scheme: str,
    normalization_eps: float,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: torch.device,
) -> float:
    """Fit ``T`` against a labeled bank and return it.

    The model is run in eval mode under ``torch.no_grad``; ``T`` is
    fitted in pure numpy via
    :func:`myocard_egm_signal.model.temperature_scaling.fit_temperature`.

    Parameters
    ----------
    model
        The trained classifier. Must produce per-trace logits shaped
        ``[batch, 1]`` (single-logit binary head); the call flattens
        to ``[N]`` before fitting.
    bank
        Labeled :class:`ClassifierBank` to fit against.
    input_length
        The model's expected per-trace length (from the checkpoint's
        ``model_meta["input_length"]``). Drives the TraceTransform's
        pad/crop step.
    normalization_scheme, normalization_eps
        The deployment-side normalization the runtime will apply.
        Calibration mirrors it so the fitted ``T`` matches what
        deployment will see.
    batch_size, num_workers, pin_memory
        DataLoader knobs forwarded from the YAML's ``calibration``
        block.
    device
        Device the model already lives on; batches are moved here
        before the forward pass.

    Returns
    -------
    float
        The fitted ``T``. Always strictly positive (the underlying
        scalar minimizer is bounded away from zero).
    """
    model.eval()
    dataset = _build_calibration_dataset(bank, input_length=input_length)
    loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=False,
    )

    all_logits: list[NDArray[np.float32]] = []
    all_labels: list[NDArray[np.int64]] = []
    for signals, targets in loader:
        signals = signals.to(device, non_blocking=True)
        signals = _apply_normalization(signals, normalization_scheme, normalization_eps)
        logits = model(signals).float().cpu().numpy()
        all_logits.append(logits)
        all_labels.append(targets.cpu().numpy())

    if not all_logits:
        raise ValueError("Calibration bank yielded zero batches; cannot fit temperature.")

    logits = np.concatenate(all_logits, axis=0).reshape(-1).astype(np.float64)
    labels = np.concatenate(all_labels, axis=0).reshape(-1).astype(np.int64)
    return fit_temperature(logits, labels)
