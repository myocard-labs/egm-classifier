"""Per-trace normalize/pad/augment transform.

Pipeline per raw trace of valid length ``L``:

  1. (train only) random gain   x <- x * U(1-g, 1+g)
  2. per-trace z-score          x <- (x - mean) / max(std, eps)
  3. place into a zero buffer of length T at an offset; (train only) the
     offset is jittered by up to +/- shift_frac*L for time-shift aug.

Two notes worth internalizing:

- **Gain is amplitude, z-score is amplitude-invariant.** Step 2
  mathematically removes any positive scalar from step 1, so random gain
  is a near-no-op while z-score normalization is on. It is implemented
  faithfully (the Phase-1 design lists it) and kept for the regime where
  normalization is disabled or for future non-normalized variants; with
  z-score on, *time-shift* is the augmentation that actually does work.
  The clinical rationale is that absolute amplitude reflects electrode
  contact, not substrate, so discarding it is intended.
- **Zero-pad == pad-with-the-mean (only when L < T).** Because we z-score
  the *valid* region before placing it (not the whole padded buffer),
  the active samples have mean 0 and any padding zeros equal that mean —
  no DC step at the boundary, and a fully-flat (degenerate) trace is
  handled by the std floor instead of dividing by zero. When L == T
  there is no padding and placement is a straight copy.

Noise augmentation is *already baked into the hybrid bank* (each trace
is clean+noise at some SNR), so there is no runtime noise mixing here.

**Scope:** ``TraceTransform`` is for per-call augmentations that vary
between training runs. Static per-file or once-per-dataset preprocessing
(bandpass, resampling, clipping, calibration) belongs in the producer
pipelines or a separate curation module — not here.

**Known limitation:** the time-shift jitter (step 3) is a silent no-op
when ``valid_len == input_length``, which is the case for the Phase-1
banks (both ship ``L = T = 512``). The shift was originally designed
for an ``L = 200`` bank with headroom inside ``T``. The intended fix is
producer-side: emit raw traces longer than ``T`` and take a random
length-``T`` crop here at train time. Tracked for Phase 1.5+; see
``project/trace_transform_review.md`` for the full discussion.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TraceTransform:
    """Configurable normalize/pad/augment applied per trace.

    No field has a default. Callers (typically egm-classifier's training
    config layer) must explicitly choose every knob — the library does
    not pre-decide policy.

    Parameters
    ----------
    input_length
        Target length T after zero-padding.
    znorm
        Apply per-trace z-score.
    znorm_eps
        Std floor to avoid divide-by-zero on degenerate traces.
    augment
        Master switch for train-time augmentation (gain + time-shift).
    max_gain
        Random gain is U(1-max_gain, 1+max_gain). e.g. 0.2 = +/-20%.
    max_shift_frac
        Time-shift jitter is up to +/- this fraction of the valid length
        (e.g. 0.10 = +/-10%).
    """

    input_length: int
    znorm: bool
    znorm_eps: float
    augment: bool
    max_gain: float
    max_shift_frac: float

    def __post_init__(self) -> None:
        """Warn when the requested config has algebraically-redundant steps.

        Random gain followed by per-trace z-score is a no-op: if x is
        scaled by any positive c, znorm(c * x) == znorm(x) exactly, so
        the gain draw is wasted work. We still run gracefully (skipping
        the gain in __call__) but flag the misconfiguration loudly so a
        caller has a chance to fix it.
        """
        if self.znorm and self.augment and self.max_gain > 0.0:
            warnings.warn(
                "TraceTransform: random gain is a no-op when znorm=True "
                "(z-score removes any positive scalar). Skipping gain "
                "step. Set max_gain=0.0 or znorm=False to silence.",
                stacklevel=2,
            )

    def __call__(self, trace: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return a length-``input_length`` float32 array for one trace."""
        x = np.asarray(trace, dtype=np.float32)
        valid_len = x.shape[0]
        target = self.input_length

        if valid_len > target:
            # Defensive: center-crop if a future bank is longer than T.
            # Written as (mid - half_target) rather than (excess // 2) so
            # the result truly centers when valid_len and target have
            # different parity (the two formulas can differ by 1 in that
            # case; for the current T=512 / even target they agree).
            start = (valid_len // 2) - (target // 2)
            x = x[start : start + target]
            valid_len = target

        # 1. Random gain (train only). Skipped when znorm is on because
        # z-score is amplitude-invariant; see __post_init__ for the warning.
        if self.augment and self.max_gain > 0.0 and not self.znorm:
            x = x * (1.0 + rng.uniform(-self.max_gain, self.max_gain))

        # 2. Per-trace z-score over the valid region.
        if self.znorm:
            mean = float(x.mean())
            std = float(x.std())
            x = (x - mean) / max(std, self.znorm_eps)

        # 3. Place into a zero buffer at a (jittered) offset.
        out = np.zeros(target, dtype=np.float32)
        base_offset = (target - valid_len) // 2
        offset = base_offset
        if self.augment and self.max_shift_frac > 0.0:
            max_shift = round(self.max_shift_frac * valid_len)
            if max_shift > 0:
                offset += int(rng.integers(-max_shift, max_shift + 1))
        offset = int(np.clip(offset, 0, target - valid_len))
        out[offset : offset + valid_len] = x
        return out
