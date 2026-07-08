"""``binned_density`` stratification — finer-grained class-rate bucketing.

For each patient, compute the per-trace positive rate ``mean(labels)``
and return the integer bin index in ``[0, n_bins - 1]``. Patients
within the same bin are considered equivalent for stratification
purposes, so the splitter distributes each bin proportionally across
train/val/test.

When this matters over
:class:`~myocard_egm_classifier.data.splits.strategies.any_positive.AnyPositiveStrategy`:

- **Local-density labels with a continuous fibrosis-density gradient.**
  Most patients have *some* fibrotic traces, so ``any_positive`` lumps
  them together regardless of how fibrotic they are. ``binned_density``
  separates lightly-fibrotic patients from heavily-fibrotic ones, so the
  val/test splits get a representative sample across the gradient.
- **When the bank is heavily skewed and ``any_positive`` still produces
  trace-class-imbalanced val/test splits.** The post-split diagnostic
  in :func:`~myocard_egm_classifier.data.datasets.build_dataloaders` warns when
  this happens; ``binned_density`` is the typical fix.

Degenerates to ``any_positive`` behavior on global-density banks:
every patient has rate 0.0 or 1.0, which always lands in the first
or last bin regardless of ``n_bins``, so any in-between bins are
empty and the stratification matches ``any_positive``.

Bin edges are uniformly spaced over ``[0, 1]``. Rate 0.0 lands in
bin 0; rate 1.0 lands in bin ``n_bins - 1`` (right edge closed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinnedDensityStrategy:
    """Patient stratum = bin index of the per-patient positive rate.

    Attributes
    ----------
    n_bins
        Number of equal-width bins over the rate range ``[0, 1]``. Must
        be ``>= 2``; ``n_bins == 2`` is equivalent to splitting at rate
        0.5 (and approximates ``any_positive`` for non-extreme banks).
        Default 3 partitions patients into low / medium / high fibrotic
        density.
    """

    n_bins: int = 3

    def __post_init__(self) -> None:
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be >= 2; got {self.n_bins}.")

    def stratum(
        self,
        simulation_id: np.ndarray,
        labels: np.ndarray,
        pid: int,
    ) -> int:
        pid_labels = labels[simulation_id == pid]
        rate = float(pid_labels.mean()) if pid_labels.size > 0 else 0.0
        # Right edge closed so rate=1.0 lands in the last bin rather
        # than overflowing past n_bins-1.
        idx = int(np.floor(rate * self.n_bins))
        return min(idx, self.n_bins - 1)
