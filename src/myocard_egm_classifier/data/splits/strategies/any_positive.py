"""``any_positive`` stratification — one bit per patient.

For each patient, the stratum is ``1`` if any trace has a positive
label, else ``0``. Two equivalence classes total: "patient contains
some fibrotic substrate" vs "patient is entirely healthy".

Why this is the default:

- **Global-density labels** make every patient single-class by
  construction (the global density is either above or below threshold).
  Under this strategy, the per-patient stratum exactly recovers the
  per-patient label — the splitter behaves identically to the original
  pre-strategy implementation.
- **Local-density labels** make a patient's traces span both classes;
  the per-patient label concept no longer exists. ``any_positive``
  collapses this to "does this patient have any fibrotic neighborhood
  at all?", which is the coarsest non-trivial stratification: it
  guarantees that splits don't accidentally end up entirely free of
  fibrotic patients.

The coarseness is the trade-off: under local-density, a patient with
1% positive traces gets stratified the same way as a patient with 50%
positive traces. If your bank skews heavily one way and the resulting
val/test splits still come out class-imbalanced, switch to
:class:`~myocard_egm_classifier.data.splits.strategies.binned_density.BinnedDensityStrategy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnyPositiveStrategy:
    """Patient stratum = 1 if any trace is positive, else 0.

    See module docstring for the design rationale.
    """

    def stratum(
        self,
        simulation_id: np.ndarray,
        labels: np.ndarray,
        pid: int,
    ) -> int:
        pid_labels = labels[simulation_id == pid]
        return int((pid_labels == 1).any())
