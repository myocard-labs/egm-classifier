"""Per-patient stratification strategies for :func:`patient_aware_split`.

Two ship out of the box:

- :class:`AnyPositiveStrategy` (default) — one bit per patient (1 if any
  trace is positive, else 0). Recovers the original behavior on global-
  density banks; handles local-density banks by collapsing them to
  "patient has any fibrotic neighborhood" vs "patient is entirely
  healthy".
- :class:`BinnedDensityStrategy` — bucket the per-patient positive rate
  into ``n_bins`` equal-width bins. Finer-grained than ``any_positive``
  under local-density labels.

New strategies are easy to add: any frozen dataclass with a
``stratum(simulation_id, labels, pid) -> int`` method satisfies
:class:`PatientStratificationStrategy` (it's a structural Protocol,
not an inheritance hierarchy). Drop the new file alongside these two,
re-export it here, and the splitter picks it up by Protocol matching.
"""

from __future__ import annotations

from ._protocol import PatientStratificationStrategy
from .any_positive import AnyPositiveStrategy
from .binned_density import BinnedDensityStrategy

__all__ = [
    "AnyPositiveStrategy",
    "BinnedDensityStrategy",
    "PatientStratificationStrategy",
]
