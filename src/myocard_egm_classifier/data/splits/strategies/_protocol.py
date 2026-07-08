"""Protocol for per-patient stratification strategies.

Patient-aware splitting has two responsibilities:

1. **Non-leakage** — every trace from one patient lands in exactly one
   split. Hard-coded in :func:`patient_aware_split`; not negotiable.
2. **Stratification** — patients with similar label distribution end up
   distributed proportionally across train/val/test, so a small,
   class-imbalanced bank doesn't accidentally land all positives in
   one split. This is what each strategy customizes.

A strategy is anything with a ``stratum(simulation_id, labels, pid)``
method that returns an integer key. Patients sharing the same key are
considered "equivalent" for stratification purposes — the splitter
allocates them proportionally across the three splits.

The Protocol is structural (no abstract base class to inherit from);
any frozen dataclass with the right method satisfies it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PatientStratificationStrategy(Protocol):
    """How to compute a stratification key for one patient.

    Implementations should be pure functions of ``(simulation_id, labels,
    pid)``: read ``labels[simulation_id == pid]`` and return an integer
    bucket. The splitter groups patients by bucket and allocates each
    bucket independently into train/val/test, then unions the per-bucket
    assignments.

    Convention: return ``int`` — small integer keys (0, 1, ...) keep the
    splitter's per-bucket bookkeeping simple. The semantics of "bucket 0
    vs bucket 1" are entirely the strategy's choice.
    """

    def stratum(
        self,
        simulation_id: np.ndarray,
        labels: np.ndarray,
        pid: int,
    ) -> int:
        """Return the stratification bucket for patient ``pid``.

        Parameters
        ----------
        simulation_id
            ``[N]`` per-trace patient id (integer-factorized upstream).
        labels
            ``[N]`` per-trace integer label (typically binary 0/1).
        pid
            The patient id whose stratum is being computed.
        """
        ...
