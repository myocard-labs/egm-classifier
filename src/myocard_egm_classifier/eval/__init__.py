"""Evaluation against ClassifierBank-format inputs.

Transparently handles two cases:

- **Labeled banks** (synthetic clean, synthetic hybrid) — emits the
  usual classification metrics (accuracy, AUROC, ECE, per-class
  precision/recall/F1) plus per-trace predictions.
- **Unlabeled / single-class banks** (e.g. IAFDB-only with the
  all-healthy labeling assumption) — emits per-trace predictions +
  confidence-calibration plots only. No accuracy/AUROC. The CLI's
  output is explicit about which kind of eval ran.

See ``docs/usage.md`` and ``project/architecture.md`` for the
IAFDB epistemic catch-22 that motivates this split — using IAFDB as
ground-truth test data is logically circular because we don't have
trustworthy fibrosis labels for it.
"""

from __future__ import annotations
