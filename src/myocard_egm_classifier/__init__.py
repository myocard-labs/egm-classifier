"""myocard-egm-classifier — 1D MobileViT binary fibrosis classifier.

Subpackages:

- :mod:`~myocard_egm_classifier.models` — backbone + head + registry. The
  Phase-1 architecture is a 1D MobileViT with a single-logit BCE head;
  the registry pattern leaves room for future backbones / heads without
  changing the training loop's signature.
- :mod:`~myocard_egm_classifier.training` — training loop, metrics,
  reporting (run.json + metrics.csv + predictions writers delegated to
  ``myocard-egm-data``'s record writers).
- :mod:`~myocard_egm_classifier.eval` — evaluation on ClassifierBank.
  Transparent handling of labeled (synthetic / hybrid) vs unlabeled
  (real IAFDB) banks — see ``docs/usage.md`` for the catch-22 that
  motivates the split.
- :mod:`~myocard_egm_classifier.export` — ONNX export (optional
  ``[onnx]`` extra).

CLIs are wired in ``[project.scripts]``: ``egm-class-train``,
``egm-class-eval``, ``egm-class-export``.
"""

from __future__ import annotations

__version__ = "0.2.0"
