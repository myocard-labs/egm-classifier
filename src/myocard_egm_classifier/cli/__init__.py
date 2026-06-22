"""CLI entry points for the classifier commands.

Three console scripts wired in ``[project.scripts]``:

- ``egm-class-train CONFIG.yaml`` — train a model end-to-end against
  one ClassifierBank.
- ``egm-class-eval CONFIG.yaml`` — load a checkpoint, predict against
  a ClassifierBank, write metrics + predictions.
- ``egm-class-export CONFIG.yaml`` — export a trained checkpoint to
  ONNX.

All three are YAML-config-driven (matching iafdb-pipeline and
synthetic-egm-pipeline). The shared YAML loader + typed config
dataclasses live in :mod:`._config`.
"""

from __future__ import annotations
