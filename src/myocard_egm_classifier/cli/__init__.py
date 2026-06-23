"""CLI entry points for the classifier commands.

Three console scripts wired in ``[project.scripts]``:

- ``egm-class-train CONFIG.yaml`` — train a model end-to-end against
  one ClassifierBank.
- ``egm-class-eval CONFIG.yaml`` — load a checkpoint, predict against
  a ClassifierBank, write metrics + predictions.
- ``egm-class-export CONFIG.yaml`` — export a trained checkpoint to
  ONNX.

All three are YAML-config-driven (matching iafdb-pipeline and
synthetic-egm-pipeline). The CLI-internal modules split by concern:

- :mod:`._common` — shared YAML/runtime helpers, the model-config
  triad, the checkpoint loader. Pulled into every CLI.
- :mod:`._train_config` — train-only typed dataclasses + YAML builders
  + ``loader_kwargs_from_config`` + run-record serializer.
- :mod:`._eval_config` — eval-only typed dataclasses + YAML builders
  + ``eval_loader_kwargs_from_config``.
- :mod:`._export_config` (PR C) — export-only typed dataclasses.

Functional code that consumes the eval config (dataset construction,
prediction stamping) lives in :mod:`myocard_egm_classifier.eval`;
``eval_cmd`` is intentionally thin (argparse + main orchestration).
"""

from __future__ import annotations
