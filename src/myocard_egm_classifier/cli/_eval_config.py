"""Eval-CLI typed config + YAML builders.

Consumed only by ``egm-class-eval`` (via :mod:`.eval_cmd`). The
generic YAML/config helpers live in :mod:`._common` (shared with
train and export); the train-side equivalents of this module live in
:mod:`._train_config`.

What lives here:

- The two eval-side dataclasses ``EvalDataCLIConfig``,
  ``EvalOutputCLIConfig``, plus the bundled
  ``EvalExperimentConfig``.
- The per-block YAML builders and key allowlists.
- ``EvalCLIOverrides`` + ``apply_eval_overrides`` for argparse flags.
- ``eval_loader_kwargs_from_config`` (config -> kwargs for the
  sequential :class:`EGMTraceDataset` constructor used by the eval
  CLI; slimmer than the train-side equivalent since eval has no
  patient-aware split or augmentation).

The companion path-derivation helper for the output bank
(``default_predictions_bank_path``) lives in
:mod:`myocard_egm_classifier.eval.predictions` alongside
``populate_predictions``, since both deal with the predictions-bank
output concept rather than YAML config parsing.

Model architecture is rebuilt from the checkpoint's ``model_meta``
block (see :func:`myocard_egm_classifier.cli._common.load_checkpoint_model`),
so the eval YAML carries no ``model:`` block.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from myocard_egm_classifier.cli._common import (
    ConfigError,
    _optional,
    _reject_unknown_keys,
    _resolve_path,
)
from myocard_egm_classifier.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_ZNORM,
    DEFAULT_ZNORM_EPS,
)

DEFAULT_EVAL_THRESHOLD = 0.5
"""Default decision threshold for the eval CLI's ``label_pred``.

At ``0.5``, label_pred is invariant under any positive temperature
scaling of the logits (so deferring calibration to PR C doesn't shift
anything the eval CLI persists today). Override only when you have a
non-default ablation-flagging threshold from clinical or downstream
calibration analysis."""


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalDataCLIConfig:
    """Bank-side knobs for one evaluation run.

    Inference-time iteration of the bank is sequential — no
    patient-aware split, no augmentation — so this config carries only
    the kwargs that genuinely matter (z-score, input length, batch
    sizing, worker count).
    """

    bank: Path | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    znorm: bool = DEFAULT_ZNORM
    znorm_eps: float = DEFAULT_ZNORM_EPS
    pin_memory: bool = DEFAULT_PIN_MEMORY


@dataclass(frozen=True)
class EvalOutputCLIConfig:
    """Where the eval run's outputs land.

    ``predictions_bank`` is the sibling ClassifierBank that the CLI
    writes with ``ClassifierPrediction`` populated on every trace.
    When ``None``, the CLI derives a sibling path from the input
    bank: ``<input_stem>_pred.cbank.h5``.
    """

    predictions_bank: Path | None = None


@dataclass(frozen=True)
class EvalExperimentConfig:
    """Bundled typed config for ``egm-class-eval``.

    Model architecture is rebuilt from the checkpoint's ``model_meta``
    block, so the YAML doesn't carry a ``model:`` section — the
    checkpoint is the source of truth for what model is being eval'd.
    """

    checkpoint: Path
    data: EvalDataCLIConfig
    output: EvalOutputCLIConfig
    threshold: float = DEFAULT_EVAL_THRESHOLD


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

_EVAL_TOP_KEYS = {"checkpoint", "data", "output", "threshold"}
_EVAL_DATA_KEYS = {
    "bank",
    "batch_size",
    "num_workers",
    "znorm",
    "znorm_eps",
    "pin_memory",
}
_EVAL_OUTPUT_KEYS = {"predictions_bank"}


def build_eval_config(doc: dict[str, Any]) -> EvalExperimentConfig:
    """Translate a parsed YAML dict into a typed eval config.

    Validates each block's keys, rejects unknowns, resolves paths
    against the YAML file's directory, and enforces ``threshold`` in
    ``(0, 1)``.
    """
    _reject_unknown_keys(
        {k: v for k, v in doc.items() if k != "_config_dir"},
        _EVAL_TOP_KEYS,
        label="top-level",
    )
    cfg_dir: Path = doc["_config_dir"]

    checkpoint_raw = doc.get("checkpoint")
    if checkpoint_raw is None:
        raise ConfigError("checkpoint is required (path to a training checkpoint).")
    checkpoint = _resolve_path(str(checkpoint_raw), cfg_dir)
    if checkpoint is None:
        raise ConfigError("checkpoint path resolved to None; check the YAML value.")

    data = _build_eval_data_block(_optional(doc, "data", default={}) or {}, cfg_dir)
    if data.bank is None:
        raise ConfigError("data.bank is required (path to a labeled ClassifierBank HDF5).")

    output = _build_eval_output_block(_optional(doc, "output", default={}) or {}, cfg_dir)

    threshold = float(doc.get("threshold", DEFAULT_EVAL_THRESHOLD))
    if not (0.0 < threshold < 1.0):
        raise ConfigError(f"threshold must satisfy 0 < threshold < 1; got {threshold}.")

    return EvalExperimentConfig(
        checkpoint=checkpoint,
        data=data,
        output=output,
        threshold=threshold,
    )


def _build_eval_data_block(block: dict[str, Any], config_dir: Path) -> EvalDataCLIConfig:
    _reject_unknown_keys(block, _EVAL_DATA_KEYS, label="data")
    bank_raw = block.get("bank")
    bank = _resolve_path(str(bank_raw) if bank_raw is not None else None, config_dir)
    return EvalDataCLIConfig(
        bank=bank,
        batch_size=int(block.get("batch_size", DEFAULT_BATCH_SIZE)),
        num_workers=int(block.get("num_workers", DEFAULT_NUM_WORKERS)),
        znorm=bool(block.get("znorm", DEFAULT_ZNORM)),
        znorm_eps=float(block.get("znorm_eps", DEFAULT_ZNORM_EPS)),
        pin_memory=bool(block.get("pin_memory", DEFAULT_PIN_MEMORY)),
    )


def _build_eval_output_block(block: dict[str, Any], config_dir: Path) -> EvalOutputCLIConfig:
    _reject_unknown_keys(block, _EVAL_OUTPUT_KEYS, label="output")
    raw = block.get("predictions_bank")
    predictions_bank = _resolve_path(str(raw) if raw is not None else None, config_dir)
    return EvalOutputCLIConfig(predictions_bank=predictions_bank)


# ---------------------------------------------------------------------------
# CLI flag overrides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCLIOverrides:
    """Per-flag overrides applied on top of the YAML-loaded eval config."""

    bank: Path | None = None
    checkpoint: Path | None = None
    predictions_bank: Path | None = None
    threshold: float | None = None


def apply_eval_overrides(
    cfg: EvalExperimentConfig, overrides: EvalCLIOverrides
) -> EvalExperimentConfig:
    """Return a new :class:`EvalExperimentConfig` with overrides applied."""
    checkpoint = cfg.checkpoint
    data = cfg.data
    output = cfg.output
    threshold = cfg.threshold

    if overrides.checkpoint is not None:
        checkpoint = overrides.checkpoint.resolve()
    if overrides.bank is not None:
        data = replace(data, bank=overrides.bank.resolve())
    if overrides.predictions_bank is not None:
        output = replace(output, predictions_bank=overrides.predictions_bank.resolve())
    if overrides.threshold is not None:
        if not (0.0 < overrides.threshold < 1.0):
            raise ConfigError(
                f"--threshold must satisfy 0 < threshold < 1; got {overrides.threshold}."
            )
        threshold = overrides.threshold

    return EvalExperimentConfig(
        checkpoint=checkpoint, data=data, output=output, threshold=threshold
    )


# ---------------------------------------------------------------------------
# Runtime translators + output-path helper
# ---------------------------------------------------------------------------


def eval_loader_kwargs_from_config(cfg: EvalExperimentConfig, input_length: int) -> dict[str, Any]:
    """Keyword args for the eval-side :class:`EGMTraceDataset` construction.

    Eval iterates the bank sequentially — no patient-aware split, no
    train-time augmentation — so this is a slimmer set than
    :func:`myocard_egm_classifier.cli._train_config.loader_kwargs_from_config`.
    ``input_length`` is supplied by the caller from the checkpoint's
    ``model_meta`` since the eval YAML doesn't carry its own model block.
    """
    return {
        "input_length": input_length,
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "znorm": cfg.data.znorm,
        "znorm_eps": cfg.data.znorm_eps,
        "pin_memory": cfg.data.pin_memory,
    }
