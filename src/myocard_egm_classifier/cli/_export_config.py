"""Export-CLI typed config + YAML builders.

Consumed only by ``egm-class-export`` (via :mod:`.export_cmd`). The
generic YAML/config helpers live in :mod:`._common` (shared with
train and eval); the train-side and eval-side equivalents of this
module live in :mod:`._train_config` and :mod:`._eval_config`.

What lives here:

- Four typed dataclasses:
  :class:`CalibrationCLIConfig` (loader knobs for the temperature
  fit), :class:`PreprocessingCLIConfig` (constants that land in the
  metadata sidecar's ``preprocessing`` block),
  :class:`DecisionCLIConfig` (threshold + class labels), and
  :class:`ExportOutputCLIConfig` (output dir + base filename),
  bundled into :class:`ExportExperimentConfig`.
- :func:`build_export_config` — YAML doc -> typed config, with
  syntax-level validation (key allowlists, threshold range, scheme
  enum, opset positivity, presence of required preprocessing fields).
- :class:`ExportCLIOverrides` + :func:`apply_export_overrides` for
  argparse flag overrides.

Model architecture is rebuilt from the checkpoint's ``model_meta``
dict (see :func:`._common.build_model_from_meta`), so the export
YAML carries no ``model:`` block; ``preprocessing.expected_trace_samples``
is derived from the checkpoint's ``input_length`` at runtime.

Why all preprocessing + decision fields are YAML-supplied (not
bank-derived): the calibration bank is a *training-data artifact* —
it stores per-trace ``freq_hz`` but doesn't preserve the bandpass
filter applied at training time, and post-aggregation across
multiple source banks it can carry mixed sampling rates. The
deployment-time constants are *training-pipeline configuration*,
not bank attributes; trying to reconstruct them from a bank is
brittle. A cleaner long-term answer is to teach training to stamp
its preprocessing config into run.json so export can read from there
— tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    DEFAULT_CLASS_LABELS,
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_EXPORT_NAME,
    DEFAULT_NORMALIZATION_SCHEME,
    DEFAULT_NUM_WORKERS,
    DEFAULT_ONNX_OPSET,
    DEFAULT_PIN_MEMORY,
    DEFAULT_ZNORM_EPS,
)

# Valid values for the metadata sidecar's normalization scheme. Mirrors the
# egm-contracts egm_class_model_metadata schema enum exactly.
_NORMALIZATION_SCHEMES = {"zscore", "zero2one", "none"}


# ---------------------------------------------------------------------------
# Typed dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationCLIConfig:
    """Loader knobs for the temperature-scaling fit.

    ``bank`` is the labeled ClassifierBank that the calibration loader
    iterates over to collect (logits, labels) for fitting T. When
    ``None``, the export skips fitting and ships with T = 1.0 (raw
    logits unchanged). The other fields are DataLoader knobs; they
    matter only when ``bank`` is set.
    """

    bank: Path | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY


@dataclass(frozen=True)
class PreprocessingCLIConfig:
    """Constants that land in the metadata sidecar's ``preprocessing`` block.

    All fields are deployment-policy values supplied directly by the
    YAML — they describe what the runtime must do to incoming traces
    before invoking the model, and they must match what the model was
    trained with. ``fs_hz`` and ``bandpass_hz`` are required (no
    sensible global default — they depend on the trained model);
    ``normalization_scheme`` defaults to ``zscore`` (matches the v1
    training default) and ``normalization_eps`` floors the per-trace
    divisor (std for zscore; max-min for zero2one).
    """

    fs_hz: float
    bandpass_hz: tuple[float, float]
    normalization_scheme: str = DEFAULT_NORMALIZATION_SCHEME
    normalization_eps: float = DEFAULT_ZNORM_EPS


@dataclass(frozen=True)
class DecisionCLIConfig:
    """Decision-rule constants for the metadata sidecar's ``decision`` block.

    Both fields are deployment-policy values. ``threshold`` defaults
    to ``DEFAULT_DECISION_THRESHOLD`` (matches the eval CLI);
    ``class_labels`` defaults to ``("healthy", "fibrotic")`` (matches
    the v1 binary-head convention pinned by ``HEALTHY_LABEL=0`` /
    ``FIBROTIC_LABEL=1``). Override in YAML for a non-v1 model.
    """

    threshold: float = DEFAULT_DECISION_THRESHOLD
    class_labels: tuple[str, ...] = field(default_factory=lambda: tuple(DEFAULT_CLASS_LABELS))


@dataclass(frozen=True)
class ExportOutputCLIConfig:
    """Where the exported artifact pair lands.

    The export CLI produces two files under :attr:`dir`:
    ``<name>.onnx`` and ``<name>.model_metadata.json``. Default name
    matches the schema docs convention.
    """

    dir: Path = Path("exports")
    name: str = DEFAULT_EXPORT_NAME


@dataclass(frozen=True)
class ExportExperimentConfig:
    """Bundled typed config for ``egm-class-export``.

    Model architecture is rebuilt from the checkpoint's ``model_meta``
    dict, so this config carries no ``model:`` block.
    ``preprocessing.expected_trace_samples`` is derived from the
    checkpoint's ``input_length`` at runtime, not from this config.
    """

    checkpoint: Path
    calibration: CalibrationCLIConfig
    preprocessing: PreprocessingCLIConfig
    decision: DecisionCLIConfig
    output: ExportOutputCLIConfig
    opset: int = DEFAULT_ONNX_OPSET


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------

_EXPORT_TOP_KEYS = {
    "checkpoint",
    "calibration",
    "preprocessing",
    "decision",
    "output",
    "opset",
}
_CALIBRATION_KEYS = {"bank", "batch_size", "num_workers", "pin_memory"}
_PREPROCESSING_KEYS = {
    "fs_hz",
    "bandpass_hz",
    "normalization_scheme",
    "normalization_eps",
}
_DECISION_KEYS = {"threshold", "class_labels"}
_OUTPUT_KEYS = {"dir", "name"}


def build_export_config(doc: dict[str, Any]) -> ExportExperimentConfig:
    """Translate a parsed YAML dict into a typed export config.

    Validates each block's keys, rejects unknowns, resolves paths
    against the YAML file's directory, and enforces field ranges
    (threshold in (0, 1), scheme in the enum, opset positive,
    bandpass low < high, fs_hz positive).
    """
    _reject_unknown_keys(
        {k: v for k, v in doc.items() if k != "_config_dir"},
        _EXPORT_TOP_KEYS,
        label="top-level",
    )
    cfg_dir: Path = doc["_config_dir"]

    checkpoint_raw = doc.get("checkpoint")
    if checkpoint_raw is None:
        raise ConfigError("checkpoint is required (path to a training checkpoint).")
    checkpoint = _resolve_path(str(checkpoint_raw), cfg_dir)
    if checkpoint is None:
        raise ConfigError("checkpoint path resolved to None; check the YAML value.")

    calibration = _build_calibration_block(_optional(doc, "calibration", default={}) or {}, cfg_dir)
    preprocessing = _build_preprocessing_block(_optional(doc, "preprocessing", default={}) or {})
    decision = _build_decision_block(_optional(doc, "decision", default={}) or {})
    output = _build_output_block(_optional(doc, "output", default={}) or {}, cfg_dir)

    opset = int(doc.get("opset", DEFAULT_ONNX_OPSET))
    if opset < 1:
        raise ConfigError(f"opset must be a positive integer; got {opset}.")

    return ExportExperimentConfig(
        checkpoint=checkpoint,
        calibration=calibration,
        preprocessing=preprocessing,
        decision=decision,
        output=output,
        opset=opset,
    )


def _build_calibration_block(block: dict[str, Any], config_dir: Path) -> CalibrationCLIConfig:
    _reject_unknown_keys(block, _CALIBRATION_KEYS, label="calibration")
    bank_raw = block.get("bank")
    bank = _resolve_path(str(bank_raw) if bank_raw is not None else None, config_dir)
    return CalibrationCLIConfig(
        bank=bank,
        batch_size=int(block.get("batch_size", DEFAULT_BATCH_SIZE)),
        num_workers=int(block.get("num_workers", DEFAULT_NUM_WORKERS)),
        pin_memory=bool(block.get("pin_memory", DEFAULT_PIN_MEMORY)),
    )


def _build_preprocessing_block(block: dict[str, Any]) -> PreprocessingCLIConfig:
    _reject_unknown_keys(block, _PREPROCESSING_KEYS, label="preprocessing")

    fs_hz_raw = block.get("fs_hz")
    if fs_hz_raw is None:
        raise ConfigError(
            "preprocessing.fs_hz is required (training-time sampling rate "
            "the deployed model expects)."
        )
    fs_hz = float(fs_hz_raw)
    if fs_hz <= 0:
        raise ConfigError(f"preprocessing.fs_hz must be > 0; got {fs_hz}.")

    bandpass_raw = block.get("bandpass_hz")
    if bandpass_raw is None:
        raise ConfigError(
            "preprocessing.bandpass_hz is required (the [low, high] band-pass "
            "edges, in Hz, applied at training time)."
        )
    if not (isinstance(bandpass_raw, list | tuple) and len(bandpass_raw) == 2):
        raise ConfigError(
            f"preprocessing.bandpass_hz must be a two-element list [low, high]; "
            f"got {bandpass_raw!r}."
        )
    low, high = float(bandpass_raw[0]), float(bandpass_raw[1])
    if not (0 < low < high):
        raise ConfigError(
            f"preprocessing.bandpass_hz must satisfy 0 < low < high; got [{low}, {high}]."
        )
    bandpass_hz = (low, high)

    scheme = str(block.get("normalization_scheme", DEFAULT_NORMALIZATION_SCHEME))
    if scheme not in _NORMALIZATION_SCHEMES:
        raise ConfigError(
            f"preprocessing.normalization_scheme must be one of "
            f"{sorted(_NORMALIZATION_SCHEMES)}; got {scheme!r}."
        )

    eps = float(block.get("normalization_eps", DEFAULT_ZNORM_EPS))
    if eps <= 0:
        raise ConfigError(f"preprocessing.normalization_eps must be > 0; got {eps}.")

    return PreprocessingCLIConfig(
        fs_hz=fs_hz,
        bandpass_hz=bandpass_hz,
        normalization_scheme=scheme,
        normalization_eps=eps,
    )


def _build_decision_block(block: dict[str, Any]) -> DecisionCLIConfig:
    _reject_unknown_keys(block, _DECISION_KEYS, label="decision")
    threshold = float(block.get("threshold", DEFAULT_DECISION_THRESHOLD))
    if not (0.0 < threshold < 1.0):
        raise ConfigError(f"decision.threshold must satisfy 0 < threshold < 1; got {threshold}.")

    labels_raw = block.get("class_labels")
    labels: tuple[str, ...] = tuple(DEFAULT_CLASS_LABELS)
    if labels_raw is not None:
        if not (isinstance(labels_raw, list | tuple) and len(labels_raw) >= 2):
            raise ConfigError(
                f"decision.class_labels must be a list of at least two strings; got {labels_raw!r}."
            )
        labels = tuple(str(x) for x in labels_raw)

    return DecisionCLIConfig(threshold=threshold, class_labels=labels)


def _build_output_block(block: dict[str, Any], config_dir: Path) -> ExportOutputCLIConfig:
    _reject_unknown_keys(block, _OUTPUT_KEYS, label="output")
    dir_raw = block.get("dir", "exports")
    out_dir = _resolve_path(str(dir_raw), config_dir)
    assert out_dir is not None  # non-empty literal
    return ExportOutputCLIConfig(
        dir=out_dir,
        name=str(block.get("name", DEFAULT_EXPORT_NAME)),
    )


# ---------------------------------------------------------------------------
# CLI flag overrides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportCLIOverrides:
    """Per-flag overrides applied on top of the YAML-loaded export config.

    Each field is ``None`` when the user didn't pass the flag, so the
    YAML value is kept. Path fields are pre-resolved against CWD by
    the argparse layer.
    """

    checkpoint: Path | None = None
    calibration_bank: Path | None = None
    output_dir: Path | None = None
    output_name: str | None = None
    threshold: float | None = None
    opset: int | None = None
    skip_calibration: bool = False


def apply_export_overrides(
    cfg: ExportExperimentConfig, overrides: ExportCLIOverrides
) -> ExportExperimentConfig:
    """Return a new :class:`ExportExperimentConfig` with overrides applied."""
    checkpoint = cfg.checkpoint
    calibration = cfg.calibration
    decision = cfg.decision
    output = cfg.output
    opset = cfg.opset

    if overrides.checkpoint is not None:
        checkpoint = overrides.checkpoint.resolve()
    if overrides.calibration_bank is not None:
        calibration = replace(calibration, bank=overrides.calibration_bank.resolve())
    if overrides.skip_calibration:
        # Belt-and-suspenders with --calibration-bank: skip wins.
        calibration = replace(calibration, bank=None)
    if overrides.output_dir is not None:
        output = replace(output, dir=overrides.output_dir.resolve())
    if overrides.output_name is not None:
        output = replace(output, name=overrides.output_name)
    if overrides.threshold is not None:
        if not (0.0 < overrides.threshold < 1.0):
            raise ConfigError(
                f"--threshold must satisfy 0 < threshold < 1; got {overrides.threshold}."
            )
        decision = replace(decision, threshold=overrides.threshold)
    if overrides.opset is not None:
        if overrides.opset < 1:
            raise ConfigError(f"--opset must be a positive integer; got {overrides.opset}.")
        opset = overrides.opset

    return ExportExperimentConfig(
        checkpoint=checkpoint,
        calibration=calibration,
        preprocessing=cfg.preprocessing,
        decision=decision,
        output=output,
        opset=opset,
    )
