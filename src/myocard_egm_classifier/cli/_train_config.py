"""Train-CLI typed config + YAML builders.

Consumed only by ``egm-class-train`` (via :mod:`.train_cmd`). The
generic YAML/config helpers (``ConfigError``, ``load_yaml``,
``_optional``, ``_resolve_path``, ``_reject_unknown_keys``) live in
:mod:`._common` because they're also used by :mod:`._eval_config`
(and eventually ``_export_config``).

The model triad — :class:`ModelCLIConfig`,
:func:`build_model_from_config`, :func:`model_meta_from_config` —
lives here rather than in ``_common`` because it's a train-time
concept: the YAML ``model:`` block deserializes into
``ModelCLIConfig``, which the trainer uses to build the architecture
and then serializes back out as the ``model_meta`` dict embedded in
the checkpoint. Eval and export consume that meta dict directly via
:func:`._common.build_model_from_meta` and never need to round-trip
through ``ModelCLIConfig``.

What lives in this module:

- :class:`ModelCLIConfig` + :func:`build_model_from_config` +
  :func:`model_meta_from_config` (train-side model triad).
- The three train-side dataclasses :class:`DataCLIConfig`,
  :class:`TrainCLIConfig`, :class:`OutputCLIConfig`, plus the bundled
  :class:`TrainExperimentConfig`.
- The per-block YAML builders and key-allowlists.
- :class:`TrainCLIOverrides` + :func:`apply_train_overrides` for
  argparse flags.
- :func:`to_train_runtime_config` (config -> ``training.TrainConfig``).
- :func:`loader_kwargs_from_config` (config -> egm-data
  ``build_dataloaders`` kwargs, including the resolved patient
  stratification strategy).
- :func:`experiment_config_to_dict` (config -> JSON-serializable dict
  for the ``run.json`` artifact's ``config`` block).
- Plumbing for the patient-stratification ``split_strategy`` block:
  :class:`SplitStrategyConfig` dataclass + :func:`to_strategy` builder.
- :func:`_expect_fractions` (split-fraction validator; train-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from myocard_egm_data.splits import (
    AnyPositiveStrategy,
    BinnedDensityStrategy,
    PatientStratificationStrategy,
)

from myocard_egm_classifier.cli._common import (
    ConfigError,
    _optional,
    _reject_unknown_keys,
    _resolve_path,
)
from myocard_egm_classifier.constants import (
    DEFAULT_AMP,
    DEFAULT_AMP_DTYPE,
    DEFAULT_AUGMENT_TRAIN,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_HEAD_EXPANSION_CHANNELS,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_LENGTH,
    DEFAULT_LR,
    DEFAULT_MAX_GAIN,
    DEFAULT_MAX_SHIFT_FRAC,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_SEED,
    DEFAULT_SELECT_METRIC,
    DEFAULT_SPLIT_FRACTIONS,
    DEFAULT_SPLIT_SEED,
    DEFAULT_STOCHASTIC_DEPTH,
    DEFAULT_WARMUP_FRAC,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_WIDTH_MULTIPLIER,
    DEFAULT_ZNORM,
    DEFAULT_ZNORM_EPS,
)
from myocard_egm_classifier.models import BlockSpec, MobileViT1D, default_v1_blocks
from myocard_egm_classifier.training import TrainConfig

# ---------------------------------------------------------------------------
# Generic helper that's only used here (kept out of _common since it
# has no eval-side caller)
# ---------------------------------------------------------------------------


def _expect_fractions(value: Any, *, field_path: str) -> tuple[float, float, float]:
    """Validate that ``value`` is a 3-element list/tuple that sums to 1."""
    if not (isinstance(value, list | tuple) and len(value) == 3):
        raise ConfigError(
            f"{field_path} must be a three-element list [train, val, test]; got {value!r}."
        )
    fractions = (float(value[0]), float(value[1]), float(value[2]))
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ConfigError(f"{field_path} must sum to 1; got {fractions} (sum={sum(fractions)}).")
    return fractions


# ---------------------------------------------------------------------------
# Model-side typed config (train-only; eval/export use the checkpoint's
# model_meta dict directly via _common.build_model_from_meta)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCLIConfig:
    """Model-architecture knobs for one training run.

    Deserialized from the YAML ``model:`` block by
    :func:`_build_model_block`, materialized into a live
    :class:`MobileViT1D` by :func:`build_model_from_config`, and
    serialized as the embedded ``model_meta`` dict in the checkpoint by
    :func:`model_meta_from_config`. Eval and export rebuild the model
    directly from that meta dict (via
    :func:`._common.build_model_from_meta`) and have no need for this
    dataclass — so it lives in the train module, not in ``_common``.
    """

    width_multiplier: float = DEFAULT_WIDTH_MULTIPLIER
    num_classes: int = DEFAULT_NUM_CLASSES
    in_channels: int = DEFAULT_INPUT_CHANNELS
    input_length: int = DEFAULT_INPUT_LENGTH
    stochastic_depth: float = DEFAULT_STOCHASTIC_DEPTH
    head_expansion_channels: int = DEFAULT_HEAD_EXPANSION_CHANNELS
    head_dropout: float = DEFAULT_HEAD_DROPOUT
    arch: str = "mobilevit_1d_v1"
    blocks: tuple[BlockSpec, ...] | None = field(default=None)
    """Explicit block list overrides the v1 default template (Sec. 9)."""

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 1


def build_model_from_config(model_cfg: ModelCLIConfig) -> MobileViT1D:
    """Construct the model described by a :class:`ModelCLIConfig`."""
    if model_cfg.blocks is not None:
        blocks = list(model_cfg.blocks)
    else:
        blocks = default_v1_blocks(
            num_outputs=model_cfg.num_classes,
            head_expansion_channels=model_cfg.head_expansion_channels,
            head_dropout=model_cfg.head_dropout,
        )
    return MobileViT1D(
        blocks=blocks,
        width_multiplier=model_cfg.width_multiplier,
        in_channels=model_cfg.in_channels,
        stochastic_depth=model_cfg.stochastic_depth,
    )


def model_meta_from_config(model_cfg: ModelCLIConfig) -> dict[str, Any]:
    """Architecture args to embed in a checkpoint for rebuild on eval/export.

    Inverse of :func:`._common.build_model_from_meta`. Trainer calls
    this once at checkpoint-save time; eval/export then call
    ``build_model_from_meta`` against the embedded dict without ever
    needing to materialize a :class:`ModelCLIConfig`.
    """
    return {
        "width_multiplier": model_cfg.width_multiplier,
        "num_classes": model_cfg.num_classes,
        "in_channels": model_cfg.in_channels,
        "input_length": model_cfg.input_length,
        "stochastic_depth": model_cfg.stochastic_depth,
        "head_expansion_channels": model_cfg.head_expansion_channels,
        "head_dropout": model_cfg.head_dropout,
        "arch": model_cfg.arch,
    }


# ---------------------------------------------------------------------------
# split_strategy block
# ---------------------------------------------------------------------------


DEFAULT_SPLIT_STRATEGY_TYPE = "any_positive"
"""Patient stratification strategy used when the YAML omits the block.

Matches egm-data's default (``AnyPositiveStrategy``): one bit per
patient, 1 if any trace is positive, else 0. Behaves identically to
the original per-patient-label stratifier on global-density banks,
and works on local-density banks without raising. Switch to
``binned_density`` when the AUROC-warning fires on a heavily-skewed
local-density bank."""

DEFAULT_BINNED_DENSITY_N_BINS = 3
"""Bin count for ``binned_density`` when the YAML omits ``n_bins``."""


@dataclass(frozen=True)
class SplitStrategyConfig:
    """How to bucket patients for the patient-aware split's stratifier.

    The YAML mirrors synthetic-egm-pipeline's ``label_policy`` pattern:
    a ``type`` discriminator plus per-strategy params. Two strategies
    ship out of the box; new ones land as one file in
    ``myocard_egm_data.splits.strategies`` and one entry in
    :func:`_build_split_strategy_block` here.

    Attributes
    ----------
    type
        ``"any_positive"`` (default) or ``"binned_density"``.
    n_bins
        Number of equal-width positive-rate bins for
        ``binned_density``. Ignored when ``type == "any_positive"``.
    """

    type: str = DEFAULT_SPLIT_STRATEGY_TYPE
    n_bins: int = DEFAULT_BINNED_DENSITY_N_BINS


# ---------------------------------------------------------------------------
# Train-side typed dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataCLIConfig:
    """Data-pipeline knobs for one training run.

    ``bank`` points at a ClassifierBank HDF5 written by either the
    synthetic producer (``synthetic-egm-pipeline``) or the IAFDB
    producer (``iafdb-pipeline``). Label policy was applied at
    conversion time; we don't re-threshold on the consumer side.
    """

    bank: Path | None = None
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = DEFAULT_NUM_WORKERS
    znorm: bool = DEFAULT_ZNORM
    znorm_eps: float = DEFAULT_ZNORM_EPS
    augment_train: bool = DEFAULT_AUGMENT_TRAIN
    max_gain: float = DEFAULT_MAX_GAIN
    max_shift_frac: float = DEFAULT_MAX_SHIFT_FRAC
    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS
    split_seed: int = DEFAULT_SPLIT_SEED
    split_strategy: SplitStrategyConfig = field(default_factory=SplitStrategyConfig)
    pin_memory: bool = DEFAULT_PIN_MEMORY


@dataclass(frozen=True)
class TrainCLIConfig:
    """Training-loop knobs for one run."""

    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_frac: float = DEFAULT_WARMUP_FRAC
    grad_clip_norm: float | None = DEFAULT_GRAD_CLIP_NORM
    amp: bool = DEFAULT_AMP
    amp_dtype: str = DEFAULT_AMP_DTYPE
    seed: int = DEFAULT_SEED
    select_metric: str = DEFAULT_SELECT_METRIC


@dataclass(frozen=True)
class OutputCLIConfig:
    """Where the run's outputs land."""

    checkpoint_dir: Path = Path("checkpoints")
    description: str = ""


@dataclass(frozen=True)
class TrainExperimentConfig:
    """Bundled typed config for ``egm-class-train``."""

    model: ModelCLIConfig
    data: DataCLIConfig
    train: TrainCLIConfig
    output: OutputCLIConfig


# ---------------------------------------------------------------------------
# YAML builder
# ---------------------------------------------------------------------------


_MODEL_KEYS = {
    "width_multiplier",
    "num_classes",
    "in_channels",
    "input_length",
    "stochastic_depth",
    "head_expansion_channels",
    "head_dropout",
    "arch",
    "blocks",
}
_DATA_KEYS = {
    "bank",
    "batch_size",
    "num_workers",
    "znorm",
    "znorm_eps",
    "augment_train",
    "max_gain",
    "max_shift_frac",
    "split_fractions",
    "split_seed",
    "split_strategy",
    "pin_memory",
}
_TRAIN_KEYS = {
    "epochs",
    "lr",
    "weight_decay",
    "warmup_frac",
    "grad_clip_norm",
    "amp",
    "amp_dtype",
    "seed",
    "select_metric",
}
_OUTPUT_KEYS = {"checkpoint_dir", "description"}
_TOP_KEYS = {"model", "data", "train", "output"}
_SPLIT_STRATEGY_TYPES = {"any_positive", "binned_density"}
_SPLIT_STRATEGY_KEYS = {"type", "n_bins"}


def build_train_config(doc: dict[str, Any]) -> TrainExperimentConfig:
    """Translate a parsed YAML dict into a typed train config.

    Validates each block's keys, rejects unknown keys with a precise
    error, resolves paths against the YAML file's directory, and
    cross-checks invariants (``split_fractions`` sums to 1,
    ``num_classes >= 1``, ``bank`` present).
    """
    _reject_unknown_keys(
        {k: v for k, v in doc.items() if k != "_config_dir"},
        _TOP_KEYS,
        label="top-level",
    )
    cfg_dir: Path = doc["_config_dir"]

    model = _build_model_block(_optional(doc, "model", default={}) or {})
    data = _build_data_block(_optional(doc, "data", default={}) or {}, cfg_dir)
    train_block = _build_train_block(_optional(doc, "train", default={}) or {})
    output = _build_output_block(_optional(doc, "output", default={}) or {}, cfg_dir)

    if data.bank is None:
        raise ConfigError("data.bank is required (path to a ClassifierBank HDF5 file).")
    if model.num_classes < 1:
        raise ConfigError("model.num_classes must be >= 1.")

    return TrainExperimentConfig(model=model, data=data, train=train_block, output=output)


def _build_model_block(block: dict[str, Any]) -> ModelCLIConfig:
    _reject_unknown_keys(block, _MODEL_KEYS, label="model")
    blocks_raw = block.get("blocks")
    blocks: tuple[BlockSpec, ...] | None = None
    if blocks_raw is not None:
        if not isinstance(blocks_raw, list):
            raise ConfigError("model.blocks must be a list of {type, params} mappings.")
        blocks = tuple(
            BlockSpec(type=str(b["type"]), params=dict(b.get("params", {}))) for b in blocks_raw
        )
    return ModelCLIConfig(
        width_multiplier=float(block.get("width_multiplier", DEFAULT_WIDTH_MULTIPLIER)),
        num_classes=int(block.get("num_classes", DEFAULT_NUM_CLASSES)),
        in_channels=int(block.get("in_channels", DEFAULT_INPUT_CHANNELS)),
        input_length=int(block.get("input_length", DEFAULT_INPUT_LENGTH)),
        stochastic_depth=float(block.get("stochastic_depth", DEFAULT_STOCHASTIC_DEPTH)),
        head_expansion_channels=int(
            block.get("head_expansion_channels", DEFAULT_HEAD_EXPANSION_CHANNELS)
        ),
        head_dropout=float(block.get("head_dropout", DEFAULT_HEAD_DROPOUT)),
        arch=str(block.get("arch", "mobilevit_1d_v1")),
        blocks=blocks,
    )


def _build_data_block(block: dict[str, Any], config_dir: Path) -> DataCLIConfig:
    _reject_unknown_keys(block, _DATA_KEYS, label="data")
    bank_raw = block.get("bank")
    bank = _resolve_path(str(bank_raw) if bank_raw is not None else None, config_dir)
    fractions = _expect_fractions(
        block.get("split_fractions", list(DEFAULT_SPLIT_FRACTIONS)),
        field_path="data.split_fractions",
    )
    strategy_block = block.get("split_strategy")
    if strategy_block is None:
        split_strategy = SplitStrategyConfig()
    else:
        if not isinstance(strategy_block, dict):
            raise ConfigError(
                "data.split_strategy must be a mapping (e.g. "
                "{type: any_positive}); got " + repr(strategy_block)
            )
        split_strategy = _build_split_strategy_block(strategy_block)
    return DataCLIConfig(
        bank=bank,
        batch_size=int(block.get("batch_size", DEFAULT_BATCH_SIZE)),
        num_workers=int(block.get("num_workers", DEFAULT_NUM_WORKERS)),
        znorm=bool(block.get("znorm", DEFAULT_ZNORM)),
        znorm_eps=float(block.get("znorm_eps", DEFAULT_ZNORM_EPS)),
        augment_train=bool(block.get("augment_train", DEFAULT_AUGMENT_TRAIN)),
        max_gain=float(block.get("max_gain", DEFAULT_MAX_GAIN)),
        max_shift_frac=float(block.get("max_shift_frac", DEFAULT_MAX_SHIFT_FRAC)),
        split_fractions=fractions,
        split_seed=int(block.get("split_seed", DEFAULT_SPLIT_SEED)),
        split_strategy=split_strategy,
        pin_memory=bool(block.get("pin_memory", DEFAULT_PIN_MEMORY)),
    )


def _build_train_block(block: dict[str, Any]) -> TrainCLIConfig:
    _reject_unknown_keys(block, _TRAIN_KEYS, label="train")
    raw_clip = block.get("grad_clip_norm", DEFAULT_GRAD_CLIP_NORM)
    grad_clip_norm: float | None = None if raw_clip is None else float(raw_clip)
    amp_dtype = str(block.get("amp_dtype", DEFAULT_AMP_DTYPE))
    if amp_dtype not in {"bf16", "fp16"}:
        raise ConfigError(f"train.amp_dtype must be 'bf16' or 'fp16'; got {amp_dtype!r}.")
    return TrainCLIConfig(
        epochs=int(block.get("epochs", DEFAULT_EPOCHS)),
        lr=float(block.get("lr", DEFAULT_LR)),
        weight_decay=float(block.get("weight_decay", DEFAULT_WEIGHT_DECAY)),
        warmup_frac=float(block.get("warmup_frac", DEFAULT_WARMUP_FRAC)),
        grad_clip_norm=grad_clip_norm,
        amp=bool(block.get("amp", DEFAULT_AMP)),
        amp_dtype=amp_dtype,
        seed=int(block.get("seed", DEFAULT_SEED)),
        select_metric=str(block.get("select_metric", DEFAULT_SELECT_METRIC)),
    )


def _build_output_block(block: dict[str, Any], config_dir: Path) -> OutputCLIConfig:
    _reject_unknown_keys(block, _OUTPUT_KEYS, label="output")
    ckpt_raw = block.get("checkpoint_dir", "checkpoints")
    checkpoint_dir = _resolve_path(str(ckpt_raw), config_dir)
    assert checkpoint_dir is not None  # we passed a non-empty literal
    return OutputCLIConfig(
        checkpoint_dir=checkpoint_dir,
        description=str(block.get("description", "")),
    )


def _build_split_strategy_block(block: dict[str, Any]) -> SplitStrategyConfig:
    """Translate a ``data.split_strategy:`` YAML block into the typed config.

    The discriminator key is ``type``; the strategy-specific parameter
    set lives flat under the block (currently just ``n_bins`` for
    ``binned_density``). Validates the type against the known set and
    rejects unknown keys so config typos fail loudly rather than
    silently using the default.
    """
    _reject_unknown_keys(block, _SPLIT_STRATEGY_KEYS, label="data.split_strategy")
    strategy_type = str(block.get("type", DEFAULT_SPLIT_STRATEGY_TYPE))
    if strategy_type not in _SPLIT_STRATEGY_TYPES:
        raise ConfigError(
            f"data.split_strategy.type must be one of "
            f"{sorted(_SPLIT_STRATEGY_TYPES)}; got {strategy_type!r}."
        )
    n_bins = int(block.get("n_bins", DEFAULT_BINNED_DENSITY_N_BINS))
    if strategy_type == "binned_density" and n_bins < 2:
        raise ConfigError(
            f"data.split_strategy.n_bins must be >= 2 for binned_density; got {n_bins}."
        )
    return SplitStrategyConfig(type=strategy_type, n_bins=n_bins)


def to_strategy(cfg: SplitStrategyConfig) -> PatientStratificationStrategy:
    """Instantiate the egm-data strategy class from the YAML config.

    Single registry point for type -> class dispatch: adding a new
    strategy means one new ``elif`` branch here, one new entry in
    :data:`_SPLIT_STRATEGY_TYPES`, and one new module in egm-data.
    """
    if cfg.type == "any_positive":
        return AnyPositiveStrategy()
    if cfg.type == "binned_density":
        return BinnedDensityStrategy(n_bins=cfg.n_bins)
    raise ConfigError(f"Unknown split_strategy.type {cfg.type!r}; this is a bug.")


# ---------------------------------------------------------------------------
# CLI flag overrides
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainCLIOverrides:
    """Per-flag overrides applied on top of the YAML-loaded config.

    Each field is ``None`` when the user didn't pass the flag, so the
    YAML value is kept. Path fields are pre-resolved against CWD by the
    argparse layer.
    """

    bank: Path | None = None
    epochs: int | None = None
    width: float | None = None
    batch_size: int | None = None
    checkpoint_dir: Path | None = None
    seed: int | None = None
    no_augment: bool = False


def apply_train_overrides(
    cfg: TrainExperimentConfig, overrides: TrainCLIOverrides
) -> TrainExperimentConfig:
    """Return a new :class:`TrainExperimentConfig` with overrides applied."""
    model = cfg.model
    data = cfg.data
    train_block = cfg.train
    output = cfg.output

    if overrides.width is not None:
        model = replace(model, width_multiplier=overrides.width)
    if overrides.bank is not None:
        data = replace(data, bank=overrides.bank.resolve())
    if overrides.batch_size is not None:
        data = replace(data, batch_size=overrides.batch_size)
    if overrides.no_augment:
        data = replace(data, augment_train=False)
    if overrides.epochs is not None:
        train_block = replace(train_block, epochs=overrides.epochs)
    if overrides.seed is not None:
        train_block = replace(train_block, seed=overrides.seed)
    if overrides.checkpoint_dir is not None:
        output = replace(output, checkpoint_dir=overrides.checkpoint_dir.resolve())

    return TrainExperimentConfig(model=model, data=data, train=train_block, output=output)


# ---------------------------------------------------------------------------
# Runtime translators: config -> downstream API kwargs
# ---------------------------------------------------------------------------


def to_train_runtime_config(cfg: TrainExperimentConfig, pos_weight: float | None) -> TrainConfig:
    """Turn the TrainCLIConfig (+ data-derived pos_weight) into a runtime TrainConfig."""
    t = cfg.train
    return TrainConfig(
        epochs=t.epochs,
        lr=t.lr,
        weight_decay=t.weight_decay,
        warmup_frac=t.warmup_frac,
        grad_clip_norm=t.grad_clip_norm,
        binary=cfg.model.is_binary,
        pos_weight=pos_weight if cfg.model.is_binary else None,
        amp=t.amp,
        amp_dtype=t.amp_dtype,
        select_metric=t.select_metric,
    )


def loader_kwargs_from_config(cfg: TrainExperimentConfig) -> dict[str, Any]:
    """Keyword args for ``myocard_egm_data.datasets.build_dataloaders``.

    Mirrors the build_dataloaders signature exactly; this function exists
    so train_cmd doesn't have to know which TrainExperimentConfig fields
    map to which build_dataloaders kwarg.
    """
    return {
        "input_length": cfg.model.input_length,
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "binary": cfg.model.is_binary,
        "znorm": cfg.data.znorm,
        "znorm_eps": cfg.data.znorm_eps,
        "augment_train": cfg.data.augment_train,
        "max_gain": cfg.data.max_gain,
        "max_shift_frac": cfg.data.max_shift_frac,
        "split_fractions": tuple(cfg.data.split_fractions),
        "split_seed": cfg.data.split_seed,
        "dataset_seed": cfg.train.seed,
        "pin_memory": cfg.data.pin_memory,
        "split_strategy": to_strategy(cfg.data.split_strategy),
    }


# ---------------------------------------------------------------------------
# Serialization for the run.json config field
# ---------------------------------------------------------------------------


def experiment_config_to_dict(cfg: TrainExperimentConfig) -> dict[str, Any]:
    """JSON-serializable dict for the ``TrainingRunRecord.config`` field.

    Paths become strings, tuples become lists, frozen dataclasses
    become plain dicts. The result is what we hand to
    :func:`myocard_egm_data.records.build_training_run_record` as the
    ``config`` block.
    """
    return {
        "model": _model_to_dict(cfg.model),
        "data": _data_to_dict(cfg.data),
        "train": _train_to_dict(cfg.train),
        "output": _output_to_dict(cfg.output),
    }


def _model_to_dict(m: ModelCLIConfig) -> dict[str, Any]:
    blocks: list[dict[str, Any]] | None = None
    if m.blocks is not None:
        blocks = [{"type": b.type, "params": dict(b.params)} for b in m.blocks]
    return {
        "width_multiplier": m.width_multiplier,
        "num_classes": m.num_classes,
        "in_channels": m.in_channels,
        "input_length": m.input_length,
        "stochastic_depth": m.stochastic_depth,
        "head_expansion_channels": m.head_expansion_channels,
        "head_dropout": m.head_dropout,
        "arch": m.arch,
        "blocks": blocks,
    }


def _data_to_dict(d: DataCLIConfig) -> dict[str, Any]:
    return {
        "bank": str(d.bank) if d.bank is not None else None,
        "batch_size": d.batch_size,
        "num_workers": d.num_workers,
        "znorm": d.znorm,
        "znorm_eps": d.znorm_eps,
        "augment_train": d.augment_train,
        "max_gain": d.max_gain,
        "max_shift_frac": d.max_shift_frac,
        "split_fractions": list(d.split_fractions),
        "split_seed": d.split_seed,
        "split_strategy": {
            "type": d.split_strategy.type,
            "n_bins": d.split_strategy.n_bins,
        },
        "pin_memory": d.pin_memory,
    }


def _train_to_dict(t: TrainCLIConfig) -> dict[str, Any]:
    return {
        "epochs": t.epochs,
        "lr": t.lr,
        "weight_decay": t.weight_decay,
        "warmup_frac": t.warmup_frac,
        "grad_clip_norm": t.grad_clip_norm,
        "amp": t.amp,
        "amp_dtype": t.amp_dtype,
        "seed": t.seed,
        "select_metric": t.select_metric,
    }


def _output_to_dict(o: OutputCLIConfig) -> dict[str, Any]:
    return {"checkpoint_dir": str(o.checkpoint_dir), "description": o.description}
