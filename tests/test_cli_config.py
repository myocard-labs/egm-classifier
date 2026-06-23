"""YAML config loading + override behavior for ``egm-class-train``.

Covers the four things a config bug would break:

- Defaults: an empty YAML (with only ``data.bank``) loads with every
  field set to its constant from :mod:`myocard_egm_classifier.constants`.
- Overrides: a YAML can override any of those defaults, and the typed
  dataclasses pick them up verbatim.
- Path resolution: every path field resolves against the YAML file's
  directory, not the current working directory.
- Error surfaces: missing ``data.bank``, unknown keys, bad
  ``split_fractions`` sum, and bad ``amp_dtype`` all raise
  :class:`ConfigError` with a precise message.

The CLI flag-override pass is tested separately via
:func:`apply_train_overrides`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from myocard_egm_data.splits import AnyPositiveStrategy, BinnedDensityStrategy

from myocard_egm_classifier.cli._common import ConfigError, load_yaml
from myocard_egm_classifier.cli._train_config import (
    TrainCLIOverrides,
    apply_train_overrides,
    build_train_config,
    experiment_config_to_dict,
    loader_kwargs_from_config,
    to_strategy,
    to_train_runtime_config,
)
from myocard_egm_classifier.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WIDTH_MULTIPLIER,
    DEFAULT_ZNORM_EPS,
)


def _write_yaml(tmp_path: Path, body: str, name: str = "cfg.yaml") -> Path:
    """Helper: drop a YAML literal into ``tmp_path/<name>`` and return the path."""
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_minimal_config_picks_up_all_defaults(tmp_path: Path) -> None:
    """A config that only sets data.bank lands with every default in place."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./fake.h5\n",
    )
    cfg = build_train_config(load_yaml(cfg_path))
    # Defaults flow through unchanged.
    assert cfg.model.width_multiplier == DEFAULT_WIDTH_MULTIPLIER
    assert cfg.data.batch_size == DEFAULT_BATCH_SIZE
    assert cfg.data.znorm_eps == DEFAULT_ZNORM_EPS
    assert cfg.train.epochs == DEFAULT_EPOCHS
    assert cfg.train.lr == DEFAULT_LR


def test_yaml_overrides_each_block(tmp_path: Path) -> None:
    """Per-block overrides from YAML reach the typed dataclasses verbatim."""
    cfg_path = _write_yaml(
        tmp_path,
        """\
model:
  width_multiplier: 0.5
  num_classes: 1
data:
  bank: ./fake.h5
  batch_size: 8
  znorm_eps: 1.0e-5
train:
  epochs: 3
  lr: 1.0e-4
  amp_dtype: fp16
output:
  checkpoint_dir: ./out
  description: "test run"
""",
    )
    cfg = build_train_config(load_yaml(cfg_path))
    assert cfg.model.width_multiplier == 0.5
    assert cfg.data.batch_size == 8
    assert cfg.data.znorm_eps == pytest.approx(1.0e-5)
    assert cfg.train.epochs == 3
    assert cfg.train.amp_dtype == "fp16"
    assert cfg.output.description == "test run"


def test_paths_resolve_against_config_dir(tmp_path: Path) -> None:
    """Relative paths (data.bank, output.checkpoint_dir) resolve to YAML dir."""
    cfg_path = _write_yaml(tmp_path, "data:\n  bank: ./banks/fake.h5\n")
    cfg = build_train_config(load_yaml(cfg_path))
    assert cfg.data.bank is not None
    assert cfg.data.bank == (tmp_path / "banks" / "fake.h5").resolve()


def test_missing_bank_raises(tmp_path: Path) -> None:
    """``data.bank`` is required; absent => ConfigError."""
    cfg_path = _write_yaml(tmp_path, "model:\n  width_multiplier: 1.0\n")
    with pytest.raises(ConfigError, match=r"data\.bank is required"):
        build_train_config(load_yaml(cfg_path))


def test_unknown_keys_caught_at_each_level(tmp_path: Path) -> None:
    """Typo'd keys at top-level / per-block raise rather than being silently dropped."""
    # Top-level typo.
    cfg_path = _write_yaml(tmp_path, "data:\n  bank: ./x.h5\nmoodel:\n  width: 1.0\n")
    with pytest.raises(ConfigError, match=r"top-level"):
        build_train_config(load_yaml(cfg_path))
    # Per-block typo: model.with_multiplier vs width_multiplier.
    cfg_path2 = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\nmodel:\n  with_multiplier: 1.0\n",
        name="cfg2.yaml",
    )
    with pytest.raises(ConfigError, match=r"model"):
        build_train_config(load_yaml(cfg_path2))


def test_split_fractions_must_sum_to_one(tmp_path: Path) -> None:
    """``data.split_fractions`` is validated at load time."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\n  split_fractions: [0.7, 0.2, 0.2]\n",
    )
    with pytest.raises(ConfigError, match=r"must sum to 1"):
        build_train_config(load_yaml(cfg_path))


def test_amp_dtype_validated(tmp_path: Path) -> None:
    """train.amp_dtype must be 'bf16' or 'fp16'."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\ntrain:\n  amp_dtype: fp64\n",
    )
    with pytest.raises(ConfigError, match=r"amp_dtype"):
        build_train_config(load_yaml(cfg_path))


def test_cli_overrides_replace_yaml_values(tmp_path: Path) -> None:
    """apply_train_overrides replaces only the fields whose flag was passed."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\n  batch_size: 32\ntrain:\n  epochs: 60\n",
    )
    cfg = build_train_config(load_yaml(cfg_path))
    overrides = TrainCLIOverrides(epochs=2, width=0.5, batch_size=16, no_augment=True)
    out = apply_train_overrides(cfg, overrides)
    assert out.train.epochs == 2
    assert out.model.width_multiplier == 0.5
    assert out.data.batch_size == 16
    assert out.data.augment_train is False
    # Untouched fields are preserved.
    assert out.train.lr == cfg.train.lr


def test_loader_kwargs_round_trips_into_build_dataloaders_shape(tmp_path: Path) -> None:
    """loader_kwargs_from_config emits exactly the kwargs build_dataloaders expects.

    We can't call build_dataloaders here without a bank; instead we
    check the kwarg dict's keys against the egm-data function signature.
    """
    from inspect import signature

    from myocard_egm_data.datasets import build_dataloaders

    cfg_path = _write_yaml(tmp_path, "data:\n  bank: ./x.h5\n")
    cfg = build_train_config(load_yaml(cfg_path))
    kwargs = loader_kwargs_from_config(cfg)

    sig = signature(build_dataloaders)
    # First positional arg is `bank`; everything else is keyword-only.
    expected = {name for name in sig.parameters if name != "bank"}
    assert set(kwargs.keys()) == expected


def test_to_train_runtime_config_carries_pos_weight(tmp_path: Path) -> None:
    """pos_weight is forwarded for binary tasks, dropped for multi-class."""
    cfg_path = _write_yaml(tmp_path, "data:\n  bank: ./x.h5\n")
    cfg = build_train_config(load_yaml(cfg_path))  # default num_classes=1 => binary
    rt = to_train_runtime_config(cfg, pos_weight=2.5)
    assert rt.binary is True
    assert rt.pos_weight == 2.5

    cfg_path2 = _write_yaml(
        tmp_path, "data:\n  bank: ./x.h5\nmodel:\n  num_classes: 3\n", name="multi.yaml"
    )
    cfg2 = build_train_config(load_yaml(cfg_path2))
    rt2 = to_train_runtime_config(cfg2, pos_weight=2.5)
    assert rt2.binary is False
    assert rt2.pos_weight is None


def test_split_strategy_defaults_to_any_positive(tmp_path: Path) -> None:
    """No split_strategy block in YAML => default AnyPositive config."""
    cfg_path = _write_yaml(tmp_path, "data:\n  bank: ./x.h5\n")
    cfg = build_train_config(load_yaml(cfg_path))
    assert cfg.data.split_strategy.type == "any_positive"
    # to_strategy resolves to the egm-data class.
    assert isinstance(to_strategy(cfg.data.split_strategy), AnyPositiveStrategy)


def test_split_strategy_binned_density_round_trip(tmp_path: Path) -> None:
    """YAML binned_density with custom n_bins reaches the strategy instance."""
    cfg_path = _write_yaml(
        tmp_path,
        """\
data:
  bank: ./x.h5
  split_strategy:
    type: binned_density
    n_bins: 5
""",
    )
    cfg = build_train_config(load_yaml(cfg_path))
    assert cfg.data.split_strategy.type == "binned_density"
    assert cfg.data.split_strategy.n_bins == 5
    strategy = to_strategy(cfg.data.split_strategy)
    assert isinstance(strategy, BinnedDensityStrategy)
    assert strategy.n_bins == 5


def test_split_strategy_unknown_type_raises(tmp_path: Path) -> None:
    """Unknown discriminator string -> ConfigError at load time."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\n  split_strategy:\n    type: median_density\n",
    )
    with pytest.raises(ConfigError, match=r"split_strategy\.type"):
        build_train_config(load_yaml(cfg_path))


def test_split_strategy_unknown_key_raises(tmp_path: Path) -> None:
    """Typo'd key inside split_strategy raises rather than silently using the default."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\n  split_strategy:\n    type: binned_density\n    bins: 3\n",
    )
    with pytest.raises(ConfigError, match=r"split_strategy"):
        build_train_config(load_yaml(cfg_path))


def test_split_strategy_binned_density_rejects_n_bins_below_two(tmp_path: Path) -> None:
    """binned_density needs n_bins >= 2; loader catches it at config time."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\n  split_strategy:\n    type: binned_density\n    n_bins: 1\n",
    )
    with pytest.raises(ConfigError, match=r"n_bins"):
        build_train_config(load_yaml(cfg_path))


def test_split_strategy_non_dict_raises(tmp_path: Path) -> None:
    """A scalar where a mapping is expected fails loudly."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\n  split_strategy: any_positive\n",
    )
    with pytest.raises(ConfigError, match=r"must be a mapping"):
        build_train_config(load_yaml(cfg_path))


def test_loader_kwargs_carries_strategy_instance(tmp_path: Path) -> None:
    """loader_kwargs_from_config emits an instantiated strategy object,
    not the config dataclass, so it can be passed straight to
    build_dataloaders."""
    cfg_path = _write_yaml(
        tmp_path,
        "data:\n  bank: ./x.h5\n  split_strategy:\n    type: binned_density\n    n_bins: 4\n",
    )
    cfg = build_train_config(load_yaml(cfg_path))
    kwargs = loader_kwargs_from_config(cfg)
    assert "split_strategy" in kwargs
    assert isinstance(kwargs["split_strategy"], BinnedDensityStrategy)
    assert kwargs["split_strategy"].n_bins == 4


def test_experiment_config_to_dict_is_json_serializable(tmp_path: Path) -> None:
    """The dict we hand to TrainingRunRecord.config has no Path / tuple types."""
    import json

    cfg_path = _write_yaml(tmp_path, "data:\n  bank: ./x.h5\n")
    cfg = build_train_config(load_yaml(cfg_path))
    d = experiment_config_to_dict(cfg)
    # split_strategy is serialized as a plain {type, n_bins} dict in
    # the run.json config block (not the live PatientStratificationStrategy).
    assert d["data"]["split_strategy"] == {"type": "any_positive", "n_bins": 3}
    # json.dumps will raise if any Path or tuple slipped through.
    json.dumps(d)
    # Sanity: top-level shape matches the four blocks.
    assert set(d.keys()) == {"model", "data", "train", "output"}
    assert d["data"]["bank"].endswith("x.h5")
    assert isinstance(d["data"]["split_fractions"], list)
