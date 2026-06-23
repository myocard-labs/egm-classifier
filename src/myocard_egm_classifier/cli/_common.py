"""Shared CLI utilities — used by every command in ``cli/``.

Two families of helpers live here, pulled into ``train_cmd``,
``eval_cmd``, and ``export_cmd`` (when it lands in PR C):

1. **Runtime helpers** — :func:`set_seed`, :func:`select_device`,
   :func:`build_model_from_meta`, :func:`load_checkpoint_model`.
   Seeding/device pieces happen at the top of every CLI's main; the
   meta-driven model builder + checkpoint loader live here so eval and
   export rebuild the model from the same embedded ``model_meta`` the
   trainer wrote.

2. **Generic YAML/config helpers** — :exc:`ConfigError`,
   :func:`load_yaml`, :func:`_optional`, :func:`_resolve_path`,
   :func:`_reject_unknown_keys`. The shape every command's YAML loader
   needs: parse, validate keys, resolve paths against the config file's
   directory, fail loudly on typos.

CLI-specific config (typed dataclasses, model triad, run-record
serializer, runtime translators) lives in sibling per-CLI modules:
:mod:`._train_config`, :mod:`._eval_config`. Notably, the
``ModelCLIConfig`` dataclass + its YAML→config builder live in
:mod:`._train_config` — eval and export rebuild the model directly
from the checkpoint's ``model_meta`` dict via
:func:`build_model_from_meta` and have no need for ``ModelCLIConfig``.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from myocard_egm_classifier.constants import (
    DEFAULT_HEAD_DROPOUT,
    DEFAULT_HEAD_EXPANSION_CHANNELS,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_NUM_CLASSES,
    DEFAULT_STOCHASTIC_DEPTH,
    DEFAULT_WIDTH_MULTIPLIER,
)
from myocard_egm_classifier.models import MobileViT1D, default_v1_blocks

# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (incl. CUDA) RNGs.

    Trained models still drift across runs because of cudnn nondeterminism
    and DataLoader worker startup; this is reproducibility insurance, not
    a guarantee.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(override: str | None) -> torch.device:
    """Pick the torch device, honoring an explicit ``--device`` override.

    Returns the override verbatim when given. Otherwise auto-picks CUDA
    if available, falling back to CPU.
    """
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_from_meta(meta: dict[str, Any]) -> MobileViT1D:
    """Construct a :class:`MobileViT1D` from a checkpoint ``model_meta`` dict.

    The inverse of the trainer's ``model_meta_from_config`` write step —
    given the dict that was embedded alongside ``model_state_dict``,
    materialize the architecture so :meth:`torch.nn.Module.load_state_dict`
    has something to attach the weights to. Used by eval and export;
    neither has any reason to round-trip through ``ModelCLIConfig``.

    The block list is always the v1 default template; the meta dict
    doesn't carry an explicit ``blocks`` field, so a checkpoint trained
    with a custom ``model.blocks`` YAML override can't be rebuilt this
    way. That's an existing limitation of the meta serialization, not
    something this function introduces.
    """
    blocks = default_v1_blocks(
        num_outputs=int(meta.get("num_classes", DEFAULT_NUM_CLASSES)),
        head_expansion_channels=int(
            meta.get("head_expansion_channels", DEFAULT_HEAD_EXPANSION_CHANNELS)
        ),
        head_dropout=float(meta.get("head_dropout", DEFAULT_HEAD_DROPOUT)),
    )
    return MobileViT1D(
        blocks=blocks,
        width_multiplier=float(meta.get("width_multiplier", DEFAULT_WIDTH_MULTIPLIER)),
        in_channels=int(meta.get("in_channels", DEFAULT_INPUT_CHANNELS)),
        stochastic_depth=float(meta.get("stochastic_depth", DEFAULT_STOCHASTIC_DEPTH)),
    )


def load_checkpoint_model(
    checkpoint_path: Path | str, device: torch.device
) -> tuple[MobileViT1D, dict[str, Any]]:
    """Rebuild a model from a training checkpoint and load its weights.

    Returns ``(model, checkpoint_dict)``. The model is built directly
    from the embedded ``model_meta`` dict via
    :func:`build_model_from_meta` — eval/export never need a separate
    ``ModelCLIConfig``, since every architecture knob they care about
    is already in that dict. Raises :class:`ValueError` if the
    checkpoint predates the model-meta embedding convention.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = ckpt.get("model_meta", {})
    if not meta:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no model_meta; cannot rebuild "
            "the architecture. Re-train with this version to embed it."
        )
    model = build_model_from_meta(meta)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Generic YAML / config helpers
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when a config file is malformed or missing required keys."""


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Resolves the parent path into the returned dict under
    ``_config_dir`` so subsequent relative paths in the doc resolve
    against the YAML's directory (same convention iafdb-pipeline and
    synthetic-egm-pipeline use).
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config {p} did not parse as a YAML mapping at the top level.")
    loaded["_config_dir"] = p.parent.resolve()
    return loaded


def _optional(doc: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Walk ``path`` into the nested dict; return ``default`` if missing.

    Used by both train and eval config builders to read optional
    blocks (``data:``, ``model:``, ``output:``, ``train:``) with a
    fallback to an empty dict so the builder always sees something to
    iterate over.
    """
    node: Any = doc
    for k in path:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def _resolve_path(value: str | None, config_dir: Path) -> Path | None:
    """Resolve a YAML-supplied path against the config file's dir.

    ``None`` / empty string returns ``None``. Absolute paths pass
    through; relative paths resolve against ``config_dir``.
    """
    if value is None or value == "":
        return None
    p = Path(value)
    return p if p.is_absolute() else (config_dir / p).resolve()


def _reject_unknown_keys(block: dict[str, Any], allowed: set[str], *, label: str) -> None:
    """Loudly fail when a YAML block has typo'd or stale keys.

    Catches things like ``model.with_multiplier`` vs ``width_multiplier``
    at config-load time instead of silently dropping the override.
    ``_config_dir`` is filtered out because :func:`load_yaml` stuffs it
    into the top-level doc as a side channel.
    """
    keys = {k for k in block if k != "_config_dir"}
    unknown = keys - allowed
    if unknown:
        raise ConfigError(f"Unknown keys in {label}: {sorted(unknown)}")
