"""Shared CLI runtime helpers: seeding, device selection, checkpoint rebuild.

These are pulled into ``train_cmd``, ``eval_cmd``, and ``export_cmd`` —
the seeding/device pieces happen at the top of every CLI's main; the
checkpoint loader lives here so eval and export both rebuild the model
from the same ``model_meta`` the trainer embedded.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from myocard_egm_classifier.cli._config import build_model_from_config, model_config_from_meta
from myocard_egm_classifier.models.mobilevit1d import MobileViT1D


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


def load_checkpoint_model(
    checkpoint_path: Path | str, device: torch.device
) -> tuple[MobileViT1D, dict[str, Any]]:
    """Rebuild a model from a training checkpoint and load its weights.

    Returns ``(model, checkpoint_dict)``. The model is rebuilt from the
    ``model_meta`` block the trainer embedded, so eval/export never need
    to be told the architecture separately. Raises ``ValueError`` if the
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
    model_cfg = model_config_from_meta(meta)
    model = build_model_from_config(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt
