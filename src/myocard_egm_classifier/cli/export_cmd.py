"""CLI: export a trained checkpoint to ONNX + a metadata sidecar.

Wired as the ``egm-class-export`` console script.

Usage
-----
::

    egm-class-export CONFIG.yaml [--checkpoint PATH] [--calibration-bank PATH]
                                  [--skip-calibration] [--output-dir PATH]
                                  [--output-name NAME] [--threshold F]
                                  [--opset N] [--device cuda|cpu]

Flow:

1. Load + validate the YAML; apply argparse overrides.
2. Rebuild the model from the checkpoint's embedded ``model_meta``
   (the eval/export shared path — same code as ``egm-class-eval``).
3. Optional calibration: if ``calibration.bank`` is set and
   ``--skip-calibration`` was not passed, run the model over the
   bank to collect logits + labels, then fit a single scalar
   temperature ``T`` via egm-signal's bounded NLL minimizer. When
   skipped, ``T = 1.0``.
4. Wrap the model with :class:`CalibratedModel` so its forward emits
   ``base_model(x) / T``.
5. Export the wrapped model to ONNX with a dynamic batch axis,
   ``input_name="signal"``, ``output_name="logit"``, opset pinned
   from the YAML.
6. Compose the ``egm_class_model_metadata.json`` sidecar (model
   artifact hash + size, input/output specs, preprocessing,
   decision, training provenance with the fitted ``T``) and write
   it next to the ONNX.

Both output files live under ``output.dir``:
``<output.name>.onnx`` and ``<output.name>.model_metadata.json``.

The CLI assumes the v1 single-logit binary head (``num_classes == 1``
in the checkpoint's ``model_meta``). Multi-class export is a future
addition.

Module layout: this file is intentionally thin — argparse, config
loading, and the orchestration that wires together
:mod:`myocard_egm_classifier.export.calibration`,
:mod:`myocard_egm_classifier.export.graph_wrapper`,
:mod:`myocard_egm_classifier.export.onnx_export`, and
:mod:`myocard_egm_classifier.export.metadata`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from myocard_egm_data.banks import load_classifier_bank

from myocard_egm_classifier.cli._common import (
    ConfigError,
    load_checkpoint_model,
    load_yaml,
    select_device,
    set_seed,
)
from myocard_egm_classifier.cli._export_config import (
    ExportCLIOverrides,
    apply_export_overrides,
    build_export_config,
)
from myocard_egm_classifier.constants import (
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_LENGTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_SEED,
)
from myocard_egm_classifier.export import (
    CalibratedModel,
    build_metadata,
    export_to_onnx,
    fit_temperature_from_bank,
    write_metadata,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egm-class-export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=Path, help="Path to the export YAML config.")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override config.checkpoint.")
    p.add_argument(
        "--calibration-bank",
        type=Path,
        default=None,
        help="Override calibration.bank (path to a labeled ClassifierBank for fitting T).",
    )
    p.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Skip the temperature fit and ship with T = 1.0. Wins over --calibration-bank.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output.dir.",
    )
    p.add_argument(
        "--output-name",
        default=None,
        help="Override output.name (base filename for <name>.onnx + <name>.model_metadata.json).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision.threshold (must be in (0, 1)).",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Override the ONNX operator-set version (positive integer).",
    )
    p.add_argument("--device", default=None, help='Override device ("cuda", "cpu").')
    return p


def _overrides_from_args(args: argparse.Namespace) -> ExportCLIOverrides:
    return ExportCLIOverrides(
        checkpoint=args.checkpoint,
        calibration_bank=args.calibration_bank,
        output_dir=args.output_dir,
        output_name=args.output_name,
        threshold=args.threshold,
        opset=args.opset,
        skip_calibration=args.skip_calibration,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        doc = load_yaml(args.config)
        cfg = apply_export_overrides(build_export_config(doc), _overrides_from_args(args))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    set_seed(DEFAULT_SEED)
    device = select_device(args.device)
    print(f"Using device: {device}")

    try:
        model, ckpt = load_checkpoint_model(cfg.checkpoint, device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR loading checkpoint: {exc}", file=sys.stderr)
        return 1

    # Pull the three architecture scalars we need straight from the
    # checkpoint's model_meta dict — no need to round-trip through a
    # ModelCLIConfig for a few lookups.
    meta: dict[str, Any] = ckpt.get("model_meta", {})
    num_classes = int(meta.get("num_classes", DEFAULT_NUM_CLASSES))
    in_channels = int(meta.get("in_channels", DEFAULT_INPUT_CHANNELS))
    input_length = int(meta.get("input_length", DEFAULT_INPUT_LENGTH))
    if num_classes != 1:
        print(
            "ERROR: export CLI only supports the single-logit binary head "
            "(num_classes == 1). Multi-class export is a future addition.",
            file=sys.stderr,
        )
        return 1

    # Optional temperature fit. When skipped (no bank, or --skip-calibration),
    # we ship with T = 1.0 — the graph still goes through CalibratedModel so
    # the export path is uniform, just with a no-op divisor.
    temperature = 1.0
    if cfg.calibration.bank is not None:
        print(f"Fitting temperature against calibration bank: {cfg.calibration.bank}")
        try:
            bank = load_classifier_bank(cfg.calibration.bank)
        except FileNotFoundError as exc:
            print(f"ERROR loading calibration bank: {exc}", file=sys.stderr)
            return 1
        try:
            temperature = fit_temperature_from_bank(
                model,
                bank,
                input_length=input_length,
                normalization_scheme=cfg.preprocessing.normalization_scheme,
                normalization_eps=cfg.preprocessing.normalization_eps,
                batch_size=cfg.calibration.batch_size,
                num_workers=cfg.calibration.num_workers,
                pin_memory=cfg.calibration.pin_memory,
                device=device,
            )
        except ValueError as exc:
            print(f"ERROR fitting temperature: {exc}", file=sys.stderr)
            return 1
        print(f"  fitted T = {temperature:.4f}")
    else:
        print("Skipping calibration fit; shipping with T = 1.0")

    wrapped = CalibratedModel(model, temperature=temperature).to(device).eval()

    # Resolve output paths up front so we can fail fast on a clobber attempt
    # before doing the expensive export pass.
    onnx_path = cfg.output.dir / f"{cfg.output.name}.onnx"
    metadata_path = cfg.output.dir / f"{cfg.output.name}.model_metadata.json"
    for existing in (onnx_path, metadata_path):
        if existing.exists():
            print(
                f"ERROR: {existing} already exists. "
                "Delete it or pass --output-name to a fresh value.",
                file=sys.stderr,
            )
            return 1

    print(f"Exporting ONNX to {onnx_path} (opset {cfg.opset})")
    try:
        export_to_onnx(
            wrapped,
            output_path=onnx_path,
            input_length=input_length,
            in_channels=in_channels,
            opset=cfg.opset,
            device=device,
        )
    except ModuleNotFoundError as exc:
        # The torch.onnx.export call pulls in `onnx` (and sometimes
        # `onnxscript`) transitively to serialize the graph. Neither is
        # imported anywhere in egm-classifier directly, so the import
        # failure only fires here. Translate into a friendly install hint
        # so the user knows which extra to install rather than parsing a
        # torch.onnx-internal traceback.
        print(
            f"ERROR: {exc}\n"
            "Hint: egm-class-export requires the optional ONNX toolchain. "
            "Install with:\n"
            "    pip install myocard-egm-classifier[onnx]",
            file=sys.stderr,
        )
        return 1

    metadata = build_metadata(
        cfg=cfg,
        checkpoint_dict=ckpt,
        onnx_path=onnx_path,
        input_length=input_length,
        in_channels=in_channels,
        num_outputs=num_classes,
        temperature=temperature,
    )
    written = write_metadata(metadata, dir_=cfg.output.dir, base_name=cfg.output.name)
    print(f"Wrote metadata to {written}")
    print()
    print("Export complete:")
    prov = ckpt.get("training_provenance", {})
    model_id = prov.get("produced_model_id") if isinstance(prov, dict) else None
    if model_id:
        print(f"  model_id: {model_id}")
    print(f"  {onnx_path.name}    ({onnx_path.stat().st_size / 1024:.1f} KiB)")
    print(f"  {written.name}      ({written.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
