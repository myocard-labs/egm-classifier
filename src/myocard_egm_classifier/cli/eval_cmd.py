"""CLI: evaluate a trained checkpoint against a labeled ClassifierBank.

Wired as the ``egm-class-eval`` console script.

Usage
-----
::

    egm-class-eval CONFIG.yaml [--checkpoint PATH] [--bank PATH]
                               [--predictions-bank PATH] [--threshold F]
                               [--device cuda|cpu]

Flow:

1. Rebuild the model from the checkpoint's embedded ``model_meta``
   (no separate ``model:`` block in the eval YAML).
2. Load the input ClassifierBank.
3. Run inference sequentially over every trace (no patient-aware
   split, no augmentation). Logits are collected in bank order.
4. Populate ``ClassifierPrediction`` on every trace, stamp the
   producing model's id + the predictions bank's own stable
   cross-artifact id (``lpred_`` labeled / ``upred_`` unlabeled), and
   write a sibling ``<stem>_pred.cbank.h5`` next to the input
   (configurable via ``output.predictions_bank``).
5. For a labeled bank, compute the standard binary-metric bundle from
   the inference logits + bank labels and print it to stdout. For an
   unlabeled bank, skip metrics. **No metrics file is written** — the
   raw logits and labels live on the predictions bank, so any
   downstream analysis can re-derive these (or any other metric) from
   there.

The CLI supports both labeled and unlabeled input banks. A fully-
labeled bank (e.g. a synthetic test bank) yields a scored eval — an
``lpred_`` predictions bank plus the scalar metric bundle on stdout.
An unlabeled bank (the IAFDB shape) yields a ``upred_`` predictions
bank with metrics skipped: predictions are still useful for
qualitative inspection, but substrate-truth metrics can't be computed
and aren't faked — see ``project_iafdb_eval_catch22`` for context.

Calibration is **not** applied in eval. The predicted probabilities
are the raw ``sigmoid(logit)``. Calibration (temperature scaling)
happens at ONNX export time (PR C), so any downstream analysis can
choose to re-fit and re-apply against the raw logits preserved on
the predictions bank.

Module layout: this file is intentionally thin — argparse, config
loading, and the orchestration that wires together the functional
helpers in :mod:`myocard_egm_classifier.eval.inference` and
:mod:`myocard_egm_classifier.eval.predictions`. Logit collection is
the shared top-level
:func:`myocard_egm_classifier.inference_helpers.collect_logits` (used by
training's per-epoch evaluator too); metric computation is the
shared top-level :func:`myocard_egm_classifier.metrics.binary_metrics`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from myocard_egm_data.banks import load_classifier_bank, write_classifier_bank
from torch.utils.data import DataLoader

from myocard_egm_classifier.cli._common import (
    ConfigError,
    load_checkpoint_model,
    load_yaml,
    select_device,
    set_seed,
)
from myocard_egm_classifier.cli._eval_config import (
    EvalCLIOverrides,
    apply_eval_overrides,
    build_eval_config,
)
from myocard_egm_classifier.constants import (
    DEFAULT_INPUT_LENGTH,
    DEFAULT_NUM_CLASSES,
    DEFAULT_SEED,
)
from myocard_egm_classifier.eval import (
    build_eval_dataset,
    default_predictions_bank_path,
    populate_predictions,
    stamp_predictions_model_id,
)
from myocard_egm_classifier.ids import derive_predictions_bank_id, validate_artifact_id
from myocard_egm_classifier.inference_helpers import collect_logits
from myocard_egm_classifier.metrics import binary_metrics


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="egm-class-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("config", type=Path, help="Path to the eval YAML config.")
    p.add_argument("--checkpoint", type=Path, default=None, help="Override config.checkpoint.")
    p.add_argument("--bank", type=Path, default=None, help="Override data.bank.")
    p.add_argument(
        "--predictions-bank",
        type=Path,
        default=None,
        help="Override output.predictions_bank.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the decision threshold for label_pred (default 0.5).",
    )
    p.add_argument("--device", default=None, help='Override device ("cuda", "cpu").')
    return p


def _overrides_from_args(args: argparse.Namespace) -> EvalCLIOverrides:
    return EvalCLIOverrides(
        bank=args.bank,
        checkpoint=args.checkpoint,
        predictions_bank=args.predictions_bank,
        threshold=args.threshold,
    )


def _format_metrics(metrics: dict[str, Any], threshold: float) -> str:
    """One-block summary of the eval metrics for stdout.

    Kept in the CLI module rather than ``eval/`` because it's pure
    presentation glue for the ``egm-class-eval`` stdout layout — it
    doesn't compute anything, just formats a finished metrics dict.
    """
    lines = [
        f"Eval metrics (threshold={threshold}):",
        f"  n={metrics['n']}",
        f"  accuracy = {metrics['accuracy']:.4f}",
        f"  auroc    = {metrics['auroc']:.4f}",
        f"  f1       = {metrics['f1']:.4f}",
        f"  precision= {metrics['precision']:.4f}",
        f"  recall   = {metrics['recall']:.4f}",
        f"  ece      = {metrics['ece']:.4f}",
        f"  confusion= {metrics['confusion']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    try:
        doc = load_yaml(args.config)
        cfg = apply_eval_overrides(build_eval_config(doc), _overrides_from_args(args))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Resolve the predictions-bank output path: explicit config wins; otherwise
    # derive the sibling _pred.cbank.h5 next to the input.
    assert cfg.data.bank is not None  # build_eval_config rejects None
    predictions_bank_path = cfg.output.predictions_bank or default_predictions_bank_path(
        cfg.data.bank
    )

    set_seed(DEFAULT_SEED)
    device = select_device(args.device)
    print(f"Using device: {device}")

    try:
        model, ckpt = load_checkpoint_model(cfg.checkpoint, device)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR loading checkpoint: {exc}", file=sys.stderr)
        return 1

    # Pull the two architecture knobs we actually need straight from the
    # checkpoint's model_meta dict — no need to round-trip through a
    # ModelCLIConfig dataclass for two scalar lookups.
    meta: dict[str, Any] = ckpt.get("model_meta", {})
    num_classes = int(meta.get("num_classes", DEFAULT_NUM_CLASSES))
    input_length = int(meta.get("input_length", DEFAULT_INPUT_LENGTH))
    if num_classes != 1:
        print(
            "ERROR: eval CLI only supports the single-logit binary head "
            "(num_classes == 1). Multi-class eval is a future addition.",
            file=sys.stderr,
        )
        return 1

    try:
        bank = load_classifier_bank(cfg.data.bank)
    except FileNotFoundError as exc:
        print(f"ERROR loading bank: {exc}", file=sys.stderr)
        return 1

    print(
        f"Checkpoint: {cfg.checkpoint}\n"
        f"Bank:       {cfg.data.bank}\n"
        f"  {bank.n_traces} traces, model input_length={input_length}"
    )

    dataset = build_eval_dataset(
        bank,
        input_length=input_length,
        znorm=cfg.data.znorm,
        znorm_eps=cfg.data.znorm_eps,
    )
    loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle=False,  # bank-order iteration required for prediction stamping
        drop_last=False,
    )

    # collect_logits returns (logits, labels); the labels are already on the
    # bank so we don't need them here. They'd just be a redundant copy of
    # bank.label_truth_array() in the same order.
    logits, _ = collect_logits(model, loader, device)
    if logits.shape[0] != bank.n_traces:
        print(
            f"ERROR: collected {logits.shape[0]} logits, bank has {bank.n_traces} traces.",
            file=sys.stderr,
        )
        return 1

    populate_predictions(bank, logits, threshold=cfg.threshold)

    # Determine labeled-ness without raising (label_truth_array() raises on
    # any unlabeled trace). A fully-labeled bank yields a scored eval
    # (lpred_, full metric suite); an unlabeled / partially-labeled bank is
    # a label-free diagnostic (upred_, predictions only) — the IAFDB shape,
    # per project_iafdb_eval_catch22 + feedback_iafdb_unlabeled_no_ml_validation.
    labeled = bool(bank.traces) and all(t.label_truth is not None for t in bank.traces)

    if labeled:
        labels = bank.label_truth_array()
        metrics = binary_metrics(logits, labels, threshold=cfg.threshold)
        print()
        print(_format_metrics(metrics, threshold=cfg.threshold))
        print()
    else:
        print()
        print(
            "Bank has unlabeled traces — writing a label-free predictions "
            "bank (upred_); metrics skipped (no truth labels)."
        )
        print()

    # Cross-artifact provenance. The producing model's id is stamped onto
    # each trace's trace_metadata (short-term home for the model->prediction
    # link; the predictions bank has no dedicated model_id field yet). The
    # predictions bank's own stable id is an explicit config override, else
    # a derived lpred_/upred_ from the labeled-ness + the producing run's
    # name (both read from the checkpoint's training_provenance).
    prov = ckpt.get("training_provenance", {})
    model_id = prov.get("produced_model_id") if isinstance(prov, dict) else None
    run_name = prov.get("run_name") if isinstance(prov, dict) else None
    if model_id:
        stamp_predictions_model_id(bank, model_id)

    if cfg.output.bank_id is not None:
        try:
            bank.id = validate_artifact_id(cfg.output.bank_id)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        bank.id = derive_predictions_bank_id(labeled=labeled, run_name=run_name)
    print(f"Predictions bank id: {bank.id}")

    try:
        out_path = write_classifier_bank(bank, predictions_bank_path)
    except FileExistsError as exc:
        print(
            f"ERROR: {exc}\n"
            "Hint: delete the previous predictions bank or pass --predictions-bank to a fresh path.",
            file=sys.stderr,
        )
        return 1
    print(f"Wrote predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
