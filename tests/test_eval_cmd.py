"""End-to-end smoke for the ``egm-class-eval`` CLI.

Builds a fresh untrained MobileViT1D, saves it as a checkpoint (with
the ``model_meta`` block the eval CLI rebuilds from), then runs the
eval CLI's ``main`` against the synthetic ClassifierBank fixture from
``conftest.py``. The untrained model's predictions are essentially
random — we don't care about metric values, we care that:

1. The CLI returns 0 (no error path).
2. The sibling ``_pred.cbank.h5`` is created at the expected location.
3. Every trace in the loaded predictions bank carries a populated
   :class:`ClassifierPrediction` with the right field shapes.
4. The default sibling-path derivation strips the right suffixes
   (covered by a separate config-level test).

Pairs with ``test_cli_config.py`` (which covers the YAML loader,
overrides, and validation paths for eval-side config).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from myocard_egm_data.banks import (
    ClassifierBank,
    ClassifierPrediction,
    load_classifier_bank,
    write_classifier_bank,
)

from myocard_egm_classifier.cli.eval_cmd import main as eval_main
from myocard_egm_classifier.eval import default_predictions_bank_path
from myocard_egm_classifier.models import MobileViT1D, default_v1_blocks


def _make_checkpoint(
    tmp_path: Path,
    input_length: int,
    training_provenance: dict[str, Any] | None = None,
) -> Path:
    """Build a tiny MobileViT1D and save it as a checkpoint with model_meta.

    Untrained — the eval CLI rebuilds the architecture from
    ``model_meta`` and loads ``model_state_dict`` from this file, so
    the actual weights' quality is irrelevant for the round-trip
    check. ``training_provenance`` (the cross-artifact ids the trainer
    stamps) is embedded only when supplied.
    """
    model = MobileViT1D(
        blocks=default_v1_blocks(num_outputs=1),
        width_multiplier=0.5,
        in_channels=1,
        stochastic_depth=0.0,
    )
    ckpt_path = tmp_path / "best.pt"
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_loss": 0.5,
        "val_metrics": {"auroc": 0.5},
        "model_meta": {
            "arch": "mobilevit_1d_v1",
            "width_multiplier": 0.5,
            "num_classes": 1,
            "in_channels": 1,
            "input_length": input_length,
            "stochastic_depth": 0.0,
            "head_expansion_channels": 320,
            "head_dropout": 0.1,
        },
    }
    if training_provenance is not None:
        payload["training_provenance"] = training_provenance
    torch.save(payload, ckpt_path)
    return ckpt_path


def _unlabeled_bank(bank: ClassifierBank) -> ClassifierBank:
    """Copy a labeled fixture bank with every trace's ``label_truth`` cleared.

    Mirrors the IAFDB shape (predictions desired, no substrate truth), so
    the eval CLI takes the ``upred_`` / metrics-skipped path.
    """
    traces = [dataclasses.replace(t, label_truth=None, prediction=None) for t in bank.traces]
    return dataclasses.replace(bank, traces=traces, id=None)


def _write_yaml(tmp_path: Path, body: str, name: str = "eval.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def _write_input_bank(tmp_path: Path, bank: ClassifierBank) -> Path:
    """Write the fixture bank to a temp file under the canonical extension."""
    input_path = tmp_path / "tiny_test.cbank.h5"
    write_classifier_bank(bank, input_path)
    return input_path


def test_eval_cmd_writes_sibling_predictions_bank(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """Run eval end-to-end; verify sibling predictions bank + populated traces."""
    input_length = 64  # matches the fixture's signal length
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    ckpt_path = _make_checkpoint(tmp_path, input_length=input_length)

    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {bank_path}
  batch_size: 8
""",
    )

    rc = eval_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0

    # Predictions bank lands at the default sibling path.
    expected_pred_path = default_predictions_bank_path(bank_path)
    assert expected_pred_path.exists()

    # Round-trip the predictions bank and verify every trace got a
    # populated ClassifierPrediction.
    pred_bank = load_classifier_bank(expected_pred_path)
    assert pred_bank.n_traces == tiny_classifier_bank.n_traces
    for trace in pred_bank.traces:
        assert trace.prediction is not None
        pred = trace.prediction
        assert isinstance(pred, ClassifierPrediction)
        assert pred.label_pred in (0, 1)
        assert 0.0 <= pred.label_prob <= 1.0
        # Binary head emits {0: -logit, 1: logit}.
        assert set(pred.pred_logits.keys()) == {0, 1}
        assert pred.pred_logits[0] == pytest.approx(-pred.pred_logits[1])


def test_eval_cmd_honors_explicit_predictions_bank_override(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """An explicit ``output.predictions_bank`` overrides the sibling default."""
    input_length = 64
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    ckpt_path = _make_checkpoint(tmp_path, input_length=input_length)
    custom_out = tmp_path / "custom_predictions.cbank.h5"

    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {bank_path}
  batch_size: 8
output:
  predictions_bank: {custom_out}
""",
    )

    rc = eval_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0
    assert custom_out.exists()
    # And the default sibling path is NOT also written.
    sibling = default_predictions_bank_path(bank_path)
    assert not sibling.exists()


def test_eval_cmd_threshold_override_flips_label_pred(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """At threshold 0.99 almost every untrained-model prediction is class 0;
    at threshold 0.01 almost every prediction is class 1. Exercises the
    threshold-override path without depending on the model's actual outputs."""
    input_length = 64
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    ckpt_path = _make_checkpoint(tmp_path, input_length=input_length)

    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {bank_path}
  batch_size: 8
""",
    )

    # High threshold => almost every prediction is class 0.
    high_out = tmp_path / "high.cbank.h5"
    rc = eval_main(
        [
            str(cfg_path),
            "--device",
            "cpu",
            "--threshold",
            "0.99",
            "--predictions-bank",
            str(high_out),
        ]
    )
    assert rc == 0
    high_bank = load_classifier_bank(high_out)
    high_class_one = sum(
        1 for t in high_bank.traces if t.prediction is not None and t.prediction.label_pred == 1
    )

    # Low threshold => almost every prediction is class 1.
    low_out = tmp_path / "low.cbank.h5"
    rc = eval_main(
        [
            str(cfg_path),
            "--device",
            "cpu",
            "--threshold",
            "0.01",
            "--predictions-bank",
            str(low_out),
        ]
    )
    assert rc == 0
    low_bank = load_classifier_bank(low_out)
    low_class_one = sum(
        1 for t in low_bank.traces if t.prediction is not None and t.prediction.label_pred == 1
    )

    # Lower threshold should monotonically classify more traces as positive.
    assert low_class_one >= high_class_one


def test_default_predictions_bank_path_strips_suffixes() -> None:
    """The sibling-path derivation handles the three filename conventions
    cleanly: .cbank.h5, .h5, and bare-stem."""
    assert default_predictions_bank_path(Path("/x/y.cbank.h5")) == Path("/x/y_pred.cbank.h5")
    assert default_predictions_bank_path(Path("/x/y.h5")) == Path("/x/y_pred.cbank.h5")
    assert default_predictions_bank_path(Path("/x/y")) == Path("/x/y_pred.cbank.h5")


def test_eval_cmd_rejects_missing_bank_with_clear_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Pointing at a nonexistent bank exits non-zero with a clear stderr message."""
    ckpt_path = _make_checkpoint(tmp_path, input_length=64)
    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {tmp_path / "does_not_exist.cbank.h5"}
""",
    )

    rc = eval_main([str(cfg_path), "--device", "cpu"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "loading bank" in err.lower() or "does_not_exist" in err


def test_eval_cmd_stamps_lpred_id_and_model_id(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """A labeled bank => an ``lpred_`` predictions-bank id; the producing
    model's id lands in every trace's ``trace_metadata`` (the short-term
    home for the model->prediction link)."""
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    prov = {
        "run_id": "run_eval_smoke_2026-06-27",
        "produced_model_id": "model_eval_smoke_2026-06-27",
        "trained_on_bank_id": "tbank_eval_smoke_2026-06-27",
        "run_name": "eval_smoke",
    }
    ckpt_path = _make_checkpoint(tmp_path, input_length=64, training_provenance=prov)

    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {bank_path}
  batch_size: 8
""",
    )

    rc = eval_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0

    pred_bank = load_classifier_bank(default_predictions_bank_path(bank_path))
    assert pred_bank.id is not None
    assert pred_bank.id.startswith("lpred_eval_smoke_")
    for trace in pred_bank.traces:
        assert trace.trace_metadata["produced_by_model_id"] == "model_eval_smoke_2026-06-27"


def test_eval_cmd_unlabeled_bank_writes_upred_and_skips_metrics(
    tmp_path: Path,
    tiny_classifier_bank: ClassifierBank,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unlabeled bank yields a ``upred_`` predictions bank, exits 0, and
    skips metrics (no substrate truth) — predictions are still stamped."""
    unlabeled = _unlabeled_bank(tiny_classifier_bank)
    bank_path = _write_input_bank(tmp_path, unlabeled)
    ckpt_path = _make_checkpoint(tmp_path, input_length=64)

    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {bank_path}
  batch_size: 8
""",
    )

    rc = eval_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "metrics skipped" in out.lower()

    pred_bank = load_classifier_bank(default_predictions_bank_path(bank_path))
    assert pred_bank.id is not None
    assert pred_bank.id.startswith("upred_")
    # Predictions are still populated even though metrics were skipped.
    assert all(t.prediction is not None for t in pred_bank.traces)


def test_eval_cmd_output_bank_id_override(
    tmp_path: Path, tiny_classifier_bank: ClassifierBank
) -> None:
    """An explicit ``output.bank_id`` is used verbatim as the predictions id."""
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    ckpt_path = _make_checkpoint(tmp_path, input_length=64)

    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {bank_path}
  batch_size: 8
output:
  bank_id: lpred_custom_override_2026-06-27
""",
    )

    rc = eval_main([str(cfg_path), "--device", "cpu"])
    assert rc == 0
    pred_bank = load_classifier_bank(default_predictions_bank_path(bank_path))
    assert pred_bank.id == "lpred_custom_override_2026-06-27"


def test_eval_cmd_rejects_invalid_bank_id_override(
    tmp_path: Path,
    tiny_classifier_bank: ClassifierBank,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A malformed ``output.bank_id`` fails fast with a clear error (rc 2)."""
    bank_path = _write_input_bank(tmp_path, tiny_classifier_bank)
    ckpt_path = _make_checkpoint(tmp_path, input_length=64)

    cfg_path = _write_yaml(
        tmp_path,
        f"""\
checkpoint: {ckpt_path}
data:
  bank: {bank_path}
  batch_size: 8
output:
  bank_id: "Not A Valid Id"
""",
    )

    rc = eval_main([str(cfg_path), "--device", "cpu"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not a valid stable artifact id" in err


def test_eval_config_parses_output_bank_id(tmp_path: Path) -> None:
    """``build_eval_config`` reads ``output.bank_id`` verbatim (the format is
    validated at use time, not config-build time)."""
    from myocard_egm_classifier.cli._common import load_yaml
    from myocard_egm_classifier.cli._eval_config import build_eval_config

    cfg_path = _write_yaml(
        tmp_path,
        "checkpoint: ./best.pt\n"
        "data:\n  bank: ./x.cbank.h5\n"
        "output:\n  bank_id: lpred_x_2026-06-27\n",
    )
    cfg = build_eval_config(load_yaml(cfg_path))
    assert cfg.output.bank_id == "lpred_x_2026-06-27"
