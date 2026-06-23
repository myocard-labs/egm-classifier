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

from pathlib import Path

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


def _make_checkpoint(tmp_path: Path, input_length: int) -> Path:
    """Build a tiny MobileViT1D and save it as a checkpoint with model_meta.

    Untrained — the eval CLI rebuilds the architecture from
    ``model_meta`` and loads ``model_state_dict`` from this file, so
    the actual weights' quality is irrelevant for the round-trip
    check.
    """
    model = MobileViT1D(
        blocks=default_v1_blocks(num_outputs=1),
        width_multiplier=0.5,
        in_channels=1,
        stochastic_depth=0.0,
    )
    ckpt_path = tmp_path / "best.pt"
    torch.save(
        {
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
        },
        ckpt_path,
    )
    return ckpt_path


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
