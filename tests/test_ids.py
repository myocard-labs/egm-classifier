"""Unit tests for the stable cross-artifact id helpers (``ids.py``).

The id *pattern* itself is single-sourced + tested in egm-contracts; these
tests cover this package's composition layer: the ``run_``/``model_``/
``lpred_``/``upred_`` recipes, the run-name descriptor sanitization +
default fallback, and the fail-fast override validator. ``today=`` is
passed explicitly so the date segment is deterministic.
"""

from __future__ import annotations

import pytest

from myocard_egm_classifier.ids import (
    DEFAULT_RUN_NAME,
    derive_model_id,
    derive_predictions_bank_id,
    derive_run_id,
    validate_artifact_id,
)

_DATE = "2026-06-27"


def test_derive_run_id_uses_run_name_and_date() -> None:
    assert derive_run_id("v1_5_courtemanche", today=_DATE) == "run_v1_5_courtemanche_2026-06-27"


def test_derive_model_id_uses_run_name_and_date() -> None:
    assert derive_model_id("v1_5_courtemanche", today=_DATE) == "model_v1_5_courtemanche_2026-06-27"


def test_derive_predictions_bank_id_labeled_vs_unlabeled() -> None:
    """``labeled`` flips the role prefix between ``lpred_`` and ``upred_``."""
    assert (
        derive_predictions_bank_id(labeled=True, run_name="holdout", today=_DATE)
        == "lpred_holdout_2026-06-27"
    )
    assert (
        derive_predictions_bank_id(labeled=False, run_name="iafdb", today=_DATE)
        == "upred_iafdb_2026-06-27"
    )


def test_empty_or_none_run_name_falls_back_to_default() -> None:
    assert derive_run_id(None, today=_DATE) == f"run_{DEFAULT_RUN_NAME}_2026-06-27"
    assert (
        derive_predictions_bank_id(labeled=True, run_name="", today=_DATE)
        == f"lpred_{DEFAULT_RUN_NAME}_2026-06-27"
    )


def test_run_name_is_sanitized_to_the_id_charset() -> None:
    """Uppercase, spaces, and punctuation collapse to ``[a-z0-9_]``."""
    assert derive_run_id("V1.5 Courtemanche!", today=_DATE) == "run_v1_5_courtemanche_2026-06-27"


def test_every_derived_id_passes_artifact_id_validation() -> None:
    """Whatever the recipes mint must satisfy the egm-contracts pattern."""
    for ident in (
        derive_run_id("v1_5", today=_DATE),
        derive_model_id("v1_5", today=_DATE),
        derive_predictions_bank_id(labeled=True, run_name="v1_5", today=_DATE),
        derive_predictions_bank_id(labeled=False, run_name="v1_5", today=_DATE),
        derive_run_id(None, today=_DATE),
    ):
        assert validate_artifact_id(ident) == ident


def test_validate_artifact_id_rejects_malformed_string() -> None:
    with pytest.raises(ValueError, match="not a valid stable artifact id"):
        validate_artifact_id("Not An Id!")


def test_validate_artifact_id_rejects_missing_date_segment() -> None:
    with pytest.raises(ValueError, match="not a valid stable artifact id"):
        validate_artifact_id("run_v1_5")
