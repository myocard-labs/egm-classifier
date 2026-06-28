"""Stable cross-artifact id helpers for egm-classifier outputs.

egm-classifier is the *consumer* end of the cross-artifact-linkage graph: a
training run produces a run record (``run.json``) and a model
(``model_metadata.json``), and an eval produces a predictions bank. Each
carries a stable egm-contracts ``ArtifactId``:

- ``run_<run_name>_<date>``   — the training run record's ``run_id``.
- ``model_<run_name>_<date>`` — the exported model's ``model_id`` (the run
  record's ``produced_model_id`` points at the same string).
- ``lpred_<run_name>_<date>`` / ``upred_<run_name>_<date>`` — the eval
  predictions bank (labeled / unlabeled source).

``run_name`` is a config-provided descriptor (``output.run_name``); it falls
back to ``egm_classifier`` when omitted. The run/model/predictions ids share
the run_name so the provenance chain reads coherently. The id *pattern* is
single-sourced in egm-contracts' ``common.ArtifactId``; this module only
composes candidate strings and validates them.
"""

from __future__ import annotations

import datetime as _dt
import re as _re

from myocard_egm_contracts import common as _contracts_common
from pydantic import ValidationError

DEFAULT_RUN_NAME = "egm_classifier"
"""Fallback descriptor when ``output.run_name`` is not set."""


def _today_utc() -> str:
    """Today's date (UTC) as ``YYYY-MM-DD`` for the id's date segment."""
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def _sanitize(text: str) -> str:
    """Collapse arbitrary text to the ``[a-z0-9_]`` id-descriptor charset."""
    cleaned = _re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned or DEFAULT_RUN_NAME


def validate_artifact_id(value: str) -> str:
    """Validate ``value`` against the egm-contracts ArtifactId pattern.

    Returns it unchanged on success; raises ``ValueError`` with a
    producer-friendly message on a malformed id. Mirrors egm-data's
    id-validation idiom so an explicit override fails fast at the boundary.
    """
    try:
        _contracts_common.ArtifactId(value)
    except ValidationError as exc:
        raise ValueError(
            f"id {value!r} is not a valid stable artifact id "
            "(egm-contracts ArtifactId pattern, e.g. "
            "'run_v1_5_courtemanche_2026-06-27')."
        ) from exc
    return value


def _descriptor(run_name: str | None) -> str:
    return _sanitize(run_name) if run_name else DEFAULT_RUN_NAME


def derive_run_id(run_name: str | None, *, today: str | None = None) -> str:
    """Default id for a training run record: ``run_<run_name>_<date>``."""
    return f"run_{_descriptor(run_name)}_{today or _today_utc()}"


def derive_model_id(run_name: str | None, *, today: str | None = None) -> str:
    """Default id for an exported model: ``model_<run_name>_<date>``."""
    return f"model_{_descriptor(run_name)}_{today or _today_utc()}"


def derive_predictions_bank_id(
    *, labeled: bool, run_name: str | None, today: str | None = None
) -> str:
    """Default id for an eval predictions bank.

    ``lpred_`` (labeled source — a scored eval) or ``upred_`` (unlabeled
    source — a label-free diagnostic, e.g. the IAFDB pass).
    """
    role = "lpred" if labeled else "upred"
    return f"{role}_{_descriptor(run_name)}_{today or _today_utc()}"
