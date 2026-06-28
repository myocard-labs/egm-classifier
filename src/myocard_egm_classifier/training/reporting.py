"""Run reporting — thin orchestrator on top of egm-data's typed writers.

As of egm-data v0.3.1, every translation step from trainer-state to
typed records lives in ``myocard-egm-data.records``:

- :func:`make_epoch_record` (re-exported here for trainer convenience)
  builds one :class:`EpochRecord` from a flat ``val_metrics`` dict,
  coercing the reliability list (which the metrics layer emits as
  plain dicts) into typed :class:`ReliabilityBin` instances.
- :func:`build_training_run_record` stamps ``schema_version`` +
  ``created_utc`` and computes the :class:`BestEpoch`.
- :func:`write_training_metrics` and :func:`write_training_run_record`
  handle the two end-of-run files.

This module owns only :func:`write_run` — the one-shot orchestration
that writes both ``metrics.csv`` and ``run.json`` into the same output
directory. Everything else is a pass-through re-export.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from myocard_egm_data.records import (
    BestEpoch,
    EpochRecord,
    HeldOutTest,
    ReliabilityBin,
    TrainingRunRecord,
    build_training_run_record,
    make_epoch_record,
    write_training_metrics,
    write_training_run_record,
)

__all__ = [
    "BestEpoch",
    "EpochRecord",
    "HeldOutTest",
    "ReliabilityBin",
    "TrainingRunRecord",
    "make_epoch_record",
    "write_run",
]


def write_run(
    out_dir: Path | str,
    *,
    config: Mapping[str, Any],
    run_meta: Mapping[str, Any],
    epoch_records: Sequence[EpochRecord],
    select_metric: str,
    test_loss: float | None = None,
    test_metrics: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    trained_on_bank_id: str | None = None,
    produced_model_id: str | None = None,
) -> tuple[Path, Path]:
    """Write both ``metrics.csv`` and ``run.json`` into ``out_dir``.

    Returns ``(csv_path, json_path)``. Both writers are owned by
    ``myocard-egm-data``; this orchestrator builds the
    :class:`TrainingRunRecord` via egm-data's
    ``build_training_run_record`` (which stamps schema_version +
    created_utc + computes the BestEpoch) and then hands the typed
    record straight to the typed writers.
    """
    out_dir = Path(out_dir)
    # Strip non-scalar entries (e.g. the ``confusion`` sub-dict that
    # :func:`binary_metrics` returns) from test_metrics before handing
    # it to build_training_run_record. The contracts schema for
    # ``HeldOutTest.metrics`` only permits scalar (number/integer/null)
    # values, while ``EpochRecord.val_metrics`` is documented as
    # accepting a nested ``confusion`` block — an asymmetry to fix
    # upstream in egm-contracts. Until that's aligned, the per-epoch
    # ``confusion`` counts are still preserved via ``val_metrics``;
    # the test block only carries the scalar summary.
    if test_metrics is not None:
        test_metrics = {k: v for k, v in test_metrics.items() if not isinstance(v, dict)}
    record = build_training_run_record(
        config=config,
        run_meta=run_meta,
        epoch_records=epoch_records,
        select_metric=select_metric,
        test_loss=test_loss,
        test_metrics=test_metrics,
        run_id=run_id,
        trained_on_bank_id=trained_on_bank_id,
        produced_model_id=produced_model_id,
    )
    csv_path = write_training_metrics(out_dir / "metrics.csv", epoch_records)
    json_path = write_training_run_record(out_dir / "run.json", record)
    return csv_path, json_path
