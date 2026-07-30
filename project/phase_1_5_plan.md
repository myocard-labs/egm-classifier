# egm-classifier — Phase 1.5 implementation plan

**Repo:** egm-classifier · **Phase:** 1.5
**Phase design doc:** `intracardiac-platform/phases/phase_1_5/design.md`
**Status:** planning · **Progress:** 0/18 steps done
**Repo estimate:** **23 points · ~26–53 h active** (cold-start estimate by analogy — the
`estimation_ledger.csv` is empty, so per-task-type rates don't exist yet and the ranges are
deliberately wide. Per-issue breakdown in [Effort tracking](#effort-tracking).)

---

## Scope — what this plan covers

Core §3 issues **CLF5 · CLF2 · CLF1 · CLF3** and the §4 backlog items this repo owns
(**B2 · B14 · B15 · B18** producer-half **· B21**).

| Phase item | What it needs from this repo | Steps |
|---|---|---|
| **CLF5** *(Wave 1, migration)* | Re-pin egm-contracts v0.6.0 + egm-data v0.5.x; adopt `training_run_record` 1.2 + the v0.6.0 `ClassifierBank` shape at write time, **current behavior** | S1–S3 |
| **CLF2** *(← CLF5)* | Emit per-split **train** metrics each epoch; best-epoch (not last-epoch) held-out test metrics; per-trace split + prediction on the predictions bank | S4–S6 |
| **CLF1** | Best-epoch **selection panel** — min val-loss (new default) · Brier · MCC · AUROC baseline (**not ECE**) — plus SWA/EMA weight averaging, and ECE on adaptive bins | S7–S9 |
| **CLF6** *(provisional id)* | **Multi-seed training + seed-variance aggregation** — so a §8.4/§8.5 gap inside seed noise isn't read as a result | S9b–S9c |
| **CLF3** | Comparator panel — pure MobileNetV2-1D (config-only) + Res-CNN-LSTM (`res_block_1d` + `lstm_1d` blocks) | S10–S12 |
| **B2** | `zero2one` normalization in train + eval (today: export-only) | S13 |
| **B14** | Relative artifact paths in the run record | rides S2 |
| **B15** | Drop `hostname` from the run record | rides S2 |
| **B18** *(producer half)* | Stop down-casting `HeldOutTest.metrics`; take test metrics at the **best** epoch | rides S2, S6 |
| **B21** | Progress bar for `egm-class-eval` | S14 |
| — | Docs + phase exit | S15 |

**Not this repo, though the roadmap used to imply otherwise:** the activation-anchoring A/B is
**SEP10 + study §8.9** — a producer flag plus two training banks, **no egm-classifier change**; and
train-time noise-mixing augmentation is **XR1, deferred to FB-9**. Both drop out of `roadmap.md`
at S15.

## Design notes

Four local decisions the steps depend on. The fifth item is an escalation, not a decision.

- **A clean train-split metrics pass needs its own loader (S4).** `build_dataloaders` builds the
  train loader with `augment=True` and `shuffle=True`; computing train metrics off it would measure
  the *augmented* distribution and interleave with the augmentation RNG stream. So `LoaderBundle`
  gains a fourth loader — `train_eval`: the **same** `split.train` indices through the **eval**
  `TraceTransform` (no augmentation), `shuffle=False`. Cost is one extra forward pass over the
  largest split per epoch (~8× the val pass), so it sits behind a `train.eval_train_split` flag,
  default **on** — seeing the divergence is the point of T5.
- **The selection panel writes one checkpoint per criterion (S8).** Study §8.4 compares the
  criteria's *test-time models*, not just which epoch each picked, so recording the argmax epoch
  isn't enough. `train.select_metrics: [val_loss, brier, mcc, auroc]` writes `best_<criterion>.pt`
  alongside the primary `best.pt` (a copy of the first entry, so every existing `best.pt` consumer —
  eval, export, the tests — keeps working unchanged). Each criterion carries a **direction**
  (minimize for `val_loss`/`brier`, maximize for the rest); `select_metric` today is maximize-only.
  **`ece` is deliberately NOT selectable** (CL-073): it is binning-dependent and biased, so
  selecting on it optimises an artifact of the binning. Selection uses a **proper scoring rule**
  (BCE / Brier); ECE + reliability stay as **diagnostics**. `select_metrics` rejects `ece` with a
  message saying so, rather than silently accepting a bad criterion.
- **Eval re-derives the split rather than being told it (S3).** The folded-in predictions work
  (design §4's *"CLF4b"* — a sub-item of the **retired** CLF4, not of CLF5; see the decisions log)
  wants a per-trace
  `split` column on the predictions bank, but `egm-class-eval` only sees a bank + a checkpoint. Rather than
  add a sidecar hop, **train stamps the split parameters** (`split_fractions`, `split_seed`,
  `split_strategy`) into the checkpoint's `training_provenance`, and eval re-derives the identical
  patient-aware split **only when** the eval bank's id equals `trained_on_bank_id`. Different bank ⇒
  the `split` column is written null, which is the honest answer. This keeps the existing
  "checkpoint is self-describing" property that `training_provenance` already established.
- **Res-CNN-LSTM is two new block types, not a new model paradigm (S11–S12).** `models/registry.py`
  already dispatches on a `type` string and `model.blocks` is already plumbed through the train YAML,
  so `res_block_1d` + `lstm_1d` register alongside `mv2_1d` and the assembler doesn't change. The
  pure-CNN arm is the existing block list minus the `mobilevit_1d` entries — a config file, no code.
  The one real risk is **ONNX export of the LSTM arm**; S12 carries the parity check as its verify.
- **`train_metrics` is optional-in-schema / required-on-write (P1, resolved 2026-07-28).** This is
  the same convention the banks already use for ids, and it's what keeps the wave boundary clean:
  **CLF5** writes records that validate *without* `train_metrics`, and **CLF2** adds the emit and
  makes it required-on-write. So S2 does not need to fabricate an empty bundle, and S4–S5 stay in
  Wave 2 where they belong.

## Steps

☐ todo · 🔨 wip · ✅ done. S1–S3 are gated on the Wave-1 tags; S4–S14 are Wave 2. Within Wave 2,
the three threads (CLF2 → CLF1, CLF3, backlog) are independent and can be reordered freely.

### CLF5 — Wave-1 migration *(← egm-contracts v0.6.0 · egm-data v0.5.x)*

#### S1 — Re-pin siblings, suite green ☐ (0.5–1.5 h)
- **Change:** `pyproject.toml` pins → egm-contracts v0.6.0, egm-data v0.5.x (egm-signal unchanged at
  v0.2.0); fix any import/type fallout; update the version table in `project/architecture.md`.
  **Also drop `label_fn`** (CL-038): under `synthetic_bank` v2.0 the converter defaults to the bank's
  own int label + `label_names`, and `label_fn` demotes to an explicit re-labeling override. Only
  `tests/test_datasets.py` L38 passes one — switch it to take the bank's label; the `loaders.py`
  docstring (L12–16) describing label semantics "at conversion time" needs the same touch.
- **Verify:** `pytest` full suite green, `mypy` clean, package imports in a fresh venv; the converted
  fixture's labels match the bank's own without a `label_fn`.
- **Depends on:** egm-contracts v0.6.0 **and** egm-data v0.5.x tagged.

#### S2 — `training_run_record` 1.2 write path (B14 · B15 · B18) ☐ (1.5–3 h)
- **Change:** `cli/train_cmd.py:_run_meta` stops writing `host` (B15) and records `bank_path` +
  the output dir **repo-relative** (B14); `training/reporting.py:write_run` drops the
  `test_metrics` dict-stripping workaround now that `HeldOutTest.metrics` mirrors the val bundle
  (B18) — the long comment explaining the asymmetry goes with it.
- **Verify:** `test_reporting.py` round-trip asserts no `host` key, relative paths, and a nested
  `confusion` surviving into `HeldOutTest.metrics`; a real `run.json` **written without
  `train_metrics`** validates against 1.2 (the optional-in-schema half of P1 — the guarantee that
  keeps CLF5 in Wave 1).
- **Depends on:** S1.

#### S3 — P3 predictions shape + `ArtifactId` role validation ☐ (1.5–3 h)
- **Change:** `eval/predictions.py` populates the new per-trace `split` + `prediction` columns
  (split re-derived per the design note; null when the eval bank isn't the training bank);
  `cli/train_cmd.py` stamps the split parameters into `training_provenance`; `ids.py` validates
  role prefixes against egm-contracts' `roles.json` instead of its local notion.
- **Verify:** `test_eval_cmd.py` — train→eval on one bank gives a populated `split` column matching
  the trainer's split; eval on a *different* bank gives nulls; a bogus role prefix now fails
  `validate_artifact_id`.
- **Depends on:** S1.

### CLF2 — train-split metrics *(← CLF5)*

#### S4 — `train_eval` loader ☐ (1–2 h)
- **Change:** `data/datasets/loaders.py` — `LoaderBundle` gains `train_eval`; built over `split.train`
  with `eval_tf`, `shuffle=False`. New `eval_train_split: bool` kwarg on `build_dataloaders`.
- **Verify:** `test_datasets.py` — `train_eval` yields the same trace count as `train`, in bank
  order, and is byte-identical across two passes (proving no augmentation).
- **Depends on:** CLF5. *(This step touches no schema, so it is technically buildable before the
  Wave-1 tags land — but CLF2 sits in Wave 2 by design, and the point of the de-risking rule is
  that nothing runs ahead of the migration. It waits.)*

#### S5 — Per-epoch train metrics into the record ☐ (1.5–3 h)
- **Change:** `training/train.py` evaluates `loaders.train_eval` each epoch and passes the bundle to
  `make_epoch_record(train_metrics=...)`; per-epoch console line shows train alongside val;
  `metrics.csv` gains the train columns; `train.eval_train_split` plumbed through `_train_config.py`.
- **Verify:** `test_train.py` — a 2-epoch run produces `EpochRecord.train_metrics` with the full
  scalar keyset and a `metrics.csv` header carrying the train columns.
- **Depends on:** S4, S2. **Two upstream prerequisites, both tracked:** `make_epoch_record` gains
  `train_metrics=None` (confirmed shipping in DAT3 S9 — CL-022 → CL-036); and
  `training_metrics.schema.json` must gain the six `train_*` CSV columns, since it is
  `additionalProperties: false` and would otherwise **reject** the new `metrics.csv` (CL-037 item 1,
  with egm-contracts). If the CSV half misses v0.6.0, the `metrics.csv` change splits out of S5 and
  the run-record half ships alone.

#### S6 — Best-epoch held-out test metrics ☐ (0.5–1.5 h)
- **Change:** `cli/train_cmd.py` reloads `best.pt` before the held-out test `evaluate` instead of
  scoring the end-of-training model (the current behavior is flagged in a comment as knowingly
  wrong for a publication number).
- **Verify:** `test_train.py` asserts the recorded test metrics match a manual evaluation of the
  best checkpoint, not the last-epoch weights.
- **Depends on:** S5.
- **Effort attribution:** rolled into **CLF2**, per §3's CLF2 note ("best-model — not last-epoch —
  test metrics fold in here"). §4 assigns the same work to CLF1 — see the decisions log; if the
  project-lead prefers CLF1, move ~1 h between the two rows at cleanup.

### CLF1 — best-epoch selection panel *(study §8.4)*

#### S7 — Brier + MCC + val-loss as selectable criteria ☐ (1–2 h)
- **Change:** `metrics.py:binary_metrics` gains `brier` and `mcc`; a criterion table maps each name
  to its **direction**, with `val_loss` selectable as a pseudo-metric and **`ece` explicitly
  excluded** (CL-073 — see the design note). MCC uses
  `torchmetrics.classification.BinaryMatthewsCorrCoef` (verified present at the pinned
  torchmetrics 1.9); **Brier has no torchmetrics binary class** — it's computed directly as
  `mean((sigmoid(logits) − labels)²)`, which is the whole definition and keeps it a proper scoring
  rule with no extra dependency.
- **Verify:** `test_metrics.py` — hand-calculated Brier/MCC on the existing perfect/worst/
  known-confusion fixtures; direction table asserts min-vs-max per criterion; `select_metrics:
  [ece]` raises at config-load with the "not a proper scoring rule" message.
- **Depends on:** none.

#### S7b — ECE on adaptive (equal-mass) bins ☐ (1.5–3 h)
- **Change:** ECE + the reliability table move to **equal-mass / quantile bins (~10–15)** as the
  primary estimator, with **15 equal-width** bins retained as a secondary reported alongside
  (CL-073). Today both come from `BinaryCalibrationError(n_bins, norm="l1")` and
  `_reliability_bins`, which are equal-width — on a saturated IAFDB predictive distribution nearly
  every sample lands in one bin, so equal-width ECE is dominated by a single bin and the reliability
  diagram is mostly empty. Equal-mass binning is what makes the T5/§8.5 calibration comparison
  readable. Both numbers are emitted (`ece`, `ece_equal_width`) so nothing that read the old key
  loses its series.
- **Verify:** `test_metrics.py` — on a deliberately skewed predictive distribution, equal-mass bins
  are all non-empty and equal-width are not; a uniform distribution makes the two agree to within
  tolerance (the sanity check that the estimator wasn't broken in the process).
- **Depends on:** S7. *Feeds the §8.5 saturation metrics and STU3's reliability rendering.*

#### S8 — Multi-criterion checkpointing ☐ (2.5–5 h)
- **Change:** `TrainConfig.select_metrics: tuple[str, ...]` (default `("val_loss",)` — the new
  default per L2, replacing max-AUROC); `train()` tracks each criterion's best independently and
  writes `best_<criterion>.pt`; `best.pt` mirrors the first entry; the chosen epoch per criterion
  lands in the run record's `run` block. `select_metric` (singular) stays accepted as a deprecated
  alias so existing configs don't break.
- **Verify:** `test_train.py` — a run with three criteria writes three checkpoints; a synthetic
  metric history where the criteria disagree selects three *different* epochs.
- **Depends on:** S7. (S5 not required, but running it after S5 means the panel sees train metrics too.)

#### S9 — SWA / EMA weight averaging ☐ (2–4 h)
- **Change:** optional `train.weight_averaging: {none|swa|ema}` — `torch.optim.swa_utils`
  `AveragedModel` (+ `SWALR` for the SWA arm, EMA via `get_ema_multi_avg_fn`), averaged from a
  configurable start epoch, with a BN-update pass before evaluation; the averaged model is scored
  and checkpointed as `best_swa.pt` / `best_ema.pt`.
- **Verify:** `test_train.py` — an averaged checkpoint is written, loads, and produces finite
  metrics; the averaged weights differ from the final-epoch weights.
- **Depends on:** S8.

### CLF6 *(provisional id — see decisions log)* — multi-seed training + seed-variance aggregation

> _Serves **§8.4 · §8.5 · §8.6 · §8.9** — an architecture or best-epoch-criterion gap that sits
> inside seed noise is not a result. Added from CL-073 after the §8 study audit; **no schema
> change** — each seed is an ordinary `training_run_record`._

#### S9b — Seed loop with per-seed artifact ids ☐ (2–4 h)
- **Change:** `train.seeds: [int, ...]` (single `seed:` stays as the one-element form); the CLI loops
  training over the list, one full run per seed. The subtlety is **artifact identity**: `run_id` /
  `produced_model_id` derive from `output.run_name`, so N seeds would collide on one id. Each seed
  gets a suffixed descriptor (`run_v1_5_x_seed0`, …) and its own `checkpoint_dir` subdirectory, so
  every seed is an independently addressable, manifest-curatable artifact. Same seed list across
  arms is what makes §8.4/§8.5's **paired** comparison possible, so the list is recorded in each run
  record.
- **Verify:** `test_train.py` — a 2-seed run writes two run records with distinct `run_id`s, two
  checkpoint dirs, and reproduces identical metrics when re-run with the same list.
- **Depends on:** none (independent of the criterion panel; do after S8 to avoid churn on the same
  file).

#### S9c — Seed-variance aggregation ☐ (1.5–3 h)
- **Change:** after the loop, emit a compact per-metric aggregate across seeds — n, mean, sd, min,
  max — to stdout and as a small `seeds_summary.json` beside the per-seed dirs. Deliberately **not**
  a new schema: it is a derived convenience over the N run records, which stay the source of truth.
- **Verify:** `test_train.py` — the summary's per-metric mean/sd match a hand computation over the
  per-seed run records.
- **Depends on:** S9b.
- **Scope boundary:** I emit per-arm spread. The **paired cross-arm statistics** (does criterion A
  beat B once seed noise is accounted for?) compare *different training runs* and belong to the
  study / STU3, not to a single `egm-class-train` invocation — flagged in my CL reply.

### CLF3 — conventional comparator panel *(study §8.5)*

#### S10 — Pure MobileNetV2-1D arm (config-only) ☐ (0.5–1 h)
- **Change:** `examples/v1_5_cnn_only.yaml` — the v1 block list with the `mobilevit_1d` entries
  swapped for `mv2_1d`; a short note in `docs/usage.md` on what the arm isolates (attention).
- **Verify:** `egm-class-train` builds the model from the config and reports a parameter count;
  `test_cli_config.py` asserts the config loads and yields a block list containing no
  `mobilevit_1d`.
- **Depends on:** none.

#### S11 — `res_block_1d` block type ☐ (1.5–3 h)
- **Change:** `models/blocks.py` — a residual conv block (Conv-BN-act ×2 + identity/projection
  shortcut, DropPath-aware, width-scaled via `make_divisible`); registered in `models/registry.py`
  with its frozen param dataclass.
- **Verify:** `test_model.py` — forward-pass shape + channel wiring, identity vs projection
  shortcut both exercised, `parse_block_params` rejects unknown params.
- **Depends on:** none.

#### S12 — `lstm_1d` block type + Res-CNN-LSTM config ☐ (2.5–5 h)
- **Change:** `models/blocks.py` — an LSTM block over the `[B, C, T]` sequence (transpose →
  `nn.LSTM` → transpose back, optional bidirectional), registered; `examples/v1_5_res_cnn_lstm.yaml`
  assembles the Chen-2022-style stack.
- **Verify:** `test_model.py` forward-pass shapes; **`test_export_cmd.py` ONNX parity** on the
  LSTM arm (elementwise diff < 1e-4) — this is the step's real risk, so it's the gating check. If
  the legacy exporter can't handle the LSTM at opset 18, the fallback is to scope CLF3's ONNX
  support out and note it in `docs/onnx_deployment.md`.
- **Depends on:** S11.

### Backlog

#### S13 — B2 `zero2one` normalization in train + eval ☐ (2–4 h)
- **Change:** `TraceTransform.znorm: bool` → `normalize: Literal["zscore","zero2one","none"]`
  (accepting the old bool for back-compat); `data.normalization_scheme` in the train + eval YAML;
  the scheme stamped into the checkpoint `model_meta` with a fallback to `zscore` for older
  checkpoints; export reads it from the checkpoint rather than duplicating it in its own YAML.
- **Verify:** `test_augmentation.py` — each scheme's output range/moments on a known trace;
  `test_cli_config.py` — the scheme round-trips YAML → `model_meta` → rebuilt transform; a legacy
  checkpoint without the key still loads as `zscore`.
- **Depends on:** none. *(The empirical zero2one-vs-zscore comparison is a run, not code — it
  belongs to §8, not this plan.)*

#### S14 — B21 eval progress bar ☐ (0.5–1.5 h)
- **Change:** wrap the `egm-class-eval` inference loop in `tqdm` with a `--no-progress` escape
  hatch, matching the producer pattern (tqdm is already a direct dependency, used by the trainer).
- **Verify:** `test_eval_cmd.py` — `--no-progress` produces no bar on a captured stderr; the
  default path still writes an identical predictions bank.
- **Depends on:** none.

### S15 — Docs + phase exit ☐ (1–2 h)
- **Change:** `roadmap.md` — drop the shipped items and the two that turned out not to be ours
  (anchoring A/B → SEP10/§8.9; train-time noise → FB-9), refresh "Schema bumps to coordinate";
  `CHANGELOG.md` — the shipped summary; `project/architecture.md` — fix the stale
  "TraceTransform / patient_aware_split live in egm-data" sections (untrue since Refactor Step 8)
  and the dead `intracardiac-platform/project/refactor_checklist.md` links (also stale in
  `project/code_placement_audit.md`); `docs/theory.md` §5 — the new Brier/MCC math + a §3 note on
  weight averaging; `docs/usage.md` — the selection panel, the comparator configs, and the
  normalization scheme.
  **Also `docs/theory.md` §1.4 (Length conditioning) — record the `T ≡ 0 (mod 64)` rule**, with the
  per-stage derivation and why it binds (stage-5 `patch_size=2` at `T/32`). It is currently written
  down only in the phase design doc §8.1 and in *this plan*, which is deleted at cleanup — so
  without this step a future maintainer picking an `input_length` has no in-repo statement of the
  constraint, and the failure mode is either a runtime error or silent zero-padding that reintroduces
  the positional shortcut. `project/architecture.md` gets a one-line cross-reference.
- **Verify:** the full pre-PR run in `intracardiac-platform/project/pr_checklist.md` passes.
- **Depends on:** all prior steps.

## Effort tracking

> **Actuals are NOT tracked for Phase 1.5** (Daniel, 2026-07-29). Marker-based session logging was
> missed across the phase — this repo's log held a single unclosed `start` from 2026-07-28 and other
> repo chats hit the same (egm-data, egm-signal, both flagged `reconstructed` in CL-032 / CL-036).
> Rather than back-fill guesses that would pollute the very corpus the method exists to build,
> 1.5 ships **estimates only**; Daniel + the project-lead chat will design the tracking methodology
> and apply it from the next phase. **So: no session log here, and the `Actual` / `Elapsed` columns
> stay empty — deliberately, not pending.** The one consequence to carry forward: `estimate ×
> actual` calibration still has **zero** data points after this phase, so Phase-2 estimates remain
> cold-start-by-analogy too.
>
> Method, for when it resumes:
> `intracardiac-platform/project/investigations/estimate_vs_actual_tracking.md`.

### Estimates by issue

Cold start: the ledger has no rows, so these are **analogy estimates with wide ranges**, not
`points × rate` arithmetic. Anchors used: `noise_bank bank_id` = S, CLF2 = M, SEP12 = L.

| Issue | Task-type | Cx | Estimate |
|---|---|---|---|
| CLF5 | schema-migration | M (3) | 3.5–7.5 h |
| CLF2 | pipeline | M (3) | 3–6.5 h |
| CLF1 | pipeline | L (5) | 7–14 h |
| CLF6 *(provisional id)* | pipeline | M (3) | 3.5–7 h |
| CLF3 | pipeline | M (3) | 4.5–9 h |
| B2 | pipeline | S (2) | 2–4 h |
| B14 | pipeline | XS (1) | *rides S2* |
| B15 | pipeline | XS (1) | *rides S2* |
| B21 | pipeline | XS (1) | 0.5–1.5 h |
| docs / phase exit | docs | XS (1) | 1–2 h |
| **Repo total** | | **23 pts** | **~26–53 h** |

**Changed 2026-07-30 by the §8 study audit (CL-073):** **+3 pts / +6–11 h.** CLF1 absorbed the
new **S7b** adaptive-ECE step (5 pts unchanged — the panel's shape didn't grow, but its hours did,
5.5–11 → 7–14); **CLF6** is net-new (M, 3 pts). CLF1 stays L rather than going XL because S7b is
mechanical (a binning change with a clear test), not novel.

Widest ranges, in order: **CLF3** (ONNX export of an LSTM is unproven here), **CLF1** (SWA/EMA is
net-new to this codebase), **CLF5** (depends on how much the v0.6.0 restructure ripples past the
schemas this repo actually writes).

## Notes / decisions log

- **2026-07-28** — Plan created at flow-down. Two items escalated to the project-lead: the Wave-1
  migration slice had no §3 id (tracked as plan-local `W1`), and P1 specified
  `EpochRecord.train_metrics` as *required* while §7 deferred the emit to Wave 2.
- **2026-07-28** — **Both resolved (project-lead).** (1) The Wave-1 migration slice became a real
  §3 issue (T5): "Migrate to egm-contracts v0.6.0 — re-pin; adopt `training_run_record` 1.2 + the
  v0.6.0 `ClassifierBank` shape at write time, current behavior." A migration issue alongside
  SEP12 / STU6, with **CLF2 depending on it**; deliberately *not* folded into CLF2, because the
  separation is what makes the wave boundary hold. 3 pts / 3.5–7.5 h. (2) P1 is now
  **optional-in-schema / required-on-write** — the migration issue writes records that validate
  without `train_metrics`; CLF2 adds the emit and makes it required-on-write.
- **2026-07-28** — **Migration issue minted as CLF4, then renamed to `CLF5`** after this chat
  flagged an id collision: `CLF4` was already spent. The §3 revision history retired the original
  CLF4 on 2026-07-24 (it became **study §8.6**, *"Classifier training-hyperparameter sweep
  (CLF4)"*), and *CLF4a* / *CLF4b* in §4 are its sub-items, redistributed to CLF1 and CLF2/STU3.
  Those back-references are **correct and stay as CLF4** — the fix was to stop recycling the id,
  not to rewrite history. Rationale that carried it: issue ids key the **cross-phase**
  `estimation_ledger.csv`, so a recycled id silently corrupts the reference-class rates; ids there
  are **append-only** (rule now in the platform template + ledger). §3 / §6 / §7 / the CON3 note
  all read CLF5; this plan renamed to match. No estimate or step-ordering change.
- **2026-07-28** — §3-vs-§4 attribution for best-model test metrics raised as **CL-023**.
  **Closed 2026-07-29 (CL-024 §6): stays under CLF2**; §4 was aligned to §3. S6 unchanged.
- **2026-07-28** — **Cross-chat items raised** in `intracardiac-platform/phases/phase_1_5/coordination_log.md`
  (the per-phase message bus; grep it for `[OPEN]` + `TO **egm-classifier**` on resume):
  **CL-020** → project-lead — the predictions bank inherits its source bank's `trace_metadata`
  verbatim, so the `sim_id` / `simulation_id` join key is reachable from a prediction row under
  whichever name the source used; argues for the rename (option 1). **CL-021** → project-lead — the
  model requires `T ≡ 0 (mod 64)` samples, which leaves study §8.1 exactly one valid value (192 ms)
  in its 150–250 ms window; affects **S13**'s `input_length` and, if ignored, silently re-introduces
  the positional regularity T1 is removing. **CL-022** → egm-data — DAT3 must add a `train_metrics`
  parameter to `make_epoch_record`, optional-with-`None`-default, or **S5** has nothing to call.
  **CL-023** → project-lead — the §3/§4 attribution above.
- **2026-07-29** — **All four adjudicated; nothing outstanding from them.** **CL-020** → CL-024 §3:
  **option 1**, producer renames `sim_id` → `simulation_id` at SEP12; this repo is unaffected (reads
  `patient_id`) and the predictions leg inherits the fix for free. **CL-021** → CL-024 §4:
  `T ≡ 0 (mod 64 samples)` is now recorded in design **§8.1 + the `T` row**; §8.1 optimises on the
  64-grid and at 1 kHz the 150–250 ms window resolves to **192 ms**. The architecture is *not* relaxed
  and padding is *not* accepted — so when §8.1 lands, `input_length` becomes 192 and no
  `TraceTransform` padding occurs. **CL-022** → DAT3 S9 (CL-036): the `train_metrics=None` call S5
  needs will exist; a `"reliability"` key inside it is dropped, not split (FB-10). **CL-023** →
  CL-024 §6, above.
- **2026-07-29** — **CL-038 (open, addressed here):** under `synthetic_bank` v2.0 the egm-data
  converter defaults to the bank's own int label + `label_names`, demoting `label_fn` to an explicit
  re-labeling override. Non-breaking, but on re-pin we should omit it. Only `tests/test_datasets.py`
  L38 passes one; folded into **S1**.
- **2026-07-29** — **CL-024 §1 (CI/ruff) applied:** `.github/workflows/ci.yml` pins `ruff==0.15.17`,
  matching this repo's existing pre-commit rev, so pre-PR lint predicts CI again. Per the ruling the
  0.16 first-aid reformat was **not** run — the pin makes it moot and `docs/onnx_deployment.md` stays
  as written. No pre-commit bump needed here (already v0.15.17; only egm-features + python-template
  were on v0.6.9).
- **2026-07-29** — **Log write discipline changed (CL-025):** post by appending at EOF with
  `cat >> ... <<'EOF'`, never a whole-file rewrite; **only the project-lead edits or resolves existing
  entries**. Don't self-resolve — a resolution comes back as a new entry.
- **2026-07-30** — **§8 study audit lands two changes (CL-073).** (1) **ECE is no longer a selectable
  criterion** — binning-dependent and biased, so selection uses a proper scoring rule (BCE/Brier) and
  ECE stays a diagnostic; this repo's plan had explicitly listed `ece` as a minimize-direction
  criterion, so it was a real defect, now fixed in the design note + S7. (2) ECE moves to
  **equal-mass/adaptive bins** (new **S7b**) — which matters more than it sounds: on the saturated
  IAFDB predictive distribution, equal-width binning puts nearly every sample in one bin, so the
  headline calibration number and the reliability diagram were both going to be near-useless for the
  §8.5 saturation comparison. (3) **Multi-seed training + spread aggregation** is net-new work
  (S9b–S9c). §8.9's positional probe needs **no** change here — it uses existing inference.
  **Cost: +3 pts, +6–11 h** (23 pts, 26–53 h).
- **2026-07-30** — **`CLF6` is a provisional, self-minted id — needs the project-lead's blessing.**
  Multi-seed serves §8.4/§8.5/§8.6/§8.9, so it belongs to no single existing issue, and CL-073 folded
  it in without minting one. Same situation as the Wave-1 slice that became CLF5. Raised in my CL
  reply; **if the project-lead assigns a different number, rename here** — ids are append-only and
  key the ledger, so it must not be invented unilaterally and left.
- **2026-07-28** — `roadmap.md`'s Phase 1.5 cluster predates the design doc: the anchoring
  investigation is now SEP10 + study §8.9 (**no egm-classifier change**) and train-time noise
  augmentation is XR1 → FB-9. Corrected at S15 rather than now, so the roadmap and CHANGELOG move
  together at phase exit.
