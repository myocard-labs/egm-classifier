# egm-classifier — roadmap

Future work only — shipped history lives in [`CHANGELOG.md`](../CHANGELOG.md). Internal
doc; public users read the README + `docs/usage.md`.

Work lands here as it's identified, sits in the **Backlog** until a phase-planning session
promotes it into a **Phase** cluster, then moves to the CHANGELOG once shipped. Phase
clusters mirror the science Project Phases in
`intracardiac-platform/project/project_plan.md`. Items scheduled into cross-cutting Phase
work carry a `→ tracked at intracardiac-platform Phase X` annotation; the rest are
component-internal.

## Phase 1.5 — sim-realism

Scheduled into the active phase. Item-level here; the **ordered steps, complexity scores, and
estimates live in [`phase_1_5_plan.md`](phase_1_5_plan.md)** (ephemeral — deleted at phase
cleanup, when the shipped work summarizes into [`CHANGELOG.md`](../CHANGELOG.md) and anything
unfinished drops back here). Issue ids are the phase design doc's:
`intracardiac-platform/phases/phase_1_5/design.md` §3 (core) and §4 (backlog).

### CLF5 — migrate to egm-contracts v0.6.0

Re-pin egm-contracts v0.6.0 + egm-data v0.5.x and adopt `training_run_record` 1.2 plus the
v0.6.0 `ClassifierBank` shape at write time, **with current behavior** — the Wave-1 migration,
deliberately separate from the feature work so a break from the refactor is caught before any
feature is added. Also drops `label_fn` at the converter boundary (the 2.0 bank carries its own
int label + `label_names`).

> → Design §3 CLF5 · plan S1–S3. Gates CLF2. Wave 1.

### CLF2 — emit per-split train metrics

Compute metrics on the train split each epoch through a non-augmented, sequential `train_eval`
loader and record them in `EpochRecord.train_metrics` + `metrics.csv`, so the train/val
divergence is visible rather than inferred. Folds in best-model (not last-epoch) held-out test
metrics, and the per-trace split + prediction columns on the predictions bank.

> → Design §3 CLF2 · plan S4–S6. Depends on CLF5 (+ CON3 / DAT3). Wave 2.

### CLF1 — best-epoch selection panel

Replace max-val-AUROC as the sole selection criterion with a panel — **min val-loss (new
default)**, Brier, MCC, AUROC retained as baseline — writing one checkpoint per criterion so
their test-time models can be compared, plus SWA/EMA weight averaging as the orthogonal
"don't pick one epoch" option. Rationale + sources: design §2 **L2**.

**ECE is explicitly not selectable** (binning-dependent and biased — select on a proper scoring
rule, keep ECE as a diagnostic), and ECE itself moves to **equal-mass / adaptive bins**, since
equal-width binning collapses on a saturated predictive distribution and would make the §8.5
calibration comparison unreadable.

> → Design §3 CLF1 · plan S7–S9 (+ S7b, from CL-073). Runs study §8.4. Wave 2.

### CLF6 (provisional id) — multi-seed training + seed-variance aggregation

Run each training over a list of seeds and report per-metric spread, so a difference between
architectures (§8.5) or between best-epoch criteria (§8.4) can be judged against seed noise
instead of read off a single run. Each seed is an ordinary `training_run_record` with its own
stable ids — no schema change; the same seed list across arms is what makes the comparison
paired. Cross-arm paired statistics belong to the study / STU3, not to a training run.

> → Added by the §8 study audit (CL-073); id provisional pending the project-lead ·
> plan S9b–S9c. Serves §8.4 / §8.5 / §8.6 / §8.9. Wave 2.

### CLF3 — conventional comparator panel

Two comparator architectures against the MobileViT: a **pure MobileNetV2-1D** (config-only —
the block list minus the `mobilevit_1d` entries, isolating the attention contribution) and a
**Res-CNN-LSTM** (`res_block_1d` + `lstm_1d` block types; the family precedented on
intracardiac EGM, Chen 2022). A shared over-call across families implicates the data, not the
model. Rationale + sources: design §2 **L3**.

> → Design §3 CLF3 · plan S10–S12. Runs study §8.5. Wave 2.

### `zero2one` normalization in train + eval (B2)

The export CLI already supports `zscore` and `zero2one`; train + eval are hardcoded to
`zscore` via the `TraceTransform`. Finish the job: add a `normalize` mode
(`'zscore' | 'zero2one' | 'none'`) to `TraceTransform` (now in `myocard_egm_classifier.data`,
currently a `znorm` bool), plumb `normalization_scheme` through the train + eval YAML, and stamp
the chosen scheme into the checkpoint `model_meta` (with a back-compat fallback). The empirical
`zero2one`-vs-`zscore` comparison is a run, not code — it belongs to §8, not the plan.

> → Design §4 B2 · plan S13.

### Run-record cleanups (B14 · B15 · B18)

Three edits riding the `training_run_record` 1.2 bump: record artifact paths **repo-relative**
rather than absolute (B14); stop writing `hostname` (B15); and stop down-casting
`HeldOutTest.metrics` once the schema mirrors the val bundle, which also means test metrics are
taken at the **best** epoch (B18).

> → Design §4 B14 / B15 / B18 · plan S2 (+ S6). Ride CLF5.

### Progress bar for `egm-class-eval` (B21)

The eval loop runs sequentially with no feedback, so a large IAFDB inference bank looks hung on
an underpowered machine. Wrap it in `tqdm` with a `--no-progress` escape hatch, matching the
producer pattern.

> → Design §4 B21 · plan S14.

### Moved out of this repo (were listed here pre-planning)

- **Activation-anchoring A/B** — turned out not to be ours. It is a producer-side opt-in flag
  (**SEP10** in synthetic-egm-pipeline) plus **study §8.9**; both arms are just different
  training banks, so egm-classifier needs no change. Design §3 SEP10.
- **Training-time additive-noise augmentation** — deferred out of the phase as **XR1** →
  [`feature_backlog.md`](../../intracardiac-platform/project/feature_backlog.md) **FB-9**
  (1.5 already large; low expected effect; both clean and noise runs already diverged).

## Phase 4 — multi-beat

### Variable-length / longer-`T` `EGMTraceDataset`

Multi-beat windows want longer (and possibly variable-length) traces; settle
fixed-length-multi-beat vs variable-length + a custom `collate_fn`. Pairs with the
`TraceTransform` time-shift fix (producer emits `L > T`; the dataset takes a random length-`T`
crop each `__getitem__` — see [`project/trace_transform_review.md`](trace_transform_review.md)).

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 4. Absorbed with the
> data layer from egm-data.

## Backlog (unscheduled — promoted into a phase at a planning session)

### `run.json` as the export-config source (downsample policy)

The provenance half shipped in v0.4.0 (export + eval auto-read the ids from the checkpoint's
`training_provenance`). What remains is the `fs_hz` / `bandpass_hz` downsample policy, still
YAML-duplicated between train and export: stamp a single training `fs_hz` into `run.json`
(downsampling everything to it, ≤ the min bank `freq_hz`), and add an optional `run_json:`
field to the export YAML that reads `fs_hz` + future preprocessing from there (inline values
warn-on-disagree). Pairs with egm-signal's `filters.decimation` (Phase 1.5). Was task #286.

### Splitting-strategy choices

The `patient_aware_split` Protocol ships `AnyPositive` + `BinnedDensity`; the trade-off space
(stratification target, bin-count default, small-patient-count K-fold fallback) isn't fully
mapped. Component-internal (absorbed with the data layer from egm-data).

### Streaming `EGMTraceDataset`

Today it loads the whole HDF5 signal array into RAM at construction; a chunked / mmap loader
may be needed as banks grow (Phase 1.5 pushes synthetic to 300–500 sims; Phase 7 multi-beat +
3D geometry is larger still).

### Migrate `torch.onnx.export` to the dynamo path

Export passes `dynamo=False` (legacy TorchScript exporter); PyTorch is moving to the dynamo
exporter (default in 2.5+). Switch to `dynamo=True`, replace `dynamic_axes` with
`dynamic_shapes`, re-run the parity smoke against a trained checkpoint, and bump the opset
default if dynamo emits anything newer. Component-internal; was task #288.

### Move `MobileViTBlock` divisibility check to `__init__`

`mobilevit_block.py` validates `T % patch_size == 0` inside `forward()`, producing ~23
`TracerWarning`s per export. Move it to `__init__` (add an `input_length` kwarg), delete the
forward-time check, and drop the `TracerWarning` suppression in `export/onnx_export.py`.
Component-internal; was task #289.

## Schema bumps to coordinate

**Inbound: the Phase-1.5 egm-contracts v0.6.0 + egm-data v0.5.x wave.** Two of its six schema
groups are written by this repo, so CLF5 adopts them and CLF2 builds on them:

- **`training_run_record` 1.1 → 1.2** (P1) — adds `EpochRecord.train_metrics` (**optional in
  schema, required on write** once CLF2 ships, which is what lets CLF5 migrate without it);
  aligns `HeldOutTest.metrics` to the val bundle (B18); drops `hostname` (B15); relative paths
  (B14). Needs egm-data's `make_epoch_record` to gain a `train_metrics=None` parameter —
  confirmed shipping in DAT3.
- **`classifier_predictions` + `ArtifactId`** (P3) — per-trace `split` + `prediction` columns on
  the egm-data `ClassifierBank`, and role-prefix validation on `ArtifactId` against
  egm-contracts' `roles.json` (B16), which `ids.py` adopts.

Also pending in the same wave, and **blocking the `metrics.csv` half of CLF2**:
`training_metrics.schema.json` is `additionalProperties: false`, so it must gain the six
`train_*` columns or it will reject the new CSV outright.

The in-flight field-level spec is the **Proposed changes** section of
`intracardiac-platform/project/cross_artifact_linkage_design.md` — the canonical §1–§3 there
reflect only what has already shipped.

Previously: the egm-contracts v0.5.0 cross-artifact-linkage wave (consumed in v0.4.0) —
`egm_class_model_metadata` → 1.2 (`model_id`) and `training_run_record` → 1.1 (`run_id` +
`produced_model_id` + `trained_on_bank_id`).

## Known issues

None open.

## Won't-do (out of scope, but documented to save the question)

- **Multi-class softmax head as a default.** The constructor supports `num_classes >= 2` (the
  loss switches to cross-entropy), but the CLIs assume the v1 binary head and export rejects
  multi-class checkpoints. `FibroticTypeLabel` is Phase 2 territory — promoting it earlier
  drags the multi-class metric / reporting / export story along with it.
- **In-process `predict()` entry point.** The three CLIs cover all consumers; notebook users
  import `MobileViT1D` directly. Deployment-time behavior (preprocessing, sigmoid + threshold)
  stays in one canonical home — the metadata sidecar consumed by the C++ runtime.
- **Ensemble training.** Single-model v1 demonstrates the pipeline; ensembling adds
  train / eval / export complexity without informing the v1 design questions.

## Open architectural questions for later

- **Calibration-method abstraction.** Temperature scaling is hardcoded as the one post-hoc
  step. A second method (Platt, isotonic) would justify a `Calibrator` Protocol
  (`fit(logits, labels) -> Self`, `apply(logits) -> logits`); speculative until then.
- **More normalization variants in `egm_class_model_metadata`** (`robust_zscore`, `meanvar`
  with per-channel stats once multi-channel). Add when a real model needs them.
- **Whether `expected_trace_samples` / `expected_fs_hz` should imply a deployment resampler.**
  Today the metadata documents what the runtime must do; it doesn't ship a resampler. A
  "feed me any sample rate, I'll resample" contract is a metadata + ONNX-graph change to
  design carefully.
