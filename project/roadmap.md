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

### `zero2one` normalization in train + eval

The export CLI already supports `zscore` and `zero2one`; train + eval are hardcoded to
`zscore` via the `TraceTransform`. Finish the job: add a `normalize` mode
(`'zscore' | 'zero2one' | 'none'`) to `TraceTransform` (now in `myocard_egm_classifier.data`,
currently a `znorm` bool), plumb `normalization_scheme` through the train + eval YAML, stamp
the chosen scheme into the checkpoint `model_meta` (with a back-compat fallback), and
empirically compare `zero2one` vs `zscore` on the v1 model in `project/investigations/`.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

### Investigation: does activation-peak anchoring help?

The v1 synthetic producer always crops each trace to a fixed window centered on the single
simulated activation peak (`docs/theory.md` §1.3 "activation-peak anchoring"; distinct from
clinical R-wave anchoring). Whether the network needs this help or could learn the activation
location itself is an open empirical question with no published intracardiac-EGM-ML answer.
Two steps: (1) an opt-in producer-side flag in synthetic-egm-pipeline to disable anchoring
(default on); (2) train v1 on anchored vs unanchored variants of the same sim set and A/B
compare AUROC / ECE / train-val gap. Possible methods-paper candidate. Phase 2's
multi-activation sims make anchoring moot, so the pre-Phase-2 window is the time to settle it.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5. Cross-repo (opt-in
> flag in synth + the A/B comparison here).

### Investigation: training-time additive-noise augmentation

v1 ships no training-time additive noise — the synthetic-egm-pipeline mixer injects
deployment-realistic IAFDB noise at *producer* time, so a second training-time layer would
shift that distribution off-target. But some synthetic-data ML groups use it as a regularizer
even alongside producer-side noise. Two steps: (1) survey the synthetic-biosignal ML
literature for consistent generalization benefit; (2) if encouraging, add a `TraceTransform`
augmentation and A/B test.

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5.

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

### Progress bar for `egm-class-eval`

The eval loop runs sequentially with no feedback, so a large IAFDB inference bank looks hung
on an underpowered machine. Wrap the loop in `tqdm` (with a `--no-progress` escape hatch +
graceful degradation when tqdm is absent, matching the producer pattern). Low priority.

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

None currently planned. The most recent was the egm-contracts v0.5.0 cross-artifact-linkage
wave (consumed in v0.4.0): `egm_class_model_metadata` → 1.2 (`model_id`) and
`training_run_record` → 1.1 (`run_id` + `produced_model_id` + `trained_on_bank_id`). No
follow-up bumps expected unless the `run.json` preprocessing extension (Backlog) needs new
`training_run_record` fields.

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
