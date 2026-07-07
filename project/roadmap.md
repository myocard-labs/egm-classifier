# egm-classifier — roadmap

What's planned for future releases. Internal doc — public users see
the README and `docs/usage.md`.

## v0.1.0 — current release (shipped)

Scope (recap; see `architecture.md` for the design rationale):

- **Model:** 1D MobileViT backbone with the registry-driven block
  list (`models/`). Single-logit binary head + BCE-with-logits loss
  for the v1 fibrotic-vs-healthy task (`num_classes >= 2` + softmax
  path stays available for the future multi-class FibroticTypeLabel).
- **`egm-class-train`** CLI: full training loop over a labeled
  `ClassifierBank` — AdamW + linear-warmup + cosine decay, optional
  AMP, optional gradient clipping, AUROC-based best-checkpoint
  selection, typed `run.json` + `metrics.csv` written via the
  egm-data records layer.
- **`egm-class-eval`** CLI: load a checkpoint, run sequential
  inference over a labeled bank, populate `ClassifierPrediction` on
  every trace, write a sibling `<stem>_pred.cbank.h5`, print the
  scalar metric bundle to stdout.
- **`egm-class-export`** CLI: optional temperature-scaling fit
  against a labeled bank, ONNX export with the fitted `T` baked into
  the graph (`logits / T`), `model_metadata.json` sidecar with
  sha256 + preprocessing + decision + best-effort training
  provenance.
- **Patient-aware splitting** with pluggable stratification
  (`AnyPositive`, `BinnedDensity`) via egm-data's
  `patient_aware_split`.
- **Per-trace augmentation:** random gain + random time-shift via
  egm-data's `TraceTransform` (train split only). Per-trace z-score
  is the v1 normalization.
- **Metrics:** torchmetrics-driven binary bundle (AUROC, accuracy,
  F1, precision, recall, ECE, confusion + reliability bins) reused
  by both the training loop's per-epoch eval and the eval CLI.

Pinned dependencies (drop the direct git references once they
publish to PyPI):

```
myocard-egm-contracts @ git+...@v0.4.0
myocard-egm-data[torch] @ git+...@v0.3.3
myocard-egm-signal     @ git+...@v0.2.0
```

The `[onnx]` extra (onnx, onnxruntime, onnxscript) is optional so
training-only images don't pull the deployment toolchain;
`egm-class-export` catches the import failure at runtime and points
the user at the extra.

## v0.4.0 — stable cross-artifact IDs (shipped)

Consumes the egm-contracts v0.5.x cross-artifact-linkage schemas.
egm-classifier is the *consumer* end that closes the provenance graph
the producers (iafdb-pipeline, synthetic-egm-pipeline) opened.

- **Train** stamps `run_id` + `produced_model_id` (derived from the new
  `output.run_name` descriptor) and `trained_on_bank_id` (the training
  bank's id) onto `run.json`, and embeds the same trio in `best.pt`'s
  `training_provenance`.
- **Export** reads the checkpoint's `training_provenance` and surfaces
  `produced_model_id` as the metadata sidecar's top-level `model_id`
  (`egm_class_model_metadata` schema 1.2); `run_id` + `trained_on_bank_id`
  + `run_name` thread through as provenance breadcrumbs.
- **Eval** stamps the predictions bank with a derived `lpred_` (labeled)
  or `upred_` (unlabeled) id — overridable via `output.bank_id` — and
  records the producing model's id in each trace's `trace_metadata`
  under `produced_by_model_id` (the short-term home; see
  `architecture.md` §"Cross-artifact stable IDs").
- **Eval now supports unlabeled banks** (`upred_`, metrics skipped) —
  the IAFDB-inspection path the old labeled-only eval rejected. Only the
  metric *computation* is skipped (no substrate truth); the predictions
  are still written.

Pins bumped: `myocard-egm-contracts v0.4.0 → v0.5.1`,
`myocard-egm-data[torch] v0.3.3 → v0.4.0` (signal stays `v0.2.0`);
`pydantic>=2` added as a direct dep.

## v0.2.0+ — concrete next steps

These are sized for "could land in one focused PR each." Items
that landed cross-cutting Phase work in the meta repo's
`project_plan.md` carry a `→ tracked at intracardiac-platform Phase X`
annotation so it's clear which items are scheduled vs which are
component-internal tech debt.

### `zero2one` normalization in train + eval — Phase 1.5

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5 (increase synthetic-data complexity + realism).

The export CLI already supports `zscore` and `zero2one` (the latter
applied directly inside the calibration loop). Train + eval are
hardcoded to `zscore` via egm-data's `TraceTransform`. To finish the
job:

- Add a `normalize` mode to `TraceTransform` (currently a `znorm`
  boolean) that accepts `'zscore' | 'zero2one' | 'none'`.
- Plumb `normalization_scheme` through the train + eval YAML
  configs.
- Stamp the chosen scheme into the checkpoint's `model_meta` dict
  so the export CLI can derive it instead of taking it from YAML
  (with a back-compat fallback for older checkpoints that don't
  carry the field).
- Empirically compare `zero2one` vs `zscore` on the v1 model and
  record the result in `project/investigations/`.

The deferred work was deliberately moved out of v0.1.0 scope so the
training side didn't have to grow a new code path mid-port.

### `run.json` as the export-config source (downsample policy) — component-internal cleanup

> → Component-internal cleanup; tracked as task #286. Do anytime; doesn't need a project-phase home. Pairs naturally with the `filters.decimation` work scheduled in egm-signal under Phase 1.5.

> **Status (v0.4.0):** the provenance-plumbing half of #286 shipped —
> the trainer embeds its cross-artifact ids in `best.pt`'s
> `training_provenance` and export + eval auto-read them, so export no
> longer needs to locate a sibling `run.json` for *ids*. What remains
> below is specifically the `fs_hz`/`bandpass_hz` downsample policy
> (still YAML-duplicated).

The export YAML currently duplicates `fs_hz` and `bandpass_hz` from
the training pipeline — typo-prone, no enforcement that the values
match what the model was trained against. To fix:

- Training-side: pick a single training `fs_hz` and stamp it into
  `run.json` alongside the model_meta. If the bank contains traces
  at multiple sample rates, downsample everything to the training
  rate (which must be ≤ the minimum bank `freq_hz`).
- Export-side: add an optional `run_json:` field to the export YAML.
  When set, `fs_hz` + future preprocessing fields are read from
  there; inline YAML values are still allowed but warn-on-disagree /
  silent-on-agree / silent-on-omission. Export-only knobs
  (threshold, opset, output paths, `class_labels` override) continue
  to live in the export YAML.

Deferred until the full polyrepo refactor is settled so the
training-side change can land cleanly.

### Progress bar for `egm-class-eval` execution — component-internal

> → Component-internal UX; do anytime, no project-phase home. Low priority.

`egm-class-eval` runs the bank through the model sequentially with no
feedback, so a large bank (e.g. a full IAFDB inference bank) looks hung on an
underpowered machine. Wrap the eval prediction loop in a progress bar
(`tqdm`, with a `--no-progress` escape hatch + graceful degradation when
`tqdm` is absent, matching the `iafdb-export-bank` producer pattern), driven
by the trace/batch count. Training already has per-epoch feedback; this brings
the eval CLI to parity. Surfaced while evaluating an IAFDB bank on a laptop.

### Validate config `bank_id` override at config-load — Refactor Step 8 (cleanup)

> → Surfaced 2026-06-30 from synthetic-egm-pipeline bank-id testing. Cross-tracked in `intracardiac-platform/project/refactor_checklist.md` Step 8.

The predictions-bank `bank_id` override is validated for id-validity at the
**write** step, so a bad hand-set id only fails after train/eval runs. Move the
check to config-load time (fail-fast). Shared cleanup with the producer
pipelines (synthetic-egm-pipeline + iafdb-pipeline); pairs with the egm-contracts
"optional date suffix" relaxation.

### Cross-project code placement audit — Refactor Step 8 (cleanup)

> → Tracked at `intracardiac-platform/project/refactor_checklist.md` Phase 8 (cleanup + verification). The audit is the comprehensive end-of-refactor sweep — by Phase 8 all repos exist and have settled, so it can resolve every misplaced piece of code at once rather than piecemeal during each scaffolding pass. TraceTransform is one known candidate; others will surface as the per-repo roadmap reviews proceed. Also tracked as task #287.

Some library code currently lives in the wrong package. Known
candidate:

- **`TraceTransform`** lives in `myocard-egm-data` but is only
  consumed by `myocard-egm-classifier` (training + eval data
  loaders, plus the export-side calibration loop). If no other
  consumer materializes by audit time, move it to egm-classifier.

The audit should sweep every shared library
(`myocard-egm-data`, `myocard-egm-signal`, `myocard-egm-contracts`)
for code that's only consumed by one downstream component.
Deliverable: a brief audit doc with each questionable item, its
current home, its consumers, a recommendation (move / keep /
split), then a follow-up commit per move.

### Migrate `torch.onnx.export` to the dynamo path — component-internal

> → Component-internal. Tracked as task #288. Do when convenient; no cross-cutting dependencies.

The export currently passes `dynamo=False` to use the legacy
TorchScript-based exporter. The legacy path handles our 1D Conv
model cleanly and works with the older `dynamic_axes={...}` argument
shape, but PyTorch is moving toward the dynamo exporter (default in
2.5+). After v0.1.0:

- Switch `export_to_onnx` to `dynamo=True`.
- Replace `dynamic_axes` with `dynamic_shapes={"signal": {0:
  torch.export.Dim("batch", min=1)}}`.
- Re-run the parity smoke against an actual trained checkpoint
  (not just an untrained model) to confirm tolerance numbers hold.
- Bump opset default if dynamo emits anything newer.

### Move `MobileViTBlock` divisibility check from `forward` to `__init__` — component-internal

> → Component-internal. Tracked as task #289.

`mobilevit_block.py:135` does `if T % p != 0: raise ValueError(...)`
inside `forward()`, which produces one `TracerWarning` per block
per export (~23 warnings on a v1 export). The check is structural —
a function of `input_length` + the architecture's stride pattern,
known at module-construction time, doesn't need to be re-validated
per forward call. Fix:

1. Add an `input_length` kwarg to `MobileViTBlock.__init__`;
   validate `T % patch_size == 0` there.
2. Delete the forward-time check.
3. Re-run the ONNX export and confirm zero `TracerWarning`s, then
   remove the `warnings.filterwarnings(..., TracerWarning)`
   suppression in `export/onnx_export.py`.

### Investigation: does activation-peak anchoring help? — Phase 1.5

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5. Cross-repo work: opt-in flag in synthetic-egm-pipeline + A/B comparison here. Also tracked as task #293.

The v1 synthetic producer always crops each trace to a fixed
window centered on the single simulated activation peak — what
`docs/theory.md` §1.3 calls "activation-peak anchoring" (distinct
from the clinical surface-ECG "R-wave anchoring"). The motivation
is signal-density-per-window: an unanchored random-offset crop
often catches mostly baseline, diluting training signal. Whether
the network actually needs this help, or could learn the
activation location on its own from enough training samples, is
an open empirical question. No published answer exists in the
intracardiac EGM ML literature.

Two-step follow-up:

1. Add an opt-in producer-side flag in `synthetic-egm-pipeline`
   so anchoring can be turned off when emitting a bank. Default
   stays on for backward compatibility.
2. Train v1 classifier on both anchored and unanchored variants
   of the same simulation set, A/B compare AUROC + ECE + the
   train/val gap.

Possible methods-paper candidate if the result is decisive
either way. Phase 2 multi-activation simulations make anchoring
moot anyway (you want long traces spanning many activations, not
a single-peak crop), so the v0.2.0–Phase-2 window is the right
time to settle this.

### Investigation: training-time additive noise augmentation — Phase 1.5

> → Tracked at `intracardiac-platform/project/project_plan.md` Phase 1.5. Also tracked as task #294.

v1 ships with no training-time additive-noise augmentation —
realistic recording noise is injected at *producer* time by the
synthetic-egm-pipeline mixer step (using noise extracted from
IAFDB recordings), so the classifier sees deployment-realistic
noise by construction. Adding a second layer of additive noise at
training time would shift that distribution off-target.

Some groups working on synthetic-data ML pipelines use
training-time additive noise as a regularizer even when the
producer already adds noise. Two-step follow-up:

1. Survey the synthetic-EGM ML literature (and adjacent
   synthetic-biosignal ML work) for groups doing training-time
   additive-noise augmentation alongside producer-side noise
   mixing. Look for consistent reports of generalization benefit.
2. If the survey is encouraging, add a `TraceTransform`
   augmentation and A/B test in the v1 classifier.

### Theory-docs convention (with egm-viewer / egm-studio) — Refactor Step 6

> → Tracked at `intracardiac-platform/project/refactor_checklist.md` Phase 6 (egm-studio). The egm-studio scaffolding pass needs to know about this convention so `docs/usage.md` includes the visual-interpretation half + cross-link back here for the math.

`docs/theory.md` in this repo is the source of truth for the
**math + operational meaning** of every metric, knob, and
algorithm (formulas, paper references). It also owns the
**troubleshooting** section that consolidates symptom-driven
"action when X" guidance — symptoms span multiple knobs, so they
collect at the end (§6) rather than scattering through the
reference sections.

When egm-viewer's `docs/usage.md` lands, it will own the
**visual interpretation** half (how to read each chart, axis
conventions, common visual misreadings) and cross-link back here
for the underlying math. Both docs stay self-contained for their
audience without duplicating the math. See
`feedback_theory_docs_split` in the workspace memory for the full
rationale; we chose 2026-06-23 to bundle visual interpretation
into the viewer's single `usage.md` rather than splitting out a
separate `visualization_guide.md`.

This convention should also land in
`intracardiac-platform/project/refactor_checklist.md` as part of
the post-v0.1.0 refactor-cleanup review so it survives across
components.

## Schema bumps to coordinate

None currently planned. The most recent change was the
`egm-contracts` v0.5.0 cross-artifact-linkage wave (consumed in
egm-classifier v0.4.0): `egm_class_model_metadata` → schema 1.2
(`model_id`) and `training_run_record` → schema 1.1 (`run_id` +
`produced_model_id` + `trained_on_bank_id`). No follow-up bumps are
expected for the v0.2.0+ items above unless the `run.json`
preprocessing extension (`v0.2.0+ — concrete next steps` §2) needs
new `training_run_record` fields.

## Won't-do (out of scope, but documented to save the question)

- **Multi-class softmax head as a v0.2.0 default.** The model
  constructor supports `num_classes >= 2` and the loss switches to
  cross-entropy automatically, but the training + eval CLIs assume
  the v1 binary head and the export CLI rejects multi-class
  checkpoints. The `FibroticTypeLabel` work is Phase 2 territory;
  promoting it earlier means dragging the multi-class metric +
  reporting + export story along with it.
- **In-process inference helper / "predict()" entry point.** The
  three CLIs cover all current consumers (training, evaluation,
  ONNX-for-deployment). Notebook users can import `MobileViT1D`
  directly. We deliberately don't expose a `Classifier.predict()`
  wrapper so deployment-time behavior (preprocessing,
  normalization, sigmoid + threshold) only has one canonical home —
  the metadata sidecar consumed by the C++ runtime.
- **Ensemble training.** Single-model v1 is enough to demonstrate
  the pipeline; ensembling adds complexity to training, eval, and
  export without informing the v1 design questions.

## Open architectural questions for later

- **Calibration method abstraction.** Currently temperature scaling
  is hardcoded as the one and only post-hoc calibration step. If a
  second method is ever needed (Platt scaling, isotonic regression),
  consider extracting a `Calibrator` Protocol with
  `fit(logits, labels) -> Self` and `apply(logits) -> logits`. Until
  there's a second method, that abstraction is speculative.
- **Whether the `egm_class_model_metadata` schema should grow more
  normalization variants** (`robust_zscore`, `meanvar` with stored
  per-channel stats once we go multi-channel). Add when a real
  model needs them.
- **Whether `expected_trace_samples` and `expected_fs_hz` should
  imply a deployment-time resampler.** Today the metadata documents
  what the runtime must do; it doesn't ship a resampler. If C++
  deployment ever wants a "feed me any sample rate, I'll resample"
  contract, that's a metadata + ONNX-graph change worth designing
  carefully.
