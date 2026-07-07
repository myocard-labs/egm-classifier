# Code-placement audit — Refactor Step 8

End-of-refactor sweep for library code living in the wrong package. The
smell we're looking for is **single-consumer code in a shared library**:
if only one downstream component ever imports a piece of a shared library,
that code usually belongs *in* that component, not in the library. By
Step 8 every repo exists and has settled, so the sweep can resolve the
whole picture at once instead of guessing mid-scaffold.

One row per questionable item: where it lived, who consumes it, the
recommendation, and what actually happened.

## Summary

| Item | Was in | Consumers | Call | Outcome |
|---|---|---|---|---|
| Training-data layer — `datasets` + `splits` + `augmentation` | `myocard-egm-data` | `myocard-egm-classifier` only | **Move** | Moved to egm-classifier `myocard_egm_classifier.data`; egm-data → **v0.5.0**, pure I/O |
| DSP primitives — filtering, calibration, threshold extraction | `myocard-egm-signal` | iafdb-pipeline, synthetic-egm-pipeline, egm-features | Keep | No change — genuinely shared |
| Per-trace feature extraction | `myocard-egm-features` | egm-studio (egm-classifier planned) | Keep | No change — library by design |
| Format schemas + validators | `myocard-egm-contracts` | every repo | Keep | No change |

## 1. Training-data layer → egm-classifier (moved)

**What moved.** Three subpackages of `myocard-egm-data`:

- `datasets/` — `EGMTraceDataset` (a PyTorch `Dataset` over a `ClassifierBank`),
  `build_dataloaders`, and `LoaderBundle`.
- `splits/` — `patient_aware_split`, `split_classifier_bank`, and the
  pluggable stratification strategies (`AnyPositive`, `BinnedDensity`).
- `augmentation/` — `TraceTransform` (per-call train-time normalize + pad +
  augment).

They now live under `myocard_egm_classifier.data` (same three subpackage
names), with their tests. `banks/` reading stays in egm-data: the moved
code imports `from myocard_egm_data.banks import …` for bank I/O.

**Why move.** Two independent signals pointed the same way:

1. **Single consumer.** Across the whole `myocard-labs` tree, nothing but
   egm-classifier imported `myocard_egm_data.{datasets,splits,augmentation}`
   (a repo-wide grep at audit time confirmed it — egm-studio and the two
   producers touch only `banks`/`records`/`phases`). Training data loaders,
   the patient-aware split, and the export-side calibration loop are all
   egm-classifier code.
2. **Torch isolation.** These three subpackages were the *only* reason
   egm-data carried a `[torch]` optional extra. torch is a direct,
   non-optional dependency of egm-classifier anyway. Moving them lets
   egm-data drop torch entirely and become a lean, numpy + h5py pure-I/O
   library that a notebook, the viewer, or an analysis script can install
   without dragging in torch.

**Version cascade.** egm-data `v0.4.2 → v0.5.0` (breaking: public API +
`[torch]` extra removed). Consumers re-pin
`myocard-egm-data[torch] v0.4.x` → `myocard-egm-data v0.5.0` (no extra).
egm-classifier is the only consumer that used the moved code; egm-studio
and the producers used egm-data for bank/record/phase I/O only, so their
re-pin is a straight version bump with no code change.

**Docs that came with the code.** The `TraceTransform` pre-processing
review (`project/trace_transform_review.md`) and the training-data-layer
future-work items (variable-length datasets, splitting-strategy choices,
streaming loaders — now in `project/roadmap.md`) moved from egm-data's
project docs into egm-classifier's, so the design thinking stays next to
the code it describes.

## 2. Shared libraries that stayed put

- **`myocard-egm-signal`** — DSP primitives (band-pass, R-wave-anchored
  calibration, voltage-threshold extraction). Consumed by both producers
  (iafdb-pipeline calibration/segmentation, synthetic-egm-pipeline mixing)
  and by egm-features. Genuinely multi-consumer; stays a shared library.
- **`myocard-egm-features`** — per-trace morphology / spectral / complexity
  features. Today's consumer is egm-studio (diagnostics + v1_baseline
  figures); egm-classifier is a planned consumer (Phase 4 feature-conditioned
  work). It is a library by design and already typed/packaged as one, so
  even at one current consumer it stays put — moving it into egm-studio
  would block the planned second consumer.
- **`myocard-egm-contracts`** — the format schemas + generated Pydantic +
  validators. Consumed by every repo; the canonical shared library.

No other single-consumer misplacements surfaced in the sweep.

## 3. Bank file-extension convention — standardized on `.classifier.h5`

Surfaced in the audit and resolved 2026-07-07: the on-disk extension for a
classifier-shaped bank was **inconsistent** across the stack. egm-classifier
wrote predictions as `_pred.cbank.h5`, while egm-studio demo banks and the
producers (iafdb-pipeline, synthetic-egm-pipeline) write `*.classifier.h5` —
two names for the same `ClassifierBank` shape. **Standardized on
`.classifier.h5`** (the producers' actual output + the ecosystem majority):
egm-classifier's `default_predictions_bank_path` now emits
`<stem>_pred.classifier.h5`, and the docs / examples / tests across every
repo were swept to match.

The convention is still implicit — it lives in reader/writer call sites, not
documented or enforced in one place. Recording it in egm-data (the I/O owner)
or egm-contracts alongside the schemas, and optionally having the writers
stamp/expect it, remains a nice-to-have follow-up.

## References

- `project/trace_transform_review.md` — the moved `TraceTransform` review.
- `project/roadmap.md` — "Cross-project code placement audit" (outcome) +
  "Training-data layer follow-ups" (migrated future work).
- `intracardiac-platform/project/refactor_checklist.md` — Step 8 tracker.
