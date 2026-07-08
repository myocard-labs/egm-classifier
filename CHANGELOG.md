# Changelog

All notable changes to `myocard-egm-classifier` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Training-data layer** — `myocard_egm_classifier.data` (`datasets` + `splits` +
  `augmentation`, incl. `EGMTraceDataset`, `build_dataloaders`, `patient_aware_split`,
  `TraceTransform`), absorbed from egm-data since egm-classifier is its sole consumer
  (Refactor Step 8 code-placement audit; see `project/code_placement_audit.md`).
- **Fail-fast eval `bank_id` validation** — a malformed `output.bank_id` override is rejected
  at config-load (`build_eval_config`), before the eval run.

### Changed

- Predictions-bank extension standardized on `<stem>_pred.classifier.h5` (was `.cbank.h5`),
  matching the producers + egm-studio.
- Re-pin `egm-contracts v0.5.1 → v0.5.3` and `egm-data[torch] v0.4.0 → egm-data v0.5.0` —
  dropping the `[torch]` extra (torch is already a direct dependency).

## [0.4.0] — 2026-06-28

### Added

- **Stable cross-artifact IDs** (consuming the egm-contracts v0.5.x linkage schemas —
  egm-classifier is the consumer end that closes the provenance graph). Train stamps
  `run_id` / `produced_model_id` (from `output.run_name`) + `trained_on_bank_id` onto
  `run.json` and into `best.pt`; export surfaces `produced_model_id` as the sidecar's
  `model_id` (`egm_class_model_metadata` 1.2); eval stamps a derived `lpred_` / `upred_` id
  (overridable) on the predictions bank.
- **Unlabeled-bank eval** (`upred_`, metrics skipped) — the IAFDB-inspection path the old
  labeled-only eval rejected; predictions are still written.

### Dependencies

Re-pins `egm-contracts v0.5.1`, `egm-data[torch] v0.4.0` (signal stays `v0.2.0`); adds
`pydantic>=2` as a direct dependency.

## [0.1.0] — 2026-06-23

First release: the Phase-1 1D MobileViT binary fibrosis classifier — train / eval / export.

### Added

- **Model** — a 1D MobileViT backbone (registry-driven block list) with a single-logit
  binary head + BCE-with-logits loss (the `num_classes >= 2` softmax path stays available for
  the future multi-class `FibroticTypeLabel`).
- **`egm-class-train`** — full training loop over a labeled `ClassifierBank` (AdamW +
  linear-warmup + cosine decay, optional AMP / grad-clip, AUROC-based best-checkpoint
  selection, typed `run.json` + `metrics.csv`).
- **`egm-class-eval`** — checkpoint → sequential inference → `ClassifierPrediction` on every
  trace → sibling predictions bank + a scalar metric bundle to stdout.
- **`egm-class-export`** — optional temperature-scaling fit + ONNX export with the fitted `T`
  baked into the graph + a `model_metadata.json` sidecar.
- **Patient-aware splitting** (`AnyPositive` / `BinnedDensity`), **per-trace augmentation**
  (random gain + time-shift, train split only) + z-score normalization, and a
  torchmetrics-driven binary metric bundle shared by training + eval.

### Dependencies

Pins `egm-contracts v0.4.0`, `egm-data[torch] v0.3.3`, `egm-signal v0.2.0`; the `[onnx]`
extra is optional so training-only images skip the deployment toolchain.

[Unreleased]: https://github.com/myocard-labs/egm-classifier/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/myocard-labs/egm-classifier/releases/tag/v0.4.0
[0.1.0]: https://github.com/myocard-labs/egm-classifier/releases/tag/v0.1.0
