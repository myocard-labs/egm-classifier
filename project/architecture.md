# egm-classifier — architecture and design rationale

What's in this repo, why it's split the way it is, and the design
calls a future maintainer would otherwise have to re-derive from the
code. Pairs with `roadmap.md` (what's next) and the public
`docs/usage.md` (how to drive the CLIs).

## Where the package sits

egm-classifier is the **consumer-side ML package** in the
myocard-labs polyrepo. It sits downstream of every producer and
upstream of the deployment runtime:

```
                ┌────────────────────────────────────────┐
                │             egm-contracts              │
                │  (JSON Schemas + Pydantic codegen)     │
                └────────────────────────────────────────┘
                                  │  pinned
       ┌──────────────────────────┼──────────────────────────┐
       │                          │                          │
       ▼                          ▼                          ▼
 ┌───────────┐            ┌───────────────┐         ┌───────────────┐
 │ egm-data  │            │  egm-signal   │         │   producers   │
 │  I/O +    │            │   DSP + ML    │         │   (iafdb /    │
 │  Pydantic │            │  primitives   │         │   synthetic)  │
 └───────────┘            └───────────────┘         └───────────────┘
       │                          │                          │
       └───────┬──────────────────┴──────────────┬───────────┘
               ▼                                  ▼
        ┌────────────────────────────────────────────┐
        │              egm-classifier                │   <-- this repo
        │   train + eval + export CLIs over a        │
        │   ClassifierBank, emit ONNX + metadata     │
        └────────────────────────────────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   C++ inference        │
                  │   (TensorRT, ONNX RT)  │
                  └────────────────────────┘
```

The contract surface is narrow on both sides:

- **Inputs:** one `ClassifierBank` HDF5 file (egm-data format), pinned
  to a `myocard-egm-contracts` schema version. Producers
  (iafdb-pipeline, synthetic-egm-pipeline) write these.
- **Outputs:** a trained `best.pt` + `run.json` + `metrics.csv` for
  training audit; a `<stem>_pred.cbank.h5` sibling for eval-time
  predictions; a `<name>.onnx` + `<name>.model_metadata.json` pair
  for deployment.

Everything downstream of the ONNX + metadata pair is a separate
project's problem; everything upstream of the bank is a separate
project's problem.

## Folder layout

```
src/myocard_egm_classifier/
├── constants.py             # every magic number the design pins
├── inference_helpers.py     # collect_logits (train + eval share it)
├── metrics.py               # binary_metrics (train + eval share it)
├── models/                  # the 1D MobileViT architecture
│   ├── blocks.py            # InvertedResidual + DropPath
│   ├── transformer.py       # the encoder block
│   ├── mobilevit_block.py   # the unfold/transformer/fold trick
│   ├── mobilevit1d.py       # full backbone assembly
│   ├── registry.py          # block-type dispatch
│   └── __init__.py
├── training/
│   ├── train.py             # the loop itself
│   ├── reporting.py         # write_run + EpochRecord helpers
│   └── __init__.py          # back-compat re-exports
├── eval/
│   ├── dataset.py           # build_eval_dataset (sequential, no-augment)
│   ├── predictions.py       # populate_predictions + path helper
│   └── __init__.py
├── export/                  # optional; needs the [onnx] extra
│   ├── graph_wrapper.py     # CalibratedModel wrapper
│   ├── calibration.py       # fit_temperature_from_bank
│   ├── onnx_export.py       # torch.onnx.export wrapper
│   ├── metadata.py          # build_metadata + write_metadata
│   └── __init__.py
└── cli/                     # console-script entry points
    ├── _common.py           # shared YAML + runtime helpers
    ├── _train_config.py     # train-only typed config
    ├── _eval_config.py      # eval-only typed config
    ├── _export_config.py    # export-only typed config
    ├── train_cmd.py         # egm-class-train
    ├── eval_cmd.py          # egm-class-eval
    ├── export_cmd.py        # egm-class-export
    └── __init__.py
```

## Why three CLIs (not one mega-CLI)

A single `egm-classifier train|eval|export` subcommand entry point
would have been less typing in `[project.scripts]`, but three reasons
to keep them separate:

1. **Different optional dependencies.** `egm-class-train` and
   `egm-class-eval` only need PyTorch + egm-data. `egm-class-export`
   adds `onnx`, `onnxruntime`, and `onnxscript` (the `[onnx]` extra).
   A training image (Docker, K8s pod) that doesn't deploy doesn't
   need to pull ~200 MB of ONNX toolchain.
2. **Different YAML config shapes.** Each CLI's YAML carries
   genuinely different fields (training has `train:` +
   `data.split_strategy:`; eval has the predictions-bank output path;
   export has calibration + sidecar fields). Forcing them into one
   schema would either bloat every config with irrelevant keys or
   gate keys per subcommand at validation time — both worse than
   three small typed configs.
3. **Pipeline ordering is explicit.** Three separate scripts in a
   shell snippet make the train → eval → export pipeline visible at
   the operator level, not buried in a subcommand graph.

The shared machinery (config loading, device selection, checkpoint
rebuild, model-meta convention) all lives in `cli/_common.py`, so the
three thin `*_cmd.py` files only do argparse + orchestration.

## Why YAML configs (and not argparse-only)

Same rationale as iafdb-pipeline and synthetic-egm-pipeline:

- **Reproducibility.** A YAML file is what you commit to a
  notebook/experiment-tracking repo. Argparse flags don't round-trip
  to disk without extra scaffolding.
- **Layered overrides.** Argparse flags act as one-off overrides on
  top of the YAML, so quick experiments don't require editing the
  config file (`egm-class-train v1.yaml --epochs 5 --device cpu`).
- **Strict validation.** Each block has an explicit `_*_KEYS`
  allowlist; a typo'd key (`with_multiplier` vs `width_multiplier`)
  raises at config-load time instead of silently dropping the
  override.
- **Path resolution.** The YAML loader stashes the config file's
  parent directory under `_config_dir`; every path field resolves
  relative to that. So `bank: ../banks/x.h5` works the same from
  anywhere you invoke the CLI.

## The model-config triad split

`ModelCLIConfig` (and its `build_model_from_config`,
`model_meta_from_config` siblings) live in `cli/_train_config.py`,
*not* in `cli/_common.py` even though all three CLIs need *some*
model knowledge. The split:

- **Train** builds a `ModelCLIConfig` from the YAML `model:` block,
  hands it to `build_model_from_config` to construct the architecture,
  then calls `model_meta_from_config` to get a dict to stamp into the
  checkpoint.
- **Eval** and **export** never see a `ModelCLIConfig`. They call
  `cli/_common.py:build_model_from_meta(meta_dict)` directly against
  the checkpoint's embedded `model_meta` dict — same architecture
  knobs, no dataclass round-trip.

Why split it this way: the `ModelCLIConfig` dataclass is a *train-
time YAML schema*, not a runtime concept. Eval and export don't
construct models from a YAML; they reconstruct them from a saved
checkpoint. Forcing them through `ModelCLIConfig` was a stutter step
that obscured the real data flow.

## The training loop

`training/train.py` implements the Phase-1 recipe pinned in
`constants.py` + the design doc:

- **Loss:** `BCEWithLogitsLoss` (single-logit head) with an optional
  `pos_weight` derived from the train-split class counts in egm-data's
  `build_dataloaders`. Multi-class falls back to `CrossEntropyLoss`
  when the head's `num_classes >= 2`, but the v1 CLIs are
  binary-only.
- **Optimizer:** AdamW with `lr=3e-4`, `weight_decay=0.05`,
  `betas=(0.9, 0.999)`. AdamW (Loshchilov & Hutter) decouples L2
  from the gradient update, which matters for transformer blocks
  inside the MobileViT.
- **Schedule:** linear warmup over 5% of total steps, then cosine
  decay to 0. Pure-Python in `cosine_warmup_lr`; no torch scheduler
  to keep the train loop transparent.
- **Selection metric:** AUROC. Threshold-free; robust under the
  fibrotic/healthy class imbalance.
- **Per-epoch state** lands as a Pydantic `EpochRecord` from
  `myocard-egm-data.records` — the same shape that `write_run`
  consumes at end-of-training. No internal schema gymnastics.

The per-epoch evaluator on the val split and the eval CLI both call
the top-level `inference_helpers.collect_logits` +
`metrics.binary_metrics` to keep their metric semantics identical.

## Patient-aware splitting

Trace-level random splits leak: traces from the same recording can
land in both train and val, making val performance look better than
test. The patient-aware split puts every trace from one patient (or
one synthetic simulation) entirely in one of train/val/test.

`patient_aware_split` lives in egm-data and takes a
`PatientStratificationStrategy`. v1 ships two:

- **`AnyPositive`** (default): one bit per patient — 1 if any of the
  patient's traces is positive, else 0. Works on global-density
  banks (where a patient's traces share a label) and degrades
  gracefully on local-density banks.
- **`BinnedDensity`**: bins patients by their positive-trace rate.
  Useful when local-density banks have wide per-patient distribution
  spread; switch to this when the AUROC-instability warning fires
  on `AnyPositive`.

Both are configured via the train YAML's `data.split_strategy:`
block; the dispatch happens in `cli/_train_config.py:to_strategy`.

## TraceTransform (and where it currently lives)

`TraceTransform` is the per-trace pad/crop + z-score + augmentation
applied inside egm-data's `EGMTraceDataset`. It's only consumed by
egm-classifier (training loop + eval data loader + export-side
calibration loop) and is a strong candidate for relocation in the
cross-project code-placement audit (`roadmap.md` v0.2.0+ §3). For
now it lives in egm-data because that's where the data loaders live
and TraceTransform is wired into `EGMTraceDataset`'s `__getitem__`.

## The eval CLI

`egm-class-eval` is intentionally thin and intentionally narrow:

- **Input:** one labeled `ClassifierBank` + one checkpoint.
- **Output:** a sibling `<stem>_pred.cbank.h5` with
  `ClassifierPrediction` stamped on every trace, and a metric
  bundle printed to stdout. **No metrics file written.**
- **Path:** sequential iteration over the bank (no patient split, no
  augmentation), `collect_logits` over the loader,
  `populate_predictions` over the bank object, `write_classifier_bank`
  out.

Two deliberate non-features:

1. **No unlabeled-bank inference.** Eval assumes every trace has
   `label_truth`. Running against an unlabeled IAFDB-only bank
   yields degenerate AUROC and single-class warnings from egm-data's
   loaders. The IAFDB-as-eval-target path is logically circular and
   tracked separately (`project_iafdb_eval_catch22` memory).
2. **No metrics file.** The raw logits + labels live on the
   predictions bank; any metric (or recomputation at a different
   threshold) can be re-derived from there. Writing a separate
   metrics file would duplicate the information and create two
   sources of truth.

## The export CLI

`egm-class-export` is a three-step pipeline behind argparse:

1. **(Optional) Calibration fit.** If `calibration.bank` is set, the
   model is run over the bank with the configured per-trace
   normalization, logits + labels collected, and
   `myocard_egm_signal.fit_temperature` invoked for a bounded scalar
   minimization. Returns `T`. When skipped (no bank, or
   `--skip-calibration`), `T = 1.0`.
2. **Graph wrapping.** The trained model is wrapped in a
   `CalibratedModel(base, T)` whose forward is `base(x) / T`. `T` is
   a `register_buffer` (not a `Parameter`) — frozen at construction
   time, serializes into the ONNX graph as a constant.
3. **ONNX + metadata emission.** `torch.onnx.export` (legacy
   `dynamo=False` path; see `roadmap.md` for the migration plan)
   emits the graph with input name `signal`, output name `logit`,
   dynamic batch axis, opset 18. The metadata sidecar carries the
   sha256 + size of the ONNX file, the preprocessing constants, the
   decision threshold, the class labels, and best-effort training
   provenance.

## Why temperature scaling at export time (not train, not eval)

Three places it *could* live, only one place it should:

- **Train time** would entangle calibration with the training loop.
  The training loop's job is to fit the model on the training
  distribution; calibration is a post-hoc fix for *systematic
  overconfidence* on the validation distribution. Different problems,
  different solvers, different consequences if they're conflated.
- **Eval time** would make the eval CLI's metrics depend on a
  calibration choice. ECE and any threshold-dependent metric at
  thresholds ≠ 0.5 would shift. The eval CLI's job is to report
  *raw model behavior* against a held-out bank; downstream tooling
  decides what to do with the numbers.
- **Export time** is when "the model is becoming a deployment
  artifact." That's the right moment for the deployment-policy
  decisions (decision threshold, class labels, *and* the
  calibration scalar). Bake `T` into the ONNX graph and the C++
  runtime doesn't need a calibration concept at all — it just runs
  the model.

## Normalization scheme (zscore / zero2one / none)

The metadata sidecar's `preprocessing.normalization.scheme` enum has
three values; `zscore` is the v1 training default. They thread
through differently today:

- **Train + eval (today):** `TraceTransform` only supports `zscore`
  via its `znorm` boolean. Train + eval banks must be zscore-trained
  models.
- **Export (today):** all three schemes are supported in the
  calibration loop. The export YAML's
  `preprocessing.normalization_scheme` field is the single source of
  truth — used by both the calibration loop (applied to traces
  before fitting `T`) and the metadata sidecar (recorded so the
  deployment runtime applies the same scheme).

The asymmetry exists because adding `zero2one` to `TraceTransform`
is a train-side change that's deferred until after v0.1.0 (roadmap
§1). Until then, the export YAML's `normalization_scheme` field can
*technically* take `zero2one`, but the corresponding training side
hasn't been wired — that combination would fit `T` against
`zero2one`-preprocessed inputs and ship a model that was trained on
`zscore` ones. Don't do that until the train + eval side ships
`zero2one` support.

## `fs_hz` and `bandpass_hz` are YAML-only

The export YAML's `preprocessing.fs_hz` and `preprocessing.bandpass_hz`
are required fields, not derived from the calibration bank, even
though it's tempting to read them off the bank's attrs:

- **`ClassifierBank` doesn't carry `bandpass_hz`.** The bandpass
  filter was applied upstream (in the synthetic producer or the
  iafdb-pipeline IAFDB R-wave extraction) and the filter parameters
  weren't preserved on the aggregated ClassifierBank.
- **`ClassifierBank` per-trace `freq_hz` can be mixed.** A bank
  aggregated from multiple source banks at different sample rates
  carries each trace's `freq_hz` individually. The "deployment-time
  expected `fs_hz`" is a training-pipeline configuration choice
  (which rate the model was trained at), not an attribute that's
  recoverable from a bank that may have mixed source rates.

So the deployment-time constants are *training-pipeline
configuration*, not bank attributes; trying to reconstruct them from
a bank is brittle. The cleaner long-term answer — teach training to
stamp its preprocessing config into `run.json` so export can read
from there — is the v0.2.0+ §2 followup. Until then, the export YAML
duplicates these two numbers, which is mildly annoying but at least
explicit.

## Provenance: what gets written where

| Artifact | Producer | Schema | Lives next to |
|---|---|---|---|
| `best.pt` | training loop, when val metric improves | torch dict: `model_state_dict`, `model_meta`, `train_config`, `val_loss`, `val_metrics` | configured `output.checkpoint_dir` |
| `run.json` | training loop, at end-of-run | `myocard-egm-contracts.training_run_record` | same dir as `best.pt` |
| `metrics.csv` | training loop, at end-of-run | `myocard-egm-contracts.training_metrics` (one row per epoch) | same dir as `best.pt` |
| `<stem>_pred.cbank.h5` | eval CLI | `myocard-egm-contracts.classifier_bank` with `ClassifierPrediction` populated | sibling to input bank |
| `<name>.onnx` | export CLI | ONNX graph (calibrated logits) | configured `output.dir` |
| `<name>.model_metadata.json` | export CLI | `myocard-egm-contracts.egm_class_model_metadata` schema 1.1 | sibling to `.onnx` |

Note the asymmetry between train and eval outputs: training writes
two files (`run.json` + `metrics.csv`) because the per-epoch CSV is
the friendly tabular form and `run.json` is the audit/JSON form; eval
writes only the predictions bank because the metrics print to stdout
(eval is one-shot; there's no per-epoch history to summarize).

## Testing strategy

`tests/` covers three layers:

1. **Pure-function unit tests** — `test_metrics.py` against
   hand-calculated perfect/worst cases, `test_model.py` against a
   tiny forward pass, `test_reporting.py` against
   `write_run`/`load_*` round-trips.
2. **CLI integration tests** — `test_train.py` + `test_eval_cmd.py` +
   `test_export_cmd.py` invoke each CLI's `main()` against tiny
   in-memory fixtures and assert on the artifacts. `test_cli_config.py`
   covers the YAML loader behavior in isolation.
3. **Cross-stack parity test** — `test_export_cmd.py::test_pytorch_onnx_parity`
   exports a model, loads the ONNX via `onnxruntime`, runs both
   side-by-side, asserts elementwise diff < 1e-4. Catches any
   regression where the ONNX graph drifts away from PyTorch's
   numerics.

CI installs both `[dev]` and `[onnx]` so the export tests can
actually run their parity check end-to-end.

## Version coordination

Three sibling pins:

```
myocard-egm-contracts @ git+...@v0.4.0  # schemas
myocard-egm-data[torch] @ git+...@v0.3.3  # bank I/O + datasets + records
myocard-egm-signal     @ git+...@v0.2.0  # temperature_scaling
```

Plus one runtime cap:

```
numpy>=1.26,<2.5  # via egm-contracts; PEP 695 stubs in numpy 2.5 break
                  # mypy under python_version="3.10"
```

When egm-contracts ships a breaking schema change (as v0.3.0 → v0.4.0
just did with the per-trace normalization redesign), egm-data has to
re-pin first, then egm-classifier. The cascade is captured in
`intracardiac-platform/project/refactor_checklist.md`.

The `[onnx]` extra (`onnx`, `onnxruntime`, `onnxscript`) is
deliberately optional so training-only images don't pull the export
toolchain. The export CLI's argparse + main are importable without
those packages; the `ModuleNotFoundError` only fires deep inside
`torch.onnx.export` at call time, and `cli/export_cmd.py` catches it
and re-emits a friendly install hint pointing at the extra.
