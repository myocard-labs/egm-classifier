# Using myocard-egm-classifier

How to drive the three CLIs end-to-end: train a 1D MobileViT
binary classifier on a `ClassifierBank`, evaluate the trained
checkpoint, export it as ONNX + a deployment-ready metadata
sidecar. For the *why* behind the design (model architecture,
training recipe, knob trade-offs) see `docs/theory.md`. For the
*deployment-side* contract (how a C++/TensorRT runtime consumes
the export) see `docs/onnx_deployment.md`.

## Install

The package is distributed via git for now (PyPI publication is
deferred until the polyrepo refactor settles).

```bash
# Base install — training + eval CLIs.
pip install git+https://github.com/myocard-labs/egm-classifier.git@v0.4.0

# With the ONNX export toolchain (onnx + onnxruntime + onnxscript).
pip install "git+https://github.com/myocard-labs/egm-classifier.git@v0.4.0#egg=myocard-egm-classifier[onnx]"
```

The `[onnx]` extra is kept optional so training-only images (Docker,
K8s) don't pull ~200 MB of deployment toolchain. The
`egm-class-export` script is installed regardless; if you invoke it
without the extra, it'll catch the import failure and tell you which
extra to install.

Three sibling packages get pulled in automatically:

- `myocard-egm-contracts==0.5.3` — JSON Schemas + Pydantic models
  (the stable cross-artifact `ArtifactId` pattern lives here).
- `myocard-egm-data==0.5.0` — `ClassifierBank` I/O (bank / record /
  phase readers + writers). The `EGMTraceDataset`, patient-aware split,
  and `TraceTransform` now live in this package (`.data`), not egm-data.
- `myocard-egm-signal==0.2.0` — temperature-scaling primitive used
  by the export CLI's calibration step.

`pydantic>=2` is also a direct dependency (it validates stable artifact
ids); it arrives transitively via egm-contracts regardless.

## CLI reference

Three console scripts, one per pipeline stage. All three are
YAML-config-driven; argparse flags act as per-invocation overrides
on top of the YAML.

### `egm-class-train`

Train a model end-to-end against one `ClassifierBank`.

```
egm-class-train CONFIG.yaml [--bank PATH] [--epochs N] [--width F]
                            [--batch-size N] [--checkpoint-dir PATH]
                            [--device cuda|cpu] [--seed N] [--no-augment]
```

| Flag | Overrides | Use case |
|---|---|---|
| `--bank PATH` | `data.bank` | Swap the source bank without editing the YAML (e.g. for an ablation). |
| `--epochs N` | `train.epochs` | Quick smoke runs (`--epochs 1`). |
| `--width F` | `model.width_multiplier` | Architecture sweep (0.5 small / 1.0 baseline / 1.5 large). |
| `--batch-size N` | `data.batch_size` | Fit a bigger / smaller GPU. |
| `--checkpoint-dir PATH` | `output.checkpoint_dir` | Direct outputs to a fresh dir without editing the YAML. |
| `--device cuda\|cpu` | (no YAML equivalent) | Force a device; otherwise auto-pick CUDA when available. |
| `--seed N` | `train.seed` | Reproducibility / multi-seed sweeps. |
| `--no-augment` | `data.augment_train` | Disable train-split augmentation. |

**Outputs** (written under `output.checkpoint_dir`):

- `best.pt` — torch checkpoint of the best-AUROC epoch. Carries
  `model_state_dict` + `model_meta` + `train_config` + `val_loss` +
  `val_metrics`.
- `run.json` — `TrainingRunRecord` Pydantic dump: full config snapshot
  + per-epoch records + held-out test metrics + run identity (host, git
  SHA, wall times) + the stable cross-artifact ids `run_id`,
  `produced_model_id`, and `trained_on_bank_id` (see "Stable artifact
  ids" below). The same ids are embedded in `best.pt`'s
  `training_provenance` so export + eval read them without a sibling
  `run.json`.
- `metrics.csv` — one row per epoch with the scalar metrics; the
  friendly tabular form of `run.json`'s `epochs` block.

### `egm-class-eval`

Evaluate a trained checkpoint against a labeled `ClassifierBank`.

```
egm-class-eval CONFIG.yaml [--checkpoint PATH] [--bank PATH]
                           [--predictions-bank PATH] [--threshold F]
                           [--device cuda|cpu]
```

| Flag | Overrides | Use case |
|---|---|---|
| `--checkpoint PATH` | `checkpoint` | Run the same eval YAML against a different `best.pt`. |
| `--bank PATH` | `data.bank` | Evaluate against a different bank. |
| `--predictions-bank PATH` | `output.predictions_bank` | Direct the output to a custom location; default is sibling `<stem>_pred.classifier.h5`. |
| `--threshold F` | (the decision threshold for `label_pred`) | Sweep thresholds without re-eval; the predictions bank carries raw logits, so any threshold can be re-applied downstream. |
| `--device cuda\|cpu` | (no YAML equivalent) | Force a device. |

**Outputs**:

- `<input_stem>_pred.classifier.h5` — sibling `ClassifierBank` with
  `ClassifierPrediction` populated on every trace (`label_pred`,
  `label_prob`, raw `pred_logits`). It carries its own stable id —
  `lpred_<run_name>_<date>` (labeled input) or
  `upred_<run_name>_<date>` (unlabeled input), with `run_name` read
  from the checkpoint's provenance — and the producing model's id is
  stamped into each trace's `trace_metadata` under
  `produced_by_model_id`. No separate metrics file — the logits +
  labels live on this bank, so any metric can be re-derived.
- **Stdout summary** (labeled input only) — `n`, `accuracy`, `auroc`,
  `f1`, `precision`, `recall`, `ece`, confusion matrix.

Both labeled and unlabeled banks are supported. A fully-labeled bank
yields a scored eval (`lpred_` bank + the metric bundle); an unlabeled
bank (the IAFDB shape) yields a `upred_` bank with predictions written
but metrics skipped — substrate-truth metrics can't be computed without
labels and aren't faked. See `project/architecture.md` and the
`project_iafdb_eval_catch22` memory for why metric-based IAFDB eval is
off the table.

### `egm-class-export`

Export a trained checkpoint to ONNX + a metadata sidecar.

```
egm-class-export CONFIG.yaml [--checkpoint PATH] [--calibration-bank PATH]
                             [--skip-calibration] [--output-dir PATH]
                             [--output-name NAME] [--threshold F]
                             [--opset N] [--device cuda|cpu]
```

| Flag | Overrides | Use case |
|---|---|---|
| `--checkpoint PATH` | `checkpoint` | Re-export from a different `best.pt` without editing the YAML (e.g. exporting from each run of an architecture sweep against the same calibration + preprocessing config). |
| `--calibration-bank PATH` | `calibration.bank` | Swap the labeled bank used to fit the temperature scalar. |
| `--skip-calibration` | forces `calibration.bank = None` | Ship with `T = 1.0` (raw logits, no calibration). Wins over `--calibration-bank` if both are passed. |
| `--output-dir PATH` | `output.dir` | Direct the export pair to a fresh directory without editing the YAML. |
| `--output-name NAME` | `output.name` | Produces `<NAME>.onnx` + `<NAME>.model_metadata.json` instead of the default `best.onnx` + `best.model_metadata.json`. |
| `--threshold F` | `decision.threshold` | Stamp a different deployment decision threshold into the metadata sidecar without editing the YAML (e.g. ship a precision-tuned `0.7` variant alongside the default `0.5`). |
| `--opset N` | `opset` | Override the ONNX operator-set version (default 18). |
| `--device cuda\|cpu` | (no YAML equivalent) | Force a device for the export pass. |

**Outputs** (written under `output.dir`):

- `<output.name>.onnx` — the calibrated graph. With `T` fitted at
  export time and baked into the graph, the runtime gets
  `logits / T` directly.
- `<output.name>.model_metadata.json` — the deployment-time sidecar
  (`egm_class_model_metadata` schema 1.2): sha256 + size_bytes,
  preprocessing constants, decision threshold + class labels, the
  model's own stable `model_id` (read from the checkpoint's
  `produced_model_id`), and training provenance — the fitted `T` plus
  the `run_id` / `trained_on_bank_id` / `run_name` breadcrumbs carried
  over from the checkpoint.

The CLI refuses to overwrite existing output files; pass
`--output-name` to a fresh value or delete the existing files first.
See `docs/onnx_deployment.md` for how a downstream runtime consumes
this pair.

## YAML schema reference

Three configs, one per CLI. Paths in every config resolve relative
to the YAML file's directory (same convention iafdb-pipeline and
synthetic-egm-pipeline use). Unknown keys at any level raise a
loud `ConfigError` at load time — typos don't get silently dropped.

### `egm-class-train` config

```yaml
# REQUIRED — path to a ClassifierBank HDF5 (label_truth on every trace).
data:
  bank: ../banks/noise_mixed_v1.classifier.h5
  batch_size: 64                  # default
  num_workers: 0                  # default
  znorm: true                     # per-trace z-score (v1 default)
  znorm_eps: 1.0e-7               # divisor floor
  augment_train: true             # gain + time-shift on the train split only
  max_gain: 0.0                   # default (no-op under znorm)
  max_shift_frac: 0.10            # default
  split_fractions: [0.8, 0.1, 0.1]  # train / val / test
  split_seed: 0                   # default
  pin_memory: false               # default
  split_strategy:                 # patient-aware stratification
    type: any_positive            # or 'binned_density'
    n_bins: 3                     # used only by binned_density

# OPTIONAL — model architecture knobs (every field has a default).
model:
  width_multiplier: 1.0           # 0.5 small / 1.0 baseline / 1.5 large
  num_classes: 1                  # 1 = single-logit BCE binary head
  in_channels: 1                  # 1 = bipolar single trace
  input_length: 512               # T at 1 kHz
  stochastic_depth: 0.1
  head_expansion_channels: 320
  head_dropout: 0.1
  # 'blocks:' is also accepted for the rare custom-architecture case.

# OPTIONAL — training-loop knobs.
train:
  epochs: 60
  lr: 3.0e-4
  weight_decay: 0.05
  warmup_frac: 0.05               # linear warmup fraction
  grad_clip_norm: 1.0
  amp: false                      # mixed precision; turn on for GPU runs
  amp_dtype: bf16                 # 'bf16' (Ampere+) or 'fp16'
  seed: 42
  select_metric: auroc            # val metric to maximize on best.pt selection

# OPTIONAL — where outputs land.
output:
  checkpoint_dir: ../checkpoints/v1_baseline
  description: "v1 baseline run"
  run_name: v1_baseline          # descriptor for the run's stable ids
                                 # (run_<run_name>_<date> + model_<run_name>_<date>);
                                 # omit to fall back to 'egm_classifier'
```

### `egm-class-eval` config

```yaml
# REQUIRED — checkpoint to evaluate.
checkpoint: ../checkpoints/v1_baseline/best.pt

# REQUIRED — labeled ClassifierBank to evaluate against.
data:
  bank: ../banks/synthetic_test.classifier.h5
  batch_size: 64                  # default
  num_workers: 0                  # default
  znorm: true                     # default; must match training
  znorm_eps: 1.0e-7               # default
  pin_memory: false               # default

# OPTIONAL — where the predictions bank lands.
output:
  # Default is sibling <input_stem>_pred.classifier.h5; override here for a
  # custom destination.
  predictions_bank: ../predictions/v1_baseline_pred.classifier.h5
  # OPTIONAL — stable id for the predictions bank. Omit to derive it
  # automatically: lpred_<run_name>_<date> (labeled input) or
  # upred_<run_name>_<date> (unlabeled input). Set to override verbatim.
  bank_id: lpred_v1_baseline_holdout_2026-06-27

# OPTIONAL — decision threshold (range (0, 1)).
threshold: 0.5                    # default
```

There's no `model:` block in the eval YAML — the architecture is
rebuilt from the checkpoint's embedded `model_meta`. Same in the
export config.

### `egm-class-export` config

```yaml
# REQUIRED — checkpoint to export.
checkpoint: ../checkpoints/v1_baseline/best.pt

# OPTIONAL — labeled bank for the temperature fit. Omit (or set
# `bank: null`) to skip calibration and ship with T = 1.0.
calibration:
  bank: ../banks/calibration_v1.classifier.h5
  batch_size: 64                  # default
  num_workers: 0                  # default
  pin_memory: false               # default

# REQUIRED — preprocessing constants for the metadata sidecar.
# These describe what the deployment runtime applies to each
# incoming trace and MUST match what the model was trained with.
preprocessing:
  fs_hz: 1000.0                   # REQUIRED — training-time sampling rate
  bandpass_hz: [30.0, 250.0]      # REQUIRED — [low, high] in Hz
  normalization_scheme: zscore    # 'zscore' (default) | 'zero2one' | 'none'
  normalization_eps: 1.0e-7       # per-trace divisor floor

# OPTIONAL — output paths. Produces <name>.onnx + <name>.model_metadata.json.
output:
  dir: ../exports/v1_baseline
  name: best                      # default

# OPTIONAL — ONNX operator-set version.
opset: 18                         # default; don't lower (current
                                  # PyTorch breaks on auto-downconvert)

# OPTIONAL — decision-rule constants for the metadata sidecar.
decision:
  threshold: 0.5                  # default
  class_labels: [healthy, fibrotic]  # default
```

The `preprocessing.fs_hz` + `bandpass_hz` requirement (no defaults)
is intentional — these depend on the trained model. See
`project/architecture.md` for why they're not derived from the
calibration bank.

## Stable artifact IDs

Every tracked artifact this package writes carries a stable
cross-artifact id — the egm-contracts `ArtifactId` pattern,
`<role>_<descriptor>_<YYYY-MM-DD>`. They let a downstream provenance
index answer "what produced what" without parsing file contents:

| Artifact | Field | Shape | Role prefix |
|---|---|---|---|
| Training run (`run.json`) | `run_id` | `run_<run_name>_<date>` | `run_` |
| Exported model (`model_metadata.json`) | `model_id` | `model_<run_name>_<date>` | `model_` |
| Predictions bank (`_pred.classifier.h5`) | `id` | `lpred_<run_name>_<date>` (labeled) / `upred_<run_name>_<date>` (unlabeled) | `lpred_` / `upred_` |

`run_name` is the single human-chosen descriptor: set it once in the
train YAML (`output.run_name`) and it threads through the run, the model,
and any predictions bank produced from that model. Omit it and
everything falls back to `egm_classifier`. Dates are stamped at write
time (UTC).

Relationship pointers ride alongside the ids so the graph has edges, not
just nodes:

- `run.json` records `trained_on_bank_id` (the bank the run trained on)
  and `produced_model_id` (the model it will export to).
- The model sidecar's `training_provenance` carries `run_id` +
  `trained_on_bank_id` back-links (the model's own id is the top-level
  `model_id`).
- The predictions bank records the producing model's id in **each
  trace's `trace_metadata`** under `produced_by_model_id` (a short-term
  home until a dedicated field lands), and the source bank via its
  embedded `banks[]` provenance entries.

Each producer derives its ids automatically; pass `output.bank_id` in
the eval YAML to override the predictions-bank id verbatim (it's
validated against the `ArtifactId` pattern, so a malformed value fails
fast).

## End-to-end walkthroughs

The three CLIs compose into a standard `train → eval → export`
pipeline.

### The v1 baseline (full recipe)

```bash
# 1. Train.
egm-class-train examples/v1_baseline.yaml

# 2. Eval against a held-out test bank.
egm-class-eval examples/v1_eval.yaml

# 3. Export with temperature calibration.
egm-class-export examples/v1_export.yaml
```

Three example configs ship under `examples/`. Edit the paths to
point at your banks + desired output locations.

### Quick smoke run on CPU

For a 1–2 minute sanity check that everything's wired:

```bash
egm-class-train examples/v1_baseline.yaml \
  --epochs 2 --device cpu --batch-size 16
```

The val curve will be terrible (2 epochs isn't enough to learn) but
the artifacts at `output.checkpoint_dir` will exist + be valid.

### Architecture sweep

Compare three model sizes without editing the YAML:

```bash
for w in 0.5 1.0 1.5; do
  egm-class-train examples/v1_baseline.yaml \
    --width $w \
    --checkpoint-dir ../checkpoints/v1_w${w}
done
```

Each run writes its own `best.pt` + `run.json` + `metrics.csv` to a
size-specific directory.

### Re-fit calibration without re-training

If your training checkpoint is fine but you want a tighter
calibration fit (e.g. you've grown your labeled calibration bank
since the last export):

```bash
egm-class-export examples/v1_export.yaml \
  --calibration-bank ../banks/calibration_v2.classifier.h5 \
  --output-name best_calv2
```

This re-uses the existing checkpoint, re-fits `T` against the new
bank, and writes a fresh `best_calv2.onnx` + `.model_metadata.json`
pair next to the originals. The clobber guard prevents accidental
overwrite of the prior export.

### Skip calibration entirely

Useful when the calibration bank isn't ready yet but you want an
ONNX for a deployment smoke test:

```bash
egm-class-export examples/v1_export.yaml --skip-calibration
```

Ships with `T = 1.0` (raw logits, no calibration). The metadata
sidecar's `training_provenance.calibration_temperature` records the
unity value so audit tooling can distinguish "calibrated to 1.0"
from "calibration was skipped."

## Programmatic use

Most users will only ever drive the CLIs. The package is
import-friendly for notebooks + custom scripts when that's not
enough.

### Loading + running a checkpoint from Python

```python
import torch
from myocard_egm_classifier.cli._common import load_checkpoint_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ckpt = load_checkpoint_model("../checkpoints/v1_baseline/best.pt", device)

# model is a ready-to-use MobileViT1D in eval mode.
# ckpt is the full torch.load dict — model_meta, train_config, val_metrics, ...

# Apply your own preprocessing (z-score, pad/crop, etc.) before forward:
with torch.no_grad():
    logit = model(x_batch_of_shape_B_1_T)  # [B, 1]
    p_fibrotic = torch.sigmoid(logit)
```

### Re-deriving metrics from a predictions bank

```python
import numpy as np
from myocard_egm_data.banks import load_classifier_bank
from myocard_egm_classifier.metrics import binary_metrics

bank = load_classifier_bank("../predictions/v1_baseline_pred.classifier.h5")

# Raw logits are stamped on every trace under prediction.pred_logits.
logits = np.array([t.prediction.pred_logits[1] for t in bank.traces])
labels = bank.label_truth_array()

metrics = binary_metrics(logits, labels, threshold=0.7)  # try a different threshold
print(metrics)
```

### Running an exported ONNX from Python

For the deployment-side reference implementation (preprocessing
pipeline + sigmoid + threshold + class label), see
`docs/onnx_deployment.md`. The full worked example there is ~75
lines of Python.

## Where to read more

- `docs/onnx_deployment.md` — how a downstream runtime (C++,
  TensorRT, Python sanity-check) consumes the export pair.
- `docs/theory.md` — math + paper references for every metric and
  every tunable knob. The "I'm tuning a new model and want to know
  what AUROC actually means" doc.
- `project/architecture.md` — internal package architecture: why
  three CLIs, why the model triad split, why calibration at export
  time, where each artifact lives. Read before making
  non-trivial changes.
- `project/roadmap.md` — what's planned for v0.2.0+ and what's
  explicitly out of scope.
- `README.md` — one-paragraph overview + install snippet.
