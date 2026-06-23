# ONNX deployment guide

How to consume the artifact pair that `egm-class-export` emits.
Audience: anyone writing inference code (C++, TensorRT, Python
sanity-check tooling, anyone else) against an exported EGM
classifier. Pairs with `docs/usage.md` (which covers the producer
side — how to run the export CLI).

## What `egm-class-export` produces

Two files, written side-by-side under the configured `output.dir`:

```
<output.name>.onnx                    # the calibrated graph
<output.name>.model_metadata.json     # the deployment-time sidecar
```

By default `output.name = "best"`, so the pair is `best.onnx` +
`best.model_metadata.json`. Both files are intended to ship together
— the ONNX graph alone doesn't carry the preprocessing and decision
constants the runtime needs.

## The ONNX graph contract

Every export emits a graph with the same input/output surface:

| Field | Value | Notes |
|---|---|---|
| **Input name** | `"signal"` | Look up by name; don't rely on positional ordering. |
| **Input shape** | `[batch, 1, T]` | `batch` is dynamic; `1` is the Conv1d channel axis; `T` is the trace length (typically 512). |
| **Input dtype** | `float32` | |
| **Output name** | `"logit"` | |
| **Output shape** | `[batch, 1]` | The `1` here is the single-logit binary head. |
| **Output dtype** | `float32` | |
| **Output semantics** | `binary_logit` | Already temperature-calibrated; apply sigmoid + threshold and you're done. |
| **Opset** | 18 | Supported by current ONNX Runtime + TensorRT 8.6+/10 releases. |

**The output is a calibrated logit.** The temperature-scaling
divisor `T` (fitted at export time against a held-out labeled bank,
or `1.0` if calibration was skipped) is **baked into the graph as a
constant** — the runtime does not need to know about `T`, apply it
separately, or look it up in the metadata. See "Calibration
semantics" below.

The batch axis is dynamic, so the same ONNX serves both:

- **Real-time scoring** (`batch = 1`): one trace at a time as the
  catheter streams in. Latency-bound.
- **Offline bulk** (`batch = N`): many traces in one shot for
  post-hoc analysis of a recorded session. Throughput-bound.

No re-export is required to switch modes.

## The metadata sidecar contract

The sidecar JSON conforms to the
`egm_class_model_metadata` schema (currently version `1.1`) in
`myocard-egm-contracts`. Top-level fields:

### `schema_version`
String. Currently `"1.1"`. **Consumers MUST refuse unknown major
versions.** Schema 1.x is back-compatible across minor bumps; a
hypothetical future 2.x is not.

### `created_utc`
ISO-8601 UTC timestamp at export time. Informational; useful for
matching against `run.json` audit trails.

### `model_artifact`
Pointer to the paired ONNX file:

- `filename` — string, relative to the metadata file's directory.
  Resolve as `dirname(metadata.json) / filename`.
- `framework` — currently always `"onnx"`. The schema reserves
  `"tensorrt"` and `"pytorch"` for future artifact types.
- `sha256` — 64-character hex string. Compute the sha256 of the
  loaded ONNX file at startup and compare; refuse to load on
  mismatch.
- `size_bytes` — integer. Cheap sanity check before the sha256
  pass.

### `input` / `output`
Redundant with the ONNX graph itself, but documented here so the
runtime can sanity-check the incoming data shape before invoking
the model. Treat as a contract check, not as authoritative — the
ONNX graph is the source of truth for shape/dtype.

### `preprocessing`
What the runtime must apply to each incoming trace **before**
invoking the model. **All four fields are required.**

| Field | What |
|---|---|
| `expected_fs_hz` | Sampling rate, Hz. Inputs at a different rate must be resampled before invocation. |
| `expected_trace_samples` | Length in samples per trace. Must equal the time axis in `input.shape`. |
| `bandpass_hz` | `[low_hz, high_hz]` for the band-pass filter applied at training time. The runtime must apply the same filter. |
| `normalization.scheme` | One of `"zscore"`, `"zero2one"`, `"none"`. Per-trace; see below. |

The normalization schemes are all **per-trace** (computed from each
trace's own samples), not per-channel or per-bank:

- **`zscore`** — `x_new = (x - mean(x)) / std(x)` along the time
  axis. The v1 training default.
- **`zero2one`** — `x_new = (x - min(x)) / (max(x) - min(x))`.
- **`none`** — pass-through. Use only if the upstream pipeline has
  already normalized.

For both `zscore` and `zero2one` the runtime should floor the
divisor at a small positive epsilon (typically `1e-7`) to avoid
division-by-zero on degenerate flat traces.

### `decision`
What the runtime applies **after** the model's sigmoid:

- `threshold` — float in `(0, 1)`. The decision rule is
  `label_pred = 1 if sigmoid(logit) >= threshold else 0`. Default
  `0.5`.
- `class_labels` — list of human-readable names indexed by class
  code. For the v1 binary head: `["healthy", "fibrotic"]` (index
  0 = healthy, index 1 = fibrotic).

### `training_provenance`
Open-ended dict (`additionalProperties: true`) for audit
breadcrumbs. Producer-specific keys the egm-classifier export
always stamps:

- `calibration_temperature` — the fitted `T` (or `1.0` if
  calibration was skipped). **Informational only;** the graph
  already bakes `1/T`. Recorded for audit + reproducibility.
- `calibration_bank_path` — absolute path to the labeled bank used
  for the fit, or `null` if calibration was skipped.
- `checkpoint_path` — absolute path to the source training
  checkpoint this export was built from.

Well-known schema keys the export passes through when the
checkpoint carries them: `run_id`, `run_json_path`,
`training_bank_path`, `training_bank_schema_version`, `git_sha`.

## The runtime inference pipeline

Step-by-step. The order matters; the model expects the
preprocessing in exactly this sequence:

1. **Receive a raw bipolar EGM trace** from the acquisition system
   in mV at the device's native sample rate.
2. **(Conditional) Resample** to `preprocessing.expected_fs_hz` if
   the acquisition rate doesn't match.
3. **Apply the band-pass filter** with edges from
   `preprocessing.bandpass_hz`. Typically a forward-backward
   Butterworth (zero-phase). The training-side filter is the same
   one egm-signal exposes; matching the exact filter design avoids
   transfer-function mismatch.
4. **Crop or zero-pad** to `preprocessing.expected_trace_samples`.
   Cropping should be centered around the R-wave if upstream
   timing carries one; padding is on the right (zero-pad).
5. **Apply per-trace normalization** per
   `preprocessing.normalization.scheme`. See the formulas above.
6. **Reshape** to `[batch, 1, expected_trace_samples]` float32 and
   batch as appropriate for your throughput target.
7. **Invoke the ONNX model** with input name `"signal"` and read
   output `"logit"`.
8. **Apply sigmoid:** `p = 1 / (1 + exp(-logit))`. The result is
   `P(class = 1)`, i.e. `P(fibrotic)` for the v1 binary head.
9. **Compare to threshold:** `label_pred = 1 if p >= decision.threshold else 0`.
10. **(Optional) Map to human label:** `decision.class_labels[label_pred]`.

If `p` itself is needed as a continuous score (e.g. for ranking
candidate ablation sites), keep it before the threshold step;
the threshold is just for binarization.

## Calibration semantics (why no `T` in the runtime)

Temperature scaling (Guo et al. 2017) is the post-hoc fix for
modern deep classifiers' systematic overconfidence — divide the
logit by a fitted scalar `T > 0` before sigmoid. We bake `T`
**into the ONNX graph at export time** for two reasons:

1. **The deployment runtime doesn't need a calibration concept.**
   It just runs the model. If the model was calibrated, the graph
   emits `logit / T`; if not, it emits `logit / 1 = logit`. Same
   code path either way.
2. **Single source of truth.** Storing `T` in the metadata and
   applying it in the runtime would invite drift if someone edits
   the metadata. Bake it into the graph, record the value in
   `training_provenance.calibration_temperature` for audit, and
   the two can't disagree.

## Verification recommendations

In rough priority order:

1. **`schema_version` check.** Refuse to load anything with a
   major version your code wasn't built against. (Today only
   `1.x` exists.)
2. **`sha256` check.** Compute the sha256 of the loaded ONNX,
   compare against `model_artifact.sha256`, refuse on mismatch.
   This catches tampering, transfer corruption, and the much more
   common "you accidentally loaded yesterday's ONNX against today's
   metadata."
3. **Shape sanity check.** Compare ONNX `input.shape` vs
   `metadata.input.shape`; refuse on disagreement (defensive — the
   exporter should keep them in sync).
4. **Preprocessing wiring check.** At runtime startup, log the
   `preprocessing.expected_fs_hz`, `bandpass_hz`, and
   `normalization.scheme` you're using. If they differ from the
   metadata, the model will silently degrade rather than crash.
5. **Threshold + class-label sanity.** Confirm
   `decision.threshold` is in `(0, 1)` and
   `decision.class_labels` length matches the output classes.

## Common pitfalls

- **Sample-rate mismatch.** If the acquisition system runs at
  500 Hz but the metadata says `expected_fs_hz = 1000.0` and you
  skip the resample, the model receives data at half the expected
  rate. No exception, just degraded accuracy.
- **Normalization-scheme mismatch.** If the metadata says
  `zscore` and the runtime applies `zero2one` (or skips
  normalization entirely), every input looks foreign to the
  model. Especially silent because the shapes and dtypes are
  fine.
- **Applying sigmoid twice.** The output is a logit, not a
  probability. Apply sigmoid exactly once after the model and
  before the threshold check.
- **Applying `T` again.** `T` is already in the graph. Reading
  `training_provenance.calibration_temperature` and dividing the
  logit by it again will distort calibration.
- **Wrong batch axis.** The model accepts dynamic batch but pins
  channel = 1 and time = `expected_trace_samples`. Feeding
  `[B, T]` (missing channel axis) or `[1, B, T]` (wrong axis
  order) will error from the ONNX runtime, but the error message
  is shape-mismatch not "you have the layout wrong."
- **Stale metadata + new ONNX.** The `sha256` check guards
  against this; don't skip it in production.

## Worked example (Python, for reference)

The Python deployment path looks like this. C++/TensorRT is
analogous; substitute the equivalent calls for sha256, signal
processing, and inference.

```python
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import butter, sosfiltfilt


def load_metadata(metadata_path: Path) -> dict:
    meta = json.loads(metadata_path.read_text())
    if meta["schema_version"].split(".")[0] != "1":
        raise RuntimeError(f"Unsupported schema major: {meta['schema_version']}")
    return meta


def verify_onnx(onnx_path: Path, meta: dict) -> None:
    sha = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    expected = meta["model_artifact"]["sha256"]
    if sha != expected:
        raise RuntimeError(f"ONNX sha256 mismatch: {sha} vs {expected}")


def preprocess(trace: np.ndarray, meta: dict) -> np.ndarray:
    p = meta["preprocessing"]
    fs = p["expected_fs_hz"]
    low, high = p["bandpass_hz"]
    target_len = p["expected_trace_samples"]
    scheme = p["normalization"]["scheme"]

    # 1) bandpass
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, trace).astype(np.float32)

    # 2) crop or pad to target_len (center-crop / right-pad)
    if x.shape[0] > target_len:
        start = (x.shape[0] - target_len) // 2
        x = x[start : start + target_len]
    elif x.shape[0] < target_len:
        x = np.pad(x, (0, target_len - x.shape[0]))

    # 3) per-trace normalize
    eps = 1e-7
    if scheme == "zscore":
        x = (x - x.mean()) / max(x.std(), eps)
    elif scheme == "zero2one":
        x = (x - x.min()) / max(x.max() - x.min(), eps)
    elif scheme == "none":
        pass
    else:
        raise RuntimeError(f"Unknown normalization scheme: {scheme}")

    # 4) reshape to [1, 1, T]
    return x.reshape(1, 1, target_len)


def score(session: ort.InferenceSession, meta: dict, trace: np.ndarray) -> dict:
    x = preprocess(trace, meta)
    logit = session.run(["logit"], {"signal": x})[0][0, 0]
    p_fibrotic = 1.0 / (1.0 + np.exp(-logit))
    label_pred = int(p_fibrotic >= meta["decision"]["threshold"])
    label_name = meta["decision"]["class_labels"][label_pred]
    return {"p_fibrotic": float(p_fibrotic), "label_pred": label_pred,
            "label_name": label_name, "logit": float(logit)}


# usage
metadata = load_metadata(Path("best.model_metadata.json"))
verify_onnx(Path("best.onnx"), metadata)
session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

result = score(session, metadata, raw_trace_from_acquisition)
print(result)
```

For batched offline inference, accumulate preprocessed traces into
a `[batch, 1, T]` array and feed the whole array in one call.

## Version compatibility

When the metadata schema bumps major version (1.x → 2.0), this
repo will publish a corresponding `egm-classifier` release that
emits the new schema. Deployment-side code targeting schema 1.x
should reject any artifact with `schema_version` starting with
`"2."` and require an explicit code update. The schema change log
lives in
`myocard-egm-contracts/project/schema_evolution.md` (the
"Schema change log" section).
