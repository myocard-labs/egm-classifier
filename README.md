# myocard-egm-classifier

> 1D MobileViT binary classifier for fibrotic vs healthy intracardiac bipolar EGM traces, with an ONNX export pipeline that bakes temperature calibration into the deployment graph.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac signal-processing toolkit.

---

## Why

Mapping atrial fibrosis from intracardiac EGMs is a binary classification problem at the per-trace level: does the bipolar waveform under this electrode pair look like a healthy activation (sharp, biphasic, high-slew) or a fibrotic one (fragmented, low-slew, lower amplitude)? A clinician answering that question voxel-by-voxel during an ablation procedure is the failure mode the myocard-labs project exists to remove — a calibrated per-trace classifier turns thousands of bipolar pairs into an automated substrate map that the ablation operator can act on directly.

`myocard-egm-classifier` is the consumer end of the myocard-labs intracardiac pipeline. It loads `ClassifierBank` HDF5 files produced by [`myocard-iafdb-pipeline`](https://github.com/myocard-labs/iafdb-pipeline) (real recordings) and [`myocard-synthetic-egm-pipeline`](https://github.com/myocard-labs/synthetic-egm-pipeline) (Finitewave simulations + IAFDB-noise overlay), trains a 1D MobileViT model with patient-aware splits, evaluates against held-out banks, and exports an ONNX file plus a typed [`egm_class_model_metadata.json`](https://github.com/myocard-labs/egm-contracts) sidecar that a C++/TensorRT runtime can deploy without re-implementing any of the pre- or post-processing.

What this repo does NOT do: HDF5 I/O ([`myocard-egm-data`](https://github.com/myocard-labs/egm-data) owns it), schema definitions ([`myocard-egm-contracts`](https://github.com/myocard-labs/egm-contracts) owns them), shared DSP primitives like bandpass or temperature scaling ([`myocard-egm-signal`](https://github.com/myocard-labs/egm-signal) owns them), or data production (the two producer repos own that). The split lets the classifier stay focused on training, evaluation, and deployment-side calibration against the project-wide contracts.

---

## Install

From source during pre-1.0 iteration:

```bash
# Base install — training + eval CLIs.
pip install "myocard-egm-classifier @ git+https://github.com/myocard-labs/egm-classifier.git"

# With the ONNX export toolchain (onnx + onnxruntime + onnxscript).
pip install "myocard-egm-classifier[onnx] @ git+https://github.com/myocard-labs/egm-classifier.git"
```

Editable install for development:

```bash
git clone https://github.com/myocard-labs/egm-classifier.git
cd egm-classifier
pip install -e ".[dev,onnx]"
pre-commit install
```

The runtime deps (`myocard-egm-contracts`, `myocard-egm-data[torch]`, `myocard-egm-signal`, `pydantic`, `torch`, `torchmetrics`, `numpy`, `tqdm`, `pyyaml`) come in transitively. The three myocard siblings are pinned to git tags during pre-1.0; drop the direct references once they publish to PyPI.

The `[onnx]` extra is kept optional so training-only images (Docker, K8s) don't pull ~200 MB of deployment toolchain. The `egm-class-export` script is installed regardless; if you invoke it without the extra, it'll catch the import failure at runtime and point you at the right `pip install` line.

---

## Quick start

End-to-end: train a model on a `ClassifierBank`, evaluate it on a held-out bank, export it as ONNX with temperature calibration baked in.

```bash
egm-class-train  examples/v1_baseline.yaml    # → checkpoints/v1_baseline/{best.pt, run.json, metrics.csv}
egm-class-eval   examples/v1_eval.yaml        # → predictions/v1_baseline_pred.classifier.h5 + stdout metrics
egm-class-export examples/v1_export.yaml      # → exports/v1_baseline/{best.onnx, best.model_metadata.json}
```

Three console scripts are installed:

| Command | Purpose |
|---|---|
| `egm-class-train` | Train a 1D MobileViT against a labeled `ClassifierBank` with patient-aware splits and AUROC-selected best-checkpoint saving. |
| `egm-class-eval` | Run sequential inference over a labeled or unlabeled bank, write a sibling `<stem>_pred.classifier.h5` (a stable `lpred_`/`upred_` id) with per-trace logits + probabilities + predictions; print the scalar metric bundle to stdout for labeled input. |
| `egm-class-export` | Fit a temperature scalar against a labeled calibration bank (optional), export the calibrated model to ONNX, write the typed metadata sidecar the C++ runtime consumes. |

All three are YAML-config-driven; argparse flags act as per-invocation overrides on top of the YAML. Full CLI reference and end-to-end walkthroughs in [`docs/usage.md`](docs/usage.md). Three pre-written example configs covering the v1 baseline pipeline live under [`examples/`](examples/).

Every tracked artifact carries a stable cross-artifact id — the training run (`run_id`), the exported model (`model_id`), and the eval predictions bank (`lpred_`/`upred_`) — all derived from a single `output.run_name` descriptor, so a downstream provenance index can trace what produced what. See [`docs/usage.md`](docs/usage.md) §"Stable artifact IDs".

---

## Programmatic usage

The model + checkpoint loader are importable when you'd rather drive inference from a notebook or a custom script:

```python
import torch
from myocard_egm_classifier.cli._common import load_checkpoint_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ckpt = load_checkpoint_model("checkpoints/v1_baseline/best.pt", device)

# model is a ready-to-use MobileViT1D in eval mode.
# ckpt is the torch.load dict — model_meta, train_config, val_metrics, ...

# Apply your own preprocessing (bandpass, z-score, pad/crop to 512) before forward:
with torch.no_grad():
    logit = model(x_batch_of_shape_B_1_T)  # [B, 1]
    p_fibrotic = torch.sigmoid(logit)
```

See [`docs/usage.md`](docs/usage.md) §"Programmatic use" for the predictions-bank reading recipe and [`docs/onnx_deployment.md`](docs/onnx_deployment.md) for the full Python ONNX-inference reference implementation (~75 lines, exercises the same preprocessing the C++ runtime should apply).

---

## Tests

```bash
pytest                  # full suite
pytest --cov            # with coverage
ruff check .            # lint
ruff format --check .   # format check
mypy                    # type check
```

CI runs the same checks on Python 3.10, 3.11, and 3.12 — see `.github/workflows/ci.yml`. The ONNX export tests build a tiny `MobileViT1D` end-to-end against `onnxruntime` so the parity check (PyTorch ↔ ONNX) runs in CI without GPU.

---

## Project status

This package is part of the in-progress [myocard-labs](https://github.com/myocard-labs) refactor. Pre-1.0 — expect breaking changes across minor versions until the schemas + CLI configs stabilise. The current release is `v0.4.0` (stable cross-artifact ids on every output, consuming the egm-contracts v0.5.x linkage schemas) and pins `myocard-egm-contracts v0.5.1`, `myocard-egm-data[torch] v0.4.0`, and `myocard-egm-signal v0.2.0`. See [`project/roadmap.md`](project/roadmap.md) for what's planned (zero2one normalization in train/eval, `run.json` as the export-config source, the activation-peak-anchoring investigation, dynamo ONNX exporter migration) and [`project/architecture.md`](project/architecture.md) for the design rationale.

---

## Citation

If you use this software in academic work, please cite both the MobileViT architecture + this classifier:

```bibtex
@inproceedings{mehta2022mobilevit,
  author    = {Mehta, Sachin and Rastegari, Mohammad},
  title     = {MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://arxiv.org/abs/2110.02178},
}

@software{klein_myocard_egm_classifier_2026,
  author  = {Klein, Daniel},
  title   = {myocard-egm-classifier: 1D MobileViT classifier for intracardiac bipolar EGMs with ONNX export and baked-in temperature calibration},
  year    = {2026},
  url     = {https://github.com/myocard-labs/egm-classifier},
}
```

The temperature-scaling post-hoc calibration follows Guo et al. 2017 (ICML); the full mathematical reference list, including the optimization + calibration citations, lives in [`docs/theory.md`](docs/theory.md) §7.

---

## License

MIT — see [LICENSE](LICENSE). Attribution requirements for upstream dependencies are listed in [NOTICE](NOTICE).
