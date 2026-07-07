# Theory — myocard-egm-classifier

The math + paper references for every metric, knob, and algorithm
the package ships, plus a consolidated troubleshooting guide for
the failure modes that show up during training, evaluation, or
deployment. The "I'm tuning a new model and want to understand why
a number looks the way it does" doc — sections §1–§5 describe each
piece of the pipeline from its own point of view, §6 turns that
around and addresses symptoms.

This doc deliberately stays on the **math + operational** side.
Plot-reading and visual-interpretation guidance (how to read a
reliability diagram, axis conventions, common misreadings) lives
in `egm-viewer/docs/usage.md` (still to be written; tracked
alongside the rest of the post-refactor work) and will cross-link
back here for derivations.

For the executable-side picture (CLIs, YAML schemas, walkthroughs)
see `docs/usage.md`; for the deployment-side picture (ONNX graph
contract + metadata sidecar) see `docs/onnx_deployment.md`; for
the internal package architecture (why three CLIs, where each
artifact lives) see `project/architecture.md`.

## Table of contents

- [Notation](#notation)
- [1. Preprocessing](#1-preprocessing)
  - [1.1 Bandpass filter](#11-bandpass-filter)
  - [1.2 Per-trace normalization](#12-per-trace-normalization)
  - [1.3 Activation-peak anchoring](#13-activation-peak-anchoring)
  - [1.4 Length conditioning](#14-length-conditioning)
- [2. Model architecture](#2-model-architecture)
  - [2.1 Why a CNN+transformer hybrid](#21-why-a-cnntransformer-hybrid)
  - [2.2 Building blocks](#22-building-blocks)
  - [2.3 Stochastic depth (DropPath)](#23-stochastic-depth-droppath)
  - [2.4 Width multiplier — the current scaling strategy](#24-width-multiplier--the-current-scaling-strategy)
  - [2.5 Single-logit binary head vs softmax](#25-single-logit-binary-head-vs-softmax)
- [3. Training knobs](#3-training-knobs)
  - [3.1 Loss](#31-loss)
  - [3.2 Optimizer — AdamW](#32-optimizer--adamw)
  - [3.3 LR schedule — linear warmup + cosine decay](#33-lr-schedule--linear-warmup--cosine-decay)
  - [3.4 Gradient clipping](#34-gradient-clipping)
  - [3.5 Mixed precision (AMP)](#35-mixed-precision-amp)
  - [3.6 Augmentation](#36-augmentation)
  - [3.7 Patient-aware splitting](#37-patient-aware-splitting)
- [4. Post-processing](#4-post-processing)
  - [4.1 Sigmoid + threshold](#41-sigmoid--threshold)
  - [4.2 Temperature scaling](#42-temperature-scaling)
- [5. Eval metrics](#5-eval-metrics)
  - [5.1 AUROC](#51-auroc)
  - [5.2 Precision, recall, F1, accuracy](#52-precision-recall-f1-accuracy)
  - [5.3 Confusion matrix](#53-confusion-matrix)
  - [5.4 ECE + reliability bins](#54-ece--reliability-bins)
- [6. Troubleshooting](#6-troubleshooting)
  - [6.1 Train AUROC ≫ val AUROC by a wide margin](#61-train-auroc--val-auroc-by-a-wide-margin)
  - [6.2 Training loss plateaus mid-run](#62-training-loss-plateaus-mid-run)
  - [6.3 Loss spikes or NaN gradients](#63-loss-spikes-or-nan-gradients)
  - [6.4 AUROC healthy but accuracy / F1 bad](#64-auroc-healthy-but-accuracy--f1-bad)
  - [6.5 ECE > 0.10 (poor calibration)](#65-ece--010-poor-calibration)
  - [6.6 Reliability diagram looks ragged](#66-reliability-diagram-looks-ragged)
  - [6.7 Suspicious zero in the confusion matrix](#67-suspicious-zero-in-the-confusion-matrix)
  - [6.8 Augmentation appears to hurt validation](#68-augmentation-appears-to-hurt-validation)
  - [6.9 Deployed model degrades over time](#69-deployed-model-degrades-over-time)
- [7. References](#7-references)

## Notation

- `T` — number of samples per trace (default 512).
- `fs_hz` — sample rate in hertz (default 1000.0).
- `x ∈ ℝᵀ` — a single per-trace input.
- `X ∈ ℝ^(B×C×T)` — a batch of `B` traces.
- `C` — channel count; `C = 1` for the v1 bipolar single-channel head.
- `z` — raw model output for one trace (a logit; scalar for the binary head).
- `p = σ(z) = 1 / (1 + e⁻ᶻ)` — sigmoid-derived probability of the positive class.
- `y ∈ {0, 1}` — ground-truth label. `0 = healthy`, `1 = fibrotic` (`HEALTHY_LABEL` / `FIBROTIC_LABEL`).
- `τ` — decision threshold for the hard prediction `ŷ = 𝟙[p ≥ τ]`.
- `T_scalar` (or just `T` in §4.2) — temperature-scaling scalar; context makes the meaning unambiguous.

---

## 1. Preprocessing

The deployment-time preprocessing pipeline — what runs on each
incoming trace before the model sees it — is the *bandpass filter*,
*per-trace normalization*, and *length conditioning* (pad/crop to
the trained `input_length`). The training pipeline does the same
three steps (plus optional augmentation, §3.6) on every train /
val / test trace, so the inference path mirrors training exactly.

The `egm_class_model_metadata.json` sidecar records every constant
of the deployed pipeline (`expected_fs_hz`, `bandpass_hz`,
`normalization.scheme`, `expected_trace_samples`) so the C++
runtime applies the same transform a Python eval would apply. See
`docs/onnx_deployment.md` §"Runtime inference pipeline" for the
deployment-side code path; this section is the *why* behind the
constants in that sidecar.

### 1.1 Bandpass filter

A bipolar EGM is the voltage difference between two closely-spaced
intracardiac electrodes. The clinical band of interest for atrial
activation analysis is roughly **30–250 Hz**: the high-frequency
content of the activation wavefront passing under the electrode
pair. Outside that band:

- **Below ~30 Hz** sits the slow far-field component (ventricular
  far-field, baseline wander from breathing + catheter motion),
  which doesn't carry local activation information and contaminates
  any amplitude-based feature.
- **Above ~250 Hz** sits electrical noise from the recording
  hardware, switching power supplies, and (in vivo) stimulator
  artifacts.

The bandpass is therefore a fixed deployment-time constant
(`preprocessing.bandpass_hz` in the metadata sidecar) — not a model
input, not a per-trace knob. It must match what training saw or
the input distribution shifts and the model degrades silently.

**Effects of a mismatched bandpass at deployment.** When the
runtime applies a different band than the training-time band, the
model sees an input distribution it was never trained on. Three
common mismatch directions and what each does:

- *Upper edge too low* (e.g. runtime 30–100 Hz vs trained
  30–250 Hz): the wavefront's high-frequency content is filtered
  out before the model sees it. Fragmented / low-slew morphology
  (the v1 fibrotic signal) gets smoothed away — expect false
  negatives on the fibrotic class.
- *Lower edge too high* (e.g. runtime 100–250 Hz vs trained
  30–250 Hz): legitimate low-frequency components of the
  activation are removed. Less of a directional class skew than
  the upper-edge case; expect AUROC to drop without a predictable
  per-class pattern.
- *Lower edge too low* (e.g. runtime 5–250 Hz vs trained
  30–250 Hz): far-field contamination leaks back in. The model
  sees baseline wander it never saw at training and can
  mis-classify quiescent segments as activations.

Reference for the 30–250 Hz band: see the
`reference_voltage_thresholds` notes in `project/` and Marchlinski
et al. (2000); the band is the de-facto standard rather than a
single-paper claim.

(See also §6.9 "Deployed model degrades over time" — bandpass
mismatch from a hardware-side filter change is the most common
upstream cause.)

### 1.2 Per-trace normalization

The metadata schema admits three normalization schemes for the
deployment runtime:

| Scheme | Per-trace transform | Removes | When to use |
|---|---|---|---|
| `zscore` (v1 default) | `(x - mean(x)) / std(x)` | per-trace amplitude **and** offset | When the model should rely on waveform morphology, not raw amplitude. |
| `zero2one` | `(x - min(x)) / (max(x) - min(x))` | per-trace dynamic range, preserves shape | When the model should see normalized morphology but absolute peak structure matters. |
| `none` | identity | nothing | Rare. Only when the upstream hardware already calibrates traces into a fixed physical-unit range. |

All three are computed **per trace** (along the time axis, with
`keepdim=True` broadcasting) — not per dataset and not per channel,
because intracardiac amplitudes vary by an order of magnitude
between patients and catheter contact conditions, and a
dataset-level statistic would mis-scale most traces. The divisor
gets a strictly positive floor (`normalization_eps`) so a
degenerate flat trace (constant signal) doesn't blow up.

**Why `zscore` as the v1 default.** Bipolar EGM amplitudes carry
information about electrode contact and tissue proximity, but the
fibrosis-vs-healthy question is fundamentally morphological — a
fragmented, low-slew wavefront looks different from a sharp,
high-amplitude one *at any amplitude scale*. Z-scoring removes a
nuisance variable that doesn't generalize across patients.

**Why `zero2one` is shipped as an option.** Z-score collapses
amplitude differences between two otherwise-identical waveforms; if
a future architecture wants to recover some amplitude information
(e.g., a multi-input model where one branch sees normalized
morphology and another sees a raw amplitude scalar), `zero2one`
preserves the peak-to-trough structure within the trace while
still controlling per-trace scale. We ship it now because the
export CLI needs to handle both schemes when the training side
eventually grows `zero2one` support (tracked in `project/roadmap.md`).

**Why never per-channel mean/std arrays.** An earlier version of
the metadata schema carried per-channel `mean` + `std` arrays. We
dropped them in v0.4.0: for `C = 1` they're per-trace stats anyway
(redundant with the per-trace zscore), and for any future
multi-channel head the deployment-time bandpass + per-trace zscore
will still apply per channel — there's no scenario where a
dataset-level statistic is the right normalization.

### 1.3 Activation-peak anchoring

**Terminology note.** "R-wave anchoring" in the clinical
literature means detecting the **surface-ECG R-wave** (the
ventricular depolarization peak) and using it as a synchronization
marker for the cardiac cycle. It's useful for sinus-rhythm
intracardiac analysis where the cycle is roughly periodic and the
R-wave gives a reliable phase reference. It's much less useful in
AF — the ventricular response is decoupled from chaotic atrial
activity, so R-waves don't synchronize the atrial information
that the v1 classifier cares about.

What the v1 synthetic producer actually does is **activation-peak
anchoring** (a project-internal term to keep this distinction
clean): each trace is a fixed window of `T = 512` samples at
1 kHz cropped around the single simulated activation peak. The
Phase-1 single-activation simulation emits one biphasic peak as
the wavefront crosses the bipolar electrode pair; the crop centers
that peak in the trace.

**Why anchor in Phase 1.** The fibrotic-vs-healthy signal is
concentrated near the activation peak. An unanchored random-offset
window often catches only quiescent baseline, which carries almost
no information — unanchored training spends most of its capacity
learning "is there a wavefront here at all" rather than "is the
wavefront shape fibrotic." This is a project-internal heuristic;
there's no canonical published reference for the
activation-anchoring choice.

**Where this stops applying — Phase 2 multi-activation.** When
the synthetic producer grows multi-activation simulations
(multiple wavefronts crossing the electrode pair over the trace
window), single-peak anchoring no longer fits: cropping around one
peak discards the multi-activation context that the model would
need to learn from. Phase 2+ traces will likely be longer, span
many activations, and not be anchored on any one of them.

**Is anchoring helping the v1 classifier? Open question.** Today
the synthetic producer always anchors, so we don't have an A/B
comparison. Two follow-ups are tracked in the v0.2.0+ roadmap:

1. Make anchoring an opt-in producer-side flag in
   synthetic-egm-pipeline so the model can be trained on both
   anchored and unanchored variants of the same simulation.
2. Run an investigation comparing the two — does anchoring
   actually improve the v1 task, or is the network capable of
   learning the activation location on its own when given enough
   training samples?

The anchoring question doesn't seem to have a published answer in
the intracardiac EGM ML literature; the investigation is a
candidate for a small methods paper.

### 1.4 Length conditioning

The model has a fixed input length `T = expected_trace_samples`
(default 512 samples at 1 kHz = 512 ms per trace). Two reasons the
length is fixed at construction time:

1. The MobileViT-1D block requires `T` to be divisible by every
   block's `patch_size` along the depth chain (§2.2). The default
   architecture has patch sizes `4 → 2 → 2` at three depths;
   combined with the stride pattern (downsample 2× per stage),
   `T = 512` gives a sequence the deepest transformer can see as
   a length-8 patch sequence. Changing `T` requires re-validating
   every patch division.
2. A fixed `T` lets us deliver the model as a single ONNX graph
   with a static time axis (only the batch axis is dynamic). C++
   deployment with TensorRT works much better with static shapes.

At deployment the runtime pads / crops each incoming trace to
exactly `T` samples before invoking the model.

---

## 2. Model architecture

The model is a 1D adaptation of MobileViT (Mehta & Rastegari 2022),
chosen for the v1 EGM-classification task because it cleanly mixes
local (CNN) and global (transformer) representations at a parameter
budget that fits a ~2k-trace training set without overfitting.

### 2.1 Why a CNN+transformer hybrid

The two extreme alternatives, and why neither was the v1 choice:

- **Pure CNN (e.g. 1D ResNet).** Fast, well-understood, easy to
  train, but the receptive field at the bottom of the stack is
  bounded by `kernel_size × num_layers`. To attend to a feature
  spanning ~256 ms of a 512-ms trace, a pure CNN needs a stack
  deep enough that the receptive field reaches that far, which
  inflates parameter count past what a 2k-trace dataset can train
  cleanly.
- **Pure transformer (e.g. ViT-style on raw samples).** A
  length-512 sequence attending position-to-position scales as
  `O(T²)` and needs a large dataset to learn position-invariant
  low-level features from scratch. With ~2k training traces, a
  pure transformer overfits before it learns local edge / peak
  detectors.

The MobileViT hybrid uses MobileNetV2 inverted-residual blocks for
the local feature extraction (cheap, locality-preserving, ImageNet-
to-1D translation well understood) and inserts transformer blocks
between MV2 stages so the global mixing happens on already-pooled
feature maps — `T = 64 → 32 → 16` at the three MobileViT block
depths, which makes `O(T²)` attention cheap. The skip-into-fusion
inside the MobileViT block (§2.2) preserves the local CNN features
even after global attention runs, so the model gets both
representations.

Compared to the pure alternatives at the same parameter budget,
this architecture trains stably on 2k traces, has a receptive field
that covers the whole input by the deepest transformer block, and
admits the standard channel-width scaling (§2.4) without redesign.

### 2.2 Building blocks

Read top-to-bottom; each block consumes the previous block's
output shape.

**Conv stem (`conv_stem_1d`).** A single `Conv1d(in=1, out=c₀,
kernel=7, stride=2)` + BatchNorm1d + SiLU. Halves the time axis
(`T → T/2`) and lifts to the base channel width `c₀ = 16` (at
`width_multiplier=1.0`). Standard ImageNet-style stem, 1D version.

**MobileNetV2 1D inverted-residual block (`mv2_1d`).** The
MobileNetV2 (Sandler et al. 2018) block, 1D'd:

```
x ─→ 1x1 Conv1d (C → C*t) → BN → SiLU  (channel expand)
   → depthwise Conv1d (kernel=3, stride=s) → BN → SiLU
   → 1x1 Conv1d (C*t → C_out) → BN     (channel project, no act)
   → + x   if stride == 1 and C == C_out (residual)
```

`t = 4` is the expand ratio (Sandler §3.3). The middle depthwise
conv is where temporal mixing happens cheaply; the 1×1 convs do
channel mixing. Stride-2 variants downsample `T → T/2` and bump
channel width; stride-1 variants preserve shape with a residual.

**MobileViT-1D block (`mobilevit_1d`).** The headline block; the
1D translation of MobileViT (Mehta & Rastegari §3.2). Five stages
on input `x ∈ ℝ^(B×C×T)`:

1. **Local rep conv.** `Conv1d(C, C, kernel=3)` + BN + SiLU.
   Standard local feature extraction.
2. **Channel lift.** `Conv1d(C, d, kernel=1)` (no activation; LN
   comes next). Lifts to the transformer dimension `d`.
3. **Unfold + L transformer layers + Fold.**
   ```
   x [B, d, T] → reshape T=(N·p) → [B, d, N, p]
              → permute            → [B, p, N, d]
              → reshape            → [B·p, N, d]
   transformer × L on the N axis (each row is a "patch sequence")
   inverse permute + reshape       → [B, d, T]
   ```
   The unfold groups **consecutive** `p` timesteps into a patch
   (so each patch is a short temporal window, like adjacent pixels
   in the 2D version). Each transformer layer then attends across
   the `N = T/p` patches at every intra-patch position. This is
   the patch-attention trick from PatchTST (Nie et al. 2023)
   applied inside MobileViT.
4. **Channel project.** `Conv1d(d, C, kernel=1)` + BN + SiLU.
   Brings the channel count back to `C`.
5. **Fusion.** `concat(x_input, x_global) → Conv1d(2C, C, kernel=3)`
   + BN + SiLU. This is the locality-preservation step — the
   original input flows around the transformer and is recombined
   with the attention-mixed features, so no temporal structure is
   lost to the attention bottleneck.

Patch sizes are `4 → 2 → 2` at the three MobileViT depths (deeper
blocks see shorter sequences, so a smaller patch keeps the patch
count above the attention floor). `T` must be divisible by `p` at
every block (validated inside `_unfold`); the canonical T=512 +
stride pattern guarantees this for the default architecture.

The internal transformer is the standard pre-LN encoder block
(Vaswani 2017, ViT pre-LN ordering from Dosovitskiy 2021): LN →
MHSA → DropPath → residual; LN → MLP (GELU, hidden = `mlp_ratio · d`,
`mlp_ratio = 2.0` for MobileViT) → DropPath → residual. `d` must
be divisible by `num_heads = 4` so per-head dim is integer.

**Classification head (`head_1d`).** `Conv1d(C, e, kernel=1)` + BN
+ SiLU (channel expand to `e = 320`), then global average pool
over time, then `Dropout(p_head) → Linear(e, num_outputs)`.
`num_outputs = 1` for the binary head (BCE); `num_outputs = K` for
a future multi-class softmax head.

### 2.3 Stochastic depth (DropPath)

Stochastic depth (Huang et al. 2016) randomly skips entire
residual branches at training time with per-block probability
`p_drop`. At inference all branches run normally — it's only a
training-time regularizer.

We schedule `p_drop` linearly from `0` at the conv stem to
`stochastic_depth` (config default `0.1`) at the final block, so
shallow blocks are never dropped and deep blocks (the ones with
larger receptive fields and more capacity to overfit) get the most
regularization. The schedule is applied uniformly inside each
MobileViT block too — the same per-block `drop_path` value
propagates into both the attention and MLP residual gates of every
transformer sub-layer.

**Effect of raising `stochastic_depth`.** Stronger regularization
of the deep blocks. Useful when the model has the capacity to
memorize the training set; typical sweep range is `0.1 → 0.3`.
Beyond `~0.3` the regularization starts to dominate and
underfitting risk grows.

### 2.4 Width multiplier — the current scaling strategy

The `width_multiplier` knob (Howard et al. 2017, MobileNet §3.2)
uniformly scales every block's channel count by a single scalar.
The base widths `(c₀, …, c₅) = (16, 32, 48, 64, 80, 96)` are the
MobileViT-XS sizing — a conservative middle ground that trains
cleanly on the ~2k-trace Phase-1 banks. Common settings:

| `width_multiplier` | Param count (approx) | When to use |
|---|---|---|
| `0.5` | ~0.6 M | Quick architecture sanity checks. |
| `1.0` (default) | ~1.6 M | v1 baseline. |
| `1.5` | ~3.5 M | Once the training set grows past ~10k traces. |

**Why uniform width scaling is the current strategy.** Uniform
scaling is a single-knob proxy for "make the whole network bigger
or smaller." Its advantages over per-layer tuning:

- One number tunes the whole network — easily comparable across
  runs.
- Preserves the architecture's structural ratios (transformer dim
  vs MV2 channel count, head expansion vs feature width) which
  are calibrated in the published MobileViT-XS / S / M sizings.
- Doesn't touch patch sizes or strides, so the patch-divisibility
  constraints from §2.2 stay satisfied without manual
  re-validation.

**Limitations.** Uniform scaling is a strict simplification — it
assumes every layer wants the same fractional capacity. For an
EGM-specific architecture the optimum is unlikely to be
MobileViT-XS-shaped. Plausibly the deepest transformer block wants
a larger `transformer_dim` (it's the only block doing global
mixing) and the early MV2 stages want less channel width (they're
processing short receptive fields with a lot of redundancy).
Single-knob scaling can't express any of that.

**The future direction.** Per-layer architecture tuning — initially
manual sweeps over the registry-driven block list, and eventually a
small architecture search — is on the post-v0.1.0 roadmap. The
pros / cons trade-off:

|  | Uniform width multiplier (today) | Per-layer architecture tuning (future) |
|---|---|---|
| Tunable knobs | 1 | dozens (per-block channels, transformer dims, depths, patches) |
| Run cost per evaluation | one training run | one training run per candidate config |
| Risk of overfitting the search to a small dataset | low (one knob) | high; needs disciplined held-out test |
| Comparability across studies | easy (cite multiplier) | requires reporting the full block list |
| Calibrated by | published MobileViT ratios | empirically by us against EGM data |

The v1 release pins on uniform scaling because the dataset isn't
yet big enough to support a meaningful architecture search; once
the training-set growth in Phase 1.5+ stabilizes, per-layer tuning
becomes worth the investment. The block list is already
registry-driven (see `models/registry.py`), so the architecture-
search step doesn't require restructuring the model code — just a
sweep harness around the YAML's `model.blocks` field.

### 2.5 Single-logit binary head vs softmax

The v1 model emits a single logit `z ∈ ℝ` per trace; we apply
`σ(z)` for the positive-class probability and threshold at
`decision.threshold` (default `0.5`) for a hard prediction.
Equivalent to a two-class softmax head trained with cross-entropy
in the limit, but with two practical advantages:

- **One output, one threshold.** A single decision threshold maps
  directly to a sweepable knob (precision-recall trade-off)
  without having to reason about which softmax output gets the
  threshold.
- **`pos_weight` works cleanly.** The BCE-with-logits loss accepts
  a scalar `pos_weight` that re-balances the positive-class
  gradient contribution (§3.1). The softmax-CE equivalent (class
  weights) works but is more error-prone with multiple class
  outputs.

The model constructor *does* support `num_classes ≥ 2` for the
future `FibroticTypeLabel` multi-class head, in which case the
loss auto-switches to cross-entropy. The training + eval CLIs
assume the v1 binary path and the export CLI rejects multi-class
checkpoints — promoting multi-class to a default is a Phase-2
scope decision (roadmap §"Won't-do").

---

## 3. Training knobs

Defaults are calibrated for the v1 task (2k traces, 60 epochs,
batch 64, 1 kHz × 512 ms inputs) — change them deliberately, not by
copy-paste from a paper that trained on ImageNet.

### 3.1 Loss

**Binary head (default).** `BCEWithLogitsLoss` — operates on raw
logits and applies a numerically-stable sigmoid internally:

```
L_BCE = (1/N) · Σᵢ [ -yᵢ · log σ(zᵢ) - (1 - yᵢ) · log(1 - σ(zᵢ)) ]
       = (1/N) · Σᵢ [ log(1 + exp(zᵢ)) - yᵢ · zᵢ ]
```

with `zᵢ` the per-trace logit and `yᵢ ∈ {0, 1}` the label. The
second form is what's actually computed (via `logaddexp(0, z)`) so
the loss stays finite for extreme logits.

**Class imbalance handling.** If the bank is K-to-1 positive-vs-
negative imbalanced, the BCE loss can be re-weighted to up-weight
the positive class:

```
L_BCE_weighted = (1/N) · Σᵢ [ -w · yᵢ · log σ(zᵢ) - (1 - yᵢ) · log(1 - σ(zᵢ)) ]
```

where `w = pos_weight`. Set `pos_weight = N_neg / N_pos` to make
the per-class gradient contributions roughly equal. The
`train.pos_weight` YAML knob plumbs this through; leaving it
`None` (the v1 default) treats the loss symmetrically and lets the
model learn whatever the data distribution suggests.

**Multi-class head (Phase 2).** `CrossEntropyLoss` auto-applies a
log-softmax then NLL; class weights would be the multi-class
equivalent of `pos_weight`. Not yet exercised end-to-end.

### 3.2 Optimizer — AdamW

`torch.optim.AdamW` with `lr = 3e-4`, `weight_decay = 0.05`, `betas
= (0.9, 0.999)`. AdamW (Loshchilov & Hutter 2019) differs from
Adam in *where* the weight decay is applied: Adam folds the L2
penalty into the gradient (so the effective decay rate couples to
the adaptive learning rate), AdamW applies the decay directly to
the parameter values after the adaptive step. For transformer-
containing architectures (where Adam-style L2 mis-regularizes) this
matters.

Why AdamW vs SGD-with-momentum: the transformer sub-layers respond
poorly to SGD without elaborate LR-schedule + warmup tuning; AdamW
trains the whole network stably out of the box at the cost of a
modest constant memory overhead (the optimizer state's first +
second moment).

**Sensitivity of `weight_decay`.** `0.05` is the MobileViT-default
value (Mehta 2022) and works for the v1 baseline at
`width_multiplier = 1.0`. Smaller models tolerate less
regularization: at `width_multiplier = 0.5` drop `weight_decay`
to ~`0.01`, otherwise underfitting can look like a bad architecture
choice.

### 3.3 LR schedule — linear warmup + cosine decay

The implemented schedule is:

```
lr(step) = base_lr · step / warmup_steps                       if step < warmup_steps
        = min_lr + (base_lr - min_lr) · 0.5 · (1 + cos(π · progress))  otherwise

where progress = (step - warmup_steps) / (total_steps - warmup_steps)
```

with `warmup_frac = 0.05` (5% of total steps), `base_lr = 3e-4`,
and `min_lr = 0.0` (cosine decays to zero by the final step).

**Why warmup.** AdamW's second-moment estimate is undefined for
the first few steps (you're dividing the mean by the square-root
of itself). Warmup gives the second moment time to populate from
real gradient statistics before the LR ramps to its base value.
Skipping warmup typically shows up as the first epoch of training
looking fine, then loss spiking around step 50–100 as the LR hits
full value with stale optimizer state.

**Why cosine decay over step-decay.** Cosine annealing (Loshchilov
& Hutter 2017) decays smoothly so the model spends more of its
training near the eventually-converged LR, rather than stepping
abruptly between LR levels. For 60-epoch runs the smooth schedule
beats step-decay in practice; for very long runs (>200 epochs) the
difference vanishes.

### 3.4 Gradient clipping

Global L2-norm clipping at `grad_clip_norm = 1.0`: rescale the
full parameter gradient vector to L2 norm ≤ `clip_norm`. Cheap
insurance against the loss-spike-and-NaN failure mode that hits
transformer training when an outlier batch produces a runaway
gradient. Set to `None` to disable; leave at `1.0` unless
profiling shows it's biting on every step (the clip rate should be
a small percentage of steps in a healthy run).

### 3.5 Mixed precision (AMP)

`torch.amp.autocast` lets the forward pass run in lower precision
(`bf16` or `fp16`) while keeping the optimizer state + master
weights in `fp32`. Two halves of memory, ~1.5× speedup on Ampere+
GPUs, and (for `bf16`) no loss-scaling required.

| Setting | Use when |
|---|---|
| `amp: false` (v1 default) | CPU runs, or when debugging numerical issues. |
| `amp: true, amp_dtype: bf16` | Ampere/Hopper+ GPUs. Recommended for production training. |
| `amp: true, amp_dtype: fp16` | Pre-Ampere GPUs (V100, T4). Requires loss-scaling (`GradScaler`); pipeline wires this in automatically when fp16 is selected on CUDA. |

**Why `bf16` is preferred when available.** `bf16` has the same
exponent range as `fp32` (8 bits) so it can't underflow on small
gradients the way `fp16` can; only the mantissa is shorter
(7 bits). For transformer training where attention scores have a
wide dynamic range, `bf16` is meaningfully more stable than
`fp16`.

### 3.6 Augmentation

Train-split only (`augment_train: true`); val + test see clean
traces. Two augmentations, both in `myocard-egm-data`'s
`TraceTransform`:

- **Random gain.** Multiply the trace by a scalar
  `a ∼ Uniform(1 - g, 1 + g)`, with `g = max_gain` per the YAML
  (v1 default `0.0` — a no-op under zscore normalization, since
  zscore is gain-invariant).
- **Random time-shift.** Roll the trace by an integer offset
  `k ∼ Uniform(-s · T, +s · T)`, with `s = max_shift_frac` (v1
  default `0.10`, ±10% of `T`). Wrap-around on the boundaries
  rather than zero-pad so the trace length stays exactly `T`.

**Interaction with anchoring.** When the producer-side
activation-peak anchoring (§1.3) is on, time-shift augmentation
encourages the model not to overfit to "wavefront at exactly
sample T/2." If the producer-side peak detection is off by tens of
milliseconds, large shifts can push the activation past the trace
boundary and clip it; see §6.8 for the symptom.

**Why no time-flip (reverse).** EGM activation waveforms have a
canonical temporal direction (depolarization rises then falls); a
time-flipped trace is morphologically wrong and would teach the
model an invariance that doesn't exist in the deployment data.
Same reasoning rules out time-warp (would distort the morphology
the model is trying to classify on).

**Why no additive noise in v1.** Realistic recording noise is
added at *producer* time, not training time. The
synthetic-egm-pipeline's mixer step injects noise from the
IAFDB-derived noise bank into the simulated activations before the
classifier bank is written — by the time the classifier sees a
trace, the noise distribution already matches what the deployment
runtime will see. Adding a second layer of training-time additive
noise on top would shift that distribution off-target.

**Whether training-time additive noise should be revisited is an
open question.** Some groups working on synthetic-data ML
pipelines use training-time additive noise as a regularizer even
when the producer already adds noise. A literature survey for this
specifically in the synthetic-EGM ML context is on the Phase 1.5
backlog (tracked in `project/roadmap.md`); if the surveyed groups
report a consistent benefit, an A/B test in the v1 classifier
becomes worth running.

### 3.7 Patient-aware splitting

The split into train/val/test is **patient-aware**: all traces
from a single patient (or, for synthetic data, a single `sim_id`)
land in exactly one of the three splits. The alternative — random
trace-level splitting — leaks information: traces from the same
heart share noise, scar geometry, and catheter contact biases
that the model can memorize and recover at test time, inflating
val metrics in a way that doesn't transfer to a held-out patient.

The implementation lives in `myocard-egm-data` as
`patient_aware_split` with a pluggable **stratification strategy**:

| Strategy | What it stratifies on | Use when |
|---|---|---|
| `any_positive` (v1 default) | Patient label = OR over the patient's trace labels (1 if any trace is fibrotic, else 0). | Binary classification with sparse positives. Keeps the positive-patient ratio comparable across splits. |
| `binned_density` | Patient bucket from the patient-level fibrotic-trace fraction binned into `n_bins`. | When the positive density per patient is meaningful (e.g., "low / medium / high fibrosis burden") and should be balanced across splits, not just "any positive." |

Both strategies first group traces by patient, compute the
patient-level stratification target, then run a standard
stratified split on patient IDs. The split is reproducible from
`split_seed` (default 0).

**Caveat — small patient counts.** Stratified splitting can fail
outright when there aren't enough patients per stratum (e.g.
fewer than 5–10 total patients in a study). In that regime,
K-fold cross-validation over patients is the right tool — the
egm-classifier CLI doesn't ship a K-fold mode today; treat it as
a one-off script.

References: patient-aware splitting is a standard requirement in
medical imaging ML (e.g. Park & Han 2018); the specific patient-
level stratification strategies here are pragmatic choices, not a
named technique.

---

## 4. Post-processing

How raw logits become deployment-ready predictions.

> **About this section:** the math + operational guidance for
> sigmoid + threshold + temperature scaling lives here. The
> visual-interpretation half (how to read a reliability diagram,
> what a "well-calibrated S-curve" looks like, common visual
> misreadings) lives in `egm-viewer/docs/usage.md` (still to be
> written); it'll cross-link back here for the math.

### 4.1 Sigmoid + threshold

The model emits a logit `z`; the binary probability is `p = σ(z)
= 1 / (1 + e⁻ᶻ)`; the hard prediction at threshold `τ` is
`ŷ = 𝟙[p ≥ τ]` with `τ = decision.threshold` from the metadata
sidecar (v1 default `0.5`).

**Why ship raw logits in the predictions bank.** The eval CLI
stamps `pred_logits` (not just `label_prob`) on every trace so
downstream analysis can re-apply any threshold without re-running
inference. A precision-recall sweep is then a pure numpy operation
over the predictions bank, not a re-eval.

**Choosing `τ` for deployment.** `τ = 0.5` is the right default
under symmetric BCE training on a balanced dataset. Under
asymmetric costs (e.g., false-negative is 10× worse than false-
positive for an ablation-guidance use case where missing fibrosis
is catastrophic), pick `τ < 0.5` to favor recall. Sweep the
validation predictions bank and pick the `τ` that maximizes your
operational cost function — the metadata-side `decision.threshold`
records the chosen value so deployment matches.

### 4.2 Temperature scaling

Modern neural networks (especially BCE-trained ones) tend to be
**overconfident**: the probability they assign to their predicted
class is systematically higher than the empirical frequency of
being correct at that probability level (Guo et al. 2017, ICML).
Temperature scaling is the simplest post-hoc fix: pick a single
positive scalar `T` and replace `p = σ(z)` with `p = σ(z / T)`.
Properties:

- **Monotonic in logits.** `σ(z/T)` preserves the ordering of
  samples by logit, so AUROC and rank-based metrics are
  *unchanged*. Only calibration-quality metrics (ECE, reliability)
  and any threshold-dependent decision at `τ ≠ 0.5` actually
  shift.
- **One parameter.** `T > 1` softens overconfident predictions;
  `T < 1` sharpens underconfident ones; `T = 1` is identity.
- **Convex NLL.** The fit objective is convex in `1/T` for
  typical logit distributions, so a bounded scalar minimizer finds
  the optimum reliably.

**Fitting `T`.** The export CLI's calibration step minimizes
binary NLL on a held-out labeled bank:

```
T̂ = argmin_T  (1/N) · Σᵢ [ log(1 + exp(zᵢ / T)) - yᵢ · zᵢ / T ]
```

`scipy.optimize.minimize_scalar` with `method='bounded'` over
`bounds = (0.05, 20.0)` (the bracket exists so a numerical
pathology — say, all-zero logits — produces a clear error rather
than a silent extreme). The fitted optimum for a healthy v1 model
typically lands in `[1.0, 3.0]`.

**Why bake `T` into the ONNX graph.** The deployment graph emits
`logits / T` directly (`CalibratedModel.forward = base(x) / T`
with `T` as an `nn.Buffer` — see `export/graph_wrapper.py`). This
means the C++ runtime gets calibrated logits at no per-inference
cost, and there's no risk of a deployment-side reimplementation
drifting out of sync with the fitted `T`. The metadata sidecar's
`training_provenance.calibration_temperature` still records `T`
explicitly so audit tooling can answer "what was the fitted
temperature for this export?" without re-parsing the ONNX graph.

**Skip-calibration semantics.** `--skip-calibration` (or omitting
the `calibration.bank` field) bakes `T = 1.0` into the graph
(identity) and records `calibration_temperature = 1.0` in the
sidecar. The metadata schema requires a numeric value, so the
audit trail distinguishes "calibrated to 1.0" from "calibration
skipped" only via the absence of `calibration_bank_path`.

Reference: Guo C, Pleiss G, Sun Y, Weinberger KQ. *On Calibration
of Modern Neural Networks.* ICML 2017. arXiv:1706.04599.

---

## 5. Eval metrics

`binary_metrics` (in `src/myocard_egm_classifier/metrics.py`)
returns the full bundle: `accuracy`, `precision`, `recall`, `f1`,
`auroc`, `ece`, `confusion` (tp/fp/tn/fn), `reliability` (per-bin
table), and `n`. The trainer projects the scalars into
`EpochRecord.val_metrics` per epoch; the eval CLI prints the same
dict to stdout. Implementations are torchmetrics primitives.

> **About this section:** the math + operational guidance for each
> metric lives here. Plot-reading conventions for the corresponding
> visualizations (ROC curves, PR curves, reliability diagrams,
> confusion matrices) live in `egm-viewer/docs/usage.md` (still to
> be written); it'll cross-link back here for derivations.

The full set of confusion counts at threshold `τ` is:

```
TP = Σᵢ 𝟙[pᵢ ≥ τ] · 𝟙[yᵢ = 1]    (true positives)
FP = Σᵢ 𝟙[pᵢ ≥ τ] · 𝟙[yᵢ = 0]    (false positives)
TN = Σᵢ 𝟙[pᵢ < τ] · 𝟙[yᵢ = 0]    (true negatives)
FN = Σᵢ 𝟙[pᵢ < τ] · 𝟙[yᵢ = 1]    (false negatives)
```

with `N = TP + FP + TN + FN`. Every threshold-dependent metric
below is a function of these four counts at the configured `τ`.

### 5.1 AUROC

The ROC curve plots `TPR = TP / (TP + FN)` against `FPR = FP /
(FP + TN)` as `τ` sweeps from 1 to 0. AUROC is the area under
that curve. Equivalent definition: AUROC is the probability that a
uniformly-random positive sample receives a higher logit than a
uniformly-random negative sample (Mann-Whitney U).

Properties:

- **Threshold-free.** The metric sweeps all thresholds, so a model
  with bad calibration but good ranking still gets a high AUROC.
- **Class-imbalance-robust** in the sense that AUROC doesn't
  collapse when the positive class is rare (unlike accuracy). It
  *does* shift slightly with imbalance — a model classifying a
  10:1 imbalanced set isn't directly comparable to one classifying
  a 1:1 set — so always report alongside the class counts.
- **Range `[0, 1]`.** `0.5` = random, `1.0` = perfectly separable.

We use AUROC as the **checkpoint-selection metric** during
training (`select_metric: auroc`) for the threshold-free +
imbalance-robust properties: when the train/val/test split has
slight class-balance drift (a hazard of patient-aware splitting on
small datasets), AUROC is the most stable per-epoch signal.

### 5.2 Precision, recall, F1, accuracy

All four are threshold-dependent functions of the confusion counts:

```
Precision = TP / (TP + FP)                   "of the positives I called, how many were right"
Recall    = TP / (TP + FN)                   "of the actual positives, how many did I catch"
F1        = 2 · P · R / (P + R)              harmonic mean of precision + recall
Accuracy  = (TP + TN) / N                    fraction of correct predictions
```

Implementation note: torchmetrics' binary versions accept raw
logits + a threshold, apply sigmoid + threshold internally, and
compute the counts from there.

**When precision matters more than recall:** false positives are
expensive (e.g., flagging a healthy region as fibrotic and
ablating it). Push `τ` up.

**When recall matters more than precision:** false negatives are
expensive (e.g., missing fibrosis and leaving an ablation target
untreated). Push `τ` down.

**When F1 is the right summary:** balanced cost of FP and FN, and
you want one number to compare two models. F1 is the standard
"single summary that respects class imbalance." Note F1 weights
precision and recall equally; for asymmetric costs, use Fβ with
`β` chosen for the cost ratio (not exposed in v1; trivially added
as a sweep over the predictions bank).

**When accuracy is misleading:** under severe class imbalance.
Accuracy on a 99:1 imbalanced set is ≥ 99% for the constant
"always predict majority" classifier; precision + recall + AUROC
catch the same model as useless.

### 5.3 Confusion matrix

`{tp, fp, tn, fn}` at threshold `τ`. The raw counts that every
threshold-dependent metric above is built from. The bundle
includes the four counts as a sub-dict so downstream tooling
(notebook analysis, the viewer's confusion-matrix plot) can
recompute any threshold-dependent metric without re-eval.

### 5.4 ECE + reliability bins

**Expected Calibration Error.** A scalar summary of calibration
quality. Walk the predicted probabilities, bin them into `n_bins`
equal-width buckets `[0, 1/n_bins), [1/n_bins, 2/n_bins), …`. For
each bucket compute the mean predicted confidence and the mean
correctness; ECE is the weighted absolute difference:

```
ECE = Σ_b  (|S_b| / N) · | conf(S_b) - acc(S_b) |
```

where `S_b` is the set of samples in bucket `b`, `conf(S_b)` is
the mean predicted confidence over `S_b`, and `acc(S_b)` is the
fraction of correct predictions in `S_b`. v1 uses `n_bins = 10`
and the L1 norm (matches Guo §2). torchmetrics'
`BinaryCalibrationError` implements this directly.

ECE is "confidence in the predicted class" — for a sample where
the model predicted negative (probability < `τ`), the confidence
is `1 - p` (probability of the predicted class), not `p`. The
reliability bin builder in `metrics._reliability_bins` follows the
same convention.

Range `[0, 1]`. Well-calibrated models score `ECE ≈ 0`. A
heuristic guideline:

| ECE | Calibration quality |
|---|---|
| `< 0.05` | Well-calibrated. |
| `0.05 – 0.10` | Acceptable for most uses. |
| `> 0.10` | Re-fit temperature scaling — see §6.5. |

**Reliability bins.** Per-bin `(lo, hi, count, mean_confidence,
mean_accuracy)` — the data the reliability diagram plots. The
trainer stamps these into `EpochRecord.val_reliability` so the
on-disk `run.json` carries them; the viewer reads them to draw the
calibration diagram. A perfectly-calibrated model has
`mean_confidence ≈ mean_accuracy` in every bin (diagonal in the
reliability plot). Per-bin counts matter for interpreting the
diagram — see §6.6.

Reference: Guo C, Pleiss G, Sun Y, Weinberger KQ. *On Calibration
of Modern Neural Networks.* ICML 2017. arXiv:1706.04599 §2.

---

## 6. Troubleshooting

Symptom-driven guide. Each entry names the observable, the most
likely cause, the first fix to try, and (where relevant)
secondary causes worth checking. Section references (§X.Y) point
back to the reference sections above.

### 6.1 Train AUROC ≫ val AUROC by a wide margin

**Symptom:** training metrics keep improving while validation
plateaus or worsens after ~epoch 20. The gap widens through the
run.

**First fix:** the model is overfitting. Raise
`model.stochastic_depth` (§2.3) from `0.1` to `0.2` or `0.3`. If
that helps but the gap is still wide, raise
`train.max_shift_frac` (§3.6) from `0.10` to `0.15` — more
aggressive time-shift augmentation.

**Less likely:** the training set is too small for the
architecture. Drop `model.width_multiplier` (§2.4) to `0.5` and
re-train; if val improves at the smaller width, the issue was
capacity-vs-data, not regularization.

**Don't:** add parameters before adding regularization. Going from
`width 1.0 → 1.5` when the train/val gap is already wide makes
the problem worse, not better.

### 6.2 Training loss plateaus mid-run

**Symptom:** train loss flattens before reaching the expected
basin (typically by epoch ~20 in a 60-epoch run).

**First fix:** lower `train.lr` from `3e-4` to `1e-4`. The cosine
decay schedule (§3.3) can leave the model bouncing around a wide
loss basin if `base_lr` is too aggressive for the dataset.

**Less likely:** the dataset has a labeling pathology causing
irreducible loss. Sort training samples by per-sample loss after
the last epoch and inspect the highest-loss handful; if they all
look like edge cases or label errors, the floor is real, not a
training failure.

### 6.3 Loss spikes or NaN gradients

**Symptom:** training loss jumps from ~0.5 to NaN over a few
steps; subsequent steps stay NaN.

**First check:** confirm gradient clipping (§3.4) is on
(`train.grad_clip_norm: 1.0`, the default). This failure mode is
exactly what the clip exists to prevent.

**Less likely:** AMP fp16 (§3.5) underflow. Switch to bf16
(Ampere+) or disable AMP and re-run.

**Less likely:** the data loader is yielding NaN signals (a
corrupt trace, or a bank with a missing-data sentinel). Log
per-batch `min` / `max` of the signal tensor for a few epochs.

### 6.4 AUROC healthy but accuracy / F1 bad

**Symptom:** AUROC is e.g. 0.85 but F1 is 0.45 at threshold 0.5.

**Cause:** the model ranks samples well but the threshold is
wrong, or the model is poorly calibrated and the natural threshold
isn't 0.5.

**First fix:** sweep `τ` over the validation predictions bank
(`<stem>_pred.classifier.h5`) and pick the F1- or precision-recall-
maximizing threshold for your operational cost function (§5.2).

**Less likely:** the calibration shifted the optimal threshold off
0.5. Re-fit temperature scaling (§4.2) and re-sweep `τ` against
the recalibrated probabilities.

### 6.5 ECE > 0.10 (poor calibration)

**On the validation set during training.** AUROC is fine but
per-epoch ECE creeps up. This is the drift temperature scaling
was designed for — the export CLI's fit step handles it at
deployment time. No training-side action needed unless you're
treating ECE as a training objective (rare; ECE is not directly
differentiable).

**On a deployed model.** Predictions are confident but wrong;
ECE measured against a fresh labeled set is > 0.10.

- **First fix:** re-fit temperature scaling (§4.2) against a fresh
  calibration bank:
  ```
  egm-class-export examples/v1_export.yaml \
    --calibration-bank ../banks/calibration_v2.classifier.h5 \
    --output-name best_calv2
  ```
  See `docs/usage.md` §"Re-fit calibration without re-training."
- **Verify before redeploying:** confirm AUROC on a fresh held-out
  bank is roughly unchanged from the trained model's val AUROC.
  Calibration drift is much more common than the underlying
  classifier degrading — but if AUROC dropped too, the issue is
  upstream (see §6.9).

### 6.6 Reliability diagram looks ragged

**Symptom:** bins jump around the diagonal rather than tracing a
smooth curve.

**First check:** bin counts. With `N_val < 200` and `n_bins = 10`,
individual bins can hold fewer than 20 samples, making
`mean_accuracy` per bin noisy. This is a sampling-noise
visualization artifact, not a real calibration issue.

**First fix:** evaluate against a larger calibration bank (target
`N ≥ 1000` for `n_bins = 10`), or reduce `n_bins` (paying the
cost of coarser resolution).

### 6.7 Suspicious zero in the confusion matrix

- **Zero `tp` and nonzero `fn`:** the model never predicted
  positive at the configured `τ`. Usually `τ` is set above the
  model's maximum probability for the positive class; check
  `max(label_prob)` on the predictions bank and lower `τ`
  accordingly.
- **Zero `fp` and zero `tp`:** the model never predicted positive
  AND something pathological is upstream — no positive samples in
  the split (check `bank.label_truth_array().sum()`), or `τ = 1.0`.
- **Zero `tn` and nonzero `fp`:** the model always predicts
  positive. Mirror image of the first case; check
  `min(label_prob)` and raise `τ`.

### 6.8 Augmentation appears to hurt validation

**Symptom:** val AUROC improves when `data.augment_train` is set
`false`.

**Likely cause:** the time-shift augmentation (§3.6,
`max_shift_frac` default `0.10`) interacts with imperfect
activation-peak anchoring (§1.3) at the producer side. If the
peak detection is off by tens of milliseconds, large shifts push
the activation past the trace boundary and clip the wavefront.

**First fix:** drop `train.max_shift_frac` to `0.05` and re-train.
If val improves, the issue is producer-side anchoring quality,
not augmentation per se.

**Adjacent open question:** §1.3 notes that whether anchoring
helps at all is unresolved. If turning off augmentation entirely
also helps, the right next investigation is "does activation-peak
anchoring help?" rather than "what's the right
`max_shift_frac`?"

### 6.9 Deployed model degrades over time

**Symptom:** deployment metrics are worse than the trained
model's eval metrics, days/weeks/months after deployment.

**First check — hardware-level filter.** Most clinical
acquisition systems have their own post-amp filter that may have
been re-cut between deployment and now. The model expects the
band recorded in `metadata.bandpass_hz` — if the upstream filter
doesn't pass that band, the model gets band-limited inputs it
wasn't trained on (see §1.1 "Effects of a mismatched bandpass at
deployment").

**Second check — catheter / electrode change.** A different
bipolar spacing or a different electrode model changes the EGM
morphology in ways that shift the model's input distribution.

**Third check — calibration vs classifier degradation.** Run a
fresh held-out labeled set through `egm-class-eval`. If AUROC is
stable but ECE is up, it's calibration drift — re-fit `T` (§6.5,
"On a deployed model"). If AUROC dropped too, the underlying
classifier is degrading and the fix is upstream (re-train against
the new acquisition setup).

---

## 7. References

Architecture:

- Mehta S, Rastegari M. *MobileViT: Light-weight, General-purpose,
  and Mobile-friendly Vision Transformer.* ICLR 2022.
  arXiv:2110.02178.
- Sandler M, Howard A, Zhu M, Zhmoginov A, Chen L-C. *MobileNetV2:
  Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
  arXiv:1801.04381.
- Howard AG et al. *MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications.* 2017. arXiv:1704.04861
  (width multiplier).
- Vaswani A et al. *Attention Is All You Need.* NeurIPS 2017.
  arXiv:1706.03762.
- Dosovitskiy A et al. *An Image Is Worth 16x16 Words: Transformers
  for Image Recognition at Scale (ViT).* ICLR 2021.
  arXiv:2010.11929.
- Nie Y, Nguyen NH, Sinthong P, Kalagnanam J. *A Time Series Is
  Worth 64 Words: Long-term Forecasting with Transformers
  (PatchTST).* ICLR 2023. arXiv:2211.14730.
- Huang G, Sun Y, Liu Z, Sedra D, Weinberger KQ. *Deep Networks
  with Stochastic Depth.* ECCV 2016. arXiv:1603.09382.

Optimization + training:

- Loshchilov I, Hutter F. *Decoupled Weight Decay Regularization
  (AdamW).* ICLR 2019. arXiv:1711.05101.
- Loshchilov I, Hutter F. *SGDR: Stochastic Gradient Descent with
  Warm Restarts.* ICLR 2017. arXiv:1608.03983 (cosine annealing).

Calibration:

- Guo C, Pleiss G, Sun Y, Weinberger KQ. *On Calibration of Modern
  Neural Networks.* ICML 2017. arXiv:1706.04599 (temperature
  scaling + ECE + reliability diagrams).

Clinical / signal background (project-internal):

- Marchlinski FE, Callans DJ, Gottlieb CD, Zado E. *Linear ablation
  lesions for control of unmappable ventricular tachycardia in
  patients with ischemic and nonischemic cardiomyopathy.*
  Circulation 2000 (bipolar voltage threshold convention).
- See `project/` for the full project-internal reading list and
  citation status; this section names only the publications
  directly referenced by the math above.
