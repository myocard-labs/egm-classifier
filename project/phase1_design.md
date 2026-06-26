# EGM Fibrosis Classifier — Phase 1 Design Considerations

*1D MobileViT for bipolar intracardiac EGM, packaged as
`myocard-egm-classifier`.*

This document records every design decision made for the first iteration
of the ML model that consumes bipolar EGM traces from the
`myocard-egm-data` `ClassifierBank` format — which the producers
([`iafdb-pipeline`](https://github.com/myocard-labs/iafdb-pipeline) and
[`synthetic-egm-pipeline`](https://github.com/myocard-labs/synthetic-egm-pipeline))
write — and predicts fibrotic vs healthy tissue. The architecture is a 1D
adaptation of MobileViT (Mehta & Rastegari, 2022): a CNN/transformer
hybrid that gives us locality-aware feature extraction in the
convolutional path and signal-wide attention in the transformer path, at
modest parameter and compute budget.

**Author:** Daniel Klein (with AI assistance)
**Project:** Intracardiac Catheter Software — ICD CNN for fibrosis
**Companion:**
[`synthetic-egm-pipeline/project/phase1_design.md`](https://github.com/myocard-labs/synthetic-egm-pipeline/blob/release/project/phase1_design.md)
**Source of truth for the architecture:** the code in
`src/myocard_egm_classifier/`. When intent and code diverge, the code
wins and this doc should be updated.

## Scope and how to use this document

This document covers the **v1 model**: a 1D MobileViT-style hybrid
trained for **binary fibrosis classification** (fibrotic vs healthy) on
the synthetic `ClassifierBank` files (clean and noise-mixed) produced
by Phase 1 of the simulator. It is not a literature review; it is a
record of which choices were made, why, what was rejected, and where
to read more.

> **Terminology note.** Throughout this document "hybrid" refers to
> the **CNN+Transformer architecture** (a *hybrid model*), never to a
> dataset. The dataset distinction is **clean synthetic** vs
> **noise-mixed synthetic** (where IAFDB-derived noise has been
> overlaid by `synthetic-egm-pipeline`'s mixer). The umbrella project
> reserves "hybrid dataset" for the future case of synthetic and real
> data merged into one training set — which is also Sánchez et al.'s
> usage. The legacy bank naming used "hybrid" for noise-mixed
> synthetic; that is being phased out across the polyrepo.

Each section follows the same shape: **v1 choice**, **alternatives
considered** with pros/cons and a brief verdict, and **recommended
reading**. The phased roadmap at the end shows when the Phase 1
simplifications get lifted.

### Foundational reading

If you only read three things to anchor the rest, these are the right
three.

- **Mehta & Rastegari (2022).** *MobileViT: Light-weight,
  General-purpose, and Mobile-friendly Vision Transformer.* ICLR.
  The original 2D architecture this design adapts.
- **Sandler et al. (2018).** *MobileNetV2: Inverted Residuals and
  Linear Bottlenecks.* CVPR. Origin of the inverted-residual block
  that is half of every MobileViT.
- **Sánchez et al. (2021).** *Using Machine Learning to Characterize
  Atrial Fibrotic Substrate From Intracardiac Signals With A Hybrid
  in silico and in vivo Dataset.* Frontiers Physiol. The reference
  paper for the broader project; defines the task formulation this
  classifier targets.

---

## 1. 2D → 1D conversion strategy

**v1 choice:** Replace every 2D convolution with its 1D counterpart,
applied along the temporal axis. This applies to (a) channel-only 1×1
convs, (b) standard spatial → temporal convs, and (c) depth-wise spatial
→ temporal convs.

EGM data is fundamentally a 1D time series, so the receptive field
should extend in time, not in any synthetic spatial dimension. `Conv1d`
directly models temporal convolutions, which (relative to `Conv2d` on a
reshaped tensor) is cheaper by a factor equal to the fictitious spatial
dimension and is supported as a first-class op in PyTorch and TensorRT.
We rename the "spatial" convs to **temporal** convs throughout the
codebase so the intent is obvious at the call site.

### Alternatives considered

**Treat EGM as a thin 2D image (H=1) and use unmodified MobileViT** —
*Pretend it's an image; reshape `(T, C)` to `(1, T, C)`.*

| Pros | Cons |
|---|---|
| Zero code change — could copy MobileViT verbatim from `timm`. | All 2D convs are wasted compute along the fictitious H=1 axis. |
| Existing pretrained weights might transfer (probably won't, but cheap to try). | Patch size `(2,2)` becomes `(2,1)` or `(1,2)` — awkward; the unfold no longer matches the intent. |
| | ONNX / TensorRT export is bumpier with degenerate dimensions. |

*Why not for v1:* Rejected. The compute savings of `Conv1d` compound
across every layer of a hybrid network; pretending it's 2D throws those
savings away.

**Convert only the spatial convs to 1D; keep 1×1 convs as `Conv2d`** —
*Mixed dimensionality.*

| Pros | Cons |
|---|---|
| Slightly less refactoring of helper utilities. | Forces tensor reshapes between blocks; bug surface. |
| | TensorRT export is harder when intermediate shapes mix 1D and 2D. |

*Why not for v1:* Strictly worse than a clean 1D conversion.

**Use a 1D ConvNeXt backbone instead of MobileViT** — *Drop the
transformer entirely.*

| Pros | Cons |
|---|---|
| Simpler architecture, fewer hyperparameters. | Loses the signal-wide attention property that makes MobileViT well-suited to sparse EGM signal (activation + long quiet baseline). |
| ConvNeXt is well-tuned for vision; 1D variants exist. | Pure CNN needs many layers to relate distant temporal positions; the transformer does this in O(1) attention layers. |

*Why not for v1:* Worth revisiting if MobileViT-1D underperforms on real
data, but the hybrid is the right starting hypothesis for EGM.

### Recommended reading

- Mehta & Rastegari (2022). MobileViT — original 2D architecture.
- Liu et al. (2022). *A ConvNet for the 2020s* (ConvNeXt). CVPR. The
  pure-CNN alternative we're explicitly not picking.
- Sandler et al. (2018). MobileNetV2 — origin of the inverted-residual
  block.

---

## 2. Patch-based unfold for 1D attention

**v1 choice:** Unfold each MobileViT block's feature map by grouping
consecutive `p` timesteps into patches, then run the transformer's
attention across patches at each intra-patch position. **Patch size
varies per block** (`p=4` at the first MobileViT block, dropping to `2`
at deeper blocks) so the attention always sees enough patches to be
informative.

MobileViT's 2D unfold groups pixels by their intra-patch position and
attends across patches. Patch-internal locality stays in the conv path;
global structure lives in attention. The 1D analog:

$$
[B, C, T] \to [B, C, p, T/p] \to [B \cdot p, T/p, C]
$$

transformer along the $T/p$ axis, then fold back. **This is exactly the
behaviour we want for EGM**: the activation patch attends globally to
baseline patches, which a pure 1D CNN would need a huge receptive field
to do. Patch divisibility ($T$ at each MobileViT block must be a
multiple of $p$) is the one gotcha — $T=512$ with stride-2 reduction
gives $T$ at each MobileViT block in $\{64, 32, 16\}$, all divisible by
$p=4$ or $p=2$.

### Alternatives considered

**No unfold — transformer attends across raw timesteps** — *Treat each
timestep as a token.*

| Pros | Cons |
|---|---|
| Simplest; no patch math. | Attention is $O(T^2)$ per layer; expensive when $T$ is large. |
| | No locality preservation — redundant with the conv path. |
| | Loses the "corresponding-position-across-patches" structure that MobileViT was designed around. |

*Why not for v1:* Rejected. Defeats the point of using MobileViT.

**Patch size constant across all MobileViT blocks** — *Single global p
(e.g., 4 everywhere).*

| Pros | Cons |
|---|---|
| One config knob; easier to document. | At deep blocks where $T$ is already small (e.g., $T=16$), $p=4$ leaves only 4 patches — barely enough for attention to be informative. |
| | Different blocks have different optimal locality/globality trade-offs. |

*Why not for v1:* Less flexible. Per-block `p` (still defaults to a
single value when not overridden) costs nothing in code and lets you
tune attention granularity per block.

**Overlapping patches** — *Patches share boundary timesteps.*

| Pros | Cons |
|---|---|
| Smoother boundary handling for fold. | Breaks the exact invertibility of unfold-fold that MobileViT relies on. |
| | Non-trivial fold reconstruction (averaging overlap). |

*Why not for v1:* Premature optimization for v1.

### Recommended reading

- Mehta & Rastegari (2022). Section 3.2 covers the unfold-fold
  mechanism.
- Nie et al. (2023). *A Time Series Is Worth 64 Words: Long-Term
  Forecasting with Transformers* (PatchTST). ICLR. The cleanest
  reference for patch-based attention on 1D time series.
- Dosovitskiy et al. (2021). *An Image is Worth 16×16 Words* (ViT).
  ICLR. Origin of patch-based attention in vision; the abstraction
  MobileViT extends.

---

## 3. Stem convolution parameters

**v1 choice:** First op: 1D conv with `kernel=7`, `stride=2`,
`in_channels=1`, `out_channels=16` (scaled by `width_multiplier`),
`bias=False`, followed by `BatchNorm1d` and SiLU.

`Conv1d` at `kernel=7` has compute cost comparable to a 2D conv at
`kernel=3` (linear vs quadratic in kernel size), so we can afford the
wider receptive field at no extra cost. EGM's signal of interest spans
tens of milliseconds at 1 kHz — a kernel of 7 samples (7 ms) captures
morphology features (sharp upstroke / downstroke) cleanly in one layer,
whereas `kernel=3` needs stacking. `stride=2` immediately halves $T$
from 512 to 256, putting the network on the standard MobileViT
downsample schedule (input/stride at stem → `T_final = stem_stride ×
product of block strides`).

### Alternatives considered

**Kernel=3 stem (image-style)** — *Borrow image-classification
convention verbatim.*

| Pros | Cons |
|---|---|
| Smaller parameter count. | Misses the wider-kernel-is-cheap-in-1D observation. |
| Stacking small kernels can reach the same receptive field with non-linearity in between. | Needs additional layers to reach a useful temporal receptive field. |

*Why not for v1:* Sub-optimal for 1D; `kernel=7` dominates at equivalent
compute.

**Kernel=15+, stride=4+ (very aggressive stem)** — *Compress hard at the
entrance.*

| Pros | Cons |
|---|---|
| Very fast through the rest of the network. | `stride=4` at the stem with our $T=512$ input lands at $T=128$ — one fewer stride-2 block of expressivity later. |
| Forces the stem to learn high-level features immediately. | Sharp activation morphology is averaged away in a single pass. |

*Why not for v1:* Tempting because EGM is sparse, but loses too much
temporal discrimination at the input. Consider for Phase 2 after
Courtemanche lands and traces have richer morphology.

**Two stacked `kernel=3 stride=2` convs (StackPatch / patchify stem)** —
*ConvNeXt-style stem.*

| Pros | Cons |
|---|---|
| Two non-linearities at the entrance. | Adds depth at the most sensitive layer. |
| Modern convention in 2D vision. | Net stride of 4 — same lost-expressivity problem as the aggressive stem. |

*Why not for v1:* Considered; `kernel=7 stride=2` is simpler and gives
the same first-layer receptive field.

### Recommended reading

- He et al. (2016). *Deep Residual Learning for Image Recognition*
  (ResNet). CVPR. The first widely-used `kernel=7` stem; conventions
  inherited from ImageNet.
- Liu et al. (2022). ConvNeXt — alternative two-conv patchify stem.
- Hannun et al. (2019). *Cardiologist-level arrhythmia detection.*
  Nat. Med. Long-kernel 1D stems on physiological signals.

---

## 4. MobileNetV2-1D (MV2-1D) block

**v1 choice:** Inverted-residual block: 1×1 expansion conv (`channels ×
expansion_ratio`) → depth-wise temporal conv (`kernel_size`, `stride`)
→ 1×1 projection conv. `BatchNorm1d` after each conv, SiLU activation
after first two convs, linear (no activation) after projection.
Residual connection when `stride=1` and `in_channels==out_channels`.

Direct 1D port of the MobileNetV2 inverted residual. The depth-wise
conv handles temporal mixing (per-channel) cheaply; the 1×1 convs
handle channel mixing. With temporal `kernel=3` and `expansion_ratio=4`,
each MV2-1D block has ~10× fewer parameters than a naive `Conv1d` block
of equivalent receptive field, with no accuracy loss in practice.

### Alternatives considered

**Plain residual block (3× `Conv1d` stack)** — *ResNet basic block
adapted to 1D.*

| Pros | Cons |
|---|---|
| Simpler; one less hyperparameter (`expansion_ratio`). | ~3–5× more parameters and FLOPs for the same receptive field. |
| | Not the standard MobileViT building block. |

*Why not for v1:* Rejected. Inverted residuals are the whole reason
MobileViT scales.

**Add Squeeze-and-Excitation (SE) to each block** — *MobileNetV3-style
SE blocks.*

| Pros | Cons |
|---|---|
| Per-channel attention; demonstrated improvement on image tasks. | Adds a hyperparameter (SE reduction ratio). |
| Cheap to compute. | Marginal gain on small datasets; the transformer already provides global mixing. |

*Why not for v1:* Phase 2 ablation candidate, not v1 default.

### Recommended reading

- Sandler et al. (2018). MobileNetV2 — Section 3 derives the inverted
  residual.
- Howard et al. (2019). *Searching for MobileNetV3.* ICCV. Adds SE and
  h-swish.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for CNNs.*
  ICML. Justifies `expansion_ratio=4 / 6` as scale-invariant defaults.

---

## 5. MobileViT-1D block (the headline block)

**v1 choice:** Five-stage pipeline:

1. **Local rep conv:** `Conv1d` (`kernel=local_kernel_size`) →
   `BatchNorm1d` → SiLU
2. **Channel expansion:** 1×1 `Conv1d`, `out=transformer_dim`
3. **Unfold + Transformer + Fold:** patch unfold (`size=p`),
   `n_transformer_layers` × pre-norm transformer block (`LayerNorm`,
   multi-head self-attention (`n_heads`), MLP with `mlp_ratio`), fold
   back to `(T, transformer_dim)`
4. **Channel projection:** 1×1 `Conv1d` back to `in_channels`
5. **Fusion conv:** concatenate output with the input to the block
   (residual-style), 1D conv with `fusion_kernel_size`, `BatchNorm1d`,
   SiLU

Direct 1D translation of MobileViT's block. Steps 1+5 are the "local
representation learning" (CNN); step 3 is the "global representation
learning" (transformer); steps 2 and 4 do the channel-count bookkeeping
that lets the transformer operate in a higher-dimensional embedding
space. The skip+fusion (step 5) is what gives MobileViT its
locality-preservation property: the input features flow around the
transformer and are recombined with the attention-mixed features, so no
spatial structure is lost. This block has the largest hyperparameter
count in the architecture — expect to spend the most ablation effort
here.

### Alternatives considered

**Drop the local rep conv (step 1)** — *Direct expansion + transformer.*

| Pros | Cons |
|---|---|
| Slightly faster. | Loses the conv-side locality that the transformer can't (efficiently) provide. |
| | Empirically, MobileViT ablates this and reports a small accuracy drop. |

*Why not for v1:* Keep the conv; cost is negligible.

**Drop the fusion conv (step 5)** — *Just sum residual.*

| Pros | Cons |
|---|---|
| Simpler; saves a conv layer. | Empirically the fusion conv contributes ~1% accuracy on ImageNet. |
| | Without it the local and global features can't interact non-linearly. |

*Why not for v1:* Keep the fusion conv.

**Use cross-attention (transformer attends to a `[CLS]` token)** —
*ViT-style aggregation instead of fold.*

| Pros | Cons |
|---|---|
| Single-token aggregation is conceptually clean. | Requires positional encoding or learned tokens. |
| | Breaks the locality-preservation contract; harder to fuse with the conv path. |

*Why not for v1:* Worth investigating in Phase 2 if pure CLS-token
aggregation beats the fold/fuse pattern on this task. Not v1 default.

### Recommended reading

- Mehta & Rastegari (2022). MobileViT — Figure 1 and Section 3.2 are
  the definitive references.
- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS. The
  transformer block itself.
- Mehta & Rastegari (2022, v2). *Separable Self-Attention for Mobile
  Vision Transformers.* arXiv 2206.02680. MobileViTv2 swaps multi-head
  self-attention for a cheaper variant; consider for Phase 2 if
  attention cost becomes a bottleneck.

---

## 6. Head — channel expansion + global pool + linear classifier

**v1 choice:** 1×1 `Conv1d` expanding to `head_expansion_channels`
(default 320 × `width_multiplier`) → `BatchNorm1d` → SiLU → Global
Average Pool (over the $T$ axis) → `Dropout(0.1)` → `Linear` to
`num_outputs` (=1 for v1 binary, single-logit BCE).

> **Code-wins-over-doc note.** The legacy PDF originally said `Linear
> to n_classes (=2 for v1)`, intending a 2-output softmax head. The
> training section (Section 12) simultaneously specified BCE with
> logits — "*the* logit ... calibrated confidence" (singular). A
> 2-output softmax and a 1-output sigmoid are the two standard ways to
> do binary classification; only the 1-output form has *the* logit. We
> ship `num_outputs == 1` + `BCEWithLogitsLoss` accordingly. The
> multi-output softmax path (`num_outputs >= 2` + cross-entropy) stays
> available for the Phase-2 multi-class severity head.

Standard MobileViT head, ported to 1D. The channel expansion gives the
global pool a wider feature vector to summarize; the dropout helps with
calibration on small datasets; the linear classifier is the simplest
possible classification head.

### Alternatives considered

**Skip the channel expansion; pool directly** — *Save a layer.*

| Pros | Cons |
|---|---|
| Smaller model. | Linear head has too few input features for a meaningful classifier. |

*Why not for v1:* Rejected. The channel expansion is cheap and
necessary.

**Use max pool instead of average pool** — *Take the strongest
activation per channel.*

| Pros | Cons |
|---|---|
| Sometimes better for sparse signals. | Loses contextual averaging; can be unstable with small batches. |

*Why not for v1:* Worth ablating in Phase 2; v1 uses average pool to
match MobileViT.

**Use a `[CLS]` token instead of pool** — *ViT-style aggregation.*

| Pros | Cons |
|---|---|
| Learned per-task aggregation. | Requires positional encoding or a learned token. |
| | Departs from MobileViT's conv-side aggregation philosophy. |

*Why not for v1:* Phase 2 ablation candidate.

### Recommended reading

- Lin et al. (2014). *Network In Network.* ICLR. Original argument for
  global average pool as a classifier head.
- Mehta & Rastegari (2022). MobileViT head detail.

---

## 7. Input trace length and pool-target dimension

**v1 choice:** Input length $T=512$ (zero-padded from the simulator's
shorter output until Phase 2 lengthens traces). Final pool dimension
$T_{\text{final}}=16$ (achieved by stem `stride=2` plus four stride-2
blocks — net 32× reduction).

$T=512$ sits in the sweet spot of the design constraints: (a)
$T_{\text{final}}=16$ at the global pool gives the linear head a useful
16-position summary without compute waste; (b) powers of 2 throughout
the network make patch-divisibility automatic at every MobileViT
block; (c) the same input length supports v2 multi-activation training
without re-design (~3 atrial AF activations fit at 1 kHz). The small
zero-pad per side of the current single-activation simulator output is
essentially free.

### Alternatives considered

**T=256** — *Closest to the current short simulator output.*

| Pros | Cons |
|---|---|
| ~2× less training compute. | $T_{\text{final}}=8$ at pool — right at the floor for global average pool. |
| Minimal zero-padding from the simulator. | No architectural headroom for multi-activation v2. |

*Why not for v1:* Premature optimization; the compute savings aren't
worth the lost head capacity.

**T=1024 (multi-activation from day 1)** — *Skip the v1
single-activation arc.*

| Pros | Cons |
|---|---|
| Final architecture from day 1; no re-training. | Requires the Phase 4 simulator extension (multi-beat pacing) before any v1 training. |
| $T_{\text{final}}=32$ — comfortable head fanout. | ~2× v1 compute cost. |
| | Couples model and simulator timelines. |

*Why not for v1:* Right answer eventually; wrong order. Doing v1
single-activation first decouples the two timelines.

**Variable-length inputs** — *Handle short and long traces in the same
model.*

| Pros | Cons |
|---|---|
| No padding waste. | Requires dynamic shapes through the transformer; TensorRT-friendly but adds export complexity. |
| | Pool needs special handling. |

*Why not for v1:* Phase 3+ feature; not Phase 1.

### Recommended reading

- Hannun et al. (2019). *Cardiologist-level arrhythmia detection.*
  Nat. Med. Long input traces (~30 s) for ECG; useful for thinking
  about how input length scales with the task.
- Nie et al. (2023). PatchTST — explicit treatment of input length for
  time-series transformers.

---

## 8. Width multiplier (single architecture, configurable size)

**v1 choice:** Single architecture definition; a `width_multiplier`
config field scales every channel count uniformly. Defaults: `0.5`
(small, for fast iteration), `1.0` (baseline, for production), `1.5`
(large, for capacity headroom). Channel counts are rounded to the
nearest multiple of 8 (`make_divisible`) to keep convs
hardware-aligned.

Borrowed directly from MobileNet v1's $\alpha$ convention. One model
definition, three sizes, no near-duplicate Python files to keep in
sync. The cost is one config field and ~10 lines of channel-count
scaling logic. Pays off the first time you want to ablate "does this
change help at all sizes?" or run a fast version on a laptop.

### Alternatives considered

**Three named architecture variants (XS / S / M)** — *MobileViT-XS /
-S / -L style.*

| Pros | Cons |
|---|---|
| Clearer when comparing against paper-equivalent sizes. | Three near-duplicate Python files; they drift. |
| | Width multiplier achieves the same with less code. |

*Why not for v1:* Width multiplier dominates strictly on engineering
grounds.

**No size variants — one fixed architecture** — *Defer the multiplier
knob.*

| Pros | Cons |
|---|---|
| Simplest. | First time you want a smaller / larger version, you'll add the multiplier anyway. |
| | Architecture ablations are weakened without a width sweep. |

*Why not for v1:* Rejected; the multiplier is cheap insurance.

### Recommended reading

- Howard et al. (2017). *MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications.* arXiv 1704.04861.
  Original $\alpha$ / $\rho$ multipliers.
- Tan & Le (2019). EfficientNet — compound scaling as a more
  principled alternative to a single width knob.

---

## 9. Configuration system

**v1 choice:** YAML config → nested frozen dataclasses + a block
registry, consistent with the strategy-registry pattern used in
`synthetic-egm-pipeline`. Each block in the architecture is declared by
`type` (`conv_stem_1d` / `mv2_1d` / `mobilevit_1d` / `head_1d`) plus its
parameters; a small registry maps `type` to a constructor.

This is exactly the pattern `timm`, MMDetection, MMSegmentation, and HF
Transformers use to describe model architectures: list of blocks, each
with a `type` and a parameter dict. The loader iterates the list, looks
up the constructor in a registry, passes the parameter dict (validated
against a per-block dataclass), and stacks the resulting modules. New
block types add themselves to the registry; the loader doesn't change.
Adopting the same shape as `synthetic-egm-pipeline`'s strategy registry
means no cross-package context-switch cost when reading either
codebase.

### Alternatives considered

**Hydra with composable configs** — *FAIR's ML config framework.*

| Pros | Cons |
|---|---|
| Composable model / optimizer / dataset configs. | Adds a non-trivial dependency. |
| Native CLI overrides without code changes. | Learning curve. |
| Built-in sweep mode for hyperparameter search. | Overkill for v1 single-task training. |

*Why not for v1:* Revisit at end of Phase 2 when hyperparameter sweep
complexity may justify the migration.

**Python configs only (timm-style)** — *Config functions return model
instances.*

| Pros | Cons |
|---|---|
| Maximum flexibility; config IS code. | Not portable to non-Python tooling. |
| | Inconsistent with the rest of the project. |

*Why not for v1:* Inconsistent with the project; rejected.

### Recommended reading

- [`timm`](https://github.com/huggingface/pytorch-image-models). The
  definitive reference for Python model registries.
- [MMDetection / MMCV](https://github.com/open-mmlab/mmdetection).
  YAML+registry pattern at industrial scale.
- [Hydra](https://hydra.cc). The alternative we're explicitly not
  picking for v1.

---

## 10. Normalization, activation, regularization

**v1 choice:** **Norm:** `BatchNorm1d` inside the conv path (after each
conv except right before the residual sum); `LayerNorm` inside the
transformer (pre-norm placement). **Activation:** SiLU (a.k.a. Swish)
in conv blocks; GELU in the transformer MLP. **Regularization:** 0.1
dropout in MLP; stochastic depth (DropPath) with linear schedule from
0.0 at the stem to 0.1 at the final block.

Standard modern recipe. `BatchNorm` performs better than `LayerNorm`
in small conv stacks where batch statistics are stable; the reverse
holds for transformer blocks where token statistics dominate. SiLU is
MobileNetV3 / MobileViT default and TensorRT-friendly; GELU is the
near-universal transformer choice. DropPath is the modern regularizer
for transformer-hybrid models with limited training data — exactly your
situation.

> **Note on dropout vs DropPath.** They are *complementary*, not
> alternatives. `nn.Dropout` is element-wise — it zeroes individual
> activations within the attention/MLP output with probability `p`,
> preventing co-adaptation of features within a layer. `DropPath`
> (stochastic depth) is sample-wise — it zeroes the *entire* residual
> branch for some samples in the batch, encouraging the network to be
> robust to layer-level redundancy. The original Vaswani / ViT recipe
> uses element-wise dropout (on attention weights, after the
> attention projection, inside the MLP); DeiT added stochastic depth
> on top and MobileViT inherited both. Our v1 defaults run DropPath
> on a linear schedule with `dropout=0.0` inside the transformer
> blocks; deeper analysis is a Phase-2 ablation candidate.

### Alternatives considered

**Use `LayerNorm` everywhere (ConvNeXt-style)** — *Drop `BatchNorm`.*

| Pros | Cons |
|---|---|
| No batch-statistics dependence at inference. | Conv-path performance typically drops slightly without BN. |
| Consistent across conv and transformer halves. | Slight TensorRT compatibility considerations. |

*Why not for v1:* Worth ablating in Phase 2.

**ReLU instead of SiLU** — *Classical CNN activation.*

| Pros | Cons |
|---|---|
| Cheaper to compute; well-understood. | Worse calibration; modern hybrids prefer smooth activations. |

*Why not for v1:* Rejected; SiLU is the established standard.

**Heavy dropout (0.5) instead of DropPath** — *Old-school
regularization.*

| Pros | Cons |
|---|---|
| Familiar. | Hurts representation quality in transformer blocks. |
| | DropPath is the modern preferred form for ViT-style models. |

*Why not for v1:* Outdated; DropPath dominates for this architecture
class.

### Recommended reading

- Ramachandran et al. (2017). *Searching for Activation Functions.*
  arXiv. The Swish/SiLU origin.
- Hendrycks & Gimpel (2016). *Gaussian Error Linear Units (GELUs).*
  arXiv.
- Huang et al. (2016). *Deep Networks with Stochastic Depth.* ECCV.
  DropPath / stochastic depth.
- Ioffe & Szegedy (2015). *Batch Normalization.* ICML.
- Ba et al. (2016). *Layer Normalization.* arXiv.

---

## 11. Trace normalization at the input

**v1 choice:** Per-trace z-score (subtract trace mean, divide by trace
std) applied as a non-trainable input transform. Standard deviation
floored at `1e-6` to avoid division by zero on degenerate (zero-padded)
inputs.

The simulator produces unitless Aliev-Panfilov amplitudes ($\sim 10^{-3}$
range under $K=1.97$); the noise mixer overlays IAFDB-derived noise
to produce traces in real-data amplitude scale ($\sim 10^{-2}$ mV
range). Without input normalization, mixed-data training fails because
the network can't generalize the absolute scale. Per-trace z-score is amplitude-invariant
by construction; it discards the absolute amplitude (which clinical
wisdom anyway associates with electrode contact quality, not substrate
— Marchlinski 2000, Sanders 2003) and keeps the morphology, which is
what the classifier should actually learn from.

### Alternatives considered

**Global dataset z-score** — *Use a single mean / std computed over the
whole training set.*

| Pros | Cons |
|---|---|
| Preserves between-trace amplitude differences. | Sim and real datasets have different global statistics; transfer is poor. |
| | Sensitive to dataset shift. |

*Why not for v1:* Rejected for sim/real transfer reasons.

**No normalization — rely on first BatchNorm** — *Let BN handle it.*

| Pros | Cons |
|---|---|
| Zero extra code. | BN's per-channel statistics aren't designed for per-trace amplitude. |
| | Sensitive to batch composition (sim-heavy vs real-heavy batches). |

*Why not for v1:* Brittle for sim/real transfer; rejected.

**Per-trace peak normalization (divide by max-abs)** — *Robust to
outliers in noise.*

| Pros | Cons |
|---|---|
| Trivial; intuitive. | Sensitive to a single noise spike. |
| | Discards distributional information that z-score preserves. |

*Why not for v1:* Inferior to z-score; rejected.

### Recommended reading

- Marchlinski et al. (2000) and Sanders et al. (2003) on the clinical
  voltage-threshold convention — relevant for understanding why
  amplitude is partly a contact-quality artifact.
- The IAFDB calibration discussion in
  [`iafdb-pipeline`](https://github.com/myocard-labs/iafdb-pipeline)'s
  `docs/usage.md`, especially the R-wave anchoring section.

---

## 12. Training recipe (loss, optimizer, schedule, augmentation)

**v1 choice:** Loss: binary cross-entropy with logits
(`BCEWithLogitsLoss`, single-logit head per Section 6). Optimizer:
AdamW (`lr=3e-4`, `weight_decay=0.05`, `betas=(0.9, 0.999)`). Schedule:
linear warmup over 5% of steps, then cosine decay to 0 over the
remainder. Batch size: 64 at `width_multiplier=1.0` (auto-tune at
startup). Augmentation: noise mixing (already applied per-trace by the
producer for noise-mixed banks; the train-time `TraceTransform` adds
random gain ±20% and random time-shift ±10% of trace length on top).
No Mixup or time warping for v1 (interacts with single-activation
morphology).

Standard modern hybrid-architecture recipe. AdamW + cosine schedule +
warmup is the most robust default for transformer-containing models.
BCE with logits is appropriate for binary classification with the
logit being directly interpretable as a calibrated confidence. The
augmentation set is deliberately conservative for v1 — we want to see
the model learn the substrate before we hammer it with augmentation
complexity. Add Mixup / CutMix in Phase 2 once baseline performance is
established.

### Alternatives considered

**Focal loss instead of BCE** — *Down-weights easy examples.*

| Pros | Cons |
|---|---|
| Better for severe class imbalance (>10:1). | Adds focal-$\gamma$ hyperparameter. |
| | Class balance after stratified sampling will likely be moderate, not severe. |

*Why not for v1:* Defer until empirical evidence demonstrates BCE
underperforms.

**SGD with momentum** — *Old-school optimizer.*

| Pros | Cons |
|---|---|
| Better generalization in some pure-CNN regimes. | Worse for transformer-containing architectures. |
| | More hyperparameters to tune. |

*Why not for v1:* Rejected; AdamW is the established choice for hybrid
models.

**Constant LR (no schedule)** — *Simplest.*

| Pros | Cons |
|---|---|
| No schedule complexity. | Doesn't converge as well; misses standard practice. |

*Why not for v1:* Rejected.

**Mixup / CutMix from day 1** — *Strong regularizer for vision.*

| Pros | Cons |
|---|---|
| Improved calibration; cheap. | Mixed-sample augmentation can confuse single-activation morphology learning. |
| | Better evaluated after baseline is established. |

*Why not for v1:* Add in Phase 2 ablation.

### Recommended reading

- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization*
  (AdamW). ICLR.
- Goyal et al. (2017). *Accurate, Large Minibatch SGD.* arXiv. Origin
  of the warmup convention.
- Loshchilov & Hutter (2017). *SGDR: Stochastic Gradient Descent with
  Warm Restarts.* ICLR. Cosine schedule.
- Zhang et al. (2018). *mixup: Beyond Empirical Risk Minimization.*
  ICLR.

---

## 13. Validation strategy and the IAFDB diagnostic

**v1 choice:** Patient-aware split where `patient_id` on each trace is
the split key. For synthetic data the producer stamps `patient_id =
simulation_id` (every trace from the same Finitewave run shares a
substrate realization); for IAFDB the producer stamps `patient_id` from
the source record. 80/10/10 train/val/test by `patient_id`.

**The second-stage IAFDB pass is a label-free diagnostic, not a scored evaluation.** Per the
[IAFDB epistemic catch-22 memo](https://github.com/myocard-labs/intracardiac-platform/tree/main/project),
IAFDB has no trustworthy fibrosis labels — the high-voltage healthy
extraction in `iafdb-pipeline` produces a *labeling assumption*, not
ground truth. Using that assumption as a test set is logically
circular: it scores the model against the labels we'd already produce
without it. The v0.2.0 eval CLI therefore:

- emits per-trace predictions for both labeled and unlabeled banks;
- emits classification metrics (accuracy, AUROC, ECE, per-class
  precision/recall/F1) **only when the input bank carries
  trustworthy labels** (the synthetic banks, clean or noise-mixed);
- for IAFDB inputs, emits predictions for a label-free look at the
  model's P(fibrotic) distribution (the saturation diagnostic) and
  explicitly *does not* report accuracy or a calibration curve — both
  need labels IAFDB lacks.

The legacy `eval_sim2real_cmd` is dropped. Real-world evaluation that
breaks the catch-22 — joint MRI + EGM measurement, or another modality
that produces a substrate ground truth — is a post-refactor research
direction, not a v1 deliverable.

Without patient-aware splitting the model sees the same fibrosis
pattern realization in train and val, which inflates val metrics —
this exact failure has burned ML cardiology papers. Patient-aware
splitting on `patient_id` guarantees that an exact substrate pattern
lives entirely in one split.

### Alternatives considered

**Random per-trace split** — *Standard sklearn split.*

| Pros | Cons |
|---|---|
| One-line implementation. | Train and val see the same substrate; inflated val metrics. |
| | Reported failure mode in multiple cardiology ML papers. |

*Why not for v1:* Strongly rejected.

**K-fold cross-validation by `patient_id`** — *Patient-aware K-fold.*

| Pros | Cons |
|---|---|
| More statistically reliable. | K× the training cost. |
| Standard in clinical ML. | Overkill for a v1 architecture sanity check. |

*Why not for v1:* Phase 2 evaluation; v1 single split is sufficient to
gate further work.

**Use IAFDB high-voltage segments as labeled test set** — *(legacy
position).*

| Pros | Cons |
|---|---|
| Real-world data; intuitive sim-to-real signal. | Catch-22: the labels are an assumption produced by the same heuristic the model would replace. |
| | Inflates apparent transfer performance because the model and the labels share a prior. |

*Why not for v1:* The new position; the legacy doc treated this as the
v1 sim-to-real signal. After the catch-22 memo, it became a
predictions-only diagnostic, not a metrics-bearing eval.

### Recommended reading

- Guo et al. (2017). *On Calibration of Modern Neural Networks.* ICML.
  Reliability diagrams and ECE; the calibration toolkit for the
  **synthetic-side** evaluation. Both need labels, so they don't apply
  to the label-free IAFDB diagnostic.
- Project memory: `project_iafdb_eval_catch22` — the full catch-22
  argument.
- Project memory: `project_icd_cnn_fibrosis` — patient-aware splitting
  discussion in the kickoff.

---

## 14. TensorRT-aware design choices

**v1 choice:** Use only ops supported by modern TensorRT (≥ 8.6):
`Conv1d`, `BatchNorm1d`, `LayerNorm`, SiLU, GELU, multi-head attention.
Export with a fixed input shape `(B, 1, 512)`. No dynamic patch sizes
or shapes that would force dynamic-shape TRT export.

TensorRT deployment is part of the broader project narrative (per the
umbrella project memory). Making design decisions consistent with
TRT-supported ops at architecture time avoids painful refactors later.
All ops in the chosen design are first-class in modern TRT; the only
thing to watch is fixed-shape export. The fold/unfold inside MobileViT
blocks should trace cleanly with fixed input shape; verify during Phase
2 by exporting and benchmarking.

### Alternatives considered

**Ignore TRT constraints; refactor later** — *Optimize for training
only.*

| Pros | Cons |
|---|---|
| No design constraints up front. | Refactor cost when TRT integration starts. |
| | Risk of architectural choices that don't translate. |

*Why not for v1:* Avoidable rework; rejected.

**ONNX-only deployment plan** — *Skip TRT, use generic runtime.*

| Pros | Cons |
|---|---|
| Less platform-specific. | Loses the TRT performance story in the umbrella project. |
| | Daniel has TRT background and it's a portfolio differentiator. |

*Why not for v1:* Rejected on portfolio grounds.

### Recommended reading

- NVIDIA TensorRT support matrix. Confirm op support per release.
- PyTorch ONNX export docs — tracing constraints.

---

## Phased roadmap

Every Phase 1 simplification corresponds to an alternative listed
above. The roadmap below shows when each is lifted.

### Phase 1 (this document)

Binary fibrotic vs healthy classification on Phase 1 simulator output.
1D MobileViT, $T=512$, single-activation traces, `width_multiplier=1.0`.
Patient-aware split by `patient_id`. Predictions-only diagnostic on
IAFDB high-voltage healthy segments; no IAFDB-derived accuracy claim.

### Phase 2 — Multi-class severity

Extend the head to 3 classes matching the clinical bipolar voltage
tiers (healthy / border zone / dense scar). Requires the stratified
sampling work in `synthetic-egm-pipeline`'s roadmap. Ablation sweep at
`width_multiplier = 0.5 / 1.0 / 1.5`.

### Phase 3 — Multi-activation training

Retrain on $T=512$ multi-activation traces (~3 atrial AF activations).
Same architecture; just different data. Requires the simulator
multi-beat-pacing upgrade. IAFDB-side diagnostic moves from
high-voltage segments to full recordings.

### Phase 4 — Pattern classification

Add a second classification head for fibrosis pattern (interstitial /
patchy / compact) per Nezlobinsky 2021. Multi-task training. Requires
the simulator pattern-strategy work in
`synthetic-egm-pipeline`'s roadmap.

### Phase 5 — CLOCS-style self-supervised pretraining

Pretrain on unlabeled IAFDB traces using a CLOCS-style contrastive
objective, then fine-tune on synthetic labeled traces. Improves
transfer to real data without relying on IAFDB fibrosis labels (which
we don't trust — see Section 13). From the project's working
architectural hypothesis (memory note).

### Phase 6 — TensorRT deployment

Export the chosen architecture to ONNX, build TRT engines, benchmark
latency and throughput. Compare against PyTorch eager-mode reference.
Optional Phase 7: C++ inference wrapper for the eventual deployment
story.

### Post-refactor research direction — break the IAFDB catch-22

Joint MRI + EGM (or any modality that produces a substrate ground
truth on the same tissue the EGM samples) is the path to real
fibrosis-labeled in vivo data. Out of scope for the Phase-1 model
itself, but the umbrella project plan tracks this as the
follow-on that turns IAFDB-side diagnostic into a real-world test set.

---

## References

### Core architectures

- Mehta S., Rastegari M. (2022). *MobileViT: Light-weight,
  General-purpose, and Mobile-friendly Vision Transformer.* ICLR.
- Sandler M., Howard A., Zhu M., Zhmoginov A., Chen L.-C. (2018).
  *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
- Howard A. et al. (2019). *Searching for MobileNetV3.* ICCV.
- Howard A. et al. (2017). *MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications.* arXiv 1704.04861.
- Vaswani A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Dosovitskiy A. et al. (2021). *An Image is Worth 16×16 Words.* ICLR.
- Liu Z., Mao H., Wu C.-Y., Feichtenhofer C., Darrell T., Xie S.
  (2022). *A ConvNet for the 2020s* (ConvNeXt). CVPR.
- Mehta S., Rastegari M. (2022). *Separable Self-Attention for Mobile
  Vision Transformers* (MobileViTv2). arXiv 2206.02680.

### Time-series transformers

- Nie Y., Nguyen N. H., Sinthong P., Kalagnanam J. (2023). *A Time
  Series Is Worth 64 Words: Long-Term Forecasting with Transformers*
  (PatchTST). ICLR.

### ECG / EGM ML

- Hannun A., Rajpurkar P. et al. (2019). *Cardiologist-level
  arrhythmia detection and classification in ambulatory
  electrocardiograms using a deep neural network.* Nature Medicine 25,
  65–69.
- Sánchez J., Nothstein M., Loewe A. et al. (2021). *Using Machine
  Learning to Characterize Atrial Fibrotic Substrate From Intracardiac
  Signals With A Hybrid in silico and in vivo Dataset.* Frontiers in
  Physiology 12, 699291.
- Kiyasseh D., Zhu T., Clifton D. A. (2021). *CLOCS: Contrastive
  Learning of Cardiac Signals Across Space, Time, and Patients.* ICML.
- Okenov A., Nezlobinsky T., Zeppenfeld K., Vandersickel N., Panfilov
  A. V. (2024). *Computer based method for identification of fibrotic
  scars from electrograms and local activation times on the epi- and
  endocardial surfaces of the ventricles.* PLOS ONE 19(4), e0300978.

### Activations, normalization, regularization

- Ioffe S., Szegedy C. (2015). *Batch Normalization.* ICML.
- Ba J. L., Kiros J. R., Hinton G. E. (2016). *Layer Normalization.*
  arXiv.
- Ramachandran P., Zoph B., Le Q. V. (2017). *Searching for Activation
  Functions* (Swish/SiLU). arXiv.
- Hendrycks D., Gimpel K. (2016). *Gaussian Error Linear Units
  (GELUs).* arXiv.
- Huang G., Sun Y., Liu Z., Sedra D., Weinberger K. Q. (2016). *Deep
  Networks with Stochastic Depth.* ECCV. (DropPath origin.)

### Training-time recipe

- Loshchilov I., Hutter F. (2019). *Decoupled Weight Decay
  Regularization* (AdamW). ICLR.
- Loshchilov I., Hutter F. (2017). *SGDR: Stochastic Gradient Descent
  with Warm Restarts.* ICLR.
- Goyal P. et al. (2017). *Accurate, Large Minibatch SGD.* arXiv.
  Warmup.
- Zhang H. et al. (2018). *mixup: Beyond Empirical Risk Minimization.*
  ICLR.

### Calibration

- Guo C., Pleiss G., Sun Y., Weinberger K. Q. (2017). *On Calibration
  of Modern Neural Networks.* ICML.

### Engineering / config systems

- [`timm` (pytorch-image-models)](https://github.com/huggingface/pytorch-image-models).
- [MMDetection / MMCV](https://github.com/open-mmlab/mmdetection).
- [Hydra](https://hydra.cc).
- [PyTorch Lightning](https://lightning.ai).

### Clinical voltage threshold conventions (relevant to normalization)

- Marchlinski F. E. et al. (2000). *Linear ablation lesions for
  control of unmappable ventricular tachycardia.* Circulation
  101(11), 1288–1296. Original bipolar voltage tiers.
- Sanders P. et al. (2003). *Electrical remodeling of the atria in
  congestive heart failure.* Circulation 108(12), 1461–1468. Atrial
  0.5 mV threshold.
- Kosiuk J. et al. PMID 30873619. 0.2 mV during AF ≈ 0.5 mV during SR.

---

*End of document. The source of truth for the architecture is the code
in `src/myocard_egm_classifier/`. This markdown captures intent and
rationale; when intent and code diverge, the code wins and this doc
should be updated.*
