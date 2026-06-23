"""Canonical Phase-1 constants for the EGM fibrosis classifier.

Every magic number the design document pins down lives here, so the
rest of the codebase references a name instead of a literal. The
legacy v1 design doc lives at
``egm_classifier_old/project/egm_classifier_phase1_design.pdf``;
section numbers below refer to that document.

Bank-schema constants that used to live here (``BANK_TRACES_GROUP``,
``BANK_SIGNAL_DATASET``, etc.) have moved upstream — schemas are
owned by ``myocard-egm-contracts``, and bank I/O is owned by
``myocard-egm-data``. This file is model + training + label
defaults only.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Input geometry (design doc Section 7)
# ---------------------------------------------------------------------------

DEFAULT_INPUT_LENGTH = 512
"""Model input length T at 1 kHz. A power-of-two T keeps
patch-divisibility automatic at every MobileViT block and yields
T_final = 16 at the global pool (stem stride 2 + four stride-2 blocks
= 32x reduction). The dataset reads the actual trace length from the
bank, so this constant is documentation, not a hard config."""

DEFAULT_INPUT_CHANNELS = 1
"""Bipolar EGM is a single-channel time series."""

EXPECTED_FS_HZ = 1000.0
"""IAFDB sample rate; the synthetic producer also writes at 1 kHz
(see synthetic-egm-pipeline's DEFAULT_OUTPUT_FS_HZ). A bank at a
different rate is a config error, not a silent resample."""

# ---------------------------------------------------------------------------
# Width multiplier (design doc Section 8)
# ---------------------------------------------------------------------------

DEFAULT_WIDTH_MULTIPLIER = 1.0
"""One architecture scaled by a single multiplier. 1.0 = baseline /
production, 0.5 = fast iteration, 1.5 = large."""

CHANNEL_ROUND = 8
"""Channel counts are rounded to a multiple of this number to keep
convs hardware-aligned."""

# ---------------------------------------------------------------------------
# Head (design doc Section 6)
# ---------------------------------------------------------------------------

DEFAULT_HEAD_EXPANSION_CHANNELS = 320
"""1x1 conv expands to this many channels before the global pool.
Scaled by ``width_multiplier``."""

DEFAULT_HEAD_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Binary task (design doc Section 6 + 12)
# ---------------------------------------------------------------------------

DEFAULT_NUM_CLASSES = 1
"""v0.2.0 ships single-logit BCE: NUM_CLASSES=1 means one output whose
sigmoid is the calibrated P(fibrotic). NUM_CLASSES >= 2 switches the
head to a softmax + cross-entropy multi-class head — kept in the model
constructor for the eventual ``FibroticTypeLabel`` upstream policy."""

# ---------------------------------------------------------------------------
# Stochastic depth (design doc Section 10)
# ---------------------------------------------------------------------------

DEFAULT_STOCHASTIC_DEPTH = 0.1
"""DropPath probability ramps linearly from 0.0 at the stem to this
value at the final block."""

# ---------------------------------------------------------------------------
# Label definition (data schema)
# ---------------------------------------------------------------------------

HEALTHY_LABEL = 0
FIBROTIC_LABEL = 1
"""Integer codes for the two binary classes. The labels_dict on the
ClassifierBank carries the {int -> name} mapping; this is just the
integer that the model's output gets compared against."""

# ---------------------------------------------------------------------------
# Patient-aware split (design doc Section 13)
# ---------------------------------------------------------------------------

DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
"""Train / val / test. Splitting is patient-aware: every trace from
one simulation (synthetic) or one patient_id (IAFDB) lands entirely
in one split. egm-data's ``patient_aware_split`` reads
``trace_metadata['patient_id']`` and the producer's builder stamps
that field."""

DEFAULT_SPLIT_SEED = 0

# ---------------------------------------------------------------------------
# ONNX export (design doc Section 14)
# ---------------------------------------------------------------------------

DEFAULT_ONNX_OPSET = 18
"""ONNX operator-set version pinned at export time.

Covers Conv1d, BatchNorm1d, LayerNorm, SiLU, GELU, MultiheadAttention
plus everything the dynamo exporter touches under modern PyTorch.

PyTorch 2.5+ requires opset >= 18 in its dynamo exporter — asking for
opset 17 triggers an auto-downconvert pass that currently fails on
``Squeeze`` / ``Unsqueeze`` axes-attribute conversions and leaves the
file at opset 18 anyway. Setting 18 directly skips the broken
downconvert. Supported by current ONNX Runtime + TensorRT 8.6+/10
releases.

Fixed input shape (B, 1, T); only the batch axis is dynamic."""

DEFAULT_NORMALIZATION_SCHEME = "zscore"
"""Per-trace normalization recorded in the exported model_metadata
sidecar and applied by the export-time calibration loader. ``zscore``
matches the v1 training default (:data:`DEFAULT_ZNORM = True`).
``zero2one`` is supported for export-side experimentation; training +
eval support for it is tracked separately."""

DEFAULT_DECISION_THRESHOLD = 0.5
"""Decision threshold baked into the exported metadata's
``decision.threshold`` field. Matches the eval CLI default; at 0.5,
label_pred is invariant under any positive temperature scaling of the
logits, so deferring calibration to export time doesn't shift the
runtime's classification behavior."""

DEFAULT_CLASS_LABELS = ("healthy", "fibrotic")
"""Human-readable class names for the v1 binary head. Position
matches the integer code (index 0 = healthy, index 1 = fibrotic;
matches :data:`HEALTHY_LABEL` and :data:`FIBROTIC_LABEL`)."""

DEFAULT_EXPORT_NAME = "best"
"""Base filename for the exported artifact pair: produces
``<name>.onnx`` + ``<name>.model_metadata.json`` under the configured
output directory. Matches the convention in the
``egm_class_model_metadata`` schema docs."""

# ---------------------------------------------------------------------------
# Data loader / per-trace augmentation (design doc Section 11 + 12)
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0
"""``0`` keeps DataLoader single-process; raise on a multi-core box once
the bank is too big to comfortably fit in the main-process working set."""

DEFAULT_ZNORM = True
"""Per-trace z-score normalization (Sec. 11). Removes the
amplitude axis the train-time gain augmentation would otherwise leak
through, and makes the loss insensitive to per-bank gain drifts."""

DEFAULT_ZNORM_EPS = 1e-7
"""Floor on the per-trace std used inside the z-score normalization
to avoid division-by-zero on flat (all-zero) trace inputs. egm-data
ships no default; the executable owns the policy value."""

DEFAULT_AUGMENT_TRAIN = True
"""Apply gain + time-shift augmentation to the train split only."""

DEFAULT_MAX_GAIN = 0.0
"""Train-time random gain in [1-max_gain, 1+max_gain]. Multiplying a
trace by any positive scalar before z-scoring cancels exactly (mean
and std scale by the same factor), so the gain step is a no-op under
:data:`DEFAULT_ZNORM = True`. Default off; set to >0 explicitly if
you also disable znorm in a given config."""

DEFAULT_MAX_SHIFT_FRAC = 0.10
"""Train-time random time shift, as a fraction of input_length."""

DEFAULT_PIN_MEMORY = False
"""``True`` only helps on CUDA-backed DataLoaders; default off so a
CPU-only laptop run doesn't pay the pinned-memory overhead."""

# ---------------------------------------------------------------------------
# Training loop (design doc Section 12)
# ---------------------------------------------------------------------------

DEFAULT_EPOCHS = 60
DEFAULT_LR = 3e-4
"""AdamW base learning rate. The cosine schedule decays this to 0
after a 5% warmup."""

DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_WARMUP_FRAC = 0.05
"""Linear warmup as a fraction of total steps."""

DEFAULT_GRAD_CLIP_NORM: float | None = 1.0
"""``None`` disables clipping; 1.0 is a safe default under AMP."""

DEFAULT_AMP = False
"""Mixed-precision toggle. Off by default — turn on once you've got
a GPU big enough to benefit."""

DEFAULT_AMP_DTYPE = "bf16"
"""``"bf16"`` on Ampere+ (no GradScaler needed); ``"fp16"`` otherwise."""

DEFAULT_SEED = 42
DEFAULT_SELECT_METRIC = "auroc"
"""Per-epoch validation metric to maximize when picking the best
checkpoint. AUROC is threshold-free and robust under the typical
fibrosis class imbalance."""
