# `TraceTransform` pre-processing review

> Moved from egm-data to egm-classifier in the Refactor Step 8 code-placement
> audit, together with the `TraceTransform` code itself
> (`myocard_egm_classifier.data.augmentation`).

Scope: the per-trace transform in `src/myocard_egm_classifier/data/augmentation/trace_transform.py` only. Producer-side conditioning (bandpass, R-wave calibration, segmentation) and inference-time preprocessing (the `preprocessing` block of `model_metadata.json`) are out of scope for this pass.

## What it does today

Given a raw trace `x` of length `L` and a target length `T = input_length`:

```
1. (defensive) center-crop to T if L > T
2. (train only, znorm=False only) random gain      x ← x * U(1-g, 1+g)
3. (znorm=True) per-trace z-score                  x ← (x - mean(x)) / max(std(x), eps)
4. place into a length-T zero buffer at an offset
   - base offset:  (T - L_valid) // 2
   - (train only) jitter: offset += U[-S, +S]  where S = round(shift_frac * L_valid)
   - offset is clipped to [0, T - L_valid]
```

Per-trace draws use `np.random.default_rng((seed, epoch, row))`, so augmentation is deterministic across runs at the same `(seed, epoch, row)`.

## Step-by-step

### Step 1: defensive center-crop

Active only when `L > T`. With the current banks producing exactly `L = T = 512`, this branch is **dead code** in practice. It exists so a future bank with longer traces (say, 1000-sample windows for a Phase-2 fractionation classifier) doesn't crash the transform. The cropping is purely positional — no resampling, no anti-aliasing — so it implicitly assumes a future bank's center is the part you want.

Open question to flag for Phase 2: if a future bank really does produce L > T, is centered cropping always the right choice, or do we want a random crop at train time (different excerpt each epoch) and a centered crop at eval?

### Step 2: random gain (now skipped when znorm=True)

`x ← x * U(1-g, 1+g)`, default `g = 0.2` (±20%). After the recent fix this step is a strict no-op when `znorm=True`, because z-score is invariant under any positive scalar: `znorm(c·x) = znorm(x)` for `c > 0`. With `znorm=False` it's a real amplitude jitter, used to simulate variation in electrode-tissue contact pressure.

Worth knowing: amplitude jitter is **only** a useful augmentation if the model is allowed to use absolute amplitude as a feature. The instant you turn z-score on, you've explicitly decided that absolute amplitude is *not* a feature, so gain becomes meaningless. This is the right call for IAFDB → synthetic transfer (different recording equipment ⇒ different absolute amplitudes; the substrate signal lives in morphology, not amplitude).

### Step 3: per-trace z-score

`x ← (x - mean(x)) / max(std(x), eps)`. The most consequential step in the file. Two things it deliberately throws away:

1. **DC offset** — usually noise (electrode drift between segments). Always safe to drop.
2. **Absolute amplitude** — this is the load-bearing decision. In the cardiology-EP literature, the classical voltage-threshold approach to scar identification (Marchlinski 2000; Sanders 2003) uses bipolar peak-to-peak amplitude < 0.5 mV as a hard scar marker. Per-trace z-score **discards exactly this feature**.

That's not a bug — it's the design choice. The rationale (also why we did R-wave calibration on the IAFDB side) is that *between recordings*, absolute amplitude varies for reasons that have nothing to do with the substrate: electrode contact, catheter angle, gain settings on the recorder. Carrying amplitude into the model in raw mV would let it shortcut on equipment artifacts. The cost is that the model has to learn *morphological* fibrosis markers (fractionation, peak count, low-frequency content) instead of just thresholding the amplitude.

If you ever want amplitude back as a feature, the right move isn't to turn z-score off — it's to add a separate "calibrated amplitude" scalar feature alongside the z-scored waveform. The amplitude can ride along the model as a second input head, and the model can learn how heavily to weight it.

The `znorm_eps = 1e-6` std floor handles degenerate traces (a completely flat segment). Without it, a flat trace would divide by zero. With it, a flat trace becomes a buffer of zeros — which is correct, because a flat trace has no information regardless.

**Subtle point on padding interaction.** Because z-score happens on the *valid* region *before* it's placed into the zero buffer, any zero-padded samples coincide with the post-z-score mean (which is 0). No DC step at the boundary, no spurious edge artifact for a convolutional first layer to latch onto. That's why step 3 has to come before step 4, not the other way around.

### Step 4: placement with optional time-shift jitter

The trace is placed into a length-T zero buffer at `offset = base_offset ± jitter`. Two things happen here.

**The shift jitter is currently a no-op for the Phase-1 bank.** This is the issue worth flagging most clearly. The shift is clipped to `[0, T - L_valid]`. For the current banks, `L_valid = T = 512`, so the allowed range collapses to `[0, 0]` — the trace can only land at offset 0, no shift is possible. The `max_shift_frac = 0.10` knob does literally nothing today.

Why did this happen? The augmentation was designed for an earlier bank that produced `L = 200` traces; with `T = 512`, that gave 312 samples of headroom and a meaningful ±10% shift range (~20 samples). The simulator was later updated to emit `L = 512` to match `T`, and the augmentation silently lost its only working knob in z-score mode. Test coverage doesn't flag it because nothing asserts that augmentation actually changes the output for a Phase-1-shaped trace.

There are three legitimate ways to restore working time-shift augmentation, all of them require a producer-side or contract-side change:

- **(a) Have the producer emit `L > T`** — the simplest fix. Generate, say, `L = 612` so there are ~100 samples of headroom. T stays at 512, the model sees the same window length, and the train-time transform can shift the window within the headroom.
- **(b) Circular shift inside the same length-T window** — `np.roll(x, k)`. Cheap, but cyclical artifacts at the boundary are unphysical for a non-periodic EGM segment. Probably not what we want.
- **(c) Replace static segmentation with stochastic segmentation** — i.e., have the producer hold longer raw windows on disk and let the dataset cut a length-T excerpt at a random start each `__getitem__`. Most flexible, biggest architectural change, and arguably the right answer for a Phase-2 system.

The hot-fix is (a). It's a contracts-side thing (synthetic_bank's `trace_duration_ms` is free to vary) and a producer-side thing (synthetic_egm_pipeline emits longer raw traces); the transform code here doesn't change.

### Other things `TraceTransform` deliberately doesn't do

For completeness, in case the absence is what you wanted to ask about:

| Step | Why it's not here |
| --- | --- |
| Bandpass filter | Producer-side; baked into the bank by IAFDB's R-wave-calibration pipeline (band_hz attr) and into the synthetic bank by the noise mixer. Re-filtering at train time would be redundant work and would risk drift if the on-disk bandpass and the train-time bandpass disagreed. Inference-time bandpass is recorded in `model_metadata.json.preprocessing.bandpass_hz`. |
| Resampling | Producer-side. Both banks ship at exactly 1000 Hz to match the model input rate (recorded as `fs_hz` in the bank's root attrs). A future bank at a different rate would need this in the transform or in inference. |
| Clipping / outlier removal | Not done. After z-score, values are typically within ±3σ. Spikes from electrode pop or non-stationary noise could exceed that; if it becomes a problem in practice, clip to `±5` after step 3. Cheap to add. |
| Detrending (linear / quadratic) | Bandpass at the producer handles low-frequency drift; per-trace z-score removes any DC residual. Adding polynomial detrending on top of those is redundant. |
| Additive noise augmentation | Already baked into the **noise-mixed** synthetic bank at producer time (clean + IAFDB noise at a target SNR). Adding more noise at train time on top of that would muddle the SNR labels the bank carries. |
| Polarity flip | Bipolar EGMs are signed (pair order matters); flipping the polarity flips the physical interpretation of activation direction. Not a valid augmentation here, even though it's standard for symmetric domains like image classification. |
| Per-channel normalization | Trivial here because every trace is 1-channel (single bipolar pair). If we ever move to multi-channel inputs (e.g. a 5×5 grid stacked as a 25-channel input), per-channel z-score would replace step 3, and we'd want to think hard about whether each electrode pair has its own statistics or shares one. |

## Recommendations

1. **Land the gain + z-score warning** — done, just noting it for completeness.
2. **Time-shift no-op: defer to Phase 1.5+ producer-side fix.** The right answer is for the synthetic-egm-pipeline (and eventually the IAFDB pipeline) to emit raw traces longer than `T`, and for the dataset to take a random length-`T` crop each `__getitem__` at train time. The zero-pad-and-shift approach was a workaround for an earlier `L < T` regime and shouldn't outlive the regeneration; it'll matter more once we move to multi-activation substrates, where stochastic windowing is the only sensible answer. Until that lands, the docstring should call out the no-op so nobody is misled.
3. ~~Consider adding a `clip_at_sigma` knob~~ **Out of scope for this module.** `TraceTransform` is solely for per-call train-time augmentations that vary between training runs. Static per-file preprocessing or once-per-dataset curation lives elsewhere (producer pipelines or a separate curation module). Clipping is a static signal-conditioning step, so it doesn't belong here even if it turns out to be useful later.
4. **"No amplitude as a feature" decision** — deferred. Worth a project-level note eventually, but not blocking the refactor. If we ever ship a model variant that learns on absolute voltage instead of z-scored signal, we revisit then.
5. **`max_gain` knob stays for now.** Refactor velocity is more important than 100% future-proofing. The warning + skip behavior is enough; we're not adding or removing knobs in this pass.

## Decisions recorded

- Time-shift augmentation will be replaced by producer-side `L > T` + dataset-level random crop. Filed as a Phase 1.5+ task. The current no-op gets documented in the docstring until then.
- No new knobs go into `TraceTransform` during the refactor.
- This module is for per-call train-time augmentations only — static per-file or once-per-dataset preprocessing belongs somewhere else.
