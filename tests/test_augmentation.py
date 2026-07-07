"""Tests for the per-trace TraceTransform.

The transform is pure numpy and torch-free, so it tests cleanly without
any optional dependency.
"""

from __future__ import annotations

import numpy as np

from myocard_egm_classifier.data.augmentation import TraceTransform


def _tf(**overrides: object) -> TraceTransform:
    """Build a TraceTransform with explicit defaults baked into the test only.

    The library itself ships no defaults; tests choose values here so the
    individual test cases stay readable. Any test-level "default" is just
    a choice this test file is making for its own legibility.
    """
    fields: dict[str, object] = {
        "input_length": 512,
        "znorm": True,
        "znorm_eps": 1e-6,
        "augment": False,
        "max_gain": 0.0,
        "max_shift_frac": 0.0,
    }
    fields.update(overrides)
    return TraceTransform(**fields)  # type: ignore[arg-type]


def test_znorm_zero_mean_unit_std_on_valid_region() -> None:
    """Per-trace z-score must produce a trace with mean ~= 0 and std ~= 1
    over the valid region. Catches a numerical bug in step 2 of the
    pipeline if anything ever broke the centering or scaling math."""
    tf = _tf(znorm=True)
    rng = np.random.default_rng(0)
    trace = rng.standard_normal(512).astype(np.float32) * 3.0 + 0.5
    out = tf(trace, rng)
    # Valid region (the whole 512) is the entire output.
    assert out.shape == (512,)
    assert abs(float(out.mean())) < 1e-5
    assert abs(float(out.std()) - 1.0) < 1e-5


def test_pad_when_shorter_than_target() -> None:
    """A trace shorter than `input_length` is z-scored first then placed
    into a zero buffer. The padding equals the post-z-score mean (0), so
    no DC step appears at the valid/padded boundary — important for the
    first conv layer not seeing an artificial edge."""
    tf = _tf(znorm=True)
    rng = np.random.default_rng(0)
    trace = rng.standard_normal(256).astype(np.float32)
    out = tf(trace, rng)
    assert out.shape == (512,)
    # The zero-padded regions match the z-scored mean (zero), so the
    # overall mean stays at zero.
    assert abs(float(out.mean())) < 1e-5


def test_center_crop_when_longer_than_target() -> None:
    """When a trace is longer than `input_length`, the transform takes a
    centered slice using `(L//2) - (T//2)` start position. Sanity-check
    the formula on a known case (L=800, T=512 -> start=144). Defensive
    branch today; will become live once banks emit raw L>T traces."""
    tf = _tf(znorm=False)
    rng = np.random.default_rng(0)
    trace = np.arange(800, dtype=np.float32)
    out = tf(trace, rng)
    assert out.shape == (512,)
    # Start = (800 // 2) - (512 // 2) = 400 - 256 = 144; first sample of
    # the valid region is 144.
    assert out[0] == 144.0


def test_augmentation_is_seed_deterministic() -> None:
    """Same input + same RNG seed -> identical augmented output. Confirms
    the augmentation stream is reproducible given (seed, epoch, row),
    which is what makes training-run comparisons meaningful."""
    tf = _tf(znorm=False, augment=True, max_gain=0.2, max_shift_frac=0.1)
    trace = np.arange(512, dtype=np.float32)
    a = tf(trace, np.random.default_rng(42))
    b = tf(trace, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_warns_when_gain_and_znorm_both_on() -> None:
    """Construct a transform with both `znorm=True` and `max_gain>0`; the
    __post_init__ check should emit a UserWarning. This is the user-facing
    signal that "random gain is mathematically zero work when z-score is
    on" — see trace_transform module docstring for the why."""
    import pytest

    with pytest.warns(UserWarning, match="random gain is a no-op when znorm=True"):
        _tf(znorm=True, augment=True, max_gain=0.2)


def test_no_warning_when_gain_zero_or_znorm_off() -> None:
    """The three legitimate configs (gain-disabled, znorm-disabled, or
    augmentation-off entirely) should NOT trip the warning. Otherwise
    every dataloader build with default settings would noise up the log."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes a failure
        _tf(znorm=True, augment=True, max_gain=0.0)
        _tf(znorm=False, augment=True, max_gain=0.2)
        _tf(znorm=True, augment=False, max_gain=0.2)
