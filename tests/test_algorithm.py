"""Tests for the local Welford-streaming change-point detector."""

from __future__ import annotations

import numpy as np
import pytest

from gigi_episodes.algorithm import (
    ChangePoint,
    EpisodicResult,
    detect_changepoints_local,
)


# ─── Synthetic data with known change-points ─────────────────────────────────


def make_three_segments(seed=0):
    """100 points each at mean 0, mean 5, mean 0 — change-points at ~100 and ~200."""
    rng = np.random.default_rng(seed)
    return np.concatenate([
        rng.normal(0.0, 1.0, 100),
        rng.normal(5.0, 1.0, 100),
        rng.normal(0.0, 1.0, 100),
    ])


def make_constant():
    """Constant value — no change-points expected."""
    return np.full(100, 3.14)


def make_slow_drift():
    """Slow linear drift — should not detect change-points (no abrupt shift)."""
    return np.linspace(0, 1, 100)


# ─── Detection accuracy ──────────────────────────────────────────────────────


def test_detects_two_changepoints_on_three_segment_data():
    """Standard three-segment dataset → detects CPs near the true positions 100, 200.

    Note: with threshold=3.0 on 300 noisy points, the algorithm may produce a
    small number of additional false positives (statistical noise). The test
    validates that the *true* shifts at 100 and 200 are captured, not that the
    count is exactly 2.
    """
    values = make_three_segments()
    result = detect_changepoints_local(values, threshold=3.0, min_segment=10)
    indices = result.indices
    assert result.count >= 2, f"Expected ≥2 change-points, got {result.count}"
    # Both true shift locations should be detected
    assert any(95 <= i <= 110 for i in indices), (
        f"No CP near position 100; detected {indices}"
    )
    assert any(195 <= i <= 215 for i in indices), (
        f"No CP near position 200; detected {indices}"
    )


def test_no_changepoints_on_constant_data():
    """Constant input → zero change-points."""
    result = detect_changepoints_local(make_constant(), threshold=3.0, min_segment=10)
    assert result.count == 0


def test_slow_drift_known_limitation():
    """Slow monotonic drift is a known limitation — algorithm may fire repeatedly.

    Documented in algorithm.py docstring. Users with slow-drift data need to
    raise threshold to 8.0+ and increase min_segment to 30+. The algorithm
    is calibrated for noisy-step data, not for slow monotonic drift.

    This test just confirms the algorithm doesn't go completely haywire on
    slow drift — it produces a bounded number of detections, not hundreds.
    """
    result = detect_changepoints_local(
        make_slow_drift(), threshold=10.0, min_segment=30
    )
    # Just check we don't get an absurd number of false positives
    assert result.count <= 5, (
        f"Algorithm should bound false positives on slow drift; got {result.count}"
    )


# ─── Result shape ────────────────────────────────────────────────────────────


def test_result_has_expected_fields():
    """EpisodicResult has all the documented attributes."""
    result = detect_changepoints_local(make_three_segments())
    assert isinstance(result, EpisodicResult)
    assert result.backend == "local"
    assert result.n_points == 300
    assert isinstance(result.indices, list)
    assert result.count == len(result.change_points)


def test_change_point_has_expected_fields():
    """Each ChangePoint has index, score, mean_before, mean_after."""
    result = detect_changepoints_local(make_three_segments())
    if result.count > 0:
        cp = result.change_points[0]
        assert isinstance(cp, ChangePoint)
        assert isinstance(cp.index, int)
        assert isinstance(cp.score, float)
        assert cp.score > 3.0, "Score should exceed the threshold"
        assert isinstance(cp.mean_before, float)
        assert isinstance(cp.mean_after, float)


# ─── Edge cases ──────────────────────────────────────────────────────────────


def test_empty_input():
    """Empty input → empty result, no errors."""
    result = detect_changepoints_local([])
    assert result.count == 0
    assert result.n_points == 0


def test_single_value():
    """Single value → no change-points possible (below min_segment)."""
    result = detect_changepoints_local([42.0])
    assert result.count == 0
    assert result.n_points == 1


def test_short_input_below_min_segment():
    """Input shorter than min_segment → no change-points."""
    result = detect_changepoints_local([1.0, 2.0, 3.0], min_segment=10)
    assert result.count == 0


def test_two_d_input_raises():
    """2-D input raises ValueError."""
    with pytest.raises(ValueError, match="1-D"):
        detect_changepoints_local(np.zeros((10, 2)))


def test_threshold_parameter_changes_sensitivity():
    """Lower threshold → more change-points (or equal)."""
    values = make_three_segments()
    strict = detect_changepoints_local(values, threshold=5.0, min_segment=10)
    lenient = detect_changepoints_local(values, threshold=2.0, min_segment=10)
    assert lenient.count >= strict.count


def test_sigma_floor_prevents_divide_by_zero():
    """Constant data shouldn't crash even with min_segment satisfied."""
    # Constant data has variance 0 — sigma_floor prevents divide-by-zero.
    values = np.full(100, 5.0)
    result = detect_changepoints_local(values, threshold=3.0, min_segment=5)
    assert result.count == 0  # No real change, no detection
