"""Tests for the LocalBackend / GigiBackend integration surface."""

from __future__ import annotations

import numpy as np
import pytest

from gigi_episodes import (
    EpisodicResult,
    GigiBackend,
    LocalBackend,
    find_changepoints,
)


def make_signal():
    rng = np.random.default_rng(42)
    return np.concatenate([
        rng.normal(0.0, 1.0, 100),
        rng.normal(5.0, 1.0, 100),
    ])


def test_default_backend_is_local():
    """Calling find_changepoints with no backend should use LocalBackend."""
    result = find_changepoints(make_signal())
    assert result.backend == "local"


def test_local_backend_returns_episodic_result():
    """LocalBackend.detect returns an EpisodicResult."""
    result = LocalBackend().detect(make_signal())
    assert isinstance(result, EpisodicResult)
    assert result.backend == "local"
    assert result.n_points == 200


def test_local_backend_detects_obvious_shift():
    """The two-segment signal should produce at least one change-point."""
    result = find_changepoints(make_signal())
    assert result.count >= 1


def test_gigi_backend_constructor():
    """GigiBackend can be constructed without connecting (lazy)."""
    backend = GigiBackend(
        url="http://localhost:3142",
        api_key="dev-local",
        bundle="test_bundle",
        field="value",
    )
    assert backend.name == "gigi"
    assert backend.url == "http://localhost:3142"
    assert backend.bundle == "test_bundle"


def test_gigi_backend_raises_on_unreachable():
    """GigiBackend raises RuntimeError when GIGI is unreachable."""
    backend = GigiBackend(
        url="http://localhost:1",  # unreachable
        bundle="test_bundle",
        field="value",
        timeout=1.0,
    )
    with pytest.raises(RuntimeError, match="GigiBackend"):
        backend.detect(values=None)


def test_threshold_pass_through_to_backend():
    """The `threshold` arg of find_changepoints reaches the backend."""
    # With very high threshold, no detections expected even on obvious data
    result = find_changepoints(make_signal(), threshold=100.0)
    assert result.count == 0
    assert result.threshold == 100.0


def test_min_segment_pass_through_to_backend():
    """The `min_segment` arg reaches the backend."""
    result = find_changepoints(make_signal(), min_segment=50)
    assert result.min_segment == 50
