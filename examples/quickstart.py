"""gigi-episodes quickstart — three escalating examples.

Run:
    python examples/quickstart.py
"""

from __future__ import annotations

import numpy as np

from gigi_episodes import find_changepoints


def example_simple_step():
    """A two-segment noisy shift. The smallest "realistic" example."""
    print("=" * 64)
    print("  Example 1 - two noisy regimes (30 + 30 points)")
    print("=" * 64)
    rng = np.random.default_rng(7)
    values = np.concatenate([
        rng.normal(1.0, 0.3, 30),
        rng.normal(5.0, 0.3, 30),
    ])
    result = find_changepoints(values, min_segment=10)
    print(f"  60 points, 2 regimes (mean 1.0 to mean 5.0, sigma 0.3)")
    print(f"  detected {result.count} change-point(s) at: {result.indices}")
    for cp in result.change_points:
        print(f"    - index {cp.index}: {cp.mean_before:.2f} to {cp.mean_after:.2f}  (score {cp.score:.1f})")


def example_noisy_segments():
    """Three Gaussian segments — realistic operational data."""
    print()
    print("=" * 64)
    print("  Example 2 — three noisy regimes")
    print("=" * 64)
    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.normal(0.0, 1.0, 100),
        rng.normal(5.0, 1.0, 100),
        rng.normal(0.0, 1.0, 100),
    ])
    result = find_changepoints(values)
    print(f"  300 points, 3 segments (mean 0 to 5 to 0)")
    print(f"  detected {result.count} change-point(s) at: {result.indices}")
    for cp in result.change_points:
        print(f"    - index {cp.index}: {cp.mean_before:.2f} -> {cp.mean_after:.2f}  (score {cp.score:.1f})")


def example_no_change():
    """Pure noise with no real change — should return no change-points."""
    print()
    print("=" * 64)
    print("  Example 3 — noise only, no real shift")
    print("=" * 64)
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 1.0, 300)
    result = find_changepoints(values)
    print(f"  300 points, pure N(0,1) noise")
    print(f"  detected {result.count} change-point(s) at: {result.indices}")
    if result.count == 0:
        print(f"    [OK] correctly returned no spurious change-points")


def main():
    example_simple_step()
    example_noisy_segments()
    example_no_change()


if __name__ == "__main__":
    main()
