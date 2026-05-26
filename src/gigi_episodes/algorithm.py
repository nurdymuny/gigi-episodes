"""Pure-numpy change-point detection — the Welford-streaming heart of gigi-episodes.

Implements an online change-point detector based on running mean / variance
estimation (Welford's algorithm) and z-score thresholding. When a new
observation lies more than `threshold` standard deviations from the running
mean (after a minimum warm-up segment), it's flagged as a change-point and
the accumulator resets.

This is a faithful local implementation of the algorithm GIGI's `/brain/episodic`
endpoint exposes. For higher-fidelity detection on multivariate or anisotropic
data, use `GigiBackend` instead — that path uses the engine's full Kähler-aware
Welford fit including L13.3 diagonal-Gaussian support and L13.7 denominator
floor for numerical stability.

The math:
    Welford accumulators per segment:
        count_n  := number of points since last reset
        mean_n   := sum(x_i) / count_n
        M2_n     := sum((x_i - mean_n)^2)
    Variance estimate:
        var_n    := M2_n / count_n
    Score for new point x_{n+1}:
        z        := |x_{n+1} - mean_n| / sqrt(var_n + sigma_floor^2)
    Decision: z > threshold AND count_n >= min_segment → change-point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Union

import numpy as np


Number = Union[int, float, np.floating, np.integer]


@dataclass
class ChangePoint:
    """A single detected change-point in a value sequence.

    Attributes
    ----------
    index : int
        Position in the input sequence where the change was detected
        (zero-based).
    score : float
        Z-score that triggered the detection — the standardized magnitude
        of the deviation from the pre-change running mean. Higher = more
        confident this is a real change.
    mean_before : float
        Running mean immediately before the change-point. Useful for
        interpreting the magnitude of the shift.
    mean_after : float
        Running mean of the segment that follows (computed post-hoc
        from the values after the change-point, up to the next CP or
        end of sequence).
    """

    index: int
    score: float
    mean_before: float
    mean_after: float


@dataclass
class EpisodicResult:
    """Result of running change-point detection on a value sequence.

    Attributes
    ----------
    change_points : list of ChangePoint
        Detected change-points, in order of appearance.
    n_points : int
        Total number of input values examined.
    threshold : float
        Z-score threshold used.
    min_segment : int
        Minimum segment length used.
    backend : str
        Which backend produced the result ("local" or "gigi").
    """

    change_points: List[ChangePoint] = field(default_factory=list)
    n_points: int = 0
    threshold: float = 0.0
    min_segment: int = 0
    backend: str = "local"

    @property
    def indices(self) -> List[int]:
        """Just the change-point indices, in order."""
        return [cp.index for cp in self.change_points]

    @property
    def count(self) -> int:
        """Number of change-points detected."""
        return len(self.change_points)


def detect_changepoints_local(
    values: Sequence[Number],
    *,
    threshold: float = 3.0,
    min_segment: int = 10,
    sigma_floor: float = 1e-6,
    window: int = None,
) -> EpisodicResult:
    """Detect change-points in a 1-D value sequence by comparing means of windows.

    Uses the "two-sample mean difference, scaled by standard error" statistic:
    for each candidate split point `i`, compare the mean of the *segment so
    far* (from the previous change-point up to `i`) against the mean of the
    next `window` points. If the difference is more than `threshold` standard
    errors, it's a change-point.

    This is dramatically more robust to single-point outliers than per-point
    z-scoring, because both sides of the comparison are window means — a
    single outlier in the forward window only shifts the forward mean by a
    fraction of itself.

    Pure-numpy / pure-Python; no external services required.

    Parameters
    ----------
    values : sequence of numbers
        The 1-D sequence to scan for change-points.
    threshold : float, default 3.0
        Number of standard errors of the mean above which a window-mean
        difference is flagged as a change-point. 3.0 ≈ 99.7% for Gaussian noise.
    min_segment : int, default 10
        Minimum number of points required in the "before" segment before any
        change-point can be flagged. Larger values give more reliable
        variance estimates and reduce false positives early in the sequence.
    sigma_floor : float, default 1e-6
        Lower bound added in quadrature to the estimated standard deviation
        when computing the standard error. Prevents division-by-zero on
        constant segments. (L13.6-style stability guard.)
    window : int, optional
        Size of the forward window used to estimate the "after" mean. Defaults
        to ``max(5, min_segment // 2)``. Larger window = more robust to
        outliers but slower to detect sharp changes.

    Returns
    -------
    EpisodicResult
        Detected change-points plus metadata.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D sequence, got shape {arr.shape}")
    n = arr.size
    if window is None:
        window = max(5, min_segment // 2)

    if n == 0:
        return EpisodicResult(
            change_points=[],
            n_points=0,
            threshold=threshold,
            min_segment=min_segment,
            backend="local",
        )

    # Robust noise-scale floor via MAD of consecutive differences.
    # Rationale: MAD of values gets inflated by genuine step shifts (the shift
    # itself contributes to dispersion from the median), which suppresses real
    # detections. MAD of *differences* is insensitive to step shifts — it only
    # captures within-segment noise. For Gaussian noise, consecutive diffs have
    # std σ√2, so we divide by √2 to recover the noise sigma.
    #
    # Limitation: slow monotonic drift (e.g., linspace 0→1) has near-zero
    # consecutive diffs, so noise_sigma ≈ 0, and small mean differences between
    # adjacent windows can still trigger detections. Slow-drift data needs
    # higher threshold and larger min_segment to avoid false positives.
    if n >= 2:
        diffs = np.diff(arr)
        diff_mad = float(np.median(np.abs(diffs - np.median(diffs))))
        noise_sigma = max(diff_mad * 1.4826 / math.sqrt(2.0), sigma_floor)
    else:
        noise_sigma = sigma_floor
    sigma_floor_sq = noise_sigma * noise_sigma

    cps: List[ChangePoint] = []
    seg_start = 0
    i = min_segment  # need at least min_segment in the "before" segment

    while i < n:
        before = arr[seg_start:i]
        if before.size < min_segment:
            i += 1
            continue

        end = min(i + window, n)
        after = arr[i:end]
        if after.size < 3:
            break  # too few forward points to compare

        mean_b = float(before.mean())
        mean_a = float(after.mean())

        # Welch-style standard error of the difference in means.
        # Floor both variances by the global MAD-based sigma — otherwise smooth
        # or short segments report variance ≈ 0 and any movement looks like ∞σ.
        var_b = max(float(before.var()), sigma_floor_sq)
        var_a = max(float(after.var()), sigma_floor_sq)
        se = math.sqrt(var_b / before.size + var_a / after.size)
        if se < sigma_floor:
            se = sigma_floor
        z = abs(mean_a - mean_b) / se

        if z > threshold:
            cps.append(
                ChangePoint(
                    index=i,
                    score=z,
                    mean_before=mean_b,
                    mean_after=mean_a,
                )
            )
            seg_start = i
            # Skip past the forward window to avoid re-detecting the same shift
            i = i + window
        else:
            i += 1

    # Post-hoc refinement: replace each CP's mean_after with the actual mean of
    # the segment that follows it (up to the next CP or end of sequence).
    if cps:
        boundaries = [cp.index for cp in cps] + [n]
        for k, cp in enumerate(cps):
            seg = arr[cp.index : boundaries[k + 1]]
            if seg.size > 0:
                cp.mean_after = float(np.mean(seg))

    return EpisodicResult(
        change_points=cps,
        n_points=n,
        threshold=threshold,
        min_segment=min_segment,
        backend="local",
    )
