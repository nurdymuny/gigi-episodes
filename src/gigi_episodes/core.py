"""Top-level public API for gigi-episodes.

The single entry point users typically need is :func:`find_changepoints`.
For lower-level access, see :mod:`gigi_episodes.algorithm` and
:mod:`gigi_episodes.backends`.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .algorithm import EpisodicResult
from .backends import LocalBackend, _BaseBackend


def find_changepoints(
    values: Optional[Sequence[float]] = None,
    *,
    backend: Optional[_BaseBackend] = None,
    threshold: float = 3.0,
    min_segment: int = 10,
) -> EpisodicResult:
    """Find change-points in a 1-D value sequence.

    The high-level entry point — wraps backend selection and parameter
    forwarding in a single function. For most users, just pass the values
    and accept the defaults.

    Parameters
    ----------
    values : sequence of numbers, optional
        The 1-D sequence to scan. Required for :class:`LocalBackend`;
        :class:`GigiBackend` ignores this argument (reads from its
        configured bundle/field instead).
    backend : backend instance, optional
        Which backend to use. Defaults to :class:`LocalBackend` if not given.
    threshold : float, default 3.0
        Number of standard errors above which a mean difference is flagged
        as a change-point. Higher = stricter = fewer detections.
    min_segment : int, default 10
        Minimum number of points required in the "before" segment before any
        change-point can be flagged.

    Returns
    -------
    EpisodicResult
        Detected change-points and metadata.

    Example
    -------
    Basic usage with the local backend::

        from gigi_episodes import find_changepoints
        result = find_changepoints([1, 1, 1, 5, 5, 5, 1, 1, 1])
        print(result.indices)

    Using a remote GIGI instance::

        from gigi_episodes import GigiBackend, find_changepoints
        backend = GigiBackend(
            url="http://localhost:3142",
            api_key="dev-local",
            bundle="latency",
            field="p99_ms",
        )
        result = find_changepoints(backend=backend)
    """
    if backend is None:
        backend = LocalBackend()

    return backend.detect(values, threshold=threshold, min_segment=min_segment)
