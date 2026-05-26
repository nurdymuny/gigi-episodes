"""gigi-episodes — change-point detection in value sequences.

A small, focused Python library that exposes GIGI's EPISODIC brain primitive
(change-point detection) as a portable tool. Use it to answer questions like:

- "When did our latency start drifting?"
- "When did this test suite start flaking?"
- "What are the episode boundaries in this time series?"

Two backends are provided:

- ``LocalBackend`` — pure-numpy, no external services required (default)
- ``GigiBackend`` — calls a running GIGI instance's ``/brain/episodic``
  endpoint for higher-fidelity detection on multivariate or anisotropic data

Quick start::

    from gigi_episodes import find_changepoints

    values = [1.0, 1.1, 0.9, 1.05, 1.0, 5.0, 5.1, 5.05, 5.0, 5.1]
    result = find_changepoints(values)
    print(result.indices)   # [5]
    print(result.count)     # 1

See https://github.com/nurdymuny/gigi-episodes for documentation and examples.
"""

from .algorithm import (
    ChangePoint,
    EpisodicResult,
    detect_changepoints_local,
)
from .backends import GigiBackend, LocalBackend
from .core import find_changepoints

__version__ = "0.1.0"
__all__ = [
    "ChangePoint",
    "EpisodicResult",
    "GigiBackend",
    "LocalBackend",
    "detect_changepoints_local",
    "find_changepoints",
]
