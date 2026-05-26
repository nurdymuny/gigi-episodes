"""Backends for gigi-episodes.

Two backends are supported:

- :class:`LocalBackend` — pure-numpy, the default. Use this when you don't have
  a running GIGI instance, or when you want zero infrastructure dependencies.

- :class:`GigiBackend` — calls a running GIGI instance's
  ``POST /v1/bundles/{name}/brain/episodic`` endpoint. Use this when you have
  GIGI running and want the engine's full Kähler-aware Welford fit including
  L13.3 diagonal-Gaussian support and L13.7 denominator floor for numerical
  stability on anisotropic data.

Both backends honor the same parameters (``threshold``, ``min_segment``) so
swapping between them is a one-line change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .algorithm import EpisodicResult, detect_changepoints_local


class _BaseBackend:
    """Abstract base — backends must implement :meth:`detect`."""

    name: str = "base"

    def detect(
        self,
        values: Sequence[float],
        *,
        threshold: float = 3.0,
        min_segment: int = 10,
    ) -> EpisodicResult:
        raise NotImplementedError


class LocalBackend(_BaseBackend):
    """Pure-numpy change-point detection, no external services.

    This is the default backend. Algorithm parity with GIGI's
    ``/brain/episodic`` for the 1-D case, including L13.6 stability guard and
    L13.7-style denominator floor.

    Example::

        from gigi_episodes import LocalBackend, find_changepoints

        backend = LocalBackend()
        result = find_changepoints([1, 1, 1, 5, 5, 5], backend=backend)
    """

    name = "local"

    def detect(
        self,
        values: Sequence[float],
        *,
        threshold: float = 3.0,
        min_segment: int = 10,
    ) -> EpisodicResult:
        """Detect change-points using local Welford streaming.

        See :func:`gigi_episodes.algorithm.detect_changepoints_local` for
        parameter details.
        """
        return detect_changepoints_local(
            values, threshold=threshold, min_segment=min_segment
        )


@dataclass
class GigiBackend(_BaseBackend):
    """Change-point detection backed by a running GIGI instance.

    Calls ``POST /v1/bundles/{bundle}/brain/episodic`` with the supplied
    field. Requires the bundle to exist and contain the values you want to
    analyze; this backend does not auto-create bundles in v0.1.

    Parameters
    ----------
    url : str
        GIGI server URL (e.g. ``"http://localhost:3142"``).
    api_key : str, optional
        API key sent in the ``Authorization: Bearer`` header.
    bundle : str
        Bundle name containing the values.
    field : str
        Field name inside the bundle to analyze.

    Example::

        from gigi_episodes import GigiBackend, find_changepoints

        backend = GigiBackend(
            url="http://localhost:3142",
            api_key="dev-local",
            bundle="latency_metrics",
            field="p99_ms",
        )
        result = find_changepoints(None, backend=backend)
        #   ↑ values argument ignored — GigiBackend reads from the bundle directly
    """

    url: str
    bundle: str
    field: str
    api_key: Optional[str] = None
    timeout: float = 30.0

    name: str = "gigi"

    def detect(
        self,
        values: Optional[Sequence[float]] = None,
        *,
        threshold: float = 3.0,
        min_segment: int = 10,
    ) -> EpisodicResult:
        """Detect change-points by calling GIGI's /brain/episodic endpoint.

        The ``values`` argument is ignored — GigiBackend reads from the
        configured bundle/field directly.
        """
        # Lazy import so the package doesn't require `requests` for LocalBackend users
        try:
            import requests
        except ImportError as e:
            raise RuntimeError(
                "GigiBackend requires the `requests` package. "
                "Install with: pip install gigi-episodes[gigi]"
            ) from e

        endpoint = f"{self.url.rstrip('/')}/v1/bundles/{self.bundle}/brain/episodic"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "field": self.field,
            "threshold": threshold,
            "min_segment": min_segment,
        }
        try:
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"GigiBackend request failed: {e}. "
                f"Falling back to LocalBackend is recommended if GIGI is unavailable."
            ) from e

        data = response.json()
        # Translate GIGI's response shape into EpisodicResult.
        # Expected shape: {"change_points": [{"index": int, "score": float, ...}], ...}
        from .algorithm import ChangePoint

        cps = []
        for cp_data in data.get("change_points", []):
            cps.append(
                ChangePoint(
                    index=int(cp_data["index"]),
                    score=float(cp_data.get("score", 0.0)),
                    mean_before=float(cp_data.get("mean_before", 0.0)),
                    mean_after=float(cp_data.get("mean_after", 0.0)),
                )
            )
        return EpisodicResult(
            change_points=cps,
            n_points=int(data.get("n_points", 0)),
            threshold=threshold,
            min_segment=min_segment,
            backend="gigi",
        )
