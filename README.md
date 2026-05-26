# gigi-episodes

**Change-point detection for time series.** The "when did this break?" tool, as a portable Python library + CLI.

```python
from gigi_episodes import find_changepoints

# Your latency over the last 1000 requests
latencies = [50.1, 49.8, 51.2, ..., 95.1, 96.3, 94.8, ...]

result = find_changepoints(latencies)
print(result.indices)   # [573]  — latency drift started around request 573
print(result.count)     # 1
```

```bash
$ gigi-episodes detect production.csv --column latency_p99
  source:      production.csv
  column:      latency_p99
  backend:     local
  threshold:   3.0
  min_segment: 10
  n_points:    1000
  changepoints: 1

  index    score   mean_before → mean_after
  -----    -----   ---------------------------
    573    12.83     50.247  →   95.103
```

## What it's for

Anywhere you have a sequence of values and want to know *when it changed*:

- **Ops** — when did latency start drifting? When did error rate spike?
- **Test flakiness** — when did this test suite start failing?
- **User behavior** — when did retention break in this cohort?
- **ML monitoring** — when did model accuracy drop?
- **Sensor data** — when did the reading shift to a new regime?

gigi-episodes is intentionally narrow: **change-point detection in 1-D value sequences, nothing else.** Other "EPISODIC" features (multivariate, anisotropic, fiber-bundle native) live in the [GIGI engine](https://davisgeometric.com) — gigi-episodes exposes one specific brain primitive as the smallest possible installable tool.

## Install

```bash
pip install gigi-episodes
```

Optional: install with the GIGI backend (requires `requests`):

```bash
pip install "gigi-episodes[gigi]"
```

## Quick start

```python
from gigi_episodes import find_changepoints

values = [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]
result = find_changepoints(values)

for cp in result.change_points:
    print(f"index {cp.index}: {cp.mean_before:.2f} → {cp.mean_after:.2f} (score {cp.score:.1f})")

# index 6: 1.00 → 5.00 (score 11.7)
# index 12: 5.00 → 1.00 (score 11.7)
```

## CLI

```bash
# Detect in a CSV column
gigi-episodes detect data.csv --column latency_ms

# JSON output for piping into other tools
gigi-episodes detect data.csv --column latency_ms --json

# Tune sensitivity
gigi-episodes detect data.csv --column latency_ms --threshold 4.0 --min-segment 30

# Read from stdin
cat values.txt | gigi-episodes detect - --column 0
```

## Tuning

| Parameter | Default | What higher means |
|-----------|---------|-------------------|
| `threshold` | `3.0` | stricter — fewer (more confident) detections; ~`σ` units of mean shift |
| `min_segment` | `10` | longer "warm-up" before any detection can fire |

Defaults are calibrated for clean operational data (latencies, counts, scores). For very noisy data, raise the threshold to 4.0 or 5.0. For long sequences with slow regime shifts, increase `min_segment` to 30+.

## Two backends

**`LocalBackend`** (default) — pure numpy. No external services. Use this 99% of the time.

```python
from gigi_episodes import LocalBackend, find_changepoints

result = find_changepoints(values, backend=LocalBackend())
```

**`GigiBackend`** — calls a running GIGI instance's `/brain/episodic` endpoint for higher-fidelity detection on multivariate or anisotropic data. Useful when you have data already in GIGI bundles and want the engine's full Kähler-aware Welford fit including L13.3 diagonal-Gaussian and L13.7 denominator-floor stability.

```python
from gigi_episodes import GigiBackend, find_changepoints

backend = GigiBackend(
    url="http://localhost:3142",
    api_key="dev-local",
    bundle="latency_metrics",
    field="p99_ms",
)
result = find_changepoints(backend=backend)
```

## The algorithm

gigi-episodes's local backend uses a **windowed two-sample mean-difference test** with global-scale floor:

1. For each candidate split point `i`, compute the mean of the segment so far (from the last change-point up to `i`) and the mean of the next `window` points.
2. Compute the standard error of the difference: `SE = sqrt(var_before/n_before + var_after/n_after)`.
3. If `|mean_after - mean_before| / SE > threshold`, it's a change-point.
4. Both variances are floored by a global MAD-based sigma estimate, so locally-tiny variance (from short or smooth segments) can't blow up the test statistic.

This is the 1-D specialization of GIGI's `/brain/episodic` endpoint — one of [twelve brain primitives](https://github.com/nurdymuny/gigi/blob/main/BRAIN_PRIMITIVES_CONSUMER_GUIDE.md) unified by the Friston master equation on a Kähler bundle. gigi-episodes makes this single primitive portable; the eleven others live in GIGI.

## License

MIT. Free for any use, commercial or otherwise. See [LICENSE](LICENSE).

## Related

- [GIGI](https://davisgeometric.com) — the fiber-bundle database engine; gigi-episodes's `GigiBackend` calls it. EPISODIC is one of twelve brain primitives.
- [DreamData](https://github.com/nurdymuny/dream-data) — synthetic-data CLI using GIGI's DREAM primitive. Sibling project.
- [gigi-mind](https://github.com/nurdymuny/gigi-mind) — VS Code extension exposing all twelve brain primitives. Sibling project.

## Status

**v0.1.0** — stable for the documented surface (1-D LocalBackend + CLI + GigiBackend skeleton). API may evolve in 0.x; will stabilize at 1.0.
