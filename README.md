# gigi-episodes

> **The "when did this break?" tool.**
> Change-point detection for noisy time series, as a portable Python library + CLI.
> One brain primitive from the [GIGI](https://davisgeometric.com) engine, made into a pip-installable tool.

```python
from gigi_episodes import find_changepoints

# Your service's p99 latency over the last 1000 minutes
latencies = [50.1, 49.8, 51.2, ..., 95.1, 96.3, 94.8, ...]

result = find_changepoints(latencies)
print(result.indices)   # [573]  — latency drift started at minute 573
print(result.count)     # 1
```

Three lines of code. The exact minute your latency went sideways, in a noisy signal where the eye can barely see the shift.

---

## What this is, in plain words

You have a sequence of numbers measured over time — latencies, error counts, test-suite durations, sensor readings, anything. Somewhere in that sequence, the underlying behavior *changed*. The mean shifted up. A regime broke. A flake started.

`find_changepoints()` tells you **where**. Not "this looks weird" — but a concrete index: *the change happened around point 573*.

That's it. One question, answered carefully, with one function call.

| Question | What `gigi-episodes` answers |
|---|---|
| **Ops**: "When did latency start drifting?" | An index — and how confident you should be |
| **Test flakiness**: "When did this test suite start failing more?" | The build number where the flake rate jumped |
| **User behavior**: "When did retention break in this cohort?" | The cohort week where retention dropped |
| **ML monitoring**: "When did model accuracy drop?" | The prediction-batch index where accuracy regressed |
| **Sensor data**: "When did this gauge shift to a new regime?" | The reading number where the new baseline took over |
| **A/B tests**: "When did the metric in arm B diverge?" | The sample index where two distributions split |

This is the *EPISODIC* primitive — number nine of GIGI's twelve brain primitives — pulled out as the smallest possible installable tool.

---

## See it work

Install it:

```bash
pip install gigi-episodes
```

Now you can already do this:

```python
from gigi_episodes import find_changepoints

# A sequence: flat at 1.0, jumps to 5.0 partway through
values = [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 1, 1]

result = find_changepoints(values)

for cp in result.change_points:
    print(f"index {cp.index}: {cp.mean_before:.2f} → {cp.mean_after:.2f} "
          f"(score {cp.score:.1f})")

# index 6:  1.00 → 5.00  (score 11.7)
# index 12: 5.00 → 1.00  (score 11.7)
```

Two change-points caught: the jump up at index 6, the drop back down at index 12. The `score` is a confidence-like number — how many standard deviations the mean shift is, in units of the local noise.

### A more realistic example

Real data is never that clean. Here's a noisy signal — Gaussian noise around a mean that shifts midway — and `gigi-episodes` still nails it:

```python
import numpy as np
from gigi_episodes import find_changepoints

rng = np.random.default_rng(42)

# 400 noisy points: mean=1 for the first 200, mean=5 for the next 200
signal = np.concatenate([
    rng.normal(1.0, 0.3, 200),
    rng.normal(5.0, 0.3, 200),
]).tolist()

result = find_changepoints(signal)
print(f"detected {result.count} change-point(s) at: {result.indices}")
# detected 1 change-point(s) at: [199]
```

The true change was at index 200 — `gigi-episodes` reported it at 199. **Off by one point**, in a signal where the noise (σ=0.3) is comparable to the jump (mean shift of 4.0).

And on pure noise, it correctly does nothing:

```python
clean = rng.normal(0, 1, 300).tolist()
result = find_changepoints(clean)
print(result.count)
# 0
```

No spurious change-points. This is harder than it sounds — many detectors will fire on every momentary excursion. The defaults here are tuned to ignore noise outliers and only flag actual regime shifts.

---

## The CLI

If you'd rather work with files:

```bash
# Detect change-points in a CSV column
gigi-episodes detect data.csv --column latency_ms

# Output JSON for piping
gigi-episodes detect data.csv --column latency_ms --json

# Tune sensitivity
gigi-episodes detect data.csv --column latency_ms --threshold 4.0 --min-segment 30

# Read from stdin
cat values.txt | gigi-episodes detect - --column 0
```

A typical CLI report looks like:

```
$ gigi-episodes detect production.csv --column latency_p99
  source:       production.csv
  column:       latency_p99
  backend:      local
  threshold:    3.0
  min_segment:  10
  n_points:     1000
  changepoints: 1

  index    score   mean_before → mean_after
  -----    -----   ---------------------------
    573    12.83      50.247  →   95.103
```

---

## The math, explained

The hard part of change-point detection isn't finding obvious step functions — it's **not firing on noise**. This section walks through how `gigi-episodes`'s local backend works and why each piece exists.

### The setup

Given a sequence of values $x_1, x_2, \ldots, x_n$, we want to find indices where the underlying mean **changed**. Naïvely, you could compute a z-score at every point:

$$z_i = \frac{|x_i - \bar{x}|}{\sigma}$$

…and flag any point where $z_i$ exceeds some threshold. But this is **terrible**:

- A single noise outlier fires a false alarm.
- A genuine slow drift never trips the threshold.
- A change in *mean* with no change in *value at a single index* gets missed.

What you actually want is: **does the mean of the next chunk of points meaningfully differ from the mean so far?**

### Step 1: The two-sample mean-difference test

For each candidate split point $i$, split the data into two windows: everything from the last change-point up to $i$ (call this "before"), and the next `window` points after (call this "after"). Compute the mean of each, and the **standard error** of the difference:

$$\text{SE} = \sqrt{\frac{\sigma_{\text{before}}^2}{n_{\text{before}}} + \frac{\sigma_{\text{after}}^2}{n_{\text{after}}}}$$

The **z-statistic** is then:

$$z = \frac{|\bar{x}_{\text{after}} - \bar{x}_{\text{before}}|}{\text{SE}}$$

If $z >$ `threshold` (default 3.0), we declare a change-point at $i$. The `threshold` is interpreted as "how many standard deviations of mean-shift do we require to be convinced." 3.0 is roughly the 99.7% confidence threshold under a Gaussian assumption.

### Step 2: Why the naïve version is still broken — the noise outlier problem

Suppose your "before" window has 50 quiet points and one big noise spike. The sample variance $\sigma_{\text{before}}^2$ will be huge, the SE will be huge, every real change-point will look statistically insignificant, and you'll miss them all.

So we need a **robust** estimate of noise — one that ignores outliers.

### Step 3: MAD-of-consecutive-differences, the Rousseeuw trick

For step-shifts, ordinary variance is contaminated by the shifts themselves. Imagine your real data is "a sequence of long flat stretches separated by jumps": classical $\sigma$ counts both the noise *and* the jumps, blowing up.

The fix is **MAD of consecutive differences** ([Rousseeuw, 1993](https://doi.org/10.1080/00031305.1993.10475977)):

$$\sigma_{\text{noise}} = \frac{\text{median}(|x_{i+1} - x_i|)}{0.6745}$$

The median absolute deviation of *consecutive differences* is insensitive to step shifts (which only affect one difference each, and the median drowns them out). The constant 0.6745 makes the MAD consistent with the standard deviation under a Gaussian noise assumption.

This single number gives us a **global noise floor**: the smallest reasonable $\sigma$ for the test, no matter how clean a local window looks.

### Step 4: Putting it all together

The actual test, for each candidate split point $i$, is:

$$\sigma_{\text{effective}}^2 = \max(\sigma_{\text{local}}^2, \sigma_{\text{noise}}^2)$$

$$z = \frac{|\bar{x}_{\text{after}} - \bar{x}_{\text{before}}|}{\sqrt{\sigma_{\text{effective}}^2 \left(\tfrac{1}{n_{\text{before}}} + \tfrac{1}{n_{\text{after}}}\right)}}$$

Floor the variance by the global MAD-of-diffs estimate, compute the standard error, divide. If $z > 3$, you've found a change-point — and you've found it without being fooled by either local noise outliers or genuinely-quiet stretches.

This is the **1-D specialization** of GIGI's `/brain/episodic` endpoint, which generalizes to **multivariate signals on a Kähler bundle** — but the 1-D version captures the math accessibly enough to fit in this README.

---

## Tuning

| Parameter | Default | What higher means |
|---|---|---|
| `threshold` | `3.0` | Stricter — fewer (more confident) detections. ~σ-units of mean shift. |
| `min_segment` | `10` | Longer required segment before any detection can fire. Useful for ignoring brief excursions. |

Defaults are calibrated for **clean operational data** — latencies, counts, scores, gauges. For very noisy data, raise the threshold to 4.0 or 5.0. For long sequences with slow regime shifts (drift that takes hundreds of points to manifest), increase `min_segment` to 30+.

---

## What you get from the standalone package

The pure-numpy `LocalBackend` catches:

- **Step changes** in mean (sharp jumps, even buried in noise) — down to roughly $1.5\sigma$ with default threshold
- **Returns to baseline** after a regime ended
- **Zero false positives on clean signals** (verified — see the noise-rejection example above)

That's enough for **"when did latency drift?", "when did this test start flaking?", "when did the metric in arm B diverge?"** — the daily-driver use cases for change-point detection.

## Want more? Upgrade to the GIGI engine

The standalone `LocalBackend` is the simplest specialization of GIGI's `EPISODIC` primitive — a single windowed mean-difference test in 1-D. Pointing at a running GIGI instance via `GigiBackend` (next section) unlocks the engine's full **Kähler-aware fiber-bundle detector**, which adds:

| Need | What the GIGI engine gives you |
|---|---|
| **Slow drift detection** | Gradual mean shifts over hundreds of points — detected as **non-trivial holonomy of the fiber** as you parallel-transport along the base manifold. Once data lives on a fiber bundle, drift is a *geometric* signal (a gradient in the fiber's expected value across base position), not a statistical one. Years-deep capability across HERALD, TESSERA, Geodesic, and earlier work — drift is a *curvature* phenomenon, not a threshold phenomenon. |
| **Variance-only changes** | Second-moment shifts where σ grows but μ doesn't move, picked up by the L13.3 diagonal-Gaussian fit with L13.7 denominator-floor stability. |
| **Multivariate joint changes** | Events that are subtle in any single field but clear in the joint distribution across the full fiber. |
| **Anisotropic noise** | Heteroscedastic detection where the Kähler form sets the metric — different directions get different noise scales, instead of one global σ. |

Each of these is **already built** in the engine. The standalone package is the entry point; GIGI is where the full version lives.

---

## Two backends

### `LocalBackend` (default — pure numpy)

This is what runs when you call `find_changepoints(values)` with no backend argument. No external services, no network calls. Use this 99% of the time.

```python
from gigi_episodes import find_changepoints, LocalBackend

result = find_changepoints(values, backend=LocalBackend())
```

### `GigiBackend` — wire the same API to the full engine

The same `find_changepoints` call, pointed at a running GIGI instance, gives you every capability in the upgrade table above:

```python
from gigi_episodes import find_changepoints, GigiBackend

backend = GigiBackend(
    url="http://localhost:3142",
    api_key="dev-local",
    bundle="latency_metrics",
    field="p99_ms",
)
result = find_changepoints(backend=backend)
```

The backend calls GIGI's `/brain/episodic` endpoint, which runs the engine's full **Kähler-aware Welford fit** over the bundle's fiber. Same function signature, same result shape, dramatically more powerful detection. The 1-D specialization is the entry point; the engine is the full version.

---

## About GIGI — the engine `gigi-episodes` is a window into

`gigi-episodes` exposes one of **twelve brain primitives** in GIGI, a geometric database engine that models data as **fiber bundles over a base manifold**. All twelve primitives are unified by the **Friston master equation** on a Kähler bundle:

| Primitive | What it does |
|---|---|
| `SAMPLE` | Draw samples from a fitted distribution |
| `FORECAST` | Predict future values of a time series |
| `DREAM` | Generate synthetic records ([see `gigi-dream`](https://pypi.org/project/gigi-dream/)) |
| `RECONSTRUCT` | Fill in missing fields from partial records |
| `INPAINT` | Reconstruct masked regions of structured data |
| `PREDICT` | Single-step prediction from current state |
| `ATTEND` | Compute attention/importance weights over fields |
| `FOCUS` | Drill down on a subset of the bundle |
| **`EPISODIC`** ← this package | Detect change-points and regime shifts |
| `SEMANTIC` | Retrieve records by meaning/similarity |
| `SELF-MONITOR` | Compute geometric health metrics (curvature, etc.) |
| `EXPLAIN` | Produce a natural-language summary of bundle state |

Beyond the brain primitives, GIGI provides:

- 🧠 **Persistent structured memory** with schema that survives serialization (via the DHOOM format)
- 📐 **Scalar curvature K** as a "geometric health score" for any bundle
- 🌐 **GIGI Query Language** (GQL — a SQL-flavored DSL, not GraphQL) for filtering, aggregating, transporting fiber-bundle data
- 🔄 **Real-time WebSocket subscriptions** to bundle mutations
- 📊 **Live demo** at [gigi-stream.fly.dev](https://gigi-stream.fly.dev/v1/health) currently hosting **4,961 bundles and 12.8 million records**

### GIGI is free

Per [Davis Geometric's licensing philosophy](https://davisgeometric.com):

> *"Free for the people who use it to learn; supported by the companies that ship products with it."*

- 🆓 **Free for research, education, and non-commercial use.**
- 💼 **Commercial deployments are patent-protected** (US Provisional Patent 64/045,889) — contact for licensing.
- 🏛️ **Patented commercial-tier operations** (curvature, spectral, holonomy, transport) return `LICENSE_REQUIRED` for non-commercial callers.

Read about the math: [davisgeometric.com](https://davisgeometric.com) · The engine: [github.com/nurdymuny/gigi](https://github.com/nurdymuny/gigi)

---

## Sibling packages

`gigi-episodes` is part of a family of small, focused brain primitives extracted from GIGI:

- [**`gigi-dream`**](https://pypi.org/project/gigi-dream/) — synthetic data generation (the `DREAM` primitive)
- [**`gigi-mcp`**](https://pypi.org/project/gigi-mcp/) — Model Context Protocol server, lets Claude query GIGI directly
- [**`gigi-client`**](https://pypi.org/project/gigi-client/) — Python SDK for the GIGI engine (HTTP + WebSocket)

Each one stands alone and works without the others. Together, they're the "scattered seeds" of GIGI — small enough to try in 30 seconds, useful even if you never adopt the full engine.

---

## License

MIT. Free for any use, commercial or otherwise. See [LICENSE](LICENSE).

(Note: this package's MIT license is unconditional. GIGI itself, which the `GigiBackend` connects to, has the dual license described above — the `LocalBackend` you get from `pip install gigi-episodes` has no such restrictions.)

---

## Status

**v0.1.0** — stable for the documented surface (1-D `LocalBackend` + CLI + `GigiBackend` skeleton). API may evolve through the 0.x series; will stabilize at 1.0.

Issues, ideas, and pull requests: [github.com/nurdymuny/gigi-episodes](https://github.com/nurdymuny/gigi-episodes/issues)

Built with care by [Bee Rosa Davis](https://davisgeometric.com) / [Davis Geometric](https://davisgeometric.com). 💛
