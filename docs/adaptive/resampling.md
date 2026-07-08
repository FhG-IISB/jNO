# Adaptive Resampling

Adaptive resampling strategies dynamically replace collocation points during training to focus on regions of high PDE residuals or high predicted error. This can accelerate convergence and improve solution accuracy by directing computational effort to the most challenging parts of the domain.

All strategies are available via the `jno.sampler` factory class.

---

## Build your own strategy

Any subclass of `ResamplingStrategy` is a valid strategy — override one abstract method:

```python
def resample(self, points, residuals, domain, tag, epoch, rng_key, candidates=None) -> jnp.ndarray
```

The base class handles `resample_every`, `resample_fraction`, and `start_epoch` for you; your `resample` only sees calls that actually need to produce new points. `candidates` is a pre-drawn pool from `domain.draw_candidates(tag)` — use it as the source of replacement points so you stay inside the domain's physical groups.

```python
from jno.utils.adaptive.resampling import ResamplingStrategy
import jax, jax.numpy as jnp

class TopResidual(ResamplingStrategy):
    """Replace the worst-residual points with fresh draws from the candidate pool."""

    def resample(self, points, residuals, domain, tag, epoch, rng_key, candidates=None):
        if residuals.ndim > 1:                         # collapse batch dim if present
            residuals = jnp.mean(residuals, axis=0)
        n_replace = int(len(points) * self.resample_fraction)
        n_keep    = len(points) - n_replace
        order     = jnp.argsort(residuals)             # ascending — keep lowest
        kept      = points[order[:n_keep]]
        idx       = jax.random.choice(rng_key, len(candidates), shape=(n_replace,))
        return jnp.concatenate([kept, candidates[idx]], axis=0)

# Wire it in exactly like the built-ins
x, y = domain.variable(
    "interior",
    sample=(None, None),
    resampling_strategy=TopResidual(resample_every=100, resample_fraction=0.2, start_epoch=1000),
)
```

---

## Built-in strategies

For the common cases, `jno.sampler` ships these out of the box.

### `sampler.random` — Random Resampling (Baseline)

Replaces a random subset of points from a new uniform sample. Helps prevent overfitting to a fixed grid without using any residual information.

```python
from jno import sampler

s = sampler.random(
    resample_every=100,     # resample every N epochs
    resample_fraction=0.1,  # fraction of points to replace
    start_epoch=1000,       # delay resampling until this epoch
)
```

---

### `sampler.rad` — Residual-Adaptive Distribution (RAD)

Selects new points by clustering around the top-`k` points with the highest PDE residuals.

```python
s = sampler.rad(
    resample_every=100,
    resample_fraction=0.1,
    start_epoch=1000,
    k=10,    # number of top-residual anchor points
)
```

**Reference:** Lu et al., "Residual-based adaptivity for two-phase flow simulation in porous media using Physics-informed Neural Networks"

---

### `sampler.rard` — Residual-Adaptive Refinement with Distribution (RARD)

Uses importance sampling based on `residual^power` to draw new points. Higher `power` concentrates points more aggressively near high-error regions.

```python
s = sampler.rard(
    resample_every=100,
    resample_fraction=0.1,
    start_epoch=1000,
    power=2.0,   # exponent for residual-based importance weighting
)
```

---

### `sampler.ha` — Hybrid Adaptive (HA)

Alternates between random-refresh phases (for exploration/regularisation) and adaptive phases (for exploitation of high-error regions). Reduces the risk of over-concentrating in a single region.

```python
s = sampler.ha(
    resample_every=100,
    resample_fraction=0.5,
    start_epoch=1000,
    alternate=True,       # True = alternate random/adaptive; False = always adaptive
    random_first=True,    # start with random phase if alternating
)
```

---

### `sampler.cr3` — Causal Retain-Resample (CR3)

Designed for **time-dependent PDEs** where causality matters (e.g., wave equations, diffusion). Uses a learnable causal gate that progressively exposes later time steps as earlier ones are solved. Prevents the network from learning a "shortcut" by ignoring causality.

```python
s = sampler.cr3(
    resample_every=100,
    resample_fraction=0.5,
    start_epoch=1000,
    t_index=-1,          # index of the time column in the collocation array
    alpha=5.0,           # gate steepness
    gamma0=-0.5,         # initial gate position (negative → only early times exposed)
    eta_g=1e-3,          # gate learning rate
    epsilon=20.0,        # gate update damping
    delta_max=0.1,       # maximum gate step per resampling
    min_keep_frac=0.1,   # never drop below this fraction of points
    max_keep_frac=0.9,   # never keep more than this fraction
)
```

**Reference:** Adapted from "Respecting Causality for Training Physics-Informed Neural Networks"

---

### `sampler.pinnfluence` — PINNFluence

Uses **gradient-based influence scores** (simplified influence functions) to identify collocation points with the highest potential impact on reducing the total loss. Computationally more expensive per step; use with a larger `resample_every`.

```python
s = sampler.pinnfluence(
    resample_every=500,      # use larger intervals (influence is expensive)
    resample_fraction=0.2,
    start_epoch=2000,
    alpha=1.0,               # score exponent for sampling distribution
    c=1.0,                   # additive smoothing constant
    candidate_factor=3.0,    # candidate pool multiplier (pool size = fraction × factor)
)
```

---

## Usage

Attach the strategy when creating variables from a tagged point set using
`resampling_strategy=...`:

```python
from jno import sampler

domain = jno.domain(constructor=jno.domain.rect(mesh_size=0.05))
x, y = domain.variable(
    "interior",
    sample=(None, None),
    resampling_strategy=sampler.rad(
        resample_every=100,
        resample_fraction=0.1,
        start_epoch=1000,
    ),
)
```

The solver applies configured strategies automatically during `solve()`.

---

## Strategy Comparison

| Strategy | Computational Cost | Sensitivity | Best For |
|----------|--------------------|-------------|----------|
| `random` | Very low | None (baseline) | Preventing grid overfitting |
| `rad` | Low | High-residual clustering | Localized high-error regions |
| `rard` | Low | Importance sampling | Smooth residual fields |
| `ha` | Low | Adaptive with regularization | General use, avoids over-concentration |
| `cr3` | Medium | Time-causal | Time-dependent PDEs |
| `pinnfluence` | High | Gradient-based influence | Small-data or high-accuracy regimes |

---

## Tips

- **Start epoch**: Always delay resampling (`start_epoch > 0`) to let the network form a rough global solution first. Resampling too early can disrupt initial training.
- **Fraction**: Values between 0.1 and 0.3 are typical. Too high a fraction discards too many well-distributed points; too low a fraction has minimal effect.
- **Resample interval**: Shorter intervals (e.g., every 50 epochs) react faster but add overhead. Longer intervals (e.g., every 500 epochs) are cheaper.
- **Combining with mini-batching**: Adaptive resampling works with `batchsize` in `solve()` — the sampler acts on the full domain, while mini-batching only affects each gradient step.
