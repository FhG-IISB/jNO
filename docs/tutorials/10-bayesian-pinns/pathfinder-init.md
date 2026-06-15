# Pathfinder warm-start via `.initialize()`

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/11_pathfinder_init.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Demonstrates jno's *logdensity-aware initializer* hook.**  Pathfinder
(Zhang et al. 2022) runs L-BFGS on the loss-derived log-density and
turns the inverse-Hessian trajectory into a normal approximation to the
posterior.  From that fitted `q` we get:

* a **warm starting position** — the MAP-ish `state.position` for K=1
  chains, or K i.i.d. samples for K>1 (proper over-dispersion);
* a **diagonal `inverse_mass_matrix`** estimate from the per-dimension
  variance of M draws from `q`.

Exposed through the existing `.initialize()` API — no new kwargs on
`.bayesian()`:

```python
a.initialize(jno.bayesian.pathfinder(maxiter=30, num_samples=200))
a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)
```

`.bayesian()`'s `warmup` + `adapt` still mean what they do today and
apply **after** pathfinder.  When `adapt=True`, window adaptation
re-runs from pathfinder's warm position and may further refine
`step_size`; when `adapt=False`, pathfinder's IMM is final and the
user's `step_size` is kept verbatim.

## The behaviour matrix

| `.initialize(pathfinder(...))` | `adapt` | What runs |
|---|---|---|
| not set | `True` (default) | window adaptation from the user's init |
| not set | `False` | user's init, user's step_size — no warmup |
| set | `True` | pathfinder → window: warm position + IMM, then window refines step_size |
| set | `False` | pathfinder only: warm position + pathfinder IMM, user's step_size |

## Numbers from the tutorial

Three side-by-side runs on the harmonic regression problem from T02
(truth `A = 50.0, B = -30.0`, default zero init).  Each run uses
`num_chains=2`.

| Run | `.initialize(...)` | `warmup, adapt` | A | B | R-hat A | R-hat B | Wall |
|---|---|---|---|---|---|---|---|
| baseline | none | `300, True` | 49.55 | -29.77 | 1.038 | 1.000 | 6.5 s |
| pf-only | `pathfinder(30, 200)` | `0, False` | 49.61 | -29.60 | 1.007 | 1.006 | 12.7 s |
| chained | `pathfinder(30, 200)` | `100, True` | 49.63 | -29.70 | 1.004 | 0.998 | 8.3 s |

The pathfinder-touched runs produce **lower R-hat** (chains agree more
tightly) on R-hat A specifically — the harder of the two coefficients
for the baseline.  Pathfinder's L-BFGS lands both chains in the same
basin before sampling, so they don't need to mix into agreement
themselves.  Wall-clock is dominated by JIT compile on this tiny
problem; on production-scale B-PINN runs the L-BFGS cost amortises
quickly.

## Extending — writing your own initializer

The `.initialize()` hook is generic.  Any class with
`requires_logdensity = True` plus a `__call__` matching the
[`_BayesianInitializer`](../../training/bayesian.md#logdensity-aware-initializers-initialize-extension)
contract will be detected and dispatched the same way:

```python
import jno

class _MyInit(jno.bayesian._BayesianInitializer):
    def __call__(self, rng_key, logdensity_fn, position, num_chains):
        # ... your algorithm ...
        return new_position, {"inverse_mass_matrix": ...}  # IMM optional

a.initialize(_MyInit())
a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)
```

Future jno initializers — Laplace approximation (Magnani et al. 2024),
SVGD (Liu & Wang 2016), MAP via Adam — slot in as additional subclasses
on the same hook.

## Composition with existing features

* **Masks** — `.mask(M).bayesian()` + pathfinder works: pathfinder
  runs against the masked subset's log-density; the unmasked complement
  stays at init.
* **Multi-chain** — when `num_chains > 1`, pathfinder samples K
  distinct starting positions from the fitted `q`.  Strictly better
  dispersion than the `init_jitter` heuristic, which is silently
  overridden when pathfinder is set.
* **Non-IMM kernels** (MALA / SGLD / SGHMC) — pathfinder's warm
  position is applied; the IMM update is silently dropped by the
  signature gate.  Sampler runs as before.
* **`substeps=`** — not compatible (initializer runs against the full
  loss, kernel sees only substep-local constraints) — raises a clear
  error at solve start.
* **`.vi(...)`** — not compatible (VI initialises its own variational
  distribution from `state.mu = position`) — raises a clear error at
  solve start.

## Reference

Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022).
*Pathfinder: Parallel quasi-Newton variational inference.*  Journal of
Machine Learning Research, 23(306), 1-49.
[arXiv:2108.03782](https://arxiv.org/abs/2108.03782)

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/11_pathfinder_init.py"
```
