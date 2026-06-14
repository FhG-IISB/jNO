# Laplace warm-start via `.initialize()`

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/12_laplace_init.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Laplace approximation as a logdensity-aware initializer.**  Slots
into the same `.initialize()` hook pathfinder uses, with a different
algorithm:

```python
a.initialize(jno.bayesian.laplace(
    map_steps=300,
    map_optimizer=optax.adam(1e-1),
    hessian_strategy="diagonal",
))
a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)
```

## Algorithm

1. **MAP via optax.**  Optimise `-log p(θ | data)` with the supplied
   optimiser (Adam by default).  Runs as a JIT-compiled
   `jax.lax.scan` over `map_steps` iterations.
2. **Hessian at the MAP** — `H = -∇²log p`.  Two strategies:

    * **`hessian_strategy="full"`** — full `(D, D)` Hessian via
      `jax.hessian`.  Numerically clean; memory cost grows as `D²`.
      Right for `D < ~1000`.
    * **`hessian_strategy="diagonal"`** (default) — diagonal of `H`
      computed by `D` Hessian-vector probes.  Memory cost `O(D)` —
      required for BNN-scale problems.  Compute cost similar to full
      but no `D×D` matrix is ever materialised.

3. **Posterior approximation** `N(MAP, H⁻¹)`.  For `num_chains=1` the
   warm position is the MAP; for `num_chains>1` we sample K
   over-dispersed warm positions from this Gaussian.  The diagonal of
   `H⁻¹` is returned as the kernel's `inverse_mass_matrix`.

A small `ridge` (default `1e-6`) is added to `H` before any inversion /
Cholesky to guard against ill-conditioned Hessians at non-converged
MAP estimates.

## Trade-offs vs Pathfinder

| Aspect | Pathfinder | Laplace |
|---|---|---|
| MAP search | L-BFGS (quasi-Newton) | gradient descent (Adam by default) |
| Hessian | low-rank inverse-Hessian from L-BFGS path | exact `∇²log p` at MAP (diagonal or full) |
| Robustness on multi-modal posteriors | better — explores the L-BFGS path | local approximation only |
| Cost for large `D` | dominated by L-BFGS line searches | full Hessian quadratic; diagonal linear |
| Failure mode | falls back to a normal approximation that may underestimate posterior variance | needs `ridge` if `H` is ill-conditioned |

Both produce a Gaussian approximation suitable as a warm start; they
just get there by different routes.

## Numbers from the tutorial

T02-scale problem (truth `A = 3.14, B = -2.71`); two side-by-side
runs with `num_chains=2`:

| Run | `.initialize(...)` | A | B | R-hat A | R-hat B | Wall |
|---|---|---|---|---|---|---|
| baseline | none | 3.148 | -2.673 | 1.016 | 0.999 | 6.6 s |
| laplace | `laplace(map_steps=300, optax.adam(1e-1))` | 3.236 | -2.553 | 1.013 | 1.005 | 9.8 s |

Both recover truth.  Laplace's R-hat is marginally better on the
harder coefficient.  Wall-clock is slower because the Hessian path
incurs its own JIT compile; for production problems where the
L-BFGS or window-adaptation runs would otherwise dominate, Laplace
amortises.

## Composition with existing features

Identical to [pathfinder's composition matrix](./pathfinder-init.md#composition-with-existing-features)
— masks, multi-chain, non-IMM kernels, substeps / VI guards all
work the same.  The mechanism is the shared `_BayesianInitializer`
hook; nothing pathfinder-specific is involved.

## When to use Laplace

* The posterior is unimodal and well-approximated by a Gaussian.
* You want an *exact* mass-matrix estimate at the MAP rather than
  the L-BFGS approximation pathfinder produces.
* You can afford the MAP-search optimiser steps (Adam can be slow
  on steep posteriors; tune `map_optimizer` and `map_steps`
  accordingly).

For multi-modal posteriors, large `D`, or when you want a single
robust warm-start with no tuning, pathfinder is usually the better
default.

## References

* MacKay, D. J. C. (1992).  *A Practical Bayesian Framework for
  Backpropagation Networks.*  §6 (Laplace approximation around the
  posterior mode).  Neural Computation, 4(3), 448-472.
  [doi:10.1162/neco.1992.4.3.448](https://doi.org/10.1162/neco.1992.4.3.448)
* Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M.,
  & Hennig, P. (2021).  *Laplace Redux — Effortless Bayesian Deep
  Learning.*  §2 (Laplace approximations for neural networks).
  NeurIPS 2021.  [arXiv:2106.14806](https://arxiv.org/abs/2106.14806)
* Magnani, E., Krämer, N., Pförtner, M., & Hennig, P. (2024).
  *Linearization Turns Neural Operators into Function-Valued Gaussian
  Processes.*  §3 (linearised-Laplace for neural operators).
  [arXiv:2406.05072](https://arxiv.org/abs/2406.05072)

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/12_laplace_init.py"
```
