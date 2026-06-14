# SVGD warm-start via `.initialize()`

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/13_svgd_init.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Stein Variational Gradient Descent (SVGD) as a logdensity-aware
initializer.**  Third concrete entry on the `.initialize()` hook
landed in Phase 12:

```python
a.initialize(jno.bayesian.svgd(num_iters=300, num_particles=32))
a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)
```

## Algorithm

1. **Seed particles.**  ``num_particles`` particles are placed around
   the user-supplied position by adding Gaussian noise of std
   ``init_jitter``.  Default ``num_particles = max(num_chains, 32)``
   so we always have enough particles for stable variance estimation
   even when the caller only asked for 1 chain.  ``init_jitter``
   defaults to ``None`` → ``max(0.1 * std(position), 1e-3)`` — a
   scale-aware spread one-tenth of the parameter scale.  Pass an
   explicit positive float (e.g. ``init_jitter=0.5``) to override.
2. **Run SVGD.**  Each particle is updated by a kernelised functional
   gradient (Liu & Wang 2016, eq. 8):

   $$
   \phi^*(x) = \frac{1}{N}\sum_{j} \left[
     k(x_j, x)\, \nabla_{x_j} \log p(x_j) + \nabla_{x_j} k(x_j, x)
   \right]
   $$

   The **first term** pulls each particle toward higher posterior
   density.  The **second term** — the gradient of the RBF kernel —
   pushes particles apart so they spread out.  jno's wrapper calls
   `blackjax.svgd` inside a `jax.lax.scan` over ``num_iters``
   iterations.
3. **Use the particle cloud as the warm-start.**

   * ``num_chains=1`` — particle-cloud **mean** as the warm position.
   * ``num_chains>1`` — first ``num_chains`` particles as K distinct
     warm positions.  The repulsive kernel dynamics already provide
     proper over-dispersion; no additional jitter is needed.

   Per-dim particle variance (plus a small ridge) is returned as the
   diagonal ``inverse_mass_matrix``.

## Trade-offs vs Pathfinder / Laplace

| Aspect | Pathfinder | Laplace | SVGD |
|---|---|---|---|
| Posterior approximation | unimodal Gaussian from L-BFGS factors | unimodal Gaussian at MAP | particle cloud (can be multi-modal) |
| Multi-modal | underestimates | local only | **captures with enough particles** |
| Compute cost per step | one L-BFGS line search | one gradient eval | ``O(N²)`` pairwise kernel evals |
| Memory | low | ``O(D²)`` full, ``O(D)`` diagonal | ``O(N × D)`` |
| When to pick | unimodal, fast, no tuning | exact Hessian wanted | suspected multi-modality, willing to pay ``N²`` cost |

## Numbers from the tutorial

T02-scale problem (truth `A = 3.14, B = -2.71`); two side-by-side
runs with `num_chains=2`:

| Run | `.initialize(...)` | A | B | R-hat A | R-hat B | Wall |
|---|---|---|---|---|---|---|
| baseline | none | 3.148 | -2.673 | 1.016 | 0.999 | 6.5 s |
| svgd | `svgd(num_iters=300, num_particles=32)` | 3.211 | -2.505 | 1.003 | 1.004 | 9.3 s |

Both recover truth.  SVGD's R-hat is the tightest of the three
initializers (Pathfinder ≈ Laplace ≈ SVGD on a unimodal problem;
SVGD's advantage shows on multi-modal posteriors which this short
tutorial doesn't exhibit).

## Composition with existing features

Identical to [pathfinder's composition matrix](./pathfinder-init.md#composition-with-existing-features).
Masks, multi-chain, non-IMM kernels, substeps / VI guards all work
the same — they're handled by the shared `_BayesianInitializer`
dispatch helpers, not by anything SVGD-specific.

## When to use SVGD

* You suspect a multi-modal posterior — SVGD's repulsive kernel can
  reach distinct modes that Pathfinder / Laplace cannot.
* You can afford ``num_particles²`` pairwise kernel evaluations
  per iteration.  For ``num_particles=32`` and ``num_iters=300`` that's
  ≈ 300k kernel evals — small for scalar PDE coefficients, modest
  for BNNs.
* You want a fully **deterministic** variational method (Pathfinder's
  ELBO sampling is stochastic; Laplace requires gradient-descent
  convergence; SVGD's only randomness is the initial particle
  seeding).

For unimodal posteriors with no multi-modality concerns, Pathfinder
is usually the cheaper default.

## Reference

Liu, Q., & Wang, D. (2016).  *Stein Variational Gradient Descent:
A General Purpose Bayesian Inference Algorithm.*  §3 (the SVGD update
rule and the kernelised Stein discrepancy it minimises).  Advances
in Neural Information Processing Systems (NeurIPS), 29, 2378-2386.
[arXiv:1608.04471](https://arxiv.org/abs/1608.04471)

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/13_svgd_init.py"
```
