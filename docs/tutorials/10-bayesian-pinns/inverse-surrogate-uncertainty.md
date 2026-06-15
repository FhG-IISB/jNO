# Inverse: Bayesian posterior over an inverted input

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/10_bayesian_pinns/05_inverse_surrogate_uncertainty.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Forward-then-freeze surrogate inversion with calibrated uncertainty.**
Phase 1 trains a PINN surrogate `u_net(x) ≈ sin(πx)` against the PDE
`u'' + π² sin(πx) = 0` (point estimate, optax).  Phase 2 freezes the
surrogate and uses NUTS to sample the posterior over an unknown input
`x_query` given an observation `u_obs ≈ u_exact(x_true)` under a
Gaussian-noise likelihood with known scale `σ`.

Compared with the deterministic [`surrogate_inversion`](../05-coupled-and-inverse/surrogate-inversion.md)
tutorial, the **input** is now Bayesian — jNO's per-parameter
`.bayesian()` attaches NUTS to `x_query`, so the inverse problem
returns a *posterior* over the inverted input plus a credible interval,
not just a single point estimate.

## Non-identifiability caveat

`u(x) = sin(πx)` is symmetric about `x = 0.5`, so `u(0.3) ≈ u(0.7)`.
The true posterior given `u_obs ≈ 0.809` is **bimodal**.  A
single-chain NUTS started near `0.3` finds the left mode and stays
there; the credible interval reported here characterises that mode
only.  Multi-chain inference (with chains started from different
initial conditions) would reveal both modes — a natural extension on
top of the same `.bayesian()` setup.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/05_inverse_surrogate_uncertainty.py"
```
