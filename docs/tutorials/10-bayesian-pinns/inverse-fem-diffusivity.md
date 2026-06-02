# Inverse: jNO-FEM forward, Bayesian diffusivity

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/10_bayesian_pinns/06_inverse_fem_diffusivity.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/10-bayesian-pinns/">Back to chapter</a>
</div>

**Bayesian inverse problem with a high-fidelity FEM forward.**  Recover
the scalar diffusivity `α` in the 2-D Poisson equation
`-α Δu = f` on `Ω = [0, 1]²` from noisy nodal observations of `u`.

The earlier tutorials in this section (`03`, `04`) used closed-form
forward models (`sin(πx)/π²`, `exp(-kt)`).  Real engineering inverse
problems rarely have closed forms — they have numerical PDE solvers.
This tutorial shows the pattern when the forward is the **FEAX-backed
FEM solver** that jNO already exposes.

## Architecture

* jNO's `domain.init_fem` + `weak.assemble` build the JAX-traceable
  stiffness matrix `A` and load vector `b`.  Both flow through
  `jnp.linalg.solve` in a way that is fully differentiable in any
  scalar parameter appearing in the weak form.
* For pure-diffusion linear PDEs (this tutorial), `A(α) = α · A_base`,
  so `u(α) = u_baseline / α`.  We exploit this to avoid re-assembling
  every NUTS step.  When the PDE has α-dependent boundary terms or
  nonlinear couplings, replace the scaling with a per-call `assemble`
  + `linalg.solve` — the rest of the pattern is identical, just
  slower.
* The likelihood `logdensity(α) = -‖u(α) - u_obs‖² / (2σ²) + log_prior`
  is a plain JAX function of `α`; we pass it to
  `blackjax.window_adaptation` and `blackjax.nuts` directly.  jNO's
  `.bayesian()` API is currently scoped to problems whose forward is
  expressible as a jNO Placeholder expression, so we drop one level
  here.

This is the right pattern whenever your forward model lives outside
jNO's tracer — FEM, finite volume, an external solver, an ODE
integrator: use jNO for the differentiable forward, blackjax for the
chain.

## Result

`α = 0.99 ± 0.02`, 90 % CI `[0.96, 1.02]`, truth `α = 1.0` —
comfortably interior, CI width set by the observation noise level
(`σ = 0.005`) and the chain length (`keep = 1000`).

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/06_inverse_fem_diffusivity.py"
```
