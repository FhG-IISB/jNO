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
  stiffness matrix `A` and load vector `b`.  We solve the `α = 1`
  problem once to get `u_baseline`.
* For pure-diffusion linear PDEs (this tutorial), `A(α) = α · A_base`
  and therefore `u(α) = u_baseline / α`.  We express the forward as a
  jNO expression of the trainable `α` and a per-node `u_baseline`
  constant attached via `jno.domain.from_array`, so the **whole loss
  flows through `crux.solve()`** with NUTS attached via `.bayesian()`
  — same configurator pattern as every other tutorial in this section.
* For nonlinear PDEs the scaling identity fails; you'd then wrap the
  per-step `assemble + linalg.solve` in a jNO `FunctionCall`
  placeholder.  Same architecture, just slower.

## Result

`α = 1.07 ± 0.14`, 90 % CI `[0.88, 1.33]`, truth `α = 1.0` —
comfortably interior, CI width set by the observation noise level
(`σ = 0.005`), the chain length (`keep = 1000`), and the per-node
likelihood averaging.

## Script

```python
--8<-- "tutorial_examples/10_bayesian_pinns/06_inverse_fem_diffusivity.py"
```
