# Diffusion-Reaction 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/diffusion_reaction_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example solves a steady 1D PDE that combines diffusion and a linear reaction term.

## Problem Setup

```text
-u''(x) + sigma u(x) = f(x),   x in [0,1],   u(0) = u(1) = 0
```

with manufactured solution `u(x) = sin(pi x)`.

## Step 1: Set the Reaction Strength

The parameter `sigma` determines how strongly the reaction term competes with diffusion. Try larger values to see the effect on convergence.

```python
σ = 10.0  # reaction coefficient — increase to make the problem stiffer
```

## Step 2: Build a Line Domain and Exact Solution

The script samples the domain, builds a manufactured forcing term, and keeps the exact solution for error tracking.

```python
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.1))
x, _ = domain.variable("interior")

u_exact = jno.np.sin(π * x)
forcing  = (π**2 + σ) * jno.np.sin(π * x)  # f = -u_exact'' + σ u_exact
```

## Step 3: Hard-Enforce Boundary Conditions

The field is defined with an `x(1-x)` factor so endpoint values are zero by construction.

```python
u_net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=64, num_layers=4, key=jax.random.PRNGKey(0))
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, 10, 1e-5))

u = u_net(x) * x * (1 - x)
```

## Step 4: Balance Diffusion and Reaction in the Residual

The residual is assembled as `-u_xx + sigma u - forcing`, making this a clean example of multiple physical effects in one PDE.

```python
u_xx = jno.np.grad(jno.np.grad(u, x), x)
pde  = -u_xx + σ * u - forcing
```

## Step 5: Measure Error and Plot

After solving, the script prints mean absolute error and saves solution and error plots.

```python
crux    = jno.core([pde.mse], domain)
history = crux.solve(5000)

_u, _u_exact = crux.eval([u, u_exact])
rel_l2 = float(jax.numpy.linalg.norm(_u - _u_exact) / (jax.numpy.linalg.norm(_u_exact) + 1e-8))
```

## What To Notice

- Increasing `sigma` makes the problem stiffer.
- The overall training loop remains similar to the basic 1D examples.
- This is a useful template before adding time dependence.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/diffusion_reaction_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/diffusion_reaction_1d.py"
```
