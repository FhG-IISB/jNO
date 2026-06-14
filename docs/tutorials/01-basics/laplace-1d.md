# Laplace 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/01_basics/laplace_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/01-basics/">Back to chapter</a>
</div>

The smallest complete jNO example. Solves the 1-D Laplace equation with **non-homogeneous** Dirichlet boundary conditions and a hard-enforced linear-interpolant ansatz.

## Problem Setup

```text
u''(x) = 0,   x in [0, 1],   u(0) = 0,  u(1) = 1
```

Exact solution: `u(x) = x` (the straight line between the two boundary values).

## Step 1: Create the Domain

```python
domain = jno.domain.line(mesh_size=0.1)
x, _ = domain.variable("interior")
```

## Step 2: Build the Network with a Non-Homogeneous Hard Ansatz

The boundary values are non-zero (`u(0)=0`, `u(1)=1`), so the multiplicative `x(1-x)` trick alone won't enforce them. Instead, use a **linear interpolant + correction** ansatz:

```python
u_net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=32, num_layers=3, key=jax.random.PRNGKey(0))
).optimizer(optax.adam(optax.exponential_decay(1e-3, 10, 0.5, end_value=1e-5)))

u = x + x * (1 - x) * u_net(x)
```

Why this works:

- `x` alone satisfies `u(0)=0` and `u(1)=1` exactly — it's the linear interpolant of the BCs.
- `x*(1-x)` vanishes at both endpoints, so `u_net(x)` can never break the BCs no matter what it learns.
- The network only has to learn the **deviation** from the linear interpolant. For pure Laplace `u''=0` that deviation is exactly zero, so this is a clean test of whether the optimiser can drive `u_net` to the constant-zero function.

## Step 3: Build the PDE Residual

```python
pde = u.d2(x)  # Laplace: u'' = 0
```

`.d2(x)` is the second-derivative shortcut on any `Placeholder`. Equivalent to `u.d(x).d(x)` but more compact.

## Step 4: Solve

```python
crux = jno.core([pde.mse])
history = crux.solve(5000)
```

## What To Notice

- The non-homogeneous BCs require an ansatz that **adds** to a satisfying solution rather than just multiplying. The general recipe is `u = boundary_lift(x) + zero_at_boundary(x) * net(x)`.
- For more interesting Poisson-style problems with a non-zero forcing, see [Poisson 1D](poisson-1d.md) — same domain, but adds a soft-BC pattern with `scheme="finite_difference"`.
- This is a *trivial* PDE in the sense that the exact solution is in the trial-space class; jNO is being asked to confirm convergence, not discover anything new.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/01_basics/laplace_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO/tutorials/01-basics/">Back to 01 Basics</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/01_basics/laplace_1d.py"
```
