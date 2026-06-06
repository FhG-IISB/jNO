# Poisson 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/poisson_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to chapter</a>
</div>

This example solves nearly the same equation as Laplace 1D, but changes two important implementation choices: it uses soft boundary constraints and a finite-difference second derivative.

## Problem Setup

We solve

```text
-u''(x) = sin(pi x),   x in [0, 1],   u(0) = u(1) = 0
```

with exact solution

```text
u(x) = sin(pi x) / pi^2
```

## Step 1: Create Interior and Boundary Variables

Unlike the hard-constraint example, this script explicitly asks for both interior and boundary points.

```python
domain = jno.domain.line(mesh_size=pick(0.01, 0.1))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")
```

Why both are needed:

- `x` is used for the PDE residual.
- `xb` is used to define the boundary-condition loss.

## Step 2: Define the Reference Solution

```python
u_exact = jnn.sin(π * x) / π**2
```

As in the previous example, this is only used to track model quality.

## Step 3: Build the Network

This version keeps the network output unconstrained.

```python
u_net = jnn.nn.mlp(
    in_features=1,
    hidden_dims=64,
    num_layers=4,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, pick(10_000, 10), 1e-5))

u = u_net(x)
```

The consequence is that boundary conditions must now be enforced through an explicit loss term.

## Step 4: Define PDE and Boundary Losses

```python
pde = -u.d2(x, scheme="finite_difference") - jnn.sin(π * x)
bc = u_net(xb)
error = jnn.tracker((u - u_exact).mse, interval=pick(100, 1))
```

Key difference from Laplace 1D:

- `u.d2(..., scheme="finite_difference")` computes the second derivative numerically.
- `bc = u_net(xb)` is minimized toward zero, which softly enforces the boundary values.

## Step 5: Solve With Multiple Constraints

```python
crux = jno.core([pde.mse, bc.mse, error], domain)
history = crux.solve(pick(10_000, 10))
```

Now the optimization balances:

- PDE residual in the interior
- boundary mismatch at the endpoints
- tracked error against the exact solution

## Step 6: Evaluate Error and Plot

After solving, the script computes the mean absolute error and saves the solution plot.

```python
mae = np.abs(pred - true).mean()
print(f"Mean absolute error vs exact: {mae:.6e}")
```

This gives a simple scalar quality check in addition to the saved figures.

## What To Notice

- Soft constraints are more flexible, especially when hard constraints are awkward to encode.
- Finite differences provide an alternative to fully automatic differentiation.
- This is a good template when you need explicit control over boundary and residual terms.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/poisson_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to 01 Basics</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/01_basics/poisson_1d.py"
```
