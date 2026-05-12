# Laplace 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/laplace_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to chapter</a>
</div>

This is the smallest complete jNO example. It solves a 1D Poisson or Laplace-type equation with homogeneous Dirichlet boundary conditions and compares the learned solution against the analytical one.

## Problem Setup

We solve

```text
-u''(x) = sin(pi x),   x in [0, 1],   u(0) = u(1) = 0
```

with exact solution

```text
u(x) = sin(pi x) / pi^2
```

## Step 1: Create the Domain

The script initializes jNO, creates a 1D line domain, and extracts interior points.

```python
π = jnn.pi
dire = jno.setup(__file__)

domain = jno.domain(constructor=jno.domain.line(mesh_size=pick(0.01, 0.1)))
x, _ = domain.variable("interior")
```

What this does:

- `jno.setup(__file__)` creates the run directory for outputs.
- `jno.domain.line(...)` builds the 1D geometry.
- `domain.variable("interior")` gives the collocation points used for the PDE residual.

## Step 2: Define the Analytical Reference

The exact solution is only used for monitoring error, not for training supervision.

```python
u_exact = jnn.sin(π * x) / π**2
```

This is useful because you can track whether the PINN is converging toward the known solution.

## Step 3: Build the Network with Hard Boundary Conditions

The model is a small MLP. Boundary conditions are enforced by multiplying the network output by `x(1-x)`.

```python
u_net = jnn.nn.mlp(
    in_features=1,
    hidden_dims=32,
    num_layers=3,
    key=jax.random.PRNGKey(0),
).optimizer(optax.adam(1), lr=lrs.exponential(1e-3, 0.5, pick(5_000, 10), 1e-5))

u = u_net(x) * x * (1 - x)
```

Why this matters:

- `u_net(x)` is unconstrained.
- Multiplying by `x(1-x)` forces `u(0)=u(1)=0` exactly.
- This removes the need for a separate boundary loss term.

## Step 4: Build the PDE Residual and Error Tracker

The residual uses automatic differentiation twice to compute the second derivative.

```python
pde = -jnn.grad(jnn.grad(u, x), x) - jnn.sin(π * x)
error = jnn.tracker((u - u_exact).mse, interval=pick(100, 1))
```

This gives you:

- `pde.mse`: the physics loss to minimize
- `error`: a tracked metric that reports the solution error during training

## Step 5: Solve the Problem

```python
crux = jno.core([pde.mse, error], domain)
history = crux.solve(pick(5_000, 10))
```

This is the standard jNO flow:

1. Bundle constraints and tracked metrics into `jno.core(...)`
2. Call `solve(...)`
3. Use the returned history for diagnostics

## Step 6: Evaluate and Plot

After training, the script sorts points, evaluates the learned field, and saves both the training history and solution plot.

```python
pts = np.array(crux.domain_data.context["interior"][0, 0, :, 0])
idx = np.argsort(pts)
xs = pts[idx]
pred = np.array(crux.eval(u)).reshape(xs.shape[0], 1)[:, 0][idx]
true = np.array(crux.eval(u_exact)).reshape(xs.shape[0], 1)[:, 0][idx]
```

You end up with:

- `training_history.png`
- `solution.png`

## What To Notice

- This example uses hard constraints, which keeps the loss simple.
- The exact solution is not part of the PDE loss, only the tracker.
- For many introductory PDEs, this is the cleanest pattern to start from.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/01_basics/laplace_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/01-basics/">Back to 01 Basics</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/01_basics/laplace_1d.py"
```
