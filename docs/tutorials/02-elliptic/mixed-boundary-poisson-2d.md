# Mixed-Boundary Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/mixed_boundary_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example shows how to combine different boundary conditions on different parts of the same domain.

## Problem Setup

```text
-Delta u = f(x,y),   (x,y) in [0,1]^2
u = 0 on x = 0 and x = 1
du/dy = 0 on y = 0 and y = 1
```

with exact solution `u(x,y) = sin(pi x) cos(pi y)`.

## Step 1: Sample Boundary Segments Separately

The script requests `top` and `bottom` variables from the domain in addition to interior points. This lets it apply separate Neumann terms to those boundaries.

```python
domain = jno.domain.rect(mesh_size=0.05)
x,  y,  _ = domain.variable("interior")
xt, yt, _ = domain.variable("top")
xb, yb, _ = domain.variable("bottom")

u_exact = jno.np.sin(pi * x) * jno.np.cos(pi * y)
forcing  = 2 * pi**2 * u_exact
```

## Step 2: Hard-Enforce the Dirichlet Part

The model is multiplied by `x(1-x)`, which automatically satisfies the zero-value condition on the left and right sides.

```python
net = jno.nn.wrap(foundax.mlp(in_features=2, hidden_dims=80, num_layers=5, key=jax.random.PRNGKey(14)))
net.optimizer(optax.adam(optax.exponential_decay(1e-3, 80, 0.5, end_value=1e-5)))

u        = net(x,  y)  * x  * (1 - x)
u_top    = net(xt, yt) * xt * (1 - xt)
u_bottom = net(xb, yb) * xb * (1 - xb)
```

## Step 3: Add Neumann Residuals

The top and bottom boundary losses are built by differentiating the boundary-evaluated field with respect to `y`.

```python
pde           = -jno.np.laplacian(u, [x, y]) - forcing
neumann_top   = u_top.d(yt)
neumann_bottom = u_bottom.d(yb)
```

## Step 4: Solve With Multiple Loss Terms

The core includes the PDE residual and both Neumann losses.

```python
crux    = jno.core([pde.mse, neumann_top.mse, neumann_bottom.mse], domain)
history = crux.solve(40_000)
```

## What To Notice

- Mixed boundary problems are common in practice.
- Tagged boundary variables make it straightforward to isolate different edges.
- You do not need to choose between all-hard or all-soft boundary handling.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/mixed_boundary_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/mixed_boundary_poisson_2d.py"
```
