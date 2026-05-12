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

## Step 2: Hard-Enforce the Dirichlet Part

The model is multiplied by `x(1-x)`, which automatically satisfies the zero-value condition on the left and right sides.

## Step 3: Add Neumann Residuals

The top and bottom boundary losses are built by differentiating the boundary-evaluated field with respect to `y`.

## Step 4: Solve With Multiple Loss Terms

The core includes the PDE residual, top Neumann loss, bottom Neumann loss, and an error tracker.

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
