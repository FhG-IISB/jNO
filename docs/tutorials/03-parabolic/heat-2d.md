# Heat 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/heat_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to chapter</a>
</div>

This example extends the heat equation to a square domain and shows how to inspect the learned solution at multiple time slices.

## Problem Setup

The PDE is `u_t = alpha Delta u` on the unit square with homogeneous Dirichlet boundaries and a sinusoidal initial state.

## Step 1: Build the 2D Space-Time Geometry

The script samples interior space-time points on a rectangular domain and uses a separate initial-time slice for the starting condition.

## Step 2: Use a DeepONet With a Hard Spatial Envelope

The model output is multiplied by `x(1-x)y(1-y)` so the boundary is satisfied on all four edges.

## Step 3: Combine PDE and Initial Losses

The transient residual enforces the heat equation, while a dedicated initial-condition residual anchors the solution at `t = 0`.

## Step 4: Plot Time Snapshots

One of the nice features of this script is explicit evaluation on selected time slices so you can inspect how the field evolves.

## What To Notice

- This is the natural 2D extension of Heat 1D.
- Snapshot evaluation is a good debugging tool for time-dependent solves.
- The same ideas generalize to more complex transient PDEs.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/heat_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/heat_2d.py"
```
