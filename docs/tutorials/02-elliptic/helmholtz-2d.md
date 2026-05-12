# Helmholtz 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/helmholtz_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example adds an oscillatory Helmholtz term to the elliptic residual, which makes the solution behavior more wave-like than Poisson-like.

## Problem Setup

```text
Delta u + k^2 u = -f(x,y),   (x,y) in [0,1]^2,
u = 0 on the boundary
```

with exact solution `u(x,y) = sin(pi x) sin(pi y)`.

## Step 1: Choose a Wave Number

The parameter `k` controls the oscillatory regime. The script encourages trying multiple values to see how convergence changes.

## Step 2: Keep Boundary Conditions Soft

This script uses an unconstrained field and adds a separate boundary loss on sampled boundary points.

## Step 3: Assemble the Helmholtz Residual

The PDE combines the Laplacian with the zeroth-order term `k^2 u`, which is the defining feature of Helmholtz problems.

## Step 4: Track Relative Error

After solving, the script computes a relative L2 error and plots exact, predicted, and absolute-error fields.

## What To Notice

- Helmholtz problems can become harder near resonant regimes.
- Soft boundary terms are convenient when the raw architecture should remain unconstrained.
- This example is a good first step toward frequency-domain PDEs.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/helmholtz_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/helmholtz_2d.py"
```
