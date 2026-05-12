# Variable-Coefficient Poisson 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example keeps the same square geometry but replaces constant diffusion with a spatially varying conductivity field.

## Problem Setup

```text
-div(kappa(x,y) grad u(x,y)) = f(x,y),   (x,y) in [0,1]^2,
u = 0 on the boundary
```

with `kappa(x,y) = 1 + x + y` and exact solution `sin(pi x) sin(pi y)`.

## Step 1: Define the Coefficient Field

The script constructs `kappa` directly from the sampled coordinates, so the PDE coefficients vary pointwise across the domain.

## Step 2: Build the Flux Instead of Just the Laplacian

Rather than writing `Delta u`, the script forms flux components and then computes their divergence.

## Step 3: Enforce Boundary Conditions Hard

As in several earlier examples, `x(1-x)y(1-y)` wraps the raw network output so the boundary condition is built into the ansatz.

## Step 4: Train Against a Spatially Heterogeneous PDE

Once coefficients vary in space, the forcing term becomes more complex, but the overall jNO workflow remains residual, tracker, core, solve.

## What To Notice

- Variable coefficients are a common bridge from toy PDEs to physically meaningful media.
- Forming fluxes explicitly keeps the PDE readable.
- The example shows how spatial heterogeneity enters both forcing and residual construction.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/variable_coefficient_poisson_2d.py"
```
