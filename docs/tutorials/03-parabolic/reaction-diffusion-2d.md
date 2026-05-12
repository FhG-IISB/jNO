# Reaction-Diffusion 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/reaction_diffusion_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to chapter</a>
</div>

This example augments diffusion with a linear reaction term in a transient 2D setting.

## Problem Setup

The PDE has the form `u_t - nu Delta u + lambda u = f`, with a manufactured exact solution used for validation.

## Step 1: Build the Space-Time Problem

The script samples interior points in a 2D domain over time and tracks an exact reference solution.

## Step 2: Use a Hard Boundary Ansatz

The model is wrapped with a boundary envelope so the solution remains zero on the outer edges.

## Step 3: Add Both Initial and PDE Residuals

The time-dependent PDE residual and the initial-condition loss are optimized together.

## Step 4: Use a Standard Training Schedule

This script is a good reference for a clean, standard jNO transient training setup with a manufactured source term.

## What To Notice

- The reaction term changes the balance of the dynamics without changing the basic workflow.
- Manufactured solutions are especially useful for validating transient codes.
- This is a useful bridge from heat equations to nonlinear parabolic systems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/reaction_diffusion_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/reaction_diffusion_2d.py"
```
