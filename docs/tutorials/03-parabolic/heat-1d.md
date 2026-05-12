# Heat 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/heat_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to chapter</a>
</div>

This example solves the transient 1D heat equation and introduces time as an explicit input to the model.

## Problem Setup

The script solves a diffusion equation of the form `u_t = alpha u_xx` on a space-time domain with zero Dirichlet boundaries and a sinusoidal initial condition.

## Step 1: Build a Space-Time Domain

The domain includes both space and time, with separate sampling for interior and initial-condition points.

## Step 2: Use a DeepONet-Style Model

The example uses a DeepONet architecture in PINN mode so the model can learn a time-dependent field over the full domain.

## Step 3: Hard-Enforce Spatial Boundary Conditions

A spatial envelope `x(1-x)` keeps the field zero at the two endpoints for every time.

## Step 4: Add the Initial Condition as a Separate Constraint

The PDE residual governs the interior, while a second loss enforces the known initial profile at `t = 0`.

## What To Notice

- Time-dependent PDEs need both interior physics and initial data.
- The jNO workflow stays similar even though the field now depends on multiple coordinates.
- This is the cleanest parabolic starting point in the tutorial set.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/heat_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/heat_1d.py"
```
