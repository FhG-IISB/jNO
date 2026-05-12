# Advection-Diffusion 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/advection_diffusion_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example combines transport and diffusion in a transient 1D problem.

## Problem Setup

The PDE has the form `u_t + c u_x = nu u_xx + f`, with a manufactured forcing term used for validation.

## Step 1: Build a Space-Time Domain

The script samples the interior and initial slice so both PDE dynamics and startup data can be enforced.

## Step 2: Choose a Time-Dependent Model

A DeepONet-style architecture maps space-time inputs to the field value.

## Step 3: Encode Transport and Diffusion Together

The residual contains both a first derivative in space and a second derivative in space, so it mixes advective and diffusive behavior.

## What To Notice

- This is a good first transport example before moving to nonlinear convection.
- Advection and diffusion terms can differ strongly in scale.
- The overall workflow still matches the other PINN examples.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/advection_diffusion_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/advection_diffusion_1d.py"
```
