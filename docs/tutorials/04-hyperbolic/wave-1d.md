# Wave 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/wave_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example solves a second-order-in-time wave equation and introduces both displacement and velocity initial data.

## Problem Setup

The PDE has the form `u_tt = c^2 u_xx`, together with initial displacement and initial velocity conditions.

## Step 1: Build a Space-Time Domain

The script samples interior and initial-time points, just as in the parabolic examples.

## Step 2: Add Two Initial Constraints

Unlike diffusion equations, the wave equation needs both `u(x,0)` and `u_t(x,0)`.

## Step 3: Build the Hyperbolic Residual

The residual uses a second derivative in time and a second derivative in space.

## What To Notice

- Hyperbolic equations often require more than one initial condition.
- This example is the cleanest starting point for second-order-in-time PDEs.
- It is a good reference before adding damping or source terms.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/wave_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/wave_1d.py"
```
