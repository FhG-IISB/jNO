# Burgers Viscous 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/burgers_viscous_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example solves the viscous Burgers equation, one of the standard nonlinear PDE benchmarks.

## Problem Setup

The PDE has the form `u_t + u u_x = nu u_xx + f`, with a manufactured exact solution used to derive the forcing term.

## Step 1: Build the Space-Time Model

The script uses the same transient PINN pattern as advection-diffusion but with a stronger nonlinear term.

## Step 2: Add Nonlinear Convection

The product `u u_x` is what makes Burgers different from linear transport models.

## Step 3: Train and Compare With the Exact Solution

The script includes diagnostics that report error and visualize the learned field.

## What To Notice

- Burgers is often the first nonlinear transport PDE people try.
- The nonlinearity appears naturally once the field and its gradient are available.
- This example is a good benchmark for model capacity and training stability.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/burgers_viscous_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/burgers_viscous_1d.py"
```
