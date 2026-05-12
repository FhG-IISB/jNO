# Diffusion-Reaction 1D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/diffusion_reaction_1d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to chapter</a>
</div>

This example solves a steady 1D PDE that combines diffusion and a linear reaction term.

## Problem Setup

```text
-u''(x) + sigma u(x) = f(x),   x in [0,1],   u(0) = u(1) = 0
```

with manufactured solution `u(x) = sin(pi x)`.

## Step 1: Set the Reaction Strength

The parameter `sigma` determines how strongly the reaction term competes with diffusion.

## Step 2: Build a Line Domain and Exact Solution

The script samples the domain, builds a manufactured forcing term, and keeps the exact solution for error tracking.

## Step 3: Hard-Enforce Boundary Conditions

The field is defined with an `x(1-x)` factor so endpoint values are zero by construction.

## Step 4: Balance Diffusion and Reaction in the Residual

The residual is assembled as `-u_xx + sigma u - forcing`, making this a clean example of multiple physical effects in one PDE.

## Step 5: Measure Error and Plot

After solving, the script prints mean absolute error and saves solution and error plots.

## What To Notice

- Increasing `sigma` makes the problem stiffer.
- The overall training loop remains similar to the basic 1D examples.
- This is a useful template before adding time dependence.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/02_elliptic/diffusion_reaction_1d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/02-elliptic/">Back to 02 Elliptic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/02_elliptic/diffusion_reaction_1d.py"
```
