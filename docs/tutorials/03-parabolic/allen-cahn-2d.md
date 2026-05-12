# Allen-Cahn 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/allen_cahn_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to chapter</a>
</div>

This example solves a manufactured 2D Allen-Cahn problem and introduces a nonlinear cubic reaction term.

## Problem Setup

The PDE has the Allen-Cahn structure `u_t = epsilon^2 Delta u + u - u^3 + f`, with a known exact solution used to build the forcing term.

## Step 1: Build a Manufactured Nonlinear Problem

The exact solution is substituted into the PDE to derive a forcing term that makes validation straightforward.

## Step 2: Set Up the Space-Time Network

The model learns a field over space and time while respecting the chosen boundary handling.

## Step 3: Encode the Nonlinear Residual

The key change relative to the heat equation is the nonlinear reaction term `u - u^3`.

## Step 4: Impose the Initial Condition

The script uses the same PDE infrastructure but anchors the solution at the initial time with an additional loss.

## What To Notice

- Nonlinear reaction terms are easy to express once the field is available symbolically.
- Manufactured solutions are especially valuable for nonlinear PDEs.
- This example is a good template for phase-field style problems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/03_parabolic/allen_cahn_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/03-parabolic/">Back to 03 Parabolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/03_parabolic/allen_cahn_2d.py"
```
