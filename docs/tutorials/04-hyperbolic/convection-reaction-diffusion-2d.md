# Convection-Reaction-Diffusion 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/convection_reaction_diffusion_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to chapter</a>
</div>

This example mixes transport, diffusion, and reaction in a transient 2D PDE.

## Problem Setup

The script solves a convection-reaction-diffusion system with drift terms `(b_x, b_y)`, diffusion strength `nu`, and reaction strength `lambda`.

## Step 1: Build a 2D Space-Time Domain

The field depends on two spatial coordinates and time, so the sampled domain is larger and the residual has more moving parts.

## Step 2: Combine Multiple Physical Effects

The residual includes:

- time evolution
- first-order transport terms
- second-order diffusion
- a linear reaction term

## Step 3: Train Against a Manufactured Solution

A manufactured forcing term keeps the problem verifiable while still exposing the full transient structure.

## What To Notice

- This is one of the richest transient tutorial examples in the set.
- The residual remains readable even with several physical terms.
- It is a good template for realistic advection-diffusion-reaction systems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/04_hyperbolic/convection_reaction_diffusion_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/04-hyperbolic/">Back to 04 Hyperbolic</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/04_hyperbolic/convection_reaction_diffusion_2d.py"
```
