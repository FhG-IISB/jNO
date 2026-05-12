# Coupled Elliptic 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_elliptic_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This example solves a stationary system with two interacting unknown fields.

## Problem Setup

The script solves a coupled system of the form `-Delta u + v = f` and `-Delta v + u = g` on the unit square.

## Step 1: Build Two Unknown Fields

Instead of one neural network, the script defines one model for `u` and one model for `v`.

## Step 2: Assemble Coupled Residuals

Each PDE residual depends on both unknowns, so the optimization must update the two fields jointly.

## Step 3: Track Each Field Against Reference Data

The script uses manufactured solutions so both coupled fields can be validated during training.

## What To Notice

- jNO can optimize multiple interacting models in one core.
- This is the simplest introduction to multi-physics style coupling.
- The pattern extends naturally to larger coupled systems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_elliptic_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/coupled_elliptic_2d.py"
```
