# Coupled Parabolic 2D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_parabolic_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This example takes the coupled-field idea into a transient setting.

## Problem Setup

The script solves two time-dependent PDEs with cross-coupling terms and manufactured transient reference solutions.

## Step 1: Build Two Time-Dependent Fields

Both unknowns depend on space and time, so the sampled domain and constraint set are larger than in the stationary case.

## Step 2: Add Initial Conditions for Both Fields

Each unknown needs its own initial condition in addition to the coupled PDE residuals.

## Step 3: Train the System Jointly

All losses are optimized together so the two models remain consistent with each other and with the data.

## What To Notice

- Coupling and time dependence can be combined cleanly in one workflow.
- The same hard-boundary ideas can be reused for both unknowns.
- This is a good reference for multi-field transient PDEs.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/coupled_parabolic_2d.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/coupled_parabolic_2d.py"
```
