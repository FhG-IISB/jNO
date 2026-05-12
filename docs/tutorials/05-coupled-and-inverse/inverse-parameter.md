# Inverse Parameter

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/inverse_parameter.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to chapter</a>
</div>

This example is an inverse problem rather than a field solve: it learns unknown scalar coefficients from residual constraints.

## Problem Setup

The script introduces trainable scalar parameters and fits them so synthetic constraints are satisfied.

## Step 1: Treat Parameters as Learnable Objects

Instead of only training a neural field, the script creates scalar parameter models that participate in optimization.

## Step 2: Build Residuals From Data Relationships

The optimization target is a set of algebraic or residual constraints rather than a spatial PDE field.

## Step 3: Solve and Inspect Learned Coefficients

After optimization, the identified parameters are printed from the trained model set.

## What To Notice

- jNO can optimize more than neural fields.
- Inverse problems often reuse the same core workflow with different residual definitions.
- This example is a good template for coefficient discovery and calibration.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/05_coupled_and_inverse/inverse_parameter.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/05-coupled-and-inverse/">Back to 05 Coupled and Inverse</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/05_coupled_and_inverse/inverse_parameter.py"
```
