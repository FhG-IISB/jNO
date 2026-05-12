# Diffusion-Reaction Robin BC

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example compares a variational PINN against a classical FEM reference under mixed Dirichlet and Robin boundary conditions.

## Problem Setup

The PDE is a diffusion-reaction or Helmholtz-like problem on a 2D domain with boundary integrals contributing Robin terms.

## Step 1: Build a Weak Form With Boundary Terms

The script includes boundary quadrature contributions directly in the variational residual.

## Step 2: Train the VPINN

Instead of minimizing a pointwise PDE residual, the network minimizes a weak-form objective.

## Step 3: Compare Against FEM

A finer FEM solve is used as a reference to assess how well the variational PINN captures the solution.

## What To Notice

- Robin conditions fit naturally into a weak-form workflow.
- This script is a strong reference for FEM versus VPINN comparisons.
- Boundary quadrature tags are one of the key implementation ideas.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/diffusion_reaction_robinBC.py"
```
