# Helmholtz Mixed BC FEM VarPINN

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem_varpinn.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example trains a variational PINN for a 2D Helmholtz problem and compares it against a fine FEM reference solution.

## Problem Setup

The PDE has the Helmholtz form with mixed Dirichlet and Neumann data and a manufactured exact solution.

## Step 1: Write a Weak Residual

The residual is assembled in variational form rather than as a pointwise strong-form loss.

## Step 2: Combine Hard Boundary Handling With VPINN Training

The neural ansatz handles some constraints structurally while the weak form captures the PDE and remaining conditions.

## Step 3: Compare Against FEM

The script solves a classical FEM reference problem on a finer mesh and visualizes the difference.

## What To Notice

- This is a representative VPINN workflow in the tutorial suite.
- Weak forms can improve how derivative-heavy PDEs are handled.
- Direct FEM comparison makes the example especially useful for benchmarking.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem_varpinn.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/helmholtz_mixedBC_fem_varpinn.py"
```
