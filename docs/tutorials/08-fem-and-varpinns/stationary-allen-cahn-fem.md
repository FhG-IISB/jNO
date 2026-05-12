# Stationary Allen-Cahn FEM

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example solves a nonlinear stationary Allen-Cahn problem using classical FEM machinery.

## Problem Setup

The weak form corresponds to a stationary Allen-Cahn equation with a nonlinear cubic term.

## Step 1: Define the Nonlinear Weak Residual

The script expresses the nonlinear form directly in terms of FEM operators rather than pointwise PINN losses.

## Step 2: Build the Jacobian and Nonlinear Solve Loop

Because the problem is nonlinear, a residual alone is not enough; the script also uses Jacobian information for iterative solution.

## Step 3: Solve With a Classical Nonlinear Method

A SciPy-style root or nonlinear solve routine is used to converge the weak-form system.

## What To Notice

- Weak-form nonlinear solves are structurally different from PINN optimization.
- This example is useful for comparing classical and neural treatments of the same PDE family.
- It also shows how jNO's weak-form abstractions extend beyond linear problems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/stationary_allen_cahn_fem.py"
```
