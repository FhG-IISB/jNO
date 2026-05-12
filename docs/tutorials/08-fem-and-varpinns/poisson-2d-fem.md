# Poisson 2D FEM

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This example is a pure finite-element solve with no neural network component.

## Problem Setup

The script assembles the weak form of a manufactured Poisson problem and solves the resulting linear system.

## Step 1: Define the Weak Form

Instead of writing a pointwise PDE residual, the script builds bilinear and linear forms using FEM symbols.

## Step 2: Assemble the Linear System

The weak form is transformed into a matrix system through jNO's FEM assembly workflow.

## Step 3: Solve and Compare

The script computes the FEM solution and compares it to a known exact field.

## What To Notice

- This chapter is about weak forms rather than PINN residuals.
- The assembly pipeline is useful as a reference even if your final method is neural.
- Pure FEM examples help validate the underlying PDE setup.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/poisson_2d_fem.py"
```
