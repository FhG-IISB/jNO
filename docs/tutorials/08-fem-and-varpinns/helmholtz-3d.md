# Helmholtz 3D

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

This is the most geometrically complex example in the tutorial set: a 3D Helmholtz problem on an extruded letter-F style geometry.

## Problem Setup

The script works with mixed boundary conditions on a 3D domain and compares VPINN and FEM-style weak-form ideas on a nontrivial mesh.

## Step 1: Build a Custom 3D Geometry

A custom geometry constructor creates the volume and tagged surfaces required for boundary conditions.

## Step 2: Assemble Weak-Form Quantities in 3D

The example uses tetrahedral-style weak-form machinery rather than a pointwise PDE residual.

## Step 3: Visualize the 3D Result

The script includes a surface or boundary visualization pipeline so the final field can be interpreted geometrically.

## What To Notice

- This is a high-end tutorial example rather than a first learning example.
- Complex geometry handling is one of the main reasons weak-form approaches become valuable.
- The workflow shows how jNO scales beyond unit-interval and unit-square toy problems.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/08-fem-and-varpinns/">Back to 08 FEM and Variational PINNs</a>
</div>

## Script Snippet

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/helmholtz_3D.py"
```
