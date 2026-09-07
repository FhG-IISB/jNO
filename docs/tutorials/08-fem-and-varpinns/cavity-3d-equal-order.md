# Equal-order flow in 3-D: where the stabilised pair pays for itself

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/cavity_3d_equal_order.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

In 2-D, Taylor–Hood costs about **2.9×** the DOFs of stabilised P1/P1 — real, but not decisive. In
3-D a P2 tetrahedron carries 10 nodes to P1's 4, and the measured ratio is **5.1×**. That is where
equal-order stops being a nicety.

The formulation is the [2-D one](navier-stokes-stabilised-2d.md) with a longer coordinate list:
`value_shape=(3,)`, `ax = [xi, yi, zi]`, and `dom.cell_metric` returning a 3×3 tensor instead of 2×2.
Nothing about SUPG/PSPG is dimension-specific.

## The resource wall is the point

Measured on one 8 GB card with the direct solve this script uses. Both failures are **cuBLAS
allocation failures — out of memory, not divergence** (checked, because "the stable pair failed to
converge" would have been a much more exciting and completely wrong claim):

| | fits | runs out |
|---|---|---|
| Taylor–Hood P2/P1 | N = 6 — 6,934 DOFs | N = 8 — 15,468 DOFs |
| **stabilised P1/P1** | N = 10 — 5,324 DOFs | N = 12 — 8,788 DOFs |

The DOF ceiling is about the same for both — it is the factorisation, not the discretisation. But the
equal-order pair spends those DOFs on roughly **5× more mesh**. On this hardware that is the
difference between a 6³ and a 10³ cube.

## Head to head at N = 6

The finest cube where both fit. Structured tets, so the vertical centreline x = y = 0.5 carries nodes
of *both* velocity spaces and the comparison needs no interpolation of the thing being compared.

| | DOFs | solve |
|---|---|---|
| stabilised P1/P1 | 1,372 | 3.4 s |
| Taylor–Hood P2/P1 | 6,934 | 8.2 s |

Centreline `u_x`: **max deviation 0.036, i.e. 3.3 %** of the profile range, at a fifth of the DOFs.

!!! warning "N = 6 is coarse, and 3.3 % reflects that"
    Seven P1 nodes on the centreline is a thin comparison. Most of that 3.3 % is discretisation error
    in the coarse P1 velocity, not a defect of the stabilisation — but this run cannot separate the
    two, and no mesh-convergence study was done. Read it as "the two agree to a few percent on a
    coarse mesh", not as an accuracy claim.

## Scope

- **Re = 100**, reached by continuation. Nothing higher was tried in 3-D.
- Steady only.
- The memory wall is specific to `lu(backend="host")`. An iterative solve with a block/Schur
  preconditioner is the way past it and is not exercised here.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/cavity_3d_equal_order.py:code"
```
