# Anisotropic mesh refinement (stretched elements for a thin layer)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/anisotropic_layer.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

Isotropic refinement makes triangles *smaller* where the error is large. For a thin
**directional** feature that is wasteful: resolving a layer of width $\varepsilon$ isotropically
needs $\sim\varepsilon$-sized triangles all along it. **Anisotropic** adaptation instead uses
*stretched* triangles — thin across the layer, long along it — resolving the same feature with
far fewer degrees of freedom.

The metric comes from the solution's recovered **Hessian**: its eigenvectors orient the
triangles with the solution's curvature and its eigenvalues set the size along each direction.
`AdaptSpec(anisotropic=True)` switches the adaptive loop from isotropic ZZ + Dörfler marking to
this metric (built by `hessian_metric` and normalized to a target vertex count each round).

Model — an **oblique** internal layer, the case isotropic handles *worst* (it must refine a wide
diagonal band):

$$-\nabla^2 u = f\quad\text{on the unit square},\qquad u = \tanh\!\big((x+y-1)/\varepsilon\big).$$

## In one flag

```python
from jno.utils.solver.fem_adapt import AdaptSpec

fem.solve(adapt=AdaptSpec(anisotropic=True, max_iters=8, refine_factor=1.6, max_dofs=2500))
# refine_factor grows the target vertex count each round; hmin/hmax bound the edge sizes.
# Under the hood: recover the Hessian -> build an anisotropic metric -> metric-remesh (Mmg).
```

## The result

![Left: an isotropic mesh with a wide band of tiny equal-sided triangles along the diagonal
(5997 dofs). Middle: an anisotropic mesh with thin stretched triangles hugging the diagonal layer
and coarse elements elsewhere (709 dofs). Right: a log-log plot of the ZZ error estimate versus
DOFs, the anisotropic curve far below the isotropic one.](/jNO/assets/anisotropic_layer.png)

The anisotropic mesh reaches a **lower error estimate (0.10 vs 0.38) with ~8× fewer DOFs**
(709 vs 5997) — its stretched elements align with the layer instead of tiling a band with tiny
triangles. The metric equidistributes to a compact, efficient mesh.

## What to notice

- **Where it pays off:** thin *directional* features — internal/boundary layers, fronts, shocks —
  especially **oblique or curved** ones. For a straight axis-aligned feature the gain over
  isotropic ZZ is smaller (isotropic already refines a thin band efficiently).
- **The metric is the instruction:** unlike marking (flag cells → halve them), a metric field sets
  a target *size and direction* everywhere; Mmg remeshes to equidistribute it, coarsening smooth
  regions as it refines the layer.
- **Works in 2D and 3D** (triangles and tetrahedra) — in 3D the metric is a 6-component tensor
  and mmg produces stretched *tets*. Scalar P1 (one DOF per vertex); metric-based DOF control is
  approximate, so `max_dofs` is a soft cap here.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/anisotropic_layer.py"
```
