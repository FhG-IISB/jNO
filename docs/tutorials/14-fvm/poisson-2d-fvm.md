# Poisson 2D (FVM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/14_fvm/poisson_2d_fvm.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

$-\Delta u = f$ on the unit square with $u = 0$ on the boundary: the constraint list of the [FDM primer](../09-fdm/poisson-2d-fdm.md), solved by finite volumes, cell- and vertex-centred.

## Result

Both centerings converge at second order. Cell-centred rel-$L^2$ goes from $4.9\times10^{-3}$ to $1.0\times10^{-3}$ from $h = 0.1$ to $0.05$ on triangles, and vertex-centred from $1.2\times10^{-2}$ to $3.1\times10^{-3}$.

## The code

```python
--8<-- "tutorial_examples/14_fvm/poisson_2d_fvm.py:code"
```

## What to notice

- `ui.xx + ui.yy` is read as a divergence: the face sum of the face gradient, compact along the centroid line and corrected for the skewness of the triangles.
- The cell-centred solve returns one value per cell. `prob.centroids` says where, and `prob.to_nodes(sol)` averages them to the mesh nodes.
- `centering="vertex"` builds median-dual volumes around the nodes. Its Dirichlet condition is a strong row at the boundary nodes, as in `jno.fdm`.

See [Finite Volume Method](../../fvm.md) for how each term is read.
