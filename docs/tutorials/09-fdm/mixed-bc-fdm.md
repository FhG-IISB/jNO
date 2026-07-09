# Mixed Boundary Conditions (FDM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/09_fdm/mixed_bc_fdm.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/09-fdm/">Back to chapter</a>
</div>

Dirichlet, Neumann, and Robin conditions on the same problem — steady conduction on the unit square
with the manufactured field $u^\* = y^2$ (so $-\Delta u = -2$):

| edge          | condition                                          |
| ------------- | -------------------------------------------------- |
| bottom ($y=0$) | Dirichlet $u = 0$                                 |
| top ($y=1$)    | Robin $\partial_n u + (u - 3) = 0$                |
| left / right   | Neumann $\partial_n u = 0$ (insulated)           |

## Result

![Left: the jno.fdm solution, a linear-in-y-squared field rising from 0 at the bottom to 1 at the top. Middle: the absolute error against u* = y-squared, concentrated near the boundaries. Right: a log-log mesh-refinement sweep of the relative L2 error, trending downward but with scatter.](/jNO/assets/mixed_bc_fdm.png)

The recovered field matches $u^\* = y^2$ to rel-$L^2 \approx 5.7\times10^{-3}$ on the working mesh.
Because the interior field is a quadratic (so the interior FD residual is essentially exact), the
error lives almost entirely at the boundaries: the one-sided Neumann/Robin edge stencils and the
corner PDE-fallback set the accuracy. Under refinement (right) the error trends down from
$\sim\!4\times10^{-2}$ to $\sim\!10^{-3}$, but **not** at a clean $O(h^2)$ rate — it is mesh-sensitive
and scatters, as the boundary stencils are re-sampled by each unstructured mesh.

## Flux conditions are written with the edge's own tags

Get the outward normal from `domain.variable(region, normals=True)`, bind the field to the edge, and
take its normal derivative. Any condition **affine in** $\partial_n u$ works the same way — `jno.fdm`
reads the coefficient of the normal derivative directly, so Neumann and Robin are the same mechanism:

```python
nr = d.variable("right", normals=True)
ur = u.bind(x=xr, y=yr)          # field bound to the right edge

ur.d(nr) - 0.0                   # Neumann: du/dn = 0
ut.d(nt) + 1.0 * (ut - 3.0)      # Robin:   du/dn + (u - 3) = 0
```

## What to notice

- **Any mix** of Dirichlet / Neumann / Robin on different edges composes in one list — no special BC
  objects.
- A flux condition binds the field to *that edge* (`ur = u.bind(x=xr, y=yr)`) and uses it for both the
  flux term and any value term, so the whole edge equation reads from one set of boundary tags.
- Corner nodes shared by two flux edges have no single outward normal, so they fall back to the
  interior PDE residual (give a corner a Dirichlet value if it needs anchoring).
- The recovered field matches $u^\* = y^2$ to the FD-discretization accuracy
  (rel-$L^2 \approx 5\times10^{-3}$).

## Full script

```python
--8<-- "tutorial_examples/09_fdm/mixed_bc_fdm.py:code"
```
