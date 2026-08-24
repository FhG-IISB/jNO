# Stokes 3D — Duct Past a Sphere

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/stokes_3d_duct_sphere.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Creeping flow through a rectangular duct past a sphere, on inf-sup-stable Taylor–Hood
tetrahedra — P2 velocity over P1 pressure:

$$-\nu\,\Delta \mathbf{u} + \nabla p = 0, \qquad \nabla\cdot\mathbf{u} = 0.$$

![Speed on the mid-plane of a duct past a sphere; the flow divides around the obstacle and the fast
core of the inflow profile is blocked.](/jNO/assets/stokes_3d_duct_sphere.png)

## The geometry is CSG

The obstacle needs no bespoke mesh — subtract it:

```python
duct = jno.Shape.box(0, 0, 0, L, H, H) - jno.Shape.sphere(CX, CY, CZ, R)
d = duct.size(0.06).domain()
```

## The outflow is written by *not* writing it

The downstream face is left untagged. An untagged boundary is **do-nothing** in jNO, which is
exactly the traction-free condition Stokes flow wants at an outlet:

```python
d.tag("inlet",  lambda x, y, z: x < EPS)
d.tag("walls",  lambda x, y, z: (y < EPS) | (y > H - EPS) | (z < EPS) | (z > H - EPS))
d.tag("sphere", lambda x, y, z: (x - CX)**2 + (y - CY)**2 + (z - CZ)**2 < (R + 1e-3)**2)
```

That also settles the pressure **level**, which is why this problem carries no `p.pin()`. An
*enclosed* flow — every wall Dirichlet — has a constant pressure null space and does need a gauge;
use `p.pin(mean=True)` there, not a point pin, or the pressure will not converge in 3-D.

## Drag as a reaction, not a surface integral

`fem.eval(term, u)` assembles the momentum residual with **no essential elimination applied**, so
the sum over the sphere's constrained DOFs is the force holding it in place. The force the fluid
exerts is its negative:

```python
R_free     = np.asarray(fem.eval(momentum, sol))
reaction_x = float(R_free[fem.region_dofs("sphere", field=u, component=0)].sum())
drag       = -reaction_x
```

This is the accurate way to get a force out of a finite element solution — more so than integrating
the stress over the surface, because the reaction is exactly conjugate to the constraint.

## What to notice

- **It is a saddle system.** `fem.solve()` warns if you point the matrix-free default at it; a
  direct factorization is used here, and past this size the block/Schur preconditioners in
  `jno.precond` are what scale.
- **The peak speed does not rise.** A *centred* sphere blocks the fast core of a parabolic profile
  and pushes fluid into the slower corners, so the maximum speed in the sphere's plane (0.94) is
  below the inflow peak (1.00). Blockage does not always mean acceleration.
- **The drag is a confined-sphere drag.** At 3.36× the unbounded Stokes-law value $6\pi\mu R U$,
  it reflects a duct only twice the sphere diameter across. That ratio is a sanity band, not a
  benchmark match — the wall correction at 50 % blockage is large.
