# Stokes flow past a cylinder (complex domain, Taylor-Hood)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/stokes_flow_around_cylinder.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The canonical viscous-flow benchmark on a domain with a hole. A creeping (Stokes) flow is driven
through a channel obstructed by a cylinder, discretised with the inf-sup-stable **Taylor-Hood** pair
— P2 velocity, P1 pressure — coupled in one block:

$$-\mu\,\Delta \mathbf{u} + \nabla p = 0,\qquad \nabla\!\cdot\mathbf{u} = 0.$$

## Complex domain, refined at the obstacle

The channel minus a cylinder is one shapely line; a named `ring` around the cylinder is meshed
finer than the open channel, where the velocity gradients are steep:

```python
cyl  = Point(1.5, 0.5).buffer(0.22)
ring = Point(1.5, 0.5).buffer(0.46).difference(cyl).intersection(box(0, 0, L, H))
dom  = jno.domain({"bulk": box(0, 0, L, H).difference(cyl).difference(ring), "ring": ring})
dom  = dom.build_mesh(0.12, sizes={"ring": 0.05})   # coarse channel, fine collar
```

## Mixed boundary data, no extra API

A parabolic profile drives the inlet **and** the outlet — exact for Stokes flow, which is fore-aft
symmetric at $Re=0$ — while the walls and the cylinder are no-slip. One `jno.np.where` on the
boundary coordinate picks the driven faces; everything else is zero:

```python
u_in = jno.np.where(xb < 1e-6, parab(yb), jno.np.where(xb > L - 1e-6, parab(yb), 0.0))
fem = jno.fem([
    mu * inner(gu, gv, n_contract=2) - pp * trace(gv),   # momentum
    -qq * trace(gu),                                      # incompressibility
    u(xb, yb)[0] - u_in,                                  # parabola at inlet/outlet, 0 on walls+cylinder
    u(xb, yb)[1] - 0.0,
    p(xpn, ypn) - 0.0,                                    # pin the pressure null space
])
sol = fem.solve(solve_fn=lambda A, b: lineax.linear_solve(lineax.MatrixLinearOperator(A), b).value)
```

## The result

![Stokes streamlines enter from the left, split symmetrically around the central cylinder —
accelerating over its top and bottom flanks — and re-converge downstream into a uniform channel
flow; the pattern is mirror-symmetric about the
centre-line.](/jNO/assets/stokes_flow_around_cylinder.png)

Streamlines split around the cylinder, accelerate at its flanks (colour = speed), and re-converge
downstream — the textbook creeping-flow picture.

## What to notice

- **Complex geometry + local refinement** in two lines of shapely; the `ring` is refined
  independently of the `bulk`.
- **Taylor-Hood multiphysics**: a P2 vector velocity and a P1 pressure as two `fem_symbols` fields,
  assembled as one inf-sup-stable block. `fem.problem.offset` slices the velocity back out.
- **Bring your own solver** (`lineax`) through `solve_fn`.
- **Verified by a physical invariant, not an analytic solution.** A centred cylinder makes Stokes
  flow top-bottom symmetric, so $u_x(x,y)=u_x(x,H-y)$ and $u_y(x,y)=-u_y(x,H-y)$; the measured
  symmetry error is $\sim2\times10^{-3}$.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/stokes_flow_around_cylinder.py"
```
