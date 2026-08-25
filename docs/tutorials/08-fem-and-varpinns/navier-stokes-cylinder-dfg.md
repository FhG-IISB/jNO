# Navier–Stokes — Cylinder (DFG benchmark 2D-1)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/navier_stokes_cylinder_dfg.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

Steady Navier–Stokes past a cylinder at $Re = 20$, checked against **published reference values**
rather than against a manufactured solution or against jNO itself:

$$(\mathbf{u}\cdot\nabla)\mathbf{u} - \nu\,\Delta\mathbf{u} + \nabla p = 0,\qquad \nabla\cdot\mathbf{u}=0.$$

![Speed with streamlines around a cylinder in a channel at Re = 20; the flow accelerates through the
gaps and a short steady recirculation sits behind the cylinder.](/jNO/assets/navier_stokes_cylinder_dfg.png)

## Result

Configuration and reference values from Schäfer & Turek (1996), the DFG/Featflow benchmark.
Taylor–Hood P2/P1, 10,038 DOF, one direct solve of about five seconds:

| quantity | jNO | reference | error |
|---|---|---|---|
| $c_D$ | 5.57377 | 5.57953523384 | **0.10 %** |
| $\Delta p$ | 0.11666 | 0.11752016697 | 0.73 % |
| $c_L$ | 0.01041 | 0.010618948146 | 1.93 % |

Refining further drives the drag error to **0.02 %** (0.39 → 0.10 → 0.04 → 0.02 % over four meshes).

## The outflow is written by *not* writing it

Only the inlet, the walls and the cylinder carry terms. The downstream face is left untagged, and an
untagged boundary is **do-nothing** in jNO — exactly the traction-free outflow this problem wants.
That also fixes the pressure **level**, which is why no `p.pin()` appears anywhere here. (An
*enclosed* flow does have a constant pressure null space and needs a gauge — `p.pin(mean=True)`.)

## Forces are reactions, not surface integrals

`fem.eval(term, u)` assembles the momentum residual with **no essential elimination applied**, so
summing it over the cylinder's constrained DOFs gives the force holding the cylinder in place; the
force the fluid exerts is its negative:

```python
free  = np.asarray(fem.eval(momentum, sol))
scale = 2.0 / (UMEAN**2 * DIA)
cD    = -scale * float(free[fem.region_dofs("cyl", field=u, component=0)].sum())
cL    = -scale * float(free[fem.region_dofs("cyl", field=u, component=1)].sum())
```

This is the accurate way to extract a force from a finite element solution — the reaction is exactly
conjugate to the constraint, whereas a surface stress integral differentiates the solution and loses
an order.

## A caveat worth stating: lift is not drag

$c_L$ is **not** held to a tight tolerance in the accompanying test, and should not be. The cylinder
centre sits at $y = 0.2$ in a channel of height $0.41$ — 0.005 off the axis — so the lift is a small
residue of two nearly-cancelling forces and is dominated by the random asymmetry of an unstructured
mesh. Measured over four refinements the drag error falls 0.39 → 0.10 → 0.04 → 0.02 % while the lift
error wanders 6.9 → 1.9 → 3.8 → 6.7 %. The sign and order of magnitude reproduce; the digits do not.
Getting lift to benchmark accuracy needs a symmetric or boundary-fitted mesh, which is a different
piece of work.

$\Delta p$ carries a smaller version of the same caveat: it is read at the nearest **pressure node**
to each probe point, so part of its error is the node offset rather than the solution — 0.00 % on
meshes that happen to land a node on the point, 0.3–0.7 % otherwise.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/navier_stokes_cylinder_dfg.py:code"
```
