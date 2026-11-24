# Rocket nozzle wall (coupled FVM and FEM)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/15_coupling/rocket_nozzle_wall.py" download>Download .py</a>
<a class="md-button" href="/jNO/#tutorials">All tutorials</a>
</div>

The gas of a converging-diverging nozzle (`jno.fvm`, the axisymmetric Euler equations) heats and presses the
shell around it (`jno.fem`, transient conduction and thermo-elasticity with the hoop strain). Each problem has its
own mesh; they meet on the liner through two interface conditions written as for one problem.

## Result

In eight coupling windows the nozzle starts (exit Mach 2.6) and the liner warms from 1.0 to 1.37 (nondimensional);
the wall's radial displacement reaches $5.5\times10^{-2}$. Both conditions are loads on the wall, so every window
takes a single sweep. It runs in under two minutes on a CPU.

## The code

```python
--8<-- "tutorial_examples/15_coupling/rocket_nozzle_wall.py:code"
```

## What to notice

- `jno.core([gas, wall, ...conditions])` with `coupling=jno.solve.coupling(...)`: the conditions mix the two
  problems' unknowns, each bound to its own boundary (`q` on the gas's liner, `T` and `U` on the wall's bore).
- The heat condition `K_W * Ts.d(ns) - h_gas * (T_aw - Ts)` differentiates only the wall, so it is a load on the
  wall, a Robin condition whose coefficient and recovery temperature are read from the gas. The same holds for the
  traction `traction + pl * ns`.
- `measure="axisymmetric"`: the interface weights and the loads the coupling writes on the wall carry $2\pi r$, as
  the wall's own forms do.
- The gas marches each window with its own explicit steps (`erk` with `cfl=`), and the wall with backward Euler.
- `domain.tag(name, f(x, n, name))` names the liner from the polygon's edges.

See [Coupling problems on their own meshes](../../coupling.md).
