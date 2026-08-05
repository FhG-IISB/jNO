# Transient Navier–Stokes: the lid-driven cavity (nonlinear flow)

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO/tutorial_examples/08_fem_and_varpinns/navier_stokes_cavity_2d.py" download>Download .py</a>
<a class="md-button" href="/jNO/tutorials/08-fem-and-varpinns/">Back to chapter</a>
</div>

The canonical viscous-flow benchmark — and a genuinely **nonlinear** one. The fluid starts at rest;
the top lid is set impulsively in motion and drives a recirculating vortex that spins up to steady
state, governed by the incompressible Navier–Stokes equations:

$$\partial_t\mathbf u + (\mathbf u\!\cdot\!\nabla)\mathbf u - \nu\,\Delta\mathbf u + \nabla p = 0,\qquad \nabla\!\cdot\mathbf u = 0,\qquad \mathrm{Re}=\tfrac{UL}{\nu}=200.$$

## The convective term is the whole point

`(u·∇)u` is written `inner(grad u, u)` — the unknown contracted **with itself**, a true
nonlinearity (unlike the bilinear `inner(grad u, grad v)`, which is trial × *test*). `jno.fem`
detects this and routes the coupled system to its nonlinear operator, with the Jacobian from
autodiff:

```python
conv = inner(gu, ub, n_contract=1)                            # (u·∇)u
momentum = inner(ub.t, vb, n_contract=1) + inner(conv, vb, n_contract=1) \
         + nu*inner(gu, gv, n_contract=2) - pp*trace(gv)
fem = jno.fem([momentum, -qq*trace(gu), ...lid + no-slip + IC...])
assert fem.is_transient and not fem.is_linear                  # transient + nonlinear + coupled
```

## `fem.solve()` does the implicit stepping

`fem.solve()` integrates the nonlinear transient system itself — **backward Euler with a Newton solve
per step** — and returns the differentiable forward trajectory. For a transient solve the result is a
*trace node*, so we read the concrete trajectory (one DOF vector per time step) through a minimal crux:

```python
sol  = fem.solve()                                   # the implicit stepping happens inside
traj = np.asarray(jno.core([sol.mse]).eval([sol]))   # (n_steps, dofs) over the time grid
```

## The result

![Animation: from a fluid at rest, the moving lid spins up a single large recirculating vortex whose
centre sits above and toward the moving-lid side; two small counter-rotating eddies form in the
bottom corners.](/jNO/assets/navier_stokes_cavity_2d.gif)

From rest, a primary vortex spins up; at Re=200 its centre sits **off-centre** (a convective effect —
Stokes flow would be symmetric) and two secondary **corner eddies** appear at the bottom — the
textbook cavity structure.

## What to notice

- **The convection is actually solved.** The convective term is detected as nonlinear, so `fem.solve`
  drives a Newton iteration on it (Taylor–Hood P2/P1, autodiff Jacobian) rather than dropping it into
  a linear solve. Steady and transient both work.
- **No hand-rolled stepping** — `fem.solve()` does the backward-Euler + Newton implicit stepping
  internally; you just read the returned trajectory.
- **Verified by physics:** the flow reaches steady state (the last frames stop changing) and shows a
  genuine recirculation — the centre-line $u_x$ is $+$ near the lid and $-$ near the floor.

## Scope note

This is a **closed** (lid-driven) flow, so it is all-Dirichlet and needs no outflow boundary. Open
flows with **vortex shedding** (a cylinder in a channel) additionally need a *natural outflow*
boundary — a current jno.fem gap — and, for a developed von Kármán street, long time integration on
a fine wake mesh; in 3D that is beyond a single 8 GB GPU. The nonlinear Navier–Stokes machinery
shown here is the prerequisite for all of that.

## Full script

```python
--8<-- "tutorial_examples/08_fem_and_varpinns/navier_stokes_cavity_2d.py:code"
```
