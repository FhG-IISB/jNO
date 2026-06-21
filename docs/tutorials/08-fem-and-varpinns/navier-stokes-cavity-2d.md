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

## Bring your own implicit stepper (backward Euler + Newton)

`fem` hands back the semidiscrete pieces as ready-to-use JAX arrays — `fem.M` (dense mass),
`fem.state0`, and `fem.residual(w, t)` / `fem.jacobian(w, t)` (flat residual, dense Jacobian) — so we
step it implicitly with a Newton solve per step (the Jacobian is `M/Δt + ∂R/∂u`), which converges
quadratically:

```python
M, w = fem.M, fem.state0                          # dense mass + initial state, ready to use
for step in range(nsteps):                       # backward Euler
    w_prev = w
    for _ in range(8):                            # Newton
        G  = M @ (w - w_prev)/dt + fem.residual(w, t_next)
        dw = jnp.linalg.solve(M/dt + fem.jacobian(w, t_next), -G)
        w  = w + dw
```

## The result

![Animation: from a fluid at rest, the moving lid spins up a single large recirculating vortex whose
centre sits above and toward the moving-lid side; two small counter-rotating eddies form in the
bottom corners.](/jNO/assets/navier_stokes_cavity_2d.gif)

From rest, a primary vortex spins up; at Re=200 its centre sits **off-centre** (a convective effect —
Stokes flow would be symmetric) and two secondary **corner eddies** appear at the bottom — the
textbook cavity structure.

## What to notice

- **Navier–Stokes works now.** The convective term used to be silently misclassified as linear; it
  is detected as nonlinear, so the convection is actually solved (Taylor–Hood P2/P1, autodiff
  Jacobian). Steady and transient both work.
- **Bring your own integrator** again — here backward Euler + Newton on the block's
  `mass`/`residual`/`jacobian`. The Newton iteration converges quadratically.
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
--8<-- "tutorial_examples/08_fem_and_varpinns/navier_stokes_cavity_2d.py"
```
