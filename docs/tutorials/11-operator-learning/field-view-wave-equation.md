# FieldView — wave-equation PINN audit

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/11_operator_learning/field_view_wave_equation.py" download>Download .py</a>
<a class="md-button" href="/jNO_docs/tutorials/11-operator-learning/">Back to chapter</a>
</div>

A two-stage tutorial for the 2-D wave equation. **Stage 1** trains a DeepONet
PINN via `crux.solve()` (AD-based `u.tt`, `u.xx`, `u.yy` through the network
graph). **Stage 2** audits the **trained network's own prediction**: it evaluates
the trained model on a structured grid and re-checks the wave equation with
second-order *finite differences* on that grid output — an independent,
discretisation-level test of whether the model satisfies the PDE, rather than a
comparison against a hand-built analytic field.

## Problem setup

```text
u_tt = c²(u_xx + u_yy)   on [0,1]², t ∈ [0, T_max]
u = 0                     on ∂[0,1]² (Dirichlet)
```

Mode-(1,1) standing wave: `u(t, x, y) = cos(ωt) sin(πx) sin(πy)`, with
`ω = c·π√2`, so `u_tt = −ω² u = c²(u_xx + u_yy)`.

## Stage 1 — PINN training

A DeepONet maps `(t, x, y)` to scalar `u`. `.scalar.bind()` registers the
coordinate variables so `u.tt`, `u.xx`, `u.yy` trace **AD** derivatives through
the network.

```python
dom_pinn = jno.domain.rect(mesh_size=0.05, time=(0, T_MAX, 6))
x_p, y_p, t_p = dom_pinn.variable("interior")
x0_p, y0_p, t0_p = dom_pinn.variable("initial")
xb_p, yb_p, tb_p = dom_pinn.variable("boundary")

net = jno.nn.wrap(foundax.deeponet(n_sensors=1, coord_dim=2, ...))

u_p = net(t_p, xy_p).scalar.bind(x=x_p, y=y_p, t=t_p)
pde_wave = u_p.tt - C**2 * (u_p.xx + u_p.yy)         # wave PDE
ic_disp  = net(t0_p, xy0_p) - sin(π*x0)*sin(π*y0)    # u(0) = sin·sin
ic_vel   = u0_p.t                                    # u_t(0) = 0
bc_wall  = net(tb_p, xy_b)                           # Dirichlet u = 0

crux_pinn = jno.core([pde_wave.mse, ic_disp.mse, ic_vel.mse, bc_wall.mse], domain=dom_pinn)
crux_pinn.solve(EPOCHS_TRAIN)
```

Both initial conditions are required for the second-order wave equation:
displacement and velocity at `t = 0`.

## Stage 2 — FieldView audit of the trained prediction

Sampled on a grid, the coordinates `x, y, t` are **axes**, not network inputs, so
`net(t, xy).field.bind(...)` differentiates the model's grid output with an FD
stencil instead of AD:

```python
T, H = 20, 32
dom_fd = jno.domain.equi_distant_rect(nx=H - 1, ny=H - 1, time=(0.0, T_MAX, T))
xg, yg, tg = dom_fd.variable("interior")

u = net(tg, jno.np.concat([xg, yg])).field.bind(x=xg, y=yg, t=tg)
wave_res = u.tt - C**2 * (u.xx + u.yy)     # nested temporal + spatial FD
```

Everything is evaluated **through the trained crux** on the FD grid, with
`min_consecutive=T` so the nested `u.tt` stencil sees every frame:

```python
pred, exact = crux_pinn.eval([net(tg, xyg), analytic], domain=dom_fd, min_consecutive=T)
res_mse = crux_pinn.eval(wave_res.expr.mse, domain=dom_fd, min_consecutive=T)
```

`crux.eval(..., domain=dom_fd)` reuses the trained weights but swaps in the grid
domain, so the FD residual is computed on exactly the network the PINN produced.

!!! note "Same PDE, two derivative engines"
    - **Stage 1** `.scalar.bind()` → **AD** through the network graph (training).
    - **Stage 2** `.field.bind()` → **FD** on the grid output (audit).

    The expression `u.tt - c²(u.xx + u.yy)` is written identically both times — one
    differentiates the graph, the other the grid values.

The audit reports the prediction's relative L2 against the closed-form solution
and the **FD wave residual** of that same prediction — how well the trained
network satisfies `u_tt = c²Δu` when checked discretely on the grid.

!!! warning "PINN accuracy is run-dependent"
    The wave PINN is ill-conditioned; combined with XLA autotune and the
    persistent compile cache, training can land in different basins between runs.
    The script's asserts are loose guards (`rel_l2 < 1`) that confirm the model
    beats a trivial predictor — treat the printed numbers as indicative and
    confirm final accuracy/timing on GPU.

## What to notice

- **Audit the model, not a stand-in.** Stage 2 differentiates the trained
  network's *own* output — a discretisation-level check that the PINN satisfies
  the PDE, independent of the AD objective it was trained on.
- **`net(t, xy).field.bind(...)` is live.** FieldView wraps the model call
  directly; there is no need to materialise and store an array first.
- **`u.tt` is nested FD.** The second temporal derivative applies the
  central-difference stencil twice over the buffered window; `min_consecutive=T`
  guarantees the frames are present.
- **`crux.eval(domain=...)`** runs the trained weights on a different (grid)
  domain — the standard way to evaluate a jNO model on new inputs.

<div class="hero-actions" markdown>
<a class="md-button md-button--primary" href="/jNO_docs/tutorial_examples/11_operator_learning/field_view_wave_equation.py" download>Download full script</a>
<a class="md-button" href="/jNO_docs/tutorials/11-operator-learning/">Back to 11 Operator Learning</a>
</div>

## Script

```python
--8<-- "tutorial_examples/11_operator_learning/field_view_wave_equation.py"
```
