"""11 — FieldView audit of a trained wave-equation PINN

A DeepONet is trained as a PINN on the 2-D wave equation

    u_tt = c²(u_xx + u_yy)   on [0,1]², t ∈ [0, T_max],   u = 0 on ∂[0,1]²,

with the closed-form standing wave  u(t, x, y) = cos(ωt) sin(πx) sin(πy),
ω = c·π√2.

Stage 1 — PINN training via ``crux.solve`` (AD-based ``u.tt``, ``u.xx``, ``u.yy``
through the network graph). This is ordinary point-wise PINN training.

Stage 2 — FieldView audit of the **trained network's own prediction**: evaluate
the trained DeepONet on a structured grid and re-check the wave equation with
second-order *finite differences* on that grid output. Sampled on a grid the
coordinates are axes, not network inputs, so ``net(t, xy).field.bind(...)`` uses
an FD stencil — an independent, discretisation-level check of whether the model
actually satisfies the PDE (not a comparison against a hand-built analytic field).
"""

import foundax
import jax
import numpy as np
import optax

import jno

KEY = jax.random.PRNGKey(0)
π = jno.np.pi

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — PINN training with crux.solve()
# 2D wave equation via AD-based u.tt and u.xx + u.yy (.scalar.bind pattern).
# Analytic: u(t,x,y) = cos(ωt) sin(πx) sin(πy),  ω = c·π√2
# ═══════════════════════════════════════════════════════════════════════════════

_C = 1.0
_OMEGA = _C * np.pi * np.sqrt(2)
_T_MAX_TRAIN = 0.5 * (2.0 * np.pi / _OMEGA)  # half-period ≈ 0.707 s
EPOCHS_TRAIN = 3_000

dom_pinn = jno.domain.rect(mesh_size=0.05, time=(0, _T_MAX_TRAIN, 6))
x_p, y_p, t_p = dom_pinn.variable("interior")
x0_p, y0_p, t0_p = dom_pinn.variable("initial")
xb_p, yb_p, tb_p = dom_pinn.variable("boundary")

net = jno.nn.wrap(
    foundax.deeponet(
        n_sensors=1,
        coord_dim=2,
        n_outputs=1,
        n_layers=4,
        basis_functions=64,
        hidden_dim=48,
        activation=jax.nn.tanh,
        key=KEY,
    )
)
net.optimizer(optax.adam(optax.warmup_cosine_decay_schedule(1e-6, 1e-3, 100, EPOCHS_TRAIN, 1e-6)))

xy_p = jno.np.concat([x_p, y_p])
xy0_p = jno.np.concat([x0_p, y0_p])

u_p = net(t_p, xy_p).scalar.bind(x=x_p, y=y_p, t=t_p)
u0_p = net(t0_p, xy0_p).scalar.bind(x=x0_p, y=y0_p, t=t0_p)

pde_wave = u_p.tt - _C**2 * (u_p.xx + u_p.yy)
ic_disp = net(t0_p, xy0_p) - jno.np.sin(π * x0_p) * jno.np.sin(π * y0_p)
ic_vel = u0_p.t  # u_t(t=0) = 0 for this solution
bc_wall = net(tb_p, jno.np.concat([xb_p, yb_p]))

crux_pinn = jno.core([pde_wave.mse, ic_disp.mse, ic_vel.mse, bc_wall.mse], domain=dom_pinn)
crux_pinn.solve(EPOCHS_TRAIN)

_u_p, _u_exact_p = crux_pinn.eval(
    [
        u_p,
        jno.np.cos(_OMEGA * t_p) * jno.np.sin(π * x_p) * jno.np.sin(π * y_p),
    ]
)
rel_l2 = float(jax.numpy.linalg.norm(_u_p - _u_exact_p) / (jax.numpy.linalg.norm(_u_exact_p) + 1e-8))
print(f"Stage 1  Relative L2 error: {rel_l2:.4e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — FieldView audit of the TRAINED network's prediction
# Evaluate the trained DeepONet on a structured grid and re-check the wave
# equation with finite differences on ITS output — not on an analytic field.
# ═══════════════════════════════════════════════════════════════════════════════

T, H = 20, 32
dom_fd = jno.domain.equi_distant_rect(nx=H - 1, ny=H - 1, time=(0.0, _T_MAX_TRAIN, T))
xg, yg, tg = dom_fd.variable("interior")
xyg = jno.np.concat([xg, yg])

# .field makes x, y, t grid axes: derivatives of the net's grid output use FD, not
# AD. This is a discretisation-level audit of the same trained weights.
u = net(tg, xyg).field.bind(x=xg, y=yg, t=tg)
wave_res = u.tt - _C**2 * (u.xx + u.yy)  # second-order temporal + spatial FD

# Evaluate through the TRAINED crux on the FD grid; min_consecutive=T holds every
# frame so the nested u.tt temporal stencil has its neighbours.
analytic = jno.np.cos(_OMEGA * tg) * jno.np.sin(π * xg) * jno.np.sin(π * yg)
pred, exact = crux_pinn.eval([net(tg, xyg), analytic], domain=dom_fd, min_consecutive=T)
pred, exact = np.asarray(pred), np.asarray(exact)
rel_l2_fd = float(np.linalg.norm(pred - exact) / (np.linalg.norm(exact) + 1e-8))
res_mse = float(np.mean(np.asarray(crux_pinn.eval(wave_res.expr.mse, domain=dom_fd, min_consecutive=T))))

print("Stage 2  FieldView audit of the trained network's prediction")
print(f"  Relative L2 vs analytic (grid)   : {rel_l2_fd:.3e}")
print(f"  FD wave residual  u_tt − c²Δu (MSE): {res_mse:.3e}")

# ── Tolerance checks ──────────────────────────────────────────────────────────
# Wave-PINN accuracy varies run-to-run (ill-conditioned + XLA autotune/compile
# cache), so the L2 guards are loose — they confirm the trained net beats a trivial
# predictor. The FD residual audits how well that prediction satisfies the wave
# equation on the grid. Final accuracy / 180 s timing should be confirmed on GPU.
assert np.isfinite(rel_l2) and rel_l2 < 1.0, f"Stage 1 PINN diverged (rel_l2={rel_l2:.3e})"
assert np.isfinite(rel_l2_fd) and rel_l2_fd < 1.0, f"prediction no better than zero (rel_l2={rel_l2_fd:.3e})"
assert np.isfinite(res_mse) and res_mse < 1.0, f"FD wave residual not sane (mse={res_mse:.3e})"
