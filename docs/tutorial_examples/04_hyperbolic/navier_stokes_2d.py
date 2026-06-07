"""04 — 2-D unsteady Navier-Stokes: Taylor-Green vortex

Problem
-------
    ∂u/∂t + u·∇u = −∇p + (1/Re)∇²u,   ∇·u = 0
    Ω = [0, 1]²,   t ∈ [0, T]

Analytical solution (Taylor-Green, Re = 100)
--------------------------------------------
    u =  −cos(πx) sin(πy) exp(−2π²t/Re)
    v =   sin(πx) cos(πy) exp(−2π²t/Re)
    p = −(cos(2πx)+cos(2πy)) exp(−4π²t/Re) / 4

The vortex decays through viscosity; the GIF shows the vorticity field
fading from its initial checkerboard pattern toward quiescence.

This exercises every unsteady-NS operator: ∂t, u·∇u, ∇p, ∇²u, ∇·u = 0.
"""

import io
from pathlib import Path

import foundax
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from PIL import Image as PILImage

import jno

π = jno.np.pi
Re = 100.0
T_end = 2.0
N_t = 10

# ── Domain ────────────────────────────────────────────────────────────────────
domain = jno.domain(
    constructor=jno.domain.rect(mesh_size=0.05),
    time=(0, T_end, N_t),
)
x, y, t = domain.variable("interior")
x0, y0, t0 = domain.variable("initial")

# ── Analytical solution ───────────────────────────────────────────────────────
decay = jno.np.exp(-2.0 * π**2 * t / Re)
u_exact = -jno.np.cos(π * x) * jno.np.sin(π * y) * decay
v_exact = jno.np.sin(π * x) * jno.np.cos(π * y) * decay
p_exact = -(jno.np.cos(2 * π * x) + jno.np.cos(2 * π * y)) * jno.np.exp(-4.0 * π**2 * t / Re) / 4.0

# ── Networks ──────────────────────────────────────────────────────────────────
def _net(key):
    return jno.nn.wrap(
        foundax.deeponet(
            n_sensors=1,
            coord_dim=2,
            n_outputs=1,
            n_layers=6,
            basis_functions=128,
            hidden_dim=96,
            activation=jax.nn.tanh,
            key=key,
        )
    )


u_net = _net(jax.random.PRNGKey(1))
v_net = _net(jax.random.PRNGKey(2))
p_net = _net(jax.random.PRNGKey(3))

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0, peak_value=1e-3, warmup_steps=200, decay_steps=49800, end_value=1e-5,
)
for net in [u_net, v_net, p_net]:
    net.optimizer(optax.adam(schedule))

# ── Symbolic fields ───────────────────────────────────────────────────────────
xy = jno.np.concat([x, y])
u = u_net(t, xy)
v = v_net(t, xy)
p = p_net(t, xy)

# ── PDE residuals ─────────────────────────────────────────────────────────────
nu = 1.0 / Re
mom_u = u.d(t) + u * u.d(x) + v * u.d(y) + p.d(x) - nu * jno.np.laplacian(u, [x, y])
mom_v = v.d(t) + u * v.d(x) + v * v.d(y) + p.d(y) - nu * jno.np.laplacian(v, [x, y])
cont  = u.d(x) + v.d(y)

# ── Initial conditions ────────────────────────────────────────────────────────
xy0 = jno.np.concat([x0, y0])
decay0 = jno.np.exp(-2.0 * π**2 * t0 / Re)
ic_u = u_net(t0, xy0) - (-jno.np.cos(π * x0) * jno.np.sin(π * y0) * decay0)
ic_v = v_net(t0, xy0) - ( jno.np.sin(π * x0) * jno.np.cos(π * y0) * decay0)

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([mom_u.mse, mom_v.mse, cont.mse, ic_u.mse, ic_v.mse], domain)
crux.solve(50_000)

# ── Assertion ─────────────────────────────────────────────────────────────────
_u, _v, _ue, _ve = crux.eval([u, v, u_exact, v_exact])
_u = jnp.asarray(_u); _v = jnp.asarray(_v)
_ue = jnp.asarray(_ue); _ve = jnp.asarray(_ve)
err = jnp.sqrt(jnp.mean((_u - _ue) ** 2 + (_v - _ve) ** 2))
ref = jnp.sqrt(jnp.mean(_ue**2 + _ve**2)) + 1e-8
rel_l2 = float(err / ref)
assert rel_l2 < 0.2, f"relative L2 = {rel_l2:.3e}"

# ── GIF — vorticity field at each snapshot ────────────────────────────────────
N_vis = 48
x_vis = jnp.linspace(0, 1, N_vis)
y_vis = jnp.linspace(0, 1, N_vis)
xx, yy = jnp.meshgrid(x_vis, y_vis)
xy_flat = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)
dx = float(x_vis[1] - x_vis[0])
t_snaps = jnp.linspace(0, T_end, N_t)

frames_data = []
for t_val in t_snaps:
    t_b = jnp.full((N_vis**2, 1), float(t_val))
    _u_g = np.asarray(jax.vmap(u_net.module)(t_b, xy_flat)).reshape(N_vis, N_vis)
    _v_g = np.asarray(jax.vmap(v_net.module)(t_b, xy_flat)).reshape(N_vis, N_vis)
    omega = np.gradient(_v_g, dx, axis=1) - np.gradient(_u_g, dx, axis=0)
    frames_data.append((_u_g, _v_g, omega))

clim = max(np.abs(w).max() for _, _, w in frames_data)

gif_frames = []
for t_val, (_u_g, _v_g, omega) in zip(t_snaps, frames_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    cf = ax.contourf(
        np.asarray(xx), np.asarray(yy), omega,
        levels=40, vmin=-clim, vmax=clim, cmap="RdBu_r",
    )
    step = N_vis // 10
    ax.quiver(
        np.asarray(xx)[::step, ::step], np.asarray(yy)[::step, ::step],
        _u_g[::step, ::step], _v_g[::step, ::step],
        scale=6, width=0.004, color="k", alpha=0.6,
    )
    plt.colorbar(cf, ax=ax, fraction=0.046, label="ω")
    ax.set_title(f"Vorticity   t = {float(t_val):.2f}", fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    gif_frames.append(PILImage.open(buf).copy())

gif_path = Path(__file__).parent / "taylor_green_vortex.gif"
gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], loop=0, duration=500)
print(f"GIF saved → {gif_path}")
