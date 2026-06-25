"""Buoyancy-driven Navier-Stokes: a laminar smoke plume rising in a rectangular domain.

Nondimensional Boussinesq system (scales: width W, diffusion time W²/ν, ΔC = 1):

    du/dt + (u·∇)u − Pr ∇²u + ∇p = Pr Ra C ĵ   (momentum + buoyancy)
    ∇·u = 0                                       (mass)
    dC/dt + u·∇C − (1/Sc) ∇²C = 0               (scalar transport)

Pr = ν/κ (Prandtl-like number, set to 1), Ra = g β ΔC W³/(ν κ) (Rayleigh),
Sc = ν/D (Schmidt, set to 1 so concentration diffuses at the same rate as momentum).
The body force Pr Ra C ĵ is upward for positive C (buoyant smoke).

Geometry: 1 × 2 (width × height) rectangular box. Smoke enters continuously at the
bottom centre (|x − 0.5| ≤ 0.12, C = 1 there). All four walls are no-slip (u = 0)
and the top/side walls are absorbing (C = 0) — the domain acts as a closed box. From
rest the buoyant plume rises, with counter-rotating recirculation on either side.

Fields: Taylor-Hood P2 velocity / P1 pressure (inf-sup stable) and P1 concentration.
Time integration uses a hand-rolled backward-Euler + Newton loop, identical to the
Rayleigh-Bénard tutorial (rayleigh_benard_2d.py).

Verified by the convective vertical scalar flux ⟨u_y C⟩ > 0: positive means the
computed velocity is genuinely transporting the buoyant scalar upward.

References
----------
[1] J. Boussinesq, "Théorie analytique de la chaleur," Gauthier-Villars (1903),
    §XI — the Boussinesq density-coupling approximation.
[2] O. Reynolds, "On the extent and action of the heating surface of steam boilers,"
    Manchester Lit. Phil. Soc. 14 (1874) — early buoyancy-driven convection context;
    see also Drazin & Reid, *Hydrodynamic Stability*, §2.1 for the modern nondimensional
    form (Ra, Pr) used here.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely.geometry import box

import jno

# ── Physical parameters ───────────────────────────────────────────────────────
Pr = 1.0  # Prandtl number
Ra = 5e3  # Rayleigh number  — vigorous laminar plume, amenable to Newton
Sc = 1.0  # Schmidt number (C diffuses like momentum at Sc=1)
W, H = 1.0, 2.0  # domain width × height
SRC_LO, SRC_HI = 0.38, 0.62  # x-extent of the bottom smoke inlet

# ── Domain ────────────────────────────────────────────────────────────────────
d = jno.domain(box(0, 0, W, H), mesh_size=0.10, time=(0.0, 1.0, 2))

# Taylor-Hood: P2 velocity (u/v trial/test), P1 pressure (p/q), P1 concentration (C/sC)
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
p, q = d.fem_symbols(names=("p", "q"), order=1)
C, sC = d.fem_symbols(names=("C", "sC"), order=1)

xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ci = d.variable("initial", split=True)

ub = u.bind(x=xi, y=yi, t=ti)
vb = v.bind(x=xi, y=yi, t=ti)
pb = p.bind(x=xi, y=yi, t=ti)
qb = q.bind(x=xi, y=yi, t=ti)
Cb = C.bind(x=xi, y=yi, t=ti)
sCb = sC.bind(x=xi, y=yi, t=ti)

ux, uy = ub[0], ub[1]
uxx, uxy = ub.x[0], ub.y[0]
uyx, uyy = ub.x[1], ub.y[1]
vx, vy = vb[0], vb[1]
vxx, vxy = vb.x[0], vb.y[0]
vyx, vyy = vb.x[1], vb.y[1]

# Weak forms (integrated by parts, all in one jno.fem block)
momentum = (
    (ub.t[0] * vx + ub.t[1] * vy)  # du/dt · v
    + ((ux * uxx + uy * uxy) * vx + (ux * uyx + uy * uyy) * vy)  # (u·∇u)·v
    + Pr * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)  # Pr ∇u:∇v
    - pb * (vxx + vyy)  # −p ∇·v
    - Pr * Ra * Cb * vy  # Pr Ra C ĵ·v  (buoyancy upward)
)
continuity = qb * (uxx + uyy)  # q ∇·u = 0
conc = (
    Cb.t * sCb  # dC/dt
    + (ux * Cb.x + uy * Cb.y) * sCb  # u·∇C
    + (1.0 / Sc) * (Cb.x * sCb.x + Cb.y * sCb.y)  # (1/Sc) ∇C·∇sC
)

# ── Boundary / initial conditions ─────────────────────────────────────────────
inlet = jno.np.where(
    yb < 1e-6,
    jno.np.where(xb >= SRC_LO, jno.np.where(xb <= SRC_HI, 1.0, 0.0), 0.0),
    0.0,
)
fem = jno.fem(
    [
        momentum,
        continuity,
        conc,
        u(xb, yb) - 0.0,  # no-slip on all walls
        C(xb, yb) - inlet,  # C=1 at smoke inlet, C=0 elsewhere (absorbing walls)
        p.pin(),  # remove pressure null space
        u(ci[0], ci[1]) - 0.0,  # start at rest
        C(ci[0], ci[1]) - 0.0,  # no smoke at t=0
    ]
)
assert fem.is_transient and not fem.is_linear

off = fem.offsets
print(f"\nSmoke plume (Ra={Ra:.0e}, Pr={Pr}, Sc={Sc}): dofs={fem.dofs}, fields={len(off)}")

# ── Time integration: backward Euler + Newton ─────────────────────────────────
M, dt, nsteps, nframes = fem.M, 0.003, 80, 16
w = fem.state0
frames, save_every = [np.asarray(w)], max(1, nsteps // nframes)
for step in range(nsteps):
    w_prev, t_next = w, (step + 1) * dt
    for it in range(10):
        G = M @ (w - w_prev) / dt + fem.residual(w, t_next)
        dw = jnp.linalg.solve(M / dt + fem.jacobian(w, t_next), -G)
        w = w + dw
        if float(jnp.linalg.norm(dw)) < 1e-8:
            break
    if (step + 1) % save_every == 0:
        frames.append(np.asarray(w))
frames = np.stack(frames)
print(f"  completed {nsteps} steps ({nsteps * dt:.2f} diffusion times), {len(frames)} frames saved")

# ── Post-processing ───────────────────────────────────────────────────────────
pts_v = np.asarray(fem.field_points[0])  # P2 velocity nodes
pts_C = np.asarray(fem.field_points[2])  # P1 concentration nodes
vel = frames[:, off[0] : off[1]].reshape(frames.shape[0], -1, 2)
Cf = frames[:, off[2] :]  # concentration DOFs per frame

tris = np.asarray(d.built_mesh.cells_dict["triangle"])
triC = mtri.Triangulation(pts_C[:, 0], pts_C[:, 1], tris)

# Convective vertical scalar flux: ⟨u_y C⟩ (C interpolated to velocity nodes)
Cvel = np.asarray(mtri.LinearTriInterpolator(triC, Cf[-1])(pts_v[:, 0], pts_v[:, 1]))
conv_flux = float(np.nanmean(vel[-1, :, 1] * Cvel))
uy_max = float(np.abs(vel[-1, :, 1]).max())
print(f"  max |u_y| = {uy_max:.3f}   convective flux ⟨u_y C⟩ = {conv_flux:+.4f}")

# ── Animation ─────────────────────────────────────────────────────────────────
step_q = max(1, len(pts_v) // 100)
qscale = max(uy_max, 0.5) / 0.07  # scale so the fastest arrow spans ~7 % of the box

fig, ax = plt.subplots(figsize=(5.0, 8.5))
tpc = ax.tripcolor(triC, Cf[0], cmap="YlOrRd", shading="gouraud", vmin=0.0, vmax=1.0)
qv = ax.quiver(
    pts_v[::step_q, 0],
    pts_v[::step_q, 1],
    vel[0, ::step_q, 0],
    vel[0, ::step_q, 1],
    color="k",
    scale_units="xy",
    scale=qscale,
    width=0.003,
)
fig.colorbar(tpc, ax=ax, shrink=0.85, label="smoke concentration $C$")
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.plot([SRC_LO, SRC_HI], [0, 0], color="steelblue", lw=4, label="smoke inlet ($C=1$)")
ax.legend(loc="upper right", fontsize=9)


def _frame(j):
    tpc.set_array(Cf[j])
    qv.set_UVC(vel[j, ::step_q, 0], vel[j, ::step_q, 1])
    ax.set_title(
        f"Buoyant smoke plume (Ra={Ra:.0e}) — frame {j}/{len(frames) - 1}\n"
        f"jno.fem, P2/P1/P1 Taylor-Hood + Boussinesq, closed box",
        fontsize=9,
    )
    return tpc, qv


ani = animation.FuncAnimation(fig, _frame, frames=len(frames), interval=130, blit=False)
_out = Path(__file__).parents[2] / "assets" / "smoke_plume_2d.gif"
ani.save(_out, writer="pillow", fps=8, dpi=90)
print(f"  saved → docs/assets/{_out.name}")

assert conv_flux > 0.0, f"upward convective transport expected: ⟨u_y C⟩ = {conv_flux:.4f}"
assert uy_max > 0.05, f"buoyancy should drive upward velocity: max |u_y| = {uy_max:.3f}"
