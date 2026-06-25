"""12 - A 2D "pot heated from below": Rayleigh-Benard convection (heat + fluid flow, fully coupled).

Heat a layer of fluid from below and it does not just conduct -- above a critical temperature
difference the hot fluid becomes buoyant, rises, cold fluid sinks, and the layer breaks into rolling
convection cells. This is the **Boussinesq** model: incompressible Navier-Stokes with a buoyancy body
force proportional to temperature, two-way coupled to an advection-diffusion equation for the heat.

    du/dt + (u.grad)u = -grad p + Pr lap u + Pr*Ra * T e_y      (momentum + buoyancy)
    div u = 0                                                    (incompressible)
    dT/dt + u.grad T  = lap T                                    (heat: advected + diffused)

Three coupled fields -- velocity ``u`` (P2), pressure ``p`` (P1), temperature ``T`` (P1) -- written as
a single ``jno.fem([...])``. The coupling is genuinely two-way: **buoyancy** ``Pr*Ra*T`` feeds heat
into the momentum balance (a linear cross term), and **advection** ``u.grad T`` feeds the flow into the
heat balance (a product of two *different* unknowns -> nonlinear). The whole system routes through the
coupled nonlinear Newton path; we march it in time with our own backward-Euler + Newton stepper (the
Navier-Stokes-cavity pattern) and watch the rolls grow from rest.

Boundary/initial conditions for the pot: **no-slip** walls (``u=0``), a **hot floor / cold lid** with
the conductive profile held on the walls (``T = 1 - y``), and the fluid starting **at rest** from the
conductive state plus a tiny perturbation that seeds the instability. The animation is the *computed*
temperature with the *computed* velocity arrows -- nothing painted in.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import jax

jax.config.update("jax_enable_x64", True)  # the assembler builds in float64

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

Pr, Ra = 1.0, 1.0e4  # Prandtl, Rayleigh (Ra >> Ra_c ~ 1708 -> vigorous convection)
Lx, Ly = 2.0, 1.0  # a wide-ish pot -> a pair of counter-rotating rolls

d = jno.domain(box(0, 0, Lx, Ly), mesh_size=0.11, time=(0.0, 0.3, 2))
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
T, sT = d.fem_symbols(names=("T", "sT"), order=1)  # P1 temperature
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ci = d.variable("initial", split=True)
ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
pb, qb = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
Tb, sb = T.bind(x=xi, y=yi, t=ti), sT.bind(x=xi, y=yi, t=ti)

ux, uy, vx, vy = ub[0], ub[1], vb[0], vb[1]
uxx, uxy, uyx, uyy = ub.x[0], ub.y[0], ub.x[1], ub.y[1]  # grad-then-index: d u_i / d x_j
vxx, vxy, vyx, vyy = vb.x[0], vb.y[0], vb.x[1], vb.y[1]
momentum = (
    (ub.t[0] * vx + ub.t[1] * vy)  # du/dt
    + ((ux * uxx + uy * uxy) * vx + (ux * uyx + uy * uyy) * vy)  # (u.grad)u  -- nonlinear
    + Pr * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)  # Pr grad u : grad v
    - pb * (vxx + vyy)  # -p div v
    - Pr * Ra * Tb * vy  # buoyancy: temperature -> momentum
)
continuity = qb * (uxx + uyy)  # div u = 0
energy = Tb.t * sb + (ux * Tb.x + uy * Tb.y) * sb + (Tb.x * sb.x + Tb.y * sb.y)  # dT/dt + u.grad T - lap T

Tcond = 1.0 - ci[1] / Ly  # conductive profile: 1 (hot) at the floor, 0 (cold) at the lid
T0 = Tcond + 0.05 * jno.np.sin(2 * np.pi * ci[0] / Lx) * jno.np.sin(np.pi * ci[1] / Ly)  # seed the rolls
fem = jno.fem(
    [
        momentum,
        continuity,
        energy,
        u(xb, yb) - 0.0,  # no-slip walls (all-component vector Dirichlet)
        T(xb, yb) - (1.0 - yb / Ly),  # hot floor / cold lid, conductive profile on the walls
        p.pin(),  # gauge-fix: remove the pressure null space
        u(ci[0], ci[1]) - 0.0,  # start at rest (all-component vector initial condition)
        T(ci[0], ci[1]) - T0,
    ]
)
assert fem.is_transient and not fem.is_linear, "Boussinesq convection is transient + nonlinear"
off = fem.offsets
M, dt, nsteps, nframes = fem.M, 0.009, 26, 13  # stop ~when the rolls establish (no static tail)
print(f"\n2D Rayleigh-Benard pot (Ra={Ra:g}, Pr={Pr:g}): dofs={fem.dofs}, steps={nsteps}")

# bring-your-own implicit integrator: backward Euler + Newton  ((M/dt + dR/du) du = -G).
# fem.residual / fem.jacobian are already jitted, so each step is fast after the first.
w = fem.state0
frames, save_every = [np.asarray(w)], max(1, nsteps // nframes)
for step in range(nsteps):
    w_prev, t_next = w, (step + 1) * dt
    for _ in range(8):  # Newton
        G = M @ (w - w_prev) / dt + fem.residual(w, t_next)
        dw = jnp.linalg.solve(M / dt + fem.jacobian(w, t_next), -G)
        w = w + dw
        if float(jnp.linalg.norm(dw)) < 1e-8:
            break
    if (step + 1) % save_every == 0:
        frames.append(np.asarray(w))
frames = np.stack(frames)

pts_v = np.asarray(fem.field_points[0])  # P2 velocity nodes
pts_T = np.asarray(fem.field_points[2])  # P1 temperature nodes
vel = frames[:, off[0] : off[1]].reshape(frames.shape[0], -1, 2)
Tf = frames[:, off[2] :]  # temperature frames
tris = np.asarray(d.built_mesh.cells_dict["triangle"])
triT = mtri.Triangulation(pts_T[:, 0], pts_T[:, 1], tris)
umax0, umaxF = float(np.abs(vel[0]).max()), float(np.abs(vel[-1]).max())
# convective vertical heat flux <u_y * T> (T interpolated to the velocity nodes): positive for
# Rayleigh-Benard convection -- hot fluid rises (u_y>0, T high), cold sinks (u_y<0, T low).
Tvel = np.asarray(mtri.LinearTriInterpolator(triT, Tf[-1])(pts_v[:, 0], pts_v[:, 1]))
conv_flux = float(np.nanmean(vel[-1, :, 1] * Tvel))
print(f"  convection onset: max|u| {umax0:.3f} (rest) -> {umaxF:.2f}  |  convective heat flux <u_y T> = {conv_flux:+.3f}")
assert umaxF > 1.0 and conv_flux > 0.0, "expected convection to develop with upward heat transport"

# ---- animate the computed temperature with the computed velocity arrows -> GIF ----
step_q = max(1, len(pts_v) // 110)  # subsample arrows so the field stays visible
qscale = max(umaxF, 1.0) / 0.09  # fixed scale (data units): the fastest arrow spans ~0.09 of the box
fig, ax = plt.subplots(figsize=(8.2, 4.4))
tpc = ax.tripcolor(triT, Tf[0], cmap="RdBu_r", shading="gouraud", vmin=0.0, vmax=1.0)
qv = ax.quiver(
    pts_v[::step_q, 0],
    pts_v[::step_q, 1],
    vel[0, ::step_q, 0],
    vel[0, ::step_q, 1],
    color="k",
    scale_units="xy",
    scale=qscale,
    width=0.0026,
)
fig.colorbar(tpc, ax=ax, shrink=0.85, label="temperature $T$")
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])


def _frame(j):
    tpc.set_array(Tf[j])
    qv.set_UVC(vel[j, ::step_q, 0], vel[j, ::step_q, 1])
    ax.set_title(f"Rayleigh–Bénard pot (Ra={Ra:g}) — hot floor drives convection, frame {j}/{len(Tf) - 1}", fontsize=10)
    return tpc, qv


ani = animation.FuncAnimation(fig, _frame, frames=len(Tf), interval=90, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "rayleigh_benard_2d.gif", writer="pillow", fps=10, dpi=84)
