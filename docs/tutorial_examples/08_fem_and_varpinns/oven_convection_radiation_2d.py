"""13 - A heated enclosure with all three heat-transfer modes: conduction + convection + radiation.

A rectangular box of air holds two solid cylinders — a **hot** one (a heater) low on the left and a
**cold** one high on the right — and the outer walls leak heat to a cooler outside through a **Robin**
(Newton-cooling) condition. Heat moves between the cylinders and out through the walls by *three*
coupled routes, all in one ``jno.fem`` problem:

* **conduction** — the diffusion term ``lap T`` smears heat through the air;
* **convection** — buoyancy lifts the warm air off the hot cylinder and lets the chilled air sink off
  the cold one (the **Boussinesq** model: incompressible Navier-Stokes with a temperature body force,
  two-way coupled to the heat equation), so two interacting plumes stir the box;
* **radiation** — the two cylinders and the four walls form a grey-body **enclosure**; each surface
  radiates across the transparent air to every other it can *see* (the cylinders also occlude each
  other), written as ``jno.np`` math on top of ``domain.enclosure(...)``.

      div u = 0,   du/dt + (u.grad)u = -grad p + Pr lap u + Pr*Ra * theta * e_y      (Boussinesq)
      dtheta/dt + u.grad theta = lap theta            (+ enclosure radiation as a surface load)
      hot cylinder: theta = 1,   cold cylinder: theta = 0,   outer walls: -dtheta/dn = Bi (theta - theta_ext)

Non-dimensional groups: Prandtl ``Pr``, Rayleigh ``Ra`` (buoyancy vs diffusion), Biot ``Bi`` (wall heat
loss), conduction-radiation number ``N_rc = sigma dT**3 L / k``, the absolute-temperature ratio
``tau = T_cold/dT`` (the absolute non-dim temperature is ``theta + tau``; radiation needs it because
``T**4`` is not offset-invariant — every surface must sit above absolute zero) and the Robin ambient
``T_ext`` (in ``theta`` units) the walls relax toward.

Fields: velocity ``u`` (P2), pressure ``p`` (P1), temperature ``T`` (P1) — one coupled ``jno.fem([...])``,
marched in time with our own backward-Euler + Newton stepper. Radiation is **lagged** (operator-split):
each step freezes the radiosity from the start-of-step wall temperatures and applies it as a consistent
surface load on the temperature equation — robust, and exact at steady state. The animation is the
*computed* temperature with the *computed* velocity arrows as the box warms from cold; nothing is painted
in (only the cylinder outlines are drawn, to mark the boundary conditions).

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4-5 (net-radiation / radiosity method
for diffuse-grey enclosures).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # CPU assembly + a sparse LU per step: fast here, avoids GPU contention/OOM
os.environ["MPLBACKEND"] = "Agg"

import jax

jax.config.update("jax_enable_x64", True)  # feax assembly is float64

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as spsp  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

import jno  # noqa: E402

# --- non-dimensional parameters ---
Pr, Ra, Bi = 0.71, 3.0e4, 1.5  # air; vigorous-but-laminar convection; moderate wall heat loss
eps_w, tau, T_ext, N_rc = 0.85, 1.0, 0.0, 0.5  # emissivity; radiation abs offset T_cold/dT; Robin ambient; cond-rad number
Lx, Ly, rc = 1.0, 0.8, 0.12
hot_c, cold_c = (0.30, 0.27), (0.70, 0.55)  # hot cylinder low-left, cold cylinder high-right -> offset plumes

# --- domain: a rectangle of fluid with two circular cut-outs (the solid cylinders) ---
fluid = box(0, 0, Lx, Ly).difference(Point(*hot_c).buffer(rc, 6)).difference(Point(*cold_c).buffer(rc, 6))
d = jno.domain(fluid, mesh_size=0.085, time=(0.0, 0.30, 2))
dh = lambda x, y: jnp.hypot(x - hot_c[0], y - hot_c[1])  # noqa: E731
dcl = lambda x, y: jnp.hypot(x - cold_c[0], y - cold_c[1])  # noqa: E731
d.tag("hot", lambda x, y: jnp.abs(dh(x, y) - rc) < 3e-2)  # hot cylinder surface
d.tag("cold", lambda x, y: jnp.abs(dcl(x, y) - rc) < 3e-2)  # cold cylinder surface
d.tag("outer", lambda x, y: (x < 1e-6) | (x > Lx - 1e-6) | (y < 1e-6) | (y > Ly - 1e-6))  # box walls (Robin)

# --- coupled Boussinesq: velocity (P2) + pressure (P1) + temperature (P1) ---
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
p, q = d.fem_symbols(names=("p", "q"), order=1)
T, sT = d.fem_symbols(names=("T", "sT"), order=1)
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
xh, yh, _ = d.variable("hot", split=True)
xc, yc, _ = d.variable("cold", split=True)
xo, yo, _ = d.variable("outer", split=True)
ci = d.variable("initial", split=True)
ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
pb, qb = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
Tb, sb = T.bind(x=xi, y=yi, t=ti), sT.bind(x=xi, y=yi, t=ti)

ux, uy, vx, vy = ub[0], ub[1], vb[0], vb[1]  # velocity / test-velocity components
# u<comp><dir> = d(u_comp)/d(dir): ub.x is the x-derivative of the vector, [i] picks component i
uxx, uxy, uyx, uyy = ub.x[0], ub.y[0], ub.x[1], ub.y[1]
vxx, vxy, vyx, vyy = vb.x[0], vb.y[0], vb.x[1], vb.y[1]
momentum = (
    (ub.t[0] * vx + ub.t[1] * vy)  # du/dt
    + ((ux * uxx + uy * uxy) * vx + (ux * uyx + uy * uyy) * vy)  # (u.grad)u  -- nonlinear
    + Pr * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)  # Pr grad u : grad v
    - pb * (vxx + vyy)  # -p div v
    - Pr * Ra * Tb * vy  # buoyancy: warm air rises
)
continuity = qb * (uxx + uyy)  # div u = 0
energy = Tb.t * sb + (ux * Tb.x + uy * Tb.y) * sb + (Tb.x * sb.x + Tb.y * sb.y)  # dT/dt + u.grad T - lap T
robin = Bi * (T.bind(x=xo, y=yo) - T_ext) * sT.bind(x=xo, y=yo)  # -dT/dn = Bi(T - T_ext): walls leak heat out
fem = jno.fem(
    [
        momentum,
        continuity,
        energy,
        robin,
        u(xb, yb) - 0.0,  # no-slip on the walls AND both cylinders
        T(xh, yh) - 1.0,  # hot cylinder
        T(xc, yc) - 0.0,  # cold cylinder
        p.pin(),  # pressure gauge-fix
        u(ci[0], ci[1]) - 0.0,  # start at rest
        T(ci[0], ci[1]) - T_ext,  # start at the outside temperature
    ]
)
assert fem.is_transient and not fem.is_linear, "the box is transient + nonlinear (Boussinesq)"
off = fem.offsets
o2 = int(off[2])  # temperature is the last field: DOFs w[o2:] are the P1 nodal temperature
nT = int(fem.dofs) - o2
pts_v = np.asarray(fem.field_points[0])  # P2 velocity nodes
pts_T = np.asarray(fem.field_points[2])  # P1 temperature nodes (== base mesh nodes)
tris = np.asarray(d.built_mesh.cells_dict["triangle"])
triT = mtri.Triangulation(pts_T[:, 0], pts_T[:, 1], tris)

# --- enclosure radiation: the cylinders + walls radiate across the *meshed* fluid -> inward normals;
#     coarse curved surfaces under-resolve F, so enforce_closure normalises it (van Leersum 1989) ---
gap = d.enclosure(["hot", "cold", "outer"], inward=True, enforce_closure=True)
gap.check()
F = gap.view_factor
eps = gap.emissivity(eps_w)
rho = 1.0 - eps
eye = jnp.eye(gap.size)
ar = np.asarray(gap.areas)
hot_m, cold_m, out_m = gap.tag_mask("hot"), gap.tag_mask("cold"), gap.tag_mask("outer")
# temperature DOFs pinned by Dirichlet (the cylinders) must NOT receive the load (it corrupts those rows)
free_T = jnp.asarray(
    ~((jnp.abs(dh(pts_T[:, 0], pts_T[:, 1]) - rc) < 3e-2) | (jnp.abs(dcl(pts_T[:, 0], pts_T[:, 1]) - rc) < 3e-2)),
    dtype=float,
)


def q_rad(Tdofs):  # grey-body radiosity (with reflections): net flux leaving each wall element
    Tk = gap.field(Tdofs) + tau  # non-dim ABSOLUTE temperature (theta + tau)
    J = jnp.linalg.solve(eye - rho[:, None] * F, eps * Tk**4)  # radiosity, units sigma*dT^4
    return J - F @ J


def rad_load(w):  # consistent surface load on the temperature block (zero on the pinned cylinder rows)
    return jnp.zeros(int(fem.dofs)).at[o2:].set(N_rc * gap.load(q_rad(w[o2:]), size=nT) * free_T)


# --- bring-your-own implicit integrator: backward Euler + Newton, radiation lagged per step ---
# The coupled Jacobian is large and sparse, so each Newton step is a **sparse LU** (a dense N x N solve
# would be far costlier); jNO hands you the raw sparse operator through ``fem.operator``.
dt, nsteps, nframes = 0.005, 30, 30  # short window, fine steps -> the early transient in detail
block = fem.operator  # assembled transient block: sparse mass / residual / Jacobian


def _csc(B):  # jax operator (BCOO or dense) -> scipy CSC (keeps only the nonzeros)
    if hasattr(B, "sum_duplicates"):  # BCOO
        B = B.sum_duplicates()
        ij = np.asarray(B.indices)
        return spsp.csc_matrix((np.asarray(B.data), (ij[:, 0], ij[:, 1])), shape=tuple(B.shape))
    return spsp.csc_matrix(np.asarray(B))


M_csc = _csc(block.M if getattr(block, "M", None) is not None else block.mass(fem.t0, {}))
print(f"\nHeated box (Ra={Ra:g}, Pr={Pr:g}, Bi={Bi:g}, N_rc={N_rc:g}): dofs={fem.dofs}, enclosure={gap.size} elements")
print(f"  view-factor closure/reciprocity = {gap.quality()[0]:.1e} / {gap.quality()[1]:.1e}")
w = np.asarray(fem.state0)
frames = [w.copy()]
for step in range(nsteps):
    w_prev, t_next = w, (step + 1) * dt
    rl = np.asarray(rad_load(jnp.asarray(w_prev)))  # lagged radiation: frozen during the step's Newton iterations
    for _ in range(8):
        wj = jnp.asarray(w)
        G = M_csc.dot((w - w_prev) / dt) + np.asarray(fem.residual(wj, t_next)) + rl
        dw = splu((M_csc / dt + _csc(block.jacobian(wj, t_next, {}))).tocsc()).solve(-G)
        w = w + dw
        if float(np.linalg.norm(dw)) < 1e-6:
            break
    if step % 8 == 0 or step == nsteps - 1:
        rate = np.linalg.norm(w - w_prev) / (dt * np.linalg.norm(w) + 1e-30)
        print(f"    step {step:2d} t={t_next:.3f}  ||dw/dt||/||w|| = {rate:.2e}")
    if (step + 1) % max(1, nsteps // nframes) == 0:
        frames.append(w.copy())
frames = np.stack(frames)

# --- diagnostics: the heat budget across the three modes, and the convection signature ---
th = np.asarray(w[o2:])
vel = np.asarray(w[off[0] : off[1]]).reshape(-1, 2)
qv_el = np.asarray(q_rad(jnp.asarray(w[o2:])))
rad_hot = N_rc * float((ar[hot_m] * qv_el[hot_m]).sum())  # radiative power leaving the hot cylinder (>0)
rad_cold = N_rc * float((ar[cold_m] * qv_el[cold_m]).sum())  # leaving the cold cylinder (<0: it absorbs)
rad_net = float((ar * qv_el).sum())  # closed grey enclosure conserves radiative energy: sum_i A_i q_i = 0
interp = mtri.LinearTriInterpolator(triT, th)
# Robin conductive loss through the walls: integral of Bi(theta - T_ext) over the outer boundary
out_e0, out_e1 = gap.elements[out_m, 0], gap.elements[out_m, 1]
out_len = np.asarray(gap.areas)[out_m]
out_th = 0.5 * (th[out_e0] + th[out_e1])
robin_loss = float((Bi * (out_th - T_ext) * out_len).sum())  # >0: heat leaves the box
plume_above = float(interp(hot_c[0], hot_c[1] + rc + 0.12))  # warm plume rises off the hot cylinder
plume_below = float(interp(hot_c[0], hot_c[1] - rc - 0.06))
print(f"  theta range [{th.min():.3f}, {th.max():.3f}]   max|u| = {np.abs(vel).max():.2f}")
print(
    f"  radiative power:  hot cylinder emits {rad_hot:+.3f}   cold cylinder {rad_cold:+.3f}   net {rad_net:+.2e} (conserved)"
)
print(f"  Robin wall loss = {robin_loss:+.3f}   |   plume above hot {plume_above:.3f} vs below {plume_below:.3f}")

# --- validation (mesh-robust property checks; no closed-form benchmark for this geometry) ---
assert -0.03 <= th.min() and th.max() <= 1.03, "temperature must stay within the cylinder bounds [0,1]"
assert np.abs(vel).max() > 3.0, "buoyant convection must develop"
assert abs(rad_net) / (ar * np.abs(qv_el)).sum() < 3e-2, "grey enclosure must conserve radiative energy (sum A*q = 0)"
assert rad_hot > 0 and rad_cold < 0, "radiation sign: the hot cylinder emits, the cold one absorbs"
assert robin_loss > 0, "Robin walls must lose heat to the cooler outside"
assert plume_above > plume_below + 0.05, "the hot cylinder must drive a rising (warm) plume"

# --- animate the box warming from cold: computed temperature + computed velocity arrows -> GIF ---
TfR = frames[:, o2:]
vR = frames[:, off[0] : off[1]].reshape(frames.shape[0], -1, 2)
step_q = max(1, len(pts_v) // 150)
qscale = max(float(np.abs(vR[-1]).max()), 1.0) / 0.08


def _draw_cylinders(ax):
    ax.add_patch(plt.Circle(hot_c, rc, facecolor="#b30000", edgecolor="k", lw=0.8, zorder=5))  # hot (theta=1)
    ax.add_patch(plt.Circle(cold_c, rc, facecolor="#08306b", edgecolor="k", lw=0.8, zorder=5))  # cold (theta=0)


fig, ax = plt.subplots(figsize=(6.6, 5.4))
tpc = ax.tripcolor(triT, TfR[0], cmap="RdBu_r", shading="gouraud", vmin=0.0, vmax=1.0)
qv = ax.quiver(
    pts_v[::step_q, 0],
    pts_v[::step_q, 1],
    vR[0, ::step_q, 0],
    vR[0, ::step_q, 1],
    color="k",
    scale_units="xy",
    scale=qscale,
    width=0.003,
)
_draw_cylinders(ax)
fig.colorbar(tpc, ax=ax, shrink=0.9, label="temperature $\\theta$")
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])


def _frame(j):
    tpc.set_array(TfR[j])
    qv.set_UVC(vR[j, ::step_q, 0], vR[j, ::step_q, 1])
    ax.set_title(
        f"Heated box — conduction + convection + radiation\nhot & cold cylinders, leaky walls · frame {j}/{len(TfR) - 1}",
        fontsize=10,
    )
    return tpc, qv


ani = animation.FuncAnimation(fig, _frame, frames=len(TfR), interval=120, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "oven_convection_radiation_2d.gif", writer="pillow", fps=8, dpi=86)

# --- static figure: the developing field + the three-mode heat budget at the surfaces ---
fig2, (axf, axb) = plt.subplots(1, 2, figsize=(11.5, 4.8))
tp2 = axf.tripcolor(triT, TfR[-1], cmap="RdBu_r", shading="gouraud", vmin=0.0, vmax=1.0)
axf.quiver(
    pts_v[::step_q, 0],
    pts_v[::step_q, 1],
    vR[-1, ::step_q, 0],
    vR[-1, ::step_q, 1],
    color="k",
    scale_units="xy",
    scale=qscale,
    width=0.003,
)
_draw_cylinders(axf)
axf.set_xlim(0, Lx)
axf.set_ylim(0, Ly)
axf.set_aspect("equal")
axf.set_xticks([])
axf.set_yticks([])
axf.set_title("developing temperature + convection plumes", fontsize=11)
fig2.colorbar(tp2, ax=axf, shrink=0.9, label="temperature $\\theta$")

# two distinct budgets: a CLOSED radiative surface exchange (passes through the transparent air, sums to
# ~0) on the left, and the actual conductive heat LOSS through the Robin walls on the right -- kept apart
# so the bars are not read as one summable column.
rad_walls = N_rc * float((ar[out_m] * qv_el[out_m]).sum())
xs_pos = [0, 1, 2, 3.6]
vals = [rad_hot, rad_cold, rad_walls, -robin_loss]
labels = ["hot cyl.", "cold cyl.", "walls", "walls"]
colors = ["#b30000", "#08306b", "#777777", "#D55E00"]
axb.bar(xs_pos, vals, color=colors, width=0.8)
axb.axhline(0, color="k", lw=0.8)
axb.axvline(2.8, color="0.6", lw=0.8, ls=":")  # separates the two (incommensurable) budgets
ytop = 1.15 * max(abs(min(vals)), abs(max(vals)))
axb.set_ylim(-ytop, ytop)
axb.text(
    1.0, 0.93 * ytop, "radiative exchange\n(emit + / absorb −, Σ≈0)", ha="center", va="top", fontsize=8, style="italic"
)
axb.text(3.6, 0.93 * ytop, "Robin\nloss", ha="center", va="top", fontsize=8, style="italic")
axb.set_xticks(xs_pos)
axb.set_xticklabels(labels, fontsize=8)
axb.set_ylabel("surface heat flow (non-dim)")
axb.set_title("radiation redistributes heat among surfaces (Σ≈0);\nthe walls then lose it to the outside", fontsize=10)
fig2.suptitle("Heated box: conduction + convection + grey-body radiation + Robin walls", fontsize=12)
fig2.tight_layout(rect=(0, 0, 1, 0.94))
fig2.savefig(Path(__file__).parents[2] / "assets" / "oven_convection_radiation_2d.png", dpi=130, bbox_inches="tight")
print("\nsaved assets/oven_convection_radiation_2d.gif and .png")
