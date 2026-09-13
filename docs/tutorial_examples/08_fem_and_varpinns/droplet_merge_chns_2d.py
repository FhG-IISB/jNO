# --8<-- [start:code]
"""Two droplets **merge** -- a change of topology -- in a Cahn–Hilliard–Navier–Stokes (diffuse-interface)
model, on a mesh that follows the interface. Nothing here is a two-phase feature: the model is the term list.

    rho (u_t + (u.grad)u) = -grad p + eta lap u + mu grad(phi),        div u = 0
    phi_t + u.grad(phi)   = div(M grad mu),       mu = lam (-lap phi + (phi^3 - phi) / eps^2)

phi = +1 in the liquid and -1 outside, and the interface is a tanh layer of width ~eps. Surface tension is
sigma = (2 sqrt2 / 3) lam / eps (Jacqmin 1999; Yue, Feng, Liu & Shen 2004). Because the interface is a
field rather than a boundary, two drops can touch and fuse with no remeshing surgery at all -- and the
mesh is free to follow it, refined where 1 - phi^2 is large and coarse in the bulk.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
tanh, sqrt = jno.np.tanh, jno.np.sqrt
dot = lambda a, b: inner(a, b, n_contract=1)  # noqa: E731  a·b
ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731  A:B

SIGMA, ETA, RHO = 1.0, 0.1, 1.0  # surface tension, viscosity, density
EPS, MOBILITY = 0.04, 1e-2  # interface width, Cahn–Hilliard mobility
LAM = 3.0 * SIGMA * EPS / (2.0 * np.sqrt(2.0))
R, CX1, CX2, CY = 0.18, 0.28, 0.72, 0.5  # two drops whose rims are 0.08 (~1.4 interface widths) apart
T_END, N_STEPS = 0.6, 12


def near_the_rims(x, y, z):  # gmsh calls a size field with THREE coordinates, in 2-D as well
    rim = min(abs(np.hypot(x - CX1, y - CY) - R), abs(np.hypot(x - CX2, y - CY) - R))
    return 0.02 + 0.04 * min(1.0, rim / 0.12)  # 0.02 at the interfaces, 0.06 in the bulk


d = jno.shape.rect(0, 0, 1, 1, size=near_the_rims).domain(time=(0.0, T_END, N_STEPS + 1))
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
c, s = d.fem_symbols(names=("phi", "psi"), order=1)  # P1 phase field
m, w = d.fem_symbols(names=("mu", "chi"), order=1)  # P1 chemical potential
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
x0, y0, _ = d.variable("initial", split=True)
B = dict(x=xi, y=yi, t=ti)
u_, v_, p_, q_ = u.bind(**B), v.bind(**B), p.bind(**B), q.bind(**B)
phi, psi, mu, chi = c.bind(**B), s.bind(**B), m.bind(**B), w.bind(**B)
grad_u, grad_v = grad(u, [xi, yi]), grad(v, [xi, yi])

momentum = (
    RHO * dot(u_.t, v_)
    + RHO * dot(dot(grad_u, u_), v_)
    + ETA * ddot(grad_u, grad_v)
    - p_ * trace(grad_v)
    - mu * (phi.x * v_[0] + phi.y * v_[1])  # the capillary force  mu grad(phi)
)
continuity = -q_ * trace(grad_u)
cahn_hilliard = phi.t * psi + (u_[0] * phi.x + u_[1] * phi.y) * psi + MOBILITY * (mu.x * psi.x + mu.y * psi.y)
chemical_potential = mu * chi - LAM * (phi * phi * phi - phi) / EPS**2 * chi - LAM * (phi.x * chi.x + phi.y * chi.y)


def drop(cx):
    return tanh((R - sqrt((x0 - cx) ** 2 + (y0 - CY) ** 2)) / (np.sqrt(2.0) * EPS))


fem = jno.fem(
    [
        momentum,
        continuity,
        cahn_hilliard,
        chemical_potential,
        u(xb, yb)[0] - 0.0,  # no-slip walls
        u(xb, yb)[1] - 0.0,
        p.pin(),  # the pressure is defined up to a constant
        u(x0, y0)[0] - 0.0,  # start at rest
        u(x0, y0)[1] - 0.0,
        c(x0, y0) - (drop(CX1) + drop(CX2) + 1.0),  # two drops
    ]
)
n0 = int(np.asarray(d.mesh.points).shape[0])
print(f"CHNS droplet merger: {fem.dofs} DOFs on {n0} vertices")

phi_now = c.bind(x=xi, y=yi)
traj = fem.solve(
    adapt=jno.solve.remesh(criterion=1.0 - phi_now * phi_now, every=3, max_dofs=n0),  # follow the interface
    nonlinear=jno.solve.newton(direct=True),  # a sparse-direct Newton per step: the saddle needs it
)
ic = fem.block_index(c)  # which block of the state is phi


def phase(k):
    """phi on its own nodes at frame k, and int (1 + phi)/2 -- the liquid area -- by the P1 vertex rule."""
    lay = traj.layouts[k]
    pts = np.asarray(lay["field_points"][ic])[:, :2]
    val = np.asarray(traj.states[k])[lay["offsets"][ic] : lay["offsets"][ic + 1]]
    tri = np.asarray(lay["cells_f"][ic])[:, :3]
    P = pts[tri]
    area = 0.5 * np.abs(
        (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1])
    )
    return pts, val, float(np.sum(area * (1.0 + val[tri]).mean(axis=1) / 2.0))


def gap(pts, val):  # phi midway between the two drops
    return float(val[np.argmin(np.hypot(pts[:, 0] - 0.5, pts[:, 1] - CY))])


def aspect(pts, val):  # width / height of the liquid region
    liquid = pts[val > 0.0]
    return float(np.ptp(liquid[:, 0]) / np.ptp(liquid[:, 1]))


areas = [phase(k)[2] for k in range(len(traj))]
pts0, phi0, liquid0 = phase(0)
ptsN, phiN, liquidN = phase(len(traj) - 1)
remeshes = sum(bool(h.get("remeshed")) for h in fem.adapt_history)
# Cahn–Hilliard conserves int phi EXACTLY on a fixed mesh -- so between two remeshes (frames that share one
# mesh) the liquid area is constant to solver precision, and any drift is the state transfer at a remesh.
same_mesh = [k for k in range(len(traj) - 1) if traj.meshes[k] is traj.meshes[k + 1]]
between = max(abs(areas[k + 1] - areas[k]) for k in same_mesh)
print(f"  gap phi: {gap(pts0, phi0):+.3f} -> {gap(ptsN, phiN):+.3f}   (two drops -> one)")
print(f"  width/height: {aspect(pts0, phi0):.2f} -> {aspect(ptsN, phiN):.2f}   (relaxing toward a circle)")
print(f"  liquid area: {liquid0:.5f} -> {liquidN:.5f}  ({(liquidN / liquid0 - 1) * 100:+.2f} %, over {remeshes} remeshes)")
print(f"  largest change between remeshes: {between:.1e}")

assert gap(pts0, phi0) < 0.0 < 0.9 < gap(ptsN, phiN), "the two drops did not merge"
assert aspect(ptsN, phiN) < 0.8 * aspect(pts0, phi0), "the merged drop is not relaxing toward a circle"
assert between < 1e-8, f"int phi changed by {between:.1e} on a fixed mesh -- Cahn–Hilliard conserves it exactly"
assert abs(liquidN / liquid0 - 1.0) < 2e-2, (
    "the remesh transfers lost or gained more liquid than expected"
)  # measured +0.81 %
assert remeshes >= 2, "the mesh never followed the interface"
# --8<-- [end:code]

# ---- figure (hidden from the docs): phi at three instants, each on the mesh that step ran on -> a PNG ----
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Frutiger 45 Light", "Frutiger", "FreeSans", "DejaVu Sans"],
        "font.weight": "light",
        "font.size": 11,
        "axes.titleweight": "light",
    }
)
frames = (0, len(traj) // 2, len(traj) - 1)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, k in zip(axes, frames):
    pts, val, _ = phase(k)
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], np.asarray(traj.layouts[k]["cells_f"][ic])[:, :3])
    im = ax.tripcolor(tri, val, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="gouraud")  # P1: linear per cell
    ax.triplot(tri, lw=0.15, color="#1A202C", alpha=0.35)  # the mesh this frame was computed on
    ax.set_title(f"$t = {float(traj.times[k]):.2f}$")
    ax.set_axis_off()
    ax.margins(0)
    ax.autoscale_view()
    x0_, x1_ = ax.get_xlim()
    y0_, y1_ = ax.get_ylim()
    ax.set_box_aspect(abs(y1_ - y0_) / abs(x1_ - x0_))  # box == field
# The colorbar is placed against the field AS DRAWN: a divider would size it to the subplot's grid cell,
# which is taller than the square field `set_box_aspect` draws inside it, and the bar would overshoot.
fig.canvas.draw()
pos = axes[-1].get_position()
cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.035 * pos.width, pos.height])
cbar = fig.colorbar(im, cax=cax)
cbar.outline.set_visible(False)
cbar.minorticks_off()
cbar.ax.tick_params(length=0)
cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
tls = cbar.ax.yaxis.get_ticklabels()
for tl in tls:
    tl.set_fontstyle("italic")
    tl.set_fontweight("light")
tls[0].set_verticalalignment("bottom")
tls[-1].set_verticalalignment("top")
fig.savefig(Path(__file__).parents[2] / "assets" / "droplet_merge_chns_2d.png")
