# --8<-- [start:code]
"""**DFG benchmark 2D-2** -- unsteady vortex shedding past a cylinder at Re = 100, on stabilised
equal-order P1/P1 marched with BDF2, checked against the published Strouhal number.

    u_t + (u.grad)u - nu lap u + grad p = 0,   div u = 0,   Re = U_mean D / nu = 100

Configuration is Schäfer & Turek (1996), benchmark 2D-2 -- the same channel and cylinder as the
steady 2D-1 case, driven three times harder so the wake goes unstable and sheds a von Karman street:

    domain     [0, 2.2] x [0, 0.41] minus a disk of radius 0.05 at (0.2, 0.2)
    inflow     u = (4 U y (0.41 - y) / 0.41^2, 0),  U = 1.5  ->  U_mean = 1.0
    Strouhal   St = f D / U_mean,  reference ~ 0.30

This run exercises three things from the stabilised-flow work at once: the vector Laplacian in the
momentum strong residual, `dom.cell_metric` in `tau`, and `jno.solve.bdf2()` for the march.

**Why the Strouhal number and not the drag.** The 2D-1 tutorial reads forces as the reaction conjugate
to the cylinder's no-slip constraint -- `fem.eval` assembles the momentum residual with no essential
elimination and the sum over the constrained DOFs is the force. That readout is **steady-only**: the
transient path publishes no free (pre-Dirichlet) residual, so it is not available here. The Strouhal
number needs no forces at all -- it is the frequency of the transverse velocity at a fixed point in
the wake, which is read straight out of the trajectory.

**One combination that is refused, and correctly.** A fully consistent transient `tau` puts `u_t` in
the strong residual. That makes the stabilisation a *state-dependent mass* `c(u)*u_t`, which
`jno.solve.bdf2()` refuses by name -- its mass action is assembled against one previous state, and
there is nowhere for BDF2's second level to enter. The residual below is therefore quasi-static
(no `u_t`), which is a real approximation and is named as one.
"""

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace, lap = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian

L, H = 2.2, 0.41
CX, CY, RR = 0.2, 0.2, 0.05
UMAX, NU = 1.5, 1e-3
UMEAN, DIA = 2.0 / 3.0 * UMAX, 2.0 * RR
EPS, C_I = 1e-9, 36.0
MS, T_END, NSTEPS = 0.022, 6.0, 1200  # ~90 s; the refinement table is in the tutorial page
ST_REF = 0.30  # Schäfer & Turek (1996), benchmark 2D-2

shape = jno.Shape.rect(0, 0, L, H, size=MS) - jno.Shape.disk(CX, CY, RR, size=MS / 5)
d = shape.domain(time=(0.0, T_END, NSTEPS + 1))
d.tag("inlet", lambda x, y: x < EPS)
d.tag("walls", lambda x, y: (y < EPS) | (y > H - EPS))
d.tag("cyl", lambda x, y: (x - CX) ** 2 + (y - CY) ** 2 < (RR + 1e-4) ** 2)

u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)  # equal order:
p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 velocity, P1 pressure
xi, yi, ti = d.variable("interior", split=True)
xin, yin, _ = d.variable("inlet", split=True)
xw, yw, _ = d.variable("walls", split=True)
xc, yc, _ = d.variable("cyl", split=True)
ci = d.variable("initial", split=True)

ub, vv = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
gp, gq = grad(p, [xi, yi]), grad(q, [xi, yi])
pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)

div = lambda gw: trace(gw)  # noqa: E731
adv = lambda gw, w: inner(gw, w, n_contract=1)  # noqa: E731

momentum = inner(ub.t, vv, 1) + inner(adv(gu, ub), vv, 1) + NU * inner(gu, gv, 2) - pp * div(gv)
continuity = -qq * div(gu)

dt = T_END / NSTEPS
G = d.cell_metric
gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
# The transient tau carries the step size. It assumes the FIXED grid from `domain(time=...)` and is
# wrong under `jno.solve.adaptive()`, where dt is chosen per step.
tau = jno.lag(((2.0 / dt) ** 2 + gG(ub) + C_I * NU**2 * inner(G, G, n_contract=2)) ** -0.5)
r_m = adv(gu, ub) - NU * lap(u, [xi, yi]) + gp  # quasi-static: no u_t (see the docstring)
momentum = momentum + tau * inner(adv(gv, ub), r_m, n_contract=1)  # SUPG -> momentum   (+)
continuity = continuity - tau * inner(gq, r_m, n_contract=1)  # PSPG -> continuity (-)

profile = 4.0 * UMAX * yin * (H - yin) / H**2
fem = jno.fem(
    [
        momentum,
        continuity,
        u(xin, yin)[0] - profile,
        u(xin, yin)[1] - 0.0,
        u(xw, yw)[0] - 0.0,
        u(xw, yw)[1] - 0.0,
        u(xc, yc)[0] - 0.0,
        u(xc, yc)[1] - 0.0,
        u(*ci)[0] - 0.0,
        u(*ci)[1] - 0.0,
    ]  # the outlet gets nothing: natural (do-nothing), which also sets the pressure level
)
assert fem.is_transient and not fem.is_linear
print(f"\nDFG 2D-2, vortex shedding at Re = {UMEAN * DIA / NU:.0f}: dofs={fem.dofs}, dt={dt:g}, T={T_END:g}")

t0 = time.time()
traj = np.asarray(
    fem.solve(
        nonlinear=jno.solve.newton(direct=True, rtol=1e-7, atol=1e-7),
        linear=jno.solve.lu(backend="host"),  # 3.8x faster here than the device factorisation
        time=jno.solve.bdf2(),  # second order AND L-stable
    ).fn()
)
print(f"  marched {traj.shape[0]} steps in {time.time() - t0:.0f} s ({1e3 * (time.time() - t0) / NSTEPS:.0f} ms/step)")

# ---- the Strouhal number, from the transverse velocity at a fixed wake probe --------------------
pts = np.asarray(fem.points)
probe = np.array([CX + 4 * RR, CY + RR])  # in the wake, off the centreline so the mode is visible
k = int(np.argmin(np.sum((pts - probe) ** 2, axis=1)))
ts = np.linspace(0.0, T_END, NSTEPS + 1)
uy = np.array([s[: fem.offsets[1]].reshape(len(pts), 2)[k, 1] for s in traj])

half = len(uy) // 2  # discard the start-up transient
tail, t_tail = uy[half:], ts[half:]
crossings = np.where(np.diff(np.sign(tail - tail.mean())))[0]
assert len(crossings) >= 4, f"the wake is not shedding ({len(crossings)} crossings) -- run longer"
period = 2.0 * np.mean(np.diff(t_tail[crossings]))
st = DIA / (UMEAN * period)

print(f"  probe at ({pts[k][0]:.3f}, {pts[k][1]:.3f}), u_y swing {tail.min():+.4f} .. {tail.max():+.4f}")
print(f"  shedding period {period:.4f} s  ->  f = {1 / period:.4f} Hz")
print(f"  St = {st:.4f}   reference ~ {ST_REF:.2f}   ({100 * (st - ST_REF) / ST_REF:+.1f}%)")

assert tail.max() - tail.min() > 0.1, "no sustained oscillation -- the wake did not go unstable"
assert 0.24 < st < 0.31, f"St = {st:.4f} is outside the band this discretisation converges through"
print("\nA von Karman street, at the right frequency to within the resolution of this run.")
# --8<-- [end:code]

# ---- figure: vorticity through one shedding cycle -----------------------------------------------
# w = dv/dx - du/dy is computed from the P1 element gradients -- the exact derivative of the velocity
# this run solved for, per cell, with nothing smoothed or interpolated in.
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

cells = np.asarray(d._cells_p1())
tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells)
P = pts[cells]
X, Y = P[..., 0], P[..., 1]
bg = np.stack([Y[:, 1] - Y[:, 2], Y[:, 2] - Y[:, 0], Y[:, 0] - Y[:, 1]], axis=1)
cg = np.stack([X[:, 2] - X[:, 1], X[:, 0] - X[:, 2], X[:, 1] - X[:, 0]], axis=1)
A2 = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
U = np.stack([s[: fem.offsets[1]].reshape(len(pts), 2) for s in traj])
vort = lambda k: np.sum(bg * U[k][cells, 1], 1) / A2 - np.sum(cg * U[k][cells, 0], 1) / A2  # noqa: E731

per = int(round(period / dt))
frames = np.arange(len(ts) - per, len(ts), max(1, per // 30))  # one cycle
w = np.stack([vort(k) for k in frames])
# Scale to the WAKE: the attached shear layer reaches |w| ~ 380 while the shed vortices peak near 50,
# so a full-range scale renders the street invisible. The shear layer saturates -- said in the caption.
cxc = P[..., 0].mean(1)
vmax = float(np.ceil(np.abs(w[:, cxc > 0.35]).max() / 10.0) * 10.0)

fig, ax = plt.subplots(figsize=(7.2, 1.55), dpi=110)
tpc = ax.tripcolor(tri, facecolors=w[0], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
ax.set_axis_off()
ax.set_xlim(0, 1.4)  # near field: the street lives here
ax.set_ylim(0, H)
ax.set_aspect("equal")
cax = make_axes_locatable(ax).append_axes("right", size="3.5%", pad=0.05)
cb = plt.colorbar(tpc, cax=cax)
cb.outline.set_visible(False)
cb.minorticks_off()
cb.ax.tick_params(length=0)
cb.set_ticks([-vmax, 0, vmax])
assets = Path(__file__).parents[2] / "assets"
FuncAnimation(fig, lambda i: (tpc.set_array(w[i]),), frames=len(frames)).save(
    assets / "vortex_shedding_dfg_2d2.gif", writer=PillowWriter(fps=12)
)
tpc.set_array(w[len(frames) // 4])
fig.savefig(assets / "vortex_shedding_dfg_2d2.png", dpi=135)
