"""Transient incompressible **Navier-Stokes** in 2D: the lid-driven cavity, the canonical viscous-flow
benchmark. The fluid starts at rest; the top lid is set impulsively in motion and drives a
recirculating vortex that spins up and settles to steady state.

    u_t + (u.grad)u - nu lap u + grad p = 0,    div u = 0,    Re = U L / nu = 200.

The point of this example is the **convective term** ``(u.grad)u`` -- written as ``inner(grad u, u)``.
That is the unknown contracted with itself (a genuine nonlinearity), so ``jno.fem`` routes the whole
system to its nonlinear coupled operator and the Jacobian comes from autodiff. ``fem.solve()`` does
the implicit time stepping internally -- **backward Euler + Newton** per step on the mass / residual /
jacobian of the transient system -- and returns the differentiable forward trajectory.

Inf-sup-stable Taylor-Hood elements (P2 velocity, P1 pressure); all-Dirichlet (a regularised moving
lid + no-slip walls), so no outflow boundary is needed.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")  # play nice on a shared GPU

import jax

jax.config.update("jax_enable_x64", True)  # the assembler builds in float64

from pathlib import Path  # noqa: E402

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

import jno  # noqa: E402

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
nu = 0.005  # Re = U L / nu = 1 * 1 / 0.005 = 200

d = jno.domain(jno.Shape.rect(0, 0, 1, 1, size=0.045), time=(0.0, 8.0, 33))
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ci = d.variable("initial", split=True)
ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
pp, qq = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
conv = inner(gu, ub, n_contract=1)  # (u.grad)u  -- the convective nonlinearity
lid = 16.0 * xb**2 * (1 - xb) ** 2  # regularised lid (peak 1 at x=0.5, 0 at the corners)
momentum = inner(ub.t, vb, n_contract=1) + inner(conv, vb, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * trace(gv)
fem = jno.fem(
    [
        momentum,
        -qq * trace(gu),
        u(xb, yb)[0] - jno.np.where(yb > 1 - 1e-6, lid, 0.0),  # moving lid on top, no-slip elsewhere
        u(xb, yb)[1] - 0.0,
        p.pin(),  # gauge-fix: remove the pressure null space (any single DOF)
        u(ci[0], ci[1])[0] - 0.0,  # start from rest
        u(ci[0], ci[1])[1] - 0.0,
    ]
)
assert fem.is_transient and not fem.is_linear, "transient Navier-Stokes must be nonlinear"
off = fem.offsets
print(f"\nTransient Navier-Stokes lid-driven cavity (Re={1 / nu:.0f}): dofs={fem.dofs}")

# fem.solve() does the implicit time stepping internally (backward Euler + Newton per step) and
# returns a differentiable trace node; evaluate the forward trajectory through a minimal crux.
sol = fem.solve()
traj = np.asarray(jno.core([sol.mse]).eval([sol]))  # (n_steps, dofs) over the domain time grid
ts = np.linspace(float(fem.t0), float(fem.t1), traj.shape[0])  # sample time of each trajectory row
settle = float(np.linalg.norm(traj[-1] - traj[-2]) / np.linalg.norm(traj[-1]))

pts_v = np.asarray(fem.field_points[0])
vel = traj[:, off[0] : off[1]].reshape(traj.shape[0], -1, 2)  # (frame, n_vel_nodes, 2)
uxN = vel[-1, :, 0]  # steady x-velocity, for the recirculation check
cl = np.abs(pts_v[:, 0] - 0.5) < 0.06  # near the vertical centre-line
top = uxN[cl & (pts_v[:, 1] > 0.7)].mean()  # driven by the lid -> u_x > 0
bot = uxN[cl & (pts_v[:, 1] < 0.3)].mean()  # return flow -> u_x < 0
print(f"  steady by final frame: {settle:.3e}  |  centre-line u_x  top={top:+.3f}  bottom={bot:+.3f} (recirculation)")

# ---- animate the spinning-up vortex: streamlines coloured by speed -> a GIF ----
gx, gy = np.meshgrid(np.linspace(0, 1, 90), np.linspace(0, 1, 90))
fig, ax = plt.subplots(figsize=(5.6, 5.4))


def _frame(j):
    ax.clear()
    UX = griddata(pts_v, vel[j, :, 0], (gx, gy), method="linear")
    UY = griddata(pts_v, vel[j, :, 1], (gx, gy), method="linear")
    ax.streamplot(gx, gy, UX, UY, color=np.hypot(UX, UY), cmap="viridis", density=1.4, linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"lid-driven cavity, Re=200 — t = {ts[j]:.1f}", fontsize=11)


ani = animation.FuncAnimation(fig, _frame, frames=vel.shape[0], interval=110, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "navier_stokes_cavity_2d.gif", writer="pillow", fps=9, dpi=85)

assert settle < 2e-2, f"flow not at steady state by the final frame: {settle:.3e}"
assert top > 0.1 and bot < -0.02, f"expected a recirculating vortex (u_x top>0, bottom<0): top={top:.3f} bot={bot:.3f}"
