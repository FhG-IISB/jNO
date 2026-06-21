"""Bring-your-own time integrator: a transient diffusion on a complex domain stepped with **diffrax**
instead of jno's built-in solver. A heat spreader -- a plate with two insulated bores -- starts cold
and is driven by a constant heat source; with a distributed heat-loss term (cooling toward ambient)
the insulated plate heats up and settles into a steady temperature pattern that wraps around the
bores:

    u_t = nu lap u - kappa u + f(x),     insulated (Neumann) on every edge, including the bores.

``jno.fem`` hands you the semidiscrete pieces ``M u_dot + A u = c`` directly -- ``fem.M`` (dense
mass), ``fem.operator.A`` (stiffness), the forcing ``c``, and ``fem.state0`` -- so you build the
diffrax ``ODETerm`` (``u_dot = M^-1(c - A u)``) yourself and hand it to **your** solver -- an
adaptive, stiff-aware ``Kvaerno5``. No call to ``fem.solve()`` anywhere; the integrator is entirely
yours, and the same pieces drive the default backward-Euler so we can cross-check the two.

Two checks, no analytic solution needed: the diffrax trajectory **agrees with backward-Euler**, and
the field is **at steady state** by the final frame (the last snapshots stop changing).
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import jax

jax.config.update("jax_enable_x64", True)  # feax assembly is float64

from pathlib import Path  # noqa: E402

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

import jno  # noqa: E402

nu, kappa = 1.0, 16.0  # diffusivity, heat-loss rate (sets the steady scale and the time constant)
dn = lambda X: jnp.asarray(X.todense()) if hasattr(X, "todense") else jnp.asarray(X)  # noqa: E731


def diffrax_solve(M, A, c, state0, save_ts):
    """Integrate M u_dot + A u = c with diffrax -- you form u_dot = M^-1(c - A u) from the flat
    block pieces and wrap it in an ODETerm; no built-in adapter needed."""

    def rhs(t, u, _args):
        return jnp.linalg.solve(M, c - A @ u)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Kvaerno5(),  # ESDIRK: implicit/adaptive, for the stiff parabolic operator
        t0=float(save_ts[0]),
        t1=float(save_ts[-1]),
        dt0=float(save_ts[1] - save_ts[0]),
        y0=jnp.asarray(state0),
        saveat=diffrax.SaveAt(ts=save_ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        max_steps=100_000,
    )
    return sol.ys  # (len(save_ts), n_dofs)


# complex domain: a plate with two insulated bores
plate = box(0, 0, 2, 1).difference(Point(0.7, 0.55).buffer(0.16)).difference(Point(1.35, 0.4).buffer(0.2))
d = jno.domain(plate, mesh_size=0.045, time=(0.0, 0.3, 61))  # dt = 0.005 so backward-Euler is accurate too
u, phi = d.fem_symbols()
xi, yi, ti = d.variable("interior", split=True)
ci = d.variable("initial", split=True)
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
src = 26.0 * jno.np.exp(-(((xi - 0.4) ** 2 + (yi - 0.5) ** 2) / (2 * 0.12**2)))  # constant-in-time hot spot
fem = jno.fem([ui.t * vi + nu * (ui.x * vi.x + ui.y * vi.y) + kappa * (u * vi) - src * vi, u(ci[0], ci[1]) - 0.0])
assert fem.is_transient

# assemble the semidiscrete pieces once from the flat API: M u_dot + A u = c
M = fem.M  # dense mass matrix (flat accessor)
A = dn(fem.operator.A)  # stiffness (raw BCOO -> dense)
c = jnp.zeros(M.shape[0])  # constant forcing (source) lives in affine_bias and/or forcing_vector_fn
if fem.operator.affine_bias is not None:
    c = c + dn(fem.operator.affine_bias).reshape(-1)
if fem.operator.forcing_vector_fn is not None:
    c = c + jnp.asarray(fem.operator.forcing_vector_fn(0.0, {})).reshape(-1)
state0, dt = fem.state0, float(fem.dt)

save_ts = jnp.linspace(float(fem.t0), float(fem.t1), 48)  # many frames for a smooth animation
traj = np.asarray(diffrax_solve(M, A, c, state0, save_ts))  # YOUR diffrax solve, no fem.solve()

# default backward-Euler over the same M, A, c for cross-check: (M + dt A) w_next = M w + dt c
w = state0
for _ in range(round((float(fem.t1) - float(fem.t0)) / dt)):
    w = jnp.linalg.solve(M + dt * A, M @ w + dt * c)
agree = float(np.linalg.norm(traj[-1] - np.asarray(w)) / np.linalg.norm(np.asarray(w)))
settled = float(np.linalg.norm(traj[-1] - traj[-2]) / np.linalg.norm(traj[-1]))  # at steady state?
print("\nTransient heat spreader (diffrax Kvaerno5, bring-your-own integrator)")
print(f"  complex domain (2 insulated bores)  dofs={fem.dofs}  snapshots={traj.shape[0]}")
print(f"  diffrax vs default backward-Euler (final field rel-L2): {agree:.3e}")
print(f"  steady by the final frame (||u[-1]-u[-2]||/||u[-1]||): {settled:.3e}")

# ---- animate the actual computed field (no invented structure) -> a looping GIF ----
pts = np.asarray(fem.points)
tris = np.asarray(fem.domain.built_mesh.cells_dict["triangle"])
triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
vmax = float(traj[-1].max())
fig, ax = plt.subplots(figsize=(8.4, 4.4))
tpc = ax.tripcolor(triang, traj[0], cmap="inferno", shading="gouraud", vmin=0.0, vmax=vmax)
fig.colorbar(tpc, ax=ax, shrink=0.85, label="temperature")
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])


def _frame(j):
    tpc.set_array(traj[j])
    ax.set_title(f"diffrax heat spreader — t = {float(save_ts[j]):.3f}", fontsize=11)
    return (tpc,)


ani = animation.FuncAnimation(fig, _frame, frames=traj.shape[0], interval=80, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "transient_diffrax_heat_spreader.gif", writer="pillow", fps=12, dpi=90)

assert agree < 5e-2, f"diffrax and backward-Euler disagree: {agree:.3e}"
assert settled < 2e-2, f"not at steady state by the final frame: {settled:.3e}"
