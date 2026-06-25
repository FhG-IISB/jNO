"""12 - Maxwell via Nédélec H(curl) edge elements (``jno.fem``, ``space="N1E"``).

The electromagnetic field lives in **H(curl)**: its *tangential* trace is continuous across material
interfaces (the normal component may jump). Nédélec first-kind **edge** elements are H(curl)-conforming
-- one tangential-moment DOF per mesh edge -- and are the textbook discretization for Maxwell. The key
contrast with the *nodal* (Lagrange) curl-curl in ``maxwell_2d_vector.py``: edge elements need **no
grad-div penalty**. The discrete gradient fields lie exactly in the curl kernel, so the spurious modes
that force a penalty on nodal elements simply do not appear -- you write the curl-curl form and solve.

jNO assembles N1E with its native push-forward engine; the 2-D curl is the
view sugar ``u.curl(x, y)`` and the essential BC is the tangential trace ``u×n`` -- the perfect-electric
-conductor (PEC) wall ``u×n = 0``, written through the outward normal as ``u[0]·ny - u[1]·nx``.

Example 1 -- magnetostatic / coercive curl-curl:  ``curl curl u + u = f`` on ``[0,1]^2``, PEC ``u×n=0``.
Manufactured ``u = (sin πy, sin πx)`` satisfies ``u×n = 0`` on ∂Ω and is a curl-curl eigenmode
(``curl curl u = π² u``), so ``f = (π²+1) u``. Lowest-order N1E converges at ``O(h)`` -- no penalty.

Example 2 -- eddy current (transient Maxwell):  ``∂ₜu + curl curl u + u = f``, same ``f``, PEC, ``u(0)=0``.
The field diffuses to the magnetostatic steady state; stepped with backward Euler from the block's own
``M`` / ``A`` / ``affine_bias`` / ``state0`` / ``dt``, ``u(T)`` relaxes onto the Example-1 solution.

Reference: J.-C. Nédélec, *Mixed finite elements in* R^3, Numer. Math. 35 (1980) 315-341.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless figure rendering

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # the edge-element systems are assembled/solved in float64

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_nonnodal import n1e_field_at_centroids  # noqa: E402
from jno.utils.solver.fem_topology import build_edge_topology  # noqa: E402

inner, sin = jno.np.inner, jno.np.sin
dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731
PI = float(np.pi)
FAC = PI**2 + 1.0  # f = (π²+1) u for the manufactured curl-curl eigenmode


def _centroids(d):
    pts, cells = np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])
    return pts, cells, pts[cells].mean(1)


def _u_exact(c):  # u = (sin πy, sin πx) at points c -- divergence-free, with u×n = 0 on the unit square
    return np.stack([np.sin(PI * c[:, 1]), np.sin(PI * c[:, 0])], axis=-1)


# ---------- Example 1: magnetostatic / coercive Maxwell  curl curl u + u = f,  PEC u×n = 0 ----------
d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")  # H(curl) edge field
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
ui, vi, ub = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), u.bind(x=xb, y=yb)
fx, fy = FAC * sin(PI * yi), FAC * sin(PI * xi)
# ∫ u·v + ∫ curl u · curl v - ∫ f·v = 0   (the +u keeps it coercive -- pure curl-curl is singular on H(curl));
# PEC tangential trace u×n = 0 on the wall (an essential BC, pins the boundary-edge DOFs).
fem = jno.fem([inner(ui, vi) + ui.curl() * vi.curl() - (fx * vi[0] + fy * vi[1]), ub[0] * ny - ub[1] * nx - 0.0])
sol = np.linalg.solve(dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))

pts, cells, cent = _centroids(d)
u_steady = np.asarray(n1e_field_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(sol)))
err1 = float(np.sqrt(np.mean(np.sum((u_steady - _u_exact(cent)) ** 2, axis=1))))
print(f"Example 1 (curl-curl + mass, PEC u×n=0): dofs={fem.dofs}  centroid L2 err = {err1:.3e}")
assert err1 < 0.07  # lowest-order N1E at h=0.1 -- and no grad-div penalty was needed

# ---------- Example 2: eddy current (transient Maxwell)  ∂ₜu + curl curl u + u = f, u(0)=0 -> steady ----------
dt_dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12, time=(0.0, 0.5, 26))
co = dt_dom.variable("interior", split=True)  # x, y, t
ci = dt_dom.variable("initial", split=True)
xb2, yb2, _, nx2, ny2 = dt_dom.variable("boundary", normals=True, split=True)
u2, v2 = dt_dom.fem_symbols(value_shape=(2,), names=("u", "v"), space="N1E")
u2i, v2i, u2b = u2.bind(x=co[0], y=co[1], t=co[2]), v2.bind(x=co[0], y=co[1], t=co[2]), u2.bind(x=xb2, y=yb2)
fx2, fy2 = FAC * sin(PI * co[1]), FAC * sin(PI * co[0])
ic = u2(ci[0], ci[1]) - jno.np.vector(0.0 * ci[0], 0.0 * ci[1])  # u(0) = 0  (projected onto the edge DOFs)
# ∫ ∂ₜu·v  is the mass term (inner(u2i.t, v2i)); the rest is the spatial operator + forcing + PEC wall + IC.
fem2 = jno.fem(
    [
        inner(u2i.t, v2i) + inner(u2i, v2i) + u2i.curl() * v2i.curl() - (fx2 * v2i[0] + fy2 * v2i[1]),
        u2b[0] * ny2 - u2b[1] * nx2 - 0.0,
        ic,
    ]
)
# Backward Euler from the block's own flat pieces (M u̇ + A u = c): (M + dt A) w_next = M w + dt c.
M, A = dense(fem2.M), dense(fem2.operator.A)
c = np.asarray(jnp.asarray(fem2.operator.affine_bias)).reshape(-1)
w, step = np.asarray(fem2.state0), float(fem2.dt)
times, norms = [float(fem2.t0)], [float(np.linalg.norm(w))]  # record the relaxation ‖u(t)‖
for k in range(round((fem2.t1 - fem2.t0) / step)):
    w = np.linalg.solve(M + step * A, M @ w + step * c)
    times.append(float(fem2.t0) + (k + 1) * step)
    norms.append(float(np.linalg.norm(w)))

pts2, cells2, cent2 = _centroids(dt_dom)
uT = np.asarray(n1e_field_at_centroids(pts2, cells2, build_edge_topology(cells2), jnp.asarray(w)))
err2 = float(np.sqrt(np.mean(np.sum((uT - _u_exact(cent2)) ** 2, axis=1))))
print(f"Example 2 (eddy current, u(0)=0 -> magnetostatic steady): u(T) L2 err = {err2:.3e}")
assert float(np.linalg.norm(np.asarray(fem2.state0))) < 1e-10  # the zero initial field
assert err2 < 0.08  # the transient relaxes onto the steady (magnetostatic) solution

# ---------- Figure (renders the COMPUTED fields -- u_h and ‖u(t)‖, not the exact solution) ----------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.4))
mag = np.linalg.norm(u_steady, axis=1)
axL.quiver(cent[:, 0], cent[:, 1], u_steady[:, 0], u_steady[:, 1], mag, cmap="viridis", scale=22, width=0.004)
axL.set_title(f"Magnetostatic curl-curl: computed H(curl) field $u_h$\n(N1E, {fem.dofs} edge DOFs, L2 err {err1:.1e})")
axL.set_xlabel("x")
axL.set_ylabel("y")
axL.set_aspect("equal")
axR.plot(times, norms, "-o", ms=3, color="crimson")
axR.set_title("Eddy current: $\\|u(t)\\|$ relaxes from 0\nto the magnetostatic steady state")
axR.set_xlabel("t")
axR.set_ylabel("$\\|u(t)\\|$")
axR.grid(alpha=0.3)
fig.tight_layout()
_out = Path(__file__).parents[2] / "assets" / "maxwell_nedelec_2d.png"
fig.savefig(_out, dpi=130, bbox_inches="tight")
print(f"saved figure -> docs/assets/{_out.name}")
