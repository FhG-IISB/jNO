"""07 - Stokes channel (Poiseuille) flow via ``jno.fem`` (coupled velocity-pressure).

Steady incompressible Stokes flow in a channel:

    -mu Delta u + grad p = 0,    div u = 0      on [0, Lx] x [0, H]

solved with an inf-sup stable Taylor-Hood pair -- P2 velocity (``order=2``) and P1 pressure
(``order=1``) -- two coupled ``fem_symbols`` fields. Fully-developed (Poiseuille) flow has the
exact parabolic profile

    u_x = (G / 2 mu) y (H - y),    u_y = 0,    p = -G x,

so we impose that profile as the velocity boundary data (zero on the walls, parabola at the
inlet/outlet), pin the pressure at one vertex to remove its constant null space, and recover
the analytic Poiseuille solution to machine precision.
"""

import jax.numpy as jnp
import numpy as np
from shapely.geometry import box

import jno

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731
G, mu, H, Lx = 1.0, 1.0, 1.0, 4.0
u_profile = lambda y: (G / (2.0 * mu)) * y * (H - y)  # noqa: E731

d = jno.domain(box(0.0, 0.0, Lx, H), mesh_size=0.2)
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)

# momentum (viscous - pressure*div(v)) ; continuity (-q div(u)) ; velocity BCs ; pressure pin
fem = jno.fem(
    [
        mu * inner(gu, gv, n_contract=2) - pp * trace(gv),
        -qq * trace(gu),
        u(xb, yb)[0] - u_profile(yb),
        u(xb, yb)[1] - 0.0,
        p.pin(),  # gauge-fix: remove the pressure null space
    ]
)

off = fem.problem.offset  # per-field DOF offsets in the coupled solution vector
sol = jnp.linalg.solve(dense(fem.A), jnp.asarray(fem.b).reshape(-1))
uu = np.asarray(sol[off[0] : off[1]]).reshape(-1, 2)  # velocity (n_vel_nodes, 2)
pts_v = np.asarray(fem.problem.mesh[0].points)
rel_ux = float(np.linalg.norm(uu[:, 0] - u_profile(pts_v[:, 1])) / np.linalg.norm(u_profile(pts_v[:, 1])))
max_uy = float(np.max(np.abs(uu[:, 1])))  # u_y == 0 and u_x x-independent => div u == 0
print(f"\nStokes Poiseuille channel (Taylor-Hood P2/P1): dofs={fem.dofs}  u_x rel_L2={rel_ux:.3e}  max|u_y|={max_uy:.3e}")
assert rel_ux < 1e-8 and max_uy < 1e-8
