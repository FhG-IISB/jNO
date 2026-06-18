"""02 - Stationary Allen-Cahn via the nonlinear ``jno.fem`` residual route.

    -eps^2 Delta u + (u^3 - u) = 0,    u = tanh((x - 0.5) / (sqrt(2) eps)) on the left/right walls.

The cubic reaction makes the weak form nonlinear in ``u``, so ``jno.fem`` returns a residual
operator (``fem.residual(u)`` / ``fem.jacobian(u)``) instead of a linear ``A, b``. Newton starts
from a smooth (over-wide) interface and sharpens it to the analytic Allen-Cahn phase interface
(Allen & Cahn, *Acta Metall.* 1979).
"""

import jax.numpy as jnp
import numpy as np
import scipy.optimize as spo
from shapely.geometry import box

import jno

eps = 0.15
exact = lambda x: np.tanh((x - 0.5) / (np.sqrt(2.0) * eps))  # noqa: E731
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)
xr, yr, _ = d.variable("right", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

# eps^2 grad(u).grad(phi) + (u^3 - u) phi = 0 (cubic -> nonlinear -> residual operator)
fem = jno.fem(
    [eps**2 * (ui.x * vi.x + ui.y * vi.y) + (u**3 - u) * vi, u(xl, yl) - exact(0.0), u(xr, yr) - exact(1.0)], quad_degree=3
)
assert not fem.is_linear

pts = np.asarray(fem.points)
u0 = np.tanh((pts[:, 0] - 0.5) / (np.sqrt(2.0) * 0.30))  # over-wide interface = the Newton start
sol = spo.root(
    lambda v: np.asarray(fem.residual(jnp.asarray(v))),
    u0,
    jac=lambda v: np.asarray(dense(fem.jacobian(jnp.asarray(v)))),
    method="hybr",
)
rel_l2 = float(jnp.linalg.norm(exact(pts[:, 0]) - sol.x) / jnp.linalg.norm(exact(pts[:, 0])))

print(f"\nAllen-Cahn (nonlinear residual route): dofs={fem.dofs}  Newton converged={sol.success}  rel_L2={rel_l2:.3e}")
assert sol.success and rel_l2 < 1e-2
