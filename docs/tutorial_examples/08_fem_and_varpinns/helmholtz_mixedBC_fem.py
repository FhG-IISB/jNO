"""04 - 2D Helmholtz with mixed Dirichlet / Neumann boundary conditions via ``jno.fem``.

    -Delta u - k^2 u = f      on the unit square                       (k = 4)
    u = 0          on the left        u = sin(pi x) on the bottom       (Dirichlet)
    du/dn given    on the right & top                                   (Neumann)

The ``-k^2 u`` reaction term makes the operator indefinite (a genuine Helmholtz term, not a
positive screened one); with k = 4 it is still below the first Dirichlet eigenvalue ~2 pi^2,
so the system stays solvable. Manufactured  u*(x, y) = sin(pi x) (cos(pi y) + y).
"""

import jax.numpy as jnp
import numpy as np
from shapely.geometry import box

import jno

pi, k = np.pi, 4.0
sin, cos = jno.np.sin, jno.np.cos
exact = lambda x, y: jnp.sin(jnp.pi * x) * (jnp.cos(jnp.pi * y) + y)  # noqa: E731
flux_right = lambda x, y: -pi * (cos(pi * y) + y)  # u_x at x = 1  # noqa: E731
flux_top = lambda x, y: sin(pi * x)  # u_y at y = 1  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.22)
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)
xbo, ybo, _ = d.variable("bottom", split=True)
xr, yr, _ = d.variable("right", split=True)
xt, yt, _ = d.variable("top", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

f = pi**2 * sin(pi * xi) * (2.0 * cos(pi * yi) + yi) - k**2 * sin(pi * xi) * (cos(pi * yi) + yi)
volume = ui.x * vi.x + ui.y * vi.y - k**2 * u * vi - f * vi
neumann_right = -flux_right(xr, yr) * phi.bind(x=xr, y=yr)  # natural term -g phi on the right
neumann_top = -flux_top(xt, yt) * phi.bind(x=xt, y=yt)
fem = jno.fem([volume, neumann_right, neumann_top, u(xl, yl) - 0.0, u(xbo, ybo) - sin(pi * xbo)], quad_degree=3)

u_fem = jnp.asarray(fem.solve(linear=jno.solve.lu()))  # indefinite Helmholtz -> sparse-direct slot
pts = np.asarray(fem.points)
rel_l2 = float(jnp.linalg.norm(exact(pts[:, 0], pts[:, 1]) - u_fem) / jnp.linalg.norm(exact(pts[:, 0], pts[:, 1])))
print(f"\n2D Helmholtz, mixed Dirichlet/Neumann via jno.fem: dofs={fem.dofs}  rel_L2={rel_l2:.3e}")
assert fem.is_linear and rel_l2 < 5e-2
