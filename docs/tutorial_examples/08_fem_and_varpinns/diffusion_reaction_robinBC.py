"""03 - 2D reaction-diffusion with mixed Dirichlet / Robin boundary conditions via ``jno.fem``.

    -Delta u + sigma u = f        on the unit square           (sigma = 4)
    u = y        on the left      u = 0 on the bottom          (Dirichlet, one non-homogeneous)
    du/dn + a u = g  on the right & top                        (Robin / natural)

A Robin condition ``du/dn + a u = g`` enters the weak form as the natural surface term
``(a u - g) phi`` on that edge. Each term carries the boundary it is bound to, so ``jno.fem``
assembles it there. Manufactured  u*(x, y) = x sin(pi y) + y.
"""

import jax.numpy as jnp
import numpy as np
from shapely.geometry import box

import jno

pi, sigma, a_r, a_t = np.pi, 4.0, 2.0, 3.0
sin = jno.np.sin
exact = lambda x, y: x * jnp.sin(jnp.pi * y) + y  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.22)
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)
xbo, ybo, _ = d.variable("bottom", split=True)
xr, yr, _ = d.variable("right", split=True)
xt, yt, _ = d.variable("top", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

f = xi * pi**2 * sin(pi * yi) + sigma * (xi * sin(pi * yi) + yi)  # -Delta u* + sigma u*
volume = ui.x * vi.x + ui.y * vi.y + sigma * u * vi - f * vi
robin_right = (a_r * u.bind(x=xr, y=yr) - (sin(pi * yr) + a_r * (sin(pi * yr) + yr))) * phi.bind(x=xr, y=yr)
robin_top = (a_t * u.bind(x=xt, y=yt) - (1.0 - pi * xt + a_t)) * phi.bind(x=xt, y=yt)
fem = jno.fem([volume, robin_right, robin_top, u(xl, yl) - yl, u(xbo, ybo) - 0.0], quad_degree=3)

u_fem = jnp.asarray(fem.solve())  # default: Jacobi-preconditioned BiCGStab on the sparse operator
pts = np.asarray(fem.points)
rel_l2 = float(jnp.linalg.norm(exact(pts[:, 0], pts[:, 1]) - u_fem) / jnp.linalg.norm(exact(pts[:, 0], pts[:, 1])))
print(f"\nReaction-diffusion, mixed Dirichlet/Robin via jno.fem: dofs={fem.dofs}  rel_L2={rel_l2:.3e}")
assert fem.is_linear and rel_l2 < 5e-2
