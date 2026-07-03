"""01 - 2D Poisson equation assembled + solved through ``jno.fem``.

    -Delta u = f on the unit square, u = 0 on the boundary.
    Manufactured  u*(x, y) = x(1-x) y(1-y),   f = -Delta u* = 2[x(1-x) + y(1-y)].

A pure finite-element solve: write the weak form as a list of residual terms (volume physics
+ the essential condition ``u(region) - g``), hand it to ``jno.fem`` -- the single FEM entry --
and solve with ``fem.solve()``. The default is a matrix-free Jacobi-preconditioned BiCGStab on
the sparse operator (never densifies); the solver slots choose anything else, e.g. CG for this
SPD system: ``fem.solve(linear=jno.solve.cg(), precond=jno.precond.jacobi())`` -- see
``docs/fem.md`` "Choosing the solver".
"""

import jax.numpy as jnp
import numpy as np
from shapely.geometry import box

import jno

exact = lambda x, y: x * (1 - x) * y * (1 - y)  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.18)
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)  # weak form + u = 0
u_fem = jnp.asarray(fem.solve(linear=jno.solve.cg(), precond=jno.precond.jacobi()))  # SPD -> CG

pts = np.asarray(fem.points)  # coordinates the DOFs live on
rel_l2 = float(jnp.linalg.norm(exact(pts[:, 0], pts[:, 1]) - u_fem) / jnp.linalg.norm(exact(pts[:, 0], pts[:, 1])))
print(f"\nPoisson via jno.fem: dofs={fem.dofs}  rel_L2={rel_l2:.3e}")
assert fem.is_linear and rel_l2 < 5e-2
