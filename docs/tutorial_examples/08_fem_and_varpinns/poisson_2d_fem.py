# --8<-- [start:code]
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

import jno

exact = lambda x, y: x * (1 - x) * y * (1 - y)  # noqa: E731

d = jno.Shape.rect(0, 0, 1, 1, size=0.18).domain()
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
# --8<-- [end:code]

# ---- solution figure: computed field | error vs exact | convergence under refinement ----
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)


def solve_at(size):
    """Rebuild + solve the whole problem on a fresh mesh of the given element size (real re-solve)."""
    dd = jno.Shape.rect(0, 0, 1, 1, size=size).domain()
    uu, pp = dd.fem_symbols()
    x_i, y_i, _ = dd.variable("interior", split=True)
    x_b, y_b, _ = dd.variable("boundary", split=True)
    uh, vh = uu.bind(x=x_i, y=y_i), pp.bind(x=x_i, y=y_i)
    rhs = 2.0 * (x_i * (1 - x_i) + y_i * (1 - y_i))
    fe = jno.fem([uh.x * vh.x + uh.y * vh.y - rhs * vh, uu(x_b, y_b) - 0.0], quad_degree=3)
    sol = jnp.asarray(fe.solve(linear=jno.solve.cg(), precond=jno.precond.jacobi()))
    p = np.asarray(fe.points)
    ex = exact(p[:, 0], p[:, 1])
    return fe.dofs, float(jnp.linalg.norm(ex - sol) / jnp.linalg.norm(ex))


err = np.asarray(u_fem) - exact(pts[:, 0], pts[:, 1])  # signed nodal error of the computed field
sizes = [0.30, 0.22, 0.16, 0.12, 0.09]
conv = [solve_at(s) for s in sizes]
dofs_c, rel_c = np.array([c[0] for c in conv]), np.array([c[1] for c in conv])

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
c0 = ax[0].tricontourf(pts[:, 0], pts[:, 1], np.asarray(u_fem), levels=20, cmap="cividis")
ax[0].set_title(f"computed $u$  (dofs={fem.dofs})")
fig.colorbar(c0, ax=ax[0], shrink=0.8)
m = float(np.max(np.abs(err)))
c1 = ax[1].tricontourf(pts[:, 0], pts[:, 1], err, levels=20, cmap="RdBu_r", vmin=-m, vmax=m)
ax[1].set_title(f"error $u-u^*$  (rel-$L^2$={rel_l2:.1e})")
fig.colorbar(c1, ax=ax[1], shrink=0.8)
for a in ax[:2]:
    a.set_aspect("equal")
    a.set_axis_off()
ax[2].loglog(dofs_c, rel_c, "o-", color="#3b6fb6")
ax[2].set_title("convergence (real re-solves)")
ax[2].set_xlabel("dofs")
ax[2].set_ylabel(r"rel-$L^2$ error")
ax[2].grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "poisson_2d_fem.png")
