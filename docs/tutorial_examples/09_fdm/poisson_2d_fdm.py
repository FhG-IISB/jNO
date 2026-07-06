"""01 - 2D Poisson equation solved through ``jno.fdm`` (finite differences, strong form).

    -Delta u = f on the unit square, u = 0 on the boundary.
    Manufactured  u*(x, y) = sin(pi x) sin(pi y),   f = 2 pi^2 sin(pi x) sin(pi y).

``jno.fdm`` is the **strong-form sibling** of ``jno.fem``: author the PDE and its boundary
conditions as the *same* constraint list, with ``u = domain.unknown()`` (a valued nodal field, the
counterpart of ``fem_symbols()``). Instead of a weak form with test functions and quadrature, the
strong residual is collocated at the mesh nodes with finite-difference stencils -- so ``ui.d2(x)`` is
the FD second derivative (autodiff is meaningless on a discrete field, so FD is the default; no
``scheme=`` needed). The Dirichlet condition is the term ``u(region) - g``, exactly as in ``jno.fem``.
"""

import jax

jax.config.update("jax_enable_x64", True)  # the strong-form solve accumulates in float64

import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
x, y, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
u = d.unknown()  # valued P1 nodal field (strong-form counterpart of fem_symbols())
ui = u.bind(x=x, y=y)  # bound view with .d / .d2 (FD by default)

f = 2.0 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
sol = jno.fdm(
    [
        -ui.d2(x) - ui.d2(y) - f,  # -Delta u = f   (finite differences at the mesh nodes)
        u(xb, yb) - 0.0,  # Dirichlet u = 0 on the boundary
    ]
).solve()

p = np.asarray(d.mesh_connectivity["points"])[:, :2]  # the nodes the DOFs live on
exact = np.sin(np.pi * p[:, 0]) * np.sin(np.pi * p[:, 1])
rel_l2 = float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))
print(f"\nPoisson via jno.fdm: nodes={p.shape[0]}  rel_L2={rel_l2:.3e}")
assert rel_l2 < 3e-2, f"relative L2 error too large: {rel_l2:.3e}"
