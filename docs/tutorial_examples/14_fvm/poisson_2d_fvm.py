# --8<-- [start:code]
"""01 - 2D Poisson equation through ``jno.fvm`` (finite volumes, cell- and vertex-centred).

    -Delta u = f on the unit square, u = 0 on the boundary,
    u*(x, y) = sin(pi x) sin(pi y),   f = 2 pi^2 sin(pi x) sin(pi y).

The constraint list is the one ``jno.fdm`` reads. ``jno.fvm`` integrates each term over a control volume:
``ui.xx + ui.yy`` becomes the face sum of the face gradient, the source is evaluated at the centroid. The
cell-centred solve returns one value per cell (``prob.centroids`` says where); ``centering="vertex"`` uses
median-dual volumes around the mesh nodes and returns nodal values.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402


def solve(h, centering):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2.0 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    prob = jno.fvm([-(ui.xx + ui.yy) - f, u(xb, yb) - 0.0], centering=centering)
    sol, c = np.asarray(prob.solve()), prob.centroids
    exact = np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1])
    return float(np.linalg.norm(sol - exact) / np.linalg.norm(exact))


for centering in ("cell", "vertex"):
    errs = [solve(h, centering) for h in (0.1, 0.05)]
    print(f"{centering}-centred: rel-L2 {errs[0]:.2e} -> {errs[1]:.2e}  (ratio {errs[0] / errs[1]:.1f}, h^2 gives 4)")
    assert errs[1] < 4e-3 and errs[0] / errs[1] > 3.5
# --8<-- [end:code]
