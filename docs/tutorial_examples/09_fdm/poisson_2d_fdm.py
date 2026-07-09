# --8<-- [start:code]
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

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.06).domain()
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
# --8<-- [end:code]

# ---- figure: computed field | signed error | mesh-refinement convergence -------------------
import os  # noqa: E402

os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

plt.rcParams.update(
    {
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "figure.dpi": 120,
    }
)


def _solve_poisson(size):
    """Re-run the identical strong-form solve at a given mesh size; return (h, rel_L2)."""
    dm = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    xx, yy, _ = dm.variable("interior", split=True)
    xxb, yyb, _ = dm.variable("boundary", split=True)
    uu = dm.unknown()
    uui = uu.bind(x=xx, y=yy)
    ff = 2.0 * np.pi**2 * jnn.sin(np.pi * xx) * jnn.sin(np.pi * yy)
    s2 = jno.fdm([-uui.d2(xx) - uui.d2(yy) - ff, uu(xxb, yyb) - 0.0]).solve()
    pp = np.asarray(dm.mesh_connectivity["points"])[:, :2]
    ex = np.sin(np.pi * pp[:, 0]) * np.sin(np.pi * pp[:, 1])
    r = float(np.linalg.norm(np.asarray(s2).reshape(-1) - ex) / np.linalg.norm(ex))
    h = float(np.sqrt(dm.mesh_connectivity["p1_area"].mean()))  # mean element size
    return h, r


sizes = [0.12, 0.09, 0.06, 0.04]
conv = [_solve_poisson(sz) for sz in sizes]
hs = np.array([c[0] for c in conv])
errs = np.array([c[1] for c in conv])
print("convergence (h, rel_L2):", [(f"{h:.3f}", f"{e:.2e}") for h, e in conv])
slope = float(np.polyfit(np.log(hs), np.log(errs), 1)[0])
print(f"fitted order p = {slope:.2f}")

pred = np.asarray(sol).reshape(-1)
tri = mtri.Triangulation(p[:, 0], p[:, 1], triangles=np.asarray(d.mesh_connectivity["triangles"]))
fig, ax = plt.subplots(1, 3, figsize=(13, 4))

im0 = ax[0].tripcolor(tri, pred, cmap="cividis", shading="gouraud")
ax[0].set_title("jno.fdm solution  u")
ax[0].set_axis_off()
ax[0].set_aspect("equal")
fig.colorbar(im0, ax=ax[0], shrink=0.8)

err = pred - exact
vmax = float(np.abs(err).max())
im1 = ax[1].tripcolor(tri, err, cmap="RdBu_r", shading="gouraud", vmin=-vmax, vmax=vmax)
ax[1].set_title(r"error  $u - u^*$")
ax[1].set_axis_off()
ax[1].set_aspect("equal")
fig.colorbar(im1, ax=ax[1], shrink=0.8)

ax[2].loglog(hs, errs, "o-", label="rel-$L^2$")
ax[2].loglog(hs, errs[0] * (hs / hs[0]) ** 2, "k--", alpha=0.6, label=r"$O(h^2)$")
ax[2].set_title(f"mesh convergence (order ≈ {slope:.2f})")
ax[2].set_xlabel("mean element size $h$")
ax[2].set_ylabel(r"relative $L^2$ error")
ax[2].grid(True, which="both", alpha=0.3)
ax[2].legend()

fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "poisson_2d_fdm.png")
