# --8<-- [start:code]
"""02 - Mixed boundary conditions through ``jno.fdm``: Dirichlet + Neumann + Robin at once.

Steady conduction on the unit square with the manufactured field ``u*(x, y) = y^2`` (so -Delta u = -2):

    * bottom (y = 0):  Dirichlet   u = 0
    * top    (y = 1):  Robin       du/dn + (u - 3) = 0     (at y=1: du/dn = 2, u = 1  ->  2 + (1-3) = 0)
    * left / right  :  Neumann     du/dn = 0               (insulated; u = y^2 is flat in x)

A flux boundary condition is written with **that edge's own tags** -- bind the field to the edge
(``ur = u.bind(x=xr, y=yr)``) and take its normal derivative ``ur.d(nr)`` against the outward normal
``nr = domain.variable(region, normals=True)``. Any condition **affine in** ``du/dn`` works the same
way: Neumann ``ur.d(n) - h``, Robin ``ur.d(n) + alpha*(u - u_inf)``, either sign -- ``jno.fdm`` reads
the coefficient of ``du/dn`` directly, no special BC objects. Corner nodes shared by two flux edges
fall back to the interior PDE residual.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).domain()
x, y, _ = d.variable("interior", split=True)
xbo, ybo, _ = d.variable("bottom", split=True)
xl, yl, _ = d.variable("left", split=True)
xr, yr, _ = d.variable("right", split=True)
xt, yt, _ = d.variable("top", split=True)
nl = d.variable("left", normals=True)  # outward-normal Variables, one per edge
nr = d.variable("right", normals=True)
nt = d.variable("top", normals=True)

u = d.unknown()
ui = u.bind(x=x, y=y)  # interior view for the PDE
ul = u.bind(x=xl, y=yl)  # edge-bound views for the flux conditions
ur = u.bind(x=xr, y=yr)
ut = u.bind(x=xt, y=yt)

sol = jno.fdm(
    [
        -ui.d2(x) - ui.d2(y) + 2.0,  # -Delta u = -2   (manufactured u = y^2)
        u(xbo, ybo) - 0.0,  # Dirichlet: bottom held at 0
        ul.d(nl) - 0.0,  # Neumann: left insulated (du/dn = 0)
        ur.d(nr) - 0.0,  # Neumann: right insulated
        ut.d(nt) + 1.0 * (ut - 3.0),  # Robin: top convects to u_inf = 3 with alpha = 1
    ]
).solve()

p = np.asarray(d.mesh_connectivity["points"])[:, :2]
exact = p[:, 1] ** 2  # u = y^2
rel_l2 = float(np.linalg.norm(np.asarray(sol).reshape(-1) - exact) / np.linalg.norm(exact))
print(f"\nMixed Dirichlet+Neumann+Robin via jno.fdm: nodes={p.shape[0]}  rel_L2={rel_l2:.3e}")
assert rel_l2 < 5e-2, f"relative L2 error too large: {rel_l2:.3e}"
# --8<-- [end:code]

# ---- figure: computed field | |error| vs u* = y^2 | mesh-refinement convergence ------------
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


def _solve_mixed(size):
    """Re-run the mixed-BC solve at a given mesh size; return (h, rel_L2) against u* = y^2."""
    dm = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    xx, yy, _ = dm.variable("interior", split=True)
    xbo2, ybo2, _ = dm.variable("bottom", split=True)
    xl2, yl2, _ = dm.variable("left", split=True)
    xr2, yr2, _ = dm.variable("right", split=True)
    xt2, yt2, _ = dm.variable("top", split=True)
    nl2 = dm.variable("left", normals=True)
    nr2 = dm.variable("right", normals=True)
    nt2 = dm.variable("top", normals=True)
    uu = dm.unknown()
    uui = uu.bind(x=xx, y=yy)
    ul2 = uu.bind(x=xl2, y=yl2)
    ur2 = uu.bind(x=xr2, y=yr2)
    ut2 = uu.bind(x=xt2, y=yt2)
    s2 = jno.fdm(
        [
            -uui.d2(xx) - uui.d2(yy) + 2.0,
            uu(xbo2, ybo2) - 0.0,
            ul2.d(nl2) - 0.0,
            ur2.d(nr2) - 0.0,
            ut2.d(nt2) + 1.0 * (ut2 - 3.0),
        ]
    ).solve()
    pp = np.asarray(dm.mesh_connectivity["points"])[:, :2]
    ex = pp[:, 1] ** 2
    r = float(np.linalg.norm(np.asarray(s2).reshape(-1) - ex) / np.linalg.norm(ex))
    h = float(np.sqrt(dm.mesh_connectivity["p1_area"].mean()))
    return h, r


# Sweep several mesh sizes. The interior field is quadratic (u* = y^2), so the interior FD residual
# is tiny; the error is set by the one-sided Neumann/Robin edge stencils and the corner PDE-fallback,
# which shrink under refinement but NOT monotonically on unstructured meshes (mesh-dependent scatter).
sizes = [0.12, 0.1, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
conv = [_solve_mixed(sz) for sz in sizes]
hs = np.array([c[0] for c in conv])
errs = np.array([c[1] for c in conv])
print("convergence (h, rel_L2):", [(f"{h:.3f}", f"{e:.2e}") for h, e in conv])

pred = np.asarray(sol).reshape(-1)
tri = mtri.Triangulation(p[:, 0], p[:, 1], triangles=np.asarray(d.mesh_connectivity["triangles"]))
fig, ax = plt.subplots(1, 3, figsize=(13, 4))

im0 = ax[0].tripcolor(tri, pred, cmap="cividis", shading="gouraud")
ax[0].set_title("jno.fdm solution  u\n(Dirichlet + Neumann + Robin)")
ax[0].set_axis_off()
ax[0].set_aspect("equal")
fig.colorbar(im0, ax=ax[0], shrink=0.8)

im1 = ax[1].tripcolor(tri, np.abs(pred - exact), cmap="magma", shading="gouraud")
ax[1].set_title(r"$|u - u^*|$,  $u^* = y^2$")
ax[1].set_axis_off()
ax[1].set_aspect("equal")
fig.colorbar(im1, ax=ax[1], shrink=0.8)

ax[2].loglog(hs, errs, "o-", label="rel-$L^2$")
ax[2].loglog(hs, errs[0] * (hs / hs[0]) ** 2, "k--", alpha=0.6, label=r"$O(h^2)$ guide")
ax[2].set_title("mesh convergence\n(boundary-stencil limited)")
ax[2].set_xlabel("mean element size $h$")
ax[2].set_ylabel(r"relative $L^2$ error")
ax[2].grid(True, which="both", alpha=0.3)
ax[2].legend()

fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "mixed_bc_fdm.png")
