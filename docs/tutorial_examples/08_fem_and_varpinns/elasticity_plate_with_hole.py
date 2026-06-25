"""Linear elasticity on a complex domain -- a plate with a hole, stretched in tension. The unknown is
the vector displacement ``u = (u_x, u_y)`` with P2 elements; the isotropic (plane-stress) bilinear
form is

    a(u, phi) = lambda (div u)(div phi) + 2 mu eps(u):eps(phi).

The hole is just ``box.difference(circle)``, with a named ``ring`` around it meshed finer to resolve
the steep strain. Verified by the method of manufactured solutions: a uniaxial-tension displacement
with a sinusoidal perturbation,

    u* = ( s (x + 0.3 sin pi y),  -nu s y ),     s = sigma / E,

gives the body force ``f = (0.3 mu s pi^2 sin pi y, 0)``; we impose ``u*`` on the whole boundary
(outer edges AND the hole) and recover it. The plate stretches in x and the circular hole deforms to
an ellipse. Solved with the built-in matrix-free (Jacobi-preconditioned) default.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import jax

jax.config.update("jax_enable_x64", True)  # the assembler builds in float64

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
E, nu, sigma = 1000.0, 0.3, 40.0  # Young's modulus, Poisson ratio, tension magnitude
lam, mu = E * nu / (1.0 - nu**2), E / (2.0 * (1.0 + nu))  # plane-stress Lame parameters
s = sigma / E
inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace
ux_star = lambda x, y: s * (x + 0.3 * jno.np.sin(PI * y))  # noqa: E731  uniaxial stretch + a wiggle
uy_star = lambda x, y: -nu * s * y  # noqa: E731

# complex domain: a 2x2 plate with a central hole; refine a named ring around the hole
hole = Point(0, 0).buffer(0.3)
ring = Point(0, 0).buffer(0.62).difference(hole)
dom = jno.domain({"plate": box(-1, -1, 1, 1).difference(hole).difference(ring), "ring": ring})
dom = dom.build_mesh(0.12, sizes={"ring": 0.045})  # coarse plate, fine collar at the hole

u, phi = dom.fem_symbols(value_shape=(2,), order=2)  # P2 vector displacement
xi, yi, _ = dom.variable("interior", split=True)
xb, yb, _ = dom.variable("boundary", split=True)
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)
fx = 0.3 * mu * s * PI**2 * jno.np.sin(PI * yi)  # body force f = -div sigma(u*)
fem = jno.fem(
    [
        weak - fx * phi.bind(x=xi, y=yi)[0],  # f_y = 0
        u(xb, yb)[0] - ux_star(xb, yb),  # manufactured Dirichlet on outer edges AND the hole
        u(xb, yb)[1] - uy_star(xb, yb),
    ]
)

sol = np.asarray(fem.solve())  # matrix-free Jacobi-preconditioned BiCGStab default (O(nnz) memory)
uu = sol.reshape(-1, 2)  # displacement (n_nodes, 2)
pts = np.asarray(fem.points)
ref = np.stack([s * (pts[:, 0] + 0.3 * np.sin(PI * pts[:, 1])), -nu * s * pts[:, 1]], 1)
rel_u = float(np.linalg.norm(uu - ref) / np.linalg.norm(ref))
print("\nElasticity: plate with a hole under tension (P2, matrix-free default solve)")
print(f"  dofs={fem.dofs}  plate/ring mesh = 0.12 / 0.045")
print(f"  MMS displacement recovery rel-L2: {rel_u:.3e}")

# ---- render the actual deformation: undeformed vs the mesh displaced by u (amplified) ----
tris = np.asarray(fem.domain.built_mesh.cells_dict["triangle"])
verts = np.asarray(fem.domain.built_mesh.points)[:, :2]
uv = griddata(pts, uu, (verts[:, 0], verts[:, 1]), method="linear")  # P2 -> linear vertices for the plot
scale = 0.3 / float(np.nanmax(np.hypot(uu[:, 0], uu[:, 1])))  # amplify for visibility
mag = np.hypot(uv[:, 0], uv[:, 1])
fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5.6))
for a, coords, title in ((a0, verts, "undeformed"), (a1, verts + scale * uv, f"stretched in x (x{scale:.0f})")):
    tr = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)
    tpc = a.tripcolor(tr, mag, cmap="viridis", shading="gouraud")
    a.triplot(tr, color="w", lw=0.18, alpha=0.4)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title(title, fontsize=11)
fig.colorbar(tpc, ax=(a0, a1), shrink=0.7, label="|u|")
fig.suptitle("Plate with a hole, manufactured uniaxial-tension field — the hole deforms to an ellipse", fontsize=12)
fig.savefig(Path(__file__).parents[2] / "assets" / "elasticity_plate_with_hole.png", dpi=130, bbox_inches="tight")

assert fem.is_linear and rel_u < 5e-3, f"MMS recovery too loose: {rel_u:.3e}"
