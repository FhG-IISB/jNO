"""Dirichlet-Laplacian eigenmodes -- a spectral verification of the assembled FEM operators.

    -lap phi = lambda phi   in Omega = (0,1)^2,    phi = 0 on the boundary.

Most FEM checks drive an operator with a forcing term; this one verifies the assembled **mass**
``M`` and **stiffness** ``K`` *spectrally*, with no forcing at all. Assembling each through
``jno.fem`` -- ``M`` from ``int phi_i phi_j`` (the ``u*v`` term), ``K`` from ``int grad.grad`` -- and
solving the generalized eigenproblem ``K x = lambda M x`` on the interior DOFs recovers the analytic
Dirichlet spectrum of the Laplacian on the unit square,

    lambda_{m,n} = pi^2 (m^2 + n^2),    eigenfunctions sin(m pi x) sin(n pi y),

including the *degenerate* pair ``lambda = 5 pi^2`` (modes (1,2) and (2,1)). A wrong mass or
stiffness assembly would shift the whole spectrum -- a check a single forced solve cannot make.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
dense = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.04)
u, w = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
ui, vi = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
fem_M = jno.fem([ui * vi])  # mass: int phi_i phi_j
M = dense(fem_M.A)
K = dense(jno.fem([ui.x * vi.x + ui.y * vi.y]).A)  # stiffness: int grad phi_i . grad phi_j
pts = np.asarray(fem_M.points)

eps = 1e-9  # interior DOFs = strictly inside the unit square (boundary nodes lie on an edge)
inside = (pts[:, 0] > eps) & (pts[:, 0] < 1 - eps) & (pts[:, 1] > eps) & (pts[:, 1] < 1 - eps)
evals, V = sla.eigh(K[np.ix_(inside, inside)], M[np.ix_(inside, inside)])  # K x = lambda M x

exact = np.array([2, 5, 5, 8, 10]) * PI**2  # pi^2(m^2+n^2): (1,1),(1,2),(2,1),(2,2),(1,3)/(3,1)
got = evals[:5]
rel = np.abs(got - exact) / exact
print("\nDirichlet-Laplacian eigenvalues  K x = lambda M x  (spectral verification)")
print(f"  dofs={fem_M.dofs}  interior modes={inside.sum()}")
print(f"  lambda_1..5 / pi^2  computed = {np.round(got / PI**2, 3)}")
print(f"  lambda_1..5 / pi^2  analytic = {np.round(exact / PI**2, 3)}  (max rel err {rel.max():.2e})")

# ---- render the first four eigenmodes ----
fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.0))
gx, gy = np.meshgrid(np.linspace(0, 1, 120), np.linspace(0, 1, 120))
for k, ax in enumerate(axes.flat):
    full = np.zeros(pts.shape[0])
    full[inside] = V[:, k] / np.max(np.abs(V[:, k]))
    G = griddata(pts, full, (gx, gy), method="cubic")
    ax.imshow(G, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(rf"$\lambda_{k + 1} = {evals[k] / PI**2:.2f}\,\pi^2$", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Dirichlet-Laplacian eigenmodes on the unit square", fontsize=12)
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "laplacian_eigenmodes_2d.png", dpi=130, bbox_inches="tight")

assert rel.max() < 4e-2, f"eigenvalues off the analytic spectrum: {got / PI**2} vs {exact / PI**2}"
assert abs(got[1] - got[2]) / got[1] < 2e-2, "the 5 pi^2 pair should be (near-)degenerate"
