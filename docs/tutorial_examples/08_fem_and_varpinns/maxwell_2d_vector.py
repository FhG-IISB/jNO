"""11 - Time-harmonic 2D Maxwell (in-plane vector E field) via jno's complex-as-coupled-real-fields.

Frequency-domain Maxwell for the in-plane electric field ``E = (Ex, Ey)`` (TE polarisation) is the
**curl-curl** equation

    curl(curl E) - k^2 E = J ,     k^2 = omega^2 * mu * eps   (complex in a lossy medium),

with ``E`` complex-valued. A complex field has no native FEM unknown in jno; instead a complex vector
is carried as **two coupled real vector fields** ``(E_r, E_i)`` (4 real DOF/node) and the complex
equation is split into its real and imaginary parts -- a coupled multifield system that jno.fem
assembles and solves like any other (the same machine behind the two-temperature / Stokes examples).

Two practical points for a *nodal* (Lagrange) discretisation of curl-curl:
  * the 2-D scalar curl is ``curl F = dFy/dx - dFx/dy`` -- written ``F.x[1] - F.y[0]`` (grad-then-index,
    since feax differentiates the trial, not a component of it);
  * nodal elements need a **grad-div penalty** ``+ s * div(E) div(v)`` to kill the spurious curl
    kernel; it is consistent here because the exact field is divergence-free (so the penalty vanishes
    at the solution). With it, P1 converges ~O(h^2) and P2 is essentially exact.

Verification (no hand-waving): a manufactured divergence-free ``E`` with **Re(E) != Im(E)** and a
genuinely complex ``k^2`` -- curl(curl E_r)=2*pi^2 E_r, curl(curl E_i)=8*pi^2 E_i -- so the forcing is
``J = curl(curl E) - k^2 E`` in closed form. The script recovers it to a tight tolerance and renders
the *computed* field (no invented structure).
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import jax

jax.config.update("jax_enable_x64", True)  # the coupled real system is float64

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

pi, sin, cos = np.pi, jno.np.sin, jno.np.cos
KR, KI = 30.0, 4.0  # k^2 = 30 + 4i  (non-resonant, lossy -> complex, non-singular)

# manufactured divergence-free fields: E_r is a curl-eigenfield (eig 2*pi^2), E_i a higher mode (8*pi^2)
E_r = lambda X, Y: (pi * sin(pi * X) * cos(pi * Y), -pi * cos(pi * X) * sin(pi * Y))  # noqa: E731
E_i = lambda X, Y: (2 * pi * sin(2 * pi * X) * cos(2 * pi * Y), -2 * pi * cos(2 * pi * X) * sin(2 * pi * Y))  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06)
Er, Pr = d.fem_symbols(value_shape=(2,), names=("Er", "Pr"), order=2)  # P2 real part of E + its test
Ei, Qi = d.fem_symbols(value_shape=(2,), names=("Ei", "Qi"), order=2)  # P2 imaginary part of E + its test
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
erb, prb = Er.bind(x=xi, y=yi), Pr.bind(x=xi, y=yi)
eib, qib = Ei.bind(x=xi, y=yi), Qi.bind(x=xi, y=yi)

curl = lambda F: F.x[1] - F.y[0]  # dFy/dx - dFx/dy  # noqa: E731
div = lambda F: F.x[0] + F.y[1]  # noqa: E731
dot = lambda a, b: a[0] * b[0] + a[1] * b[1]  # noqa: E731
s = KR  # grad-div penalty weight (consistent: exact E is divergence-free)

exr, eyr = E_r(xi, yi)
exi, eyi = E_i(xi, yi)
# J = curl(curl E) - k^2 E, split: J_r = (2pi^2 - kr) E_r + ki E_i ; J_i = (8pi^2 - kr) E_i - ki E_r
jxr, jyr = (2 * pi**2 - KR) * exr + KI * exi, (2 * pi**2 - KR) * eyr + KI * eyi
jxi, jyi = (8 * pi**2 - KR) * exi - KI * exr, (8 * pi**2 - KR) * eyi - KI * eyr

eq_re = (
    curl(erb) * curl(prb)
    + s * div(erb) * div(prb)
    - (KR * dot(erb, prb) - KI * dot(eib, prb))
    - (jxr * prb[0] + jyr * prb[1])
)
eq_im = (
    curl(eib) * curl(qib)
    + s * div(eib) * div(qib)
    - (KR * dot(eib, qib) + KI * dot(erb, qib))
    - (jxi * qib[0] + jyi * qib[1])
)
brx, bry = E_r(xb, yb)
bix, biy = E_i(xb, yb)  # E = exact on the wall (per-component: a coordinate-dependent vector BC is not yet a single term)
fem = jno.fem(
    [eq_re, eq_im, Er(xb, yb)[0] - brx, Er(xb, yb)[1] - bry, Ei(xb, yb)[0] - bix, Ei(xb, yb)[1] - biy],
    quad_degree=6,
)

A = jno.np.asarray(fem.A.todense()) if hasattr(fem.A, "todense") else jno.np.asarray(fem.A)
sol = np.asarray(np.linalg.solve(np.asarray(A), np.asarray(fem.b).reshape(-1)))
off = np.asarray(fem.problem.offset)
pts = np.asarray(fem.problem.mesh[0].points)
n = int(off[1])
E_re = sol[:n].reshape(-1, 2)  # computed Re(E) = (Ex_r, Ey_r) at every node
E_im = sol[n:].reshape(-1, 2)  # computed Im(E)

px, py = pts[:, 0], pts[:, 1]  # the manufactured reference, in plain numpy (E_r/E_i use jno.np for the trace)
ex_re = np.stack([pi * np.sin(pi * px) * np.cos(pi * py), -pi * np.cos(pi * px) * np.sin(pi * py)], axis=1)
ex_im = np.stack(
    [2 * pi * np.sin(2 * pi * px) * np.cos(2 * pi * py), -2 * pi * np.cos(2 * pi * px) * np.sin(2 * pi * py)], axis=1
)
rel = float(np.linalg.norm(np.concatenate([E_re - ex_re, E_im - ex_im])) / np.linalg.norm(np.concatenate([ex_re, ex_im])))
print("\n2D vector Maxwell (curl-curl, complex k^2) via complex-as-coupled-real-fields")
print(f"  4 real DOF/node, {fem.dofs} DOFs total;  L2 rel error vs manufactured E: {rel:.2e}")
assert rel < 2e-3, f"manufactured Maxwell not recovered: {rel:.2e}"

# ---- render the actual computed field (|E| and the Re(E) vectors) -- no invented structure ----
tris = np.asarray(fem.domain.built_mesh.cells_dict["triangle"])
triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
Emag = np.sqrt((E_re**2 + E_im**2).sum(1))  # |E| = sqrt(|Ex|^2 + |Ey|^2)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
tpc = ax[0].tripcolor(triang, Emag, cmap="magma", shading="gouraud")
fig.colorbar(tpc, ax=ax[0], shrink=0.85)
ax[0].set_title("|E|  (computed magnitude)", fontsize=10)
step = max(1, len(pts) // 400)
ax[1].tripcolor(triang, Emag, cmap="Greys", shading="gouraud", alpha=0.35)
ax[1].quiver(pts[::step, 0], pts[::step, 1], E_re[::step, 0], E_re[::step, 1], color="C0", scale=60, width=0.004)
ax[1].set_title("Re(E)  (computed vector field)", fontsize=10)
for a in ax:
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle(f"Time-harmonic 2D Maxwell, in-plane E (k² = {KR:g} + {KI:g}i);  L2 err {rel:.1e}", fontsize=11)
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "maxwell_2d_vector.png", dpi=130, bbox_inches="tight")
