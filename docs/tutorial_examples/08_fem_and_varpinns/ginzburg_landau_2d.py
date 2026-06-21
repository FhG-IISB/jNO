"""Vector Ginzburg-Landau -- a nonlinear *vector* phase field via ``jno.fem``.

    -lap u + (|u|^2 - 1) u = f,    u = (u_1, u_2) a 2-vector,    u = u* on the boundary.

The Ginzburg-Landau energy underlies superconductivity and pattern formation. Here the unknown is a
two-component vector field, and the reaction ``(|u|^2 - 1) u`` couples the components through
``|u|^2 = u.u`` -- a cubic nonlinearity in which the unknown is contracted with itself. ``jno.fem``
detects this, routes the system to its coupled nonlinear operator (autodiff Jacobian), and a Newton
root-find recovers the field.

Verified by the method of manufactured solutions: we pick a transcendental
``u* = (sin(pi x) sin(pi y),  sin(2 pi x) sin(pi y))``, compute the matching forcing ``f``, impose
``u*`` on the boundary, and check the recovered field against it. (The same problem's order of
accuracy is certified in the convergence suite, ``tests/test_fem_convergence.py``.)
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.optimize as spo  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

inner, grad = jno.np.inner, jno.np.grad
sin = jno.np.sin
PI = np.pi
dense = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
u, w = d.fem_symbols(value_shape=(2,), names=("u", "w"))
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ub, vv = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)

# manufactured u* and the matching forcing f = -lap u* + (|u*|^2 - 1) u*
U1, U2 = sin(PI * xi) * sin(PI * yi), sin(2 * PI * xi) * sin(PI * yi)
m2 = U1 * U1 + U2 * U2
f1 = 2 * PI**2 * U1 + (m2 - 1.0) * U1
f2 = 5 * PI**2 * U2 + (m2 - 1.0) * U2
react = (inner(ub, ub, n_contract=1) - 1.0) * inner(ub, vv, n_contract=1)  # (|u|^2 - 1) u . v
weak = inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=2) + react - (f1 * vv[0] + f2 * vv[1])
fem = jno.fem([weak, u(xb, yb) - (0.0, 0.0)])
assert not fem.is_linear, "the cubic reaction must make the form nonlinear"

res = spo.root(lambda v: np.asarray(fem.residual(v)), np.zeros(fem.dofs), jac=lambda v: dense(fem.jacobian(v)), tol=1e-11)
pts = np.asarray(fem.points)
uu = res.x.reshape(-1, 2)
ex = np.stack(
    [np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1]), np.sin(2 * PI * pts[:, 0]) * np.sin(PI * pts[:, 1])], axis=-1
)
rel = float(np.linalg.norm(uu - ex) / np.linalg.norm(ex))
print("\nVector Ginzburg-Landau  -lap u + (|u|^2-1)u = f  (nonlinear, Newton residual route)")
print(f"  dofs={fem.dofs}  Newton |residual|={np.linalg.norm(np.asarray(fem.residual(res.x))):.1e}")
print(f"  manufactured recovery: rel-L2 = {rel:.2e}")

# ---- render the computed magnitude |u| ----
mag = np.hypot(uu[:, 0], uu[:, 1])
gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
G = griddata(pts, mag, (gx, gy), method="cubic")
fig, ax = plt.subplots(figsize=(4.8, 4.4))
im = ax.imshow(G, origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=0.0)
fig.colorbar(im, ax=ax, shrink=0.85, label=r"$|\mathbf{u}|$")
ax.set_title(r"Ginzburg--Landau $|\mathbf{u}|$ — nonlinear vector field")
ax.set_xticks([])
ax.set_yticks([])
fig.savefig(Path(__file__).parents[2] / "assets" / "ginzburg_landau_2d.png", dpi=130, bbox_inches="tight")

assert res.success and rel < 1e-2, f"Ginzburg-Landau field not recovered: rel-L2 {rel:.2e}"
