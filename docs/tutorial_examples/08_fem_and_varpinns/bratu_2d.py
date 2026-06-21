"""The Bratu (Gelfand) problem -- a nonlinear combustion model via the ``jno.fem`` residual route.

    -lap u = lambda e^u   in Omega = (0,1)^2,    u = 0 on the boundary.

A classic model of a thermal explosion: the exponential reaction ``lambda e^u`` (heat release that
grows with temperature) competes with diffusion. Solutions exist only for ``lambda <= lambda_c ~
6.81`` on the unit square, in two branches (the lower one stable). The ``e^u`` term is nonlinear in
the unknown, so ``jno.fem`` returns a residual operator (``fem.residual`` / ``fem.jacobian`` from
autodiff) and we drive it with a Newton root-find from rest, landing on the stable lower-branch
bump at ``lambda = 2``.

No closed form exists in 2D, so we verify the prediction by **mesh convergence**: the peak value of
the computed solution must be mesh-independent (it agrees between a coarse and a fine mesh).

Reference: G. Bratu, "Sur les equations integrales non lineaires", Bull. Soc. Math. France 42
(1914), 113-142; Frank-Kamenetskii / Gelfand combustion theory.
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

dense = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)  # noqa: E731
lam = 2.0  # < lambda_c ~ 6.81: the stable lower branch


def solve_bratu(mesh_size):
    """-lap u = lam e^u, u=0; Newton from rest. Returns (peak, points, u_h, dofs)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    # grad u . grad v - lambda e^u v = 0 ; the e^u term makes the form nonlinear
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - lam * jno.np.exp(ui) * vi, u(xb, yb) - 0.0])
    assert not fem.is_linear, "the exponential reaction must make the form nonlinear"
    res = spo.root(
        lambda w: np.asarray(fem.residual(w)), np.zeros(fem.dofs), jac=lambda w: dense(fem.jacobian(w)), tol=1e-11
    )
    return float(res.x.max()), np.asarray(fem.points), res.x, fem.dofs


peak_c, _, _, _ = solve_bratu(0.06)  # coarse
peak_f, pts, uh, dofs = solve_bratu(0.04)  # fine
mesh_err = abs(peak_f - peak_c) / peak_f
print("\nBratu problem  -lap u = lambda e^u  (nonlinear, Newton residual route)")
print(f"  lambda={lam}  dofs={dofs}  lower-branch peak u_max = {peak_f:.4f}")
print(f"  mesh convergence of the peak (coarse vs fine): {mesh_err:.2e}")

# ---- render the computed solution (the combustion bump) ----
gx, gy = np.meshgrid(np.linspace(0, 1, 220), np.linspace(0, 1, 220))
G = griddata(pts, uh, (gx, gy), method="cubic")
fig, ax = plt.subplots(figsize=(4.8, 4.4))
im = ax.imshow(G, origin="lower", extent=(0, 1, 0, 1), cmap="inferno", vmin=0.0)
fig.colorbar(im, ax=ax, shrink=0.85, label=r"$u$")
ax.set_title(rf"Bratu $-\Delta u=\lambda e^u$, $\lambda={lam:g}$ — lower branch")
ax.set_xticks([])
ax.set_yticks([])
fig.savefig(Path(__file__).parents[2] / "assets" / "bratu_2d.png", dpi=130, bbox_inches="tight")

assert peak_f > 0.0 and mesh_err < 3e-2, f"Bratu peak not mesh-converged: {mesh_err:.2e}"
