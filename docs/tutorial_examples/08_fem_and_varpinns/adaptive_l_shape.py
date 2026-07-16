# --8<-- [start:code]
r"""Adaptive mesh refinement on the L-shape, driven entirely by ``fem.solve(adapt=...)``.

    -lap u = 0  on the L-shape,  Dirichlet = the exact singular mode  u = r^(2/3) sin(2 phi / 3)
    about the reentrant corner (0.5, 0.5).

``u`` is harmonic, so all the discretization error is in resolving the ``r^(2/3)`` corner
singularity. ``fem.solve(adapt=AdaptSpec(...))`` runs the whole loop internally --
``solve -> Zienkiewicz-Zhu error estimate -> Dörfler mark -> local remesh`` -- with no hand-rolled
code: it concentrates elements at the corner, mutates the domain to the final adapted mesh, and
records the per-round trace on ``fem.adapt_history``.

Run::

    JAX_PLATFORMS=cpu pixi run -e fem python docs/tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small FEM solves; keep off the GPU

import jax

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402
import jno.jnp_ops as J  # noqa: E402
from jno.utils.solver.fem_adapt import AdaptSpec  # noqa: E402


def _mod(a, m):
    # jno.np has no `mod`; build it from `floor` (works on trace symbols and numpy)
    return a - m * J.floor(a / m)


def u_singular(x, y, xp, mod):
    X, Y = x - 0.5, y - 0.5
    r = xp.sqrt(X * X + Y * Y)
    th = mod(xp.arctan2(Y, X), 2.0 * np.pi)
    phi = mod(th - np.pi / 2.0, 2.0 * np.pi)  # material wedge is 3*pi/2 wide
    return (r ** (2.0 / 3.0)) * xp.sin(2.0 / 3.0 * phi)


# ``-lap u = 0`` with the singular mode as Dirichlet data on the whole boundary.
d = jno.Shape.polygon([(0, 0), (1.0, 0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0, 1.0)], size=0.3).domain()
u, phi = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - u_singular(xb, yb, J, _mod)])

# The whole adaptive loop in one call: solve -> ZZ estimate -> Dörfler mark -> local remesh, repeated.
# It returns the solution on the final adapted mesh; `d`/`fem` now refer to that mesh and
# `fem.adapt_history` traces {n_dofs, estimate} per round.
sol = np.asarray(fem.solve(adapt=AdaptSpec(theta=0.6, max_iters=6, refine_factor=1.7))).reshape(-1)
hist = fem.adapt_history
dofs = np.array([h["n_dofs"] for h in hist])
est = np.array([h["estimate"] for h in hist])
print(f"adaptive rounds (n_dofs, ZZ estimate): {[(int(n), round(float(e), 5)) for n, e in zip(dofs, est)]}")

# Adaptive refinement must add DOFs at the corner AND drive the ZZ error estimate down.
assert len(hist) >= 2 and dofs[-1] > dofs[0] and est[-1] < 0.6 * est[0], (
    f"adaptive loop should reduce the ZZ estimate {est[0]:.3e} -> {est[-1]:.3e} "
    f"while refining {int(dofs[0])} -> {int(dofs[-1])} dofs"
)
# --8<-- [end:code]

# --- figure: the final adapted mesh, its solution, and the convergence history ---
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

plt.rcParams.update({"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10})
pts = np.asarray(d.mesh.points)[:, :2]
tris = np.asarray(d.mesh.cells_dict["triangle"])
tri = Triangulation(pts[:, 0], pts[:, 1], tris)
fig, ax = plt.subplots(1, 3, figsize=(12.5, 4.0))
ax[0].triplot(tri, color="0.2", lw=0.3)
ax[0].set_title(f"final adapted mesh · {len(sol)} dofs")
tpc = ax[1].tripcolor(tri, sol, shading="gouraud", cmap="cividis")
ax[1].set_title(r"solution $u = r^{2/3}\sin(2\varphi/3)$")
fig.colorbar(tpc, ax=ax[1], shrink=0.8)
ax[2].loglog(dofs, est, "o-", color="#0072B2")
ax[2].set_xlabel("dofs")
ax[2].set_ylabel("ZZ error estimate")
ax[2].set_title("convergence")
ax[2].grid(True, which="both", alpha=0.3)
for a in ax[:2]:
    a.set_aspect("equal")
    a.set_axis_off()
fig.suptitle("Adaptive refinement at the L-shape reentrant corner", fontweight="bold")
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "adaptive_l_shape.png")
