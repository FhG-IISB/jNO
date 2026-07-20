# --8<-- [start:code]
r"""Two ways to resolve the L-shape reentrant-corner singularity — **h-adaptivity** (add elements) and
**differentiable r-adaptivity** (relocate a *fixed* set of nodes), on the same problem, side by side.

    -lap u = 0  on the L-shape,  Dirichlet = the exact singular mode  u = r^(2/3) sin(2 phi / 3)
    about the reentrant corner (0.5, 0.5).

``u`` is harmonic, so all the discretization error is the ``r^(2/3)`` corner singularity, measured here by
the **energy-norm error** ``E - E_ref`` with ``E = ½∫|∇u_h|²`` and ``E_ref`` from a fine mesh.

* **h-adaptivity** — ``fem.solve(adapt=AdaptSpec(...))`` runs the whole classical loop internally
  (``solve -> Zienkiewicz-Zhu estimate -> Dörfler mark -> local remesh``): it **adds** elements at the
  corner. The remesh is a discrete, non-differentiable outer loop.
* **r-adaptivity** — make the interior vertex coordinates trainable (``x.trainable()``) and move them down
  the energy gradient, computed *through the differentiable solve* (``∂(fem.solve())/∂X``, the keystone).
  It **relocates** a fixed node set — no new DOFs, fixed connectivity, one JAX graph, no remeshing.

Run::

    JAX_PLATFORMS=cpu pixi run -e fem python docs/tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small FEM solves; keep off the GPU

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as J  # noqa: E402
from jno.utils.solver.fem_adapt import AdaptSpec  # noqa: E402

L_SHAPE = [(0, 0), (1.0, 0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0, 1.0)]  # reentrant corner at (0.5, 0.5)
MARGIN = 0.025  # keep the movable interior nodes off the boundary (incl. the two notch edges)


def _mod(a, m):  # jno.np has no `mod`; build it from `floor` (works on trace symbols and numpy)
    return a - m * J.floor(a / m)


def u_singular(x, y, xp, mod):
    X, Y = x - 0.5, y - 0.5
    r = xp.sqrt(X * X + Y * Y)
    phi = mod(mod(xp.arctan2(Y, X), 2.0 * np.pi) - np.pi / 2.0, 2.0 * np.pi)  # 3*pi/2-wide material wedge
    return (r ** (2.0 / 3.0)) * xp.sin(2.0 / 3.0 * phi)


def _interior(x, y):
    """True interior L-shape vertices: inside the material, off the outer box and the two notch edges."""
    inside = ~((x > 0.5) & (y > 0.5))
    off_box = (x > MARGIN) & (x < 1 - MARGIN) & (y > MARGIN) & (y < 1 - MARGIN)
    off_notch = ~(((jnp.abs(x - 0.5) < MARGIN) & (y > 0.5 - MARGIN)) | ((jnp.abs(y - 0.5) < MARGIN) & (x > 0.5 - MARGIN)))
    return inside & off_box & off_notch


def build(size, movable=False):
    d = jno.Shape.polygon(L_SHAPE, size=size).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    if movable:
        # ---- the r-adaptivity API ----------------------------------------------------------------------
        # `.trainable()` on a spatial coordinate turns that region's mesh VERTICES into a design variable:
        # the assembler routes them into the element geometry, so `fem.solve()` becomes differentiable in
        # the node positions. Literal, per component (x and y are separate). Must be called BEFORE jno.fem.
        # The boundary is left fixed, so the L-shape itself never changes — only its interior nodes move.
        xm, ym, _ = d.variable("mov", where=_interior, split=True)
        xm.trainable(name="ix")
        ym.trainable(name="iy")
        # -------------------------------------------------------------------------------------------------
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - u_singular(xb, yb, J, _mod)])
    return d, fem


def dirichlet_energy(pts, sol, cells):
    """``½∫|∇u_h|²`` for a P1 field, straight from the vertices + nodal values — differentiable in BOTH
    (so it works as the r-adaptivity objective ``∂E/∂X`` through the solve). The FE energy is bounded below
    by the true energy and falls as the mesh resolves the corner, so minimizing it is the energy-norm-error
    goal (Ciarlet, *The Finite Element Method for Elliptic Problems*, 1978)."""
    v = pts[cells]
    s = sol[cells]
    x0, x1, x2 = v[:, 0, 0], v[:, 1, 0], v[:, 2, 0]
    y0, y1, y2 = v[:, 0, 1], v[:, 1, 1], v[:, 2, 1]
    detJ = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    gx = (s[:, 0] * (y1 - y2) + s[:, 1] * (y2 - y0) + s[:, 2] * (y0 - y1)) / detJ
    gy = (s[:, 0] * (x2 - x1) + s[:, 1] * (x0 - x2) + s[:, 2] * (x1 - x0)) / detJ
    return 0.25 * jnp.sum((gx * gx + gy * gy) * jnp.abs(detJ))


def _solve(fem):
    A, b = fem.A, fem.b
    Ad = jnp.asarray(A.todense() if hasattr(A, "todense") else A)
    return jnp.linalg.solve(Ad, jnp.asarray(b).reshape(-1))


def _min_detj(pts, cells):
    v = pts[cells]
    a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    return float(np.min(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))


def relocate(fem, d, e_ref, n_steps=60, lr=3e-3):
    """Descend the FE-energy gradient to relocate the trainable interior vertices.

    The built-in part is one line — ``jax.value_and_grad(energy)`` — which is exact only because
    ``.trainable()`` made ``fem.solve()`` differentiable in the mesh (``fem.operator.evaluate`` is
    differentiable in the coordinate parameters). The rest is a plain compact optimiser: a per-node
    RMS-normalised step (the near-corner gradients are huge, so unnormalised descent would stall the rest)
    with a backtracking safeguard so the mesh never tangles. A relocation companion to
    ``fem.solve(adapt=...)`` — it concentrates a *fixed* node set instead of adding new ones. (Because the
    gradient is exact, a learned monitor network can be trained through it via ``jno.core`` — the ML path.)"""
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    cj = jnp.asarray(cells)
    pts0 = np.asarray(d.mesh.points)[:, :2].copy()
    pts0_j = jnp.asarray(pts0)
    (sx, sy) = (d._trainable_coords[0], d._trainable_coords[1])
    ids, nx, ny = sx["ids"], sx["name"], sy["name"]

    def energy(xy):
        A, b = fem.operator.evaluate({nx: xy[0], ny: xy[1]})
        u = jnp.linalg.solve(jnp.asarray(A.todense()), jnp.asarray(b).reshape(-1))
        pts = pts0_j.at[ids, 0].set(xy[0]).at[ids, 1].set(xy[1])
        return dirichlet_energy(pts, u, cj)

    val_grad = jax.jit(jax.value_and_grad(energy))
    floor = 0.1 * _min_detj(pts0, cells)  # never let any cell fall below 10% of the initial worst quality

    def moved(xy):
        p = pts0.copy()
        p[ids, 0], p[ids, 1] = np.asarray(xy[0]), np.asarray(xy[1])
        return p

    xy = (pts0_j[ids, 0], pts0_j[ids, 1])
    msq = (jnp.zeros_like(xy[0]), jnp.zeros_like(xy[1]))  # RMSProp running average of squared gradient
    hist = [float(energy(xy)) - e_ref]
    for _ in range(n_steps):
        e, g = val_grad(xy)
        msq = (0.9 * msq[0] + 0.1 * g[0] ** 2, 0.9 * msq[1] + 0.1 * g[1] ** 2)
        step = (g[0] / jnp.sqrt(msq[0] + 1e-8), g[1] / jnp.sqrt(msq[1] + 1e-8))
        a = lr
        for _ in range(20):  # backtracking on mesh validity
            cand = (xy[0] - a * step[0], xy[1] - a * step[1])
            if _min_detj(moved(cand), cells) > floor:
                break
            a *= 0.5
        else:
            break  # no admissible step -> at the mesh-quality limit
        xy = cand
        hist.append(float(e) - e_ref)
    return moved(xy), np.array(hist)


# --- reference energy (fine mesh) and a common coarse starting mesh ---------------------------------------
d_ref, fem_ref = build(0.03)
E_REF = float(
    dirichlet_energy(
        jnp.asarray(np.asarray(d_ref.mesh.points)[:, :2]), _solve(fem_ref), jnp.asarray(d_ref.mesh.cells_dict["triangle"])
    )
)

# --- (1) h-adaptivity: the built-in adaptive driver, the whole loop in ONE call ---------------------------
d_h, fem_h = build(0.12)
n0 = len(d_h.mesh.points)
E0 = float(
    dirichlet_energy(
        jnp.asarray(np.asarray(d_h.mesh.points)[:, :2]), _solve(fem_h), jnp.asarray(d_h.mesh.cells_dict["triangle"])
    )
)
# ---- the h-adaptivity API: `fem.solve(adapt=...)` runs solve -> ZZ-estimate -> Dörfler-mark -> remesh
#      internally and returns the solution on the final adapted mesh (`d_h`/`fem_h` now refer to it). ----
sol_h = np.asarray(fem_h.solve(adapt=AdaptSpec(theta=0.6, max_iters=4, refine_factor=1.7))).reshape(-1)
pts_h, tris_h = np.asarray(d_h.mesh.points)[:, :2], np.asarray(d_h.mesh.cells_dict["triangle"])
E_h = float(dirichlet_energy(jnp.asarray(pts_h), jnp.asarray(sol_h), jnp.asarray(tris_h)))
n_h = len(sol_h)

# --- (2) r-adaptivity: `.trainable()` coordinates + the differentiable solve ------------------------------
# `build(movable=True)` calls `xm.trainable()` / `ym.trainable()` on the interior coordinates (see build()),
# so `fem_r.solve()` is now differentiable w.r.t. the mesh. The exact shape gradient ∂E/∂X is therefore just
# `jax.grad` — no adjoint code — and `relocate` simply descends it (no new DOFs, fixed connectivity).
d_r, fem_r = build(0.12, movable=True)
pts_r, hist_r = relocate(fem_r, d_r, E_REF)
tris_r = np.asarray(d_r.mesh.cells_dict["triangle"])
E_r = float(hist_r[-1] + E_REF)

print(f"energy-norm error  (E - E_ref),  E_ref = {E_REF:.4f}")
print(f"  coarse start   : {E0 - E_REF:.3e}   ({n0} dofs)")
print(
    f"  h-adaptivity   : {E_h - E_REF:.3e}   ({n_h} dofs, +{n_h - n0})   {100 * (1 - (E_h - E_REF) / (E0 - E_REF)):.0f}% lower"
)
print(f"  r-adaptivity   : {E_r - E_REF:.3e}   ({n0} dofs, +0)   {100 * (1 - (E_r - E_REF) / (E0 - E_REF)):.0f}% lower")

# h-adaptivity adds DOFs and lowers the error; r-adaptivity lowers it at FIXED DOFs, without tangling.
assert n_h > n0, "h-adaptivity should add DOFs at the corner"
assert E_h - E_REF < 0.4 * (E0 - E_REF), "h-adaptivity should sharply cut the energy error"
assert E_r - E_REF < 0.75 * (E0 - E_REF), "r-adaptivity should cut the energy error at fixed DOFs"
assert _min_detj(pts_r, tris_r) > 0, "r-adaptivity must not tangle the mesh"
# --8<-- [end:code]

# --- figure: h-refined mesh vs r-relocated mesh, and the two mechanisms on one error-vs-DOF axis ----------
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

plt.rcParams.update(
    {
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Frutiger 45 Light", "FreeSans", "DejaVu Sans"],
        "axes.titleweight": "light",
        "axes.labelweight": "bold",
        "axes.edgecolor": "#94A3B8",
    }
)
TEAL, INK, BLUE, GRAY = "#0D9488", "#1A202C", "#0072B2", "#94A3B8"
fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.2))


def mesh_panel(a, pts, tris, title):
    a.triplot(Triangulation(pts[:, 0], pts[:, 1], tris), color=INK, lw=0.4, alpha=0.7)
    a.plot([0.5], [0.5], "x", color=TEAL, ms=9, mew=2.2)  # the reentrant corner
    a.set_aspect("equal")
    a.set_axis_off()
    a.set_title(title)


mesh_panel(ax[0], pts_h, tris_h, f"h-adaptivity  ·  {n_h} dofs  (+{n_h - n0})")
mesh_panel(ax[1], pts_r, tris_r, f"r-adaptivity  ·  {n0} dofs  (relocated)")

e0, eh, er = E0 - E_REF, E_h - E_REF, E_r - E_REF
ax[2].plot([n0, n_h], [e0, eh], "o-", color=BLUE, lw=1.4, label="h-adaptivity (+DOFs)")
ax[2].plot([n0, n0], [e0, er], "o-", color=TEAL, lw=1.4, label="r-adaptivity (fixed DOFs)")
ax[2].plot([n0], [e0], "o", color=GRAY, ms=7)
ax[2].annotate("coarse start", (n0, e0), (n0 + 6, e0 * 1.15), color=GRAY, fontsize=9)
ax[2].set_yscale("log")
ax[2].set_xlabel("degrees of freedom")
ax[2].set_ylabel(r"energy-norm error  $E - E_{\mathrm{ref}}$")
ax[2].legend(frameon=False, fontsize=9, loc="upper right")
ax[2].grid(True, which="both", alpha=0.25, ls="--")
ax[2].set_box_aspect(1.0)
for s in ("top", "right"):
    ax[2].spines[s].set_visible(False)

fig.tight_layout(w_pad=2.2)
fig.savefig(Path(__file__).parents[2] / "assets" / "adaptive_l_shape.png")
print("saved figure -> assets/adaptive_l_shape.png")
