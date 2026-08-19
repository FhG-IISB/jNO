# --8<-- [start:code]
r"""**h**, **r** and **p** adaptivity on ONE problem — same mesh, same reference, same functional.

jNO adapts a discretisation three ways, all through the same ``fem.solve(adapt=...)`` slot:

* **h** — ``jno.solve.remesh()`` / ``refine()``: *add* elements where the error is.
* **r** — ``jno.solve.relocate()``: *move* a fixed node set down a differentiable objective.
* **p** — ``jno.solve.enrich()``: *raise the local order*, leaving the mesh alone, by switching
  interpolation covers on at the marked nodes (``space="cover"``).

They are measured against each other here, so they all run on the SAME problem: an L-shape driven by a
compact smooth source,

    -lap u = f  on the L-shape,   u = 0 on the whole boundary,
    f = a C^3 bump supported in the lower-right arm,

which carries **two** features of different character at once -- the re-entrant corner's ``r^(2/3)``
singularity, and a smooth localized bump. The boundary condition is homogeneous on purpose: a cover
field's trace is only the P1 interpolant of an INHOMOGENEOUS ``g``, so ``u = 0`` keeps that documented
scope limit out of a comparison it would otherwise distort.

Everything is measured with one functional, ``E = 1/2 integral |grad u_h|^2``, read straight off the
assembled form with ``fem.eval``, so it means the same thing for P1, for a relocated mesh and for an
enriched space. The geometric alternative -- a formula on the vertex values -- is blind to p: a cover's
coefficients are the local gradient, so they move ``u_h`` BETWEEN nodes and barely move the nodal
values. Measured that way, enriching 5% of the nodes and enriching all of them are indistinguishable.

With a source and homogeneous data the Galerkin solution minimises ``J = 1/2 a(v,v) - (f,v)`` and
``J_h = -E_h``, so the energy RISES toward the truth and the error is ``E_ref - E_h >= 0``.

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
from jno.utils.solver.fem_cover import cover_block  # noqa: E402

L_SHAPE = [(0, 0), (1.0, 0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0, 1.0)]  # reentrant corner at (0.5, 0.5)
BUMP = (0.74, 0.22, 0.17, 60.0)  # cx, cy, R, amplitude -- the source, compactly supported inside the arm
CORNER = (0.5, 0.5)
H = 0.075  # the common coarse mesh every method starts from
MARGIN = 0.04
BLK = cover_block(2)  # 3 in 2-D: a cover node carries its value plus two cover coefficients


def source(x, y, xp):
    """A C^3 bump, ``(1-s)^4`` with ``s = |x-c|^2/R^2``, supported strictly inside the lower-right arm."""
    cx, cy, R, A = BUMP
    s = ((x - cx) ** 2 + (y - cy) ** 2) / R**2
    return A * xp.where(s < 1.0, 1.0 - s, 0.0) ** 4


def _movable(x, y):
    """Interior L-shape vertices: inside the material, off the outer box and the two notch edges."""
    inside = ~((x > 0.5) & (y > 0.5))
    off_box = (x > MARGIN) & (x < 1 - MARGIN) & (y > MARGIN) & (y < 1 - MARGIN)
    off_notch = ~(((jnp.abs(x - 0.5) < MARGIN) & (y > 0.5 - MARGIN)) | ((jnp.abs(y - 0.5) < MARGIN) & (x > 0.5 - MARGIN)))
    return inside & off_box & off_notch


def build(size, space="Lagrange", movable=False):
    d = jno.Shape.polygon(L_SHAPE, size=size).domain()
    if movable:
        # `.trainable()` on a spatial coordinate turns that region's mesh VERTICES into a design
        # variable: the assembler routes them into the element geometry, so `fem.solve()` becomes
        # differentiable in the node positions. Literal, per component. Must be called BEFORE jno.fem.
        xm, ym, _ = d.variable("mov", where=_movable, split=True)
        xm.trainable(name="ix")
        ym.trainable(name="iy")
    u, phi = d.fem_symbols(space=space)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    stiff = ui.x * vi.x + ui.y * vi.y
    fem = jno.fem([stiff - source(xi, yi, jno.np) * vi, u(xb, yb) - 0.0])
    return d, fem, stiff


def solve_fn(a, b):
    """Sparse-direct, handed to ``adapt=`` positionally.

    ``adapt=`` does not take the ``linear=`` slot on a steady problem, so a non-default linear solver
    has to arrive as ``solve_fn``. The reference reaches ~12k DOFs, where a dense factorisation is a
    gigabyte; an enriched system is also the least well-conditioned of the three."""
    from jno.utils.solver.linear import sparse_lu_solve

    return sparse_lu_solve(a, b)


def energy(fem, stiff, sol):
    """``E = 1/2 integral |grad u_h|^2``, from the assembled form: ``1/2 u . (A u)``.

    Space-agnostic ON PURPOSE -- ``fem.eval`` integrates the weak form with whatever basis the field
    actually carries, so P1, a relocated mesh and an enriched space are measured by the same rule."""
    return 0.5 * float(np.dot(np.asarray(sol).reshape(-1), np.asarray(fem.eval(stiff, sol)).reshape(-1)))


def active_dofs(d, fem):
    """DOFs the solve actually carries. An unenriched cover node has its slots PINNED, so the padded
    size ``(1+dim) * n_nodes`` is the same every round and would hide the whole effect of p."""
    pins = getattr(d, "_fem_native_dirichlet_pairs", None) or []
    return int(fem.dofs) - len({int(i) for i, _ in pins})


def _min_detj(pts, cells):
    v = pts[cells]
    a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    return float(np.min(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))


def _near(pts, centre, radius):
    return np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1]) < radius


def _min_angle(pts, cells):
    """Smallest angle in the mesh, in degrees -- the quality number relocation can quietly destroy."""
    v = pts[cells]
    a = np.linalg.norm(v[:, 1] - v[:, 0], axis=1)
    b = np.linalg.norm(v[:, 2] - v[:, 1], axis=1)
    c = np.linalg.norm(v[:, 0] - v[:, 2], axis=1)
    cos = np.stack(
        [
            (b * b + c * c - a * a) / (2 * b * c),
            (a * a + c * c - b * b) / (2 * a * c),
            (a * a + b * b - c * c) / (2 * a * b),
        ],
        axis=1,
    )
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).min())


# --- the reference, and the common coarse start ----------------------------------------------------------
# The reference must OUT-RESOLVE everything measured against it. A P1 reference is not good enough for
# this comparison: an enriched run beats a P1 reference of several times the DOF count outright, which
# shows up as a negative error. A fully enriched (cover) reference is third order instead. The drift from
# h=0.020 to h=0.015 is 4.8e-06 -- well under the smallest error below, which is what this is quoted to.
d_ref, fem_ref, stiff_ref = build(0.015, space="cover")
E_REF = energy(fem_ref, stiff_ref, solve_fn(fem_ref.A, fem_ref.b))

d0, fem0, stiff0 = build(H)
E0, n0 = energy(fem0, stiff0, solve_fn(fem0.A, fem0.b)), active_dofs(d0, fem0)
pts0, tris0 = np.asarray(d0.mesh.points)[:, :2], np.asarray(d0.mesh.cells_dict["triangle"])

# (1) h -- ADD elements. One call runs solve -> ZZ estimate -> Dorfler mark -> local remesh.
d_h, fem_h, stiff_h = build(H)
fem_h.solve(solve_fn, adapt=jno.solve.remesh(theta=0.6, max_iters=4, refine_factor=1.7))
E_h, n_h = energy(fem_h, stiff_h, solve_fn(fem_h.A, fem_h.b)), active_dofs(d_h, fem_h)
pts_h, tris_h = np.asarray(d_h.mesh.points)[:, :2], np.asarray(d_h.mesh.cells_dict["triangle"])

# (2) r -- RELOCATE a fixed node set, down a mesh functional, through the differentiable solve.
#     NOT `objective="energy"` here, and the reason is this problem's source term. The Ritz functional
#     is J(v) = 1/2 a(v,v) - (f,v); with no load J = E and descending the energy descends the error,
#     which is what the classic L-shape wants. With a load, J_h = -E_h at the discrete solution, so
#     minimising the error means MAXIMISING E -- and descending it walks away from the solution while
#     flattening elements, which is the cheapest way to lower the integral of |grad u|^2. Measured
#     here: E fell 0.12252 -> 0.10788 exactly as asked, the true error rose 3.6x, and the smallest
#     angle in the mesh collapsed from 40.8 degrees to 3.2. The default (arclength equidistribution)
#     targets resolution instead and keeps the mesh sane.
d_r, fem_r, stiff_r = build(H, movable=True)
pts_r0 = np.asarray(d_r.mesh.points)[:, :2].copy()  # coarse start, for the animation
sol_r = np.asarray(fem_r.solve(solve_fn, adapt=jno.solve.relocate(max_iters=30, lr=5e-4))).reshape(-1)
E_r, n_r = energy(fem_r, stiff_r, sol_r), active_dofs(d_r, fem_r)
pts_r, tris_r = np.asarray(d_r.mesh.points)[:, :2], np.asarray(d_r.mesh.cells_dict["triangle"])

# (3) p -- RAISE THE ORDER on the same mesh. The loop returns the VERTEX VIEW, which for a cover field
#     is only the value slots, so re-solve on the final space for the full coefficient vector (the loop
#     leaves `fem` bound to it -- one solve, not a rebuild).
d_p, fem_p, stiff_p = build(H, space="cover")
fem_p.solve(solve_fn, adapt=jno.solve.enrich(theta=0.8, max_iters=5))
E_p, n_p = energy(fem_p, stiff_p, solve_fn(fem_p.A, fem_p.b)), active_dofs(d_p, fem_p)
mask_p = np.asarray(d_p._fem_enriched_nodes, dtype=bool)
pts_p = np.asarray(d_p.mesh.points)[:, :2]

# (4) hp -- the two composed across successive solves: refine the mesh, then enrich on it.
#     NOTE a rough edge: `enrich` always starts from plain P1, so calling it on a field that was already
#     uniformly `space="cover"` REMOVES enrichment before adding it back selectively -- the active DOF
#     count goes DOWN at that call. That is the loop's design (there is no "start from what is there"),
#     but it makes h-then-p read oddly unless you know it.
d_hp, fem_hp, stiff_hp = build(H, space="cover")
fem_hp.solve(solve_fn, adapt=jno.solve.remesh(theta=0.6, max_iters=3, refine_factor=1.7))
n_hp_after_h = active_dofs(d_hp, fem_hp)
fem_hp.solve(solve_fn, adapt=jno.solve.enrich(theta=0.8, max_iters=4))
E_hp, n_hp = energy(fem_hp, stiff_hp, solve_fn(fem_hp.A, fem_hp.b)), active_dofs(d_hp, fem_hp)
mask_hp = np.asarray(d_hp._fem_enriched_nodes, dtype=bool)

err = lambda E: E_REF - E  # noqa: E731  -- energy rises toward the truth; see the module docstring
print(f"ONE problem, one mesh, one reference.   E_ref = {E_REF:.6f}   (cover, {active_dofs(d_ref, fem_ref)} dofs)")
print(f"  coarse start   : {err(E0):.3e}   ({n0} dofs)")
print(f"  h-adaptivity   : {err(E_h):.3e}   ({n_h} dofs)   {err(E0) / err(E_h):5.1f}x lower")
print(f"  r-adaptivity   : {err(E_r):.3e}   ({n_r} dofs, +0)   {err(E0) / err(E_r):5.1f}x   <- not r's regime here")
print(f"  p-adaptivity   : {err(E_p):.3e}   ({n_p} dofs, {mask_p.mean():.0%} enriched)   {err(E0) / err(E_p):5.1f}x lower")
print(
    f"  h then p       : {err(E_hp):.3e}   ({n_hp} dofs, {mask_hp.mean():.0%} enriched)   {err(E0) / err(E_hp):5.1f}x lower"
)

# --- where did each method spend its DOFs? ---------------------------------------------------------------
# The two features want different treatment, so this is the interesting half of the comparison.
cx, cy, R, _A = BUMP
near_c, in_b = _near(pts_p, CORNER, 0.22), _near(pts_p, (cx, cy), R)
print("\np put its covers:")
print(
    f"  corner {mask_p[near_c].mean():5.0%} | bump {mask_p[in_b].mean():5.0%} | elsewhere {mask_p[~(near_c | in_b)].mean():5.0%}"
)

cent = pts_h[tris_h].mean(axis=1)
e1, e2 = pts_h[tris_h][:, 1] - pts_h[tris_h][:, 0], pts_h[tris_h][:, 2] - pts_h[tris_h][:, 0]
hsz = np.sqrt(0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]))
nc, ib = _near(cent, CORNER, 0.22), _near(cent, (cx, cy), R)
print("h put its elements (mean cell size, smaller = more refined):")
print(f"  corner {hsz[nc].mean():.4f} | bump {hsz[ib].mean():.4f} | elsewhere {hsz[~(nc | ib)].mean():.4f}")

assert err(E0) > 0 and err(E_p) > 0, "the reference must bound every run -- otherwise it is not a reference"
assert err(E_h) < 0.25 * err(E0), "h-adaptivity should sharply cut the energy error"
assert err(E_p) < err(E_h), "p should beat h per DOF on this smooth-source problem"
assert err(E_hp) < err(E_p), "h then p should beat either alone"
assert 0.0 < mask_p.mean() < 1.0, "the p run enriched everything or nothing -- nothing was chosen"
assert _min_detj(pts_r, tris_r) > 0, "r-adaptivity must not tangle the mesh"
# Not tangling is a low bar -- `objective="energy"` on this source-driven problem passed it while
# flattening the smallest angle from 40.8 degrees to 3.2 (it lowers the integral of |grad u|^2 by
# squashing elements). Assert the mesh stays USABLE, not merely non-inverted.
_ang0, _ang_r = _min_angle(pts0, tris0), _min_angle(pts_r, tris_r)
print(f"\nmesh quality (smallest angle):  start {_ang0:.1f} deg  ->  after relocation {_ang_r:.1f} deg")
assert _ang_r > 0.6 * _ang0, f"relocation wrecked the mesh: {_ang0:.1f} deg -> {_ang_r:.1f} deg"
# --8<-- [end:code]

# --- figure ----------------------------------------------------------------------------------------------
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
TEAL, INK, BLUE, GRAY, RUST = "#0D9488", "#1A202C", "#0072B2", "#94A3B8", "#B45309"
fig, ax = plt.subplots(1, 4, figsize=(17.0, 4.3))

sol0 = np.asarray(solve_fn(fem0.A, fem0.b)).reshape(-1)
tp = ax[0].tripcolor(Triangulation(pts0[:, 0], pts0[:, 1], tris0), sol0, shading="gouraud", cmap="magma")
ax[0].add_patch(plt.Circle((cx, cy), R, fill=False, ec="white", lw=0.8, alpha=0.7))
ax[0].plot([CORNER[0]], [CORNER[1]], "x", color="white", ms=8, mew=2.0)
ax[0].set_aspect("equal")
ax[0].set_axis_off()
ax[0].set_title(r"$u_h$  ·  corner + bump")
cb = fig.colorbar(tp, ax=ax[0], fraction=0.046, pad=0.03)
cb.outline.set_visible(False)
cb.ax.tick_params(length=0)


def mesh_panel(a, pts, tris, title):
    a.triplot(Triangulation(pts[:, 0], pts[:, 1], tris), color=INK, lw=0.4, alpha=0.7)
    a.add_patch(plt.Circle((cx, cy), R, fill=False, ec=BLUE, lw=0.9, alpha=0.8))
    a.plot([CORNER[0]], [CORNER[1]], "x", color=TEAL, ms=9, mew=2.2)
    a.set_aspect("equal")
    a.set_axis_off()
    a.set_title(title)


mesh_panel(ax[1], pts_h, tris_h, f"h  ·  add elements  ·  {n_h} dofs")

ax[2].triplot(
    Triangulation(pts_p[:, 0], pts_p[:, 1], np.asarray(d_p.mesh.cells_dict["triangle"])), color=GRAY, lw=0.3, alpha=0.45
)
ax[2].scatter(pts_p[mask_p, 0], pts_p[mask_p, 1], s=14, color=RUST, linewidths=0)
ax[2].add_patch(plt.Circle((cx, cy), R, fill=False, ec=BLUE, lw=0.9, alpha=0.8))
ax[2].plot([CORNER[0]], [CORNER[1]], "x", color=TEAL, ms=9, mew=2.2)
ax[2].set_aspect("equal")
ax[2].set_axis_off()
ax[2].set_title(f"p  ·  enrich  ·  {n_p} dofs ({mask_p.mean():.0%})")

ax[3].plot([n0, n_h], [err(E0), err(E_h)], "o-", color=BLUE, lw=1.5, label="h  (+DOFs)")
ax[3].plot([n0, n_r], [err(E0), err(E_r)], "o-", color=TEAL, lw=1.5, label="r  (fixed DOFs)")
ax[3].plot([n0, n_p], [err(E0), err(E_p)], "o-", color=RUST, lw=1.5, label="p  (same mesh)")
ax[3].plot([n_p, n_hp], [err(E_p), err(E_hp)], "o--", color=INK, lw=1.3, label="h then p")
ax[3].set_yscale("log")
ax[3].set_xscale("log")
ax[3].set_xlabel("active degrees of freedom")
ax[3].set_ylabel(r"energy-norm error  $E_{\mathrm{ref}} - E$")
ax[3].set_title("one problem, four runs")
ax[3].legend(frameon=False, fontsize=9, loc="lower left")
ax[3].grid(True, which="both", alpha=0.25, ls="--")
ax[3].set_box_aspect(1.0)
for s in ("top", "right"):
    ax[3].spines[s].set_visible(False)

fig.tight_layout(w_pad=2.0)
fig.savefig(Path(__file__).parents[2] / "assets" / "adaptive_l_shape.png")
print("\nsaved figure -> assets/adaptive_l_shape.png")

# --- animation: the three mechanisms on the SAME problem --------------------------------------------------
# All three record their state per round in `fem.adapt_history`: h grows the mesh, r moves the points,
# p leaves both alone and switches nodes on.
import matplotlib.animation as animation  # noqa: E402

h_seq = [(hh["points"], hh["cells"]) for hh in fem_h.adapt_history] + [(pts_h, tris_h)]
r_seq = [pts_r0] + [hh["points"] for hh in fem_r.adapt_history]
p_seq = [np.asarray(hh["enriched"], dtype=bool) for hh in fem_p.adapt_history] + [mask_p]
p_tris = np.asarray(d_p.mesh.cells_dict["triangle"])
N_FRAMES = 40
figA, axes3 = plt.subplots(1, 3, figsize=(12.6, 4.5))


def _frame(i):
    frac = i / (N_FRAMES - 1)
    hp_, hc_ = h_seq[round(frac * (len(h_seq) - 1))]
    rp_ = r_seq[round(frac * (len(r_seq) - 1))]
    pm_ = p_seq[round(frac * (len(p_seq) - 1))]
    for a in axes3:
        a.clear()
        a.set_aspect("equal")
        a.set_axis_off()
        a.set_xlim(-0.03, 1.03)
        a.set_ylim(-0.03, 1.03)
        a.plot([CORNER[0]], [CORNER[1]], "x", color=TEAL, ms=8, mew=2.0)
        a.add_patch(plt.Circle((cx, cy), R, fill=False, ec=BLUE, lw=0.8, alpha=0.6))
    axes3[0].triplot(Triangulation(hp_[:, 0], hp_[:, 1], hc_), color=INK, lw=0.4, alpha=0.75)
    axes3[0].set_title(f"h — add elements  ·  {len(hp_)} nodes")
    axes3[1].triplot(Triangulation(rp_[:, 0], rp_[:, 1], tris_r), color=TEAL, lw=0.5, alpha=0.85)
    axes3[1].set_title(f"r — relocate  ·  {len(rp_)} nodes (fixed)")
    axes3[2].triplot(Triangulation(pts_p[:, 0], pts_p[:, 1], p_tris), color=GRAY, lw=0.3, alpha=0.4)
    axes3[2].scatter(pts_p[pm_, 0], pts_p[pm_, 1], s=12, color=RUST, linewidths=0)
    axes3[2].set_title(f"p — raise the order  ·  {pm_.mean():.0%} enriched")
    return []


figA.tight_layout(w_pad=1.5)
gif_path = Path(__file__).parents[2] / "assets" / "adaptive_l_shape.gif"
animation.FuncAnimation(figA, _frame, frames=N_FRAMES, interval=100).save(
    gif_path, writer=animation.PillowWriter(fps=10), dpi=100
)
plt.close(figA)
print("saved animation -> assets/adaptive_l_shape.gif")
