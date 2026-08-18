# --8<-- [start:code]
r"""**h**, **r** and **p** adaptivity side by side — and the rule for choosing between them.

jNO adapts a discretisation three ways, all through the same ``fem.solve(adapt=...)`` slot:

* **h** — ``jno.solve.remesh()`` / ``refine()``: *add* elements where the error is.
* **r** — ``jno.solve.relocate()``: *move* a fixed node set down a differentiable objective.
* **p** — ``jno.solve.enrich()``: *raise the local order*, leaving the mesh alone, by switching
  interpolation covers on at the marked nodes (``space="cover"``).

Which one wins is decided by the **regularity of the solution**, so this tutorial runs all three on two
problems that answer differently:

    Part 1  the L-shape re-entrant corner,  u = r^(2/3) sin(2 phi / 3)     -- a SINGULARITY
    Part 2  smooth structures on a plate: a wave packet, a dome, a spike   -- SMOOTH, but localized

Part 1 is h's home ground: ``r^(2/3)`` is not in ``H^2``, and raising the polynomial order buys its rate
from smoothness the solution does not have. Part 2 is p's: everything there is analytic, the demand is
purely *local resolution*, and order is far cheaper than nodes.

Everything is measured with ONE functional, ``E = 1/2 integral |grad u_h|^2``, read straight off the
assembled form (``fem.eval``) so it means the same thing for P1, for a relocated mesh and for an enriched
space. A geometric formula on the vertex values -- the obvious way to write it -- is blind to the cover
coefficients: they change ``u_h`` BETWEEN nodes and barely move the nodal values, so measured that way
p-adaptivity appears to do nothing at all.

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
from jno.utils.solver.fem_cover import cover_block  # noqa: E402

L_SHAPE = [(0, 0), (1.0, 0), (1.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0, 1.0)]  # reentrant corner at (0.5, 0.5)
MARGIN = 0.025  # keep the movable interior nodes off the boundary (incl. the two notch edges)
BLK = cover_block(2)  # 3 in 2-D: a cover node carries its value plus two cover coefficients


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


def dense_solve(a, b):
    """A direct solve, handed to ``adapt=`` as ``solve_fn``.

    ``adapt=`` does not take the ``linear=`` slot on a steady problem, so a non-default linear solver has
    to arrive positionally. These systems are small; an enriched one is also the least well-conditioned
    of the three, which is the other reason not to leave it to the matrix-free default."""
    return jnp.linalg.solve(jnp.asarray(a.todense() if hasattr(a, "todense") else a), jnp.asarray(b).reshape(-1))


def sparse_solve(a, b):
    """Part 2's systems reach ~24k DOFs, where a dense factorisation is ~4.6 GB. Sparse-direct instead."""
    from jno.utils.solver.linear import sparse_lu_solve

    return sparse_lu_solve(a, b)


def energy(fem, stiff, sol):
    """``E = 1/2 integral |grad u_h|^2``, from the assembled form: ``1/2 u . (A u)``.

    Space-agnostic ON PURPOSE. ``fem.eval`` integrates the weak form with whatever basis the field
    actually carries, so P1, a relocated mesh and an enriched space are all measured by the same rule.
    """
    return 0.5 * float(np.dot(np.asarray(sol).reshape(-1), np.asarray(fem.eval(stiff, sol)).reshape(-1)))


def _min_detj(pts, cells):
    v = pts[cells]
    a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    return float(np.min(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))


def active_dofs(d, fem):
    """DOFs the solve actually carries: an unenriched cover node has its slots PINNED, so the padded size
    ``(1+dim) * n_nodes`` is the same every round and would hide the whole effect of p."""
    pins = getattr(d, "_fem_native_dirichlet_pairs", None) or []
    return int(fem.dofs) - len({int(i) for i, _ in pins})


# =========================================================================================================
# Part 1 -- a SINGULARITY: the L-shape re-entrant corner.  u = r^(2/3) is not in H^2.
# =========================================================================================================


def build_l(size, space="Lagrange", movable=False):
    d = jno.Shape.polygon(L_SHAPE, size=size).domain()
    u, phi = d.fem_symbols(space=space)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    if movable:
        # `.trainable()` on a spatial coordinate turns that region's mesh VERTICES into a design variable:
        # the assembler routes them into the element geometry, so `fem.solve()` becomes differentiable in
        # the node positions. Literal, per component. Must be called BEFORE jno.fem.
        xm, ym, _ = d.variable("mov", where=_interior, split=True)
        xm.trainable(name="ix")
        ym.trainable(name="iy")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    stiff = ui.x * vi.x + ui.y * vi.y
    fem = jno.fem([stiff, u(xb, yb) - u_singular(xb, yb, J, _mod)])
    return d, fem, stiff


# Reference energy from a fine mesh. With Dirichlet data and no source the Galerkin solution MINIMISES E
# over the admissible set, so E_h >= E_exact and the error E_h - E_ref falls as the space improves.
d_ref, fem_ref, stiff_ref = build_l(0.03)
E_REF = energy(fem_ref, stiff_ref, dense_solve(fem_ref.A, fem_ref.b))

d0, fem0, stiff0 = build_l(0.12)
n0 = len(d0.mesh.points)
E0 = energy(fem0, stiff0, dense_solve(fem0.A, fem0.b))

# (1) h -- ADD elements. One call runs solve -> ZZ estimate -> Dorfler mark -> local remesh.
d_h, fem_h, stiff_h = build_l(0.12)
sol_h = np.asarray(fem_h.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=4, refine_factor=1.7))).reshape(-1)
pts_h, tris_h = np.asarray(d_h.mesh.points)[:, :2], np.asarray(d_h.mesh.cells_dict["triangle"])
E_h, n_h = energy(fem_h, stiff_h, sol_h), len(sol_h)

# (2) r -- RELOCATE a fixed node set. `objective="energy"` descends the FE energy, which is what is
#     measured here: for a Ritz method E_h - E_exact = 1/2 ||u - u_h||_E^2, so minimising the energy at
#     fixed DOFs IS minimising the energy-norm error. That is the right objective for a FIXED
#     singularity; the default ("equidistribution") wins instead on a moving or under-resolved front.
d_r, fem_r, stiff_r = build_l(0.12, movable=True)
pts_r0 = np.asarray(d_r.mesh.points)[:, :2].copy()  # coarse start, for the animation
sol_r = np.asarray(fem_r.solve(adapt=jno.solve.relocate(objective="energy", max_iters=60, lr=3e-3))).reshape(-1)
pts_r, tris_r = np.asarray(d_r.mesh.points)[:, :2], np.asarray(d_r.mesh.cells_dict["triangle"])
E_r = energy(fem_r, stiff_r, sol_r)

# (3) p -- RAISE THE ORDER on the same mesh, and here it is the WRONG CHOICE. Two reasons, both
#     structural rather than incidental:
#       * u = r^(2/3) is not in H^2. A higher-order space buys its rate from smoothness the solution
#         lacks, so p converges slowly at a corner while h keeps its rate.
#       * the Dirichlet data here is INHOMOGENEOUS, and a cover field's trace is only the P1 interpolant
#         of g (the tangential covers pin to zero, not to dg/ds) -- a documented scope limit.
#     It is in the tutorial anyway: a method's failing regime is half of knowing when to reach for it.
d_p, fem_p, stiff_p = build_l(0.12, space="cover")
fem_p.solve(dense_solve, adapt=jno.solve.enrich(theta=0.7, max_iters=5))
# An adapt= driver RETURNS the vertex view -- the nodal values. For P1 that is the whole coefficient
# vector, but a cover field also carries the cover coefficients, so a functional computed from the
# returned array would see only the P1 part. Re-solve on the final space instead: the loop leaves `fem`
# bound to it, so this is one solve on the adapted system, not a rebuild.
sol_p = dense_solve(fem_p.A, fem_p.b)
E_p = energy(fem_p, stiff_p, sol_p)
n_p, frac_p = active_dofs(d_p, fem_p), float(np.asarray(d_p._fem_enriched_nodes).mean())

print(f"PART 1 -- L-shape singularity.   energy-norm error (E - E_ref),  E_ref = {E_REF:.4f}")
print(f"  coarse start   : {E0 - E_REF:.3e}   ({n0} dofs)")
print(
    f"  h-adaptivity   : {E_h - E_REF:.3e}   ({n_h} dofs, +{n_h - n0})   {100 * (1 - (E_h - E_REF) / (E0 - E_REF)):.0f}% lower"
)
print(f"  r-adaptivity   : {E_r - E_REF:.3e}   ({n0} dofs, +0)   {100 * (1 - (E_r - E_REF) / (E0 - E_REF)):.0f}% lower")
print(
    f"  p-adaptivity   : {E_p - E_REF:.3e}   ({n_p} dofs, {frac_p:.0%} enriched)   "
    f"{100 * (1 - (E_p - E_REF) / (E0 - E_REF)):.0f}% lower   <- the wrong tool here"
)

assert n_h > n0, "h-adaptivity should add DOFs at the corner"
assert E_h - E_REF < 0.4 * (E0 - E_REF), "h-adaptivity should sharply cut the energy error"
assert E_r - E_REF < 0.75 * (E0 - E_REF), "r-adaptivity should cut the energy error at fixed DOFs"
assert _min_detj(pts_r, tris_r) > 0, "r-adaptivity must not tangle the mesh"

# =========================================================================================================
# Part 2 -- SMOOTH but localized: a wave packet, a broad dome and a narrow spike.
# =========================================================================================================
# Each structure is C^3 and compactly supported inside its own disc, so u and grad u vanish IDENTICALLY on
# the wall: the homogeneous condition is exact, and the covers pin to zero exactly -- which keeps Part 1's
# inhomogeneous-trace limitation out of this measurement.
#
#     g(s) = (1-s)^4,   s = |x-c|^2 / R^2,   supported on s < 1
REGIONS = (  # (cx, cy, R, amplitude, omega)   omega = 0 is a plain dome
    (0.30, 0.68, 0.26, 0.70, 5.0 * np.pi),  # packet: oscillatory -> the highest demand
    (0.72, 0.30, 0.26, 1.00, 0.0),  # dome:   the LARGEST amplitude, almost no demand
    (0.72, 0.72, 0.10, 0.55, 0.0),  # spike:  small, extreme curvature
)


def _u_regions(x, y, xp):
    out = 0.0
    for cx, cy, R, A, w in REGIONS:
        s = ((x - cx) ** 2 + (y - cy) ** 2) / R**2
        g = xp.where(s < 1.0, 1.0 - s, 0.0) ** 4
        out = out + A * (g * xp.sin(w * (x - cx) / R) if w else g)
    return out


def source_regions(x, y, xp):
    """``f = -Laplace(u)``. For ``g(s)``: ``Lap g = g'' |grad s|^2 + g' Lap s`` with ``|grad s|^2 = 4s/R^2``
    and ``Lap s = 4/R^2``; the oscillation adds ``2 grad g . grad sin`` and ``-g (w/R)^2 sin``."""
    f = 0.0
    for cx, cy, R, A, w in REGIONS:
        s = ((x - cx) ** 2 + (y - cy) ** 2) / R**2
        one_s = xp.where(s < 1.0, 1.0 - s, 0.0)
        g, lap_g = one_s**4, (16.0 / R**2) * (3.0 * s * one_s**2 - one_s**3)
        if w:
            xi = w * (x - cx) / R
            dgdx = -8.0 * one_s**3 * (x - cx) / R**2
            lap = lap_g * xp.sin(xi) + 2.0 * dgdx * (w / R) * xp.cos(xi) - g * (w / R) ** 2 * xp.sin(xi)
        else:
            lap = lap_g
        f = f - A * lap
    return f


def build_p(size, space="Lagrange", movable=False):
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    tol = 1e-9
    d.tag("walls", lambda *c: np.logical_or.reduce([(x < tol) | (x > 1 - tol) for x in c]))
    if movable:
        xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.03) & (x < 0.97) & (y > 0.03) & (y < 0.97), split=True)
        xm.trainable(name="ix")
        ym.trainable(name="iy")
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    u, phi = d.fem_symbols(space=space)
    xi, yi = co[0], co[1]
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    stiff = ui.x * vi.x + ui.y * vi.y
    fem = jno.fem([stiff - source_regions(xi, yi, jno.np) * vi, u(cw[0], cw[1]) - 0.0])
    return d, fem, stiff


# With a SOURCE and homogeneous data the Galerkin solution minimises J = 1/2 a(v,v) - (f,v), and at the
# solution J_h = -E_h. So the energy RISES toward the truth here and the error is E_ref - E_h: the same
# functional as Part 1, read with the opposite sign.
H2 = 0.02
# The reference must be MORE ACCURATE than everything measured against it, and a P1 one is not here: the
# enriched run beats a 17.8k-DOF P1 reference outright, which showed up as a NEGATIVE error. A fully
# enriched (cover) reference is third order and converges far faster per node. Measured drift between
# h=0.016 and h=0.012 references is 6.7e-3 -- about a third of the smallest error below, which is the
# accuracy this comparison is quoted to.
d2ref, fem2ref, stiff2ref = build_p(0.012, space="cover")
E2_REF = energy(fem2ref, stiff2ref, sparse_solve(fem2ref.A, fem2ref.b))

d2_0, fem2_0, stiff2_0 = build_p(H2)
n2_0 = active_dofs(d2_0, fem2_0)
E2_0 = energy(fem2_0, stiff2_0, sparse_solve(fem2_0.A, fem2_0.b))

d2_h, fem2_h, stiff2_h = build_p(H2)
sol2_h = np.asarray(fem2_h.solve(sparse_solve, adapt=jno.solve.remesh(theta=0.6, max_iters=3, refine_factor=1.6))).reshape(
    -1
)
E2_h, n2_h = energy(fem2_h, stiff2_h, sol2_h), active_dofs(d2_h, fem2_h)

# r on THIS problem makes the answer worse, with either objective (energy 12.23 -> 9.87,
# equidistribution -> 7.13), and it is reported rather than quietly dropped. What is NOT established is
# WHY: relocation is known to write back a mesh that differs from the one its loop validated
# (FhG-IISB/jNO#114), so this row may be measuring that defect rather than the method. Read it as
# "unresolved", not as "r-adaptivity does not work on smooth problems".
d2_r, fem2_r, stiff2_r = build_p(H2, movable=True)
sol2_r = np.asarray(
    fem2_r.solve(sparse_solve, adapt=jno.solve.relocate(objective="energy", max_iters=40, lr=4e-3))
).reshape(-1)
E2_r, n2_r = energy(fem2_r, stiff2_r, sol2_r), active_dofs(d2_r, fem2_r)

d2_p, fem2_p, stiff2_p = build_p(H2, space="cover")
fem2_p.solve(sparse_solve, adapt=jno.solve.enrich(theta=0.8, max_iters=5))
sol2_p = sparse_solve(fem2_p.A, fem2_p.b)  # the full coefficient vector; see the note in Part 1
E2_p, n2_p = energy(fem2_p, stiff2_p, sol2_p), active_dofs(d2_p, fem2_p)
mask2 = np.asarray(d2_p._fem_enriched_nodes, dtype=bool)

print(f"\nPART 2 -- smooth, localized.     energy-norm error (E_ref - E),  E_ref = {E2_REF:.4f}")
print(f"  coarse start   : {E2_REF - E2_0:.3e}   ({n2_0} dofs)")
print(f"  h-adaptivity   : {E2_REF - E2_h:.3e}   ({n2_h} dofs)")
print(f"  r-adaptivity   : {E2_REF - E2_r:.3e}   ({n2_r} dofs, +0)   <- worse than the start; see jNO#114")
print(f"  p-adaptivity   : {E2_REF - E2_p:.3e}   ({n2_p} dofs, {mask2.mean():.0%} enriched)   <- p's regime")

assert E2_REF - E2_p > 0, "the reference must bound the enriched solve -- if not, it is not a reference"
assert E2_REF - E2_p < 0.25 * (E2_REF - E2_0), "p-adaptivity should sharply cut the energy error here"
assert E2_REF - E2_p < E2_REF - E2_h, "p should beat h on a smooth, localized problem"
assert 0.0 < mask2.mean() < 1.0, "the p run enriched everything or nothing -- nothing was chosen"
# --8<-- [end:code]

# --- figure: the three mechanisms, on both regimes -------------------------------------------------------
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
fig, ax = plt.subplots(2, 3, figsize=(13.0, 8.2))


def mesh_panel(a, pts, tris, title, mark=True):
    a.triplot(Triangulation(pts[:, 0], pts[:, 1], tris), color=INK, lw=0.4, alpha=0.7)
    if mark:
        a.plot([0.5], [0.5], "x", color=TEAL, ms=9, mew=2.2)  # the reentrant corner
    a.set_aspect("equal")
    a.set_axis_off()
    a.set_title(title)


mesh_panel(ax[0, 0], pts_h, tris_h, f"h  ·  add elements  ·  {n_h} dofs")
mesh_panel(ax[0, 1], pts_r, tris_r, f"r  ·  relocate  ·  {n0} dofs")

e0, eh, er, ep = E0 - E_REF, E_h - E_REF, E_r - E_REF, E_p - E_REF
ax[0, 2].plot([n0, n_h], [e0, eh], "o-", color=BLUE, lw=1.4, label="h  (+DOFs)")
ax[0, 2].plot([n0, n0], [e0, er], "o-", color=TEAL, lw=1.4, label="r  (fixed DOFs)")
ax[0, 2].plot([n0, n_p], [e0, ep], "o-", color=RUST, lw=1.4, label="p  (wrong regime)")
ax[0, 2].annotate("coarse start", (n0, e0), (n0 + 6, e0 * 1.2), color=GRAY, fontsize=9)
ax[0, 2].set_yscale("log")
ax[0, 2].set_xlabel("degrees of freedom")
ax[0, 2].set_ylabel(r"energy-norm error  $E - E_{\mathrm{ref}}$")
ax[0, 2].set_title("Part 1 · singularity")
ax[0, 2].legend(frameon=False, fontsize=9, loc="upper right")
ax[0, 2].grid(True, which="both", alpha=0.25, ls="--")
ax[0, 2].set_box_aspect(1.0)

g = np.linspace(0, 1, 400)
gx, gy = np.meshgrid(g, g)
ax[1, 0].imshow(_u_regions(gx, gy, np), cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", extent=(0, 1, 0, 1))
ax[1, 0].set_axis_off()
ax[1, 0].set_title(r"$u$  ·  packet, dome, spike")

pts2 = np.asarray(d2_p.mesh.points)[:, :2]
ax[1, 1].scatter(pts2[~mask2, 0], pts2[~mask2, 1], s=0.7, color=GRAY, alpha=0.3, linewidths=0)
ax[1, 1].scatter(pts2[mask2, 0], pts2[mask2, 1], s=2.2, color=TEAL, linewidths=0)
ax[1, 1].set_aspect("equal")
ax[1, 1].set_axis_off()
ax[1, 1].set_title(f"p  ·  enriched nodes  ·  {mask2.mean():.0%}")

f0, fh, fr, fp = E2_REF - E2_0, E2_REF - E2_h, E2_REF - E2_r, E2_REF - E2_p
ax[1, 2].plot([n2_0, n2_h], [f0, fh], "o-", color=BLUE, lw=1.4, label="h  (+DOFs)")
ax[1, 2].plot([n2_0, n2_r], [f0, fr], "o-", color=TEAL, lw=1.4, label="r  (fixed DOFs)")
ax[1, 2].plot([n2_0, n2_p], [f0, fp], "o-", color=RUST, lw=1.4, label="p  (right regime)")
ax[1, 2].set_yscale("log")
ax[1, 2].set_xscale("log")
ax[1, 2].set_xlabel("active degrees of freedom")
ax[1, 2].set_ylabel(r"energy-norm error  $E_{\mathrm{ref}} - E$")
ax[1, 2].set_title("Part 2 · smooth, localized")
ax[1, 2].legend(frameon=False, fontsize=9, loc="upper right")
ax[1, 2].grid(True, which="both", alpha=0.25, ls="--")
ax[1, 2].set_box_aspect(1.0)

for s in ("top", "right"):
    ax[0, 2].spines[s].set_visible(False)
    ax[1, 2].spines[s].set_visible(False)
fig.tight_layout(w_pad=2.2, h_pad=2.0)
fig.savefig(Path(__file__).parents[2] / "assets" / "adaptive_l_shape.png")
print("saved figure -> assets/adaptive_l_shape.png")

# --- animation: the three mechanisms, side by side (a GIF) -----------------------------------------------
# All three drivers record their state per round in `fem.adapt_history`: h grows the mesh, r moves the
# points, p leaves both alone and switches nodes on. The animation shows that difference faithfully.
import matplotlib.animation as animation  # noqa: E402

h_seq = [(h["points"], h["cells"]) for h in fem_h.adapt_history] + [(pts_h, tris_h)]
r_seq = [pts_r0] + [h["points"] for h in fem_r.adapt_history]
p_seq = [np.asarray(h["enriched"], dtype=bool) for h in fem2_p.adapt_history] + [mask2]
p_tris = np.asarray(d2_p.mesh.cells_dict["triangle"])
N_FRAMES = 40
figA, axes3 = plt.subplots(1, 3, figsize=(12.6, 4.5))


def _frame(i):
    frac = i / (N_FRAMES - 1)
    hp, hc = h_seq[round(frac * (len(h_seq) - 1))]
    rp = r_seq[round(frac * (len(r_seq) - 1))]
    pm = p_seq[round(frac * (len(p_seq) - 1))]
    for a in axes3:
        a.clear()
        a.set_aspect("equal")
        a.set_axis_off()
        a.set_xlim(-0.03, 1.03)
        a.set_ylim(-0.03, 1.03)
    for a in axes3[:2]:
        a.plot([0.5], [0.5], "x", color=TEAL, ms=8, mew=2.0)
    axes3[0].triplot(Triangulation(hp[:, 0], hp[:, 1], hc), color=INK, lw=0.4, alpha=0.75)
    axes3[0].set_title(f"h — add elements  ·  {len(hp)} dofs")
    axes3[1].triplot(Triangulation(rp[:, 0], rp[:, 1], tris_r), color=TEAL, lw=0.5, alpha=0.85)
    axes3[1].set_title(f"r — relocate  ·  {len(rp)} dofs (fixed)")
    axes3[2].triplot(Triangulation(pts2[:, 0], pts2[:, 1], p_tris), color=GRAY, lw=0.25, alpha=0.35)
    axes3[2].scatter(pts2[pm, 0], pts2[pm, 1], s=2.0, color=RUST, linewidths=0)
    axes3[2].set_title(f"p — raise the order  ·  {pm.mean():.0%} enriched")
    return []


figA.tight_layout(w_pad=1.5)
gif_path = Path(__file__).parents[2] / "assets" / "adaptive_l_shape.gif"
animation.FuncAnimation(figA, _frame, frames=N_FRAMES, interval=100).save(
    gif_path, writer=animation.PillowWriter(fps=10), dpi=100
)
plt.close(figA)
print("saved animation -> assets/adaptive_l_shape.gif")
