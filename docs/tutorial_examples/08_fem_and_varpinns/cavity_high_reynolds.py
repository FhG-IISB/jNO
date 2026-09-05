# --8<-- [start:code]
"""**How far the stabilised pair goes** -- the lid-driven cavity from Re = 100 to Re = 5000, on
equal-order P1/P1, reached by continuation in the Reynolds number.

    (u.grad)u - nu lap u + grad p = 0,   div u = 0,   Re = U L / nu

Two questions this answers, both of which the stabilised-flow tutorial leaves open:

  1. **How high can Re go?** That tutorial verifies the formulation against a closed form at Re = 20.
     Stabilisation exists for convection-dominated flow, so Re = 20 is not an envelope.
  2. **Is the equal-order answer right?** P1/P1 is not an inf-sup-stable pair -- it works only because
     PSPG compensates. The check is Taylor-Hood P2/P1, the stable pair, on the same mesh.

Two things carry the run:

**A regularised lid.** The classical cavity drives the lid at a constant speed, which puts a
discontinuity -- and a pressure singularity -- in each top corner. Newton from rest does not survive
it: it fails here at Re = 100 already. `16 x^2 (1-x)^2` is the standard regularisation and is what the
other cavity tutorial uses too.

**Continuation in Re.** Even regularised, a cold Newton solve at Re = 1000 drives the iterate
somewhere its own tangent is singular. `nu` is a `jno.np.parameter`, so `fem.solve(continuation=...)`
sweeps it and warm-starts each solve from the last -- which is the textbook way to reach a high
Reynolds number, expressed as a solver slot rather than a hand-written loop.

NOT a Ghia comparison, and it should not be read as one. Ghia, Ghia & Shin (*J. Comput. Phys.* **48**
(1982) 387) tabulate the cavity with a CONSTANT lid; the regularisation above changes the driving
profile, so the numbers here are not comparable to that table. What this is instead is an INTERNAL
cross-validation -- the new, unstable-on-its-own pair against the established stable one, on the same
mesh and the same problem. That is the question this branch actually raises.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace, lap = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian

N = 48  # structured n x n mesh
C_I = 36.0  # inverse-estimate constant, linear elements (Tezduyar & Osawa, CMAME 190 (2000) Sec. 3)
RE_LADDER = [100.0, 200.0, 400.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0]


def cavity(n=N, order=1, stabilised=True):
    """The cavity with `nu` left as a runtime parameter, so continuation can sweep it."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).domain()
    d.tag("lid", lambda x, y: y > 1 - 1e-9)
    d.tag("wall", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9))
    d.point_region("ppin", (0.0, 0.0))

    nu = jno.np.parameter((1,), name="nu")
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=order)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("lid", split=True)[:2]
    xw, yw = d.variable("wall", split=True)[:2]
    xpn, ypn = d.variable("ppin", split=True)[:2]

    ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    gp, gq = grad(p, [xi, yi]), grad(q, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)

    div = lambda gw: trace(gw)  # noqa: E731
    adv = lambda gw, w: inner(gw, w, n_contract=1)  # noqa: E731  -- (w.grad)w

    momentum = inner(adv(gu, ub), vv, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * div(gv)
    continuity = -qq * div(gu)

    if stabilised:
        G = d.cell_metric
        gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
        tau = jno.lag((gG(ub) + C_I * nu**2 * inner(G, G, n_contract=2)) ** -0.5)
        r_m = adv(gu, ub) - nu * lap(u, [xi, yi]) + gp
        momentum = momentum + tau * inner(adv(gv, ub), r_m, n_contract=1)  # SUPG -> momentum   (+)
        continuity = continuity - tau * inner(gq, r_m, n_contract=1)  # PSPG -> continuity (-)

    lid_profile = 16.0 * xl**2 * (1 - xl) ** 2  # regularised: zero at both top corners
    return d, jno.fem(
        [
            momentum,
            continuity,
            u(xl, yl)[0] - lid_profile,
            u(xl, yl)[1] - 0.0,
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
            p(xpn, ypn) - 0.0,
        ]
    )


def velocity(fem, sol):
    """`(points, u)` for the velocity block -- `fem.points` is that field's own node set."""
    pts = np.asarray(fem.points)
    return pts, sol[fem.offsets[0] : fem.offsets[1]].reshape(len(pts), 2)


def centreline(pts, uv, axis, at=0.5, tol=1e-9):
    """The profile along a cavity centreline, as `(coordinate, u_x, u_y)` sorted along the line."""
    keep = np.abs(pts[:, axis] - at) < tol
    other = 1 - axis
    order = np.argsort(pts[keep, other])
    return pts[keep, other][order], uv[keep, 0][order], uv[keep, 1][order]


def climb(fem, ladder, keep="all"):
    """Climb the Reynolds ladder, warm-starting each rung from the one below.

    ONE `fem.solve`, not one per rung: the form is compiled once and `nu` arrives as a runtime
    argument, so an n-rung ladder is n solves rather than n rebuilds and n compilations.
    """
    sol = fem.solve(
        nonlinear=jno.solve.newton(direct=True, rtol=1e-9, atol=1e-9),
        linear=jno.solve.lu(backend="host"),
        continuation=jno.solve.continuation(nu=[1.0 / r for r in ladder], keep=keep),
    )
    return np.asarray(sol)


print(f"\nLid-driven cavity, {N}x{N} structured mesh")
print("=" * 62)

# ---- 1. how far the stabilised equal-order pair goes -------------------------------------------
d_s, fem_s = cavity(order=1, stabilised=True)
print(f"stabilised P1/P1 : {fem_s.dofs:6d} dofs")
# `fem.solve(continuation=...)` RAISES on the first rung that fails to converge, so arriving at the
# top of the ladder is itself the convergence statement -- there is no quietly-unconverged rung.
family = climb(fem_s, RE_LADDER, keep="all")  # (n_rungs, n_dofs)
reached = {re: family[i] for i, re in enumerate(RE_LADDER)}
print("     Re    min u_x on x=0.5    u_y range on y=0.5")
for re in RE_LADDER:
    pts, uv = velocity(fem_s, reached[re])
    assert np.isfinite(uv).all(), f"Re = {re:.0f} returned a non-finite field"
    _y, ux, _ = centreline(pts, uv, axis=0)
    _x, _, uy = centreline(pts, uv, axis=1)
    # The primary vortex strengthens monotonically with Re -- the classic cavity signature.
    print(f"  {re:6.0f}      {ux.min():+.4f}          {uy.min():+.4f} .. {uy.max():+.4f}")

# ---- 2. is it right? Taylor-Hood, the inf-sup-STABLE pair, on the same mesh ----------------------
d_t, fem_t = cavity(order=2, stabilised=False)
print(f"\nTaylor-Hood P2/P1: {fem_t.dofs:6d} dofs  ({fem_t.dofs / fem_s.dofs:.1f}x the equal-order pair)")
sol_t = climb(fem_t, [r for r in RE_LADDER if r <= 1000.0], keep="last")

pts_s, uv_s = velocity(fem_s, reached[1000.0])
pts_t, uv_t = velocity(fem_t, sol_t)
ys, ux_s, _ = centreline(pts_s, uv_s, axis=0)  # u_x along the vertical centreline x = 0.5
yt, ux_t, _ = centreline(pts_t, uv_t, axis=0)
xs, _, uy_s = centreline(pts_s, uv_s, axis=1)  # u_y along the horizontal centreline y = 0.5
xt, _, uy_t = centreline(pts_t, uv_t, axis=1)

# Compare on the coarse (P1) node set; the P2 profile is sampled there by interpolation along the line.
ux_ref = np.interp(ys, yt, ux_t)
uy_ref = np.interp(xs, xt, uy_t)
e_u = float(np.max(np.abs(ux_s - ux_ref)))
e_v = float(np.max(np.abs(uy_s - uy_ref)))
span = float(max(np.ptp(ux_ref), np.ptp(uy_ref)))

print("\nRe = 1000 centreline profiles, equal-order vs Taylor-Hood on the same mesh:")
print(f"  max |du_x| along x=0.5 : {e_u:.4f}   ({100 * e_u / span:.1f}% of the profile range)")
print(f"  max |du_y| along y=0.5 : {e_v:.4f}   ({100 * e_v / span:.1f}% of the profile range)")
print(f"  extrema  u_x: {ux_s.min():+.4f} / {ux_s.max():+.4f}   Taylor-Hood {ux_ref.min():+.4f} / {ux_ref.max():+.4f}")
print(f"           u_y: {uy_s.min():+.4f} / {uy_s.max():+.4f}   Taylor-Hood {uy_ref.min():+.4f} / {uy_ref.max():+.4f}")

assert e_u < 0.10 * span, f"equal-order disagrees with the stable pair by {e_u:.3f}"
assert e_v < 0.10 * span, f"equal-order disagrees with the stable pair by {e_v:.3f}"
print(f"\nStabilised equal-order P1/P1 reached Re = {max(reached):.0f} and tracks the stable pair at Re = 1000.")
# --8<-- [end:code]

# ---- figure: speed with streamlines as Re climbs -------------------------------------------------
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402

ctri = mtri.Triangulation(pts_s[:, 0], pts_s[:, 1], np.asarray(d_s._cells_p1()))
gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
fig, axs = plt.subplots(1, 3, figsize=(9, 3.2), dpi=140)
for ax, re in zip(axs, (100.0, 1000.0, 5000.0)):
    _, uv = velocity(fem_s, reached[re])
    tp = ax.tripcolor(ctri, np.hypot(uv[:, 0], uv[:, 1]), cmap="viridis", vmin=0.0, vmax=1.0, shading="gouraud")
    ui = np.asarray(mtri.LinearTriInterpolator(ctri, uv[:, 0])(gx, gy))
    vi = np.asarray(mtri.LinearTriInterpolator(ctri, uv[:, 1])(gx, gy))
    ax.streamplot(gx, gy, ui, vi, color="white", linewidth=0.45, density=0.9, arrowsize=0.5)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Re = {re:.0f}", pad=6)  # a label naming the panel's data, not a describing title
cax = make_axes_locatable(axs[-1]).append_axes("right", size="3.5%", pad=0.05)
cb = plt.colorbar(tp, cax=cax)
cb.outline.set_visible(False)
cb.minorticks_off()
cb.ax.tick_params(length=0)
cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "cavity_high_reynolds.png")
