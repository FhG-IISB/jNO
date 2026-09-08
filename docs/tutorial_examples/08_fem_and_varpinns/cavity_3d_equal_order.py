# --8<-- [start:code]
"""**Equal-order flow in 3-D** -- where the stabilised pair actually pays for itself.

    (u.grad)u - nu lap u + grad p = 0,   div u = 0,   Re = U L / nu = 100

In 2-D, Taylor-Hood P2/P1 costs about 2.9x the DOFs of stabilised P1/P1. In **3-D** the gap is wider,
because a P2 tetrahedron carries 10 nodes to P1's 4: measured below at **5.1x**.

That is not an accounting curiosity. Measured on one 8 GB card with the direct solve this script uses,
the mesh each pair reaches before it runs out of GPU memory (a cuBLAS allocation failure, not a
convergence failure -- both were checked):

    Taylor-Hood P2/P1    N = 6  fits (6,934 dofs)    N = 8  OOM (15,468 dofs)
    stabilised P1/P1     N = 10 fits (5,324 dofs)    N = 12 OOM (8,788 dofs)

The DOF ceiling is roughly the same for both -- it is the factorisation -- but the equal-order pair
spends those DOFs on about **5x more mesh**. The head-to-head below therefore runs at N = 6, the
finest cube where BOTH fit.

This is the same formulation as the 2-D stabilised tutorial -- SUPG on momentum, PSPG on continuity,
`dom.cell_metric` for a direction-aware `tau` -- written for three dimensions by changing the
coordinate list and `value_shape`. Nothing else about it is 3-D-specific.

The mesh is **structured**, so the vertical centreline x = y = 0.5 carries nodes of both the P1 and
the P2 velocity space and the two solutions can be compared where they both live, with no
interpolation in the comparison itself.
"""

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace, lap = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian

N = 6  # structured n x n x n cube; both discretisations must fit a direct solve together
C_I = 36.0
RE_LADDER = [5.0, 10.0, 25.0, 50.0, 100.0]


def cavity3d(n=N, order=1, stabilised=True):
    d = jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=n).domain()
    d.tag("lid", lambda x, y, z: z > 1 - 1e-9)
    d.tag("wall", lambda x, y, z: (z < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    d.point_region("ppin", (0.0, 0.0, 0.0))

    nu = jno.np.parameter((1,), name="nu")
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=order)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xl, yl, zl = d.variable("lid", split=True)[:3]
    xw, yw, zw = d.variable("wall", split=True)[:3]
    xn, yn, zn = d.variable("ppin", split=True)[:3]
    ax = [xi, yi, zi]

    ub, vv = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    gu, gv = grad(u, ax), grad(v, ax)
    gp, gq = grad(p, ax), grad(q, ax)
    pp, qq = p.bind(x=xi, y=yi, z=zi), q.bind(x=xi, y=yi, z=zi)

    div = lambda gw: trace(gw)  # noqa: E731
    adv = lambda gw, w: inner(gw, w, n_contract=1)  # noqa: E731

    momentum = inner(adv(gu, ub), vv, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * div(gv)
    continuity = -qq * div(gu)

    if stabilised:
        G = d.cell_metric  # (3, 3) per quadrature point in 3-D -- the symbol is dimension-generic
        gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
        tau = jno.lag((gG(ub) + C_I * nu**2 * inner(G, G, n_contract=2)) ** -0.5)
        r_m = adv(gu, ub) - nu * lap(u, ax) + gp  # the VECTOR Laplacian, in 3-D
        momentum = momentum + tau * inner(adv(gv, ub), r_m, n_contract=1)  # SUPG (+)
        continuity = continuity - tau * inner(gq, r_m, n_contract=1)  # PSPG (-)

    # A lid driven in x, tapered to zero on all four top edges so the corners carry no singularity.
    lid = 16.0 * xl**2 * (1 - xl) ** 2 * 16.0 * yl**2 * (1 - yl) ** 2
    return d, jno.fem(
        [
            momentum,
            continuity,
            u(xl, yl, zl)[0] - lid,
            u(xl, yl, zl)[1] - 0.0,
            u(xl, yl, zl)[2] - 0.0,
            u(xw, yw, zw)[0] - 0.0,
            u(xw, yw, zw)[1] - 0.0,
            u(xw, yw, zw)[2] - 0.0,
            p(xn, yn, zn) - 0.0,
        ]
    )


def climb(fem):
    return np.asarray(
        fem.solve(
            nonlinear=jno.solve.newton(direct=True, rtol=1e-8, atol=1e-8),
            linear=jno.solve.lu(backend="host"),
            continuation=jno.solve.continuation(nu=[1.0 / r for r in RE_LADDER]),
        )
    )


def centreline(fem, sol, tol=1e-9):
    """`u_x` up the vertical centreline x = y = 0.5, at that field's own nodes."""
    pts = np.asarray(fem.points)
    uv = sol[fem.offsets[0] : fem.offsets[1]].reshape(len(pts), 3)
    on = (np.abs(pts[:, 0] - 0.5) < tol) & (np.abs(pts[:, 1] - 0.5) < tol)
    order = np.argsort(pts[on, 2])
    return pts[on, 2][order], uv[on, 0][order]


print(f"\n3-D lid-driven cavity, {N}x{N}x{N} structured tets, Re = {RE_LADDER[-1]:.0f}")
print("=" * 66)

runs = {}
for label, order, stab in (("stabilised P1/P1", 1, True), ("Taylor-Hood P2/P1", 2, False)):
    d, fem = cavity3d(order=order, stabilised=stab)
    t0 = time.time()
    sol = climb(fem)
    runs[label] = (fem, sol, time.time() - t0)
    print(f"{label:<18}: {fem.dofs:6d} dofs   solved in {runs[label][2]:5.1f} s")

fem_s, sol_s, _ = runs["stabilised P1/P1"]
fem_t, sol_t, _ = runs["Taylor-Hood P2/P1"]
print(f"\nTaylor-Hood costs {fem_t.dofs / fem_s.dofs:.1f}x the DOFs (a P2 tet carries 10 nodes to P1's 4)")

zs, ux_s = centreline(fem_s, sol_s)
zt, ux_t = centreline(fem_t, sol_t)
assert len(zs) > 3 and len(zt) > 3, "the centreline must carry nodes of both spaces"
ux_ref = np.interp(zs, zt, ux_t)  # sample the finer P2 profile at the P1 nodes
err = float(np.max(np.abs(ux_s - ux_ref)))
span = float(np.ptp(ux_ref))

print(f"\nCentreline u_x (x = y = 0.5), {len(zs)} P1 nodes vs {len(zt)} P2 nodes:")
print(f"  max deviation {err:.4f}  ({100 * err / span:.1f}% of the profile range {span:.4f})")
print(
    f"  extrema   equal-order {ux_s.min():+.4f} / {ux_s.max():+.4f}   Taylor-Hood {ux_ref.min():+.4f} / {ux_ref.max():+.4f}"
)

assert err < 0.15 * span, f"equal-order disagrees with the stable pair by {err:.3f} ({100 * err / span:.0f}%)"
print("\nEqual-order tracks the stable pair in 3-D, at a fifth of the degrees of freedom.")
# --8<-- [end:code]
