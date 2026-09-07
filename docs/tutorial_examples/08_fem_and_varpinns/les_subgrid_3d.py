"""An LES subgrid eddy viscosity, inside a real 3-D solve.

A subgrid model adds ``nu_t(grad u)`` to the molecular viscosity. It needs no library API: the filter
width is ``d.cell_size`` and the model is arithmetic on ``grad(u)``, so it goes in the term list like
any other coefficient. Three of them here:

* **Smagorinsky** (1963)                              nu_t = (Cs d)^2 |S|
* **Vreman**, Phys. Fluids 16 (2004) 3670, eq. (5)    nu_t = c sqrt(B_beta / (g:g))
* **WALE**, Nicoud & Ducros, Flow Turb. Combust. 62 (1999) 183, eq. (13)

THE POINT OF THIS SCRIPT is the property that separates them, measured on a SOLVED flow rather than on
a prescribed gradient. Two flows, because they answer different questions:

1. **Plane Couette, u = (z, 0, 0)** -- SIMPLE SHEAR: one non-zero velocity gradient, so the gradient
   tensor is nilpotent and both Vreman's B_beta and WALE's Sd:Sd vanish identically. There is no
   subgrid turbulence in a laminar shear layer, so the correct nu_t is ZERO. Smagorinsky reports a
   large one anyway, because |S| is non-zero in any shear -- the defect Van Driest damping exists to
   patch. This is the discriminating case.
2. **Lid-driven cavity at Re = 100** -- a general 3-D strain field with recirculation. Here the models
   SHOULD act: B_beta and Sd:Sd are non-zero, and a laminar cavity is not a case any of them claims to
   vanish on. Measuring it shows the models are live and how far they move a real answer, and stops
   case 1 being mistaken for "these models never do anything".

SCOPE, up front: this is a wall-bounded LAMINAR demonstration that the models behave as designed
inside a Newton solve. It is **not validated LES** -- that needs a turbulent benchmark against DNS
(channel flow, decaying isotropic turbulence), which this has not run and does not claim.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace, lap, sym, einsum, sqrt, where = (
    jno.np.inner,
    jno.np.grad,
    jno.np.trace,
    jno.np.laplacian,
    jno.np.sym,
    jno.np.einsum,
    jno.np.sqrt,
    jno.np.where,
)

N = int(os.environ.get("N", 6))  # structured n x n x n cube
C_I, NU_MOL = 36.0, 1.0e-2  # Re = U L / nu with U = L = 1
CS, CV, CW, EPS = 0.17, 0.07, 0.325, 1e-30

# Both invariants below are non-negative in exact arithmetic and are computed as a DIFFERENCE of two
# nearly equal numbers, so round-off can push them just below zero -- and a negative one is not a small
# error but a NaN, through sqrt() and **1.5. Measured in pure shear: B_beta lands at 1.4e-20 where it
# should be 0, from a relative cancellation of 2.7e-16. The clamp is part of the model.
clamp = lambda z: where(z > 0.0, z, 0.0)  # noqa: E731


def eddy_viscosity(name, g, delta, dim=3):
    """The three models, written through second invariants so one spelling serves 2-D and 3-D."""
    ss = inner(sym(g), sym(g), n_contract=2)
    if name == "none":
        return 0.0
    if name == "smagorinsky":
        return (CS * delta) ** 2 * sqrt(2.0 * ss)
    if name == "vreman":
        gg = inner(g, g, n_contract=2)
        tr_b = delta**2 * gg  # tr(beta),  beta = delta^2 g gᵀ
        tr_b2 = delta**4 * einsum("...ik,...jk,...jl,...il->...", g, g, g, g)  # tr(beta^2)
        return CV * sqrt(clamp(0.5 * (tr_b**2 - tr_b2)) / (gg + EPS) + EPS)
    if name == "wale":
        tr_g2 = einsum("...ij,...ji->...", g, g)  # tr(g @ g)
        a1 = einsum("...ik,...kj,...il,...lj->...", g, g, g, g)  # tr(g^2 (g^2)ᵀ)
        a2 = einsum("...ik,...kj,...jl,...li->...", g, g, g, g)  # tr(g^2 g^2)
        sd = clamp(0.5 * (a1 + a2) - tr_g2**2 / dim)  # Sd:Sd, deviator never formed
        return (CW * delta) ** 2 * (sd**1.5 / (ss**2.5 + sd**1.25 + EPS))
    raise ValueError(f"unknown model {name!r}")


def cavity3d(model="none", n=N, nu=NU_MOL):
    """Stabilised P1/P1 lid-driven cavity, plus a scalar block that PROJECTS nu_t so it can be read.

    The projection block is present in every variant, `none` included, so the systems being compared
    have identical structure and the difference between them is the model and nothing else.
    """
    d = jno.Shape.box(0, 0, 0, 1, 1, 1).structured(n=n).domain()
    d.tag("lid", lambda x, y, z: z > 1 - 1e-9)
    d.tag("wall", lambda x, y, z: (z < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    d.point_region("ppin", (0.0, 0.0, 0.0))

    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    nut, w = d.fem_symbols(names=("nut", "w"), order=1)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xl, yl, zl = d.variable("lid", split=True)[:3]
    xw, yw, zw = d.variable("wall", split=True)[:3]
    xn, yn, zn = d.variable("ppin", split=True)[:3]
    ax = [xi, yi, zi]

    ub, vv = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    pp, qq = p.bind(x=xi, y=yi, z=zi), q.bind(x=xi, y=yi, z=zi)
    nb, wb = nut.bind(x=xi, y=yi, z=zi), w.bind(x=xi, y=yi, z=zi)
    gu, gv, gp, gq = grad(u, ax), grad(v, ax), grad(p, ax), grad(q, ax)

    div = lambda gw: trace(gw)  # noqa: E731
    adv = lambda gw, ww: inner(gw, ww, n_contract=1)  # noqa: E731

    # LAGGED, and this is not optional: nu_t is a square root, so its slope at u = 0 is infinite and
    # Newton diverges outright from a rest state. Freezing it within each linearisation is the same
    # Picard treatment a Carman-Kozeny drag needs.
    nu_t = jno.lag(eddy_viscosity(model, gu, d.cell_size))
    nu_eff = nu + nu_t

    momentum = inner(adv(gu, ub), vv, n_contract=1) + nu_eff * inner(gu, gv, n_contract=2) - pp * div(gv)
    continuity = -qq * div(gu)

    G = d.cell_metric
    gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
    tau = jno.lag((gG(ub) + C_I * nu**2 * inner(G, G, n_contract=2)) ** -0.5)
    r_m = adv(gu, ub) - nu * lap(u, ax) + gp
    momentum = momentum + tau * inner(adv(gv, ub), r_m, n_contract=1)  # SUPG (+)
    continuity = continuity - tau * inner(gq, r_m, n_contract=1)  # PSPG (-)

    lid = 16.0 * xl**2 * (1 - xl) ** 2 * 16.0 * yl**2 * (1 - yl) ** 2
    fem = jno.fem(
        [
            momentum,
            continuity,
            nb * wb - nu_t * wb,  # read-out only: nu_t never feeds back through this block
            u(xl, yl, zl)[0] - lid,
            u(xl, yl, zl)[1] - 0.0,
            u(xl, yl, zl)[2] - 0.0,
            u(xw, yw, zw)[0] - 0.0,
            u(xw, yw, zw)[1] - 0.0,
            u(xw, yw, zw)[2] - 0.0,
            p(xn, yn, zn) - 0.0,
        ]
    )
    return d, fem, u, nut


def couette3d(model="none", n=N, nu=NU_MOL):
    """Plane Couette, u = (z, 0, 0), imposed on the boundary and SOLVED in the interior.

    The profile is linear, hence an exact solution of the momentum equation, so the interior nodes
    reproduce it and the interior velocity gradient is pure simple shear -- exactly the structure on
    which Vreman and WALE are constructed to vanish. Note the flow itself cannot discriminate: nu_t is
    spatially constant here, and a constant viscosity leaves a Couette profile unchanged. The
    discriminating quantity IS nu_t, which is why it is projected and read out."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1).structured(n=n).domain()
    d.tag("all", lambda x, y, z: (z < 1e-9) | (z > 1 - 1e-9) | (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    d.point_region("ppin", (0.0, 0.0, 0.0))

    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    nut, w = d.fem_symbols(names=("nut", "w"), order=1)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xa, ya, za = d.variable("all", split=True)[:3]
    xn, yn, zn = d.variable("ppin", split=True)[:3]
    ax = [xi, yi, zi]

    ub, vv = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    pp, qq = p.bind(x=xi, y=yi, z=zi), q.bind(x=xi, y=yi, z=zi)
    nb, wb = nut.bind(x=xi, y=yi, z=zi), w.bind(x=xi, y=yi, z=zi)
    gu, gv = grad(u, ax), grad(v, ax)

    nu_t = jno.lag(eddy_viscosity(model, gu, d.cell_size))
    momentum = (
        inner(inner(gu, ub, n_contract=1), vv, n_contract=1) + (nu + nu_t) * inner(gu, gv, n_contract=2) - pp * trace(gv)
    )
    continuity = -qq * trace(gu)
    # PSPG/SUPG, exactly as in the cavity. Not optional: equal-order P1/P1 without it is not inf-sup
    # stable and the factorisation is EXACTLY SINGULAR. It also costs nothing in accuracy here -- the
    # strong residual of a Couette profile is identically zero ((u.grad)u = 0, lap u = 0, grad p = 0),
    # so a consistent stabilisation leaves the exact solution exact.
    G = d.cell_metric
    gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
    tau = jno.lag((gG(ub) + C_I * nu**2 * inner(G, G, n_contract=2)) ** -0.5)
    r_m = inner(gu, ub, n_contract=1) - nu * lap(u, ax) + grad(p, ax)
    momentum = momentum + tau * inner(inner(gv, ub, n_contract=1), r_m, n_contract=1)
    continuity = continuity - tau * inner(grad(q, ax), r_m, n_contract=1)
    fem = jno.fem(
        [
            momentum,
            continuity,
            nb * wb - nu_t * wb,
            u(xa, ya, za)[0] - za,  # u = (z, 0, 0) on the boundary -> exact Couette inside
            u(xa, ya, za)[1] - 0.0,
            u(xa, ya, za)[2] - 0.0,
            p(xn, yn, zn) - 0.0,
        ]
    )
    return d, fem, u, nut


def run(model, n=N, nu=NU_MOL, case="cavity"):
    d, fem, u, nut = (cavity3d if case == "cavity" else couette3d)(model, n=n, nu=nu)
    t0 = time.time()
    sol = np.asarray(
        fem.solve(
            nonlinear=jno.solve.newton(direct=True, rtol=1e-10, atol=1e-10),
            linear=jno.solve.lu(backend="host"),
        )
    )
    dt = time.time() - t0
    bu, bn = fem.blocks[fem.block_index(u)], fem.blocks[fem.block_index(nut)]
    return sol[bu.start : bu.stop], np.abs(sol[bn.start : bn.stop]).max(), fem.dofs, dt


if __name__ == "__main__":
    nu = float(sys.argv[1]) if len(sys.argv) > 1 else NU_MOL

    print(f"(1) PLANE COUETTE, u = (z, 0, 0) -- simple shear, {N}x{N}x{N} tets, Re = {1.0 / nu:.0f}")
    print("    Simple shear carries no subgrid turbulence, so the correct nu_t is ZERO.")
    base_c, nut_c, dofs_c, t_c = run("none", nu=nu, case="couette")
    ref = max(nut_c, 0.0)
    print(f"  {'no model':<14} {dofs_c:6d} dofs  {t_c:6.1f} s   max nu_t/nu = {nut_c / nu:.3e}")
    got = {}
    for model in ("smagorinsky", "vreman", "wale"):
        _uu, nut, _d, dt = run(model, nu=nu, case="couette")
        got[model] = (nut, dt)
    # Judged RELATIVE to Smagorinsky, not against an absolute floor: Vreman's invariant is a difference
    # of nearly equal numbers, so in exact-zero cases it lands on the cancellation floor (~1e-6 of nu
    # here) rather than on 0. "Vanishes" means orders below the model that does not.
    ref = got["smagorinsky"][0]
    for model in ("vreman", "wale", "smagorinsky"):
        nut, dt = got[model]
        tag = "~ ZERO (correct)" if nut < 1e-3 * ref else "NON-ZERO in laminar shear"
        print(f"  {model:<14} {'':6} {'':4} {dt:6.1f} s   max nu_t/nu = {nut / nu:11.4e}   {tag}")
    print(
        f"    -> Smagorinsky invents nu_t/nu = {ref / nu:.3f} where the answer is zero -- "
        f"{ref / max(got['vreman'][0], 1e-300):.0f}x Vreman's\n"
        f"       and {ref / max(got['wale'][0], 1e-300):.1e}x WALE's. That gap IS the difference "
        f"between the models.\n"
    )

    print(f"(2) LID-DRIVEN CAVITY at Re = {1.0 / nu:.0f} -- a general 3-D strain field, where the models SHOULD act")
    base, nut0, dofs, t0 = run("none", nu=nu)
    scale = np.abs(base).max()
    print(f"  {'no model':<14} {dofs:6d} dofs  {t0:6.1f} s   max|u| = {scale:.6f}   max nu_t/nu = {nut0 / nu:.3e}")
    for model in ("vreman", "wale", "smagorinsky"):
        uu, nut, _d, dt = run(model, nu=nu)
        print(
            f"  {model:<14} {'':6} {'':4} {dt:6.1f} s   d|u|/|u| = {np.abs(uu - base).max() / scale:.3e}"
            f"   max nu_t/nu = {nut / nu:8.3e}"
        )
    print(
        "\n    A laminar cavity is NOT a case any of these models claims to vanish on -- it has real\n"
        "    3-D strain. The point of (2) is that the models are live and bounded, and that the\n"
        "    separation seen in (1) is a property of the flow structure, not of the implementation."
    )
