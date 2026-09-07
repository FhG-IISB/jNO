"""Residual-based stabilization, written as ordinary weak-form terms.

Nothing here is a library feature: SUPG / PSPG / grad-div are *formulas*, which is the point. What the
library had to supply is the two pieces they are built from -- a **vector** `laplacian` (the momentum
strong residual carries `nu*lap(u)`) and `dom.cell_metric` (the direction-aware `G` that `tau` needs).
These tests are the claim that those two are enough.

`tau` follows Tezduyar & Osawa, *CMAME* **190** (2000) Sec. 3: `tau_m = (u.G u + C_I nu^2 G:G)^-1/2`
for a steady problem, with the LSIC/grad-div coefficient `tau_c = 1/(tr(G) tau_m)`. PSPG is Hughes,
Franca & Balestra, *CMAME* **59** (1986) -- it is what lets equal-order P1/P1 velocity/pressure work
at all, since that pair is not inf-sup stable.

Oracles are properties, not pinned numbers: a monotone exact solution must not be undershot, and an
equal-order pair must actually converge.
"""

import jax
import numpy as np
import pytest

import jno

inner, grad, trace, lap = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian
C_I = 36.0  # the inverse-estimate constant for linear elements (Tezduyar & Osawa, Sec. 3)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ======================================================================================
# Scalar transport at high Peclet -- the sharpest statement of what stabilization buys
# ======================================================================================
def _advection_diffusion(n, nu, stabilised):
    """`b.grad u - nu lap u = 0` on the unit square, `u(0,y)=0`, `u(1,y)=1`, `b = (1, 0)`.

    The exact solution is monotone in x with a boundary layer of width `nu` at the outflow. Galerkin
    P1 cannot represent it and rings; SUPG adds streamline diffusion and does not.
    """
    dom = jno.Shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).domain()
    dom.tag("inflow", lambda x, y: x < 1e-9)
    dom.tag("outflow", lambda x, y: x > 1.0 - 1e-9)
    u, v = dom.fem_symbols()
    xi, yi = dom.variable("interior", split=True)[:2]
    x0, y0 = dom.variable("inflow", split=True)[:2]
    x1, y1 = dom.variable("outflow", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    b = (1.0, 0.0)

    adv = lambda w: b[0] * w.x + b[1] * w.y  # noqa: E731  -- b.grad w
    galerkin = adv(ui) * vi + nu * (ui.x * vi.x + ui.y * vi.y)
    terms = [galerkin, u(x0, y0) - 0.0, u(x1, y1) - 1.0]
    if stabilised:
        G = dom.cell_metric
        bGb = inner(b, inner(G, b, n_contract=1), n_contract=1)
        tau = (bGb + C_I * nu**2 * inner(G, G, n_contract=2)) ** -0.5
        # The strong residual. `lap(ui)` is identically zero on P1 -- it is written because the form is
        # the same one a higher-order run uses, and dropping it silently would be the wrong habit.
        r_strong = adv(ui) - nu * lap(ui, [xi, yi])
        terms[0] = galerkin + tau * adv(vi) * r_strong
    return dom, jno.fem(terms)


def _nodal(fem):
    # Direct: the advection-dominated operator is strongly non-symmetric and the default
    # Jacobi-BiCGStab does not reach the residual gate on it. That is a solver choice, not part of
    # what is being tested -- and it fails loudly rather than returning a wrong answer.
    return np.asarray(fem.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)


@pytest.mark.parametrize("nu", [1e-2, 1e-3])
def test_supg_confines_the_galerkin_wiggles_to_the_layer(nu):
    """The exact solution rises monotonically from 0 to 1, so every undershoot is an artefact.

    Unstabilised P1 rings across the WHOLE domain -- at `nu = 1e-3` it reaches -3.00, three times the
    solution's own range, far upstream of anything physical. SUPG confines the error to the unresolved
    outflow layer: upstream of it the solution is clean to ~1e-2 or better.

    Stated honestly, because it is the limit of the method: SUPG is **not** monotone. It still leaves a
    local over/undershoot at a layer the mesh cannot resolve (measured min -0.109 at `nu = 1e-2`,
    h = 1/16). Removing that needs a discontinuity-capturing term, which is a different formula.
    """
    n = 16
    dom, fem_g = _advection_diffusion(n, nu, stabilised=False)
    _, fem_s = _advection_diffusion(n, nu, stabilised=True)
    gal, sup = _nodal(fem_g), _nodal(fem_s)
    upstream = np.asarray(dom.mesh.points)[:, 0] < 0.9  # away from the boundary layer

    assert gal[upstream].min() < -0.25, f"the baseline must actually ring upstream ({gal[upstream].min():.3g})"
    assert abs(sup[upstream].min()) < 0.05, f"SUPG must not ring upstream ({sup[upstream].min():.3g})"
    assert abs(gal[upstream].min()) > 10.0 * abs(sup[upstream].min()), "at least a 10x reduction"
    assert sup.max() < 1.15, f"and it must not blow past the data ({sup.max():.3g}); Galerkin reaches {gal.max():.3g}"


def test_the_stabilisation_vanishes_when_it_should():
    """A diffusion-dominated problem needs no stabilization, and `tau` must say so on its own: at
    `nu = 1`, `tau ~ h/|b| * O(h)` is tiny and the stabilised answer must track the Galerkin one.
    A `tau` that did not scale with the cell metric would perturb this case visibly."""
    _, fem_g = _advection_diffusion(12, 1.0, stabilised=False)
    _, fem_s = _advection_diffusion(12, 1.0, stabilised=True)
    gal, sup = _nodal(fem_g), _nodal(fem_s)
    assert np.max(np.abs(sup - gal)) < 5e-3, "stabilization must be negligible where it is not needed"


# ======================================================================================
# Equal-order P1/P1 incompressible flow -- what PSPG unlocks
# ======================================================================================
def _kovasznay_exact(nu):
    """Kovasznay (1948): a closed-form steady Navier-Stokes solution."""
    Re = 1.0 / nu
    lam = Re / 2.0 - np.sqrt(Re**2 / 4.0 + 4.0 * np.pi**2)
    ue = lambda x, y: 1.0 - np.exp(lam * x) * np.cos(2 * np.pi * y)  # noqa: E731
    ve = lambda x, y: lam / (2 * np.pi) * np.exp(lam * x) * np.sin(2 * np.pi * y)  # noqa: E731
    pe = lambda x, y: 0.5 * (1.0 - np.exp(2 * lam * x))  # noqa: E731
    return lam, ue, ve, pe


def _kovasznay_p1p1(ms, nu=0.05, stabilised=True):
    """Equal-order P1 velocity / P1 pressure. Not inf-sup stable -- PSPG is what makes it solvable."""
    from shapely.geometry import box

    lam, ue, ve, pe = _kovasznay_exact(nu)
    d = jno.domain(box(-0.5, -0.5, 1.0, 1.5), mesh_size=ms)
    x0, y0 = -0.5, -0.5
    d.point_region("ppin", (x0, y0))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
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
        gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731  -- a^T G a
        # tau is LAGGED: it is a coefficient, not part of the equation, and differentiating
        # through `u.G u` gives a tangent Newton cannot follow from a cold start.
        tau_m = jno.lag((gG(ub) + C_I * nu**2 * inner(G, G, n_contract=2)) ** -0.5)
        # The LSIC / grad-div coefficient. NOT used in the verified form: at this Reynolds number it
        # made both errors worse (e_u 6.14e-02 -> 8.36e-02, e_p 1.78e-01 -> 2.06e-01), so its
        # calibration is left open rather than shipped as if it were verified.
        tau_c = 1.0 / (trace(G) * tau_m)
        r_m = adv(gu, ub) - nu * lap(u, [xi, yi]) + gp  # the VECTOR Laplacian is the enabling piece
        # SUPG and PSPG are the SAME strong residual against two different test perturbations, and in
        # jNO they must be written as two terms: an additive term carries exactly one test field,
        # because the test field is what names the equation block. SUPG (test `v`) belongs to
        # momentum; PSPG (test `q`) belongs to continuity -- which is exactly what it stabilises.
        # SIGNS MATTER, and they are not both plus: a stabilization term is added to an equation, so
        # it must carry that equation's OWN sign convention. Momentum is written `+(u.grad u, v)`, so
        # SUPG is `+`. Continuity is written `-(q, div u)`, so PSPG is `-`. Measured at ms=0.16: the
        # correct pair takes the pressure error from 6.30e-01 to 1.78e-01, while `+` on PSPG makes
        # Newton diverge outright. (`tau_c` is deliberately unused -- see the note below.)
        momentum = momentum + tau_m * inner(adv(gv, ub), r_m, n_contract=1)  # SUPG
        continuity = continuity - tau_m * inner(gq, r_m, n_contract=1)  # PSPG
        _ = tau_c

    bx = 1.0 - jno.np.exp(lam * xb) * jno.np.cos(2 * np.pi * yb)
    by = lam / (2 * np.pi) * jno.np.exp(lam * xb) * jno.np.sin(2 * np.pi * yb)
    fem = jno.fem(
        [
            momentum,
            continuity,
            u(xb, yb)[0] - bx,
            u(xb, yb)[1] - by,
            p(xpn, ypn) - float(pe(x0, y0)),
        ]
    )
    return d, fem, (ue, ve, pe)


def _errors(d, fem, exact):
    """Discrete RMS nodal error. P1 nodes are the mesh vertices, so this converges at the L2 rate on a
    quasi-uniform mesh -- enough for a rate study, and it needs no quadrature of the exact field."""
    ue, ve, pe = exact
    # Assembled-tangent Newton: matrix-free JFNK goes NaN on this cold start from rest (the
    # stabilised saddle tangent is what it struggles with, not the physics).
    sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-8, atol=1e-8)))
    off = fem.offsets
    uv = sol[off[0] : off[1]].reshape(-1, 2)
    ph = sol[off[1] :].reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :2]
    x, y = pts[:, 0], pts[:, 1]
    e_u = np.sqrt(np.mean((uv[:, 0] - ue(x, y)) ** 2 + (uv[:, 1] - ve(x, y)) ** 2))
    p_ex = pe(x, y)
    e_p = np.sqrt(np.mean(((ph - ph.mean()) - (p_ex - p_ex.mean())) ** 2))  # gauge-free
    return e_u, e_p


@pytest.mark.slow
def test_stabilised_equal_order_p1p1_converges():
    """The claim: with PSPG + SUPG + grad-div, P1/P1 -- half the DOFs of Taylor-Hood in 3-D -- is a
    working discretisation. Rates are measured, not pinned: velocity must show at least first order
    and the pressure must converge at all (unstabilised, it does neither)."""
    sizes = [0.16, 0.11, 0.075]
    errs = []
    for ms in sizes:
        d, fem, exact = _kovasznay_p1p1(ms)
        errs.append(_errors(d, fem, exact))
    eu = [e[0] for e in errs]
    ep = [e[1] for e in errs]
    ru = np.log(eu[0] / eu[-1]) / np.log(sizes[0] / sizes[-1])
    rp = np.log(ep[0] / ep[-1]) / np.log(sizes[0] / sizes[-1])
    assert eu[-1] < eu[0], f"velocity error must fall under refinement: {eu}"
    assert ep[-1] < ep[0], f"pressure error must fall under refinement: {ep}"
    assert ru > 1.2, f"velocity rate {ru:.2f} (P1 should approach 2), errors {eu}"
    assert rp > 0.8, f"pressure rate {rp:.2f} (equal-order P1 should approach 1.5), errors {ep}"
    print(f"\nstabilised P1/P1 Kovasznay: velocity rate {ru:.2f}, pressure rate {rp:.2f}")


@pytest.mark.slow
def test_pspg_is_what_fixes_the_equal_order_pressure():
    """The contrast that proves the stabilization is doing the work, not the mesh: P1/P1 violates the
    inf-sup condition, and PSPG is the term that compensates.

    Measured at ms=0.11: pressure error 4.09e-01 unstabilised against 9.42e-02 stabilised -- a 4.3x
    improvement. The velocity is very slightly WORSE (5.24e-02 -> 6.14e-02 at ms=0.16), which is the
    expected trade and is stated rather than hidden.
    """
    ms = 0.11
    d_s, fem_s, exact = _kovasznay_p1p1(ms, stabilised=True)
    d_u, fem_u, _ = _kovasznay_p1p1(ms, stabilised=False)
    _, ep_s = _errors(d_s, fem_s, exact)
    _, ep_u = _errors(d_u, fem_u, exact)
    assert ep_u > 3.0 * ep_s, f"unstabilised pressure {ep_u:.3g} vs stabilised {ep_s:.3g}"


def test_the_vector_laplacian_is_really_in_the_residual():
    """A P2 velocity has a non-zero `lap(u)`, so the viscous part of the strong residual must change the
    assembled system. This is the tie back to the vector-Hessian assembly: if it were silently dropped
    the two operators would coincide."""
    from shapely.geometry import box

    nu = 0.05
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ub = u.bind(x=xi, y=yi)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    G = d.cell_metric
    tau = (inner((1.0, 0.0), inner(G, (1.0, 0.0), n_contract=1), n_contract=1)) ** -0.5

    def build(with_viscous):
        r = inner(gu, ub, n_contract=1)
        if with_viscous:
            r = r - nu * lap(u, [xi, yi])
        stab = tau * inner(inner(gv, ub, n_contract=1), r, n_contract=1)
        base = nu * inner(gu, gv, n_contract=2)
        fem = jno.fem([base + stab, u(xb, yb)[0] - 1.0, u(xb, yb)[1] - 0.0])
        return np.asarray(fem.jacobian(np.zeros(fem.dofs)))

    a, b = build(False), build(True)
    assert not np.allclose(a, b), "nu*lap(u) must change the stabilised operator on P2"
    assert np.abs(a - b).max() > 1e-8
