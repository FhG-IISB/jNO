"""Component-indexed derivatives of vector Lagrange fields — ``u[i].x`` as a first-class spelling.

A vector field's components used to be second-class: ``u[i]`` *values* worked, but ``u[i].d(x)``
raised ("gradients of TrialFunction/TestFunction only"), so any form that is not expressible through
``inner``/``symgrad`` — above all **finite-strain elasticity**, whose ``det F`` / ``F^{-T}`` are
tensor-nonlinear in the displacement gradient — forced either the coupled-scalar workaround or a
different library. The lowering existed for the non-nodal families (RT/N1E ``.div()``/``.curl()``
sugar); these tests pin its Lagrange twin.

Conventions (must match the whole-field branches so spellings mix in one term):
trial ``u[i].d(x_l)`` -> (n_quad,); test ``v[i].d(x_l)`` -> (n_quad, n_local, n_comp) with only the
DOF-component column ``i`` nonzero (node-major ravel).
"""

import jax
import numpy as np
import pytest

import jno
import jno.jnp_ops as J


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _cantilever(value_shape=(2,)):
    d = jno.Shape.rect(0.0, 0.0, 2.0, 0.5, size=0.15).domain()
    d.tag("left", lambda x, n, names: x[:, 0] < 1e-6)
    d.tag("right", lambda x, n, names: x[:, 0] > 2.0 - 1e-6)
    u, v = d.fem_symbols(value_shape=value_shape)
    xi, yi = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    xr, yr = d.variable("right", split=True)[:2]
    return d, u, v, u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), (xi, yi), (xl, yl), v.bind(x=xr, y=yr)


MU, LAM = 1.0, 0.5


def test_component_spelling_matches_symgrad_linear_elasticity():
    """The same Hooke form written component-wise and through ``symgrad``/``inner`` must agree — the
    component route feeds the SAME linear assembly (A is jacfwd of the shared residual kernel), so a
    disagreement would mean the two spellings evaluate different physics."""
    d1, u1, _v1, ui, vi, _c, (xl, yl), vr = _cantilever()
    e11, e22, e12 = ui[0].x, ui[1].y, 0.5 * (ui[0].y + ui[1].x)
    s11 = LAM * (e11 + e22) + 2 * MU * e11
    s22 = LAM * (e11 + e22) + 2 * MU * e22
    s12 = 2 * MU * e12
    weak_c = s11 * vi[0].x + s12 * vi[0].y + s12 * vi[1].x + s22 * vi[1].y
    sol_c = np.asarray(jno.fem([weak_c, u1(xl, yl)[0] - 0.0, u1(xl, yl)[1] - 0.0, -0.01 * vr[1]]).solve())

    d2, u2, _v2, ui2, vi2, (xi2, yi2), (xl2, yl2), vr2 = _cantilever()
    eps_u, eps_v = J.symgrad(ui2, [xi2, yi2]), J.symgrad(vi2, [xi2, yi2])
    div_u, div_v = ui2[0].x + ui2[1].y, vi2[0].x + vi2[1].y  # component divergence in a mixed spelling
    weak_s = 2 * MU * J.inner(eps_u, eps_v, n_contract=2) + LAM * div_u * div_v
    sol_s = np.asarray(jno.fem([weak_s, u2(xl2, yl2)[0] - 0.0, u2(xl2, yl2)[1] - 0.0, -0.01 * vr2[1]]).solve())

    assert np.abs(sol_c).max() > 0.1, "the load never arrived"
    assert sol_c == pytest.approx(sol_s, abs=5e-9), f"spellings disagree: max|d|={np.abs(sol_c - sol_s).max():.3e}"


@pytest.mark.parametrize("comp, expect", [(0, "axial"), (1, "bending")])
def test_a_component_traction_loads_the_named_component(comp, expect):
    """``-t * v[i]`` on a boundary is a traction on component ``i`` alone. Comp 0 on the tip of a
    cantilever stretches it (|ux| dominant); comp 1 bends it (|uy| dominant). A mis-indexed surface
    table would swap or collapse these."""
    d, u, _v, ui, vi, _c, (xl, yl), vr = _cantilever()
    e11, e22, e12 = ui[0].x, ui[1].y, 0.5 * (ui[0].y + ui[1].x)
    s11 = LAM * (e11 + e22) + 2 * MU * e11
    s22 = LAM * (e11 + e22) + 2 * MU * e22
    s12 = 2 * MU * e12
    weak = s11 * vi[0].x + s12 * vi[0].y + s12 * vi[1].x + s22 * vi[1].y
    fem = jno.fem([weak, u(xl, yl)[0] - 0.0, u(xl, yl)[1] - 0.0, -0.01 * vr[comp]])
    n = len(d.mesh.points)
    uxy = np.asarray(fem.solve()).reshape(n, 2)
    ax, bend = np.abs(uxy[:, 0]).max(), np.abs(uxy[:, 1]).max()
    if expect == "axial":
        assert ax > 5 * bend, f"comp-0 traction should stretch, got |ux| {ax:.4f} vs |uy| {bend:.4f}"
    else:
        assert bend > 5 * ax, f"comp-1 traction should bend, got |uy| {bend:.4f} vs |ux| {ax:.4f}"


def _neo_hookean_P(F11, F12, F21, F22):
    Jd = F11 * F22 - F12 * F21
    iT11, iT12 = F22 / Jd, -F21 / Jd
    iT21, iT22 = -F12 / Jd, F11 / Jd
    c = LAM * J.log(Jd)
    P11 = MU * (F11 - iT11) + c * iT11
    P12 = MU * (F12 - iT12) + c * iT12
    P21 = MU * (F21 - iT21) + c * iT21
    P22 = MU * (F22 - iT22) + c * iT22
    return P11, P12, P21, P22


def test_finite_strain_neo_hookean_in_the_natural_spelling():
    """The use case the fix exists for: compressible Neo-Hookean written as every textbook writes it —
    ``F = I + ∇u`` with ``det F`` and ``F^{-T}`` — on one ``value_shape=(2,)`` field, solved by Newton.

    Verified against the coupled-scalar spelling of the SAME problem (two scalar fields, previously the
    only working route): identical answers, well into the geometrically nonlinear regime. The
    coupled-scalar reference itself was validated against linear elasticity in the small-load limit
    (rel 5.4e-4) when this capability was scoped."""

    def solve_vector(load):
        d, u, _v, ui, vi, _c, (xl, yl), _vr = _cantilever()
        P11, P12, P21, P22 = _neo_hookean_P(1.0 + ui[0].x, ui[0].y, ui[1].x, 1.0 + ui[1].y)
        weak = P11 * vi[0].x + P12 * vi[0].y + P21 * vi[1].x + P22 * vi[1].y + load * vi[1]
        fem = jno.fem([weak, u(xl, yl)[0] - 0.0, u(xl, yl)[1] - 0.0])
        sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(line_search=True)))
        return sol.reshape(len(d.mesh.points), 2)

    def solve_scalar(load):
        d = jno.Shape.rect(0.0, 0.0, 2.0, 0.5, size=0.15).domain()
        d.tag("left", lambda x, n, names: x[:, 0] < 1e-6)
        a, qa = d.fem_symbols(names=("a", "qa"))
        b, qb = d.fem_symbols(names=("b", "qb"))
        xi, yi = d.variable("interior", split=True)[:2]
        xl, yl = d.variable("left", split=True)[:2]
        ai, bi = a.bind(x=xi, y=yi), b.bind(x=xi, y=yi)
        va, vb = qa.bind(x=xi, y=yi), qb.bind(x=xi, y=yi)
        P11, P12, P21, P22 = _neo_hookean_P(1.0 + ai.x, ai.y, bi.x, 1.0 + bi.y)
        w1 = P11 * va.x + P12 * va.y
        w2 = P21 * vb.x + P22 * vb.y + load * vb
        fem = jno.fem([w1, w2, a(xl, yl) - 0.0, b(xl, yl) - 0.0])
        sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(line_search=True)))
        n = len(d.mesh.points)
        return np.stack([sol[:n], sol[n:]], axis=1)

    for load in (0.001, 0.05):
        uv, us = solve_vector(load), solve_scalar(load)
        assert np.isfinite(uv).all()
        assert uv == pytest.approx(us, abs=1e-10), (
            f"load={load}: vector spelling disagrees with the coupled-scalar route by {np.abs(uv - us).max():.3e}"
        )
    assert np.abs(uv).max() > 1.0, "the large-load case should be well into the finite-strain regime"


@pytest.mark.parametrize("order", [1, 2])
def test_decoupled_component_laplacians_match_the_scalar_solve(order):
    """Two decoupled component Laplacians with the source on component 1 only: component 0 must stay
    identically zero and component 1 must equal the plain scalar Poisson solution — at P1 and P2, since
    the component tables are built from the same order-k basis as everything else."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    u, v = d.fem_symbols(value_shape=(2,), order=order)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    weak = ui[0].x * vi[0].x + ui[0].y * vi[0].y + ui[1].x * vi[1].x + ui[1].y * vi[1].y - 1.0 * vi[1]
    fem = jno.fem([weak, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0])
    sol = np.asarray(fem.solve()).reshape(-1, 2)

    d2 = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    us, vs = d2.fem_symbols(order=order)
    xi2, yi2 = d2.variable("interior", split=True)[:2]
    xb2, yb2 = d2.variable("boundary", split=True)[:2]
    usi, vsi = us.bind(x=xi2, y=yi2), vs.bind(x=xi2, y=yi2)
    ref = np.asarray(jno.fem([usi.x * vsi.x + usi.y * vsi.y - 1.0 * vsi, us(xb2, yb2) - 0.0]).solve())

    assert np.abs(sol[:, 0]).max() < 1e-9, f"P{order}: unforced component drifted to {np.abs(sol[:, 0]).max():.2e}"
    assert sol[:, 1] == pytest.approx(ref, abs=5e-9), f"P{order}: forced component != scalar Poisson"


def test_a_component_out_of_range_fails_loud():
    """``u[2].x`` on a 2-component field must name the problem, not index garbage."""
    _d, u, _v, ui, vi, _c, (xl, yl), _vr = _cantilever()
    weak = ui[2].x * vi[0].x - 1.0 * vi[1]
    with pytest.raises(IndexError, match="out of range"):
        jno.fem([weak, u(xl, yl)[0] - 0.0, u(xl, yl)[1] - 0.0]).solve()


def test_component_and_whole_field_spellings_mix_in_one_term():
    """Half the stress written through ``symgrad``/``inner``, half through components, in ONE weak form.
    The shape conventions are shared, so the mixed form must equal the pure ones."""
    d1, u1, _v1, ui, vi, (xi, yi), (xl, yl), vr = _cantilever()
    eps_u, eps_v = J.symgrad(ui, [xi, yi]), J.symgrad(vi, [xi, yi])
    div_u = ui[0].x + ui[1].y
    weak_mixed = 2 * MU * J.inner(eps_u, eps_v, n_contract=2) + LAM * div_u * (vi[0].x + vi[1].y)
    sol_m = np.asarray(jno.fem([weak_mixed, u1(xl, yl)[0] - 0.0, u1(xl, yl)[1] - 0.0, -0.01 * vr[1]]).solve())

    d2, u2, _v2, ui2, vi2, _c2, (xl2, yl2), vr2 = _cantilever()
    e11, e22, e12 = ui2[0].x, ui2[1].y, 0.5 * (ui2[0].y + ui2[1].x)
    s11 = LAM * (e11 + e22) + 2 * MU * e11
    s22 = LAM * (e11 + e22) + 2 * MU * e22
    s12 = 2 * MU * e12
    weak_c = s11 * vi2[0].x + s12 * vi2[0].y + s12 * vi2[1].x + s22 * vi2[1].y
    sol_c = np.asarray(jno.fem([weak_c, u2(xl2, yl2)[0] - 0.0, u2(xl2, yl2)[1] - 0.0, -0.01 * vr2[1]]).solve())

    assert sol_m == pytest.approx(sol_c, abs=5e-9), f"mixed spelling drifted: {np.abs(sol_m - sol_c).max():.3e}"


# ==================================================================================================
# One spelling, one meaning: `field[i]` is the i-th COMPONENT, on a raw field exactly as on a view.
#
# `Placeholder.__getitem__` indexes the LEADING axis, which is right for a plain array and wrong for
# a FEM field — at assembly the leading axis is quadrature points (then DOFs, for a test function)
# and the component axis is last. So `u.vector[0]` and `u(region)[0]` (both built as `expr[..., i]`
# by the typed views) selected a component while a bare `u[0]` sliced quadrature points and died in
# the assembler with a raw broadcast error that named nothing.
# ==================================================================================================
def _bar(size=0.5):
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=size).domain()
    d.tag("left", lambda x, y: x < 1e-9)
    return d


def _elastic_term(d, load_term_of):
    """Assemble the bar's volume term at a FIXED state, via ``fem.eval``.

    Comparing the assembled term rather than the solve is what makes "these spellings are the same
    term" testable exactly: two identical operators can still differ by ~1e-11 after an iterative
    solve, because a Krylov reduction is not order-deterministic on GPU. Evaluating at a fixed state
    also works whether the form classifies linear or nonlinear."""
    LAM, MU = 115.4, 76.9
    sym, grad, trace, inner = J.sym, J.grad, J.trace, J.inner
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols(value_shape=(2,))
    eps = lambda w: sym(grad(w, X))
    bulk = LAM * trace(eps(u)) * trace(eps(phi)) + 2 * MU * inner(eps(u), eps(phi), 2)
    term = bulk - load_term_of(u, phi)
    fem = jno.fem([term, u(*cl)[0] - 0.0, u(*cl)[1] - 0.0])
    rng = np.random.default_rng(3)
    u0 = rng.normal(scale=0.01, size=fem.dofs)
    return np.asarray(fem.eval(term, u0))


def test_every_component_spelling_assembles_the_same_term():
    """The relationship that was broken: a raw `phi[0]`, the ellipsis form, the typed view and the
    idiomatic contraction must be the SAME term. Three of the four already agreed; `phi[0]` raised."""
    b = 2.5
    spellings = {
        "raw phi[0]": lambda u, phi: b * phi[0],
        "phi[..., 0]": lambda u, phi: b * phi[..., 0],
        "phi.vector[0]": lambda u, phi: b * phi.vector[0],
        "inner(b, phi)": lambda u, phi: J.inner(J.asarray([b, 0.0]), phi, 1),
    }
    terms = {k: _elastic_term(_bar(), f) for k, f in spellings.items()}
    ref = terms["inner(b, phi)"]
    assert np.abs(ref).max() > 1e-6  # a real term, not a trivially-zero agreement
    scale = max(1.0, float(np.abs(ref).max()))
    for k, v in terms.items():
        # Not bit-for-bit: a GPU reduction is not order-deterministic, so the same term reassociates
        # to ~1e-15. The bound is still ~9 orders tighter than any real difference — indexing the
        # wrong axis changes the answer by O(1) or fails outright.
        assert np.abs(v - ref).max() <= 1e-12 * scale, f"{k} assembles a different term"


def test_a_raw_trial_component_works_in_a_weak_term():
    """The trial side of the same fix — `u[0]` used as a VALUE (not a derivative) in a volume term."""
    K = 30.0
    raw = _elastic_term(_bar(0.4), lambda u, phi: J.inner(J.asarray([2.5, 0.0]), phi, 1) - K * u[0] * phi[0])
    ell = _elastic_term(_bar(0.4), lambda u, phi: J.inner(J.asarray([2.5, 0.0]), phi, 1) - K * u[..., 0] * phi[..., 0])
    assert np.abs(raw).max() > 1e-6
    assert np.abs(raw - ell).max() <= 1e-12 * max(1.0, float(np.abs(ell).max()))


def test_component_index_still_recovers_for_per_component_dirichlet():
    """The pin path reads `getitem_key` to recover the component. Inserting the ellipsis must leave
    that untouched, so a per-component (roller) BC still pins exactly ONE component's DOFs.

    Asserted on the pin set rather than a solve: pinning only u_x leaves rigid-body motion in y, so a
    solve would be singular for reasons that have nothing to do with the component index."""
    LAM, MU = 115.4, 76.9
    sym, grad, trace, inner = J.sym, J.grad, J.trace, J.inner
    d = _bar(0.4)
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols(value_shape=(2,))
    eps = lambda w: sym(grad(w, X))
    bulk = LAM * trace(eps(u)) * trace(eps(phi)) + 2 * MU * inner(eps(u), eps(phi), 2)
    body = inner(J.asarray([1.0, 1.0]), phi, 1)

    fem = jno.fem([bulk - body, u(*cl)[0] - 0.0])  # roller: x only
    pinned = {int(dof) for dof, _v in (d._fem_native_dirichlet_pairs or [])}
    assert pinned == set(int(i) for i in fem.region_dofs("left", component=0)), (
        "a per-component pin must reach exactly the x-DOFs of the region"
    )

    fem_all = jno.fem([bulk - body, u(*cl) - 0.0])  # all components, for contrast
    pinned_all = {int(dof) for dof, _v in (d._fem_native_dirichlet_pairs or [])}
    assert pinned_all == set(int(i) for i in fem_all.region_dofs("left"))
    assert len(pinned_all) == 2 * len(pinned)


def test_component_index_on_a_scalar_field_raises():
    """A scalar has no components. Before, `u[0]` there sliced quadrature points; now it says so."""
    d = _bar()
    u, _ = d.fem_symbols()
    with pytest.raises(TypeError, match="scalar"):
        u[0]


def test_explicit_ellipsis_is_left_alone():
    """`u[..., i]` was already unambiguous and must keep its exact meaning (and its getitem_key)."""
    d = _bar()
    u, _ = d.fem_symbols(value_shape=(2,))
    assert u[..., 1].getitem_key == (Ellipsis, 1)
    assert u[1].getitem_key == (Ellipsis, 1)  # the raw spelling now agrees
