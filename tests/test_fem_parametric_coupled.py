"""A runtime parameter in a **coupled** (multi-field) form.

The single-field paths have been differentiable in a material parameter for a long time; the coupled
ones were not, unless the form happened to be nonlinear or carry step history. That excluded exactly
the shape a mixed design problem takes -- a Taylor-Hood Stokes system with a design field in a
Brinkman friction term, which is linear in the unknown:

    nu <grad u, grad v> - <p, div v> + alpha(rho) <u, v> = f,   <q, div u> = 0

The exclusion was never an assembly limitation. ``assemble_fem_native`` already builds a parametric
``FemLinearSystem(A(args), b(args))`` from an ``_assemble_at(args)`` closure over the same
residual/jacobian every other path uses, and that closure is field-agnostic. Only the gate that
decides whether a coupled form may reach the native assembler refused.

These tests pin the behaviour that gate was hiding, and the refusals that must still stand.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
NU, EPS = 0.05, 1e-9
A_MAX, Q_RAMP = 5.0e2, 0.1


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stokes_brinkman(size=0.22, parametric=True, alpha_const=0.0, H=0.4, L=1.0):
    """Taylor-Hood Stokes in a duct plus a Brinkman friction term.

    ``parametric`` chooses whether the friction coefficient is a trainable field or a plain constant
    -- the same physics either way, which is what lets the two be compared.
    """
    d = jno.Shape.rect(0, 0, L, H, size=size).domain()
    d.tag("inlet", lambda x, y: x < EPS)
    d.tag("walls", lambda x, y: (y < EPS) | (y > H - EPS))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xin, yin = d.variable("inlet", split=True)[:2]
    xw, yw = d.variable("walls", split=True)[:2]
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)

    rho = None
    if parametric:
        rho_t, _ = d.fem_symbols(names=("rho", "rho_w"), order=1)
        rho = jno.np.parameter(rho_t, name="rho")
        ri = rho.bind(x=xi, y=yi) if hasattr(rho, "bind") else rho
        alpha = A_MAX * Q_RAMP * (1.0 - ri) / (Q_RAMP + ri)
    else:
        alpha = alpha_const

    profile = 4.0 * yin * (H - yin) / H**2
    fem = jno.fem(
        [
            NU * inner(gu, gv, n_contract=2) - pp * trace(gv) + alpha * inner(ub, vv, n_contract=1),
            -qq * trace(gu),
            u(xin, yin)[0] - profile,
            u(xin, yin)[1] - 0.0,
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
        ]
    )
    return d, fem, rho


def test_a_design_field_in_a_coupled_linear_form_assembles():
    """The headline: this used to raise NotImplementedError."""
    _, fem, _ = _stokes_brinkman()
    assert fem.is_linear, "Brinkman friction is linear in u; the form must not be reclassified"
    assert len(fem.offsets) == 3, "two blocks (velocity, pressure) expected"
    assert getattr(fem._op, "is_parametric", False), "the operator must re-assemble at runtime args"
    assert getattr(fem._op, "operator_fn", None) is not None


def test_a_non_parametric_coupled_form_still_takes_the_plain_matrix_route():
    """The change must not quietly move existing users onto a parametric operator: without a
    parameter, a coupled linear form still assembles to a frozen (A, b) pair."""
    _, fem, _ = _stokes_brinkman(parametric=False, alpha_const=7.0)
    assert fem.is_linear
    assert not getattr(fem._op, "is_parametric", False), "no parameter => no re-assembly machinery"
    assert np.asarray(fem.A.todense() if hasattr(fem.A, "todense") else fem.A).ndim == 2


def test_the_parametric_solve_matches_the_constant_form_at_the_same_coefficient():
    """The oracle: freeze the design field at a uniform value and the parametric system must give the
    same field as writing that coefficient as a plain constant. Same physics, two spellings."""
    rho0 = 0.5
    alpha0 = A_MAX * Q_RAMP * (1.0 - rho0) / (Q_RAMP + rho0)

    _, fem_c, _ = _stokes_brinkman(parametric=False, alpha_const=alpha0)
    ref = np.asarray(fem_c.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)

    _, fem_p, rho = _stokes_brinkman(parametric=True)
    rho.dtype(jnp.float64)
    rho.initialize(jax.nn.initializers.constant(rho0))
    sol = fem_p.solve(linear=jno.solve.lu(backend="host"))
    crux = jno.core([sol.mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    got = np.asarray(crux.eval([sol])).reshape(-1)

    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-11)


def test_the_gradient_to_a_design_FIELD_matches_a_finite_difference():
    """What fluid topology optimisation actually needs: d(objective)/d(rho_i) for a nodal design field.

    Differentiated straight through ``A(args) x = b(args)``, so the check covers the re-assembly the
    parametric path is built on -- and, because the derivative is taken per node, it is sensitive to
    the gather being right. A wrong-node gather would still produce finite gradients; it would just
    attribute them to the wrong nodes, which a component-wise finite difference catches.
    """
    _, fem, rho = _stokes_brinkman(size=0.25, parametric=True)
    n_design = int(np.asarray(fem.field_points[1]).shape[0])
    rng = np.random.default_rng(0)
    rho0 = jnp.asarray(0.35 + 0.3 * rng.random(n_design))  # non-uniform, safely inside (0, 1)

    def objective(rv):
        args = {"rho": rv}
        A = fem._op.operator_fn(args)
        b = fem._op.rhs_fn(args)
        dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A)
        x = jnp.linalg.solve(dense, jnp.asarray(b).reshape(-1))
        return jnp.sum(x**2)

    g = np.asarray(jax.grad(objective)(rho0))
    assert g.shape == (n_design,)
    assert np.all(np.isfinite(g)) and np.abs(g).max() > 0

    # component-wise central differences on a few nodes, including a boundary one
    h = 1e-6
    for i in (0, n_design // 3, n_design - 1):
        e = jnp.asarray(np.eye(1, n_design, i).reshape(-1))
        fd = float((objective(rho0 + h * e) - objective(rho0 - h * e)) / (2 * h))
        rel = abs(g[i] - fd) / max(abs(fd), 1e-9)
        assert rel < 1e-5, f"node {i}: AD {g[i]:.8e} vs FD {fd:.8e} (rel {rel:.2e})"


def test_a_parametric_dirichlet_value_on_a_coupled_form_still_refuses_by_name():
    """The adjacent case the old gate also stood in front of. If it cannot assemble, it must say so
    specifically -- not with the general 'no parametric route' claim, which is no longer true."""
    H, L = 0.4, 1.0
    d = jno.Shape.rect(0, 0, L, H, size=0.25).domain()
    d.tag("inlet", lambda x, y: x < EPS)
    d.tag("walls", lambda x, y: (y < EPS) | (y > H - EPS))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    g_t, _ = d.fem_symbols(names=("gpar", "gpar_w"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xin, yin = d.variable("inlet", split=True)[:2]
    xw, yw = d.variable("walls", split=True)[:2]
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    gpar = jno.np.parameter(g_t, name="gpar")
    gb = gpar.bind(x=xin, y=yin) if hasattr(gpar, "bind") else gpar
    terms = [
        NU * inner(gu, gv, n_contract=2) - pp * trace(gv),
        -qq * trace(gu),
        u(xin, yin)[0] - gb,  # a PARAMETER in the essential value
        u(xin, yin)[1] - 0.0,
        u(xw, yw)[0] - 0.0,
        u(xw, yw)[1] - 0.0,
    ]
    with pytest.raises(NotImplementedError, match="ESSENTIAL VALUE") as ei:
        jno.fem(terms)
    msg = str(ei.value)
    # It must name ITS OWN reason. The coupled linear assembly does have a parametric route now, so
    # blaming that one would send the reader to work around a limitation they do not have.
    assert "no parametric route" not in msg
    assert "node ids" in msg or "numbering" in msg, f"the refusal should say why: {msg}"


def test_a_non_uniform_design_field_is_evaluated_at_the_right_nodes():
    """The sharp version of the oracle above.

    A *uniform* design field agrees with a constant coefficient no matter which nodes the parameter is
    read from, so it cannot detect a mis-indexed lookup. Initialise rho to a known function of position
    and compare against the same coefficient written directly as a coordinate expression: the two agree
    only if the parameter's nodal values are gathered at the coordinates they belong to.
    """
    H, L, size = 0.4, 1.0, 0.22
    f = lambda x, y: 0.3 + 0.4 * x / L + 0.2 * y / H  # noqa: E731  -- varies in BOTH directions

    # (a) coefficient written as a plain coordinate expression, no parameter anywhere
    d = jno.Shape.rect(0, 0, L, H, size=size).domain()
    d.tag("inlet", lambda x, y: x < EPS)
    d.tag("walls", lambda x, y: (y < EPS) | (y > H - EPS))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xin, yin = d.variable("inlet", split=True)[:2]
    xw, yw = d.variable("walls", split=True)[:2]
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    rho_expr = 0.3 + 0.4 * xi / L + 0.2 * yi / H
    alpha_expr = A_MAX * Q_RAMP * (1.0 - rho_expr) / (Q_RAMP + rho_expr)
    profile = 4.0 * yin * (H - yin) / H**2
    fem_e = jno.fem(
        [
            NU * inner(gu, gv, n_contract=2) - pp * trace(gv) + alpha_expr * inner(ub, vv, n_contract=1),
            -qq * trace(gu),
            u(xin, yin)[0] - profile,
            u(xin, yin)[1] - 0.0,
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
        ]
    )
    ref = np.asarray(fem_e.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)

    # (b) the same coefficient carried by a trainable field initialised to that function
    d2, fem_p, rho = _stokes_brinkman(size=size, parametric=True, H=H, L=L)
    nodes = np.asarray(fem_p.field_points[1])  # the P1 (pressure/design) nodes
    rho.dtype(jnp.float64)
    # Ignore the dtype jNO passes in. The initializer runs BEFORE the float64 cast, so honouring it
    # rounds the design field to float32 and puts a ~3e-8 floor under every comparison below.
    rho.initialize(lambda key, shape, dtype=None: jnp.asarray(f(nodes[:, 0], nodes[:, 1]), jnp.float64))
    sol = fem_p.solve(linear=jno.solve.lu(backend="host"))
    crux = jno.core([sol.mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    got = np.asarray(crux.eval([sol])).reshape(-1)

    assert got.shape == ref.shape
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    # Gate at 1e-6, set by a measured and unrelated precision floor rather than by taste: the parameter
    # store initialises in float32 and casts to float64 AFTERWARDS, so the design field jNO holds differs
    # from the values handed to `initialize` by ~2.6e-8 whatever dtype the initializer returns. That puts
    # ~2e-8 under this comparison and no interpolation change can move it.
    #
    # It still bites hard on what it exists to catch. Gathering P1 nodal values with a P2 field's node
    # ids -- which JAX CLAMPS rather than refusing, so the excess silently reads the last node -- measured
    # 4.4e-1 here. That is five orders above this gate, and invisible to a uniform design field, which is
    # exactly why this test uses one that varies in both directions.
    assert err < 1e-6, f"a non-uniform design field disagrees with the same coefficient inline: {err:.3e}"


def test_a_field_parameter_shared_across_fields_of_DIFFERING_orders():
    """Previously refused; now correct, which is what licenses removing the refusal.

    A material coefficient common to two coupled equations -- here a P1 field `k(x)` multiplying both a
    P2-tested and a P1-tested term -- used to raise: the parameter was interpolated on "the" field's
    space, so two fields of different order had no shared node set and the choice was ambiguous. It
    interpolates on its own P1 basis now, so the ambiguity is gone: there is nothing to choose.

    The oracle is the same coefficient written inline as a coordinate expression. `k` is affine, so its
    P1 interpolant is exact and the two must agree to the parameter store's precision floor.
    """
    H, L, size = 0.4, 1.0, 0.25
    kf = lambda x, y: 0.5 + 0.3 * x / L - 0.2 * y / H  # noqa: E731  affine -> exact in P1

    def build(parametric):
        d = jno.Shape.rect(0, 0, L, H, size=size).domain()
        d.tag("walls", lambda x, y: (x < EPS) | (x > L - EPS) | (y < EPS) | (y > H - EPS))
        a, pa = d.fem_symbols(names=("a", "pa"), order=2)  # P2 field
        b, pb = d.fem_symbols(names=("b", "pb"), order=1)  # P1 field
        xi, yi = d.variable("interior", split=True)[:2]
        xw, yw = d.variable("walls", split=True)[:2]
        ai, vi = a.bind(x=xi, y=yi), pa.bind(x=xi, y=yi)
        bi, wi = b.bind(x=xi, y=yi), pb.bind(x=xi, y=yi)
        if parametric:
            kt, _ = d.fem_symbols(names=("kk", "kk_w"), order=1)
            par = jno.np.parameter(kt, name="kk")
            k = par.bind(x=xi, y=yi)
        else:
            par, k = None, kf(xi, yi)
        fem = jno.fem(
            [
                k * (ai.x * vi.x + ai.y * vi.y) - 1.0 * vi - bi * vi,  # k in a P2-TESTED term
                k * (bi.x * wi.x + bi.y * wi.y) + bi * wi - 2.0 * ai * wi,  # ...and a P1-TESTED one
                a(xw, yw) - 0.0,
                b(xw, yw) - 0.0,
            ]
        )
        return d, fem, par

    _, fem_ref, _ = build(parametric=False)
    ref = np.asarray(fem_ref.solve(linear=jno.solve.lu(backend="host"))).reshape(-1)

    with pytest.warns(UserWarning, match="resolved more coarsely"):
        _, fem_p, par = build(parametric=True)
    nodes = np.asarray(fem_p.field_points[1])  # the P1 block's nodes == the parameter's own
    par.dtype(jnp.float64)
    par.initialize(lambda key, shape, dtype=None: jnp.asarray(kf(nodes[:, 0], nodes[:, 1]), jnp.float64))
    sol = fem_p.solve(linear=jno.solve.lu(backend="host"))
    crux = jno.core([sol.mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    got = np.asarray(crux.eval([sol])).reshape(-1)

    assert got.shape == ref.shape
    err = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
    assert err < 1e-6, f"a shared field parameter across differing orders disagrees with inline: {err:.3e}"


def test_the_coarser_resolution_warning_is_not_raised_when_every_field_is_P1():
    """The warning must mark a real asymmetry, not fire on every coupled parametric form."""
    H, L = 0.4, 1.0
    d = jno.Shape.rect(0, 0, L, H, size=0.25).domain()
    d.tag("walls", lambda x, y: (x < EPS) | (x > L - EPS) | (y < EPS) | (y > H - EPS))
    a, pa = d.fem_symbols(names=("a1", "pa1"), order=1)
    b, pb = d.fem_symbols(names=("b1", "pb1"), order=1)
    kt, _ = d.fem_symbols(names=("kk1", "kk1_w"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xw, yw = d.variable("walls", split=True)[:2]
    ai, vi = a.bind(x=xi, y=yi), pa.bind(x=xi, y=yi)
    bi, wi = b.bind(x=xi, y=yi), pb.bind(x=xi, y=yi)
    k = jno.np.parameter(kt, name="kk1").bind(x=xi, y=yi)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jno.fem(
            [
                k * (ai.x * vi.x + ai.y * vi.y) - 1.0 * vi - bi * vi,
                k * (bi.x * wi.x + bi.y * wi.y) + bi * wi - 2.0 * ai * wi,
                a(xw, yw) - 0.0,
                b(xw, yw) - 0.0,
            ]
        )
    assert [w for w in caught if "resolved more coarsely" in str(w.message)] == []
