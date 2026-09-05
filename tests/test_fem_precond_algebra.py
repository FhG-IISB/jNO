"""An algebra over preconditioner specs: `M1 + M2` and `M1 @ M2`.

A materialized spec IS a linear map `v -> M^-1 v`, so the classical physics-based Schur
approximations are arithmetic on specs rather than a menu of built-ins:

    Mp + Ap                                  Cahouet & Chabard (1988) -- which jNO already has as a
                                             built-in, so it doubles as the self-check below
    Ap @ Fp @ Mp   (Fp applied, not inverted) PCD -- Kay, Loghin & Wathen, *SIAM J. Sci. Comput.*
                                             **24** (2002) 237

`jno.precond.saddle()` gains no argument for either. The oracle throughout is the same one the rest
of the preconditioner suite uses: a preconditioner may change how fast a Krylov method converges,
never what it converges to.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(mesh_size=0.35):
    """A single-field system, so an auxiliary form over the same symbols sizes to the whole operator."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    return fem, ui, vi, xi, yi


def _apply(spec, fem, v):
    """Materialize `spec` against `fem`'s own operator and apply it once. Stays in JAX arrays so a
    result can be fed straight back in (the host-LU inner solve will not take a numpy input)."""
    spec.prepare(fem)
    op = LinearOperator(fem.A)
    return materialize_precond(spec, PrecondContext(op, fem))(jnp.asarray(v))


def _apply_T(spec, fem, v):
    spec.prepare(fem)
    op = LinearOperator(fem.A)
    return materialize_precond(spec, PrecondContext(op, fem)).T(jnp.asarray(v))


# ======================================================================================
# the arithmetic itself
# ======================================================================================
def test_a_sum_applies_both_and_adds():
    """`(M1 + M2) v == M1 v + M2 v`, to machine precision."""
    fem, ui, vi, xi, yi = _poisson()
    M1 = jno.precond.form([ui * vi], inner=jno.solve.lu(backend="host"))
    M2 = jno.precond.form([ui * vi + ui.x * vi.x + ui.y * vi.y], inner=jno.solve.lu(backend="host"))
    rng = np.random.default_rng(0)
    v = rng.standard_normal(fem.dofs)
    np.testing.assert_allclose(_apply(M1 + M2, fem, v), _apply(M1, fem, v) + _apply(M2, fem, v), rtol=1e-11)


def test_a_product_applies_right_to_left():
    """`(M1 @ M2) v == M1(M2 v)` -- in the order the operator product reads."""
    fem, ui, vi, xi, yi = _poisson()
    M1 = jno.precond.form([ui * vi], inner=jno.solve.lu(backend="host"))
    M2 = jno.precond.form([ui * vi + ui.x * vi.x + ui.y * vi.y], inner=jno.solve.lu(backend="host"))
    rng = np.random.default_rng(1)
    v = rng.standard_normal(fem.dofs)
    np.testing.assert_allclose(_apply(M1 @ M2, fem, v), _apply(M1, fem, _apply(M2, fem, v)), rtol=1e-11)


def test_inner_false_applies_the_form_instead_of_inverting_it():
    """`form(..., inner=False)` is `M v = A v`, the middle factor of a product like PCD. Checked
    against the assembled auxiliary itself, and shown NOT to be its inverse."""
    fem, ui, vi, xi, yi = _poisson()
    terms = [ui * vi]
    applied = jno.precond.form(terms, inner=False)
    inverted = jno.precond.form(terms, inner=jno.solve.lu(backend="host"))
    rng = np.random.default_rng(2)
    v = rng.standard_normal(fem.dofs)

    # A(A^-1 v) == v is the statement that one really is the other's inverse.
    round_trip = _apply(applied, fem, _apply(inverted, fem, v))
    np.testing.assert_allclose(round_trip, v, rtol=1e-9, atol=1e-11)
    assert not np.allclose(_apply(applied, fem, v), _apply(inverted, fem, v)), "apply must not be invert"


def test_the_product_transpose_reverses_the_order():
    """`(AB)^T = B^T A^T`. This is the reason the product is a spec and not a lambda: the reverse pass
    of a differentiable solve preconditions `A^T`, and a mis-ordered transpose runs it nearly
    unpreconditioned. Uses a NON-symmetric factor, or the test could not tell."""
    fem, ui, vi, xi, yi = _poisson()
    # An advection form: A != A^T, so the order genuinely matters.
    skew = jno.precond.form([ui.x * vi + ui * vi], inner=False)
    mass = jno.precond.form([ui * vi], inner=jno.solve.lu(backend="host"))
    rng = np.random.default_rng(3)
    v = rng.standard_normal(fem.dofs)

    got = _apply_T(mass @ skew, fem, v)
    want = _apply_T(skew, fem, _apply_T(mass, fem, v))  # B^T(A^T v)
    np.testing.assert_allclose(got, want, rtol=1e-10)

    wrong = _apply_T(mass, fem, _apply_T(skew, fem, v))  # the mis-ordered version
    assert not np.allclose(got, wrong), "the factor must be non-symmetric or this proves nothing"


def test_combining_with_a_non_spec_is_refused():
    """A spec plus a number is a type error, not a silently-wrapped scalar."""
    fem, ui, vi, _xi, _yi = _poisson()
    M = jno.precond.form([ui * vi])
    with pytest.raises(TypeError):
        _ = M + 3.0
    with pytest.raises(TypeError):
        _ = M @ "mass"


def test_the_combination_reprs_as_the_math():
    fem, ui, vi, _xi, _yi = _poisson()
    A, B = jno.precond.form([ui * vi]), jno.precond.form([ui.x * vi.x])
    assert " + " in repr(A + B)
    assert " @ " in repr(A @ B)


# ======================================================================================
# the self-check: the algebra must rebuild a built-in
# ======================================================================================
def _stokes(mesh_size=0.3, mu=1.0):
    """Taylor-Hood Poiseuille channel -- the saddle system these blocks exist for."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    G, H, Lx = 1.0, 1.0, 4.0
    u_profile = lambda y: (G / (2 * mu)) * y * (H - y)  # noqa: E731
    d = jno.domain(box(0.0, 0.0, Lx, H), mesh_size=mesh_size)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            mu * inner_(gu, gv, n_contract=2) - pp * trace(gv),
            -qq * trace(gu),
            u(xb, yb)[0] - u_profile(yb),
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )
    return fem, u, p, q, pp, qq, mu, xi, yi


@pytest.mark.slow
def test_a_user_written_schur_solves_the_saddle_system():
    """The claim the algebra exists for.

    `Mp + Ap` -- a weighted pressure mass plus a gauged pressure Laplacian, both ordinary weak forms
    in the same language as the PDE -- is the Cahouet & Chabard (1988) Schur approximation, which jNO
    ships as `saddle(laplace_weight=...)`. Being able to rebuild an existing built-in out of the
    algebra is the check that the algebra is right; and it means `saddle()` needs no `schur=`
    argument for the user to write their own.
    """
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    fem, u, p, q, pp, qq, mu, xi, yi = _stokes()
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))

    host = jno.solve.lu(backend="host")
    # Fresh P1 symbols on the domain, the way the built-in auxiliaries are built: an auxiliary is its
    # own little FEM problem on the pressure space, not a re-use of the primal system's symbols.
    a_sym, b_sym = fem.domain.fem_symbols()
    ai, bi = a_sym.bind(x=xi, y=yi), b_sym.bind(x=xi, y=yi)
    schur = jno.precond.form([(1.0 / mu) * ai * bi], inner=host) + jno.precond.form(
        [0.05 * (ai.x * bi.x + ai.y * bi.y), a_sym.pin()], inner=host
    )
    got = np.asarray(
        fem.solve(
            linear=jno.solve.fgmres(tol=1e-10, restart=60, maxiter=400),
            precond=jno.precond.triangular((u, jno.precond.amg()), (p, schur)),
        )
    )
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
    assert rel < 5e-3, f"a preconditioner changes speed, never the answer (rel {rel:.2e})"
