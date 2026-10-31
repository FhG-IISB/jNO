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
# a combination is a spec like any other: it compiles, keeps its children's flags, and the
# tree walkers see through it
# ======================================================================================
def test_a_combination_of_traceable_specs_compiles():
    """`jacobi() + jacobi()` is as traceable as its children, so it takes the compiled slot path; equal
    combinations are equal values (one program), and a float32 child is a different one."""
    from jno.utils.solver.solver_api import _compilable

    J = jno.precond.jacobi
    assert _compilable(jno.solve.cg(), J() + J())
    assert _compilable(jno.solve.cg(), J() @ J())
    assert J() + J() == J() + J() and hash(J() + J()) == hash(J() + J())
    assert J() + J() != J() + J(float32=True)
    fem, *_ = _poisson()
    ref = np.asarray(fem.solve())
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10), precond=J() + J()))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-12)


def test_a_combination_with_a_host_leaf_does_not_compile():
    from jno.utils.solver.solver_api import _compilable

    pytest.importorskip("pyamg")
    assert not _compilable(jno.solve.cg(), jno.precond.jacobi() + jno.precond.amg())


def test_a_product_is_flagged_nonsymmetric_and_cg_still_solves():
    """`M1 M2` of two symmetric factors is not symmetric. Flagged, `cg` switches to flexible CG on its
    own (`jno.solve.cg`), instead of running standard CG on a preconditioner it is not valid for."""
    fem, *_ = _poisson()
    J = jno.precond.jacobi
    op = LinearOperator(fem.A)
    assert materialize_precond(J() @ J(), PrecondContext(op, fem)).nonsymmetric
    assert not materialize_precond(J() + J(), PrecondContext(op, fem)).nonsymmetric
    lo = materialize_precond(J() + J(float32=True), PrecondContext(op, fem))
    assert lo.low_precision, "a single-precision child makes the combination single-precision"
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10), precond=J() @ J()))
    np.testing.assert_allclose(got, np.asarray(fem.solve()), rtol=1e-7, atol=1e-12)


def test_the_tree_walkers_see_inside_a_combination():
    """A leaf inside `+` is reached by every walk: the refresh hook fires once per child (not twice, as
    it would if the combination also forwarded it), an untraceable leaf is found, and `_uncached` strips a
    `reuse=k` child so a march's chunk cadence is the only policy."""
    from jno.precond import _Spec
    from jno.utils.solver import solver_api as sa

    calls = []

    class _Hook(_Spec):
        traceable = True
        key = ("hook",)

        def refresh_from(self, sol, fem):
            calls.append(self)

        def materialize(self, ctx):
            return lambda v: v

    a, b = _Hook(), _Hook()
    tree = a + jno.precond.jacobi() @ b
    sa.refresh_hooks(tree, None, None)
    assert sorted(map(id, calls)) == sorted([id(a), id(b)])
    calls.clear()
    tree.refresh_from(None, None)  # a direct call on the composition reaches the same leaves, once each
    assert sorted(map(id, calls)) == sorted([id(a), id(b)])
    assert (jno.precond.jacobi() + jno.precond.jacobi()).refresh_from is None

    pytest.importorskip("pyamg")
    leaves, untraceable = sa._untraceable_leaves(jno.precond.jacobi() + jno.precond.amg())
    assert untraceable and len(leaves) == 2

    kept = jno.precond.jacobi(reuse=3)
    assert sa._refresh_cadence(jno.precond.jacobi() + kept) == 3
    stripped = sa._uncached(jno.precond.jacobi() + kept)
    assert type(stripped).__name__ == "_Sum" and getattr(stripped.right, "refresh", None) is None


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


def test_an_auxiliary_form_is_assembled_sparse():
    """`PrecondContext.assemble` reached for `fem.A`, which is DOCUMENTED dense ("use fem.operator for
    the raw sparse form on large problems"). So every auxiliary cost n^2 -- measured at 1.74 GB for a
    velocity-space mass at 14,739 dofs, i.e. a preconditioner running out of memory doing the one
    thing it exists to avoid. It was the memory wall of the whole iterative path while every other
    stage stayed sparse.

    A mass matrix has O(n) entries; anything near n^2 means the dense path came back."""
    fem, ui, vi, _xi, _yi = _poisson(mesh_size=0.12)
    op = PrecondContext(None, fem).assemble([ui * vi])
    assert op.bcoo is not None, "the auxiliary must be sparse, not a densified array"
    n = int(op.shape[0])
    assert int(op.bcoo.nse) < 40 * n, f"{op.bcoo.nse} entries for an {n}-dof mass is not sparse"


# ======================================================================================
# `M1 >> M2`: multiplicative combination (subspace correction)
# ======================================================================================
def _dense_spec(M, *, symmetric=False, count=None):
    """A spec applying the dense matrix `M` (and `M^T` as its transpose), for exact-algebra checks."""
    from jno.precond import _Spec
    from jno.utils.solver.solver_api import PrecondApplier

    M = jnp.asarray(M)

    class _D(_Spec):
        traceable = True

        def materialize(self, ctx):
            if count is not None:
                count.append(self)
            return PrecondApplier(lambda v: M @ v, None if symmetric else (lambda v: M.T @ v))

        def __repr__(self):
            return "D"

    return _D()


def _rand(n, seed, *, spd=False):
    a = np.random.default_rng(seed).standard_normal((n, n)) / n
    return a @ a.T + np.eye(n) if spd else a + np.eye(n)


def test_then_is_the_multiplicative_combination():
    """`(M1 >> M2) r = M1 r + M2 (r - A M1 r)`: the error propagator is the PRODUCT of the two,
    `I - B A = (I - M2 A)(I - M1 A)` (Xu, SIAM Rev. 34, 1992, Sec. 3), for a non-symmetric A and factors,
    and a chain of three is the product of three."""
    n = 40
    A, M1, M2, M3 = _rand(n, 0), _rand(n, 1), _rand(n, 2), _rand(n, 3)
    ctx = PrecondContext(LinearOperator(jax.experimental.sparse.BCOO.fromdense(jnp.asarray(A))), None)
    eye = np.eye(n)
    B2 = np.column_stack([np.asarray(materialize_precond(_dense_spec(M1) >> _dense_spec(M2), ctx)(e)) for e in eye])
    np.testing.assert_allclose(B2, M1 + M2 - M2 @ A @ M1, rtol=1e-12, atol=1e-12)
    chain = _dense_spec(M1) >> _dense_spec(M2) >> _dense_spec(M3)
    B3 = np.column_stack([np.asarray(materialize_precond(chain, ctx)(e)) for e in eye])
    E = (eye - M3 @ A) @ (eye - M2 @ A) @ (eye - M1 @ A)
    np.testing.assert_allclose(eye - B3 @ A, E, rtol=1e-10, atol=1e-10)
    assert len(chain.children) == 3, "a chain is flattened, not nested"


def test_then_has_the_exact_transpose():
    """The adjoint solve preconditions `A^T` with `B^T`, which is the reversed chain of transposes on `A^T`."""
    n = 30
    A, M1, M2, M3 = _rand(n, 4), _rand(n, 5), _rand(n, 6), _rand(n, 7)
    ctx = PrecondContext(LinearOperator(jax.experimental.sparse.BCOO.fromdense(jnp.asarray(A))), None)
    ap = materialize_precond(_dense_spec(M1) >> _dense_spec(M2) >> _dense_spec(M3), ctx)
    eye = np.eye(n)
    B = np.column_stack([np.asarray(ap(e)) for e in eye])
    Bt = np.column_stack([np.asarray(ap.T(e)) for e in eye])
    np.testing.assert_allclose(Bt, B.T, rtol=1e-11, atol=1e-11)


def test_then_symmetry_and_shared_children():
    """`Q >> M >> Q` of symmetric children is symmetric (for symmetric A), so plain CG may use it; `M >> Q`
    is not, and says so. A child that appears twice is materialized once."""
    n = 30
    A = _rand(n, 8, spd=True)
    ctx = PrecondContext(LinearOperator(jax.experimental.sparse.BCOO.fromdense(jnp.asarray(A))), None)
    built = []
    Q = _dense_spec(np.linalg.inv(A) * 0.5, symmetric=True, count=built)
    M = _dense_spec(np.diag(1 / np.diag(A)), symmetric=True)
    bnn = materialize_precond(Q >> M >> Q, ctx)
    assert not bnn.nonsymmetric and len(built) == 1
    B = np.column_stack([np.asarray(bnn(e)) for e in np.eye(n)])
    np.testing.assert_allclose(B, B.T, rtol=1e-11, atol=1e-12)
    assert materialize_precond(M >> Q, ctx).nonsymmetric
    assert repr(Q >> M >> Q) == "(D >> D >> D)"


def test_then_solves_through_fem_solve():
    """A preconditioner changes the iteration count, never the answer -- also compiled (both children trace)."""
    from jno.utils.solver.solver_api import _compilable

    fem, *_ = _poisson()
    J = jno.precond.jacobi
    assert _compilable(jno.solve.cg(), J() >> J())
    ref = np.asarray(fem.solve())
    for spec in (J() >> J(), J() >> J() >> J()):
        got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10), precond=spec))
        np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-12)
    with pytest.raises(TypeError):
        _ = J() >> 2.0
