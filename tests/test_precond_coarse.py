"""jno.precond.coarse(): the coarse-space correction ``Q r = Z (Z^T A Z)^{-1} Z^T r`` and its combinations.

Oracles: the dense projection (``Q`` is exactly ``Z E^{-1} Z^T``, global or split over parts); the spectrum of
the deflated operator (with ``Z`` the lowest eigenvectors, A-DEF2 maps them to 1 and leaves the rest of the
spectrum alone -- Tang, Nabben, Vuik & Erlangga, J. Sci. Comput. 39, 2009, Sec. 3); the default solve (a
preconditioner never changes the answer); iteration counts where the coarse space is the point (constants
per part on Poisson, high-contrast inclusions, rigid-body modes on elasticity); and the refusals (a kernel in
the space, a non-nodal field, a size that does not match).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

pytest.importorskip("pymetis", reason="partitioned coarse spaces use METIS: the [metis] extra")

R = jno.shape.rect


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.08, *, dirichlet=True):
    d = R(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    terms = [ui.x * vi.x + ui.y * vi.y - 10.0 * vi]
    if dirichlet:
        terms.append(u(xb, yb) - 0.0)
    return jno.fem(terms), (x, y)


def _matrix(spec, fem):
    """The dense matrix of ``spec`` applied to ``fem``'s own operator, column by column."""
    spec.prepare(fem)
    A = fem._op[0]
    ap = materialize_precond(spec, PrecondContext(LinearOperator(A), fem))
    n = A.shape[0]
    return np.column_stack([np.asarray(ap(jnp.asarray(e))) for e in np.eye(n)]), np.asarray(A.todense())


def _iso(A):
    """Rows with only a diagonal: the eliminated Dirichlet rows the coarse space leaves out."""
    off = A - np.diag(np.diag(A))
    return ~np.any(off != 0, axis=1) & ~np.any(off != 0, axis=0)


def _iterations(fem, precond, linear=None):
    fem.solve(linear=linear or jno.solve.cg(tol=1e-10), precond=precond)
    return fem.stats["linear_iterations"]["total"]


# ======================================================================================
# the operator itself
# ======================================================================================
def test_a_global_space_is_the_dense_projection():
    fem, _ = _poisson()
    n = fem._op[0].shape[0]
    Z = np.random.default_rng(0).standard_normal((n, 3))
    Q, A = _matrix(jno.precond.coarse(Z), fem)
    Z = np.where(_iso(A)[:, None], 0.0, Z)
    np.testing.assert_allclose(Q, Z @ np.linalg.solve(Z.T @ A @ Z, Z.T), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(Q @ A @ Z, Z, rtol=1e-8, atol=1e-10)  # exact on span Z


def test_a_partitioned_space_is_the_dense_projection_of_its_split_vectors():
    """``[1, x]`` over 6 METIS parts is 12 coarse vectors: each formula restricted to each part."""
    fem, (x, y) = _poisson()
    spec = jno.precond.coarse([1, x], parts=6)
    Q, A = _matrix(spec, fem)
    pat = spec._pattern
    part, null = np.asarray(pat.part), np.asarray(pat.null)
    pts = np.asarray(fem.field_points[0])
    np.testing.assert_allclose(null[part < pat.p, 1], pts[part < pat.p, 0])  # the formula x, at the nodes
    Z = np.zeros((A.shape[0], pat.p * 2))
    for i in range(pat.p):
        Z[part == i, 2 * i : 2 * i + 2] = null[part == i]
    assert pat.p == 6 and np.all(part[_iso(A)] == pat.p)
    np.testing.assert_allclose(Q, Z @ np.linalg.solve(Z.T @ A @ Z, Z.T), rtol=1e-8, atol=1e-11)


def test_adef2_maps_the_deflated_eigenvectors_to_one_and_keeps_the_rest():
    """With ``M = I`` and ``Z`` the ``k`` lowest eigenvectors, ``(M >> Q) A`` has eigenvalues 1 (k times) and
    the remaining ``n - k`` eigenvalues of ``A`` -- the small ones are gone, which is what deflation is."""
    from jno.precond import _Spec

    class _Identity(_Spec):
        traceable = True
        key = ("identity",)

        def materialize(self, ctx):
            return lambda v: v

    fem, _ = _poisson(0.12)
    A = np.asarray(fem._op[0].todense())
    keep = ~_iso(A)
    lam, V = np.linalg.eigh(A[np.ix_(keep, keep)])
    k = 4
    Z = np.zeros((A.shape[0], k))
    Z[keep] = V[:, :k]
    B, _ = _matrix(_Identity() >> jno.precond.coarse(Z), fem)
    got = np.sort(np.linalg.eigvals((B @ A)[np.ix_(keep, keep)]).real)
    want = np.sort(np.concatenate([np.ones(k), lam[k:]]))
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-8)


def test_bnn_is_symmetric_and_the_others_say_they_are_not():
    fem, (x, y) = _poisson()
    J, Q = jno.precond.jacobi(), jno.precond.coarse([1, x, y])
    B, _ = _matrix(Q >> J >> Q, fem)
    np.testing.assert_allclose(B, B.T, rtol=1e-10, atol=1e-12)
    ctx = PrecondContext(LinearOperator(fem._op[0]), fem)
    assert not materialize_precond(Q >> J >> Q, ctx).nonsymmetric
    assert materialize_precond(J >> Q, ctx).nonsymmetric


# ======================================================================================
# through fem.solve: the answer is the default one, the iterations drop
# ======================================================================================
@pytest.mark.parametrize("form", ["adef1", "adef2", "bnn", "additive"])
def test_every_combination_gives_the_default_answer(form):
    fem, (x, y) = _poisson(0.04)
    ref = np.asarray(fem.solve())
    J, Q = jno.precond.jacobi(), jno.precond.coarse([1], parts=16)
    spec = {"adef1": Q >> J, "adef2": J >> Q, "bnn": Q >> J >> Q, "additive": Q + J}[form]
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=spec))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())


def test_constants_per_part_cut_the_jacobi_iterations():
    fem, _ = _poisson(0.015)
    plain = _iterations(fem, jno.precond.jacobi())
    deflated = _iterations(fem, jno.precond.jacobi() >> jno.precond.coarse([1], parts=64))
    assert deflated < 0.6 * plain, (plain, deflated)


def _inclusions(size=0.02, contrast=1e6):
    d = (
        (
            R(0.20, 0.20, 0.30, 0.30).name("inc1").attach(k=contrast)
            + R(0.60, 0.50, 0.70, 0.60).name("inc2").attach(k=contrast)
            + R(0.40, 0.75, 0.50, 0.85).name("inc3").attach(k=contrast)
            + R(0.00, 0.00, 1.00, 1.00).name("matrix").attach(k=1.0)
        )
        .sized(size)
        .domain()
    )
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    return jno.fem([d.k * (ui.x * vi.x + ui.y * vi.y) - vi, u(xb, yb) - 0.0])


def test_one_vector_per_inclusion_removes_the_contrast_from_the_iterations():
    """Each stiff inclusion contributes one tiny eigenvalue (its near-constant mode); one deflation vector per
    inclusion removes them (Vuik, Segal & Meijerink, J. Comput. Phys. 152, 1999). Same answer; Jacobi-CG
    iterations grow with the contrast (measured 226 -> 382 for 1e2 -> 1e6), deflated they barely move
    (160 -> 192). (Smoothed-aggregation AMG already resolves these modes here; deflation buys nothing on top.)"""
    its = {}
    for contrast in (1e2, 1e6):
        fem = _inclusions(contrast=contrast)
        Q = jno.precond.coarse([1], parts=["inc1", "inc2", "inc3"])
        its[contrast] = (
            _iterations(fem, jno.precond.jacobi(), jno.solve.cg(tol=1e-8)),
            _iterations(fem, jno.precond.jacobi() >> Q, jno.solve.cg(tol=1e-8)),
        )
        if contrast == 1e2:  # the answer, where the conditioning lets a Krylov solve reach a tight tolerance
            ref = np.asarray(fem.solve(linear=jno.solve.lu()))
            got = np.asarray(fem.solve(linear=jno.solve.fgmres(tol=1e-11, restart=200), precond=jno.precond.jacobi() >> Q))
            np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())
    assert Q._pattern.p == 4, "three inclusions and the rest of the domain"
    (p_lo, d_lo), (p_hi, d_hi) = its[1e2], its[1e6]
    assert p_hi > 1.4 * p_lo and d_hi < 1.3 * d_lo and d_hi < 0.6 * p_hi, its


def _cantilever(size=0.05):
    d = R(0.0, 0.0, 4.0, 1.0, size=size).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, phi = d.fem_symbols(value_shape=(2,))
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    eu, ev = jno.np.symgrad(ui, [x, y]), jno.np.symgrad(vi, [x, y])
    ddot = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    fem = jno.fem(
        [
            2.0 * ddot(eu, ev) + jno.np.trace(eu) * jno.np.trace(ev) + 0.1 * vi[1],
            u(xl, yl)[0] - 0.0,
            u(xl, yl)[1] - 0.0,
        ]
    )
    return fem, (x, y)


def test_rigid_body_formulas_with_shared_schwarz_parts_keep_elasticity_flat():
    """The rigid-body modes written as formulas, on the Schwarz subdomains: BNN iterations stay flat as the
    parts grow (Toselli & Widlund 2005, Sec. 3.8), and the answer is the default one."""
    fem, (x, y) = _cantilever()
    ref = np.asarray(fem.solve())
    its = {}
    for p in (8, 64):
        S = jno.precond.schwarz(parts=p, coarse=False)
        Q = jno.precond.coarse([(1, 0), (0, 1), (-y, x)], parts=S)
        got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=Q >> S >> Q))
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6 * np.abs(ref).max())
        its[p] = fem.stats["linear_iterations"]["total"]
    assert its[64] < 1.3 * its[8], its


# ======================================================================================
# spaces: formulas per field, arrays, callables
# ======================================================================================
def test_a_formula_space_on_a_vector_field_and_a_scalar_broadcast():
    fem, (x, y) = _cantilever(0.2)
    from jno.utils.solver.deflation import resolve_space

    n = fem._op[0].shape[0]
    Z = resolve_space([(1, 0), (0, 1), (-y, x), 1], fem, n)
    pts = np.asarray(fem.field_points[0])
    np.testing.assert_array_equal(Z[0::2, 0], 1.0)
    np.testing.assert_array_equal(Z[1::2, 0], 0.0)
    np.testing.assert_allclose(Z[0::2, 2], -pts[:, 1])
    np.testing.assert_allclose(Z[1::2, 2], pts[:, 0])
    np.testing.assert_array_equal(Z[:, 3], 1.0)  # one formula fills every component
    with pytest.raises(ValueError, match="components"):
        resolve_space([(1, 0, 0)], fem, n)


def test_a_dict_space_names_its_field_and_a_list_is_refused_on_several():
    pytest.importorskip("shapely")
    d = R(0.0, 0.0, 2.0, 1.0, size=0.25).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    gu, gv = jno.np.grad(u, [x, y]), jno.np.grad(v, [x, y])
    pp, qq = p.bind(x=x, y=y), q.bind(x=x, y=y)
    fem = jno.fem(
        [
            jno.np.inner(gu, gv, n_contract=2) - pp * jno.np.trace(gv),
            -qq * jno.np.trace(gu),
            u(xb, yb) - (0.0, 0.0),
            p.pin(),
        ]
    )
    from jno.utils.solver.deflation import resolve_space

    n = fem.blocks[-1].stop
    Z = resolve_space({p: [1]}, fem, n)
    sl = fem.blocks[fem.block_index(p)]
    assert Z.shape == (n, 1) and np.all(Z[sl, 0] == 1.0) and np.all(np.delete(Z[:, 0], np.r_[sl]) == 0.0)
    with pytest.raises(ValueError, match="fields"):
        resolve_space([1], fem, n)


def test_arrays_and_callables_are_checked_against_the_solved_size():
    fem, _ = _poisson()
    n = fem._op[0].shape[0]
    from jno.utils.solver.deflation import resolve_space

    assert resolve_space(lambda ctx: np.ones((ctx.n, 2)), fem, n).shape == (n, 2)
    with pytest.raises(ValueError, match="rows"):
        resolve_space(np.ones((n + 3, 1)), fem, n)
    with pytest.raises(ValueError, match="non-finite"):
        resolve_space(np.full((n, 1), np.nan), fem, n)


def test_a_formula_space_on_a_periodic_system_is_restricted_to_the_kept_dofs():
    """On ``P^T A P`` the vectors are built on the full space and gathered at the kept DOFs; a translation
    along the periodic direction is in the reduced space exactly, so it stays in the kernel of the
    traction-free operator."""
    pytest.importorskip("shapely")
    d = R(0.0, 0.0, 2.0, 1.0, size=0.25).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    u, phi = d.fem_symbols(value_shape=(2,))
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    eu, ev = jno.np.symgrad(ui, [x, y]), jno.np.symgrad(vi, [x, y])
    fem = jno.fem([2.0 * jno.np.inner(eu, ev, n_contract=2) + jno.np.trace(eu) * jno.np.trace(ev), u(xl, yl) - u(xr, yr)])
    from jno.precond import _representative_operator
    from jno.utils.solver.deflation import resolve_space

    A = _representative_operator(fem)
    n = A.shape[0]
    assert n < fem.blocks[-1].stop, "the periodic tie must reduce the system for this test to mean anything"
    Z = resolve_space([(1, 0), (0, 1), (-y, x)], fem, n)
    Ad = np.asarray(A.todense())
    for c in (0, 1):  # both translations: in the reduced space, and in the kernel
        assert np.abs(Ad @ Z[:, c]).max() < 1e-10 * np.abs(Ad).max()


# ======================================================================================
# refusals
# ======================================================================================
def test_a_kernel_in_the_space_is_refused_with_the_fix():
    """Pure Neumann Poisson: the constant is the operator's kernel, not a slow mode."""
    fem, _ = _poisson(dirichlet=False)
    with pytest.raises(ValueError, match="KERNEL"):
        _matrix(jno.precond.coarse([1]), fem)


def test_a_near_kernel_on_a_fine_mesh_is_not_mistaken_for_a_kernel():
    """A Dirichlet condition on one short edge only: the constant is a slow mode, not a kernel."""
    d = R(0.0, 0.0, 1.0, 1.0, size=0.015).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - vi, u(xl, yl) - 0.0])
    ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    got = np.asarray(
        fem.solve(linear=jno.solve.cg(tol=1e-11), precond=jno.precond.jacobi() >> jno.precond.coarse([1, x, y], parts=64))
    )
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())


def test_a_part_of_only_dirichlet_rows_is_dropped_not_refused():
    fem, _ = _poisson()
    n = fem._op[0].shape[0]
    A = np.asarray(fem._op[0].todense())
    Z = np.zeros((n, 2))
    Z[~_iso(A), 0] = 1.0
    Z[_iso(A), 1] = 1.0  # lives only on eliminated rows: no vector once they are left out
    spec = jno.precond.coarse(Z)
    Q, _ = _matrix(spec, fem)
    assert spec._pattern.dropped == 1
    z = Z[:, :1]
    np.testing.assert_allclose(Q, z @ np.linalg.solve(z.T @ A @ z, z.T), rtol=1e-9, atol=1e-12)


def test_formulas_on_a_non_nodal_field_are_refused():
    d = R(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    x, y, _ = d.variable("interior", split=True)
    u, phi = d.fem_symbols(space="Hermite")
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui * vi - vi])
    from jno.utils.solver.deflation import resolve_space

    with pytest.raises(ValueError, match="Hermite"):
        resolve_space([1], fem, fem.dofs)


def test_other_refusals():
    fem, _ = _poisson()
    with pytest.raises(ValueError, match="parts"):
        jno.precond.coarse([1], parts="many").prepare(fem)
    with pytest.raises(ValueError, match="second"):
        jno.precond.coarse([1], parts=jno.precond.schwarz(parts=4)).prepare(fem)
    with pytest.raises(ValueError, match="problem attached"):
        jno.precond.coarse([1]).build(fem._op[0])
    with pytest.raises(TypeError, match="matvec"):
        spec = jno.precond.coarse(np.ones((fem._op[0].shape[0], 1)), parts=4)
        spec.prepare(fem)
        n = fem._op[0].shape[0]
        mv_only = LinearOperator.from_matvec(lambda v: fem._op[0] @ v, shape=(n, n))
        materialize_precond(spec, PrecondContext(mv_only, fem))


# ======================================================================================
# compiled, and on jno.fdm
# ======================================================================================
def test_a_built_coarse_space_compiles_and_reuses_its_program():
    from jno.utils.solver.solver_api import _compilable

    fem, (x, y) = _poisson(0.04)
    spec = jno.precond.jacobi() >> jno.precond.coarse([1, x, y], parts=8)
    assert not _compilable(jno.solve.cg(), spec), "unbuilt: the space and partition are host work"
    spec.prepare(fem)
    assert _compilable(jno.solve.cg(), spec)
    ref = np.asarray(fem.solve())
    for _ in range(2):
        got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=spec))
        np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())


def test_a_single_precision_coarse_space_gives_the_default_answer():
    fem, (x, y) = _poisson(0.04)
    ref = np.asarray(fem.solve())
    Q = jno.precond.coarse([1, x, y], parts=8, float32=True)
    got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-11), precond=Q >> jno.precond.jacobi() >> Q))
    np.testing.assert_allclose(got, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())


def test_a_global_space_works_matrix_free():
    """A global space needs only products with the operator, so a Jacobian-free Newton solve can use it."""
    d = R(0.0, 0.0, 1.0, 1.0, size=0.08).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    fem = jno.fem([(1.0 + ui**2) * (ui.x * vi.x + ui.y * vi.y) - vi, u(xb, yb) - 0.0])
    ref = np.asarray(fem.solve())
    Z = np.asarray(fem.field_points[0])[:, :1] * 0 + 1.0
    got = np.asarray(
        fem.solve(
            nonlinear=jno.solve.newton(direct=False),
            linear=jno.solve.gmres(tol=1e-11),
            precond=jno.precond.jacobi() >> jno.precond.coarse(Z),
        )
    )
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-8)


def test_an_fdm_vector_problem_gives_the_default_answer():
    import jno.jnp_ops as jnn

    d = jno.domain(R(0.0, 0.0, 1.0, 1.0, size=0.1).structured())
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    U = d.unknown(value_shape=(2,))
    Ui = U.vector.bind(x=x, y=y)
    f = jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    zero = jno.np.stack([0.0 * xb, 0.0 * xb], axis=-1)
    prob = jno.fdm([-Ui.laplacian() - jno.np.stack([f, 2.0 * f], axis=-1), U(xb, yb) - zero])
    ref = np.asarray(prob.solve())
    Q = jno.precond.coarse([(1, 0), (0, 1), (-y, x)], parts=4)
    got = np.asarray(prob.solve(linear=jno.solve.gmres(tol=1e-11), precond=jno.precond.jacobi() >> Q))
    assert np.abs(got - ref).max() < 1e-8 * np.abs(ref).max()


@pytest.mark.parametrize("restricted", [False, True])
def test_two_level_schwarz_is_the_coarse_space_around_the_one_level_method(restricted):
    """`schwarz(coarse=Z)` is `Q >> S >> Q` (balanced) with `S = schwarz(coarse=False)` and `Q = coarse(Z,
    parts=S)`, and `Q >> S` when restricted -- the same operator to rounding, on the same partition."""
    fem, (x, y) = _cantilever(0.1)
    rigid = [(1, 0), (0, 1), (-y, x)]
    two = jno.precond.schwarz(parts=8, coarse=rigid, restricted=restricted)
    S = jno.precond.schwarz(parts=8, coarse=False, restricted=restricted)
    Q = jno.precond.coarse(rigid, parts=S)
    B_two, _ = _matrix(two, fem)
    B_alg, _ = _matrix(Q >> S if restricted else Q >> S >> Q, fem)
    np.testing.assert_array_equal(np.asarray(two._pattern.part), np.asarray(S._pattern.part))
    np.testing.assert_allclose(B_alg, B_two, rtol=1e-10, atol=1e-10 * np.abs(B_two).max())  # GPU summation order
