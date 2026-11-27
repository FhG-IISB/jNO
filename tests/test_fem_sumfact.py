"""Sum factorisation of a Lagrange Q_k field on quadrilaterals and hexahedra.

A Q_k field (k >= 3) is evaluated at the quadrature points by 1-D contractions along each axis of the cell's DOF
tensor (``fem_lagrange.TensorFactors``) instead of the dense ``(n_quad, n_dof)`` tables. Checks: the factors
reproduce the tabulation; the assembled operator and load equal the dense-table ones on a distorted (non-affine)
mesh, for a scalar and a vector field and both quadrature rules.
"""

import jax
import jno.numpy as jnn
import numpy as np
import pytest

import jno
from jno.utils.solver import fem_lagrange as fl


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("cell", ["quad", "hexahedron"])
@pytest.mark.parametrize("degree", [3, 5])
@pytest.mark.parametrize("rule", [None, "gll"])
def test_factors_reproduce_the_tabulation(cell, degree, rule, monkeypatch):
    from jno.utils.solver import fem_utils

    monkeypatch.setattr(fem_utils, "_QUAD_RULE", [rule])
    spec = fl.lagrange_on(cell, degree)
    tf = fl.tensor_factors(cell, degree, spec.quad_points, spec.ref_values, spec.ref_grads)
    assert tf is not None
    assert all(tf._ident) == (rule == "gll")  # collocated: the value tables are the identity and are skipped
    rng = np.random.default_rng(0)
    u = jax.numpy.asarray(rng.standard_normal((spec.n_dof, 2)))
    V, dV = spec.ref_values[..., 0], spec.ref_grads[..., 0, :]
    assert np.abs(np.asarray(tf.values(u)) - V @ np.asarray(u)).max() < 1e-11
    ref = np.einsum("qnd,nc->qcd", dV, np.asarray(u))
    assert np.abs(np.asarray(tf.ref_grads(u)) - ref).max() < 1e-10 * max(1.0, np.abs(ref).max())
    tdim = dV.shape[-1]
    K = jax.numpy.asarray(rng.standard_normal((V.shape[0], tdim, tdim)))
    phys = np.einsum("qcd,qdD->qcD", ref, np.asarray(K))
    assert np.abs(np.asarray(tf.grads(u, K)) - phys).max() < 1e-10 * max(1.0, np.abs(phys).max())


def test_a_simplex_or_a_non_tensor_rule_has_no_factors():
    spec = fl.lagrange_on("triangle", 3)
    assert fl.tensor_factors("triangle", 3, spec.quad_points, spec.ref_values, spec.ref_grads) is None
    q = fl.lagrange_on("quad", 3)
    assert fl.tensor_factors("quad", 3, q.quad_points[:-1], q.ref_values[:-1], q.ref_grads[:-1]) is None


def _distorted(dim, n=2):
    box = jno.shape.rect(0, 0, 1, 1) if dim == 2 else jno.shape.box(0, 0, 0, 1, 1, 1)
    d = box.structured(n=n).quad().domain(compute_mesh_connectivity=False)
    pts = np.asarray(d.mesh.points)
    mid = int(np.argmin(np.abs(pts[:, :dim] - 0.5).sum(axis=1)))
    d.mesh.points[mid, :dim] += np.array([0.11, -0.07, 0.05])[:dim]  # trapezia: the Jacobian varies in a cell
    return d


def _scalar_problem(dim, order, quadrature, **kw):
    d = _distorted(dim)
    u, v = d.fem_symbols(order=order)
    X = d.variable("interior", split=True)[:dim]
    ui, vi = u.bind(**dict(zip("xyz", X))), v.bind(**dict(zip("xyz", X)))
    grad = sum(getattr(ui, a) * getattr(vi, a) for a in "xyz"[:dim])
    kappa = 1.0 + X[0] * X[1]
    form = kappa * grad + X[0] * ui.x * vi + 3.0 * ui * vi - jnn.sin(3 * X[0]) * vi
    return jno.fem([form], quadrature=quadrature, **kw)


def _scalar(dim, order, quadrature):
    prob = _scalar_problem(dim, order, quadrature)
    return np.asarray(prob.A), np.asarray(prob.b)


def _vector_problem(order, quadrature, **kw):
    d = _distorted(2)
    u, v = d.fem_symbols(order=order, value_shape=(2,))
    x, y, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    X = [x, y]
    form = 2.0 * jnn.inner(jnn.symgrad(ui, X), jnn.symgrad(vi, X), 2)  # the whole-field gradient rule
    form = form + 1.5 * (ui[0].d(x) + ui[1].d(y)) * (vi[0].d(x) + vi[1].d(y))  # the component-gradient rule
    form = form + jnn.inner(ui, vi, 1)  # the value rule
    return jno.fem([form - y * vi[0] + x * vi[1]], quadrature=quadrature, **kw)


def _vector(order, quadrature):
    prob = _vector_problem(order, quadrature)
    return np.asarray(prob.A), np.asarray(prob.b)


@pytest.mark.parametrize("case", ["scalar2d", "scalar3d", "vector2d"])
@pytest.mark.parametrize("quadrature", [None, "gll"])
def test_the_operator_equals_the_dense_table_one(case, quadrature, monkeypatch):
    build = {
        "scalar2d": lambda: _scalar(2, 4, quadrature),
        "scalar3d": lambda: _scalar(3, 3, quadrature),
        "vector2d": lambda: _vector(4, quadrature),
    }[case]
    calls, real = [], fl.tensor_factors
    monkeypatch.setattr(fl, "tensor_factors", lambda *a, **k: calls.append(1) or real(*a, **k))
    A, b = build()
    assert calls, "the build did not take the sum-factorised path"
    monkeypatch.setattr(fl, "SUMFACT_MIN_DEGREE", 99)  # the dense tables
    del calls[:]
    A0, b0 = build()
    assert not calls
    assert A.shape == A0.shape and np.abs(A0).max() > 0
    assert np.abs(A - A0).max() < 1e-11 * np.abs(A0).max()
    assert np.abs(b - b0).max() < 1e-11 * max(np.abs(b0).max(), 1.0)


# ------------------------------------------------------------------------------------------ matrix-free operator


def _poisson(dim, order, **kw):
    d = _distorted(dim)
    u, v = d.fem_symbols(order=order)
    X = d.variable("interior", split=True)[:dim]
    Xb = d.variable("boundary", split=True)[:dim]
    ui, vi = u.bind(**dict(zip("xyz", X))), v.bind(**dict(zip("xyz", X)))
    grad = sum(getattr(ui, a) * getattr(vi, a) for a in "xyz"[:dim])
    exact = lambda *P: jnn.sin(np.pi * P[0]) * jnn.cos(P[1]) + (P[2] if dim == 3 else 0.0)  # noqa: E731
    f = (np.pi**2 + 1.0) * jnn.sin(np.pi * X[0]) * jnn.cos(X[1])
    return jno.fem([grad - f * vi, u(*Xb) - exact(*Xb)], **kw), exact


@pytest.mark.parametrize("dim,order", [(2, 5), (3, 3)])
def test_the_matrix_free_operator_is_the_assembled_one(dim, order):
    mf, _ = _poisson(dim, order, matrix_free=True)
    ref, _ = _poisson(dim, order)
    A, b = mf.operator
    A0 = np.asarray(ref.A)
    assert getattr(A, "matrix_free", False) and not hasattr(A, "todense")
    assert np.abs(np.asarray(mf.A) - A0).max() < 1e-10 * np.abs(A0).max()
    assert np.abs(np.asarray(b) - np.asarray(ref.b)).max() < 1e-10 * np.abs(np.asarray(ref.b)).max()
    assert np.abs(np.asarray(A.diag()) - np.diag(A0)).max() < 1e-10 * np.abs(A0).max()
    v = jax.numpy.asarray(np.random.default_rng(1).standard_normal(A0.shape[0]))
    assert np.abs(np.asarray(A.T @ v) - A0.T @ np.asarray(v)).max() < 1e-9 * np.abs(A0).max()


def test_a_matrix_free_solve_matches_the_assembled_solve_and_the_exact_field():
    mf, exact = _poisson(2, 6, matrix_free=True)
    ref, _ = _poisson(2, 6)
    u = np.asarray(mf.solve()).reshape(-1)
    u0 = np.asarray(ref.solve()).reshape(-1)
    P = np.asarray(mf.points)
    assert np.abs(u - u0).max() < 1e-6
    assert np.abs(u - np.sin(np.pi * P[:, 0]) * np.cos(P[:, 1])).max() < 1e-5
    cg = np.asarray(mf.solve(linear=jno.solve.cg(tol=1e-12), precond=jno.precond.jacobi())).reshape(-1)
    assert np.abs(cg - np.sin(np.pi * P[:, 0]) * np.cos(P[:, 1])).max() < 1e-6  # the Q6 discretisation error


def test_a_direct_solver_on_a_matrix_free_operator_says_so():
    mf, _ = _poisson(2, 3, matrix_free=True)
    with pytest.raises(Exception, match="(?i)matrix|assembled|matvec"):
        mf.solve(linear=jno.solve.lu())


def test_a_matrix_free_build_never_plans_the_sparsity_pattern(monkeypatch):
    """The pattern of a Q_k operator has ``n_local**2`` entries per cell (2.2e9 at Q8 and 2M DOFs): planning it was
    the host-memory peak of a matrix-free build (48 GB there), for an operator that is never assembled."""
    from jno.utils.solver import fem_native as fn

    calls = []
    real = fn.compress_plan_blocks
    monkeypatch.setattr(fn, "compress_plan_blocks", lambda *a, **k: calls.append(1) or real(*a, **k))
    mf, _ = _poisson(2, 3, matrix_free=True)
    A, _b = mf.operator
    assert np.isfinite(np.asarray(A.diag())).all() and not calls
    _poisson(2, 3)
    assert calls  # the spy sees the assembled build


def _cubic(order, quadrature=None):
    d = _distorted(2)
    u, v = d.fem_symbols(order=order)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = 5.0 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
    prob = jno.fem([ui.x * vi.x + ui.y * vi.y + ui**3 * vi - f * vi, u(xb, yb) - 0.0], quadrature=quadrature)
    return np.asarray(prob.solve()).reshape(-1)


@pytest.mark.parametrize("quadrature", [None, "gll"])
def test_a_nonlinear_solve_equals_the_dense_table_one(quadrature, monkeypatch):
    """Newton's residual and tangent both go through the factorised field evaluation."""
    u = _cubic(4, quadrature)
    monkeypatch.setattr(fl, "SUMFACT_MIN_DEGREE", 99)
    u0 = _cubic(4, quadrature)
    assert np.abs(u0).max() > 0.1 and np.abs(u - u0).max() < 1e-9


def _equations(jaxpr):
    for eqn in jaxpr.eqns:
        yield eqn
        for p in eqn.params.values():
            for sub in p if isinstance(p, (list, tuple)) else (p,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    yield from _equations(inner)


@pytest.mark.parametrize("dim,order", [(2, 5), (3, 4)])
def test_the_matrix_free_apply_is_one_contraction_per_axis_each_way(dim, order):
    """The cost claim, structurally: ``A v`` of a Laplacian contracts once along each axis for the gradient and
    once for its transpose (2 x dim products), and holds no array the size of an ``(n_quad, n_dof)`` table.
    Three separate gradients for ``u.x``, ``u.y``, ``u.z`` -- or a transposed pass per direction -- would show
    up here as 6 x dim."""
    mf, _ = _poisson(dim, order, matrix_free=True, quadrature="gll")
    A, b = mf.operator
    eqns = list(_equations(jax.make_jaxpr(A.mv)(jax.numpy.ones(b.shape[0])).jaxpr))
    assert sum(e.primitive.name == "dot_general" for e in eqns) == 2 * dim
    # each at full precision by request: a float32 product on a GPU otherwise runs in TensorFloat-32, and the
    # operator was then wrong by 5.6e-2 at 2M DOFs
    for e in eqns:
        if e.primitive.name == "dot_general":
            assert all(q == jax.lax.Precision.HIGHEST for q in e.params["precision"]), e.params["precision"]
    n_cells, n_local = 2**dim, (order + 1) ** dim
    largest = max(int(np.prod(o.aval.shape)) for e in eqns for o in e.outvars)
    assert largest < n_cells * n_local * n_local  # nothing table-sized: at most (cells, n_quad, dim, dim)


@pytest.mark.parametrize("case", ["scalar2d", "scalar3d", "vector2d"])
@pytest.mark.parametrize("quadrature", [None, "gll"])
def test_the_matrix_free_diagonal_is_summed_by_factors_and_exact(case, quadrature, monkeypatch):
    """The diagonal of an unstored operator, by sum factorisation (``_vol_elem_diag``: the integrand's bilinear
    coefficients per point against squared 1-D tables), against the assembled matrix's: variable coefficient,
    advection (non-symmetric), reaction, distorted cells, and a vector form with symgrad and divergence terms."""
    from jno.utils.solver import fem_native as fn

    build = {
        "scalar2d": lambda **kw: _scalar_problem(2, 4, quadrature, **kw),
        "scalar3d": lambda **kw: _scalar_problem(3, 3, quadrature, **kw),
        "vector2d": lambda **kw: _vector_problem(4, quadrature, **kw),
    }[case]
    verdicts, real = [], fn._fast_diag_agrees
    monkeypatch.setattr(fn, "_fast_diag_agrees", lambda *a, **k: verdicts.append(real(*a, **k)) or verdicts[-1])
    A, _b = build(matrix_free=True).operator
    d = np.asarray(A.diag())
    assert verdicts and all(verdicts), "the diagonal did not take the sum-factorised route"
    exact = np.diag(np.asarray(build().A))
    assert np.abs(d - exact).max() < 1e-11 * np.abs(exact).max()


def test_a_diagonal_the_fast_route_cannot_read_keeps_the_exact_pass(monkeypatch):
    """A term the factored route evaluates wrongly is caught by the check against the exact pass on sample cells."""
    from jno.utils.solver import fem_native as fn

    verdicts, agrees = [], fn._fast_diag_agrees
    monkeypatch.setattr(
        fn,
        "_fast_diag_agrees",
        lambda fast, exact, *a, **k: (
            verdicts.append(agrees(lambda c, la: 2.0 * fast(c, la), exact, *a, **k)) or verdicts[-1]
        ),
    )
    A, _b = _scalar_problem(2, 4, "gll", matrix_free=True).operator
    d = np.asarray(A.diag())
    assert verdicts and not any(verdicts)
    exact = np.diag(np.asarray(_scalar_problem(2, 4, "gll").A))
    assert np.abs(d - exact).max() < 1e-11 * np.abs(exact).max()


def test_a_matrix_free_operator_first_applied_inside_a_trace_still_applies(monkeypatch):
    """Its coefficients are moved to the solving device on first use and kept: a first use inside ``jit`` or
    ``make_jaxpr`` must not keep traced values (on the GPU the next eager apply then failed with "Execution
    supplied 15 buffers but compiled program expected 23"). On a CPU the move is a no-op, so it is made a real
    ``device_put`` here, as on a GPU."""
    from jno.utils.solver import placement

    monkeypatch.setattr(placement, "to_solve_device", lambda tree: jax.tree_util.tree_map(jax.device_put, tree))
    A, b = _poisson(2, 4, matrix_free=True)[0].operator
    v = jax.numpy.ones(b.shape[0])
    jax.make_jaxpr(A.mv)(v)
    traced = jax.jit(A.mv)(v)
    assert np.allclose(np.asarray(A.mv(v)), np.asarray(traced))
    assert np.isfinite(np.asarray(A.diag())).all()
