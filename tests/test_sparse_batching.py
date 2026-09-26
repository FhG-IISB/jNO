"""vmap rules for JAX's sparse primitives (``jno.utils.solver.sparse_batching``, installed by ``import jno``).

Oracle: dense numpy. Every batched product must equal the per-item dense product and, on GPU, still
lower to cuSPARSE."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from jax.experimental import sparse as js

import jno  # noqa: F401  -- importing jNO installs the rules
from jno.utils.solver import sparse_batching as csr_batching


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


GPU = jax.default_backend() == "gpu"


def _pattern(m=37, n=29, density=0.15, seed=0):
    S = sp.random(m, n, density=density, random_state=seed, format="csr")
    S.sort_indices()
    return S


def _parts(S, dtype):
    return (
        jnp.asarray(S.data.astype(dtype)),
        jnp.asarray(S.indices.astype(np.int32)),
        jnp.asarray(S.indptr.astype(np.int32)),
    )


def _mv(data, indices, indptr, v, shape, transpose=False):
    return (
        js.CSR((data, indices, indptr), shape=shape) @ v
        if not transpose
        else (js.CSR((data, indices, indptr), shape=shape).T @ v)
    )


def _tol(dtype):
    return dict(rtol=1e-12, atol=1e-12) if dtype == np.float64 else dict(rtol=2e-5, atol=2e-5)


def _uses_cusparse(f, *args):
    return "cusparse" in jax.jit(f).lower(*args).compile().as_text()


@pytest.mark.parametrize("spmm", [False, True])  # both paths, whatever this machine's measurement picks
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_vmap_over_the_vector(dtype, transpose, axis, spmm, monkeypatch):
    monkeypatch.setattr(csr_batching, "_prefer_spmm", lambda *a, **k: spmm)
    B = 6
    S = _pattern()
    d, i, p = _parts(S, dtype)
    A = S.toarray().astype(dtype)
    k = S.shape[0] if transpose else S.shape[1]
    V = np.random.default_rng(1).standard_normal((B, k)).astype(dtype)
    Vin = V if axis == 0 else V.T
    f = lambda v: _mv(d, i, p, v, S.shape, transpose)
    got = jax.vmap(f, in_axes=axis)(jnp.asarray(Vin))
    ref = (V @ A) if transpose else (V @ A.T)
    np.testing.assert_allclose(np.asarray(got), ref, **_tol(dtype))
    if GPU:
        assert _uses_cusparse(jax.vmap(f, in_axes=axis), jnp.asarray(Vin))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("v_batched", [False, True])
def test_vmap_over_values_with_a_shared_pattern(dtype, transpose, v_batched):
    S = _pattern()
    d, i, p = _parts(S, dtype)
    rng = np.random.default_rng(2)
    D = rng.standard_normal((4, S.nnz)).astype(dtype)
    k = S.shape[0] if transpose else S.shape[1]
    V = rng.standard_normal((4, k)).astype(dtype) if v_batched else rng.standard_normal(k).astype(dtype)
    f = lambda dd, v: _mv(dd, i, p, v, S.shape, transpose)
    got = jax.vmap(f, in_axes=(0, 0 if v_batched else None))(jnp.asarray(D), jnp.asarray(V))
    for b in range(4):
        Ab = sp.csr_matrix((D[b], S.indices, S.indptr), shape=S.shape).toarray()
        vb = V[b] if v_batched else V
        np.testing.assert_allclose(np.asarray(got[b]), (Ab.T if transpose else Ab) @ vb, **_tol(dtype))
    if GPU:
        assert _uses_cusparse(jax.vmap(f, in_axes=(0, 0 if v_batched else None)), jnp.asarray(D), jnp.asarray(V))


def test_vmap_over_different_patterns_of_equal_size():
    mats = [_pattern(seed=s) for s in range(3)]
    nse = min(m.nnz for m in mats)
    mats = [sp.csr_matrix((m.data[:nse], m.indices[:nse], np.minimum(m.indptr, nse)), shape=m.shape) for m in mats]
    D, Ind, P = (
        jnp.stack([jnp.asarray(getattr(m, a).astype(t)) for m in mats])
        for a, t in (("data", np.float64), ("indices", np.int32), ("indptr", np.int32))
    )
    v = jnp.asarray(np.random.default_rng(3).standard_normal(mats[0].shape[1]))
    got = jax.vmap(lambda d, i, p: _mv(d, i, p, v, mats[0].shape))(D, Ind, P)
    for b, m in enumerate(mats):
        np.testing.assert_allclose(np.asarray(got[b]), m.toarray() @ np.asarray(v), rtol=1e-12, atol=1e-12)


def test_nested_vmap_values_and_vectors():
    S = _pattern()
    d, i, p = _parts(S, np.float64)
    rng = np.random.default_rng(4)
    D = rng.standard_normal((3, S.nnz))
    V = rng.standard_normal((6, S.shape[1]))
    got = jax.vmap(lambda dd: jax.vmap(lambda v: _mv(dd, i, p, v, S.shape))(jnp.asarray(V)))(jnp.asarray(D))
    for b in range(3):
        Ab = sp.csr_matrix((D[b], S.indices, S.indptr), shape=S.shape).toarray()
        np.testing.assert_allclose(np.asarray(got[b]), V @ Ab.T, rtol=1e-12, atol=1e-12)


def test_jacfwd_and_vmap_of_grad_and_grad_of_vmap():
    S = _pattern()
    d, i, p = _parts(S, np.float64)
    A = S.toarray()
    x = jnp.asarray(np.random.default_rng(5).standard_normal(S.shape[1]))
    J = jax.jacfwd(lambda v: _mv(d, i, p, v, S.shape))(x)  # the thing that raised before
    np.testing.assert_allclose(np.asarray(J), A, rtol=1e-12, atol=1e-12)
    Jd = jax.jacfwd(lambda dd: _mv(dd, i, p, x, S.shape))(d)
    row = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
    ref = np.zeros((S.shape[0], S.nnz))
    ref[row, np.arange(S.nnz)] = np.asarray(x)[S.indices]
    np.testing.assert_allclose(np.asarray(Jd), ref, rtol=1e-12, atol=1e-12)
    loss = lambda dd, v: jnp.sum(jnp.sin(_mv(dd, i, p, v, S.shape)))
    D = jnp.stack([d, 2 * d, -d])
    g_vmap = jax.vmap(jax.grad(loss, argnums=(0, 1)), in_axes=(0, None))(D, x)
    g_loop = [jax.grad(loss, argnums=(0, 1))(D[b], x) for b in range(3)]
    for b in range(3):
        np.testing.assert_allclose(np.asarray(g_vmap[0][b]), np.asarray(g_loop[b][0]), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.asarray(g_vmap[1][b]), np.asarray(g_loop[b][1]), rtol=1e-12, atol=1e-12)
    total = lambda DD: jnp.sum(jax.vmap(lambda dd: loss(dd, x))(DD))
    np.testing.assert_allclose(np.asarray(jax.grad(total)(D)), np.stack([g[0] for g in g_loop]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("transpose", [False, True])
def test_csr_matmat_batching(transpose):
    S = _pattern()
    d, i, p = _parts(S, np.float64)
    rng = np.random.default_rng(6)
    k = S.shape[0] if transpose else S.shape[1]
    X = rng.standard_normal((3, k, 4))
    A = S.toarray()
    Aop = A.T if transpose else A
    mm = lambda dd, x: (js.CSR((dd, i, p), shape=S.shape).T if transpose else js.CSR((dd, i, p), shape=S.shape)) @ x
    got = jax.vmap(lambda x: mm(d, x))(jnp.asarray(X))
    np.testing.assert_allclose(np.asarray(got), np.einsum("ij,bjk->bik", Aop, X), rtol=1e-12, atol=1e-12)
    D = rng.standard_normal((3, S.nnz))
    got = jax.vmap(mm)(jnp.asarray(D), jnp.asarray(X))
    for b in range(3):
        Ab = sp.csr_matrix((D[b], S.indices, S.indptr), shape=S.shape).toarray()
        np.testing.assert_allclose(np.asarray(got[b]), (Ab.T if transpose else Ab) @ X[b], rtol=1e-12, atol=1e-12)


def test_install_is_idempotent_and_announces_only_what_it_registers(capsys):
    assert set(csr_batching._INSTALLED) >= {"csr_matvec", "csr_matmat"}  # done by `import jno`
    capsys.readouterr()
    assert csr_batching.install() == []  # nothing new: no second registration, no second warning
    assert "registered vmap" not in capsys.readouterr().out


def test_block_diagonal_refuses_an_index_overflow_instead_of_wrapping():
    data = jnp.ones((4, 5))
    idx = jnp.zeros((4, 5), jnp.int32)
    ptr = jnp.zeros((4, 4), jnp.int32)
    with pytest.raises(NotImplementedError, match="overflows int32"):
        csr_batching._block_diagonal(data, idx, ptr, (3, 2**30))


# ---------------------------------------------------------------------------------------------- spsolve
def _square(n=60, seed=0):
    S = (sp.diags([-1.0, 4.2, -1.3], [-1, 0, 1], (n, n)) + 0.2 * sp.random(n, n, 0.03, random_state=seed)).tocsr()
    S.sort_indices()
    return S, jnp.asarray(S.data), jnp.asarray(S.indices.astype(np.int32)), jnp.asarray(S.indptr.astype(np.int32))


def _solve(d, i, p, b):
    from jax.experimental.sparse.linalg import spsolve

    return spsolve(d, i, p, b, tol=1e-12)


def test_spsolve_vmap_over_rhs_values_and_both():
    S, d, i, p = _square()
    rng = np.random.default_rng(0)
    B = rng.standard_normal((5, S.shape[0]))
    D = (1 + 0.1 * rng.standard_normal((5, S.nnz))) * S.data
    A = S.toarray()
    got = jax.vmap(lambda b: _solve(d, i, p, b))(jnp.asarray(B))
    np.testing.assert_allclose(np.asarray(got), np.linalg.solve(A, B.T).T, rtol=1e-10, atol=1e-12)
    got = jax.vmap(lambda dd, b: _solve(dd, i, p, b))(jnp.asarray(D), jnp.asarray(B))
    got_shared_b = jax.vmap(lambda dd: _solve(dd, i, p, jnp.asarray(B[0])))(jnp.asarray(D))
    for k in range(5):
        Ak = sp.csr_matrix((D[k], S.indices, S.indptr), shape=S.shape).toarray()
        np.testing.assert_allclose(np.asarray(got[k]), np.linalg.solve(Ak, B[k]), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(got_shared_b[k]), np.linalg.solve(Ak, B[0]), rtol=1e-10, atol=1e-12)


def test_spsolve_jacfwd_jacrev_and_value_jacobian():
    S, d, i, p = _square(n=40)
    b = jnp.asarray(np.random.default_rng(1).standard_normal(S.shape[0]))
    Ainv = np.linalg.inv(S.toarray())
    np.testing.assert_allclose(np.asarray(jax.jacfwd(lambda bb: _solve(d, i, p, bb))(b)), Ainv, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jacrev(lambda bb: _solve(d, i, p, bb))(b)), Ainv, rtol=1e-9, atol=1e-12)
    # w.r.t. the matrix values: one tangent per nonzero -> nnz solves, one at a time
    J = jax.jacfwd(lambda dd: _solve(dd, i, p, b))(d)
    x = np.linalg.solve(S.toarray(), np.asarray(b))
    row = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
    # d x / d a_k = -A^{-1} e_{row_k} x_{col_k}
    ref = -Ainv[:, row] * x[S.indices][None, :]
    np.testing.assert_allclose(np.asarray(J), ref, rtol=1e-8, atol=1e-11)


def test_jacrev_through_jnos_sparse_lu_solve_matches_rowwise_jacobian():
    """The path `jno.solve.lu()` takes. `jax.jacrev` used to raise here; it must now agree with jNO's own
    vmap-free `rowwise_jacobian`."""
    from jno.utils.ad_mode import rowwise_jacobian
    from jno.utils.solver.linear import sparse_lu_solve

    S, d, i, p = _square(n=50, seed=3)
    coo = S.tocoo()
    idx = jnp.asarray(np.stack([coo.row, coo.col], 1).astype(np.int32))
    b = jnp.asarray(np.random.default_rng(2).standard_normal(S.shape[0]))

    def f(theta):  # a parametric operator: values scaled by exp(theta), a load scaled by theta
        A = js.BCOO((jnp.asarray(coo.data) * jnp.exp(theta[0]), idx), shape=S.shape)
        return sparse_lu_solve(A, b * (1.0 + theta[1]))[:7]

    theta = jnp.asarray([0.1, -0.2])
    J_rev = jax.jacrev(f)(theta)
    J_row = rowwise_jacobian(f, theta, range(7))
    np.testing.assert_allclose(np.asarray(J_rev), np.asarray(J_row), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jax.jacfwd(f)(theta)), np.asarray(J_row), rtol=1e-9, atol=1e-12)


def test_spmm_choice_is_measured_per_device_and_cached():
    csr_batching._SPMM_DECISIONS.clear()
    first = csr_batching._prefer_spmm((500, 500), 3000, 16, np.float64, False)
    assert isinstance(first, bool)
    (key,) = csr_batching._SPMM_DECISIONS
    dev = jax.devices()[0]
    assert key[:2] == (dev.platform, dev.device_kind)  # keyed on the machine, not hard-coded
    csr_batching._SPMM_DECISIONS[key] = not first  # a cached decision is reused, not re-measured
    assert csr_batching._prefer_spmm((500, 500), 3000, 16, np.float64, False) is (not first)


def test_a_failed_measurement_falls_back_to_the_spmv_loop_and_says_so(monkeypatch):
    from jno.utils import logger as _logger

    got = []  # a spy, not stdout capture: `jno.setup` elsewhere re-binds the logger past pytest's capture
    monkeypatch.setattr(
        _logger, "get_logger", lambda *a, **k: type("Spy", (), {"warning": lambda self, m: got.append(m)})()
    )
    csr_batching._SPMM_DECISIONS.clear()

    def broken(*a, **k):
        raise RuntimeError("no SpMM here")

    monkeypatch.setattr(csr_batching._csr, "_csr_matmat", broken)
    assert csr_batching._prefer_spmm((300, 300), 1500, 20, np.float64, False) is False
    assert any("using the SpMV loop" in m for m in got)
    csr_batching._SPMM_DECISIONS.clear()


# ---------------------------------------------------------------------------- factor once / lu_stack
def _bcoo_square(n=80, seed=4):
    S = (sp.diags([-1.0, 4.2, -1.3], [-1, 0, 1], (n, n)) + 0.2 * sp.random(n, n, 0.03, random_state=seed)).tocoo()
    A = js.BCOO((jnp.asarray(S.data), jnp.asarray(np.stack([S.row, S.col], 1).astype(np.int32))), shape=S.shape)
    return S, A


def test_host_lu_factors_once_for_a_batch_against_one_matrix(monkeypatch):
    import scipy.sparse.linalg as spla

    from jno.utils.solver import linear

    S, A = _bcoo_square()
    calls = {"n": 0}
    orig = spla.splu

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(spla, "splu", counting)
    B = jnp.asarray(np.random.default_rng(0).standard_normal((7, S.shape[0])))
    X = jax.vmap(lambda b: linear.host_lu_solve(A, b, reuse=False))(B)
    np.testing.assert_allclose(np.asarray(X), np.linalg.solve(S.toarray(), np.asarray(B).T).T, rtol=1e-10, atol=1e-12)
    assert calls["n"] == 1, "a batch against one matrix must factor once"
    calls["n"] = 0
    J = jax.jacrev(lambda b: linear.host_lu_solve(A, b, reuse=False))(B[0])
    np.testing.assert_allclose(np.asarray(J), np.linalg.inv(S.toarray()), rtol=1e-9, atol=1e-12)
    assert calls["n"] == 2  # the forward solve + ONE for all 80 adjoint rows


def test_host_lu_batched_values_are_factored_each():
    from jno.utils.solver import linear

    S, A = _bcoo_square()
    rng = np.random.default_rng(1)
    D = jnp.asarray(1 + 0.1 * rng.standard_normal((3, S.nnz))) * A.data
    B = jnp.asarray(rng.standard_normal((3, S.shape[0])))
    X = jax.vmap(lambda d, b: linear.host_lu_solve(js.BCOO((d, A.indices), shape=A.shape), b, reuse=False))(D, B)
    for k in range(3):
        Ak = sp.coo_matrix((np.asarray(D[k]), (S.row, S.col)), shape=S.shape).toarray()
        np.testing.assert_allclose(np.asarray(X[k]), np.linalg.solve(Ak, np.asarray(B[k])), rtol=1e-10, atol=1e-12)


def _spsolve_sizes(f, *args):
    import re

    return [int(m) for m in re.findall(r"f64\[(\d+)\] = spsolve", str(jax.make_jaxpr(f)(*args)))]


@pytest.fixture
def _restore_lu_stack():
    yield
    csr_batching.set_lu_stack(1)


def test_lu_stack_stacks_takes_effect_when_changed_and_pads_correctly(_restore_lu_stack):
    S, d, i, p = _square(n=30)
    B = jnp.asarray(np.random.default_rng(2).standard_normal((5, S.shape[0])))
    f = jax.vmap(lambda b: _solve(d, i, p, b))
    csr_batching.set_lu_stack(1)
    assert _spsolve_sizes(f, B) == [30]  # one system per lax.map step
    csr_batching.set_lu_stack(2)  # 5 systems -> chunks of 2, last one padded
    assert _spsolve_sizes(f, B) == [60]
    np.testing.assert_allclose(np.asarray(f(B)), np.linalg.solve(S.toarray(), np.asarray(B).T).T, rtol=1e-10, atol=1e-12)
    csr_batching.set_lu_stack(8)  # >= batch: one call
    assert _spsolve_sizes(f, B) == [150]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4"])
def test_lu_stack_rejects_non_positive_integers(bad):
    with pytest.raises(ValueError, match="lu_stack"):
        csr_batching.set_lu_stack(bad)


def test_setup_sets_lu_stack_from_the_argument_and_from_toml(tmp_path, monkeypatch, _restore_lu_stack):
    from jno.utils import config

    script = tmp_path / "run.py"
    script.write_text("")
    jno.setup(str(script), lu_stack=3)
    assert csr_batching._LU_STACK == 3
    monkeypatch.setattr(config, "get_config", lambda: {"jno": {"lu_stack": 5}})
    jno.setup(str(script))
    assert csr_batching._LU_STACK == 5


# ------------------------------------------------------------- factor once: cuDSS and PARDISO as well
def _backend(name):
    from jno.utils.solver import linear

    avail = {"cudss": linear._cudss_available, "pardiso": linear._pardiso_available}[name]
    if not avail():
        pytest.skip(f"{name} not installed (jno extra [{name}])")
    return linear, {"cudss": linear.cudss_lu_solve, "pardiso": linear.pardiso_lu_solve}[name]


@pytest.mark.parametrize("name", ["cudss", "pardiso"])
def test_factor_once_backends_solve_a_vmapped_batch_in_one_call(name, monkeypatch):
    linear, solve = _backend(name)
    S, A = _bcoo_square()
    host_fn = f"_{name}_host_solve"
    calls = {"n": 0}
    orig = getattr(linear, host_fn)

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(linear, host_fn, counting)
    B = jnp.asarray(np.random.default_rng(3).standard_normal((6, S.shape[0])))
    X = jax.vmap(lambda b: solve(A, b))(B)
    np.testing.assert_allclose(np.asarray(X), np.linalg.solve(S.toarray(), np.asarray(B).T).T, rtol=1e-9, atol=1e-11)
    assert calls["n"] == 1, "a vmapped batch against one matrix must reach the backend as ONE block solve"


@pytest.mark.parametrize("name", ["cudss", "pardiso"])
def test_factor_once_backends_jacobians_and_batched_values(name):
    linear, solve = _backend(name)
    S, A = _bcoo_square()
    Ainv = np.linalg.inv(S.toarray())
    b = jnp.asarray(np.random.default_rng(4).standard_normal(S.shape[0]))
    np.testing.assert_allclose(np.asarray(jax.jacrev(lambda bb: solve(A, bb))(b)), Ainv, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(jax.jacfwd(lambda bb: solve(A, bb))(b)), Ainv, rtol=1e-8, atol=1e-10)
    rng = np.random.default_rng(5)
    D = jnp.asarray(1 + 0.1 * rng.standard_normal((3, S.nnz))) * A.data
    Bs = jnp.asarray(rng.standard_normal((3, S.shape[0])))
    X = jax.vmap(lambda d, bb: solve(js.BCOO((d, A.indices), shape=A.shape), bb))(D, Bs)
    for k in range(3):
        Ak = sp.coo_matrix((np.asarray(D[k]), (S.row, S.col)), shape=S.shape).toarray()
        np.testing.assert_allclose(np.asarray(X[k]), np.linalg.solve(Ak, np.asarray(Bs[k])), rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("name", ["cudss", "pardiso"])
def test_a_vmapped_block_right_hand_side(name):
    linear, solve = _backend(name)
    S, A = _bcoo_square()
    Bk = jnp.asarray(np.random.default_rng(6).standard_normal((4, S.shape[0], 3)))  # 4 blocks of 3 columns
    X = jax.vmap(lambda blk: solve(A, blk))(Bk)
    ref = np.stack([np.linalg.solve(S.toarray(), np.asarray(Bk[j])) for j in range(4)])
    np.testing.assert_allclose(np.asarray(X), ref, rtol=1e-9, atol=1e-11)
