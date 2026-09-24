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


@pytest.mark.parametrize("B", [5, csr_batching.SPMM_MIN_BATCH + 3])  # SpMV-loop path and SpMM path
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_vmap_over_the_vector(dtype, transpose, axis, B):
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
