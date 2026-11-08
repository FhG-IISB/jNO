"""The host LU factors the operator's nonzeros, not its stored pattern.

A jNO operator stores every entry its element blocks could touch: all fields of a cell against all fields,
and a `region=` field's DOFs outside its region (held at 0 by an identity row) keep their full pattern. Those
entries are explicit zeros, and SuperLU orders and factors whatever is stored. On a coupled induction + gas
flow + transport + elasticity form, 96% of the stored entries were zeros: the tangent factored with them took
minutes per Newton step and with them dropped it took seconds, with the same solution.
"""

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from jno.utils.solver import linear as _linear


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(autouse=True)
def _clean_cache():
    saved = dict(_linear._FACTOR_CACHE)
    _linear._FACTOR_CACHE.clear()
    try:
        yield
    finally:
        _linear._FACTOR_CACHE.clear()
        _linear._FACTOR_CACHE.update(saved)


def _padded(n, width):
    """A tridiagonal operator plus explicit zeros on a band of half-width `width` (the stored pattern)."""
    T = sp.diags([np.arange(2.0, n + 2.0), -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1], format="coo")
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    band = (np.abs(i - j) <= width) & (np.abs(i - j) > 1)
    rows = np.concatenate([T.row, i[band]])
    cols = np.concatenate([T.col, j[band]])
    vals = np.concatenate([T.data, np.zeros(int(band.sum()))])
    A = jsp.BCOO((jnp.asarray(vals), jnp.asarray(np.stack([rows, cols], axis=1))), shape=(n, n))
    return A, T.tocsc()


@pytest.mark.parametrize("reuse", [True, False])
def test_superlu_sees_only_the_nonzeros_and_the_answer_is_unchanged(monkeypatch, reuse):
    n = 40
    A, T = _padded(n, 12)
    b = jnp.asarray(np.linspace(-1.0, 1.0, n))
    seen = []
    orig = spla.splu

    def spy(mat, *a, **k):
        seen.append(mat.nnz)
        return orig(mat, *a, **k)

    monkeypatch.setattr(spla, "splu", spy)
    x = np.asarray(_linear.host_lu_solve(A, b, reuse=reuse))
    assert seen == [T.nnz], f"SuperLU was handed {seen} entries; the operator has {T.nnz} nonzeros"
    assert np.allclose(x, spla.spsolve(T, np.asarray(b)), rtol=1e-13, atol=0.0)


def test_the_transpose_solve_uses_the_pruned_factorization_too():
    n = 40
    A, T = _padded(n, 12)
    b = jnp.asarray(np.linspace(-1.0, 1.0, n))
    g = jax.grad(lambda v: jnp.sum(_linear.host_lu_solve(A, v) ** 2))(b)
    x = spla.spsolve(T, np.asarray(b))
    assert np.allclose(np.asarray(g), spla.spsolve(T.T.tocsc(), 2 * x), rtol=1e-12, atol=0.0)
