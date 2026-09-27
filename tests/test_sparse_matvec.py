"""``sparse_matvec``: the BCOO matvec with its index work hoisted out of the solver loop.

The oracle everywhere is JAX's own ``BCOO @ v`` (and its transpose / gradients / vmap): the helper
must be the same linear map, only cheaper to call repeatedly.
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from jno.utils.solver.linear import sparse_matvec


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _bcoo(n=60, m=None, density=0.08, seed=0, dtype=np.float64):
    m = n if m is None else m
    S = sp.random(n, m, density=density, random_state=seed, format="coo", dtype=np.float64)
    S.sum_duplicates()
    return jsp.BCOO.from_scipy_sparse(S.astype(dtype))


def _close(a, b, dtype=np.float64):
    tol = 1e-12 if dtype == np.float64 else 1e-5
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(60, 60), (40, 70), (70, 40)])
def test_matches_bcoo_for_vectors_and_blocks(dtype, shape):
    A = _bcoo(*shape, dtype=dtype)
    rng = np.random.default_rng(1)
    v = jnp.asarray(rng.standard_normal(shape[1]).astype(dtype))
    V = jnp.asarray(rng.standard_normal((shape[1], 3)).astype(dtype))
    _close(sparse_matvec(A)(v), A @ v, dtype)
    _close(sparse_matvec(A)(V), A @ V, dtype)  # multi-RHS block
    w = jnp.asarray(rng.standard_normal(shape[0]).astype(dtype))
    _close(sparse_matvec(A, transpose=True)(w), A.T @ w, dtype)
    _close(sparse_matvec(A.T)(w), A.T @ w, dtype)  # a transposed BCOO (unsorted indices) as input


def test_duplicate_triplets_are_summed_like_bcoo():
    idx = jnp.array([[0, 1], [0, 1], [2, 0], [2, 0], [2, 0], [1, 1]], jnp.int32)
    data = jnp.array([1.0, 2.0, 3.0, -1.0, 0.5, 4.0])
    A = jsp.BCOO((data, idx), shape=(3, 2))
    v = jnp.array([10.0, -3.0])
    _close(sparse_matvec(A)(v), A @ v)
    _close(sparse_matvec(A)(v), [-9.0, -12.0, 25.0])


def test_padded_out_of_bound_triplets_contribute_nothing_even_against_nonfinite_input():
    """``sum_duplicates`` (remove_zeros=True) pads with index (n, m). BCOO ignores those entries; so
    must we -- including when the clamped neighbour of the pad is inf/NaN (0 * inf would be NaN)."""
    A = _bcoo(30, seed=3)
    A = A.sum_duplicates(nse=A.nse + 7)  # 7 padding slots at the out-of-bound index
    assert int((np.asarray(A.indices) >= 30).any(axis=1).sum()) == 7
    v = jnp.ones(30).at[-1].set(jnp.inf)
    ref = A @ v
    got = sparse_matvec(A)(v)
    np.testing.assert_array_equal(np.isfinite(np.asarray(got)), np.isfinite(np.asarray(ref)))
    fin = np.isfinite(np.asarray(ref))
    _close(np.asarray(got)[fin], np.asarray(ref)[fin])
    w = jnp.ones(30).at[-1].set(jnp.nan)
    got_t, ref_t = sparse_matvec(A, transpose=True)(w), A.T @ w
    np.testing.assert_array_equal(np.isnan(np.asarray(got_t)), np.isnan(np.asarray(ref_t)))


def test_jit_grad_and_vmap_match_bcoo():
    A = _bcoo(50, seed=5)
    idx = A.indices
    rng = np.random.default_rng(2)
    v = jnp.asarray(rng.standard_normal(50))
    w = jnp.asarray(rng.standard_normal(50))

    def L_ref(data, v):
        return w @ (jsp.BCOO((data, idx), shape=A.shape) @ v)

    def L_new(data, v):
        return w @ sparse_matvec(jsp.BCOO((data, idx), shape=A.shape))(v)

    for f in (jax.grad(L_new, argnums=(0, 1)), jax.jit(jax.grad(L_new, argnums=(0, 1)))):
        gd, gv = f(A.data, v)
        rd, rv = jax.grad(L_ref, argnums=(0, 1))(A.data, v)
        _close(gd, rd)
        _close(gv, rv)
    # forward mode too (jacfwd vmaps tangents -- the thing CSR's matvec cannot do)
    _close(jax.jacfwd(L_new, argnums=1)(A.data, v), jax.jacfwd(L_ref, argnums=1)(A.data, v))
    # vmap over right-hand sides, and over operator VALUES with a shared pattern (parametric solves)
    Vs = jnp.asarray(rng.standard_normal((4, 50)))
    _close(jax.vmap(sparse_matvec(A))(Vs), jax.vmap(lambda x: A @ x)(Vs))
    Ds = jnp.asarray(rng.standard_normal((4, A.nse)))
    got = jax.vmap(lambda d: sparse_matvec(jsp.BCOO((d, idx), shape=A.shape))(v))(Ds)
    ref = jax.vmap(lambda d: jsp.BCOO((d, idx), shape=A.shape) @ v)(Ds)
    _close(got, ref)


def test_non_bcoo_operators_pass_through_unchanged():
    D = jnp.asarray(np.random.default_rng(0).standard_normal((5, 5)))
    v = jnp.arange(5.0)
    _close(sparse_matvec(D)(v), D @ v)
    _close(sparse_matvec(D, transpose=True)(v), v @ D)
    # a batched BCOO (n_batch=1) is left to BCOO's own matvec
    Ab = jsp.BCOO.fromdense(jnp.stack([D, 2 * D]), n_batch=1)
    _close(sparse_matvec(Ab)(v), Ab @ v)


def test_default_linear_solve_has_no_bcoo_matvec_inside_its_krylov_loop():
    """The point of the helper: BCOO's per-call index preparation (split + wrap + bounds select) is
    not hoisted out of a while loop by XLA, so the loop must not call BCOO's matvec. Structural check
    on the traced default solve, forward and with its implicit-diff firewall."""
    from jno._fem import _bicgstab_jacobi, _firewalled_bicgstab

    n = 40
    S = sp.diags([-1.0, 2.2, -1.0], [-1, 0, 1], (n, n)).tocoo()
    A = jsp.BCOO.from_scipy_sparse(S)
    b = jnp.ones(n)
    for f in (lambda A, b: _bicgstab_jacobi(A, b, 1e-10, 500), lambda A, b: _firewalled_bicgstab(A, b, 1e-10, 500)):
        text = str(jax.make_jaxpr(f)(A, b))
        assert "while" in text  # the Krylov loop is there ...
        assert "bcoo_dot_general" not in text  # ... and no BCOO matvec anywhere in it
    # and it still solves the system
    x = _firewalled_bicgstab(A, b, 1e-10, 500)
    np.testing.assert_allclose(S.toarray() @ np.asarray(x), np.ones(n), atol=1e-8)


def test_linear_operator_prepares_the_split_only_for_a_traced_operator():
    """Traced (a jit argument, how every compiled solve receives it): split once, same map as BCOO.
    Concrete: left to BCOO's own ``@`` -- XLA folds the index work of a compile-time constant, and a
    cached split would hold 8 B/nonzero of device memory for nothing."""
    from jno.utils.solver.solver_api import LinearOperator

    A = _bcoo(40, 55, seed=7)
    assert LinearOperator(A)._split is None
    rng = np.random.default_rng(4)
    v, w = jnp.asarray(rng.standard_normal(55)), jnp.asarray(rng.standard_normal(40))

    @jax.jit
    def apply(A, v, w):
        op = LinearOperator(A)
        assert op._split is not None and op.T._split is op._split
        return op @ v, op.T @ w, op.T.T @ v

    y, yt, ytt = apply(A, v, w)
    _close(y, A @ v)
    _close(yt, A.T @ w)
    _close(ytt, A @ v)


def test_slot_composed_krylov_solve_has_no_bcoo_matvec():
    """``fem.solve(linear=jno.solve.cg())`` compiles through ``_composed_compiled`` with the operator
    as an argument: its loop must use the prepared split, not BCOO's ``@``."""
    import jno
    from jno.utils.solver.solver_api import _composed_compiled

    n = 40
    S = sp.diags([-1.0, 2.2, -1.0], [-1, 0, 1], (n, n)).tocoo()
    A = jsp.BCOO.from_scipy_sparse(S)
    b = jnp.ones(n)
    for linear in (jno.solve.cg(tol=1e-12), jno.solve.bicgstab(tol=1e-12), jno.solve.gmres(tol=1e-12)):
        f = lambda A, b: _composed_compiled(A, b, None, linear=linear, precond=jno.precond.jacobi())  # noqa: E731
        text = str(jax.make_jaxpr(f)(A, b))
        assert "bcoo_dot_general" not in text, linear
        x = f(A, b)
        np.testing.assert_allclose(S.toarray() @ np.asarray(x), np.ones(n), atol=1e-8)
