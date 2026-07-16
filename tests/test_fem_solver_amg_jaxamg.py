"""`jno.solve.amg()` — the optional GPU AMG solver slot backed by jaxamg (NVIDIA AmgX).

jaxamg needs a prebuilt AmgX / CUDA / MPI stack, so the end-to-end solve is skipped where it (or a GPU)
is unavailable; the jNO-side plumbing (slot construction, direct-solver contract, matrix-free rejection,
optional-dependency error) is always tested.
"""

import importlib.util

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import pytest

import jno
from jno.utils.solver.solver_api import LinearOperator

_HAS_JAXAMG = importlib.util.find_spec("jaxamg") is not None


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _spd_operator(n=8):
    dense = jnp.diag(2.0 * jnp.ones(n)) + jnp.diag(-1.0 * jnp.ones(n - 1), 1) + jnp.diag(-1.0 * jnp.ones(n - 1), -1)
    return LinearOperator(jsparse.BCOO.fromdense(dense)), dense


def test_amg_slot_constructs():
    s = jno.solve.amg()
    assert s.name == "amg" and s.direct is True  # self-contained AMG solve, owns its preconditioner
    assert "amg" in jno.solve.__all__


def test_amg_is_a_direct_slot_rejecting_precond():
    """A direct solver takes no outer preconditioner — the contract is enforced before any jaxamg import."""
    op, _ = _spd_operator()
    with pytest.raises(ValueError, match="direct"):
        jno.solve.amg()(op, jnp.ones(op.shape[0]), M=lambda v: v)


def test_amg_rejects_matrix_free_operator():
    """No assembled matrix → clear error (jaxamg needs the sparse matrix); checked before the import."""
    n = 6
    op = LinearOperator.from_matvec(lambda v: 2.0 * v, shape=(n, n))
    with pytest.raises(ValueError, match="matrix-free"):
        jno.solve.amg()(op, jnp.ones(n))


@pytest.mark.skipif(_HAS_JAXAMG, reason="jaxamg installed; this checks the missing-dependency message")
def test_amg_missing_dependency_message():
    op, _ = _spd_operator()
    with pytest.raises(ImportError, match="jaxamg"):
        jno.solve.amg()(op, jnp.ones(op.shape[0]))


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_JAXAMG, reason="requires jaxamg + AmgX + CUDA")
def test_amg_matches_direct_on_poisson():
    """End-to-end: an SPD (1-D Poisson) system solved by jno.solve.amg() matches the sparse-direct LU."""
    op, dense = _spd_operator(64)
    b = jnp.ones(op.shape[0])
    x_ref = jnp.linalg.solve(dense, b)
    x_amg = jno.solve.amg(tol=1e-10)(op, b)
    assert jnp.max(jnp.abs(x_amg - x_ref)) < 1e-6


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_JAXAMG, reason="requires jaxamg + AmgX + CUDA")
def test_amg_composes_as_preconditioner():
    """`jno.precond.inner(jno.solve.amg(...))` uses the AMG solve as M⁻¹ inside an outer Krylov."""
    op, dense = _spd_operator(64)
    b = jnp.ones(op.shape[0])
    x_ref = jnp.linalg.solve(dense, b)
    solve = jno.solve.cg(tol=1e-10)
    x = solve(op, b, M=jno.precond.inner(jno.solve.amg(maxiter=1, krylov=None)))
    assert jnp.max(jnp.abs(x - x_ref)) < 1e-6
