"""The PARDISO backend's host solve: single-precision operators and the non-finite guard."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import pytest


def _skip_unless_available():
    from jno.utils.solver.linear import _pardiso_available

    if not _pardiso_available():
        pytest.skip("pypardiso (with the private phase hooks this backend drives) is not installed")


def _poisson(n, dtype):
    A = np.diag(np.full(n, 2.0)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    return jsparse.BCOO.fromdense(jnp.asarray(A, dtype=dtype), nse=3 * n - 2), A


def test_a_float32_operator_is_factored_in_double_and_answered_in_the_rhs_dtype():
    """PyPardiso refuses float32 with a TypeError; the solve now widens for the factorisation and casts back."""
    _skip_unless_available()
    from jno.utils.solver.linear import pardiso_lu_solve

    A32, A = _poisson(8, jnp.float32)
    b = jnp.arange(1.0, 9.0, dtype=jnp.float32)
    x = pardiso_lu_solve(A32, b)
    assert x.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(A, np.asarray(b, np.float64)), rtol=1e-5)


def test_the_float32_solve_keeps_its_dtype_through_the_adjoint():
    """The transpose solve of ``custom_linear_solve`` runs the same callback: cotangents stay float32."""
    _skip_unless_available()
    from jno.utils.solver.linear import pardiso_lu_solve

    A32, A = _poisson(8, jnp.float32)
    b = jnp.arange(1.0, 9.0, dtype=jnp.float32)
    g_b, g_data = jax.grad(
        lambda bb, dd: jnp.sum(pardiso_lu_solve(jsparse.BCOO((dd, A32.indices), shape=A32.shape), bb) ** 2), (0, 1)
    )(b, A32.data)
    assert g_b.dtype == jnp.float32 and g_data.dtype == jnp.float32
    x = np.linalg.solve(A, np.asarray(b, np.float64))
    np.testing.assert_allclose(np.asarray(g_b), 2.0 * np.linalg.solve(A.T, x), rtol=1e-4)


def test_a_float64_operator_still_answers_in_float64():
    _skip_unless_available()
    from jno.utils.solver.linear import pardiso_lu_solve

    with jax.enable_x64(True):
        A64, A = _poisson(8, jnp.float64)
        b = jnp.arange(1.0, 9.0, dtype=jnp.float64)
        x = pardiso_lu_solve(A64, b)
        assert x.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(x), np.linalg.solve(A, np.asarray(b)), rtol=1e-12)


GUARD = textwrap.dedent(
    """
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp, jax.experimental.sparse as jsparse, numpy as np
    from jno.utils.solver.linear import pardiso_lu_solve
    n = 6
    A = jsparse.BCOO.fromdense(jnp.eye(n) * 2.0, nse=n)
    bad = jsparse.BCOO((A.data.at[2].set(jnp.nan), A.indices), shape=A.shape)
    try:
        pardiso_lu_solve(bad, jnp.ones(n)).block_until_ready()
    except Exception as e:  # the callback error surfaces wrapped in a JaxRuntimeError
        assert "non-finite" in str(e), str(e)[-400:]
        print("GUARDED")
    else:
        raise SystemExit("no error for a NaN operator")
    # the process is still alive and PARDISO still works
    print(float(pardiso_lu_solve(A, jnp.ones(n))[0]))
    """
)


def test_a_nan_operator_raises_the_guards_error_and_the_process_survives():
    """MKL PARDISO aborts the interpreter (SIGABRT) on a NaN system, so the guard in ``_pardiso_host_solve`` must
    fire first. Run in a subprocess: if the guard is gone, this test sees the abort instead of dying with it."""
    _skip_unless_available()
    r = subprocess.run([sys.executable, "-c", GUARD], capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""})  # fmt: skip
    assert r.returncode == 0, f"exit {r.returncode} (134/-6 = SIGABRT, the guard is gone):\n{r.stderr[-800:]}"
    assert "GUARDED" in r.stdout and r.stdout.split()[-1] == "0.5", r.stdout + r.stderr[-400:]


def test_the_guard_message_counts_the_non_finite_entries():
    _skip_unless_available()
    from jno.utils.solver.linear import _pardiso_host_solve

    data = np.array([2.0, np.nan, 2.0, np.inf])
    idx = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    with pytest.raises(FloatingPointError, match=r"2 of 4 matrix value"):
        _pardiso_host_solve(data, idx, np.ones(4), (4, 4), False)


def _exact_2x2(A, b):
    """The exact solution of a 2x2 system in rational arithmetic: a pivot of 1e-13 makes the system far too
    ill-conditioned for a dense double-precision solve to serve as the oracle."""
    from fractions import Fraction as F

    (a, c), (d, e) = [[F(float(v)) for v in row] for row in A]
    r0, r1 = F(float(b[0])), F(float(b[1]))
    det = a * e - c * d
    return np.array([float((r0 * e - c * r1) / det), float((a * r1 - d * r0) / det)])


def _perturbed_pivot_system(transpose=False, pivot=9e-14, size=1e7):
    """Nonsingular, but its second pivot (``pivot``) is below the 1e-13 PARDISO replaces: the factorization is of
    a perturbed matrix, and the raw solution has a residual of 1e-7 (under the singularity gate) and no digits."""
    A = np.array([[1.0, 1.0], [1.5, 1.5 + pivot]])
    # the right-hand side puts a solution entry of ``size`` on the pivot's row, for A x = b or for A^T x = b
    return A, np.array([1.0, (1.0 if transpose else 1.5) + size * pivot])


@pytest.mark.parametrize("transpose", [False, True])
def test_a_perturbed_pivot_of_a_nonsingular_operator_is_refined_and_reported(transpose):
    """PARDISO perturbed the pivot, the residual (8e-8) is far under the singularity gate, and the raw vector
    was 16 % off (cond 1e13) with nothing said. Now the solve is refined against the true operator and the
    perturbed pivots and the residual before and after are reported."""
    _skip_unless_available()
    import scipy.sparse as sp

    from jno.utils.solver import linear as lin

    A, b = _perturbed_pivot_system(transpose)
    M = sp.coo_matrix(A)
    idx = np.stack([M.row, M.col], axis=1)
    oracle = _exact_2x2(A.T if transpose else A, b)
    lin._PARDISO_CACHE.clear()
    lin._warned_pardiso_perturbed = False
    with pytest.warns(
        RuntimeWarning, match=r"perturbed 1 pivot\(s\).*raw solution had relative residual .*brought it to"
    ) as w:
        x = lin._pardiso_host_solve(M.data, idx, b, (2, 2), transpose)
    assert "refined" in str(w[0].message)
    assert next(iter(lin._PARDISO_CACHE.values()))["perturbed"] == 1
    op = A.T if transpose else A
    assert np.linalg.norm(op @ x - b) / np.linalg.norm(b) < 1e-9  # the raw solve is at 8e-8
    assert np.linalg.norm(x - oracle) / np.linalg.norm(oracle) < 1e-2  # the raw solve is 16 % off
    # a second solve against the same factorization is refined too, and says nothing more
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x2 = lin._pardiso_host_solve(M.data, idx, b, (2, 2), transpose)
    np.testing.assert_allclose(x2, x)


def test_a_perturbed_pivot_of_a_singular_operator_still_raises():
    _skip_unless_available()
    import scipy.sparse as sp

    from jno.utils.solver import linear as lin

    M = sp.coo_matrix(np.array([[1.0, 1.0], [1.0, 1.0 + 1e-14]]))
    lin._PARDISO_CACHE.clear()
    with pytest.raises(RuntimeError, match=r"SINGULAR.*perturbed 1 pivot.*before refinement"):
        lin._pardiso_host_solve(M.data, np.stack([M.row, M.col], axis=1), np.array([1.0, 2.0]), (2, 2), False)
