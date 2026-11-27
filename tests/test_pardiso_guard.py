"""The PARDISO backend's host solve: single-precision operators and the non-finite guard."""

from __future__ import annotations

import subprocess
import sys
import textwrap

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
