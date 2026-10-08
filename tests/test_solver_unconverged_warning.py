"""A Krylov solve that stops on its step cap short of the requested tolerance must say so.

The residual gate used to be a fixed 1e-4 whatever the request: a solve asked for 1e-8 that left on
its iteration cap at 1.8e-5 returned a 1.7 %-wrong answer with no signal. Between 100x the request and
the hard gate it now warns (a consistent singular system floors there legitimately); above the hard
gate it still raises."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import UnconvergedSolveWarning


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _system(n=200, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(np.logspace(0, 2, n)) @ Q.T  # SPD, condition 1e2: CG needs ~70 steps to 1e-8
    return jnp.asarray(A), jnp.asarray(rng.standard_normal(n))


def _residual(A, b, x):
    return float(jnp.linalg.norm(A @ x - b) / jnp.linalg.norm(b))


@pytest.mark.parametrize("solver", ["cg", "gmres", "fgmres"])
def test_step_cap_short_of_the_tolerance_warns(solver):
    A, b = _system()
    s = {
        "cg": jno.solve.cg(tol=1e-10, maxiter=60),
        "gmres": jno.solve.gmres(tol=1e-10, restart=10, maxiter=8),
        "fgmres": jno.solve.fgmres(tol=1e-10, restart=10, maxiter=70),
    }[solver]
    with pytest.warns(UnconvergedSolveWarning, match="short of the 1e-10"):
        x = s(A, b)
    assert 1e-8 < _residual(A, b, x) < 1e-4  # in the warning band: kept, but flagged


def test_converged_solve_is_silent():
    A, b = _system()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnconvergedSolveWarning)
        x = jno.solve.cg(tol=1e-10, maxiter=5000)(A, b)
    assert _residual(A, b, x) < 1e-8
