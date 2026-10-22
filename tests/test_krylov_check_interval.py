"""BiCGStab tests convergence once per block of iterations, and still returns the sequential loop's answer.

On a GPU, XLA runs a ``while_loop`` by copying its condition to the host after every trip. Running ``m``
iterations per trip, with every iteration after convergence or breakdown FROZEN, must give bitwise the
same ``x``, iteration count and breakdown flag as one iteration per trip -- including convergence in the
middle of a block, a ``maxiter`` that is not a multiple of ``m``, and a breakdown.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno.utils.solver.krylov as kr


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _run(m, A, b, **kw):
    prev = kr._CHECK_EVERY[0]
    kr._CHECK_EVERY[0] = m
    try:
        x, k, broke = jax.jit(lambda b: kr.bicgstab_counted(lambda v: A @ v, b, M=lambda v: v / jnp.diag(A), **kw))(b)
    finally:
        kr._CHECK_EVERY[0] = prev
    return np.asarray(x), int(k), bool(broke)


def _system(n=60, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) * 0.1 + np.diag(np.linspace(1.0, 20.0, n))  # non-symmetric, well posed
    return jnp.asarray(A), jnp.asarray(rng.standard_normal(n))


@pytest.mark.parametrize("m", [2, 3, 8])
@pytest.mark.parametrize("case", ["converges", "maxiter-not-multiple", "zero-rhs"])
def test_a_block_of_iterations_per_trip_is_the_sequential_loop_exactly(m, case):
    # Break: letting the iterations after convergence inside a block keep updating x (or the count).
    A, b = _system()
    kw = {"tol": 1e-10, "maxiter": 1000}
    if case == "maxiter-not-multiple":
        kw = {"tol": 1e-30, "maxiter": 13}  # never converges: stops on the cap, which m does not divide
    if case == "zero-rhs":
        b = jnp.zeros_like(b)  # converged before the first iteration
    ref = _run(1, A, b, **kw)
    got = _run(m, A, b, **kw)
    assert got[1] == ref[1] and got[2] == ref[2], (got[1:], ref[1:])
    np.testing.assert_array_equal(got[0], ref[0])
    if case == "converges":
        assert ref[1] % m != 0 or m == 1  # the interesting case: convergence inside a block


def test_a_breakdown_inside_a_block_stops_where_the_sequential_loop_stops():
    # rhat orthogonal to A r after one step: rho = 0 -> breakdown flag, count kept.
    A = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
    b = jnp.asarray([1.0, 0.0])
    ref = _run(1, A, b, tol=1e-12, maxiter=50)
    got = _run(8, A, b, tol=1e-12, maxiter=50)
    assert got[1:] == ref[1:]
    np.testing.assert_array_equal(got[0], ref[0])  # NaN where the sequential loop has NaN
