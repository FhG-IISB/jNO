"""MMA's subproblem is JAX: one compiled float64 program on the design's device, batchable with `jax.vmap`.

Oracles: a batch of subproblems solved by one `vmap` equals each solved alone, and each meets its KKT conditions
(the rows feasible, tight where the multiplier is positive: tests/test_optimizers_mma.py's oracle); a process in
float32 gets the same float64 answer as one in float64; the compiled step holds no host callback.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno  # noqa: F401  (registers the package)

M = sys.modules["jno.optimizers.mma"]  # the package's `mma` is the constructor
from tests.test_optimizers_mma import _kkt, _random_subproblem  # noqa: E402


def _args(seed):
    x, df0, g, dg, low, upp, xmin, xmax, move = _random_subproblem(seed)
    span = np.maximum(xmax - xmin, 1e-12)
    return x, df0, g, dg, low, upp, xmin, xmax, move * np.ones_like(x), span


def _jit_args(a):
    x, df0, g, dg, low, upp, xmin, xmax, move, _span = a
    return tuple(jnp.asarray(v, jnp.float64) for v in (x, df0, g, dg, low, upp, xmin, xmax, move)) + (
        jnp.float64(1e-5),
        jnp.float64(1e-5),
    )


def test_a_vmapped_batch_of_subproblems_equals_each_solved_alone():
    seeds = [0, 1, 2, 3]
    with jax.enable_x64(True):
        batch = [_jit_args(_args(s)) for s in seeds]
        stacked = tuple(jnp.stack([b[i] for b in batch]) for i in range(len(batch[0])))
        solve = lambda *a: M._approximation_jit(*a, None, sweeps=5, has_start=False)[:2]  # noqa: E731
        xb, lb = jax.vmap(solve)(*stacked)
        for k, s in enumerate(seeds):
            x1, l1 = solve(*batch[k])
            np.testing.assert_allclose(np.asarray(xb[k]), np.asarray(x1), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(np.asarray(lb[k]), np.asarray(l1), rtol=1e-10, atol=1e-14, equal_nan=True)
            x, df0, g, dg, low, upp, *_ = _args(s)
            rows, size, lam = _kkt(np.asarray(xb[k]), np.asarray(lb[k]), x, g, dg, low, upp)
            ok = np.isfinite(lam)
            assert np.all(rows[ok] <= 1e-8 * size[ok])


def test_the_subproblem_is_float64_in_a_float32_process():
    a = _args(5)
    prev = jax.config.jax_enable_x64
    try:
        jax.config.update("jax_enable_x64", True)
        x64, l64 = M._solve_approximation(*a[:8], a[8], 1e-5, 1e-5, 400)
        x64, l64 = np.asarray(x64), np.asarray(l64)
        jax.config.update("jax_enable_x64", False)
        x32, l32 = M._solve_approximation(*a[:8], a[8], 1e-5, 1e-5, 400)
        assert x32.dtype == jnp.float64 and l32.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(x32), x64)
        np.testing.assert_array_equal(np.asarray(l32), l64)
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("has_start", [False, True])
def test_the_compiled_step_has_no_host_callback(has_start):
    with jax.enable_x64(True):
        args = _jit_args(_args(0))
        lam0 = jnp.zeros(args[2].shape[0]) if has_start else None
        text = str(jax.make_jaxpr(lambda *a: M._approximation_jit(*a, lam0, sweeps=5, has_start=has_start))(*args))
    for prim in ("pure_callback", "io_callback", "debug_callback", "host_callback"):
        assert prim not in text, prim
    assert "while" in text  # the dual's iterations are in the program, not in Python
