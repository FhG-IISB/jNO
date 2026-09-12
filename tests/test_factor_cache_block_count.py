"""The factorization cache must hold one entry per block a preconditioner sweeps.

``_FACTOR_CACHE`` exists so a repeated operator is factorised once. It was bounded at 2, which the
docstring justified as "an alternating pair (a coupled two-field march)" -- correct for a monolithic
solve, where only one or two distinct operators ever exist.

A BLOCK preconditioner breaks that assumption. ``jno.precond.triangular((T, inner(lu())), (u, ...),
(p, ...), (w, ...))`` calls the host solve once per block, with a DIFFERENT operator each time, and
it does so again on every Krylov application. With a bound below the block count the cache evicts an
entry before it is ever reused, so the hit rate is not merely reduced -- it is exactly zero, and
every application re-factorises.

Measured on the 4-field melt pool (T, u, p, w) at h=8um, 120 steps, where the answer is identical in
every configuration (peak T 1917 K, peak |u| 5.1616e-01 m/s):

    bound 2, reuse=True    OOM -- SIGKILL, host RSS climbs until the kernel intervenes
    bound 2, reuse=False   3330 s
    bound 8, reuse=True    1998 s

The bound is a CAP, not an allocation: a monolithic solve inserts one key and holds one
factorization whatever the cap is, so raising it cannot cost that case anything. What it costs is
bounded by what a caller actually puts in.

These tests pin the mechanism -- eviction against block count -- not a wall-clock number.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp

import jno
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
    """The cache is module-level; leave it as found."""
    saved = dict(_linear._FACTOR_CACHE)
    _linear._FACTOR_CACHE.clear()
    try:
        yield
    finally:
        _linear._FACTOR_CACHE.clear()
        _linear._FACTOR_CACHE.update(saved)


def _operator(n, shift):
    """A distinct, well-conditioned SPD-ish operator per block."""
    d = np.arange(1, n + 1, dtype=float) + float(shift)
    A = sp.diags([d, -np.ones(n - 1), -np.ones(n - 1)], [0, 1, -1], format="coo")
    return jsp.BCOO(
        (jnp.asarray(A.data), jnp.asarray(np.stack([A.row, A.col], axis=1))),
        shape=(n, n),
    )


def test_the_bound_is_at_least_a_four_field_block_sweep():
    """The regression this file exists for. Four fields is an ordinary multiphysics system -- the
    melt pool is exactly (T, u, p, w) -- so a bound below 4 makes block preconditioning re-factorise
    every application. Stated as a floor, not an equality, so raising it further stays legal."""
    assert _linear._FACTOR_CACHE_MAX >= 4, (
        f"_FACTOR_CACHE_MAX is {_linear._FACTOR_CACHE_MAX}: a block preconditioner over 4 fields "
        "evicts every entry before reuse, so the hit rate is zero and each application re-factorises"
    )


def test_a_round_robin_over_n_blocks_reuses_every_factorization():
    """Sweep distinct operators the way `triangular` does, twice. On the second pass every operator
    must already be cached -- that is the whole point, and it is what fails at a bound below N."""
    n_blocks, n = 4, 24
    ops = [_operator(n, 10 * k) for k in range(n_blocks)]
    b = jnp.asarray(np.ones(n))

    for op in ops:  # first sweep: populate
        _linear.host_lu_solve(op, b)
    assert len(_linear._FACTOR_CACHE) == n_blocks, (
        f"{len(_linear._FACTOR_CACHE)} entries held for {n_blocks} distinct blocks -- "
        "an entry was evicted before it could be reused"
    )

    keys_after_first = set(_linear._FACTOR_CACHE)
    for op in ops:  # second sweep: must be all hits
        _linear.host_lu_solve(op, b)
    assert set(_linear._FACTOR_CACHE) == keys_after_first, (
        "the second sweep re-factorised instead of hitting the cache"
    )


def test_the_cached_solve_is_the_same_answer_as_an_uncached_one():
    """Caching is an optimisation, so it must not move the answer. `reuse=False` factorises inside
    the callback and frees it, which is the independent path."""
    n = 24
    op = _operator(n, 3)
    b = jnp.asarray(np.linspace(-1.0, 1.0, n))
    cached = np.asarray(_linear.host_lu_solve(op, b))
    _linear._FACTOR_CACHE.clear()
    uncached = np.asarray(_linear.host_lu_solve(op, b, reuse=False))
    assert np.allclose(cached, uncached, rtol=0.0, atol=0.0), (
        f"cached and uncached solves differ by {np.abs(cached - uncached).max():.3e}"
    )


def test_the_cache_still_evicts_so_it_cannot_grow_without_bound():
    """The bound must still BE a bound: a caller with more distinct operators than the cap must not
    accumulate them forever. This is what stops a Newton march -- whose tangent changes every
    iteration and never repeats -- from retaining one factorization per iteration."""
    n = 16
    over = _linear._FACTOR_CACHE_MAX + 3
    b = jnp.asarray(np.ones(n))
    for k in range(over):
        _linear.host_lu_solve(_operator(n, 100 * k), b)
    assert len(_linear._FACTOR_CACHE) <= _linear._FACTOR_CACHE_MAX, (
        f"{len(_linear._FACTOR_CACHE)} entries against a cap of {_linear._FACTOR_CACHE_MAX}"
    )
