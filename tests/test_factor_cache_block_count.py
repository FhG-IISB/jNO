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

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

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


def _count_factorizations(monkeypatch):
    """Count the SuperLU factorizations ``host_lu_solve`` performs."""
    import scipy.sparse.linalg as spla

    calls = [0]
    real = spla.splu

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(spla, "splu", counted)
    return calls


def test_a_round_robin_over_n_blocks_reuses_every_factorization(monkeypatch):
    """Sweep distinct operators the way `triangular` does. A factorization is kept once its operator RECURS, so
    the first sweep keeps none, the second factors each block once more and keeps it, and every later sweep is
    all hits -- which is what fails at a bound below N."""
    n_blocks, n = 4, 24
    ops = [_operator(n, 10 * k) for k in range(n_blocks)]
    b = jnp.asarray(np.ones(n))
    calls = _count_factorizations(monkeypatch)

    for op in ops:  # first sweep: nothing recurs yet
        _linear.host_lu_solve(op, b)
    assert len(_linear._FACTOR_CACHE) == 0
    for op in ops:  # second sweep: each block is met again and kept
        _linear.host_lu_solve(op, b)
    assert len(_linear._FACTOR_CACHE) == n_blocks, (
        f"{len(_linear._FACTOR_CACHE)} entries held for {n_blocks} distinct blocks -- "
        "an entry was evicted before it could be reused"
    )
    assert calls[0] == 2 * n_blocks, f"{calls[0]} factorizations over two sweeps: more than one extra per block"

    keys_after_second = set(_linear._FACTOR_CACHE)
    for op in ops:  # third sweep: must be all hits
        _linear.host_lu_solve(op, b)
    assert set(_linear._FACTOR_CACHE) == keys_after_second
    assert calls[0] == 2 * n_blocks, "the third sweep re-factorised instead of hitting the cache"


def test_a_resolve_loop_over_changing_operators_keeps_no_factorization(monkeypatch):
    """A coefficient rescaled on every call meets each operator in one solve and its adjoint, never again. The
    adjoint shares the forward's factorization, and nothing is kept in the cache (it used to hold eight dead
    ones: 886 -> 2441 MB over 13 calls of a 13k-unknown 3-D problem)."""
    import jax

    n = 24
    base = _operator(n, 5)
    b = jnp.asarray(np.ones(n))
    calls = _count_factorizations(monkeypatch)

    def loss(s):
        A = jsp.BCOO((base.data * s, base.indices), shape=base.shape)
        return jnp.sum(_linear.host_lu_solve(A, b) ** 2)

    g = jax.jit(jax.grad(loss))
    scales = [1.0 + 0.1 * i for i in range(_linear._FACTOR_CACHE_MAX + 3)]
    for s in scales:
        g(s).block_until_ready()
    assert len(_linear._FACTOR_CACHE) == 0, f"{len(_linear._FACTOR_CACHE)} factorizations kept by a loop that never repeats"
    assert calls[0] == len(scales), f"{calls[0]} factorizations for {len(scales)} operators (forward and adjoint share one)"


def test_a_constant_operator_march_factors_once_and_then_keeps_it(monkeypatch):
    """The constant-operator case the cache exists for: one factorization serves every step, and it is kept."""
    n = 30
    op, b = _operator(n, 1.0), jnp.ones(n)
    calls = _count_factorizations(monkeypatch)
    for _ in range(6):
        _linear.host_lu_solve(op, b)
    assert calls[0] == 1, f"{calls[0]} factorizations over 6 solves of one operator"
    assert len(_linear._FACTOR_CACHE) == 1


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


def _two_root_newton(linear):
    """eps lap u + (u^2 - 1) v = 0 from u = 0.5: Newton takes several steps to the root u = 1."""
    import jno

    d = jno.shape.rect(0, 0, 1, 1).sized(0.1).domain()
    x, y, _ = d.variable("interior", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([1e-3 * (ui.d(x) * vi.d(x) + ui.d(y) * vi.d(y)) + (ui**2 - 1.0) * vi])
    x0 = jnp.full(len(np.asarray(d.mesh.points)), 0.5)
    sol = fem.solve(x0=x0, nonlinear=jno.solve.newton(direct=True, rtol=1e-12, atol=0.0, max_steps=40), linear=linear)
    return np.asarray(sol)


def test_a_newton_solve_keeps_none_of_its_step_factorizations():
    """A Newton tangent changes every step, so a kept factorization is never hit again: the default keeps none
    (a large coupled problem otherwise holds up to eight dead factorizations). An explicit reuse=True still keeps
    them, and both give the same root."""
    import jno

    plain = _two_root_newton(jno.solve.lu(backend="host"))
    assert len(_linear._FACTOR_CACHE) == 0, f"{len(_linear._FACTOR_CACHE)} Newton-step factorizations kept"
    _linear._FACTOR_CACHE.clear()
    kept = _two_root_newton(jno.solve.lu(backend="host", reuse=True))
    assert len(_linear._FACTOR_CACHE) == 0  # no tangent repeats, so none recurs either way
    np.testing.assert_allclose(plain, 1.0, atol=1e-10)
    np.testing.assert_allclose(plain, kept, rtol=0, atol=1e-13)


def test_outside_a_newton_solve_the_default_still_keeps():
    """The constant-operator case the cache exists for: the default (reuse=None) keeps once the operator has
    been asked for three times, and a repeat hits."""
    n = 30
    op, b = _operator(n, 1.0), jnp.ones(n)
    for _ in range(3):
        _linear.host_lu_solve(op, b)
    assert len(_linear._FACTOR_CACHE) == 1
    with _linear.factorizations_not_kept():
        _linear.host_lu_solve(_operator(n, 2.0), b)
        for _ in range(3):  # an explicit choice wins inside too
            _linear.host_lu_solve(_operator(n, 3.0), b, reuse=True)
    assert len(_linear._FACTOR_CACHE) == 2


def test_a_forked_child_gets_a_working_factor_thread():
    """The host factorizations run on one worker thread; a child forked after it started must not wait on a
    thread that does not exist in the child."""
    import os
    import time

    from jno.utils.solver import linear

    assert linear._factor_worker().submit(lambda: 1).result() == 1
    pid = os.fork()
    if pid == 0:
        try:
            ok = linear._factor_worker().submit(lambda: 2).result(timeout=20) == 2
        except BaseException:
            ok = False
        os._exit(0 if ok else 1)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            assert os.waitstatus_to_exitcode(status) == 0
            return
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    raise AssertionError("the forked child hung on the factor worker")
