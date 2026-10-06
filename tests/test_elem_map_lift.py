"""elem_map LIFTS large captured arrays to jit arguments instead of baking them in as constants.

Oracles:
* the lifted path returns exactly what the baked path returns (plain, chunked, scatter);
* two closures with the same code and shapes but DIFFERENT captured values share one compilation,
  and each still gets its own answer -- the program depends on the arrays' shapes, not their values;
* the compiled program carries no large constant, and nothing keeps the captured array alive;
* a tracer among the captures (an enclosing trace) falls back to the baked path, gradients intact;
* sibling closures that share a cell still share it after the rebuild.
"""

from __future__ import annotations

import gc
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver import fem_utils as fu

N = 20_000  # 160 KB of float64: above _LIFT_MIN_BYTES


@pytest.fixture(autouse=True)
def _x64_and_clean():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    fu._ELEM_MAP_LIFTED.clear()
    fu._ELEM_MAP_CACHE.clear()
    fu._ELEM_MAP_CONTENT.clear()
    yield
    jax.config.update("jax_enable_x64", prev)


def _kernel(table, w):
    """A two-level closure like the assembler's: an inner helper captures the big table."""

    def gather(c):
        return table[c] * w

    def elem(c, x):
        return gather(c) + x * x

    return elem


def _run(fn, xs, chunk=None, scatter=None, lift=True):
    prev = fu._LIFT_ENABLED
    fu._LIFT_ENABLED = lift
    try:
        return fu.elem_map(fn, xs, chunk, scatter=scatter)
    finally:
        fu._LIFT_ENABLED = prev


def _inputs(seed):
    rng = np.random.default_rng(seed)
    table = jnp.asarray(rng.standard_normal(N))
    c = jnp.arange(N)
    x = jnp.asarray(rng.standard_normal(N))
    return table, c, x


@pytest.mark.parametrize("chunk", [None, 4096])
def test_lifted_equals_baked(chunk):
    table, c, x = _inputs(0)
    fn = _kernel(table, 2.5)
    got = _run(fn, (c, x), chunk)
    ref = _run(fn, (c, x), chunk, lift=False)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
    np.testing.assert_allclose(np.asarray(got), np.asarray(table) * 2.5 + np.asarray(x) ** 2, rtol=1e-14, atol=1e-15)
    assert len(fu._ELEM_MAP_LIFTED) == 1, "the lifted path was not taken"


@pytest.mark.parametrize("chunk", [None, 4096])
def test_scatter_lifted_equals_baked(chunk):
    table, c, x = _inputs(1)
    fn = _kernel(table, -1.0)
    index = jnp.asarray(np.random.default_rng(2).integers(0, 500, size=N))
    out = jnp.zeros(500)
    got = _run(fn, (c, x), chunk, scatter=(out, index))
    ref = _run(fn, (c, x), chunk, scatter=(out, index), lift=False)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-12)


def test_same_shapes_different_values_share_one_compile_and_stay_correct():
    t1, c, x = _inputs(3)
    t2, _, _ = _inputs(4)
    m0 = fu._ELEM_MAP_STATS["misses"]
    r1 = _run(_kernel(t1, 1.0), (c, x))
    r2 = _run(_kernel(t2, 1.0), (c, x))
    assert fu._ELEM_MAP_STATS["misses"] - m0 == 1, "a same-shaped mesh recompiled"
    np.testing.assert_allclose(np.asarray(r1), np.asarray(t1) + np.asarray(x) ** 2, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(r2), np.asarray(t2) + np.asarray(x) ** 2, rtol=1e-14, atol=1e-15)
    # a different SCALAR is baked, so it must not share the program
    _run(_kernel(t1, 3.0), (c, x))
    assert fu._ELEM_MAP_STATS["misses"] - m0 == 2


def test_no_large_constant_in_the_program_and_nothing_pinned():
    table, c, x = _inputs(5)
    ref = weakref.ref(table)
    _run(_kernel(table, 1.0), (c, x))
    (jf,) = fu._ELEM_MAP_LIFTED.values()
    text = jf.lower((table,), c, x).as_text()
    assert len(text) < 20_000, f"the lowered program is {len(text)} chars: a constant was baked in"
    del table
    gc.collect()
    assert ref() is None, "the compiled-kernel cache keeps the captured array alive"


def test_tracer_capture_falls_back_and_differentiates():
    _, c, x = _inputs(6)
    t0 = jnp.asarray(np.random.default_rng(7).standard_normal(N))

    def loss(table):
        return jnp.sum(_run(_kernel(table, 2.0), (c, x)))

    g = jax.grad(loss)(t0)
    np.testing.assert_allclose(np.asarray(g), 2.0, rtol=0, atol=0)


def test_shared_cells_stay_shared():
    table, c, x = _inputs(8)

    def make(tab):
        scale = 3.0

        def a(i):
            return tab[i] * scale

        def b(i, y):
            return a(i) + tab[i] + y  # `tab` is ONE cell shared by `a` and `b`

        return b

    got = _run(make(table), (c, x))
    np.testing.assert_allclose(np.asarray(got), 4.0 * np.asarray(table) + np.asarray(x), rtol=1e-15)


class _Opaque:
    """A leaf the content tokenizer cannot key by value (like a trace ``FunctionCall`` node)."""

    def __init__(self, k):
        self.k = k


def test_untokenizable_leaf_is_id_keyed_still_lifted_and_not_pinned():
    t1, c, x = _inputs(9)
    t2, _, _ = _inputs(10)
    op = _Opaque(2.0)

    def make(tab):
        def elem(i, y):
            return tab[i] * op.k + y

        return elem

    m0 = fu._ELEM_MAP_STATS["misses"]
    k0 = fu._ELEM_MAP_STATS.get("lift_id_keyed", 0)
    r1 = _run(make(t1), (c, x))
    r2 = _run(make(t2), (c, x))  # same opaque object, same shapes: one program
    assert fu._ELEM_MAP_STATS.get("lift_id_keyed", 0) - k0 == 2
    assert fu._ELEM_MAP_STATS["misses"] - m0 == 1
    np.testing.assert_allclose(np.asarray(r1), 2.0 * np.asarray(t1) + np.asarray(x), rtol=1e-15)
    np.testing.assert_allclose(np.asarray(r2), 2.0 * np.asarray(t2) + np.asarray(x), rtol=1e-15)
    ref = weakref.ref(t1)
    del t1, r1
    gc.collect()
    assert ref() is None, "an id-keyed lifted entry pins the captured array"


def _memo_kernel(table, pts):
    """Like fem_native's ``_static_geometry``: a memo filled at trace time from CONCRETE captures."""
    memo = []

    def geometry():
        if not memo:
            with jax.ensure_compile_time_eval():
                memo.append(pts * 2.0)
        return memo[0]

    def elem(c, x):
        return table[c] + geometry()[c] + x

    return elem, memo


@pytest.mark.parametrize("chunk", [None, 4096])
def test_trace_time_memo_stays_concrete_and_does_not_leak(chunk):
    table, c, x = _inputs(11)
    pts = jnp.asarray(np.random.default_rng(12).standard_normal(N))
    fn, memo = _memo_kernel(table, pts)
    got = _run(fn, (c, x), chunk)
    np.testing.assert_allclose(np.asarray(got), np.asarray(table) + 2 * np.asarray(pts) + np.asarray(x), rtol=1e-14)
    assert memo and not isinstance(memo[0], jax.core.Tracer), "the memo holds a tracer"
    got2 = _run(fn, (c, x), chunk)  # a second call must not trip over a leaked tracer
    np.testing.assert_array_equal(np.asarray(got2), np.asarray(got))


def test_untouched_function_is_value_keyed_no_false_sharing():
    """Arrays behind a trace-time function are BAKED, so two meshes that differ only there must not
    share a program even though every array has the same shape."""
    table, c, x = _inputs(13)
    p1 = jnp.asarray(np.random.default_rng(14).standard_normal(N))
    p2 = jnp.asarray(np.random.default_rng(15).standard_normal(N))
    r1 = _run(_memo_kernel(table, p1)[0], (c, x))
    r2 = _run(_memo_kernel(table, p2)[0], (c, x))
    np.testing.assert_allclose(np.asarray(r1), np.asarray(table) + 2 * np.asarray(p1) + np.asarray(x), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(r2), np.asarray(table) + 2 * np.asarray(p2) + np.asarray(x), rtol=1e-14)


def test_bookkeeping_list_keeps_its_identity():
    table, c, x = _inputs(16)
    log = []

    def elem(i, y):
        log.append(y.shape)  # trace-time bookkeeping on a list with nothing to lift
        return table[i] + y

    _run(elem, (c, x))
    assert log, "the trace wrote into a copy of the list, not the list itself"
