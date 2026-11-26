"""A finished transient build must not stay in memory: nothing but jNO's own bounded caches may keep it.

A loop of builds-and-solves in one process grew by ~30 MB per build on a 100-cell mesh. Two caches kept every
finished problem alive without bound: ``functools.lru_cache`` on the evaluator's ``_accepts_key`` (a strong
key on the per-build closure) and JAX's memo of ``closure_convert`` (a strong key on the Newton residual,
with its constants as the value). The test builds and solves a small nonlinear transient, empties jNO's
bounded kernel caches (which hold a problem on purpose, see ``_ELEM_MAP_CACHE``) and requires the problem
to be collected.
"""

from __future__ import annotations

import gc
import weakref

import jax
import jax.numpy as jnp
import pytest

import jno
from jno.trace_evaluator import TraceEvaluator
from jno.utils.solver import fem_utils


@pytest.fixture(autouse=True)
def _x64():
    """These builds run in float64; save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _transient_build(k):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).sized(0.25).domain(time=(0.0, 0.2, 3))
    u, p = d.fem_symbols(names=("u", "p"))
    x, y, t = d.variable("interior", split=True)
    ui, q = u.bind(x=x, y=y, t=t), p.bind(x=x, y=y, t=t)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    fem = jno.fem(
        [
            ui.t * q + (1.0 + 0.5 * ui * ui) * (ui.x * q.x + ui.y * q.y) * 0.1 - k * q,
            u(xb, yb, tb) - 0.0,
            u(ci[0], ci[1]) - 0.0,
        ]
    )
    node = fem.solve()
    traj = TraceEvaluator({}).evaluate(node.expr if hasattr(node, "expr") else node, context={})
    return d, float(jnp.sum(traj[-1]))


def _clear_bounded_kernel_caches():
    for name in ("_ELEM_MAP_CACHE", "_ELEM_MAP_CONTENT", "_ELEM_MAP_LIFTED", "_PLAN_CACHE", "_LEAF_DIGEST_CACHE"):
        getattr(fem_utils, name).clear()


def test_a_finished_nonlinear_transient_build_is_not_kept_alive():
    refs = []
    for k in (1.0, 1.5):
        d, value = _transient_build(k)
        assert jnp.isfinite(value)
        refs.append(weakref.ref(d))
        del d
    _clear_bounded_kernel_caches()
    gc.collect()
    alive = sum(r() is not None for r in refs)
    assert alive == 0, (
        f"{alive} of {len(refs)} finished builds are still referenced after jNO's kernel caches were emptied: "
        "something outside them (a module-level cache, a JAX memo keyed on the problem's closures) pins the problem"
    )


def test_accepts_key_does_not_pin_its_function():
    from jno.trace_evaluator import _accepts_key

    def with_key(x, key=None):
        return x

    def without(x):
        return x

    assert _accepts_key(with_key) is True and _accepts_key(without) is False
    assert _accepts_key(with_key) is True  # answered from the cache
    ref = weakref.ref(with_key)
    del with_key
    gc.collect()
    assert ref() is None
