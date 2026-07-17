"""`jno.precond.cached(spec)` — memoise any preconditioner's setup across solves (backend-agnostic)."""

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import pytest

import jno
from jno.precond import _Spec  # base that grants the fluent .cached() (user specs can inherit it too)
from jno.utils.solver.solver_api import (
    LinearOperator,
    PrecondApplier,
    PrecondContext,
    materialize_precond,
    prepare_precond,
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _op(n=6):
    dense = jnp.diag(2.0 * jnp.ones(n)) + jnp.diag(-1.0 * jnp.ones(n - 1), 1) + jnp.diag(-1.0 * jnp.ones(n - 1), -1)
    return LinearOperator(jsparse.BCOO.fromdense(dense)), dense


class _Counting(_Spec):
    """A minimal spec that counts how many times its (notional) setup is built."""

    def __init__(self):
        self.n = 0

    def materialize(self, ctx):
        self.n += 1
        d = ctx.diag()
        inv = 1.0 / jnp.where(jnp.abs(d) > 1e-30, d, 1.0)
        return PrecondApplier(lambda v: inv * v)


def test_cached_builds_once_and_reuses_applier():
    inner = _Counting()
    c = jno.precond.cached(inner)
    op, _ = _op()
    m1 = materialize_precond(c, PrecondContext(op, None))
    m2 = materialize_precond(c, PrecondContext(op, None))
    assert inner.n == 1  # setup built exactly once across two solves
    assert m1 is m2  # same applier object reused


def test_cached_frozen_reuses_even_when_operator_changes():
    inner = _Counting()
    c = jno.precond.cached(inner)  # refresh=False (frozen)
    materialize_precond(c, PrecondContext(_op(6)[0], None))
    materialize_precond(c, PrecondContext(_op(8)[0], None))  # different operator, still reuses
    assert inner.n == 1


def test_cached_refresh_rebuilds_on_structure_change():
    inner = _Counting()
    c = jno.precond.cached(inner, refresh=True)
    materialize_precond(c, PrecondContext(_op(6)[0], None))
    materialize_precond(c, PrecondContext(_op(8)[0], None))  # shape/nnz changed → rebuild
    assert inner.n == 2


def test_cached_custom_key():
    inner = _Counting()
    c = jno.precond.cached(inner, refresh=lambda ctx: ctx.A.shape[0])  # key on size
    materialize_precond(c, PrecondContext(_op(6)[0], None))
    materialize_precond(c, PrecondContext(_op(6)[0], None))  # same size → reuse
    materialize_precond(c, PrecondContext(_op(8)[0], None))  # new size → rebuild
    assert inner.n == 2


def test_cached_matches_uncached_solve():
    """The cached preconditioner produces the same converged solution as the bare one.

    (Calling a solver directly takes a materialised applier as ``M``; ``fem.solve(precond=...)``
    does that materialisation from the spec internally.)"""
    op, dense = _op(48)
    b = jnp.ones(op.shape[0])
    ctx = PrecondContext(op, None)
    x_ref = jnp.linalg.solve(dense, b)
    solve = jno.solve.cg(tol=1e-12)
    x_bare = solve(op, b, M=materialize_precond(jno.precond.jacobi(), ctx))
    x_cached = solve(op, b, M=materialize_precond(jno.precond.cached(jno.precond.jacobi()), ctx))
    assert jnp.max(jnp.abs(x_cached - x_ref)) < 1e-8
    assert jnp.max(jnp.abs(x_cached - x_bare)) < 1e-10


def test_fluent_cached_on_every_spec():
    """Every jno.precond.* spec has a fluent .cached() equivalent to cached(spec)."""
    for spec in (jno.precond.jacobi(), jno.precond.amg(), jno.precond.form([]), jno.precond.chebyshev()):
        wrapped = spec.cached()
        assert type(wrapped).__name__ == "_Cached" and wrapped.spec is spec


def test_fluent_cached_builds_once():
    inner = _Counting()
    c = inner.cached()  # fluent form
    materialize_precond(c, PrecondContext(_op()[0], None))
    materialize_precond(c, PrecondContext(_op()[0], None))
    assert inner.n == 1


def test_fluent_cached_passes_refresh():
    inner = _Counting()
    c = inner.cached(refresh=True)
    materialize_precond(c, PrecondContext(_op(6)[0], None))
    materialize_precond(c, PrecondContext(_op(8)[0], None))
    assert inner.n == 2  # refresh=True rebuilt on the structure change


def test_cached_is_idempotent():
    c = jno.precond.amg().cached()
    assert c.cached() is c  # .cached() on an already-cached spec is a no-op


def test_amg_no_longer_self_caches_but_cached_does(monkeypatch):
    """Unification: `jno.precond.amg()` rebuilds the hierarchy each solve (no silent cross-solve
    cache); `.cached()` is the explicit build-once mechanism. Verified without pyamg by counting the
    (monkeypatched) hierarchy build."""
    import jno.utils.solver.amg as amgmod

    builds = []
    monkeypatch.setattr(amgmod, "build_hierarchy", lambda A, **kw: builds.append(1) or "LEVELS")
    monkeypatch.setattr(amgmod, "vcycle_apply", lambda levels, r: r)  # identity apply
    ctx = PrecondContext(_op()[0], None)

    bare = jno.precond.amg()
    materialize_precond(bare, ctx)
    materialize_precond(bare, ctx)
    assert len(builds) == 2  # rebuilt each solve — no implicit self-cache

    builds.clear()
    wrapped = jno.precond.amg().cached()
    materialize_precond(wrapped, ctx)
    materialize_precond(wrapped, ctx)
    assert len(builds) == 1  # .cached() builds once, reuses


def test_cached_forwards_prepare_hook():
    class _WithPrepare:
        def __init__(self):
            self.prepared = False

        def prepare(self, fem):
            self.prepared = True

        def materialize(self, ctx):
            return PrecondApplier(lambda v: v)

    inner = _WithPrepare()
    prepare_precond(jno.precond.cached(inner), fem=object())  # prepare_precond no-ops on fem=None
    assert inner.prepared  # the eager build hook is forwarded to the wrapped spec
