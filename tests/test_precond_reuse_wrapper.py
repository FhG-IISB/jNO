"""``reuse=`` -- keep any preconditioner's setup across solves (backend-agnostic). A spec built by a
``jno.precond`` constructor takes ``reuse=`` directly; a user-written spec goes through the same wrapper,
``jno.precond._reusing(spec, reuse)``."""

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import pytest

import jno
from jno.precond import _reusing, _Spec  # _Spec: the base a user-written spec inherits
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


def test_reuse_builds_once_and_reuses_applier():
    inner = _Counting()
    c = _reusing(inner, True)
    op, _ = _op()
    m1 = materialize_precond(c, PrecondContext(op, None))
    m2 = materialize_precond(c, PrecondContext(op, None))
    assert inner.n == 1  # setup built exactly once across two solves
    assert m1 is m2  # same applier object reused


def test_reuse_true_keeps_it_even_when_operator_changes():
    inner = _Counting()
    c = _reusing(inner, True)  # frozen
    materialize_precond(c, PrecondContext(_op(6)[0], None))
    materialize_precond(c, PrecondContext(_op(8)[0], None))  # different operator, still reuses
    assert inner.n == 1


def test_a_kept_preconditioner_matches_a_rebuilt_one():
    """The cached preconditioner produces the same converged solution as the bare one.

    (Calling a solver directly takes a materialised applier as ``M``; ``fem.solve(precond=...)``
    does that materialisation from the spec internally.)"""
    op, dense = _op(48)
    b = jnp.ones(op.shape[0])
    ctx = PrecondContext(op, None)
    x_ref = jnp.linalg.solve(dense, b)
    solve = jno.solve.cg(tol=1e-12)
    x_bare = solve(op, b, M=materialize_precond(jno.precond.jacobi(), ctx))
    x_cached = solve(op, b, M=materialize_precond(jno.precond.jacobi(reuse=True), ctx))
    assert jnp.max(jnp.abs(x_cached - x_ref)) < 1e-8
    assert jnp.max(jnp.abs(x_cached - x_bare)) < 1e-10


def test_amg_rebuilds_per_solve_unless_reuse_is_set(monkeypatch):
    """Unification: `jno.precond.amg()` rebuilds the hierarchy each solve (no silent cross-solve
    cache); `reuse=True` is the explicit build-once mechanism. Verified without pyamg by counting the
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
    wrapped = jno.precond.amg(reuse=True)
    materialize_precond(wrapped, ctx)
    materialize_precond(wrapped, ctx)
    assert len(builds) == 1  # reuse=True builds once, reuses


def test_reuse_forwards_prepare_hook():
    class _WithPrepare:
        def __init__(self):
            self.prepared = False

        def prepare(self, fem):
            self.prepared = True

        def materialize(self, ctx):
            return PrecondApplier(lambda v: v)

    inner = _WithPrepare()
    prepare_precond(_reusing(inner, True), fem=object())  # prepare_precond no-ops on fem=None
    assert inner.prepared  # the eager build hook is forwarded to the wrapped spec


def test_frozen_cache_reaches_the_compiled_path_once_built():
    """`reuse=True` is the recipe for many-iteration work -- a transient run or a Newton loop, where
    one preconditioner serves every step. It inherited `traceable = False`, so it stayed on the eager
    path on exactly the workloads with the most preconditioner applications to dispatch.

    A frozen cache that has already built does no setup at all -- `materialize` hands back the stored
    applier -- so it can be materialised inside a trace. The first solve still runs eager (it has to,
    to build); every solve after it can compile. A cadence (`reuse=k`) is excluded: it rebuilds through the
    inner spec, which for amg/ams/form is host-side work.
    """
    from jno.utils.solver.solver_api import PrecondContext, _compilable, materialize_precond

    op, _ = _op()
    lin = jno.solve.cg(tol=1e-10)

    spec = jno.precond.jacobi(reuse=True)
    assert spec.traceable is False, "an unbuilt cache has nothing stored yet"
    assert spec.key is None
    assert not _compilable(lin, spec)

    materialize_precond(spec, PrecondContext(op, None))  # the first (eager) solve builds it
    assert spec.traceable is True, "a frozen, built cache materialises to a stored applier"
    assert spec.key is not None
    assert _compilable(lin, spec), "a built frozen cache must be able to compile -- that is the point"

    # a cadence rebuilds through the inner spec (host-side for amg/ams/form), so it never claims traceability
    r = jno.precond.jacobi(reuse=2)
    materialize_precond(r, PrecondContext(op, None))
    assert r.traceable is False, "reuse=2 must stay eager"
    assert not _compilable(lin, r)


def test_compiled_kept_precond_gives_the_same_answer():
    """Compiling the solve around a frozen cache must not change what it returns."""
    import numpy as np
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.15)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)

    ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    spec = jno.precond.jacobi(reuse=True)
    lin = jno.solve.cg(tol=1e-12, maxiter=5000)
    first = np.asarray(fem.solve(linear=lin, precond=spec))  # eager: builds the cache
    second = np.asarray(fem.solve(linear=lin, precond=spec))  # compiled: reuses it
    assert np.abs(first - ref).max() < 1e-8
    assert np.abs(second - ref).max() < 1e-8
    assert np.abs(second - first).max() < 1e-10


# --- the wrapper must not hide what it wraps -------------------------------------------------


def test_reuse_forwards_complex_native():
    """``complex_native`` selects HOW a complex system is solved, so a wrapper that hides it changes
    the answer rather than the speed.

    Declaring it routes the solve to the genuinely complex ``n``-sized operator; without it the
    problem falls through to the fused real-equivalent ``2n`` block, where two things break at once:
    ``fem.blocks`` describes the ``n`` layout so every slice covers the wrong half, and AMS is handed
    the skew-dominated block its own docs record diverging to 1e+20 on. ``_Cached`` defined only its
    own behaviour, so a kept ``ams()`` reported ``False`` and silently took that path -- making
    ``reuse=`` and complex-native mutually exclusive, which is precisely backwards, since a
    complex block preconditioner is the one whose setup most wants reusing.
    """
    assert jno.precond.ams().complex_native is True
    assert getattr(jno.precond.ams(reuse=True), "complex_native", False) is True


def test_reuse_forwards_complex_ok():
    """``complex_ok`` says a spec can be applied to a complex operator WITHOUT reformulation. AMS
    reads it to decide whether its auxiliary blocks get split, so hiding it makes AMS silently revert
    to the real-equivalent form that AMG is documented to diverge on."""
    assert jno.precond.amg().complex_ok is True
    assert getattr(jno.precond.amg(reuse=True), "complex_ok", False) is True


def test_a_block_composition_sees_through_reuse():
    """The case that matters: the flag has to survive BOTH wrappers, since a block composition
    inherits it from its children."""
    tri = jno.precond.triangular(("u", jno.precond.ams(reuse=True)), ("p", jno.precond.jacobi()))
    assert tri.complex_native, "triangular lost complex_native through the cached child"


def test_keeping_a_plain_spec_still_reports_no_complex_support():
    """Forwarding must report what the inner spec actually is, not assert capability."""
    assert not getattr(jno.precond.jacobi(reuse=True), "complex_native", False)
