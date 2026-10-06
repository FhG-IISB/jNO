"""``jno.precond.cached(spec)`` keeps only a setup built from CONCRETE values.

A cache that first builds inside a trace -- the per-step Newton solve of a nonlinear march (a scan body), a
solve under ``jax.jit`` or ``jax.grad`` -- gets an applier that closes over that trace's intermediate
values. It used to keep it, and hand it to the next trace: ``UnexpectedTracerError`` on the FIRST solve of a
nonlinear march, on the second steady ``newton(direct=True)`` solve, and on a second jit trace or a grad.

Oracles: every solve equals the same solve without the cache (a preconditioner changes the route, never
the answer); the gradient equals the uncached gradient; a setup built from concrete values is still kept.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


E = 1e-9


def _wall(d):
    d.tag("wall", lambda x, y: (x < E) | (x > 1 - E) | (y < E) | (y > 1 - E))


def _heat_march():
    """``u_t = div((1 + u²) grad u) + 1``: a nonlinear march, so each step's Newton solve runs in the scan."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain(time=(0.0, 0.2, 5))
    _wall(d)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, ti = d.variable("interior", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ci = d.variable("initial", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    return jno.fem([ub.t * vb + (1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - vb, u(xw, yw) - 0.0, u(*ci) - ic])


def _steady_nonlinear():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain()
    _wall(d)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, *_ = d.variable("interior", split=True)
    xw, yw, *_ = d.variable("wall", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return jno.fem([(1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - 10 * vb, u(xw, yw) - 0.0])


def _eager(out):
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


@pytest.mark.parametrize("refresh", [False, True], ids=["frozen", "on_pattern_change"])
def test_a_nonlinear_march_with_a_cached_preconditioner(refresh):
    # Compared at 1e-12, below the march Newton's own tolerance: both marches solve tightly.
    tight = jno.solve.newton(rtol=1e-13, atol=1e-15)
    ref = _eager(_heat_march().solve(nonlinear=tight))
    spec = jno.precond.cached(jno.precond.jacobi(), refresh=refresh)
    for _ in range(2):  # the second march must not meet the first one's tracers either
        got = _eager(_heat_march().solve(precond=spec, nonlinear=tight))
        assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()
    assert spec._applier is None, "a setup built inside the march's trace must not be kept"


def test_repeated_steady_direct_newton_solves_with_a_cached_preconditioner():
    nl = jno.solve.newton(direct=True)
    ref = _eager(_steady_nonlinear().solve(nonlinear=nl))
    spec = jno.precond.cached(jno.precond.jacobi())
    for _ in range(3):
        got = _eager(_steady_nonlinear().solve(nonlinear=jno.solve.newton(direct=True), precond=spec))
        assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max()


def _parametric_poisson():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain()
    _wall(d)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, *_ = d.variable("interior", split=True)
    xw, yw, *_ = d.variable("wall", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = jno.np.parameter((1,), name="k")
    return jno.fem([k * (ub.x * vb.x + ub.y * vb.y) - vb, u(xw, yw) - 0.0])


def test_one_cached_spec_across_jit_traces_and_grad():
    fem = _parametric_poisson()
    ref_node = fem.solve(linear=jno.solve.cg())
    spec = jno.precond.cached(jno.precond.jacobi())
    node = fem.solve(linear=jno.solve.cg(), precond=spec)
    k = jnp.array([2.0])
    ref = np.asarray(ref_node.fn(k))
    for f in (jax.jit(lambda kk: node.fn(kk)), jax.jit(lambda kk: 1.0 * node.fn(kk))):  # two distinct traces
        assert np.abs(np.asarray(f(k)) - ref).max() <= 1e-10 * np.abs(ref).max()
    g = float(jax.grad(lambda kk: node.fn(kk).sum())(k)[0])
    g_ref = float(jax.grad(lambda kk: ref_node.fn(kk).sum())(k)[0])
    assert abs(g) > 1e-3
    assert abs(g - g_ref) <= 1e-8 * abs(g_ref)


def test_a_setup_built_from_concrete_values_is_still_kept():
    spec = jno.precond.cached(jno.precond.jacobi())
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain()
    _wall(d)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, *_ = d.variable("interior", split=True)
    xw, yw, *_ = d.variable("wall", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ub.x * vb.x + ub.y * vb.y - vb, u(xw, yw) - 0.0])
    fem.solve(linear=jno.solve.cg(), precond=spec)
    built = spec._applier
    assert built is not None
    fem.solve(linear=jno.solve.cg(), precond=spec)
    assert spec._applier is built, "a frozen cache rebuilt on the second solve"
