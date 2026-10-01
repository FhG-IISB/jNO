"""``jno.precond.chebyshev()`` without explicit bounds works inside a trace (a Newton linearisation).

The bounds are measured by Lanczos, and an interval that broke down (non-finite, non-positive, zero width)
falls back to power iteration. That choice was a Python ``if`` on the Ritz values -- traced inside a Newton
step -- so every steady Newton solve with a Chebyshev preconditioner raised TracerBoolConversionError,
although the solver docs list chebyshev among the preconditioners that work on the nonlinear path. The
choice is now a ``lax.cond``.

Oracles: the preconditioned Newton solve equals the unpreconditioned one; the traced bounds equal the
eager ones on both branches (a usable Lanczos interval, and a degenerate one that takes the fallback).
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


def _steady_nonlinear():
    d = jno.shape.rect(0, 0, 1, 1).structured(n=6).domain()
    e = 1e-9
    d.tag("wall", lambda x, y: (x < e) | (x > 1 - e) | (y < e) | (y > 1 - e))
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, *_ = d.variable("interior", split=True)
    xw, yw, *_ = d.variable("wall", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return jno.fem([(1 + ub * ub) * (ub.x * vb.x + ub.y * vb.y) - 10 * vb, u(xw, yw) - 0.0])


def _eager(out):
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


def test_a_newton_solve_with_an_estimated_chebyshev_preconditioner():
    ref = _eager(_steady_nonlinear().solve(nonlinear=jno.solve.newton()))
    got = _eager(_steady_nonlinear().solve(nonlinear=jno.solve.newton(), precond=jno.precond.chebyshev()))
    assert np.abs(ref).max() > 0.1
    assert np.abs(got - ref).max() <= 1e-10 * np.abs(ref).max()


@pytest.mark.parametrize(
    "diag",
    [np.linspace(1.0, 50.0, 40), np.ones(40)],
    ids=["lanczos_interval", "degenerate_interval_falls_back"],
)
def test_traced_bounds_equal_the_eager_ones(diag):
    pytest.importorskip("matfree", reason="the Lanczos branch needs matfree; without it both paths are power iteration")
    from jno.utils.solver.krylov import spectrum_bounds

    d = jnp.asarray(diag)

    def bounds(scale):
        return spectrum_bounds(lambda v: scale * d * v, d.shape[0], dtype=d.dtype, iters=20)

    eager = [float(x) for x in bounds(1.0)]
    traced = [float(x) for x in jax.jit(bounds)(jnp.asarray(1.0))]
    assert np.allclose(traced, eager, rtol=1e-12)
    if diag.min() == diag.max():  # the fallback: lmin = lmax / 30 against the power-iteration lmax
        assert np.isclose(traced[0], traced[1] / 30.0, rtol=1e-12)
    else:  # Lanczos brackets the true spectrum after the safety factor
        assert traced[0] <= diag.min() * 1.0001 and traced[1] >= diag.max()
