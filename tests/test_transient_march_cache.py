"""The eager transient march (``fem.solve().fn()``) is traced once and reused, and stays correct.

Oracle: a dense backward-Euler loop on the block's own ``M``, ``A``, ``state0`` and ``dt`` -- the
scheme the default march implements -- as in ``tests/test_fem_3d.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver import backend_blocks


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(steps=8):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.12).domain(time=(0.0, 0.05, steps))
    co = d.variable("interior", split=True)
    x, y, t = co[0], co[1], co[2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    return jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y), u(cb[0], cb[1]) - 0.0, ic])


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _oracle(block, scale_A=1.0):
    M, A = _dense(block.M), scale_A * _dense(block.A)
    w, dt = np.asarray(block.state0).reshape(-1), float(block.dt)
    n = max(1, round((float(block.t1) - float(block.t0)) / dt))
    traj = [w]
    for _ in range(n):
        w = np.linalg.solve(M + dt * A, M @ w)
        traj.append(w)
    return np.stack(traj)


def _count_traces(block):
    calls = {"n": 0}
    orig = type(block).step

    def counting(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    return calls, counting


def test_repeated_eager_evaluations_trace_once_and_match_the_oracle(monkeypatch):
    fem = _heat()
    block = fem.operator
    calls, counting = _count_traces(block)
    monkeypatch.setattr(type(block), "step", counting)

    ref = _oracle(block)
    first = np.asarray(fem.solve().fn())
    traced_first = calls["n"]
    second = np.asarray(fem.solve().fn())  # a NEW node from a new solve(): same block, same config
    node = fem.solve()
    third, fourth = np.asarray(node.fn()), np.asarray(node.fn())  # the SAME node twice

    assert traced_first >= 1
    assert calls["n"] == traced_first, "a warm evaluation re-traced block.step"
    for traj in (first, second, third, fourth):
        np.testing.assert_allclose(traj, ref, rtol=1e-7, atol=1e-9)
    # not bitwise: GPU scatter-add is not reproducible run-to-run (~1e-16)
    np.testing.assert_allclose(first, fourth, rtol=0, atol=1e-12)


def test_reassigning_a_block_field_retraces_instead_of_reusing_stale_constants():
    fem = _heat()
    block = fem.operator
    np.asarray(fem.solve().fn())  # populate the cache with the original A
    block.A = 3.0 * block.A  # jNO reassigns block fields (contact / coupled residuals); must not go stale
    got = np.asarray(fem.solve().fn())
    np.testing.assert_allclose(got, _oracle(block), rtol=1e-7, atol=1e-9)
    assert not np.allclose(got, _oracle(block, scale_A=1.0 / 3.0), rtol=1e-3)


def test_traced_inputs_bypass_the_cache_and_still_differentiate():
    """Under jit/grad with a TRACED march input (here the initial state) the march runs inline -- the
    enclosing jit owns caching -- and no jaxpr closed over tracers is stored on the block."""
    fem = _heat(steps=4)
    block = fem.operator
    block.__dict__.pop("_march_cache", None)
    grid = backend_blocks._block_time_grid(block)
    s0 = jnp.asarray(block.state0).reshape(-1)
    calls = {"n": 0}

    def march(s0, grid_ts, args):
        calls["n"] += 1
        return jnp.cumsum(jnp.outer(grid_ts[1:], s0), axis=0)  # linear in s0

    def total(s0):
        return backend_blocks._cached_march(block, ("probe",), march, s0, grid, {}).sum()

    g = jax.grad(total)(s0)
    np.testing.assert_allclose(np.asarray(g), np.full(s0.shape, float(jnp.cumsum(grid[1:]).sum())), rtol=1e-12)
    assert "_march_cache" not in block.__dict__ or not block.__dict__["_march_cache"]
    # and the same call with CONCRETE inputs caches: traced once over two calls
    calls["n"] = 0
    a = backend_blocks._cached_march(block, ("probe",), march, s0, grid, {})
    b = backend_blocks._cached_march(block, ("probe",), march, s0, grid, {})
    assert calls["n"] == 1 and len(block.__dict__["_march_cache"]) == 1
    np.testing.assert_allclose(np.asarray(a), np.asarray(march(s0, grid, {})), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)


def test_a_changed_jno_setup_setting_is_not_served_from_a_stale_march_cache():
    """`matvec_format` / `lu_stack` are baked into a march when it is traced; their setters clear JAX's
    caches but not this block-level one, so they must be part of its key."""
    from jno.utils.solver import matvec_format

    fem = _heat(steps=4)
    block = fem.operator
    try:
        matvec_format.set_matvec_format("coo")
        a = np.asarray(fem.solve().fn())
        matvec_format.set_matvec_format("csr")
        b = np.asarray(fem.solve().fn())
        assert len(block.__dict__["_march_cache"]) == 2  # re-traced for the new setting, not reused
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    finally:
        matvec_format.set_matvec_format("auto")


@pytest.mark.parametrize(
    "kw",
    [
        dict(time="sdirk"),
        dict(time="bdf2"),
        dict(time="rosenbrock"),
        dict(nonlinear="newton"),
    ],
)
def test_a_march_with_time_or_solver_slots_is_traced_once(kw):
    """fem.solve composes fresh per-step solvers on every call; the march cache keyed on those objects missed
    every time, re-tracing a warm nonlinear march on every call (~1.1 s where the march takes 30 ms)."""
    import jax as _jax

    import jno as _jno

    d = _jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain(time=(0.0, 0.1, 6))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = _jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y + ui**3 * vi, u(cb[0], cb[1]) - 0.0, u(ci[0], ci[1]) - 1.0])
    blk = fem.operator
    count, orig = [0], blk.residual

    def counted(w, *a, **k):
        if isinstance(w, _jax.core.Tracer):
            count[0] += 1
        return orig(w, *a, **k)

    object.__setattr__(blk, "residual", counted)
    specs = {
        "time": {"sdirk": _jno.solve.sdirk, "bdf2": _jno.solve.bdf2, "rosenbrock": _jno.solve.rosenbrock},
        "nonlinear": {"newton": _jno.solve.newton},
    }
    make = lambda: {k: specs[k][v]() for k, v in kw.items()}  # noqa: E731  fresh spec objects every call
    first = np.asarray(fem.solve(**make()).fn())
    traced = count[0]
    again = np.asarray(fem.solve(**make()).fn())
    assert traced > 0 and count[0] == traced, "the warm march was traced again"
    np.testing.assert_allclose(again, first, rtol=1e-12, atol=1e-14)
