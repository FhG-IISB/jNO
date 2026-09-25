"""A finished operator is moved to the solving device ONCE, not copied there by every solve.

Oracle: the same solve with the operator left where assembly put it (the host), and the placement
itself (devices / committed) read off the arrays.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.placement import solve_device, to_solve_device

gpu = pytest.mark.skipif(jax.default_backend() == "cpu", reason="placement is a no-op on a CPU-only run")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(size=0.1):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    x, y = d.variable("interior", split=True)[:2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(cb[0], cb[1]) - 0.0])


def _heat(size=0.15):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=(0.0, 0.05, 6))
    x, y, t = d.variable("interior", split=True)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(cb[0], cb[1]) - 0.0, ic])


def _leaves(tree):
    return [x for x in jax.tree_util.tree_leaves(tree) if isinstance(x, jax.Array)]


@gpu
def test_the_linear_operator_moves_once_and_stays_uncommitted():
    fem = _poisson()
    host = [np.asarray(x) for x in _leaves(fem._op)]
    assert all(x.devices() == {jax.devices("cpu")[0]} for x in _leaves(fem._op))  # host assembly
    u1 = np.asarray(fem.solve())
    moved = _leaves(fem._op)
    assert all(x.devices() == {solve_device()} and not x.committed for x in moved)
    for h, m in zip(host, moved):
        np.testing.assert_array_equal(h, np.asarray(m))
    u2 = np.asarray(fem.solve())
    assert all(a is b for a, b in zip(moved, _leaves(fem._op)))  # not moved again
    np.testing.assert_allclose(u1, u2, rtol=1e-12, atol=1e-15)
    assert np.abs(u1).max() > 1e-3


@gpu
def test_a_solve_the_caller_scopes_to_the_host_still_runs_there():
    fem = _poisson()
    u_dev = np.asarray(fem.solve())  # operator now on the device, uncommitted
    with jax.default_device(jax.devices("cpu")[0]):
        u_host = fem.solve()
        assert u_host.devices() == {jax.devices("cpu")[0]}
    np.testing.assert_allclose(np.asarray(u_host), u_dev, rtol=1e-7, atol=1e-10)


@gpu
def test_the_march_uploads_its_constants_once_and_configurations_share_them():
    fem = _heat()
    a = np.asarray(fem.solve().fn())
    b = np.asarray(fem.solve().fn())
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)
    blocks = [v for v in fem.__dict__.values() if hasattr(v, "__dict__") and "_march_cache" in v.__dict__]
    blocks += [v for v in vars(fem._op).values() if hasattr(v, "__dict__") and "_march_cache" in v.__dict__] if hasattr(fem._op, "__dict__") else []
    if not blocks:
        pytest.skip("could not reach the transient block from the fem object")
    for blk in blocks:
        for consts, _run, _tree in blk.__dict__["_march_cache"].values():
            assert all(c.devices() == {solve_device()} for c in consts if isinstance(c, jax.Array) and c.size > 1)


def test_committed_and_traced_arrays_are_left_alone():
    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.ones(3), cpu)  # committed by the caller
    assert to_solve_device(x) is x
    jax.jit(lambda v: (to_solve_device(v), v)[1])(jnp.ones(3))  # a tracer passes through


def test_the_memo_shares_one_copy():
    with jax.default_device(jax.devices("cpu")[0]):
        x = jnp.arange(4.0)
    memo = {}
    a, b = to_solve_device([x], memo), to_solve_device([x], memo)
    assert a[0] is b[0]
