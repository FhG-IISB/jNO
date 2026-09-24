"""Time-invariant per-step pieces of a transient march are evaluated once, and only those.

Oracles: a dense backward-Euler loop driven by the block's OWN forcing at every step (the scheme the
default march implements), a finite-difference gradient, and the un-hoisted march itself.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver import backend_blocks as bb


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(source, steps=6):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.12).domain(time=(0.0, 0.1, steps))
    co = d.variable("interior", split=True)
    x, y, t = co[0], co[1], co[2]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), phi.bind(x=x, y=y, t=t)
    ic = u(ci[0], ci[1]) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [ci[0], ci[1]])
    f = jno.fn(source, [x, y, t])
    return jno.fem([ui.t * vi + (ui.x * vi.x + ui.y * vi.y) - f * vi, u(cb[0], cb[1]) - 0.0, ic])


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _oracle(block):
    """Backward Euler: (M + dt A) w⁺ = M w + dt (c + f(t⁺)), with the block's own f at every step."""
    M, A = _dense(block.M), _dense(block.A)
    c = np.zeros(M.shape[0]) if block.affine_bias is None else np.asarray(block.affine_bias).reshape(-1)
    w, dt = np.asarray(block.state0).reshape(-1), float(block.dt)
    n = max(1, round((float(block.t1) - float(block.t0)) / dt))
    traj = [w]
    for k in range(n):
        f = np.asarray(block.forcing_vector_fn(float(block.t0) + (k + 1) * dt, {})).reshape(-1)
        w = np.linalg.solve(M + dt * A, M @ w + dt * (c + f))
        traj.append(w)
    return np.stack(traj)


def test_static_source_is_hoisted_and_the_march_matches_the_oracle():
    fem = _heat(lambda x, y, t: 5.0 * x * (1 - x) * y * (1 - y))
    block = fem.operator
    hoisted = bb.hoist_time_invariant(block, {}, float(block.t0))
    assert hoisted.forcing_vector_fn is not block.forcing_vector_fn
    # A static source is assembled into `affine_bias`; the per-step forcing re-assembles the whole
    # residual only to return ~0 on top of it -- exactly the work the hoist removes.
    c = np.asarray(block.affine_bias).reshape(-1)
    f = np.asarray(block.forcing_vector_fn(0.0, {}))
    assert np.abs(c + f).max() > 1e-3  # a real load, not the trivial zero
    np.testing.assert_allclose(np.asarray(hoisted.forcing_vector_fn(0.07, {})), f, rtol=0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(fem.solve().fn()), _oracle(block), rtol=1e-7, atol=1e-10)


def test_time_dependent_source_is_not_hoisted_and_still_matches_the_oracle():
    fem = _heat(lambda x, y, t: 20.0 * jnp.sin(10.0 * t) * x * (1 - x) * y * (1 - y))
    block = fem.operator
    assert not bb._independent_of_t(block.forcing_vector_fn, 0.0, {})
    assert bb.hoist_time_invariant(block, {}, 0.0).forcing_vector_fn is block.forcing_vector_fn
    got = np.asarray(fem.solve().fn())
    np.testing.assert_allclose(got, _oracle(block), rtol=1e-7, atol=1e-10)


def test_dead_reads_of_t_hoist_live_ones_do_not_and_gradients_flow_through_the_hoisted_value():
    base = jnp.linspace(1.0, 2.0, 7)

    def dead_t(t, args):
        _unused = jnp.sin(t) * base  # t is read but nothing returned depends on it
        return args["k"] * base

    def live_t(t, args):
        return args["k"] * base * (1.0 + t)

    assert bb._independent_of_t(dead_t, 0.0, {"k": 2.0})
    assert not bb._independent_of_t(live_t, 0.0, {"k": 2.0})

    import dataclasses

    @dataclasses.dataclass
    class Blk:
        forcing_vector_fn: object = None
        operator_fn: object = None

    def loss(k):
        blk = bb.hoist_time_invariant(Blk(forcing_vector_fn=dead_t, operator_fn=live_t), {"k": k}, 0.0)
        assert blk.operator_fn is live_t  # the t-dependent one is left alone
        return jnp.sum(blk.forcing_vector_fn(0.3, {"k": k}) ** 2) + jnp.sum(blk.operator_fn(0.3, {"k": k}))

    k0, eps = 1.7, 1e-6
    fd = (loss(k0 + eps) - loss(k0 - eps)) / (2 * eps)
    np.testing.assert_allclose(float(jax.grad(loss)(k0)), float(fd), rtol=1e-7)


def test_bdf2_march_is_unchanged_by_hoisting(monkeypatch):
    fem = _heat(lambda x, y, t: 5.0 * x * (1 - x) * y * (1 - y), steps=8)
    got = np.asarray(fem.solve(time=jno.solve.bdf2()).fn())
    monkeypatch.setattr(bb, "hoist_time_invariant", lambda block, args, t0: block)
    fem2 = _heat(lambda x, y, t: 5.0 * x * (1 - x) * y * (1 - y), steps=8)
    ref = np.asarray(fem2.solve(time=jno.solve.bdf2()).fn())
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_parametric_operator_is_hoisted_under_grad_and_the_gradient_matches_finite_differences():
    """The optimisation-loop case: a trainable coefficient inside the stiffness. ``operator_fn(t, args)``
    is t-independent, so it is assembled once per march instead of once per step -- under ``jax.grad``,
    with ``args`` traced -- and the gradient is unchanged."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.15).domain(time=(0.0, 0.05, 6))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    k = jno.np.parameter((1,), name="k")
    u0 = jno.np.sin(jnp.pi * ci[0]) * jno.np.sin(jnp.pi * ci[1])
    fem = jno.fem([ui.t * vi + k * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
    block = fem.operator
    (name,) = block.runtime_parameter_exprs
    assert block.operator_fn is not None
    save = bb._block_time_grid(block)

    seen = {}

    def loss(kv):
        args = {name: jnp.asarray([kv])}
        seen["hoisted"] = bb.hoist_time_invariant(block, args, 0.0).operator_fn is not block.operator_fn
        return jnp.sum(bb._default_transient_integrate(block, args, save) ** 2)

    g = float(jax.grad(loss)(0.8))
    assert seen["hoisted"], "the t-independent parametric operator was not hoisted under grad"
    fd = float((loss(0.8 + 1e-6) - loss(0.8 - 1e-6)) / 2e-6)
    np.testing.assert_allclose(g, fd, rtol=1e-5)
