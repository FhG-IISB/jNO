"""A trainable amplitude in a τ-dependent essential value on a load-path march -- ``u(top)[1] - g*tau`` (#103).

Displacement control is how a softening test is driven at all, and recovering its grip amplitude from a
measured response is the inverse problem that wants this. The two held-value mechanisms used to refuse
the combination: the parametric one would hold the value constant in τ (un-ramping the load), the temporal
one would freeze ``g`` at its stored value (un-training it). The march's residual is called with
``(u, args, τ)``, so the ramp is now evaluated at the step's τ WITH the runtime args.

Oracles: the march at ``g`` equals its twin with ``g`` written as a constant (bit for bit on CPU, to
round-off on GPU); the grip holds ``g*τ_k`` on every step; the gradient in ``g`` matches central
differences; and ``jno.core`` recovers ``g`` from the trajectory it produced.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno

G = 0.02
TAU = (0.0, 1.0, 5)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _param(value):
    g = jno.np.parameter((1,), name="g", key=jax.random.PRNGKey(0))
    g.initialize(jax.nn.initializers.constant(value))
    return g


def _plate(g):
    """A plate clamped at the bottom and pulled at the top by ``u_y = g*tau``, whose stiffness hardens with
    the accumulated strain energy ``s`` -- a history the march carries, so a wrong ramp at any one step
    changes every later one."""
    inner, grad = jno.np.inner, jno.np.grad
    d = jno.shape.rect(0, 0, 0.5, 1, size=0.25).domain(tau=TAU)
    u, v = d.fem_symbols(value_shape=(2,))
    s, _ = d.fem_symbols(value_shape=())
    xi, yi, _ = d.variable("interior", split=True)
    ct = d.variable("top", where=lambda x, y: y > 1 - 1e-9, split=True)
    cb = d.variable("bottom", where=lambda x, y: y < 1e-9, split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    fem = jno.fem(
        [
            (1.0 + 50.0 * s.i(-1)) * inner(gu, gv, n_contract=2),
            s.evolves(s.i(-1) + inner(gu, gu, n_contract=2)),
            u(*ct[:2])[1] - g * ct[-1],
            u(*cb[:2])[0] - 0.0,
            u(*cb[:2])[1] - 0.0,
        ]
    )
    return d, fem


def test_the_march_equals_its_constant_amplitude_twin_and_holds_the_grip():
    g = _param(0.0)
    d, fem = _plate(g)
    assert "g" in fem.operator.runtime_parameter_exprs
    ys = np.asarray(fem.solve(g=jnp.array([G])))
    twin = np.asarray(_plate(G)[1].solve())
    assert np.abs(twin).max() > 0.5 * G
    # The same march: bit-identical on CPU, round-off on GPU (the parametric one compiles differently).
    assert np.abs(ys - twin).max() <= 1e-12 * np.abs(twin).max()

    pts = np.asarray(fem.points)
    top = np.flatnonzero(pts[:, 1] > 1 - 1e-9)
    taus = np.linspace(*TAU)
    for k, tau in enumerate(taus):
        assert np.abs(ys[k, 2 * top + 1] - G * tau).max() < 1e-12, f"step {k}: the grip is not at g*tau"


def test_the_gradient_in_the_amplitude_matches_finite_differences():
    _, fem = _plate(_param(0.0))
    node = fem.solve()

    def loss(gv):
        return jnp.sum(node.fn(gv) ** 2)

    grad = float(jax.grad(loss)(jnp.array([G]))[0])
    h = 1e-6
    fd = float((loss(jnp.array([G + h])) - loss(jnp.array([G - h]))) / (2 * h))
    assert abs(grad) > 1e-6
    assert abs(grad - fd) < 1e-6 * abs(fd), f"d/dg {grad:.10e} vs FD {fd:.10e}"


@pytest.mark.slow
def test_jno_core_recovers_the_grip_amplitude():
    truth = jnp.asarray(_plate(G)[1].solve())
    g = _param(0.005)
    d, fem = _plate(g)
    crux = jno.core([(fem.solve() - truth).mse], domain=d)
    g.optimizer(optax.adam(2e-3))
    crux.solve(300)
    rec = float(np.asarray(crux.eval([g])).reshape(-1)[0])
    assert abs(rec - G) < 1e-3 * G, f"recovered g = {rec:.6g}, truth {G}"
