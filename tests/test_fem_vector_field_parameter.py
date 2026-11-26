"""A VECTOR nodal field parameter in a weak form: ``f = jno.np.parameter(ft)`` with ``ft`` a test symbol of
``value_shape=(2,)`` holds one value per vertex and component, and the kernel must read both.

It used to gather the cell's values as ``flat[node ids]`` from the interleaved ``(n_vertices * 2,)`` vector -- node j
read component j % 2 of node j // 2 -- and interpolate that one scalar into every component: ``f = (1, 0)`` gave
``u_y = u_x``, with no error.

Oracles: the same nodal values held by a frozen field (a separate gather); ``f = (1, 0)`` leaves the second
component zero; the gradient in ``f`` against a central difference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno
from jno.trace import Placeholder

inner, vec = jno.np.inner, jno.np.vector
LU = jno.solve.lu()


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _problem(load):
    """-lap u + u = f on the unit square, u = 0 on the boundary, u and f vectors; ``load(d, g, gt, x, y)`` gives f"""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    grad = lambda w: jno.np.grad(w, [x, y])  # noqa: E731
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"))
    _g, gt = d.fem_symbols(value_shape=(2,), names=("g_", "gt"))
    U, V = u.bind(x=x, y=y), v.bind(x=x, y=y)
    f = load(d, _g, gt, x, y)
    form = jno.fem(
        [
            inner(grad(u), grad(v), n_contract=2) + inner(U, V) - inner(f, V),
            u(xb, yb) - vec(0.0 * xb, 0.0 * xb),
        ]
    )
    return d, f, form.solve(linear=LU)


def _field_parameter(values):
    def load(d, _g, gt, x, y):
        f = jno.np.parameter(gt, name="f")
        f.dtype(jnp.float64)
        f.initialize(lambda *_a, **_k: jnp.asarray(values(np.asarray(d.mesh.points)[:, :2])))
        return f

    return load


def _eval(*nodes):
    if len(nodes) == 1 and not isinstance(nodes[0], Placeholder):  # a form with no runtime input solves eagerly
        return np.asarray(nodes[0])
    out = jno.core([nodes[0]]).eval(list(nodes))
    return [np.asarray(o) for o in out] if len(nodes) > 1 else np.asarray(out)


def _smooth(p):
    return np.stack([np.sin(3.0 * p[:, 0]) + p[:, 1], 0.5 - p[:, 0] * p[:, 1]], axis=-1)


def test_a_load_along_x_leaves_the_y_component_zero():
    _d, _f, sol = _problem(_field_parameter(lambda p: np.stack([np.ones(len(p)), np.zeros(len(p))], -1)))
    u = _eval(sol).reshape(-1, 2)
    assert np.abs(u[:, 0]).max() > 1e-2, "the oracle needs a load that moves u_x"
    np.testing.assert_allclose(u[:, 1], 0.0, rtol=0, atol=1e-14)


def test_the_parameter_equals_the_same_values_frozen():
    d, _f, got = _problem(_field_parameter(_smooth))
    pts = np.asarray(d.mesh.points)[:, :2]
    _d2, _f2, want = _problem(lambda d2, g, gt, x, y: g.bind(x=x, y=y).freeze(_smooth(pts).reshape(-1)))
    np.testing.assert_allclose(_eval(got).reshape(-1), _eval(want).reshape(-1), rtol=0, atol=1e-13)


def test_the_gradient_in_the_parameter_matches_a_central_difference():
    """d(sum of u * u)/df along a fixed direction: one optimiser step of plain SGD reads the AD gradient"""

    def readout(scale, lr=None):
        d, f, sol = _problem(_field_parameter(lambda p: scale * _smooth(p)))
        loss = jno.np.sum(sol * sol)
        if lr is None:
            return float(np.ravel(_eval(loss))[0])
        f.optimizer(optax.sgd(lr))
        crux = jno.core([loss])
        f0 = np.asarray(crux.eval([f])).reshape(-1)
        crux.solve(1)
        g = (f0 - np.asarray(crux.eval([f])).reshape(-1)) / lr
        return float(g @ _smooth(np.asarray(d.mesh.points)[:, :2]).reshape(-1))

    s0, h = 1.0, 1e-5
    ad = readout(s0, lr=1e-6)
    fd = (readout(s0 + h) - readout(s0 - h)) / (2 * h)
    assert abs(fd) > 1e-4, "the oracle needs a readout that moves with f"
    np.testing.assert_allclose(ad, fd, rtol=1e-6)
