"""One solver's scalar readout as another FEM's coefficient: resolved once per solve and handed to the
kernel (a "global scalar"), not called at every quadrature point. Here a FEM integral on one domain is
the source of a FEM on another, and a shape parameter of the first reaches the second. Oracle: central
differences of the chained solve, and the parametric chain at the reference equals numbers.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno._fem import _walk
from jno.trace_evaluator import TraceEvaluator

jax.config.update("jax_enable_x64", True)


def _poisson(d, f):
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0]), ui


def _chain(w):
    fem1, u1 = _poisson(jno.shape.rect(0.0, 0.0, w, 1.0, size=0.15).domain(), 1.0)
    q = (1.0 * u1).integrate(fem1)  # one number per solve
    fem2, _u2 = _poisson(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.15).domain(), q)
    return fem2.solve()


def test_an_integral_of_one_fem_is_the_source_of_another():
    w = jno.np.parameter((1,), name="w")
    w.dtype(jnp.float64)
    w.initialize(jax.nn.initializers.constant(2.0))
    node = _chain(w)
    (mc,) = {m.model._parameter_name: m for m in _walk(node) if type(m).__name__ == "ModelCall"}.values()

    def umax(wv):
        table = {mc.model.layer_id: eqx.tree_at(lambda q: q.value, mc.model.module, jnp.reshape(wv, (1,)))}
        return jnp.max(jnp.asarray(TraceEvaluator(table).evaluate(node, context={})).reshape(-1))

    numbers = _chain(2.0).eval()  # still a node: its source is the first solve's integral
    assert float(umax(2.0)) == pytest.approx(float(np.max(np.asarray(numbers).reshape(-1))), rel=1e-7)
    g = float(jax.grad(umax)(2.0))
    h = 1e-5
    fd = (float(umax(2.0 + h)) - float(umax(2.0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-6)
    assert g > 0  # a wider first domain holds more of its solution, so a stronger source in the second
