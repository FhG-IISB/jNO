"""One geometry parameter drives two solvers on one domain: a PEEC network's loss heats a FEM problem,
and the parameter moves both -- the network's faces and the FEM mesh (the shape's stretch).

A PEEC readout is one value per solve; used as a FEM coefficient it is resolved once and handed to
the kernel (a global scalar), not evaluated per quadrature point. Oracle: central differences of
the chained solve, and the parametric chain at the reference equals the same chain built with numbers.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno._fem import _walk
from jno.trace_evaluator import TraceEvaluator


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64; save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


mm, CU, P = 1e-3, 5.8e7, 1.0e-3


def _chain(y0):
    a = jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=P).name("t1").attach(sigma=CU, k=390.0)
    b = jno.shape.box(0, y0, 0, 12 * mm, 7 * mm, 0.5 * mm, size=P).name("t2").attach(sigma=CU, k=390.0)
    d = (a + b).domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 11.1 * mm)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    em = jno.peec([v(*at("A")) - v(*at("B")) - 0.01], freq=1e3, grid=jno.peec.faces(pitch=P))
    for r, q in em.solve(matrix_free=False).dissipation().items():
        d.attach(r, q=q)
    T, s = d.fem_symbols()
    x = d.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=x[0], y=x[1], z=x[2]), s.bind(x=x[0], y=x[1], z=x[2])
    return jno.fem([d.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - d.q * si, T(*at("B")) - 0.0]).solve()


def test_a_geometry_parameter_drives_a_peec_heated_fem_solve():
    y0 = jno.np.parameter((1,), name="y0")
    y0.dtype(jnp.float64)
    y0.initialize(jax.nn.initializers.constant(5 * mm))
    node = _chain(y0)
    (mc,) = {m.model._parameter_name: m for m in _walk(node) if type(m).__name__ == "ModelCall"}.values()

    def tmax(yv):
        table = {mc.model.layer_id: eqx.tree_at(lambda q: q.value, mc.model.module, jnp.reshape(yv, (1,)))}
        return jnp.max(jnp.asarray(TraceEvaluator(table).evaluate(node, context={})).reshape(-1))

    numbers = float(np.max(np.asarray(_chain(5 * mm)).reshape(-1)))
    # the numbers-only FEM solves iteratively (BiCGStab, tol 1e-8), the parametric one by sparse LU
    assert float(tmax(5 * mm)) == pytest.approx(numbers, rel=1e-7)  # measured 7.3e-09
    g = float(jax.grad(tmax)(5 * mm))
    h = 1e-4 * mm
    fd = (float(tmax(5 * mm + h)) - float(tmax(5 * mm - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-6)  # measured 2.4e-10


def _coupled(x1):
    """Two conductors meeting at x = x1 (one expression for both faces), sigma(T) from the FEM field,
    heated by their own loss: the two-way fixed point `heat.solve()` / `em.solve()` share."""
    A = jno.shape.box(0, 0, 1e-3, x1, 4e-3, 1.5e-3, size=1e-3).name("A")
    B = jno.shape.box(x1, 0, 1e-3, 16e-3, 4e-3, 1.5e-3, size=1e-3).name("B")
    d = (A + B).domain()
    d.tag("P", lambda x, y, z: x < 1.1e-3)
    d.tag("N", lambda x, y, z: x > 14.9e-3)
    T, s = d.fem_symbols()
    d.attach(sigma=CU / (1 + 0.004 * (T - 300.0)), k=400.0)
    i, v = d.peec_symbols()
    pad = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    # the network's solve settings live on it, so the coupled solve uses them too
    em = jno.peec([v(*pad("P")) - v(*pad("N")) - 0.05], freq=1e3, grid=jno.peec.faces(pitch=1e-3), matrix_free=False)
    x = d.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=x[0], y=x[1], z=x[2]), s.bind(x=x[0], y=x[1], z=x[2])
    heat = jno.fem([d.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - em.loss * si, T(*pad("N")) - 300.0])
    return heat.solve(), em


def test_a_geometry_parameter_through_the_two_way_electrothermal_fixed_point():
    x1 = jno.np.parameter((1,), name="x1")
    x1.dtype(jnp.float64)
    x1.initialize(jax.nn.initializers.constant(6e-3))
    node, _em = _coupled(x1)
    (mc,) = {m.model._parameter_name: m for m in _walk(node) if type(m).__name__ == "ModelCall"}.values()

    def tmax(wv):
        table = {mc.model.layer_id: eqx.tree_at(lambda q: q.value, mc.model.module, jnp.reshape(wv, (1,)))}
        return jnp.max(jnp.asarray(TraceEvaluator(table).evaluate(node, context={})).reshape(-1))

    assert float(tmax(6e-3)) == pytest.approx(float(np.max(np.asarray(_coupled(6e-3)[0]))), rel=1e-10)  # measured 2e-14
    g = float(jax.grad(tmax)(6e-3))
    h = 1e-7
    fd = (float(tmax(6e-3 + h)) - float(tmax(6e-3 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-6)  # measured 8.8e-09


def test_the_network_at_the_converged_field_is_a_trace_node_too():
    """`em.solve().R` of the parametric coupled pair: the network's resistance at the converged field, equal
    to the eager `ElectroThermalSolution.em.R` at the reference, and differentiable in the geometry."""
    x1 = jno.np.parameter((1,), name="x1")
    x1.dtype(jnp.float64)
    x1.initialize(jax.nn.initializers.constant(6e-3))
    R = _coupled(x1)[1].solve().R  # the network at the converged field, as a trace node
    (mc,) = {m.model._parameter_name: m for m in _walk(R) if type(m).__name__ == "ModelCall"}.values()

    def r(wv):
        table = {mc.model.layer_id: eqx.tree_at(lambda q: q.value, mc.model.module, jnp.reshape(wv, (1,)))}
        return jnp.real(jnp.asarray(TraceEvaluator(table).evaluate(R, context={}))).reshape(())

    assert float(r(6e-3)) == pytest.approx(float(np.real(_coupled(6e-3)[1].solve().R)), rel=1e-10)
    g = float(jax.grad(r)(6e-3))
    h = 1e-7
    fd = (float(r(6e-3 + h)) - float(r(6e-3 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-5)
