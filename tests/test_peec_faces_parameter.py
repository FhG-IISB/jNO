"""A solid's faces as `jno.core` design variables: `.attach(faces=(x0, y0, z0, x1, y1, z1))` with
entries in `jno.np.parameter`s, the solid counterpart of a wire's `.attach(path=..., radius=...)`.

The oracle is `BuiltPEEC.solve(geometry=)`, whose gradient `test_peec_traced_grid` pins against
networks rebuilt at each design: the parametric readout must give the same value and the same
derivative, and its own derivative must match central differences of itself. Also: the attachments
that cannot mean anything are refused, and a property `jno.peec` does not read is named.
"""

import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

mm, CU = 1e-3, 5.8e7
P = 1.0 * mm
T2 = (0.0, 5 * mm, 0.0, 12 * mm, 7 * mm, 0.5 * mm)  # the second trace's faces at the reference


def _y0(value=T2[1]):
    p = jno.np.parameter((1,), name="y0")
    p.dtype(jnp.float64)
    p.initialize(jax.nn.initializers.constant(value))
    return p


def _domain(t2_props, wire_props=None, t1_props=None):
    a = jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=CU, **(t1_props or {})).name("t1")
    b = jno.shape.box(*T2, size=(P,) * 3).attach(sigma=CU, **t2_props).name("t2")
    sh = a + b
    if wire_props is not None:
        sh = sh + jno.shape.line(
            [
                (6 * mm, 1 * mm, 0.5 * mm),
                (6 * mm, 1 * mm, 1.5 * mm),
                (6 * mm, 6 * mm, 1.5 * mm),
                (6 * mm, 6 * mm, 0.5 * mm),
            ],
            r=0.15 * mm,
            size=P,
        ).attach(sigma=3.8e7, **wire_props).name("w")
    d = sh.domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 11.1 * mm)
    return d


def _network(d, faces=True):
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6, **({"grid": jno.peec.faces(pitch=P)} if faces else {}))


def _value_fn(node):
    """The readout as a function of the parameter's value -- what `jno.core` threads in."""
    import equinox as eqx

    from jno._fem import _walk
    from jno.trace_evaluator import TraceEvaluator

    (mc,) = {m.model._parameter_name: m for m in _walk(node) if type(m).__name__ == "ModelCall"}.values()

    def value(y):
        table = {mc.model.layer_id: eqx.tree_at(lambda q: q.value, mc.model.module, jnp.reshape(y, (1,)))}
        return jnp.real(jnp.asarray(TraceEvaluator(table).evaluate(node, context={}))).reshape(())

    return value


def test_a_parametric_face_is_the_geometry_solve_with_its_gradient():
    """`faces=` with y0 a parameter: the value and dL/dy0 equal `solve(geometry=)` at the same y0
    (the tested path), at the reference and moved off it, and dL/dy0 matches central differences."""
    y0 = _y0()
    net = _network(_domain({"faces": (T2[0], y0, T2[2], T2[3], T2[4], T2[5])}))
    built = net.build()
    value = _value_fn(built.solve(matrix_free=False).L)
    geo = lambda yy: jnp.real(built.solve(sigma={}, matrix_free=False, geometry={"t2": (T2[0], yy, *T2[2:])}).L)
    for yy in (T2[1], T2[1] - 0.4 * mm):
        assert float(value(yy)) == pytest.approx(float(geo(yy)), rel=1e-12)
        g = float(jax.grad(value)(yy))
        assert g == pytest.approx(float(jax.grad(geo)(yy)), rel=1e-10)
        h = 1e-6 * mm
        fd = (float(value(yy + h)) - float(value(yy - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-5)
        assert g > 0, "a wider gap between the traces is a larger loop"


def test_faces_as_one_expression_of_six_values():
    """The other spelling: one expression of shape (6,), as `path=` is one expression of the vertices."""
    y0 = _y0()
    e = np.zeros(6)
    e[1] = 1.0
    f0 = np.asarray(T2) - e * T2[1]
    tup = _value_fn(_network(_domain({"faces": (T2[0], _y0(), *T2[2:])})).build().solve(matrix_free=False).L)
    one = _value_fn(_network(_domain({"faces": y0 * e + f0})).build().solve(matrix_free=False).L)
    yy = T2[1] - 0.3 * mm
    assert float(one(yy)) == pytest.approx(float(tup(yy)), rel=1e-12)


def test_attachments_that_cannot_mean_anything_are_refused():
    y0 = _y0()
    with pytest.raises(ValueError, match="needs 6 values"):
        _network(_domain({"faces": (T2[0], y0, T2[2], T2[3], T2[4])})).build()
    with pytest.raises(ValueError, match="no `jno.np.parameter`"):
        _network(_domain({"faces": T2})).build()
    with pytest.raises(ValueError, match="path="):
        _network(_domain({"path": y0 * np.ones((2, 3))})).build()
    with pytest.raises(ValueError, match="Shape.line and attaches `faces=`"):
        _network(_domain({}, wire_props={"faces": (0, y0, 0, 1, 1, 1)})).build()
    with pytest.raises(ValueError, match="grid=jno.peec.faces"):
        _network(_domain({"faces": (T2[0], y0, *T2[2:])}), faces=False).build()


def test_a_property_peec_does_not_read_is_named(caplog):
    with caplog.at_level(logging.WARNING, logger="jno"):
        _network(_domain({}, t1_props={"sgima": 1.0, "k_th": 390.0})).build()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "'sgima' on ['t1']" in msg and "'k_th' on ['t1']" in msg and "reads only" in msg
