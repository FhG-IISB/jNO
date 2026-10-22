"""Design variables in the shape's own coordinates: `jno.shape.box(0, y0, ...)` with `y0` a
`jno.np.parameter`, and a line's vertices and radius, drive a `jno.peec` network through `jno.core`.

The oracle is `solve(geometry=)`, whose gradient `test_peec_traced_grid` pins against networks
rebuilt at each design: the parametric readout must give the same value and derivative, and its own
derivative must match central differences of itself. Also: what cannot move is refused by name, the
old `.attach(path=, radius=, faces=)` spelling points at the new one, and a property `jno.peec` does
not read is named.
"""

import logging

import jax
import jax.numpy as jnp
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64; save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


mm, CU = 1e-3, 5.8e7
P = 1.0 * mm
T2 = (0.0, 5 * mm, 0.0, 12 * mm, 7 * mm, 0.5 * mm)  # the second trace's faces at the reference


def _param(name, value):
    p = jno.np.parameter((1,), name=name)
    p.dtype(jnp.float64)
    p.initialize(jax.nn.initializers.constant(value))
    return p


def _tag(sh):
    d = sh.domain()
    d.tag("A", lambda x, y, z: x < 0.9 * mm)
    d.tag("B", lambda x, y, z: x > 11.1 * mm)
    return d


def _traces(t2, t1_props=None):
    a = jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=CU, **(t1_props or {})).name("t1")
    return a + t2.attach(sigma=CU).name("t2")


def _network(d, faces=True):
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6, **({"grid": jno.peec.faces(pitch=P)} if faces else {}))


def _value_fn(node):
    """The readout as a function of the parameters' values -- what `jno.core` threads in."""
    import equinox as eqx

    from jno._fem import _walk
    from jno.trace_evaluator import TraceEvaluator

    mcs = {m.model._parameter_name: m for m in _walk(node) if type(m).__name__ == "ModelCall"}

    def value(**vals):
        table = {
            mcs[n].model.layer_id: eqx.tree_at(lambda q: q.value, mcs[n].model.module, jnp.reshape(v, (1,)))
            for n, v in vals.items()
        }
        return jnp.real(jnp.asarray(TraceEvaluator(table).evaluate(node, context={}))).reshape(())

    return value


def test_a_parametric_face_is_the_geometry_solve_with_its_gradient():
    """y0 of the second trace a parameter: the value and dL/dy0 equal `solve(geometry=)` at the same
    y0 (the tested path), at the reference and moved off it, and dL/dy0 matches central differences."""
    y0 = _param("y0", T2[1])
    net = _network(_tag(_traces(jno.shape.box(T2[0], y0, *T2[2:], size=(P,) * 3))))
    value = _value_fn(net.solve(matrix_free=False).L)
    geo = lambda yy: jnp.real(net.solve(sigma={}, matrix_free=False, geometry={"t2": (T2[0], yy, *T2[2:])}).L)
    for yy in (T2[1], T2[1] - 0.4 * mm):
        assert float(value(y0=yy)) == pytest.approx(float(geo(yy)), rel=1e-12)
        g = float(jax.grad(lambda q: value(y0=q))(yy))
        assert g == pytest.approx(float(jax.grad(geo)(yy)), rel=1e-10)
        h = 1e-6 * mm
        fd = (float(value(y0=yy + h)) - float(value(y0=yy - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-5)
        assert g > 0, "a wider gap between the traces is a larger loop"


def test_an_expression_and_a_translation_move_the_same_face():
    """`y0 = gap + 3 mm` in the box, and a box translated by `gap`: one design, two spellings, and the
    translated box moves BOTH its y faces (a rigid shift), so it is compared with that geometry."""
    gap = _param("gap", 2 * mm)
    net = _network(_tag(_traces(jno.shape.box(T2[0], gap + 3 * mm, *T2[2:], size=(P,) * 3))))
    expr = _value_fn(net.solve(matrix_free=False).L)
    geo = lambda yy: jnp.real(net.solve(sigma={}, matrix_free=False, geometry={"t2": (T2[0], yy, *T2[2:])}).L)
    assert float(expr(gap=1.6 * mm)) == pytest.approx(float(geo(4.6 * mm)), rel=1e-12)

    sh = _param("sh", 0.0)
    moved = jno.shape.box(*T2, size=(P,) * 3).translate((0.0, sh, 0.0))
    net2 = _network(_tag(_traces(moved)))
    tr = _value_fn(net2.solve(matrix_free=False).L)
    rigid = jnp.real(
        net2.solve(
            sigma={}, matrix_free=False, geometry={"t2": (T2[0], T2[1] - 0.3 * mm, T2[2], T2[3], T2[4] - 0.3 * mm, T2[5])}
        ).L
    )
    assert float(tr(sh=-0.3 * mm)) == pytest.approx(float(rigid), rel=1e-12)


def test_a_wires_vertices_and_radius_are_parameters():
    """A bond wire over two traces: its apex height and radius in the line's own coordinates. Pinned
    against central differences; a thicker wire has less self inductance."""
    h, rr = _param("h", 0.0), _param("rr", 0.15 * mm)
    wire = (
        jno.shape.line(
            [
                (6 * mm, 1 * mm, 0.5 * mm),
                (6 * mm, 1 * mm, 1.5 * mm + h),
                (6 * mm, 6 * mm, 1.5 * mm + h),
                (6 * mm, 6 * mm, 0.5 * mm),
            ],
            r=rr,
            size=P,
        )
        .attach(sigma=3.8e7)
        .name("w")
    )
    a = jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=CU).name("t1")
    b = jno.shape.box(*T2, size=(P,) * 3).attach(sigma=CU).name("t2")
    d = (a + b + wire).domain()
    d.tag("A", lambda x, y, z: (x < 0.9 * mm) & (y < 3 * mm))
    d.tag("B", lambda x, y, z: (x < 0.9 * mm) & (y > 4 * mm))
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    value = _value_fn(jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).solve(matrix_free=False).L)
    gh, gr = jax.grad(lambda p, q: value(h=p, rr=q), argnums=(0, 1))(0.0, 0.15 * mm)
    fh = (float(value(h=1e-6, rr=0.15 * mm)) - float(value(h=-1e-6, rr=0.15 * mm))) / 2e-6
    fr = (float(value(h=0.0, rr=0.15 * mm + 1e-7)) - float(value(h=0.0, rr=0.15 * mm - 1e-7))) / 2e-7
    assert float(gh) == pytest.approx(fh, rel=1e-5)
    assert float(gr) == pytest.approx(fr, rel=1e-5)
    assert float(gh) > 0 and float(gr) < 0  # a taller loop is more inductive, a thicker wire less


def test_what_cannot_move_is_refused_by_name():
    y0 = _param("y0", T2[1])
    with pytest.raises(NotImplementedError, match="disk"):
        jno.shape.disk(0, y0, 1.0)
    with pytest.raises(NotImplementedError, match="rotate"):
        jno.shape.box(*T2).rotate((0, 0, 0), (0, 0, 1), y0)
    with pytest.raises(ValueError, match="no initial value"):
        jno.shape.box(T2[0], jno.np.parameter((1,), name="bare"), *T2[2:])
    # a parametric solid that is not a box
    t2 = jno.shape.box(T2[0], y0, *T2[2:], size=(P,) * 3) - jno.shape.box(1 * mm, 5 * mm, 0, 2 * mm, 6 * mm, 0.5 * mm)
    with pytest.raises(NotImplementedError, match="not an axis-aligned box"):
        _network(_tag(_traces(t2))).solve()
    # the uniform lattice cannot move a face
    with pytest.raises(ValueError, match="grid=jno.peec.faces"):
        _network(_tag(_traces(jno.shape.box(T2[0], y0, *T2[2:], size=(P,) * 3))), faces=False).solve()


def test_the_old_attach_spelling_points_at_the_new_one():
    y0 = _param("y0", T2[1])
    d = _tag(
        jno.shape.box(0, 0, 0, 12 * mm, 2 * mm, 0.5 * mm, size=(P,) * 3).attach(sigma=CU).name("t1")
        + jno.shape.box(*T2, size=(P,) * 3).attach(sigma=CU, faces=(T2[0], y0, *T2[2:])).name("t2")
    )
    with pytest.raises(ValueError, match="no longer read"):
        _network(d).solve()


def test_a_property_peec_does_not_read_is_named(caplog):
    with caplog.at_level(logging.WARNING, logger="jno"):
        _network(_tag(_traces(jno.shape.box(*T2, size=(P,) * 3), t1_props={"sgima": 1.0, "k_th": 390.0})))._build()
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "'sgima' on ['t1']" in msg and "'k_th' on ['t1']" in msg and "reads only" in msg
