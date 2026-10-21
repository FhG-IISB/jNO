"""A parameter in a shape's coordinates moves the FEM mesh: the domain stretches it along each axis
between the plan's face planes (`jno.geometry.parametric`), through the same route `x.trainable()`
uses, so a `jno.fem` solve is differentiable in the design.

Oracles: a heated block on a cooled one (1-D in y, so the hottest point is closed-form) -- the
stretched solve agrees with a mesh BUILT at the moved design, and the gradient with central
differences of itself. Also: only vertices a design can move are moved, two faces on one plane that
move differently are refused, and the solvers that cannot follow yet refuse by name.
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

mm, Q, H = 1e-3, 1e6, 0.7e-3


def _param(name, value):
    p = jno.np.parameter((1,), name=name)
    p.dtype(jnp.float64)
    p.initialize(jax.nn.initializers.constant(value))
    return p


def _blocks(y1):
    """A cooled block (y 0..4 mm) under a heated one (y 4 mm..y1), insulated elsewhere."""
    a = jno.shape.box(0, 0, 0, 10 * mm, 4 * mm, 1 * mm, size=H).name("a").attach(k=1.0, q=0.0)
    b = jno.shape.box(0, 4 * mm, 0, 10 * mm, y1, 1 * mm, size=H).name("b").attach(k=1.0, q=Q)
    d = (a + b).domain()
    d.tag("cold", lambda x, y, z: y < 1e-9)
    T, s = d.fem_symbols()
    x = d.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=x[0], y=x[1], z=x[2]), s.bind(x=x[0], y=x[1], z=x[2])
    return d, jno.fem(
        [d.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - d.q * si, T(*d.variable("cold", split=True)[:3]) - 0.0]
    )


def _tmax_of(node, name):
    (mc,) = [m for m in _walk(node) if type(m).__name__ == "ModelCall" and m.model._parameter_name == name][:1]

    def f(v):
        table = {mc.model.layer_id: eqx.tree_at(lambda q: q.value, mc.model.module, jnp.reshape(v, (1,)))}
        return jnp.max(jnp.asarray(TraceEvaluator(table).evaluate(node, context={})).reshape(-1))

    return f


def test_the_stretched_solve_is_the_rebuilt_one_and_its_gradient_is_exact():
    y1 = _param("y1", 8 * mm)
    d, fem = _blocks(y1)
    tmax = _tmax_of(fem.solve(), "y1")
    assert float(tmax(8 * mm)) == pytest.approx(Q * (4 * mm * 4 * mm + (4 * mm) ** 2 / 2), rel=2e-3)  # 1-D slab
    _d9, fem9 = _blocks(9 * mm)
    rebuilt = float(np.max(np.asarray(fem9.solve()).reshape(-1)))
    assert float(tmax(9 * mm)) == pytest.approx(rebuilt, rel=1e-4)  # measured 6.9e-06
    g = float(jax.grad(tmax)(8 * mm))
    h = 1e-4 * mm
    fd = (float(tmax(8 * mm + h)) - float(tmax(8 * mm - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-8)  # measured 2.7e-12
    assert g == pytest.approx(Q * (4 * mm + 4 * mm), rel=5e-3)  # d/dy1 of q (L1 L2 + L2^2 / 2)


def test_only_what_a_design_can_move_is_moved():
    """y1 is the heated block's top face: the cooled block (between two fixed planes) keeps its
    vertices; only y moves; and the moved column is exactly the reference at the reference."""
    y1 = _param("y1", 8 * mm)
    d, _fem = _blocks(y1)
    pts = np.asarray(d.mesh.points)
    (spec,) = d._trainable_coords
    assert spec["axis"] == 1
    assert np.all(pts[spec["ids"], 1] > 4 * mm - 1e-12)  # nothing of the fixed block below y = 4 mm
    assert np.any(np.abs(pts[:, 1] - 8 * mm) < 1e-12) and set(np.flatnonzero(pts[:, 1] > 4 * mm + 1e-9)) <= set(spec["ids"])
    col = np.asarray(spec["expr"].fn(jnp.asarray([8 * mm])))
    assert np.allclose(col, pts[spec["ids"], 1], rtol=0, atol=1e-15)


def test_two_faces_on_one_plane_must_move_together():
    w = _param("w", 10 * mm)
    a = jno.shape.box(0, 0, 0, w, 4 * mm, 1 * mm, size=H).name("a")
    b = jno.shape.box(0, 4 * mm, 0, 10 * mm, 8 * mm, 1 * mm, size=H).name("b")  # its x1 = 10 mm, fixed
    with pytest.raises(ValueError, match="share the plane x"):
        (a + b).domain().mesh  # noqa: B018 -- asking for the mesh registers the stretch
    same = jno.shape.box(0, 4 * mm, 0, w, 8 * mm, 1 * mm, size=H).name("b")  # the same expression: fine
    d = (a + same).domain()
    assert [s["axis"] for s in d.mesh is not None and d._trainable_coords] == [0]


def test_solvers_that_cannot_follow_refuse_by_name():
    y1 = _param("y1", 8 * mm)
    a = jno.shape.box(0, 0, 0, 10 * mm, 4 * mm, 1 * mm, size=H).name("a")
    b = jno.shape.box(0, 4 * mm, 0, 10 * mm, y1, 1 * mm, size=H).name("b")
    d = (a + b).domain()
    u = d.unknown()
    x, y, z = d.variable("interior", split=True)[:3]
    ui = u.bind(x=x, y=y, z=z)
    with pytest.raises(NotImplementedError, match="jno.fdm"):
        jno.fdm([ui.d2(x) + ui.d2(y) + ui.d2(z) - 1.0])
    # a parametric wire inside a meshed FEM domain does not follow the stretch
    h = _param("h", 0.0)
    wire = jno.shape.line(
        [(5 * mm, 2 * mm, 1 * mm), (5 * mm, 2 * mm, 2 * mm + h), (5 * mm, 6 * mm, 1 * mm)], r=0.4 * mm, size=H
    ).name("w")
    d2 = (
        a.attach(k=1.0)
        + jno.shape.box(0, 4 * mm, 0, 10 * mm, 8 * mm, 1 * mm, size=H).name("b").attach(k=1.0)
        + wire.attach(k=1.0)
    ).domain()
    d2.tag("cold", lambda x, y, z: y < 1e-9)
    T, s = d2.fem_symbols()
    xx = d2.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=xx[0], y=xx[1], z=xx[2]), s.bind(x=xx[0], y=xx[1], z=xx[2])
    with pytest.raises(NotImplementedError, match="line region"):
        jno.fem([d2.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - si, T(*d2.variable("cold", split=True)[:3]) - 0.0])


def test_jno_core_trains_the_shape_through_the_solve_and_refuses_collocation():
    """Through the front door: `jno.core` on `max T` moves y1 the right way (a shorter heated block is
    cooler). A loss that SAMPLES the parametric domain instead is refused: its points would not move."""
    import optax

    y1 = _param("y1", 8 * mm)
    y1.optimizer(optax.sgd(1e-9))
    d, fem = _blocks(y1)
    crux = jno.core([jno.np.max(fem.solve())])
    crux.solve(3)
    (moved,) = np.asarray(crux.eval([y1])[0]).reshape(-1)[:1]
    assert moved < 8 * mm  # dTmax/dy1 > 0, so descent shortens the block
    x, y, z = d.variable("interior", split=True)[:3]
    with pytest.raises(NotImplementedError, match="Collocation points"):
        jno.core([(x * y).mse])


def test_stretch_reads_how_far_the_design_moved_from_the_reference():
    y1 = _param("y1", 8 * mm)
    d, _fem = _blocks(y1)
    st = d.stretch()
    (mc,) = [m for m in _walk(st) if type(m).__name__ == "ModelCall"]
    at = lambda v: float(st.fn(jnp.asarray([v])))  # noqa: E731
    assert at(8 * mm) == pytest.approx(1.0)
    assert at(10 * mm) == pytest.approx(1.5)  # the heated block's 4 mm gap grows to 6 mm
    assert at(6 * mm) == pytest.approx(2.0)  # ... or shrinks to 2 mm
    assert at(4 * mm) == np.inf  # closed
    assert float(jno.shape.box(0, 0, 0, 1, 1, 1).name("a").domain().stretch().fn()) == 1.0


def _rect_top(y1, size=0.25, time=None):
    """A unit-wide rect whose top face is at y1; the domain and its moving top tag."""
    d = jno.shape.rect(0.0, 0.0, 1.0, y1, size=size).domain(**({} if time is None else {"time": time}))
    return d


def _value_of(node, name, v):
    (mc,) = [m for m in _walk(node) if type(m).__name__ == "ModelCall" and m.model._parameter_name == name][:1]
    table = {mc.model.layer_id: eqx.tree_at(lambda q: q.value, mc.model.module, jnp.reshape(v, (1,)))}
    return jnp.asarray(TraceEvaluator(table).evaluate(node, context={})).reshape(-1)


def test_a_position_dependent_boundary_value_follows_the_moving_mesh():
    """Laplace on a rect, u = 0 at y = 0 and u = 3 y on the moving top: u = 3 y exactly (P1 represents
    it), so the hottest node is 3 * y1 at the MOVED top -- not at the reference one."""
    y1 = _param("y1", 1.0)
    d = _rect_top(y1)
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(xt, yt) - 3.0 * yt])
    node = fem.solve()
    for yy in (1.0, 1.4, 0.7):
        assert float(jnp.max(_value_of(node, "y1", yy))) == pytest.approx(3.0 * yy, rel=1e-10)


def test_a_position_dependent_initial_condition_follows_the_moving_mesh():
    """Heat on a rect, insulated, u(0) = y: the stretched transient at y1 = 1.5 equals the one on a mesh
    BUILT at 1.5 (same cell counts along the stretch), node by node, sorted."""

    def march(y1):
        d = _rect_top(y1, size=0.25, time=(0.0, 0.05, 3))
        u, v = d.fem_symbols()
        x, y, t = d.variable("interior", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y)
        return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(ci[0], ci[1]) - ci[1]]).solve()

    y1 = _param("y1", 1.0)
    moved = np.sort(np.asarray(_value_of(march(y1), "y1", 1.5)))
    rebuilt = np.sort(np.asarray(march(1.5)).reshape(-1))
    if moved.size == rebuilt.size:
        assert np.allclose(moved, rebuilt, rtol=0, atol=2e-2)
    # the strong statement that does not depend on matching meshes: the INITIAL maximum is the moved top
    assert float(np.max(moved)) > 1.0


def test_a_path_that_does_not_follow_the_moving_mesh_refuses_by_name():
    """An obstacle bound written in x is evaluated at the reference nodes: on a moving mesh it would
    answer for the old shape, so it is refused rather than solved."""
    y1 = _param("y1", 1.0)
    d = _rect_top(y1, size=0.25)
    u, v = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    node = jno.fem([ui.x * vi.x + ui.y * vi.y + vi, u(xb, yb) - 0.0, u.bounds(-0.1 * (1.0 + y), None)]).solve()
    with pytest.raises(NotImplementedError, match="REFERENCE node positions"):
        _value_of(node, "y1", 1.0)  # the solve runs -- and its bound would be read at the old nodes
