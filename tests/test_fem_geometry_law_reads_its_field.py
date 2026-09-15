"""A geometry law reads the field it names — its own block of the solved state, component by component.

A mesh-motion law may read the solved state through a frozen field (``u.bind(...).freeze(...)``): a Stefan
front reads ``grad T . n``, a free surface reads the liquid's velocity. The driver used to hand EVERY frozen
field the whole flat state, and the readout then took the state's first ``n_vertices`` entries -- the first
field's block. A law reading any other field of a coupled problem therefore read the wrong field, with no
error (measured: a law on a field that is identically 0 moved the mesh 0.1). A vector field's component
(``uf[1]``) could not be read at all.

The oracles are exact. Every field here is constant in time (a no-flux diffusion of a constant initial
state), so a law ``y.d(t) - 0.5 * f`` on the top edge moves it by exactly ``0.5 * f * T`` -- forward Euler of
a constant speed -- and a law reading a field that is 0 does not move it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.trace import frozen_fields_in, refreeze

T_END, N_STEPS = 0.2, 4


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _domain():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, T_END, N_STEPS + 1))
    xi, yi, ti = d.variable("interior", split=True)
    xt, yt, tt = d.variable("top", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    return d, (xi, yi, ti), (xt, yt, tt), (x0, y0)


def _heat(f, s, xi, yi, ti):
    """f_t = 0.1 lap f with no-flux boundaries: a constant initial state stays exactly constant."""
    fb, sb = f.bind(x=xi, y=yi, t=ti), s.bind(x=xi, y=yi, t=ti)
    return fb.t * sb + 0.1 * (fb.x * sb.x + fb.y * sb.y)


def _vector_heat(u, v, xi, yi, ti):
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    return (
        ub.t[0] * vb[0]
        + ub.t[1] * vb[1]
        + 0.1 * (ub.x[0] * vb.x[0] + ub.y[0] * vb.y[0] + ub.x[1] * vb.x[1] + ub.y[1] * vb.y[1])
    )


def _top_shift(traj):
    p0, p1 = np.asarray(traj.meshes[0][0]), np.asarray(traj.meshes[-1][0])
    top = np.abs(p0[:, 1] - 1.0) < 1e-9
    return p1[top, 1] - p0[top, 1]


def _two_scalars(read):
    """Two coupled P1 scalar fields, a = 1 and b = 0; the top edge moves at 0.5 times the one it reads."""
    d, (xi, yi, ti), (xt, yt, tt), (x0, y0) = _domain()
    a, sa = d.fem_symbols(names=("a", "sa"))
    b, sb = d.fem_symbols(names=("b", "sb"))
    read_field = {"a": a, "b": b}[read]
    ff = read_field.bind(x=xt, y=yt).freeze(np.zeros(len(d.mesh.points)))
    fem = jno.fem(
        [
            _heat(a, sa, xi, yi, ti),
            _heat(b, sb, xi, yi, ti),
            yt.d(tt) - 0.5 * ff,
            a(x0, y0) - 1.0,
            b(x0, y0) - 0.0,
        ]
    )
    return fem, d


@pytest.mark.parametrize("read, value", [("a", 1.0), ("b", 0.0)])
def test_a_law_reads_the_field_it_names_in_a_coupled_problem(read, value):
    """Whichever block the assembler puts first, each law must read its own field. Before the fix, one of
    these two moved by the OTHER field's value."""
    fem, _ = _two_scalars(read)
    shift = _top_shift(fem.solve())
    assert np.allclose(shift, 0.5 * value * T_END, atol=1e-9), (
        f"the law reading {read!r} (= {value}) moved the top edge by {shift.min():.6f}..{shift.max():.6f}, "
        f"expected {0.5 * value * T_END:.6f}"
    )


def _vector_and_scalar(read, order):
    """A vector field u = (0, 1) of the given order beside a P1 scalar c = 0.25."""
    d, (xi, yi, ti), (xt, yt, tt), (x0, y0) = _domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=order)
    c, s = d.fem_symbols(names=("c", "s"))
    n = len(d.mesh.points)
    uf = u.bind(x=xt, y=yt).freeze(np.zeros((n, 2)))
    cf = c.bind(x=xt, y=yt).freeze(np.zeros(n))
    law = {"u0": uf[0], "u1": uf[1], "c": cf}[read]
    fem = jno.fem(
        [
            _vector_heat(u, v, xi, yi, ti),
            _heat(c, s, xi, yi, ti),
            yt.d(tt) - 0.5 * law,
            u(x0, y0)[0] - 0.0,
            u(x0, y0)[1] - 1.0,
            c(x0, y0) - 0.25,
        ]
    )
    return fem


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("read, value", [("u1", 1.0), ("u0", 0.0), ("c", 0.25)])
def test_a_law_reads_one_component_of_a_vector_field(read, value, order):
    """`uf[1]` is how a free surface reads the liquid's vertical velocity. It used to raise; the P2 case
    also checks that the law reads the VERTEX values of a higher-order block."""
    shift = _top_shift(_vector_and_scalar(read, order).solve())
    assert np.allclose(shift, 0.5 * value * T_END, atol=1e-9), (
        f"reading {read!r} (= {value}) with a P{order} vector field moved the top edge by "
        f"{shift.min():.6f}..{shift.max():.6f}, expected {0.5 * value * T_END:.6f}"
    )


@pytest.mark.parametrize("order", [1, 2])
def test_a_law_reads_the_vertex_values_of_a_varying_field(order):
    """One explicit step reads the INITIAL state exactly, so with c = 1 + x the top edge's first step is
    exactly 0.5 (1 + x_i) dt at each vertex. This pins WHICH DOFs are read: a constant field (as above)
    cannot tell a P2 block's vertex DOFs from its edge DOFs. `a = 7` sits in front of it, so reading the
    first block by mistake would show."""
    dt = 0.05
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, dt, 2))
    xi, yi, ti = d.variable("interior", split=True)
    xt, yt, tt = d.variable("top", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    a, sa = d.fem_symbols(names=("a", "sa"))
    c, sc = d.fem_symbols(names=("c", "sc"), order=order)
    cf = c.bind(x=xt, y=yt).freeze(np.zeros(len(d.mesh.points)))
    fem = jno.fem(
        [
            _heat(a, sa, xi, yi, ti),
            _heat(c, sc, xi, yi, ti),
            yt.d(tt) - 0.5 * cf,
            a(x0, y0) - 7.0,
            c(x0, y0) - (1.0 + x0),
        ]
    )
    traj = fem.solve()
    p0, p1 = np.asarray(traj.meshes[0][0]), np.asarray(traj.meshes[1][0])
    top = np.abs(p0[:, 1] - 1.0) < 1e-9
    assert np.allclose(p1[top, 1] - p0[top, 1], 0.5 * (1.0 + p0[top, 0]) * dt, atol=1e-9), (
        p1[top, 1] - p0[top, 1],
        0.5 * (1.0 + p0[top, 0]) * dt,
    )


def test_the_host_oracle_reads_the_same_block_as_the_march():
    """The host evaluator is the parity oracle for the traced one, and it had the same defect, so the
    parity test could not catch it. Both must read b -- not a -- off a state where the two differ
    everywhere. (A law on 'boundary' sampled with its normals: the host oracle's own scope, see
    `_geometry_velocity`.)"""
    from jno.utils.solver.fem_adapt import _geometry_motion_specs, _geometry_velocity, _geometry_velocity_fn

    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.2, 5))
    a, sa = d.fem_symbols(names=("a", "sa"))
    b, sb = d.fem_symbols(names=("b", "sb"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb, _nx, _ny = d.variable("boundary", normals=True, split=True)
    ci = d.variable("initial", split=True)
    bf = b.bind(x=xb, y=yb).freeze(np.zeros(len(d.mesh.points)))
    fem = jno.fem(
        [
            _heat(a, sa, xi, yi, ti),
            _heat(b, sb, xi, yi, ti),
            yb.d(tb) - 0.5 * bf,
            a(ci[0], ci[1]) - 1.0,
            b(ci[0], ci[1]) - 0.0,
        ]
    )
    pts = np.asarray(d.mesh.points)[:, :2]
    off = list(fem.offsets)
    ia, ib = fem.block_index(a), fem.block_index(b)
    state = np.zeros(off[-1])
    state[off[ia] : off[ia + 1]] = 1.0 + pts[:, 0]  # a = 1 + x
    state[off[ib] : off[ib + 1]] = 0.3 * pts[:, 1]  # b = 0.3 y
    spec = _geometry_motion_specs(fem, d)[0]
    expected = 0.5 * 0.3 * pts[np.asarray(spec["ids"]), 1]  # the law reads b at the driven vertices
    host = np.asarray(_geometry_velocity(spec, d, state))
    traced = np.asarray(_geometry_velocity_fn(spec, d)(jnp.asarray(pts), state))
    assert host == pytest.approx(expected, abs=1e-6), "the host oracle read the wrong field"
    assert traced == pytest.approx(expected, abs=1e-6), "the traced velocity read the wrong field"


def test_a_law_reading_a_field_this_problem_does_not_solve_is_refused():
    """A frozen field of a symbol that no weak form solves has no values in the state to read."""
    d, (xi, yi, ti), (xt, yt, tt), (x0, y0) = _domain()
    a, sa = d.fem_symbols(names=("a", "sa"))
    w, _sw = d.fem_symbols(names=("w", "sw"))  # never solved
    wf = w.bind(x=xt, y=yt).freeze(np.zeros(len(d.mesh.points)))
    fem = jno.fem([_heat(a, sa, xi, yi, ti), yt.d(tt) - 0.5 * wf, a(x0, y0) - 1.0])
    with pytest.raises(ValueError, match="does not solve"):
        fem.solve()


def test_a_frozen_value_reads_out_like_a_coordinate():
    """`yb * tf` must be the pointwise product. The value read out as (N,) against the coordinate's
    (1, N, 1), so the product broadcast into an (N, N) outer product -- no error, wrong numbers."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    u, _v = d.fem_symbols()
    w, _sw = d.fem_symbols(value_shape=(2,), names=("w", "sw"))
    xb, yb, _tb, _nx, _ny = d.variable("boundary", normals=True, split=True)
    pts = np.asarray(d.mesh.points)[:, :2]
    x, y = np.asarray(xb.eval()).reshape(-1), np.asarray(yb.eval()).reshape(-1)
    shape = np.asarray(yb.eval()).shape

    tf = u.bind(x=xb, y=yb).freeze(1.0 + pts[:, 0])  # f = 1 + x
    prod = np.asarray((yb * tf).eval())
    assert prod.shape == shape, f"scalar value read out as {prod.shape}, the coordinate as {shape}"
    assert np.allclose(prod.reshape(-1), y * (1.0 + x), atol=1e-6)

    wf = w.bind(x=xb, y=yb).freeze(np.stack([pts[:, 0], 2.0 + pts[:, 1]], axis=1))  # w = (x, 2 + y)
    comp = np.asarray((yb * wf[1]).eval())
    assert comp.shape == shape, f"a vector component read out as {comp.shape}, the coordinate as {shape}"
    assert np.allclose(comp.reshape(-1), y * (2.0 + y), atol=1e-6)


def test_a_law_mixing_a_coordinate_and_a_field_reads_each_at_its_own_vertex():
    """The top edge rises at 0.5 * a * x with a = 1, so each vertex moves by its OWN 0.5 * x * T. Under the
    outer-product broadcast the march read one vertex's x for all of them."""
    d, (xi, yi, ti), (xt, yt, tt), (x0, y0) = _domain()
    a, sa = d.fem_symbols(names=("a", "sa"))
    af = a.bind(x=xt, y=yt).freeze(np.zeros(len(d.mesh.points)))
    fem = jno.fem([_heat(a, sa, xi, yi, ti), yt.d(tt) - 0.5 * af * xt, a(x0, y0) - 1.0])
    traj = fem.solve()
    p0, p1 = np.asarray(traj.meshes[0][0]), np.asarray(traj.meshes[-1][0])
    top = np.abs(p0[:, 1] - 1.0) < 1e-9
    assert np.allclose(p1[top, 1] - p0[top, 1], 0.5 * p0[top, 0] * T_END, atol=1e-9), (
        p1[top, 1] - p0[top, 1],
        0.5 * p0[top, 0] * T_END,
    )


def test_refreeze_keeps_a_vector_fields_shape():
    """`refreeze` flattened every field, so a vector field came back (2n,) and the next assembly failed
    with an unrelated reshape error."""
    d, (xi, yi, _ti), _top, _ic = _domain()
    u, _v = d.fem_symbols(value_shape=(2,), names=("u", "v"))
    n = len(d.mesh.points)
    node = frozen_fields_in(u.bind(x=xi, y=yi).freeze(np.zeros((n, 2)))._expr)[0]
    assert refreeze(node, np.ones((n, 2))).values.shape == (n, 2)
    assert refreeze(node, np.ones(2 * n)).values.shape == (n, 2)
