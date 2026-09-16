"""Signed distance on a shape: negative inside, zero on the boundary, positive outside.

`contains` answers a yes/no; an SDF answers *how far*, which is what a hard boundary condition and a
smoothed material indicator both need. Two calling forms serve the two places it is useful -- an
(N, dim) array for sampling and membership, and per-axis `Variable`s for a weak form or a PINN
ansatz, where arithmetic is emitted through jno.np rather than evaluated. Both come from the same
per-primitive formulas; only the array module differs.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

S = jno.shape


@pytest.fixture(autouse=True)
def _x64():
    """float64 for the exactness checks, restored after each test.

    This was a module-level ``jax.config.update``, which flips the flag for the whole process at
    COLLECTION time and never puts it back -- every test collected after this file then ran in
    float64, the leak ``conftest._restore_x64`` exists to catch. It also hid the float32 NaN
    gradient below, since nothing here ever ran in the precision a PINN trains in.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


SHAPES = {
    "rect": (S.rect(0, 0, 2, 1), 2),
    "disk": (S.disk(0, 0, 1), 2),
    "polygon": (S.polygon([(0, 0), (2, 0), (1.6, 1), (0.5, 1.2)]), 2),
    "box": (S.box(0, 0, 0, 1, 1, 1), 3),
    "sphere": (S.sphere(0, 0, 0, 1), 3),
    "cylinder": (S.cylinder(0, 0, 0, 0, 0, 1, 0.4), 3),
    "cut": (S.rect(0, 0, 2, 1) - S.disk(0.7, 0.5, 0.28), 2),
    "fuse": (S.disk(0, 0, 1) | S.disk(1.2, 0, 1), 2),
    "inter": (S.rect(0, 0, 1, 1) & S.rect(0.4, 0, 0.6, 1), 2),
    "translate": (S.disk(0, 0, 0.4).translate((1.0, 1.0, 0.0)), 2),
    "rotate": (S.box(0, 0, 0, 1, 1, 1).rotate((0, 0, 0), (0, 0, 1), 0.3), 3),
    "extrude": ((S.rect(0, 0, 1, 1) - S.disk(0.5, 0.5, 0.2)).extrude(0.6), 3),
}


def test_primitive_distances_are_exact():
    assert np.allclose(S.disk(0, 0, 1).sdf(np.array([[0.0, 0], [0.5, 0], [1.0, 0], [2.0, 0]])), [-1.0, -0.5, 0.0, 1.0])
    assert np.allclose(S.rect(0, 0, 2, 1).sdf(np.array([[1.0, 0.5], [0.0, 0.5], [3.0, 2.0]])), [-0.5, 0.0, np.sqrt(2)])
    assert np.allclose(
        S.box(0, 0, 0, 1, 1, 1).sdf(np.array([[0.5, 0.5, 0.5], [0, 0.5, 0.5], [2, 0.5, 0.5]])), [-0.5, 0.0, 1.0]
    )


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_sign_is_exactly_contains(name):
    """contains(p) == (sdf(p) <= 0). If these ever disagree, one of them is wrong."""
    shape, dim = SHAPES[name]
    q = np.random.default_rng(0).uniform(-0.6, 1.8, (4000, dim))
    assert np.array_equal(np.asarray(shape.sdf(q)) <= 0, np.asarray(shape.contains(q), dtype=bool))


@pytest.mark.parametrize("name", list(SHAPES))
def test_traces_and_matches_eager(name):
    shape, dim = SHAPES[name]
    q = np.random.default_rng(1).uniform(-0.6, 1.8, (256, dim))
    assert np.allclose(np.asarray(jax.jit(shape.sdf)(jnp.asarray(q))), np.asarray(shape.sdf(q)))


def test_the_gradient_is_the_unit_outward_direction():
    """|grad sdf| = 1 is the defining property of a true distance function."""
    shape = S.disk(0.0, 0.0, 1.0)
    one = lambda q: shape.sdf(q[None, :])[0]
    g = jax.jit(jax.vmap(jax.grad(one)))(jnp.asarray(np.random.default_rng(2).uniform(-2, 2, (32, 2))))
    assert np.abs(np.linalg.norm(np.asarray(g), axis=1) - 1.0).max() < 1e-8


@pytest.mark.parametrize("name", list(SHAPES))
def test_the_trace_form_agrees_with_the_array_form(name):
    """sdf(x, y) on Variables must be the same function as sdf(pts) on an array."""
    shape, dim = SHAPES[name]
    dom = shape.domain()
    c = dom.variable("interior", sample=(200, None), split=True)
    expr = shape.sdf(*c[:dim])
    got = np.asarray(expr.eval()).reshape(-1)
    pts = np.stack([np.asarray(c[i].eval()).reshape(-1) for i in range(dim)], axis=1)
    assert np.abs(got - np.asarray(shape.sdf(pts))).max() < 1e-12


def test_hard_boundary_condition_is_exactly_zero_on_the_boundary():
    """The point of the trace form: u = g + sdf*net satisfies u = g on the boundary BY CONSTRUCTION,
    not by penalty -- and on a shape with no hand-writable x(1-x) factor."""
    from jno.geometry.shape import sample_on_boundary

    shape = S.rect(0, 0, 2, 1) - S.disk(0.7, 0.5, 0.28)
    p, _, _ = jax.jit(functools.partial(sample_on_boundary, shape, n=2000))(jax.random.PRNGKey(0))
    assert np.abs(np.asarray(shape.sdf(np.asarray(p)[:, :2]))).max() < 1e-12


def test_no_closed_form_says_so():
    with pytest.raises(NotImplementedError, match="no closed-form signed distance"):
        S.box(0, 0, 0, 1, 1, 1).fillet(0.2).sdf(np.zeros((4, 3)))


@pytest.mark.parametrize(
    "name,point,expected",
    [
        ("rect interior, nearest the left edge", (0.3, 0.6), (-1.0, 0.0)),
        ("rect centre, on the medial axis", (0.5, 0.5), None),
        ("disk centre, on the medial axis", (0.0, 0.0), None),
        ("box interior", (0.2, 0.5, 0.7), (-1.0, 0.0, 0.0)),
    ],
)
def test_the_gradient_is_finite_in_float32(name, point, expected):
    """float32 is what a PINN trains in, and a hard-BC ansatz is what the trace form is for.

    A ``s + 1e-300`` guard under each square root rounds to exactly 0 in float32, so the box's
    "outside" term was ``sqrt(0)`` at every interior point: ``inf * 0`` in the chain rule, NaN out.
    Measured before the fix: ``[nan, nan]`` at every interior point of a rect and at a disk's centre,
    while float64 gave the right answer. Where the nearest boundary is unique the gradient is the
    analytic inward unit normal; on the medial axis the distance has a kink and any subgradient is
    acceptable, so there only finiteness is asserted.
    """
    shapes = {"rect": S.rect(0, 0, 1, 1), "disk": S.disk(0, 0, 1), "box": S.box(0, 0, 0, 1, 1, 1)}
    shape = shapes[name.split()[0]]
    jax.config.update("jax_enable_x64", False)  # the autouse fixture restores it
    q = jnp.asarray(point, dtype=jnp.float32)
    g = np.asarray(jax.grad(lambda p: shape.sdf(p[None, :])[0])(q))
    assert g.dtype == np.float32
    assert np.isfinite(g).all(), f"{name}: NaN gradient in float32 -- {g}"
    if expected is not None:
        assert np.allclose(g, expected, atol=1e-6), f"{name}: grad {g}, want {expected}"


def test_a_boundary_point_reads_exactly_zero():
    """The old guard left a bias: a point on a box edge read 1e-150, not 0. None now."""
    pts = np.array([[1.0, 0.5], [0.0, 0.25], [0.5, 1.0]])
    assert np.array_equal(np.asarray(S.rect(0, 0, 1, 1).sdf(pts)), np.zeros(3))
