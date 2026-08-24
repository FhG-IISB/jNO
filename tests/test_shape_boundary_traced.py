"""Sampling a shape's BOUNDARY inside jit, with analytic normals and no mesh.

The interior is a rejection problem; the boundary is not. It has measure zero, so no proposal drawn
from a volume ever lands on it. Points come from the primitives' own surfaces instead -- a circle by
angle, a box face by area, a polygon edge by length -- and which of them survive is decided by asking
`contains` on BOTH SIDES: a point is on the composite's boundary exactly when the two sides disagree.
That same test orients the normal, so a primitive never needs to know its own winding and a
subtracted operand's normal comes out flipped with no special case.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.geometry.shape import sample_on_boundary

jax.config.update("jax_enable_x64", True)

S = jno.Shape
SHAPES = {
    "rect": (S.rect(0, 0, 2, 1), 2),
    "disk": (S.disk(0, 0, 1), 2),
    "rect - disk": (S.rect(0, 0, 2, 1) - S.disk(0.7, 0.5, 0.28), 2),
    "polygon": (S.polygon([(0, 0), (2, 0), (1.6, 1), (0.5, 1.2)]), 2),
    "fused disks": (S.disk(0, 0, 1) | S.disk(1.2, 0, 1), 2),
    "translated disk": (S.disk(0, 0, 0.4).translate((1.0, 1.0, 0.0)), 2),
    "box": (S.box(0, 0, 0, 1, 1, 1), 3),
    "box - sphere": (S.box(0, 0, 0, 1, 1, 1) - S.sphere(0.5, 0.5, 0.5, 0.25), 3),
}


def _draw(shape, n=800, seed=0, **kw):
    f = jax.jit(functools.partial(sample_on_boundary, shape, n=n, **kw))
    p, nrm, filled = f(jax.random.PRNGKey(seed))
    return np.asarray(p), np.asarray(nrm), int(filled)


@pytest.mark.parametrize("name", list(SHAPES))
def test_points_are_on_the_boundary_with_outward_unit_normals(name):
    shape, dim = SHAPES[name]
    p, nrm, filled = _draw(shape)
    assert filled == 800, f"{name} only filled {filled}"
    q, d = p[:, :dim], nrm[:, :dim]
    assert np.abs(np.linalg.norm(d, axis=1) - 1).max() < 1e-12, f"{name} normals are not unit"
    h = 1e-5
    assert not np.asarray(shape.contains(q + h * d), dtype=bool).any(), f"{name} normal points inward"
    assert np.asarray(shape.contains(q - h * d), dtype=bool).all(), f"{name} point is not on the surface"


def test_a_circle_is_exact_not_a_chord():
    """The reason to sample the geometry rather than a mesh: no discretisation error at all."""
    p, nrm, _ = _draw(S.disk(0.0, 0.0, 1.0), n=4000)
    r = np.linalg.norm(p[:, :2], axis=1)
    assert np.abs(r - 1.0).max() < 1e-12, "points are not exactly on the circle"
    radial = p[:, :2] / r[:, None]
    assert np.abs((nrm[:, :2] * radial).sum(1) - 1.0).max() < 1e-12, "normals are not exactly radial"


def test_a_cut_contributes_its_own_surface_in_proportion():
    """`rect - disk` must sample BOTH the outer perimeter and the cut arc, weighted by length."""
    shape = S.rect(0, 0, 2, 1) - S.disk(0.7, 0.5, 0.28)
    p, _, _ = _draw(shape, n=6000)
    on_arc = np.abs(np.linalg.norm(p[:, :2] - np.array([0.7, 0.5]), axis=1) - 0.28) < 1e-9
    frac = on_arc.mean()
    expected = (2 * np.pi * 0.28) / (2 * (2 + 1) + 2 * np.pi * 0.28)
    assert abs(frac - expected) < 0.05, f"arc share {frac:.3f}, expected about {expected:.3f}"


def test_the_cut_surface_normal_is_flipped():
    """On a hole, outward for the DOMAIN points into the hole -- toward the disk's centre."""
    shape = S.rect(0, 0, 2, 1) - S.disk(0.7, 0.5, 0.28)
    p, nrm, _ = _draw(shape, n=6000)
    c = np.array([0.7, 0.5])
    on_arc = np.abs(np.linalg.norm(p[:, :2] - c, axis=1) - 0.28) < 1e-9
    inward = (p[on_arc, :2] - c) / 0.28
    assert (nrm[on_arc, :2] * inward).sum(1).max() < -0.999, "hole normals do not point into the hole"


@pytest.mark.parametrize("name", list(SHAPES))
def test_every_call_is_a_fresh_draw(name):
    shape, _ = SHAPES[name]
    seen = set()
    for i in range(10):
        p, _, _ = _draw(shape, n=256, seed=i)
        seen.update(map(tuple, np.round(p, 12)))
    assert len(seen) > 0.99 * 10 * 256, f"{name} repeated points across calls"


def test_unfilled_is_reported_not_hidden():
    """A boundary that is almost entirely swallowed still reports what it managed."""
    swallowed = S.disk(0, 0, 1) | S.disk(0, 0, 3)      # the inner circle is interior to the union
    _, _, filled = _draw(swallowed, n=400, max_rounds=1, batch=64)
    assert filled < 400


def test_extrude_says_where_to_go_instead_of_failing_obscurely():
    with pytest.raises(NotImplementedError, match="host-side"):
        sample_on_boundary(S.rect(0, 0, 1, 1).extrude(0.5), jax.random.PRNGKey(0), 16)
