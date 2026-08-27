"""Guards for :meth:`jno.Shape.line` — a tube along a polyline.

The primitive exists for one reason, and ``test_line_can_be_a_named_region_with_its_own_size``
is that reason: a swept solid has no closed-form membership, so it cannot carry its own ``size=``
inside a region set (per-region mesh sizing calls ``contains`` and a sweep dies inside the boundary
tessellation). A tube's membership is "distance to the polyline <= r", which is exact — so the same
geometry that ``disk.sweep(path)`` cannot express as a sized region, ``Shape.line`` can.
"""

import math

import numpy as np
import pytest

import jno

PTS = [(0.0, 0.0, 0.0), (0.0, 5.0, 2.0), (0.0, 10.0, 0.0)]


def _distance_to_polyline(x, pts):
    """Independent reference: segment-wise point-to-polyline distance."""
    P = np.asarray(pts, float)
    a, b = P[:-1], P[1:]
    d = b - a
    t = np.clip(((x[:, None, :] - a) * d).sum(2) / (d * d).sum(1), 0.0, 1.0)
    proj = a + t[:, :, None] * d
    return np.linalg.norm(x[:, None, :] - proj, axis=2).min(1)


def test_contains_is_the_distance_to_the_polyline():
    """The closed form, against a reference written from the definition."""
    line = jno.Shape.line(PTS, d=0.75)
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 11.0, (4000, 3))
    got = np.asarray(line.contains(x))
    ref = _distance_to_polyline(x, PTS) <= 0.375 + 1e-6
    assert np.array_equal(got, ref)
    assert got.any() and not got.all(), "the sample should straddle the surface"


def test_line_can_be_a_named_region_with_its_own_size():
    """The whole reason this is a primitive: a SIZED, NAMED region in a multi-material set.

    Per-region mesh sizing resolves each region by a containment test, so a shape without closed-form
    membership cannot participate -- which is exactly why ``disk.sweep(path)`` cannot be used here.

    The host is sized at 0.8 rather than something coarser on purpose; see
    :func:`test_a_coarse_host_cannot_mesh_around_a_thin_inclusion` for why.
    """
    wire = jno.Shape.line(PTS, d=0.75, size=0.30).name("wire")
    host = (jno.Shape.box(-2, -2, -2, 2, 12, 4, size=0.8) - wire).name("air")
    d = (wire + host).domain()
    sets = d.mesh.cell_sets
    assert "wire" in sets and "air" in sets
    n_wire = len(np.asarray(sets["wire"][0]))
    assert n_wire > 100, f"the wire region got only {n_wire} cells"


def test_a_coarse_host_cannot_mesh_around_a_thin_inclusion():
    """Records a limitation that is NOT specific to ``Shape.line``, and its shape.

    Embedding a thin curved solid in a host whose cells are larger than the inclusion's DIAMETER
    fails in the surface mesher ("duplicated facets", "a segment and a facet intersect") and yields
    zero tetrahedra. Refining the inclusion does not help; refining the HOST does. A plain
    ``Shape.cylinder`` behaves identically, so this pins the host-size dependence rather than
    blaming the primitive -- and it is why the test above sizes its host at 0.8.

    Measured for a 0.75 mm tube: host 1.6 -> fails, host 0.8 -> ~9.7k tets, host 0.4 -> ~32k.

    It must RAISE, not return an empty domain: gmsh hands back zero points with the cell blocks
    present but empty, and without a guard that object behaves like a domain until some later
    reduction over an empty set fails far from the geometry that caused it.
    """
    wire = jno.Shape.line(PTS, d=0.75, size=0.30).name("wire")
    coarse = (jno.Shape.box(-2, -2, -2, 2, 12, 4, size=1.6) - wire).name("air")
    with pytest.raises(RuntimeError, match="EMPTY 3-D mesh"):
        (wire + coarse).domain().points


def test_meshes_to_roughly_the_analytic_volume():
    """A faceted cylinder is an INSCRIBED polygon, so the mesh must be under the true volume -- and
    not by more than the faceting at this resolution. Bounds it on both sides rather than asserting
    a single number that would move with the mesher."""
    line = jno.Shape.line(PTS, d=0.75, size=0.30)
    d = line.domain()
    P, C = np.asarray(d.points), np.asarray(d._cells_p1())
    e = np.stack([P[C[:, i + 1]] - P[C[:, 0]] for i in range(3)], -1)
    vol = float(np.abs(np.linalg.det(e)).sum() / 6.0)
    seg = float(np.linalg.norm(np.diff(np.asarray(PTS, float), axis=0), axis=1).sum())
    exact = math.pi * 0.375**2 * seg + 4.0 / 3.0 * math.pi * 0.375**3
    assert 0.80 * exact < vol < exact, f"{vol:.4f} vs analytic {exact:.4f}"


def test_bounds_enclose_the_tube():
    lo, hi = jno.Shape.line(PTS, d=0.75)._node[1].bounds()
    assert np.allclose(lo, (-0.375, -0.375, -0.375))
    assert np.allclose(hi, (0.375, 10.375, 2.375))


def test_two_dimensional_points_are_padded():
    """A planar polyline is written in 2-D and lifted, like every other primitive."""
    a = jno.Shape.line([(0, 0), (3, 4)], d=0.5)._node[1].points
    assert a == ((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(), "exactly one of d="),
        (dict(d=0.5, r=0.25), "exactly one of d="),
        (dict(d=-1.0), "radius must be positive"),
    ],
)
def test_refuses_an_ambiguous_or_impossible_cross_section(kwargs, match):
    with pytest.raises(ValueError, match=match):
        jno.Shape.line(PTS, **kwargs)


def test_refuses_a_degenerate_polyline():
    with pytest.raises(ValueError, match="at least two points"):
        jno.Shape.line([(0.0, 0.0, 0.0)], d=0.5)
