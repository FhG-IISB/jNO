"""The interface frame that tied/periodic ties project through.

Matching a slave node against the master face means comparing the two **in the interface**, with the
across-interface coordinate removed. That used to be done by dropping the single global axis whose tag
means differed most, which assumes the two faces are planar, axis-aligned and separated by a pure
translation along that axis. :func:`_interface_frame` replaces it with an SVD fit of the master face's
own tangent plane.

Two things are checked here:

* **No regression.** On a planar axis-aligned pair the frame spans the same plane the transverse axes
  did, so the conforming and non-matching periodic paths are unchanged (``tests/test_fem_periodic*``
  cover those end-to-end; the equivalence is asserted directly below).
* **The new case.** A **coincident** (tied) interface — both faces at the same location, as two
  independently meshed bodies glued along a shared surface. The mean difference is then ~0, so the old
  axis-drop selected whichever coordinate carried the largest rounding error; when that landed on an
  in-plane axis the projection collapsed the face onto a line and produced silently wrong weights.

**Not** covered, because it is not supported by either the old or the new code: a *sheared* lattice,
where the offset between the two faces has a component **along** the interface. Projecting onto the
tangent plane removes the across-interface offset only, so an in-plane shift would tie each slave to
the wrong master location. A periodic cell whose lattice vector is normal to its faces is fine.
"""

import numpy as np
import pytest

from jno.utils.solver.fem_utils import _interface_frame, build_periodic_prolongation


def _plane_frame(normal):
    """An orthonormal ``(t1, t2, n)`` with ``n`` along ``normal`` — to *build* test planes with."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, seed)
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1), n


def _square_patch(origin, t1, t2):
    """Unit square as 4 corners + 2 triangles, embedded in the plane ``origin + s*t1 + q*t2``."""
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = origin + uv[:, :1] * t1 + uv[:, 1:] * t2
    return pts, np.array([[0, 1, 2], [0, 2, 3]])


# --------------------------------------------------------------------------- the frame itself


def test_frame_spans_the_face_plane_when_axis_aligned():
    """A z = const face: the two frame rows are orthonormal and orthogonal to the face normal, so
    they span exactly the plane the old transverse axes (x, y) did."""
    m = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    frame, origin = _interface_frame(m, m)
    assert frame.shape == (2, 3)
    assert np.allclose(frame @ frame.T, np.eye(2), atol=1e-12)  # orthonormal rows
    assert np.allclose(frame @ np.array([0.0, 0.0, 1.0]), 0.0, atol=1e-12)  # orthogonal to n
    assert np.allclose(origin, [0.5, 0.5, 1.0])


def test_frame_projection_preserves_in_plane_distances():
    """The frame is only defined up to a rotation *within* the plane, and every consumer (nearest-
    neighbour matching, edge parameters, barycentric weights) is rotation-invariant. Pin that down:
    projected pairwise distances equal the true in-plane distances."""
    t1, t2, n = _plane_frame([1.0, 2.0, -0.5])
    pts, _ = _square_patch(np.array([0.3, -0.2, 0.7]), t1, t2)
    frame, origin = _interface_frame(pts, pts)
    loc = (pts - origin) @ frame.T
    d_full = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    d_proj = np.linalg.norm(loc[:, None, :] - loc[None, :, :], axis=-1)
    assert np.allclose(d_full, d_proj, atol=1e-12)  # the face is planar -> projection is an isometry


def test_frame_fits_a_face_whose_normal_is_not_a_global_axis():
    t1, t2, n = _plane_frame([1.0, 1.0, 1.0])
    pts, _ = _square_patch(np.zeros(3), t1, t2)
    frame, _origin = _interface_frame(pts, pts)
    assert np.allclose(frame @ n, 0.0, atol=1e-12)


def test_frame_handles_coincident_faces():
    """The tied case: master and slave occupy the SAME plane, so the mean difference is ~0 and the
    old axis-drop had nothing to key on. The frame comes from the master face's own geometry, so it
    is unaffected."""
    m = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    s = np.array([[0.3, 0.2, 0.0], [0.6, 0.7, 0.0]])
    delta = m.mean(axis=0) - s.mean(axis=0)
    assert delta[2] == 0.0  # coincident: nothing separates the faces along the normal ...
    assert int(np.argmax(np.abs(delta))) in (0, 1)  # ... so the old axis-drop lands IN the plane
    frame, _ = _interface_frame(m, s)
    assert np.allclose(frame @ np.array([0.0, 0.0, 1.0]), 0.0, atol=1e-12)


def test_frame_is_empty_in_1d():
    """A 1-D interface is a single point: no in-interface coordinate exists, and every slave ties to
    the master exactly (distance 0), which is what dropping the only axis did."""
    frame, origin = _interface_frame(np.array([[1.0]]), np.array([[0.0]]))
    assert frame.shape == (0, 1)
    loc = (np.array([[0.0], [1.0]]) - origin) @ frame.T
    assert loc.shape == (2, 0)
    d2 = np.sum((loc[:, None, :] - loc[None, :, :]) ** 2, axis=-1)
    assert np.allclose(d2, 0.0)  # every pair is at distance 0 -> exact tie


def test_frame_refuses_a_collinear_3d_face():
    """A 3-D tie whose 'face' is an edge has no tangent plane. Refuse loudly rather than fit noise."""
    line = np.column_stack([np.linspace(0, 1, 5), np.zeros(5), np.zeros(5)])
    with pytest.raises(ValueError, match="collinear"):
        _interface_frame(line, line)


def test_frame_refuses_a_point_face():
    pt = np.zeros((3, 2))
    with pytest.raises(ValueError, match="single point"):
        _interface_frame(pt, pt)


# ------------------------------------------------------- end-to-end through the prolongation


def _coincident_faces(slave_uv, *, z=0.0):
    """A master unit square (4 corners, 2 triangles) and a slave node set in the SAME plane."""
    master = np.array([[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    slave = np.column_stack([slave_uv, np.full(len(slave_uv), z)])
    pts = np.vstack([master, slave])
    tags = {"a": np.arange(4), "b": np.arange(4, 4 + len(slave))}
    return pts, tags, {"a": tris}


def test_tied_coincident_interface_reproduces_constant_and_linear():
    """The Phase-B enabler: two faces at the SAME location with different node layouts. The
    prolongation must be a partition of unity and reproduce a linear field exactly."""
    uv = np.array([[0.3, 0.2], [0.7, 0.6], [0.5, 0.5], [0.2, 0.8], [0.9, 0.1]])
    pts, tags, facets = _coincident_faces(uv)
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets=facets)
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])

    assert res["n_red"] == 4  # the 5 slave nodes are eliminated onto the 4 master corners
    assert np.allclose(P.sum(axis=1), 1.0)
    for a, b, c in [(0.0, 0.0, 1.0), (2.0, -1.5, 0.3)]:
        field = a * pts[:, 0] + b * pts[:, 1] + c
        assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_tied_interface_on_a_tilted_plane_reproduces_linear():
    """Same, on a plane whose normal is not a global axis — the frame has to be fitted, not guessed."""
    t1, t2, n = _plane_frame([1.0, 1.0, 1.0])
    master, tris = _square_patch(np.zeros(3), t1, t2)
    uv = np.array([[0.3, 0.2], [0.7, 0.6], [0.25, 0.65]])
    slave = uv[:, :1] * t1 + uv[:, 1:] * t2
    pts = np.vstack([master, slave])
    tags = {"a": np.arange(4), "b": np.arange(4, 4 + len(slave))}
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets={"a": tris})
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    for coef in [(0.0, 0.0, 0.0, 1.0), (2.0, -1.5, 0.4, 0.3)]:
        field = pts @ np.array(coef[:3]) + coef[3]
        assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_tied_interface_survives_a_high_density_ratio():
    """Extreme: one master facet pair against a 1:8-refined slave face (81 nodes)."""
    g = np.linspace(0.02, 0.98, 9)
    uv = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    pts, tags, facets = _coincident_faces(uv)
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets=facets)
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert res["n_red"] == 4 and len(kept) == 4
    assert np.allclose(P.sum(axis=1), 1.0)
    field = 1.3 * pts[:, 0] - 0.7 * pts[:, 1] + 2.0
    assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_tied_interface_with_a_slave_node_on_a_master_node():
    """Degenerate overlap: a slave node coincident with a master node ties exactly (distance 0),
    the rest interpolate. Both branches must coexist in one tie."""
    uv = np.array([[0.0, 0.0], [1.0, 1.0], [0.4, 0.3]])  # first two land ON master corners
    pts, tags, facets = _coincident_faces(uv)
    res = build_periodic_prolongation(pts, [("a", "b")], tags, facets=facets)
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    field = 0.9 * pts[:, 0] + 1.7 * pts[:, 1] - 0.4
    assert np.allclose(P @ field[kept], field, atol=1e-10)


def test_periodic_pair_still_ties_across_a_normal_offset():
    """No-regression at the prolongation level: the ordinary periodic case (faces separated along
    their own normal) is unchanged — a linear in-plane field is still reproduced exactly."""
    master = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    slave = np.array([[0.3, 0.2, 0.0], [0.7, 0.6, 0.0], [0.5, 0.5, 0.0]])
    pts = np.vstack([master, slave])
    tags = {"top": np.arange(4), "bot": np.arange(4, 7)}
    res = build_periodic_prolongation(pts, [("top", "bot")], tags, facets={"top": tris})
    P = np.asarray(res["P_node"].todense())
    kept = np.asarray(res["kept_nodes"])
    assert np.allclose(P.sum(axis=1), 1.0)
    field = 2.0 * pts[:, 0] - 1.5 * pts[:, 1] + 0.3
    assert np.allclose(P @ field[kept], field, atol=1e-10)
