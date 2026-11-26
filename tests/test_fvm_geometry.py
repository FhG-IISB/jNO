"""The control volumes and faces of ``jno.fvm`` (:mod:`jno._fvm.geometry`), on every cell type."""

import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402
from jno._fvm.geometry import cell_graph, face_geometry  # noqa: E402


def _graph(domain):
    cells, kind = domain._cells_topo()
    pts = np.asarray(domain.mesh_connectivity["points"])[:, : domain.dimension]
    return cell_graph(pts, cells, kind)


MESHES = {
    "line": lambda: jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.1).domain(),
    "triangle": lambda: jno.shape.rect(0, 0, 1, 1, size=0.1).domain(),
    "quad": lambda: jno.shape.rect(0, 0, 1, 1).structured(n=8).quad().domain(),
    "tetra": lambda: jno.shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain(),
    "hexahedron": lambda: jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=4).quad().domain(),
}


@pytest.fixture(scope="module", params=list(MESHES))
def graph(request):
    g = _graph(MESHES[request.param]())
    assert g.cell_type == request.param
    return g


def test_every_control_volume_is_closed(graph):
    """The outward area vectors of a closed volume sum to zero (the divergence theorem for a constant)."""
    acc = np.zeros((graph.n, graph.dim))
    np.add.at(acc, graph.owner, graph.area)
    np.add.at(acc, graph.neigh, -graph.area)
    np.add.at(acc, graph.b_owner, graph.b_area)
    assert np.abs(acc).max() < 1e-14


def test_volumes_and_centroids_add_up_to_the_unit_box(graph):
    assert graph.volume.min() > 0
    assert graph.volume.sum() == pytest.approx(1.0, abs=1e-13)
    first_moment = (graph.volume[:, None] * graph.centroid).sum(axis=0)
    np.testing.assert_allclose(first_moment, 0.5, atol=1e-13)


def test_boundary_faces_cover_the_boundary_and_point_outward(graph):
    measure = np.linalg.norm(graph.b_area, axis=1).sum()
    assert measure == pytest.approx(2.0 * graph.dim, abs=1e-12)  # the surface of the unit box
    outward = np.einsum("fd,fd->f", graph.b_area, graph.b_centroid - 0.5)
    assert np.all(outward > 0)


def test_interior_area_vectors_point_from_owner_to_neighbour(graph):
    assert np.all(np.einsum("fd,fd->f", graph.area, graph.delta) > 0)
    np.testing.assert_allclose(graph.delta, graph.centroid[graph.neigh] - graph.centroid[graph.owner])


def test_one_triangle_and_one_tet_by_hand():
    g = cell_graph(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), np.array([[0, 1, 2]]), "triangle")
    assert g.volume[0] == pytest.approx(0.5)
    np.testing.assert_allclose(g.centroid[0], [1 / 3, 1 / 3])
    g = cell_graph(np.eye(4, 3, k=-1) + 0.0, np.array([[0, 1, 2, 3]]), "tetra")
    assert g.volume[0] == pytest.approx(1 / 6)
    np.testing.assert_allclose(g.centroid[0], [0.25, 0.25, 0.25])


def test_a_planar_polygon_face_is_exact():
    """A unit square face split about its vertex mean: area 1, centroid at its middle, normal along z."""
    sq = np.array([[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]]])
    area, centroid = face_geometry(sq)
    np.testing.assert_allclose(area, [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(centroid, [[0.5, 0.5, 2.0]])


def test_a_periodic_pair_becomes_interior_faces():
    g = _graph(MESHES["quad"]())
    left = np.flatnonzero(np.isclose(g.b_centroid[:, 0], 0.0))
    right = np.flatnonzero(np.isclose(g.b_centroid[:, 0], 1.0))
    p = g.with_periodic(left, right, shift=(-1.0, 0.0))
    assert p.b_owner.size == g.b_owner.size - 16 and p.owner.size == g.owner.size + 8
    new = slice(g.owner.size, None)
    np.testing.assert_allclose(p.delta[new], [[-1 / 8, 0.0]] * 8, atol=1e-14)  # one cell width, across the seam
    assert np.all(np.einsum("fd,fd->f", p.area[new], p.delta[new]) > 0)
    acc = np.zeros((p.n, 2))
    np.add.at(acc, p.owner, p.area)
    np.add.at(acc, p.neigh, -p.area)
    np.add.at(acc, p.b_owner, p.b_area)
    assert np.abs(acc).max() < 1e-14


def test_a_non_conforming_periodic_pair_raises():
    g = _graph(MESHES["triangle"]())
    left = np.flatnonzero(np.isclose(g.b_centroid[:, 0], 0.0))
    bottom = np.flatnonzero(np.isclose(g.b_centroid[:, 1], 0.0))
    with pytest.raises(ValueError, match="periodic"):
        g.with_periodic(left, bottom[: left.size], shift=(0.0, 1.0))
