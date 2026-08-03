"""Nodal measures and the point-in-polygon test behind boundary-normal orientation.

``compute_nodal_ds`` and ``compute_nodal_volumes`` partition an exactly known quantity -- the
boundary measure and the domain measure -- so every test here checks against the analytic value
rather than against a second implementation of the same loop.
"""

import numpy as np
import pytest
from shapely.geometry import box

import jno
from jno.domain.mesh_utils import MeshUtils

SQUARE_HOLE = box(0.0, 0.0, 1.0, 1.0).difference(box(0.4, 0.4, 0.6, 0.6))


@pytest.fixture(scope="module")
def square():
    return jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08).mesh_connectivity


@pytest.fixture(scope="module")
def annulus():
    return jno.domain(SQUARE_HOLE, mesh_size=0.05).mesh_connectivity


@pytest.fixture(scope="module")
def cube():
    return jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.25).domain().mesh_connectivity


@pytest.fixture(scope="module")
def line():
    return jno.domain.line((0.0, 2.0), 0.05).mesh_connectivity


# --------------------------------------------------------------------------------------------
# nodal ds -- sums to the boundary measure
# --------------------------------------------------------------------------------------------
def test_nodal_ds_sums_to_the_perimeter(square):
    assert MeshUtils.compute_nodal_ds(square).sum() == pytest.approx(4.0, rel=1e-12)


def test_nodal_ds_counts_a_hole_boundary(annulus):
    # outer perimeter 4 plus the 0.2-square hole's own perimeter 0.8
    assert MeshUtils.compute_nodal_ds(annulus).sum() == pytest.approx(4.8, rel=1e-12)


def test_nodal_ds_sums_to_the_surface_area_in_3d(cube):
    assert MeshUtils.compute_nodal_ds(cube).sum() == pytest.approx(6.0, rel=1e-12)


def test_nodal_ds_sums_to_the_length_in_1d(line):
    assert MeshUtils.compute_nodal_ds(line).sum() == pytest.approx(2.0, rel=1e-12)


def test_nodal_ds_is_zero_away_from_the_boundary(square):
    ds = MeshUtils.compute_nodal_ds(square)
    assert np.all(ds >= 0.0)
    assert np.count_nonzero(ds) < square["n_points"]  # interior nodes carry no boundary measure


def test_nodal_ds_selects_the_requested_nodes(square):
    ds = MeshUtils.compute_nodal_ds(square)
    picked = np.array([0, 3, 7])
    assert np.array_equal(MeshUtils.compute_nodal_ds(square, boundary_indices=picked), ds[picked])


# --------------------------------------------------------------------------------------------
# nodal volumes -- sums to the domain measure
# --------------------------------------------------------------------------------------------
def test_nodal_volumes_sum_to_the_area(square):
    assert MeshUtils.compute_nodal_volumes(square).sum() == pytest.approx(1.0, rel=1e-12)


def test_nodal_volumes_exclude_the_hole(annulus):
    assert MeshUtils.compute_nodal_volumes(annulus).sum() == pytest.approx(1.0 - 0.04, rel=1e-12)


def test_nodal_volumes_sum_to_the_volume_in_3d(cube):
    assert MeshUtils.compute_nodal_volumes(cube).sum() == pytest.approx(1.0, rel=1e-12)


def test_nodal_volumes_sum_to_the_length_in_1d(line):
    assert MeshUtils.compute_nodal_volumes(line).sum() == pytest.approx(2.0, rel=1e-12)


def test_every_node_carries_volume(square):
    vols = MeshUtils.compute_nodal_volumes(square)
    assert np.all(vols > 0.0)  # every node belongs to at least one element


def test_unsupported_dimension_is_rejected(square):
    bogus = dict(square, dimension=4)
    with pytest.raises(ValueError, match="Unsupported dimension"):
        MeshUtils.compute_nodal_volumes(bogus)
    with pytest.raises(ValueError, match="Unsupported dimension"):
        MeshUtils.compute_nodal_ds(bogus)


# --------------------------------------------------------------------------------------------
# point-in-polygon
# --------------------------------------------------------------------------------------------
#: unit square with a square hole, as explicit points and edges
PIP_POINTS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]])
PIP_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]])


def test_points_in_polygon_respects_the_hole():
    pts = np.array(
        [
            [0.1, 0.1],  # in the material
            [0.1, 0.5],  # in the material, ray crosses the hole
            [0.5, 0.5],  # inside the HOLE -- not material
            [1.5, 0.5],  # outside altogether
            [-0.5, 0.5],  # outside, ray crosses everything
        ]
    )
    got = MeshUtils._points_in_polygon_2d(pts, PIP_EDGES, PIP_POINTS)
    assert list(got) == [True, True, False, False, False]


def test_points_in_polygon_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.2, 1.2, size=(97, 2))
    whole = MeshUtils._points_in_polygon_2d(pts, PIP_EDGES, PIP_POINTS)
    chunked = MeshUtils._points_in_polygon_2d(pts, PIP_EDGES, PIP_POINTS, block=1)
    assert np.array_equal(whole, chunked)


def test_points_in_polygon_without_edges_assumes_inside():
    pts = np.array([[0.5, 0.5], [9.0, 9.0]])
    assert np.all(MeshUtils._points_in_polygon_2d(pts, None, PIP_POINTS))
    assert np.all(MeshUtils._points_in_polygon_2d(pts, np.empty((0, 2), dtype=int), PIP_POINTS))


def test_points_in_polygon_ignores_horizontal_edges():
    # a horizontal edge can never be crossed by a horizontal ray; it must not divide by its own
    # zero height either
    flat = np.array([[0, 1], [2, 3]])  # the two horizontal edges of the outer square
    got = MeshUtils._points_in_polygon_2d(np.array([[0.5, 0.5]]), flat, PIP_POINTS)
    assert got.tolist() == [False]  # no crossable edge -> zero crossings -> outside


def test_points_in_polygon_accepts_a_single_point():
    assert MeshUtils._points_in_polygon_2d(np.array([0.1, 0.1]), PIP_EDGES, PIP_POINTS).tolist() == [True]


# --------------------------------------------------------------------------------------------
# boundary-edge extraction -- the measures and the normals are both built on it
# --------------------------------------------------------------------------------------------
def _closes_into_loops(edges):
    """A boundary is a union of closed loops, so every node on it has exactly two edges."""
    _, counts = np.unique(np.asarray(edges).ravel(), return_counts=True)
    return set(counts.tolist()) == {2}


@pytest.mark.parametrize("geom", [box(0.0, 0.0, 1.0, 1.0), SQUARE_HOLE])
def test_extracted_boundary_edges_close_into_loops(geom):
    mesh = jno.domain(geom, mesh_size=0.06).mesh
    edges = np.asarray(MeshUtils.extract_boundary_edges(mesh.cells_dict["triangle"], len(mesh.points)))
    assert len(edges) > 0
    assert _closes_into_loops(edges)


def test_extracted_boundary_edges_are_exactly_the_once_used_ones():
    tris = np.array([[0, 1, 2], [1, 3, 2]])  # two triangles sharing edge (1, 2)
    edges = np.asarray(MeshUtils.extract_boundary_edges(tris, 4))
    assert {tuple(e) for e in edges.tolist()} == {(0, 1), (0, 2), (1, 3), (2, 3)}  # the shared edge is interior


def test_volume_boundary_returns_plain_int_tuples():
    """``_chain_edges_to_loops`` consumes these as dictionary keys."""
    from jno.domain.domain_class import domain

    edges = domain._extract_volume_boundary(np.array([[0, 1, 2], [1, 3, 2]]))
    assert set(edges) == {(0, 1), (0, 2), (1, 3), (2, 3)}
    assert all(isinstance(v, int) for e in edges for v in e)
