import numpy as np
import pytest
from shapely.geometry import box

import jno
from jno.domain.mesh_utils import MeshUtils


def test_compute_normals_from_boundary_faces_cube_outward_unit():
    # Unit cube vertices
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    # 12 boundary triangles (orientation may be arbitrary)
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )

    normals, boundary_indices = MeshUtils._compute_normals_from_boundary_faces(points, faces)

    assert normals.shape == (8, 3)
    assert boundary_indices.shape == (8,)

    # Unit-length normals
    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-7)

    # Outward check: dot with (vertex - centroid) should be positive.
    centroid = np.mean(points, axis=0)
    radial = points[boundary_indices] - centroid
    dots = np.sum(normals * radial, axis=1)
    assert np.all(dots > 0.0)


@pytest.fixture(scope="module")
def annulus_normals():
    """PCA normals on a unit square with a central square hole."""
    mesh = jno.domain(box(0.0, 0.0, 1.0, 1.0).difference(box(0.4, 0.4, 0.6, 0.6)), mesh_size=0.05).mesh
    boundary_indices = np.unique(MeshUtils._get_boundary_elements(mesh.cells_dict["triangle"], "triangle"))
    normals, idx = MeshUtils._compute_normals_pca(mesh.points, boundary_indices, 2, mesh=mesh)
    return normals, mesh.points[idx, :2]


def test_pca_normals_are_unit_length(annulus_normals):
    normals, _ = annulus_normals
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)


def test_pca_normals_point_out_of_the_material_on_both_boundaries(annulus_normals):
    """The hole is what makes this a real test.

    A normal points OUT of the material, so on the hole it points INWARD, toward the domain
    centre -- the opposite of the outer boundary. The centroid heuristic alone gets that backwards;
    only the point-in-polygon test distinguishes the two boundaries.
    """
    normals, xy = annulus_normals
    on_hole = np.all((xy > 0.39) & (xy < 0.61), axis=1)
    assert on_hole.sum() > 0 and (~on_hole).sum() > 0

    radial = np.einsum("ij,ij->i", normals, xy - 0.5)
    assert np.all(radial[~on_hole] > 0.0)  # outer boundary: away from the centre
    assert np.all(radial[on_hole] < 0.0)  # hole boundary: toward the centre
