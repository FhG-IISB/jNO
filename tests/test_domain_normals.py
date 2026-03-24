import numpy as np

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
