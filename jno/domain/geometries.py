from __future__ import annotations

import meshio
import numpy as np


class Geometries:
    @staticmethod
    def line(x_range=(0, 1), mesh_size=0.1):
        """Create a 1D line domain.

        Built directly as a two-block ``meshio.Mesh`` (a ``line`` volume block + a
        ``vertex`` boundary block for the two endpoints), rather than via pygmsh: a
        pygmsh 0-D *point* physical group round-trips to a malformed scalar
        ``cell_set`` with no vertex block, so ``variable("left"/"right"/"boundary")``
        could not resolve. The explicit two-block form is exactly the contract
        ``_extract_points_from_mesh`` consumes (endpoints named ``left``/``right``).
        """

        def constructor(geo):
            import meshio

            x0, x1 = float(x_range[0]), float(x_range[1])
            n = max(int(round(abs(x1 - x0) / float(mesh_size))), 1)  # number of segments
            xs = np.linspace(x0, x1, n + 1)
            points = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])
            lines = np.column_stack([np.arange(n), np.arange(1, n + 1)]).astype(np.int64)
            verts = np.array([[0], [n]], dtype=np.int64)  # the two endpoint nodes
            empty = np.array([], dtype=np.int64)
            mesh = meshio.Mesh(
                points=points,
                cells=[("line", lines), ("vertex", verts)],
                cell_sets={
                    "interior": [np.arange(n, dtype=np.int64), empty.copy()],
                    "left": [empty.copy(), np.array([0], dtype=np.int64)],
                    "right": [empty.copy(), np.array([1], dtype=np.int64)],
                    "boundary": [empty.copy(), np.array([0, 1], dtype=np.int64)],
                },
            )
            return mesh, 1, mesh_size

        return constructor

    @staticmethod
    def equi_distant_rect(x_range=(0, 1), y_range=(0, 1), nx=10, ny=10):
        """
        Create a structured triangular mesh with proper cell_sets for boundary extraction.

        Returns:
            meshio.Mesh with cell_sets for boundaries and domain
        """

        def constructor(geo):
            x0 = x_range[0]
            x1 = x_range[1]

            y0 = y_range[0]
            y1 = y_range[1]

            # Create structured grid points
            x = np.linspace(x0, x1, nx + 1)
            y = np.linspace(y0, y1, ny + 1)
            xx, yy = np.meshgrid(x, y, indexing="ij")

            # Flatten to create points array
            points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros((nx + 1) * (ny + 1))])

            # Helper function to get point index
            def idx(i, j):
                return i * (ny + 1) + j

            # =========================================================================
            # Create triangles (2D cells)
            # =========================================================================
            triangles = []
            for i in range(nx):
                for j in range(ny):
                    p0 = idx(i, j)
                    p1 = idx(i + 1, j)
                    p2 = idx(i + 1, j + 1)
                    p3 = idx(i, j + 1)

                    triangles.append([p0, p1, p2])
                    triangles.append([p0, p2, p3])

            triangles = np.array(triangles)

            # =========================================================================
            # Create boundary edges (1D cells)
            # =========================================================================
            bottom_edges = []
            top_edges = []
            left_edges = []
            right_edges = []

            # Bottom boundary (j = 0)
            for i in range(nx):
                bottom_edges.append([idx(i, 0), idx(i + 1, 0)])

            # Top boundary (j = ny)
            for i in range(nx):
                top_edges.append([idx(i, ny), idx(i + 1, ny)])

            # Left boundary (i = 0)
            for j in range(ny):
                left_edges.append([idx(0, j), idx(0, j + 1)])

            # Right boundary (i = nx)
            for j in range(ny):
                right_edges.append([idx(nx, j), idx(nx, j + 1)])

            bottom_edges = np.array(bottom_edges)
            top_edges = np.array(top_edges)
            left_edges = np.array(left_edges)
            right_edges = np.array(right_edges)

            # Combine all edges into one array
            all_edges = np.vstack([bottom_edges, top_edges, left_edges, right_edges])

            # Track indices within the combined edge array
            n_bottom = len(bottom_edges)
            n_top = len(top_edges)
            n_left = len(left_edges)
            n_right = len(right_edges)

            bottom_indices = np.arange(0, n_bottom)
            top_indices = np.arange(n_bottom, n_bottom + n_top)
            left_indices = np.arange(n_bottom + n_top, n_bottom + n_top + n_left)
            right_indices = np.arange(n_bottom + n_top + n_left, n_bottom + n_top + n_left + n_right)
            all_boundary_indices = np.arange(len(all_edges))

            # =========================================================================
            # Create cells list
            # =========================================================================
            cells = [
                ("triangle", triangles),  # Block 0: triangles
                ("line", all_edges),  # Block 1: all boundary edges
            ]

            # =========================================================================
            # Create cell_sets
            # cell_sets format: {name: [array_for_block_0, array_for_block_1, ...]}
            # Each array contains indices of cells within that block
            # =========================================================================
            cell_sets = {
                # domain (all triangles)
                "interior": [
                    np.arange(len(triangles)),  # Block 0: all triangle indices
                    np.array([], dtype=np.int64),  # Block 1: no edges
                ],
                # Boundary edges
                "bottom": [
                    np.array([], dtype=np.int64),  # Block 0: no triangles
                    bottom_indices,  # Block 1: bottom edge indices
                ],
                "top": [
                    np.array([], dtype=np.int64),
                    top_indices,
                ],
                "left": [
                    np.array([], dtype=np.int64),
                    left_indices,
                ],
                "right": [
                    np.array([], dtype=np.int64),
                    right_indices,
                ],
                "boundary": [
                    np.array([], dtype=np.int64),
                    all_boundary_indices,
                ],
            }

            # =========================================================================
            # Create mesh
            # =========================================================================
            mesh = meshio.Mesh(
                points=points,
                cells=cells,
                cell_sets=cell_sets,
            )

            return (
                mesh,
                2,
                min((x_range[1] - x_range[0]) / nx, (y_range[1] - y_range[0]) / ny),
            )

        return constructor

    @staticmethod
    def poseidon(nx: int = 128, ny: int = 128):
        """
        Create a structured 2-D grid for foundation models (Poseidon, Walrus, …).

        The grid has exactly ``nx × ny`` vertices on [0, 1]×[0, 1], matching
        the pixel resolution that these models expect.  Triangulation and
        boundary edge connectivity are built so that ``scheme='finite_difference'``
        works out of the box with ``jnn.laplacian`` / ``jnn.grad``.

        Args:
            nx: Number of grid points along x.  Default 128.
            ny: Number of grid points along y.  Default 128.
        """
        x_range = (0, 1)
        y_range = (0, 1)

        def constructor(geo):
            x0 = x_range[0]
            x1 = x_range[1]

            y0 = y_range[0]
            y1 = y_range[1]

            # Create structured grid points — exactly nx × ny vertices
            x = np.linspace(x0, x1, nx)
            y = np.linspace(y0, y1, ny)
            xx, yy = np.meshgrid(x, y, indexing="ij")

            # Flatten to create points array (N = nx*ny, 3)
            points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])

            # Helper function to get point index
            def idx(i, j):
                return i * ny + j

            # =========================================================================
            # Create triangles (2D cells) — (nx-1)*(ny-1)*2 triangles
            # =========================================================================
            triangles = []
            for i in range(nx - 1):
                for j in range(ny - 1):
                    p0 = idx(i, j)
                    p1 = idx(i + 1, j)
                    p2 = idx(i + 1, j + 1)
                    p3 = idx(i, j + 1)

                    triangles.append([p0, p1, p2])
                    triangles.append([p0, p2, p3])

            triangles = np.array(triangles)

            # =========================================================================
            # Create boundary edges (1D cells)
            # =========================================================================
            bottom_edges = []
            top_edges = []
            left_edges = []
            right_edges = []

            # Bottom boundary (j = 0)
            for i in range(nx - 1):
                bottom_edges.append([idx(i, 0), idx(i + 1, 0)])

            # Top boundary (j = ny - 1)
            for i in range(nx - 1):
                top_edges.append([idx(i, ny - 1), idx(i + 1, ny - 1)])

            # Left boundary (i = 0)
            for j in range(ny - 1):
                left_edges.append([idx(0, j), idx(0, j + 1)])

            # Right boundary (i = nx - 1)
            for j in range(ny - 1):
                right_edges.append([idx(nx - 1, j), idx(nx - 1, j + 1)])

            bottom_edges = np.array(bottom_edges)
            top_edges = np.array(top_edges)
            left_edges = np.array(left_edges)
            right_edges = np.array(right_edges)

            # Combine all edges into one array
            all_edges = np.vstack([bottom_edges, top_edges, left_edges, right_edges])

            # Track indices within the combined edge array
            n_bottom = len(bottom_edges)
            n_top = len(top_edges)
            n_left = len(left_edges)
            n_right = len(right_edges)

            bottom_indices = np.arange(0, n_bottom)
            top_indices = np.arange(n_bottom, n_bottom + n_top)
            left_indices = np.arange(n_bottom + n_top, n_bottom + n_top + n_left)
            right_indices = np.arange(n_bottom + n_top + n_left, n_bottom + n_top + n_left + n_right)
            all_boundary_indices = np.arange(len(all_edges))

            # =========================================================================
            # Create cells list
            # =========================================================================
            cells = [
                ("triangle", triangles),  # Block 0: triangles
                ("line", all_edges),  # Block 1: all boundary edges
            ]

            # =========================================================================
            # Create cell_sets
            # cell_sets format: {name: [array_for_block_0, array_for_block_1, ...]}
            # Each array contains indices of cells within that block
            # =========================================================================
            cell_sets = {
                # domain (all triangles)
                "interior": [
                    np.arange(len(triangles)),  # Block 0: all triangle indices
                    np.array([], dtype=np.int64),  # Block 1: no edges
                ],
                # Boundary edges
                "bottom": [
                    np.array([], dtype=np.int64),  # Block 0: no triangles
                    bottom_indices,  # Block 1: bottom edge indices
                ],
                "top": [
                    np.array([], dtype=np.int64),
                    top_indices,
                ],
                "left": [
                    np.array([], dtype=np.int64),
                    left_indices,
                ],
                "right": [
                    np.array([], dtype=np.int64),
                    right_indices,
                ],
                "boundary": [
                    np.array([], dtype=np.int64),
                    all_boundary_indices,
                ],
            }

            # =========================================================================
            # Create mesh
            # =========================================================================
            mesh = meshio.Mesh(
                points=points,
                cells=cells,
                cell_sets=cell_sets,
            )

            return mesh, 2, 0.1

        return constructor
