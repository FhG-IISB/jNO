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
    def equi_distant_rect(x_range=(0, 1), y_range=(0, 1), nx=10, ny=10, cell="triangle"):
        """Structured 2-D mesh of a rectangle, with ``cell_sets`` for boundary extraction.

        Args:
            cell: ``"triangle"`` (default) splits every grid rectangle into two triangles;
                ``"quad"`` keeps it whole, giving ``nx*ny`` quadrilaterals. The grid nodes are
                identical either way — only the connectivity differs.

        Quadrilaterals are emitted in VTK/meshio order, counterclockwise from the lower-left
        corner ``(p00, p10, p11, p01)``, so ``det J > 0``. The boundary block stays 2-node
        ``line`` cells: a quad's facet is a straight edge, exactly as a triangle's is.

        Returns:
            ``constructor(geo) -> (meshio.Mesh, 2, ds)`` — the ``jno.domain`` geometry-func contract.
        """
        if cell not in ("triangle", "quad"):
            raise ValueError(f"equi_distant_rect(): cell={cell!r} is not a 2-D cell; use 'triangle' or 'quad'.")

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
            # Create volume cells (2D): two triangles per rectangle, or one quad
            # =========================================================================
            volume_cells = []
            for i in range(nx):
                for j in range(ny):
                    p0 = idx(i, j)
                    p1 = idx(i + 1, j)
                    p2 = idx(i + 1, j + 1)
                    p3 = idx(i, j + 1)

                    if cell == "quad":
                        volume_cells.append([p0, p1, p2, p3])  # ccw from lower-left
                    else:
                        volume_cells.append([p0, p1, p2])
                        volume_cells.append([p0, p2, p3])

            vcells = np.array(volume_cells, dtype=np.int64)

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
                (cell, vcells),  # Block 0: volume cells ("triangle" or "quad")
                ("line", all_edges),  # Block 1: all boundary edges
            ]

            # =========================================================================
            # Create cell_sets
            # cell_sets format: {name: [array_for_block_0, array_for_block_1, ...]}
            # Each array contains indices of cells within that block
            # =========================================================================
            cell_sets = {
                # domain (all volume cells)
                "interior": [
                    np.arange(len(vcells)),  # Block 0: all volume-cell indices
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
    def equi_distant_box(x_range=(0, 1), y_range=(0, 1), z_range=(0, 1), nx=8, ny=8, nz=8, cell="tetra"):
        """Structured 3-D mesh of a box with boundary-face ``cell_sets``
        (``left/right/bottom/top/front/back`` + ``boundary``, ``interior`` for the volume).

        Args:
            cell: ``"tetra"`` (default) splits every voxel into six Kuhn tets, all sharing the
                voxel's main diagonal so the subdivision is conforming across voxels; ``"hex"``
                keeps the voxel whole, giving ``nx*ny*nz`` hexahedra and **quadrilateral**
                boundary faces.

        Node order is ``idx(i, j, k) = (i·(ny+1) + j)·(nz+1) + k`` (C-order), so a nodal field reshapes
        cleanly to ``(nx+1, ny+1, nz+1)`` for the structured FD stencils.

        Hexahedra are emitted in VTK/meshio order — the bottom face counterclockwise seen from
        below-to-above (nodes 0-3) then the matching top face (4-7), so node ``k+4`` sits directly
        above node ``k`` and the cell is right-handed (positive volume).

        Returns:
            ``constructor(geo) -> (meshio.Mesh, 3, ds)`` — the ``jno.domain`` geometry-func contract.
        """
        if cell not in ("tetra", "hex"):
            raise ValueError(f"equi_distant_box(): cell={cell!r} is not a 3-D cell; use 'tetra' or 'hex'.")

        def constructor(geo):
            x0, x1 = x_range
            y0, y1 = y_range
            z0, z1 = z_range
            X = np.linspace(x0, x1, nx + 1)
            Y = np.linspace(y0, y1, ny + 1)
            Z = np.linspace(z0, z1, nz + 1)
            xx, yy, zz = np.meshgrid(X, Y, Z, indexing="ij")
            points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])  # C-order: idx below
            ny1, nz1 = ny + 1, nz + 1

            def idx(i, j, k):
                return (i * ny1 + j) * nz1 + k

            # Kuhn / Freudenthal split: 6 tets per voxel, all sharing the main diagonal 000→111.
            # Local corner c encodes (Δi, Δj, Δk) as bits (c>>2, c>>1, c) ∈ {0,1}³.
            kuhn = [(0, 4, 6, 7), (0, 4, 5, 7), (0, 2, 6, 7), (0, 2, 3, 7), (0, 1, 5, 7), (0, 1, 3, 7)]
            # VTK hexahedron order in the same local-corner encoding: the z=0 face counterclockwise
            # (as seen from +z) then the z=1 face directly above it.
            vtk_hex = (0, 4, 6, 2, 1, 5, 7, 3)
            tets = []
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        corner = [idx(i + ((c >> 2) & 1), j + ((c >> 1) & 1), k + (c & 1)) for c in range(8)]
                        if cell == "hex":
                            tets.append([corner[c] for c in vtk_hex])
                        else:
                            for a, b, cc, dd in kuhn:
                                tets.append([corner[a], corner[b], corner[cc], corner[dd]])
            tets = np.array(tets, dtype=np.int64)

            # Boundary faces: each box face is a quad grid. A tet mesh splits every quad into two
            # triangles; a hex mesh keeps it, since a hexahedron's facet IS a quadrilateral.
            def quad_tris(node, na, nb):
                tris = []
                for a in range(na):
                    for b in range(nb):
                        p00, p10, p11, p01 = node(a, b), node(a + 1, b), node(a + 1, b + 1), node(a, b + 1)
                        if cell == "hex":
                            tris.append([p00, p10, p11, p01])
                        else:
                            tris += [[p00, p10, p11], [p00, p11, p01]]
                return tris

            faces = {
                "left": quad_tris(lambda j, k: idx(0, j, k), ny, nz),
                "right": quad_tris(lambda j, k: idx(nx, j, k), ny, nz),
                "bottom": quad_tris(lambda i, k: idx(i, 0, k), nx, nz),
                "top": quad_tris(lambda i, k: idx(i, ny, k), nx, nz),
                "front": quad_tris(lambda i, j: idx(i, j, 0), nx, ny),
                "back": quad_tris(lambda i, j: idx(i, j, nz), nx, ny),
            }
            all_tris = []
            face_ranges = {}
            for name, tris in faces.items():
                start = len(all_tris)
                all_tris.extend(tris)
                face_ranges[name] = np.arange(start, len(all_tris), dtype=np.int64)
            all_tris = np.array(all_tris, dtype=np.int64)

            vol_block, face_block = ("hexahedron", "quad") if cell == "hex" else ("tetra", "triangle")
            cells = [(vol_block, tets), (face_block, all_tris)]
            empty = np.array([], dtype=np.int64)
            cell_sets = {"interior": [np.arange(len(tets), dtype=np.int64), empty]}
            cell_sets["boundary"] = [empty, np.arange(len(all_tris), dtype=np.int64)]
            for name, rng in face_ranges.items():
                cell_sets[name] = [empty, rng]

            mesh = meshio.Mesh(points=points, cells=cells, cell_sets=cell_sets)
            ds = min((x1 - x0) / nx, (y1 - y0) / ny, (z1 - z0) / nz)
            return mesh, 3, ds

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
