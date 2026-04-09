from __future__ import annotations

import numpy as np
import meshio

# Ordinal labels used for boundary segment naming.
_ORDINALS = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]


def _ensure_ccw(vertices):
    """Return *vertices* in counter-clockwise order (positive signed area)."""
    v = np.asarray(vertices, dtype=np.float64)
    # Shoelace signed area
    xs, ys = v[:, 0], v[:, 1]
    area2 = np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys)
    if area2 < 0:
        v = v[::-1]
    return v


def _order_from_top_right(vertices):
    """Return an index permutation that starts at the topmost vertex and
    proceeds clockwise (i.e. the next vertex is to the right).

    The caller already guarantees counter-clockwise winding, so "starting from
    the top going right" means we find the top-most vertex (highest y, break
    ties with rightmost x) and then *reverse* the vertex order so the
    traversal goes clockwise.
    """
    v = np.asarray(vertices, dtype=np.float64)
    # Pick top-most vertex; break ties by rightmost x.
    start = int(np.lexsort((v[:, 0] * -1, v[:, 1] * -1))[0])
    n = len(v)
    # CCW order starting from *start*, then reversed → clockwise from top.
    ccw_order = [(start + i) % n for i in range(n)]
    cw_order = [ccw_order[0]] + ccw_order[1:][::-1]
    return cw_order


class Geometries:

    @staticmethod
    def line(x_range=(0, 1), mesh_size=0.1):
        """Create a 1D line domain."""

        def constructor(geo):
            x0, x1 = x_range
            p0 = geo.add_point([x0, 0], mesh_size=mesh_size)
            p1 = geo.add_point([x1, 0], mesh_size=mesh_size)
            line = geo.add_line(p0, p1)

            geo.add_physical(line, "interior")
            geo.add_physical([p0], "left")
            geo.add_physical([p1], "right")
            geo.add_physical([p0, p1], "boundary")

            return geo, 1, mesh_size

        return constructor

    @staticmethod
    def polygon(vertices, mesh_size=0.1, _aliases=None):
        """Create a 2D polygon domain from an arbitrary list of vertices.

        Vertices are automatically oriented to counter-clockwise winding.
        Boundary segments are labeled ``"one"``, ``"two"``, … starting from
        the topmost vertex and proceeding clockwise (to the right).

        Parameters
        ----------
        vertices : sequence of (x, y) pairs
            At least 3 corner points defining the polygon.
        mesh_size : float
            Target mesh element size.
        """

        def constructor(geo):
            verts = np.asarray(vertices, dtype=np.float64)
            if verts.ndim != 2 or verts.shape[0] < 3 or verts.shape[1] != 2:
                raise ValueError(f"Expected >= 3 vertices of shape (N, 2), got {verts.shape}")

            verts = _ensure_ccw(verts)
            label_order = _order_from_top_right(verts)
            n = len(verts)

            points = [geo.add_point([verts[i, 0], verts[i, 1]], mesh_size=mesh_size) for i in range(n)]
            lines = [geo.add_line(points[i], points[(i + 1) % n]) for i in range(n)]

            curve_loop = geo.add_curve_loop(lines)
            surface = geo.add_plane_surface(curve_loop)

            geo.add_physical(surface, "interior")
            geo.add_physical(lines, "boundary")

            # Label each segment starting from the top-right vertex, going clockwise.
            for label_idx, vi in enumerate(label_order):
                name = _ORDINALS[label_idx] if label_idx < len(_ORDINALS) else str(label_idx + 1)
                geo.add_physical([lines[vi]], name)
                # Add backward-compatible aliases (e.g. "top", "right", …)
                if _aliases and label_idx in _aliases:
                    geo.add_physical([lines[vi]], _aliases[label_idx])

            return geo, 2, mesh_size

        return constructor

    @staticmethod
    def rect(x_range=(0, 1), y_range=(0, 1), mesh_size=0.1):
        """Create a rectangular domain as a polygon.

        Boundary labels: ``"one"`` / ``"top"``, ``"two"`` / ``"right"``,
        ``"three"`` / ``"bottom"``, ``"four"`` / ``"left"``.
        """
        x0, x1 = x_range
        y0, y1 = y_range
        return Geometries.polygon(
            vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            mesh_size=mesh_size,
            _aliases={0: "top", 1: "right", 2: "bottom", 3: "left"},
        )

    @staticmethod
    def triangle(vertices=((0, 0), (1, 0), (0, 1)), mesh_size=0.1):
        """Create a triangular domain as a polygon.

        Parameters
        ----------
        vertices : sequence of 3 (x, y) pairs
            The three corner points.
        mesh_size : float
            Mesh element size.
        """
        v = list(vertices)
        if len(v) != 3:
            raise ValueError(f"Triangle requires exactly 3 vertices, got {len(v)}")
        return Geometries.polygon(vertices=v, mesh_size=mesh_size)

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

            return mesh, 2, min((x_range[1] - x_range[0]) / nx, (y_range[1] - y_range[0]) / ny)

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

    @staticmethod
    def cube(x_range=(0, 1), y_range=(0, 1), z_range=(0, 1), mesh_size=0.1):
        """Create a cubic domain for pygmsh."""

        def construct(geo):
            x0, x1 = x_range
            y0, y1 = y_range
            z0, z1 = z_range

            # Create 8 corner points
            points = [
                geo.add_point([x0, y0, z0], mesh_size=mesh_size),  # 0: bottom-left-front
                geo.add_point([x1, y0, z0], mesh_size=mesh_size),  # 1: bottom-right-front
                geo.add_point([x1, y1, z0], mesh_size=mesh_size),  # 2: bottom-right-back
                geo.add_point([x0, y1, z0], mesh_size=mesh_size),  # 3: bottom-left-back
                geo.add_point([x0, y0, z1], mesh_size=mesh_size),  # 4: top-left-front
                geo.add_point([x1, y0, z1], mesh_size=mesh_size),  # 5: top-right-front
                geo.add_point([x1, y1, z1], mesh_size=mesh_size),  # 6: top-right-back
                geo.add_point([x0, y1, z1], mesh_size=mesh_size),  # 7: top-left-back
            ]

            # Create lines for each edge of the cube (12 edges)
            # Bottom face (z=z0)
            lines_bottom = [
                geo.add_line(points[0], points[1]),  # front
                geo.add_line(points[1], points[2]),  # right
                geo.add_line(points[2], points[3]),  # back
                geo.add_line(points[3], points[0]),  # left
            ]

            # Top face (z=z1)
            lines_top = [
                geo.add_line(points[4], points[5]),  # front
                geo.add_line(points[5], points[6]),  # right
                geo.add_line(points[6], points[7]),  # back
                geo.add_line(points[7], points[4]),  # left
            ]

            # Vertical edges
            lines_vertical = [
                geo.add_line(points[0], points[4]),  # front-left
                geo.add_line(points[1], points[5]),  # front-right
                geo.add_line(points[2], points[6]),  # back-right
                geo.add_line(points[3], points[7]),  # back-left
            ]

            # Create curve loops for each face
            loop_bottom = geo.add_curve_loop(lines_bottom)
            loop_top = geo.add_curve_loop(lines_top)
            loop_front = geo.add_curve_loop([lines_bottom[0], lines_vertical[1], -lines_top[0], -lines_vertical[0]])
            loop_right = geo.add_curve_loop([lines_bottom[1], lines_vertical[2], -lines_top[1], -lines_vertical[1]])
            loop_back = geo.add_curve_loop([lines_bottom[2], lines_vertical[3], -lines_top[2], -lines_vertical[2]])
            loop_left = geo.add_curve_loop([lines_bottom[3], lines_vertical[0], -lines_top[3], -lines_vertical[3]])

            # Create surfaces for each face
            surface_bottom = geo.add_plane_surface(loop_bottom)
            surface_top = geo.add_plane_surface(loop_top)
            surface_front = geo.add_plane_surface(loop_front)
            surface_right = geo.add_plane_surface(loop_right)
            surface_back = geo.add_plane_surface(loop_back)
            surface_left = geo.add_plane_surface(loop_left)

            # Create surface loop and volume
            surface_loop = geo.add_surface_loop([surface_bottom, surface_top, surface_front, surface_right, surface_back, surface_left])
            volume = geo.add_volume(surface_loop)

            # Add physical groups
            geo.add_physical(volume, "interior")
            geo.add_physical([surface_bottom, surface_top, surface_front, surface_right, surface_back, surface_left], "boundary")
            geo.add_physical([surface_bottom], "bottom")
            geo.add_physical([surface_top], "top")
            geo.add_physical([surface_front], "front")
            geo.add_physical([surface_back], "back")
            geo.add_physical([surface_left], "left")
            geo.add_physical([surface_right], "right")

            return geo, 3, mesh_size

        return construct

    @staticmethod
    def disk(center=(0, 0), radius=1.0, mesh_size=0.1, num_points=32):
        """Create a circular domain using polygon approximation."""

        def constructor(geo):
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            cx, cy = center
            points = [geo.add_point([cx + radius * np.cos(a), cy + radius * np.sin(a)], mesh_size=mesh_size) for a in angles]

            lines = []
            for i in range(num_points):
                lines.append(geo.add_line(points[i], points[(i + 1) % num_points]))

            loop = geo.add_curve_loop(lines)
            surface = geo.add_plane_surface(loop)

            geo.add_physical(surface, "interior")
            geo.add_physical(lines, "boundary")

            return geo, 2, mesh_size

        return constructor

    @staticmethod
    def l_shape(size=1.0, mesh_size=0.1, separate_boundary=False):
        """Create an L-shaped domain."""

        def constructor(geo):
            points = [
                geo.add_point([0, 0], mesh_size=mesh_size),
                geo.add_point([size, 0], mesh_size=mesh_size),
                geo.add_point([size, size / 2], mesh_size=mesh_size),
                geo.add_point([size / 2, size / 2], mesh_size=mesh_size),
                geo.add_point([size / 2, size], mesh_size=mesh_size),
                geo.add_point([0, size], mesh_size=mesh_size),
            ]

            lines = [
                geo.add_line(points[0], points[1]),
                geo.add_line(points[1], points[2]),
                geo.add_line(points[2], points[3]),
                geo.add_line(points[3], points[4]),
                geo.add_line(points[4], points[5]),
                geo.add_line(points[5], points[0]),
            ]

            curve_loop = geo.add_curve_loop(lines)
            surface = geo.add_plane_surface(curve_loop)

            geo.add_physical(surface, "interior")
            geo.add_physical(lines, "boundary")

            if separate_boundary:
                geo.add_physical([lines[0]], "bottom")
                geo.add_physical([lines[1]], "right_lower")
                geo.add_physical([lines[2]], "inner_horizontal")
                geo.add_physical([lines[3]], "inner_vertical")
                geo.add_physical([lines[4]], "top")
                geo.add_physical([lines[5]], "left")

            return geo, 2, mesh_size

        return constructor

    @staticmethod
    def rectangle_with_hole(outer_size=1.0, hole_size=0.4, mesh_size=0.1, separate_boundary=False):
        """Create a rectangular domain with a rectangular hole in the middle.

        Parameters
        ----------
        outer_size : float
            Size of the outer rectangle (width and height).
        hole_size : float
            Size of the inner rectangular hole (width and height).
        mesh_size : float
            Mesh element size.
        separate_boundary : bool
            If True, assign separate physical groups to each boundary segment.
        """

        def constructor(geo):
            # Outer rectangle points (counter-clockwise)
            outer_points = [
                geo.add_point([0, 0], mesh_size=mesh_size),
                geo.add_point([outer_size, 0], mesh_size=mesh_size),
                geo.add_point([outer_size, outer_size], mesh_size=mesh_size),
                geo.add_point([0, outer_size], mesh_size=mesh_size),
            ]

            # Inner rectangle (hole) points (clockwise for hole)
            # Centered in the middle
            hole_offset = (outer_size - hole_size) / 2
            inner_points = [
                geo.add_point([hole_offset, hole_offset], mesh_size=mesh_size),
                geo.add_point([hole_offset + hole_size, hole_offset], mesh_size=mesh_size),
                geo.add_point([hole_offset + hole_size, hole_offset + hole_size], mesh_size=mesh_size),
                geo.add_point([hole_offset, hole_offset + hole_size], mesh_size=mesh_size),
            ]

            # Outer boundary lines (counter-clockwise)
            outer_lines = [
                geo.add_line(outer_points[0], outer_points[1]),  # bottom
                geo.add_line(outer_points[1], outer_points[2]),  # right
                geo.add_line(outer_points[2], outer_points[3]),  # top
                geo.add_line(outer_points[3], outer_points[0]),  # left
            ]

            # Inner boundary lines (clockwise for hole)
            inner_lines = [
                geo.add_line(inner_points[0], inner_points[1]),  # hole_bottom
                geo.add_line(inner_points[1], inner_points[2]),  # hole_right
                geo.add_line(inner_points[2], inner_points[3]),  # hole_top
                geo.add_line(inner_points[3], inner_points[0]),  # hole_left
            ]

            # Create curve loops
            outer_loop = geo.add_curve_loop(outer_lines)
            inner_loop = geo.add_curve_loop(inner_lines)

            # Create surface with hole (outer loop first, then hole loop)
            surface = geo.add_plane_surface(outer_loop, holes=[inner_loop])

            # Physical groups
            geo.add_physical(surface, "interior")
            geo.add_physical(outer_lines + inner_lines, "boundary")

            if separate_boundary:
                # Outer boundary
                geo.add_physical([outer_lines[0]], "bottom")
                geo.add_physical([outer_lines[1]], "right")
                geo.add_physical([outer_lines[2]], "top")
                geo.add_physical([outer_lines[3]], "left")
                # Inner boundary (hole)
                geo.add_physical([inner_lines[0]], "_bottom")
                geo.add_physical([inner_lines[1]], "_right")
                geo.add_physical([inner_lines[2]], "_top")
                geo.add_physical([inner_lines[3]], "_left")
                # Group all hole boundaries together as well
                geo.add_physical(inner_lines, "_boundary")

            return geo, 2, mesh_size

        return constructor

    @staticmethod
    def rect_pml(
        x_range=(0, 1),
        y_range=(0, 1),
        mesh_size=0.1,
        pml_thickness_top=0.2,
        pml_thickness_bottom=0.2,
    ):
        """Rectangle with top and bottom PMLs (pygmsh-compatible)."""

        def construct(geo):
            x0, x1 = x_range
            y0, y1 = y_range

            yb = y0 - pml_thickness_bottom
            yt = y1 + pml_thickness_top

            # ------------------------------------------------------------
            # Points
            # ------------------------------------------------------------
            p00 = geo.add_point([x0, y0], mesh_size)
            p10 = geo.add_point([x1, y0], mesh_size)
            p11 = geo.add_point([x1, y1], mesh_size)
            p01 = geo.add_point([x0, y1], mesh_size)

            pb0 = geo.add_point([x0, yb], mesh_size)
            pb1 = geo.add_point([x1, yb], mesh_size)

            pt1 = geo.add_point([x1, yt], mesh_size)
            pt0 = geo.add_point([x0, yt], mesh_size)

            # ------------------------------------------------------------
            # Interior rectangle
            # ------------------------------------------------------------
            l0 = geo.add_line(p00, p10)  # bottom
            l1 = geo.add_line(p10, p11)  # right
            l2 = geo.add_line(p11, p01)  # top
            l3 = geo.add_line(p01, p00)  # left

            s_int = geo.add_plane_surface(geo.add_curve_loop([l0, l1, l2, l3]))

            # ------------------------------------------------------------
            # Bottom PML (CCW)
            # ------------------------------------------------------------
            lb0 = geo.add_line(pb0, pb1)
            lb1 = geo.add_line(pb1, p10)
            lb2 = geo.add_line(p00, pb0)

            s_pb = geo.add_plane_surface(
                geo.add_curve_loop(
                    [
                        lb0,  # pb0 -> pb1
                        lb1,  # pb1 -> p10
                        -l0,  # p10 -> p00  (reverse interior bottom)
                        lb2,  # p00 -> pb0
                    ]
                )
            )

            # ------------------------------------------------------------

            # Top PML (CCW)

            # ------------------------------------------------------------

            lt0 = geo.add_line(p11, pt1)  # p11 → pt1
            lt1 = geo.add_line(pt1, pt0)  # pt1 → pt0
            lt2 = geo.add_line(pt0, p01)  # pt0 → p01

            s_pt = geo.add_plane_surface(
                geo.add_curve_loop(
                    [
                        -l2,  # p01 → p11  (reversed: l2 was p11 → p01)
                        lt0,  # p11 → pt1
                        lt1,  # pt1 → pt0
                        lt2,  # pt0 → p01
                    ]
                )
            )

            # ------------------------------------------------------------
            # Physical groups
            # ------------------------------------------------------------
            geo.add_physical(s_int, "interior")
            geo.add_physical(s_pb, "pml_bottom")
            geo.add_physical(s_pt, "pml_top")

            geo.add_physical([l0, l1, l2, l3], "boundary")
            geo.add_physical([l0], "bottom")
            geo.add_physical([l2], "top")
            geo.add_physical([l1], "right")
            geo.add_physical([l3], "left")

            return geo, 2, mesh_size

        return construct

    @staticmethod
    def rectangle_with_holes(outer_size=(2.0, 1.0), holes=None, mesh_size=0.1, separate_boundary=True):
        """Create a rectangular domain with multiple rectangular holes.

        Parameters
        ----------
        outer_size : tuple
            (width, height) of the outer rectangle.
        holes : list of dict
            List of hole specifications. Each dict should contain:
            - 'origin': (x, y) bottom-left corner of the hole
            - 'size': (width, height) of the hole
            - 'type': str, name for this hole's boundary (e.g., 'hole1', 'heater')

        mesh_size : float
            Mesh element size.
        separate_boundary : bool
            If True, assign separate physical groups to each boundary segment.

        Example
        -------
        >>> holes = [
        ...     {'origin': (0.3, 0.3), 'size': (0.2, 0.2), 'type': 'obstacle'},
        ...     {'origin': (0.7, 0.3), 'size': (0.15, 0.3), 'type': 'heater'},
        ... ]
        >>> geom = Geometries.rectangle_with_holes(
        ...     outer_size=(1.5, 1.0),
        ...     holes=holes,
        ...     mesh_size=0.05
        ... )
        """

        if holes is None:
            holes = []

        def constructor(geo):
            outer_w, outer_h = outer_size

            # =================================================================
            # Outer rectangle points (counter-clockwise)
            # =================================================================
            outer_points = [
                geo.add_point([0, 0], mesh_size=mesh_size),
                geo.add_point([outer_w, 0], mesh_size=mesh_size),
                geo.add_point([outer_w, outer_h], mesh_size=mesh_size),
                geo.add_point([0, outer_h], mesh_size=mesh_size),
            ]

            # Outer boundary lines (counter-clockwise)
            outer_lines = [
                geo.add_line(outer_points[0], outer_points[1]),  # bottom
                geo.add_line(outer_points[1], outer_points[2]),  # right
                geo.add_line(outer_points[2], outer_points[3]),  # top
                geo.add_line(outer_points[3], outer_points[0]),  # left
            ]

            # Create outer loop
            outer_loop = geo.add_curve_loop(outer_lines)

            # =================================================================
            # Create holes
            # =================================================================
            hole_loops = []
            hole_data = []  # Store (lines, type) for each hole

            for hole_spec in holes:
                origin = hole_spec["origin"]
                size = hole_spec["size"]
                hole_type = hole_spec.get("type", "hole")

                x0, y0 = origin
                w, h = size

                # Hole corner points (counter-clockwise)
                hole_points = [
                    geo.add_point([x0, y0], mesh_size=mesh_size),  # bottom-left
                    geo.add_point([x0 + w, y0], mesh_size=mesh_size),  # bottom-right
                    geo.add_point([x0 + w, y0 + h], mesh_size=mesh_size),  # top-right
                    geo.add_point([x0, y0 + h], mesh_size=mesh_size),  # top-left
                ]

                # Hole boundary lines (counter-clockwise)
                hole_lines = [
                    geo.add_line(hole_points[0], hole_points[1]),  # bottom
                    geo.add_line(hole_points[1], hole_points[2]),  # right
                    geo.add_line(hole_points[2], hole_points[3]),  # top
                    geo.add_line(hole_points[3], hole_points[0]),  # left
                ]

                # Create hole loop
                hole_loop = geo.add_curve_loop(hole_lines)
                hole_loops.append(hole_loop)
                hole_data.append((hole_lines, hole_type))

            # =================================================================
            # Create surface with all holes
            # =================================================================
            surface = geo.add_plane_surface(outer_loop, holes=hole_loops)

            # =================================================================
            # Physical groups
            # =================================================================

            # Interior surface
            geo.add_physical(surface, "interior")

            # Outer boundary (all outer lines)
            geo.add_physical(outer_lines, "boundary")

            # Collect all hole lines for a combined hole boundary
            all_hole_lines = []
            for hole_lines, _ in hole_data:
                all_hole_lines.extend(hole_lines)

            if all_hole_lines:
                geo.add_physical(all_hole_lines, "hole_boundary")

            if separate_boundary:
                # Outer boundary segments
                geo.add_physical([outer_lines[0]], "bottom")
                geo.add_physical([outer_lines[1]], "right")
                geo.add_physical([outer_lines[2]], "top")
                geo.add_physical([outer_lines[3]], "left")

                # Each hole gets its own boundary groups
                for hole_lines, hole_type in hole_data:
                    # Combined boundary for this hole
                    geo.add_physical(hole_lines, f"{hole_type}_boundary")

                    # Individual sides
                    geo.add_physical([hole_lines[0]], f"{hole_type}_bottom")
                    geo.add_physical([hole_lines[1]], f"{hole_type}_right")
                    geo.add_physical([hole_lines[2]], f"{hole_type}_top")
                    geo.add_physical([hole_lines[3]], f"{hole_type}_left")

            return geo, 2, mesh_size

        return constructor
