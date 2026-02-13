import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from scipy.spatial import KDTree
import jax.numpy as jnp
import jax
from dataclasses import dataclass
import meshio

from .trace import Variable, TensorTag
from .utils.logger import get_logger, Logger


@dataclass
class DomainData:
    """Pre-processed domain data for training."""

    points_by_tag: Dict[str, jax.Array]
    tensor_tags: Dict[str, jax.Array]
    dimension: int


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
    def rect(x_range=(0, 1), y_range=(0, 1), mesh_size=0.1):
        """Create a rectangular domain for pygmsh."""

        def construct(geo):
            x0, x1 = x_range
            y0, y1 = y_range

            points = [
                geo.add_point([x0, y0], mesh_size=mesh_size),
                geo.add_point([x1, y0], mesh_size=mesh_size),
                geo.add_point([x1, y1], mesh_size=mesh_size),
                geo.add_point([x0, y1], mesh_size=mesh_size),
            ]

            lines = [
                geo.add_line(points[0], points[1]),
                geo.add_line(points[1], points[2]),
                geo.add_line(points[2], points[3]),
                geo.add_line(points[3], points[0]),
            ]

            curve_loop = geo.add_curve_loop(lines)
            surface = geo.add_plane_surface(curve_loop)

            geo.add_physical(surface, "interior")
            geo.add_physical(lines, "boundary")
            geo.add_physical([lines[0]], "bottom")
            geo.add_physical([lines[1]], "right")
            geo.add_physical([lines[2]], "top")
            geo.add_physical([lines[3]], "left")

            return geo, 2, mesh_size

        return construct

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
    def poseidon():
        """
        Create a structured grid needed for using the poseidon foundation model
        """
        nx = 128
        ny = 128
        x_range = (0, 1)
        y_range = (0, 1)

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


class MeshUtils:

    @staticmethod
    def _preprocess_mesh_connectivity(mesh, dimension, boundary_indices):
        """Preprocess mesh to build FEM connectivity matrices for finite differences."""
        if mesh is None:
            return

        points = mesh.points[:, :dimension]
        n_points = len(points)

        if dimension == 1:
            # Check for line elements in the mesh
            if "line" not in mesh.cells_dict:
                raise ValueError("1D finite difference support requires line meshes")

            elements = mesh.cells_dict["line"]
            element_type = "lines"
            n_vertices_per_element = 2

            # Precompute 1D element lengths and shape function gradients
            length, grad_phi = MeshUtils.precompute_p1_line_geometry(points, elements)

            # Create all directed edges from line elements
            # Each line element [a, b] creates edges a->b and b->a
            edges = np.concatenate([elements[:, [0, 1]], elements[:, [1, 0]]], axis=0)

        elif dimension == 2:
            if "triangle" not in mesh.cells_dict:
                raise ValueError("2D finite difference support requires triangular meshes")

            elements = mesh.cells_dict["triangle"]
            element_type = "triangles"
            n_vertices_per_element = 3

            area, grad_phi = MeshUtils.precompute_p1_triangle_geometry(points, elements)

            # Create all directed edges from triangles
            edges = np.concatenate(
                [
                    elements[:, [0, 1]],  # a -> b
                    elements[:, [0, 2]],  # a -> c
                    elements[:, [1, 0]],  # b -> a
                    elements[:, [1, 2]],  # b -> c
                    elements[:, [2, 0]],  # c -> a
                    elements[:, [2, 1]],  # c -> b
                ],
                axis=0,
            )

        elif dimension == 3:
            if "tetra" not in mesh.cells_dict:
                raise ValueError("3D finite difference support requires tetrahedral meshes")

            elements = mesh.cells_dict["tetra"]
            element_type = "tetrahedra"
            n_vertices_per_element = 4

            # Create all directed edges from tetrahedra (6 edges per tetrahedron)
            edges = np.concatenate(
                [
                    elements[:, [0, 1]],  # a -> b
                    elements[:, [0, 2]],  # a -> c
                    elements[:, [0, 3]],  # a -> d
                    elements[:, [1, 2]],  # b -> c
                    elements[:, [1, 3]],  # b -> d
                    elements[:, [2, 3]],  # c -> d
                    # Reverse directions
                    elements[:, [1, 0]],
                    elements[:, [2, 0]],
                    elements[:, [3, 0]],
                    elements[:, [2, 1]],
                    elements[:, [3, 1]],
                    elements[:, [3, 2]],
                ],
                axis=0,
            )

        else:
            raise ValueError(f"Finite difference not supported for dimension {dimension}")

        # Build neighbor lists efficiently
        neighbors = {}
        for i in range(n_points):
            # Find all edges starting from vertex i
            mask = edges[:, 0] == i
            neighbor_ids = np.unique(edges[mask, 1]).tolist()
            neighbors[i] = neighbor_ids

        # Store connectivity info
        mesh_connectivity = {"points": points, element_type: elements, "neighbors": neighbors, "n_points": n_points, "dimension": dimension}

        if dimension == 2:
            mesh_connectivity["p1_area"] = np.array(area)
            mesh_connectivity["p1_grad_phi"] = np.array(grad_phi)

        msg = f"Preprocessed mesh connectivity: {n_points} points, {len(elements)} {element_type}"

        mesh_connectivity["nodal_ds"] = MeshUtils.compute_nodal_ds(mesh_connectivity)
        mesh_connectivity["boundary_indices"] = boundary_indices

        bp = points[boundary_indices]
        all_indices = np.arange(len(points))
        non_boundary_indices = np.setdiff1d(all_indices, boundary_indices)
        _bp = points[non_boundary_indices]

        mesh_connectivity["boundary_points"] = bp
        if dimension > 1:  # TODO temporary fix -> hardencode for 1D geometries
            bpe = MeshUtils.extract_boundary_edges(mesh.cells_dict["triangle"], len(bp))
            mesh_connectivity["VM"] = MeshUtils.get_visibility_matrix_raytrace(bp, bpe, _bp[0], 40)

        msg = f"Preprocessed mesh connectivity: {n_points} points, {len(elements)} {element_type}"

        return mesh_connectivity, msg

    @staticmethod
    def compute_nodal_ds(mesh_connectivity, boundary_indices=None):
        """
        Compute element measure (length/area/volume) attributed to each node.

        For boundary view factors:
        - 1D domain: ds = length of adjacent line segments (÷2 per node)
        - 2D domain: ds = length of adjacent boundary edges (÷2 per node)
        - 3D domain: ds = area of adjacent boundary faces (÷3 per node)

        Parameters
        ----------
        mesh_connectivity : dict
            Preprocessed mesh connectivity from _preprocess_mesh_connectivity
        boundary_indices : array, optional
            Indices of boundary nodes. If None, returns ds for all nodes.

        Returns
        -------
        ds : ndarray of shape (n_boundary_points,)
        """
        from collections import Counter

        dimension = mesh_connectivity["dimension"]
        points = mesh_connectivity["points"]
        n_points = mesh_connectivity["n_points"]

        ds = np.zeros(n_points)

        if dimension == 1:
            # Boundary = endpoints, ds = half of adjacent line element
            elements = mesh_connectivity["lines"]
            for elem in elements:
                p0, p1 = points[elem[0]], points[elem[1]]
                length = np.linalg.norm(p1 - p0)
                ds[elem[0]] += 0.5 * length
                ds[elem[1]] += 0.5 * length

        elif dimension == 2:
            # Boundary = edges that appear once in the triangulation
            triangles = mesh_connectivity["triangles"]

            edge_count = Counter()
            edge_lengths = {}

            for tri in triangles:
                for i in range(3):
                    n0, n1 = int(tri[i]), int(tri[(i + 1) % 3])
                    edge = (min(n0, n1), max(n0, n1))
                    edge_count[edge] += 1
                    if edge not in edge_lengths:
                        edge_lengths[edge] = np.linalg.norm(points[n1] - points[n0])

            # Boundary edges appear exactly once
            for edge, count in edge_count.items():
                if count == 1:
                    length = edge_lengths[edge]
                    ds[edge[0]] += 0.5 * length
                    ds[edge[1]] += 0.5 * length

        elif dimension == 3:
            # Boundary = faces that appear once in the tetrahedralization
            tetra = mesh_connectivity["tetrahedra"]

            face_count = Counter()
            face_areas = {}

            # 4 faces per tetrahedron
            face_local = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]

            for tet in tetra:
                for fl in face_local:
                    nodes = tuple(sorted([int(tet[fl[0]]), int(tet[fl[1]]), int(tet[fl[2]])]))
                    face_count[nodes] += 1
                    if nodes not in face_areas:
                        p0, p1, p2 = points[nodes[0]], points[nodes[1]], points[nodes[2]]
                        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
                        face_areas[nodes] = area

            # Boundary faces appear exactly once
            for face, count in face_count.items():
                if count == 1:
                    area = face_areas[face]
                    for node in face:
                        ds[node] += area / 3.0  # Divide among 3 vertices

        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        if boundary_indices is not None:
            return ds[boundary_indices]

        return ds

    @staticmethod
    def get_boundary_normals(mesh, k=8):
        points = mesh.points
        if "tetra" in mesh.cells_dict:
            boundary_elements = MeshUtils._get_boundary_elements(mesh.cells_dict["tetra"], "tetra")
            actual_dim = 3
        elif "triangle" in mesh.cells_dict:
            boundary_elements = MeshUtils._get_boundary_elements(mesh.cells_dict["triangle"], "triangle")
            actual_dim = 2
        else:
            raise ValueError("Unsupported mesh type.")

        boundary_indices = np.unique(boundary_elements)
        return MeshUtils._compute_normals_pca(points, boundary_indices, actual_dim, k, mesh=mesh)

    @staticmethod
    def _get_boundary_elements(cells, cell_type):
        """Finds elements (lines/triangles) that appear only once."""
        if cell_type == "tetra":
            faces = np.sort(np.vstack([cells[:, [0, 1, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 3]], cells[:, [1, 2, 3]]]), axis=1)
        else:  # triangle
            faces = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)

        unique_elements, counts = np.unique(faces, axis=0, return_counts=True)
        return unique_elements[counts == 1]

    @staticmethod
    def _compute_normals_pca(points, boundary_indices, dim, k=8, mesh=None):
        """
        Compute outward-pointing normals for boundary points using PCA.

        Handles both outer boundaries and inner boundaries (holes) correctly.
        Normals always point OUT of the domain material.
        """
        coords = points[boundary_indices, :dim]

        tree = KDTree(coords)
        _, neighbors = tree.query(coords, k=min(k, len(coords)))

        v_normals = np.zeros((len(boundary_indices), dim))
        mesh_centroid = np.mean(points[:, :dim], axis=0)

        # Get boundary edges for point-in-polygon test
        boundary_edges = None
        if dim == 2 and mesh is not None and "triangle" in mesh.cells_dict:
            from collections import Counter

            triangles = mesh.cells_dict["triangle"]
            edge_count = Counter()
            for tri in triangles:
                for j in range(3):
                    n0, n1 = int(tri[j]), int(tri[(j + 1) % 3])
                    edge = (min(n0, n1), max(n0, n1))
                    edge_count[edge] += 1
            boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

        def point_in_domain_2d(pt, edges, all_points):
            """Ray casting point-in-polygon for 2D."""
            if edges is None:
                return True  # Fallback

            x, y = pt[0], pt[1]
            crossings = 0

            for e0, e1 in edges:
                x0, y0 = all_points[e0, 0], all_points[e0, 1]
                x1, y1 = all_points[e1, 0], all_points[e1, 1]

                if ((y0 > y) != (y1 > y)) and (y1 != y0):
                    x_int = x0 + (x1 - x0) * (y - y0) / (y1 - y0)
                    if x < x_int:
                        crossings += 1

            return crossings % 2 == 1

        for i in range(len(boundary_indices)):
            patch = coords[neighbors[i]]
            centered_patch = patch - np.mean(patch, axis=0)

            # SVD finds directions of variance
            _, _, vh = np.linalg.svd(centered_patch)

            # The last row of vh is the direction of LEAST variance (the normal)
            normal = vh[-1, :]
            normal = normal / (np.linalg.norm(normal) + 1e-12)

            # Test which direction is "inside" the domain
            step_size = 1e-4 * np.max(np.abs(coords.max(axis=0) - coords.min(axis=0)))

            test_point_positive = coords[i] + step_size * normal
            test_point_negative = coords[i] - step_size * normal

            if dim == 2:
                inside_positive = point_in_domain_2d(test_point_positive, boundary_edges, points[:, :dim])
                inside_negative = point_in_domain_2d(test_point_negative, boundary_edges, points[:, :dim])

                # Normal should point OUT of domain (toward the side that is NOT inside)
                if inside_positive and not inside_negative:
                    # Positive direction is inside, so normal should point negative
                    normal = -normal
                elif inside_negative and not inside_positive:
                    # Negative direction is inside, normal already points out
                    pass
                else:
                    # Fallback to centroid-based heuristic
                    point_vec = coords[i] - mesh_centroid
                    if np.dot(normal, point_vec) < 0:
                        normal = -normal
            else:
                # 3D or fallback: use centroid heuristic
                point_vec = coords[i] - mesh_centroid
                if np.dot(normal, point_vec) < 0:
                    normal = -normal

            v_normals[i] = normal

        return v_normals, boundary_indices

    @staticmethod
    @jax.jit
    def get_visibility_matrix_ordered(P_bnd: jnp.ndarray, P_int: jnp.ndarray) -> jnp.ndarray:
        """
        Compute visibility matrix for boundary points considering interior points.

        Parameters
        ----------
        P_bnd : jnp.ndarray
            (n_bnd, 2) boundary points of the mesh (simple polygon).
        P_int : jnp.ndarray
            (n_int, 2) interior points of the mesh.

        Returns
        -------
        jnp.ndarray
            (n_bnd, n_bnd) boolean visibility matrix.
            VM[i,j] = 1 if boundary point i can "see" boundary point j.

        Two boundary points are visible if:
        1. The line segment between them does not intersect any boundary edge

        (except at the endpoints themselves).
        2. The ray passes through the interior (midpoint inside polygon).
        3. No other boundary point lies on the segment between them.

        """

        def order_boundary_points(P):
            """Order boundary points counter-clockwise by angle from centroid."""
            center = jnp.mean(P, axis=0)
            angles = jnp.arctan2(P[:, 1] - center[1], P[:, 0] - center[0])
            return jnp.argsort(angles)

        # Order boundary points
        order = order_boundary_points(P_bnd)
        P = P_bnd[order]

        @jax.jit
        def _compute(P, P_interior):
            n_bnd = P.shape[0]
            n_int = P_interior.shape[0]
            ks = jnp.arange(n_bnd)

            # Polygon edges: edge k connects point k to point (k+1) mod n
            C = P  # Start points of edges (n_bnd, 2)
            D = jnp.roll(P, -1, axis=0)  # End points of edges (n_bnd, 2)

            def orient(p, q, r):
                """
                Compute orientation of triplet (p, q, r).
                Returns positive if counter-clockwise, negative if clockwise, 0 if collinear.
                """
                return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

            def point_in_polygon(pt):
                """
                Ray-casting point-in-polygon test for a single point pt (2,).
                Returns True if pt is inside (or on) the polygon defined by C-D.
                """
                x = pt[0]
                y = pt[1]

                x0 = C[:, 0]
                y0 = C[:, 1]
                x1 = D[:, 0]
                y1 = D[:, 1]

                # Edges that straddle the horizontal ray at y
                cond = ((y0 > y) != (y1 > y)) & (y1 != y0)

                # x-coordinate where the ray at height y intersects the edge
                x_int = x0 + (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12)

                crossings = cond & (x < x_int)

                # Inside if number of crossings is odd
                inside = jnp.mod(jnp.sum(crossings.astype(jnp.int32)), 2) == 1
                return inside

            def segments_intersect_strict(A, B, C_pt, D_pt):
                """
                Check if segment AB strictly intersects segment CD.
                Returns True only if they cross each other (not just touch at endpoints).
                """
                o1 = orient(A, B, C_pt)
                o2 = orient(A, B, D_pt)
                o3 = orient(C_pt, D_pt, A)
                o4 = orient(C_pt, D_pt, B)

                # Strict intersection: both segments must straddle each other
                return (o1 * o2 < 0.0) & (o3 * o4 < 0.0)

            def point_on_segment(A, B, P_test, tol=1e-8):
                """
                Check if point P_test lies on segment AB (excluding endpoints).

                Parameters
                ----------
                A, B : points defining the segment
                P_test : point to test
                tol : tolerance for collinearity and bounds checking

                Returns
                -------
                bool : True if P_test is strictly on segment AB (not at endpoints)
                """
                # Vector from A to B
                AB = B - A
                # Vector from A to P
                AP = P_test - A

                # Length squared of AB
                AB_len_sq = jnp.dot(AB, AB)

                # Parameter t where P = A + t * AB
                # t = dot(AP, AB) / dot(AB, AB)
                t = jnp.dot(AP, AB) / (AB_len_sq + 1e-12)

                # Point on line closest to P_test
                closest = A + t * AB

                # Distance from P_test to the line
                dist_sq = jnp.sum((P_test - closest) ** 2)

                # Check if:
                # 1. Point is close to the line (collinear)
                # 2. t is strictly between 0 and 1 (not at endpoints)
                is_collinear = dist_sq < tol**2
                is_between = (t > tol) & (t < 1.0 - tol)

                return is_collinear & is_between

            def boundary_point_blocks_segment(A, B, i, j):
                """
                Check if any OTHER boundary point (not i or j) lies on segment AB.

                This ensures the ray terminates at the first boundary point it hits.
                """

                def check_single_point(k):
                    # Skip the endpoints themselves
                    is_endpoint = (k == i) | (k == j)
                    P_k = P[k]
                    on_segment = point_on_segment(A, B, P_k)
                    return (~is_endpoint) & on_segment

                # Check all boundary points
                blocked_by = jax.vmap(check_single_point)(ks)
                return jnp.any(blocked_by)

            def seg_visible(i, j):
                """
                Check if the segment from boundary point i to boundary point j is visible.

                Visible means:
                1. The segment does not intersect any polygon edge (except adjacent ones)
                2. The midpoint lies inside the polygon
                3. No other boundary point lies on the segment

                """
                A = P[i]
                B = P[j]

                # === Check 1: No intersection with polygon edges ===
                k2 = (ks + 1) % n_bnd

                # Skip edges that share an endpoint with the query segment
                is_adjacent = (ks == i) | (ks == j) | (k2 == i) | (k2 == j)

                # Broadcast A and B for vectorized computation
                A_b = jnp.broadcast_to(A, (n_bnd, 2))
                B_b = jnp.broadcast_to(B, (n_bnd, 2))

                # Check intersection with each edge
                intersects = segments_intersect_strict(A_b, B_b, C, D)

                # Mask out adjacent edges
                intersects = intersects & (~is_adjacent)

                # No edge intersection
                no_edge_intersection = ~jnp.any(intersects)

                # === Check 2: Midpoint inside polygon ===
                mid = 0.5 * (A + B)
                midpoint_inside = point_in_polygon(mid)

                # === Check 3: No other boundary point on the segment ===
                no_blocking_point = ~boundary_point_blocks_segment(A, B, i, j)

                # All conditions must be satisfied
                return no_edge_intersection & midpoint_inside & no_blocking_point

            def outer_body(i, VM):
                def inner_body(j, row):
                    is_same = i == j

                    # Adjacent boundary points are always visible (they share an edge)
                    is_adjacent_point = (j == (i + 1) % n_bnd) | (j == (i - 1 + n_bnd) % n_bnd)

                    visible_ij = jax.lax.cond(is_same, lambda: False, lambda: jax.lax.cond(is_adjacent_point, lambda: True, lambda: seg_visible(i, j)))  # Diagonal is always 0 (can't see itself)  # Adjacent boundary points are always visible
                    row = row.at[j].set(visible_ij)
                    return row

                row = VM[i]
                row = jax.lax.fori_loop(0, n_bnd, inner_body, row)
                VM = VM.at[i].set(row)
                return VM

            VM0 = jnp.zeros((n_bnd, n_bnd), dtype=jnp.float32)
            VM = jax.lax.fori_loop(0, n_bnd, outer_body, VM0)

            return VM

        # Compute visibility in ordered space
        visible_ord = _compute(P, P_int)

        # Reorder back to original point ordering
        inv_order = jnp.argsort(order)
        visible = visible_ord[jnp.ix_(inv_order, inv_order)]

        # Ensure diagonal is zero (point can't see itself)
        n = visible.shape[0]
        VM_jax = visible.at[jnp.diag_indices(n)].set(0.0)

        return VM_jax

    @staticmethod
    def get_visibility_matrix_raytrace(boundary_points: jnp.ndarray, boundary_edges: jnp.ndarray, interior_point: jnp.ndarray = None, n_ray_samples: int = 10) -> jnp.ndarray:
        """
        Compute visibility matrix using ray tracing.
        Works for arbitrary domains with any number of holes.

        Parameters
        ----------
        boundary_points : jnp.ndarray
            (n_bnd, 2) boundary points.
        boundary_edges : jnp.ndarray
            (n_edges, 2) boundary edges as index pairs.
            Each edge [i, j] connects boundary_points[i] to boundary_points[j].
        interior_point : jnp.ndarray, optional
            (2,) a single point known to be inside the domain.
            Used to determine inside/outside orientation.
        n_ray_samples : int
            Number of sample points along each ray to verify it stays inside.

        Returns
        -------
        jnp.ndarray
            (n_bnd, n_bnd) visibility matrix.
        """
        n_bnd = boundary_points.shape[0]
        n_edges = boundary_edges.shape[0]

        # Precompute edge geometry
        E0 = boundary_points[boundary_edges[:, 0]]  # (n_edges, 2)
        E1 = boundary_points[boundary_edges[:, 1]]  # (n_edges, 2)

        @jax.jit
        def compute(P, E0, E1, edges, interior_pt):

            def cross2d(a, b):
                return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

            # =====================================================================
            # Winding number for inside/outside test
            # =====================================================================
            def winding_number(pt):
                """
                Compute winding number of pt w.r.t. all boundary edges.
                |winding| > 0.5 means inside.
                """
                v0 = E0 - pt  # (n_edges, 2)
                v1 = E1 - pt  # (n_edges, 2)

                cross = cross2d(v0, v1)
                dot = jnp.sum(v0 * v1, axis=-1)
                angles = jnp.arctan2(cross, dot)

                return jnp.sum(angles) / (2.0 * jnp.pi)

            def is_inside(pt):
                wn = winding_number(pt)
                # Use interior_pt to determine sign convention
                ref_wn = winding_number(interior_pt)
                # Inside if same sign as reference
                return (wn * ref_wn) > 0.25

            # =====================================================================
            # Ray-edge intersection
            # =====================================================================
            def ray_intersects_edge(ray_origin, ray_dir, e0, e1, tol=1e-10):
                """
                Check if ray from ray_origin in direction ray_dir intersects edge e0-e1.
                Returns (intersects, t_ray, t_edge)
                """
                edge_dir = e1 - e0
                denom = cross2d(ray_dir, edge_dir)

                parallel = jnp.abs(denom) < tol

                diff = e0 - ray_origin
                t_ray = cross2d(diff, edge_dir) / (denom + tol)
                t_edge = cross2d(diff, ray_dir) / (denom + tol)

                # Valid intersection: t_ray > 0 (in front of ray) and t_edge in [0, 1]
                valid = (~parallel) & (t_ray > tol) & (t_edge > tol) & (t_edge < 1.0 - tol)

                return valid, t_ray, t_edge

            def segment_intersects_edge_proper(A, B, e0, e1, tol=1e-10):
                """
                Check if segment AB properly intersects edge e0-e1.
                Proper = crosses through interior, not at endpoints.
                """
                AB = B - A
                edge = e1 - e0
                denom = cross2d(AB, edge)

                parallel = jnp.abs(denom) < tol

                diff = e0 - A
                t_seg = cross2d(diff, edge) / (denom + tol)
                t_edge = cross2d(diff, AB) / (denom + tol)

                # Proper intersection: both parameters strictly in (0, 1)
                eps = 1e-12
                proper = (~parallel) & (t_seg > eps) & (t_seg < 1 - eps) & (t_edge > eps) & (t_edge < 1 - eps)

                return proper

            # =====================================================================
            # Main visibility check
            # =====================================================================
            def check_visibility(i, j):
                A = P[i]
                B = P[j]

                # Same point
                same = i == j

                # ------------------------------------------------------------------
                # Check 1: Segment doesn't cross any boundary edge
                # ------------------------------------------------------------------
                def check_edge_crossing(k):
                    e0 = E0[k]
                    e1 = E1[k]
                    ei0 = edges[k, 0]
                    ei1 = edges[k, 1]

                    # Skip edges adjacent to our segment endpoints
                    adjacent = (ei0 == i) | (ei0 == j) | (ei1 == i) | (ei1 == j)

                    crosses = segment_intersects_edge_proper(A, B, e0, e1)
                    return crosses & (~adjacent)

                edge_crossings = jax.vmap(check_edge_crossing)(jnp.arange(n_edges))
                any_crossing = jnp.any(edge_crossings)

                # ------------------------------------------------------------------
                # Check 2: Ray stays inside domain (sample along segment)
                # ------------------------------------------------------------------
                t_samples = jnp.linspace(0.1, 0.9, n_ray_samples)
                sample_points = A[None, :] + t_samples[:, None] * (B - A)[None, :]

                inside_checks = jax.vmap(is_inside)(sample_points)
                all_inside = jnp.all(inside_checks)

                # ------------------------------------------------------------------
                # Check 3: No other boundary point blocks the ray
                # ------------------------------------------------------------------
                def point_blocks_ray(k):
                    """Check if boundary point k lies strictly on segment AB."""
                    Pk = P[k]

                    # Skip endpoints
                    is_endpoint = (k == i) | (k == j)

                    # Project Pk onto line AB
                    AB = B - A
                    AP = Pk - A
                    AB_len_sq = jnp.dot(AB, AB) + 1e-12
                    t = jnp.dot(AP, AB) / AB_len_sq

                    # Distance to line
                    proj = A + t * AB
                    dist_sq = jnp.sum((Pk - proj) ** 2)

                    # On segment if: close to line AND t in (0, 1)
                    tol = 1e-6
                    on_line = dist_sq < tol**2
                    in_segment = (t > tol) & (t < 1.0 - tol)

                    return (~is_endpoint) & on_line & in_segment

                blocking = jax.vmap(point_blocks_ray)(jnp.arange(n_bnd))
                any_blocking = jnp.any(blocking)

                # ------------------------------------------------------------------
                # Combine all checks
                # ------------------------------------------------------------------
                visible = (~same) & (~any_crossing) & all_inside & (~any_blocking)

                return visible.astype(jnp.float32)

            # Vectorize over all pairs
            ii, jj = jnp.meshgrid(jnp.arange(n_bnd), jnp.arange(n_bnd), indexing="ij")
            VM = jax.vmap(lambda i: jax.vmap(lambda j: check_visibility(i, j))(jnp.arange(n_bnd)))(jnp.arange(n_bnd))

            return VM

        # Default interior point: centroid (works for simple domains)
        if interior_point is None:
            interior_point = jnp.mean(boundary_points, axis=0)

        return compute(boundary_points, E0, E1, boundary_edges, interior_point)

    @staticmethod
    def extract_boundary_edges(triangles: jnp.ndarray, n_points: int) -> jnp.ndarray:
        """
        Extract boundary edges from triangle connectivity.
        Boundary edges appear in exactly one triangle.

        Parameters
        ----------
        triangles : jnp.ndarray
            (n_tri, 3) triangle connectivity.
        n_points : int
            Total number of points.

        Returns
        -------
        jnp.ndarray
            (n_boundary_edges, 2) boundary edge indices.
        """
        import numpy as np

        triangles = np.asarray(triangles)

        # Collect all edges (sorted to make undirected)
        edges = []
        for tri in triangles:
            for k in range(3):
                e = tuple(sorted([tri[k], tri[(k + 1) % 3]]))
                edges.append(e)

        # Count occurrences
        from collections import Counter

        edge_count = Counter(edges)

        # Boundary edges appear exactly once
        boundary_edges = [list(e) for e, c in edge_count.items() if c == 1]

        return jnp.array(boundary_edges)

    @staticmethod
    @jax.jit
    def get_view_factor_3d(P, VM, Nrm, ds):

        n_pts = P.shape[0]

        v = P[None, :, :] - P[:, None, :]  # (N,N,3), x_j - x_i
        r = jnp.linalg.norm(v, axis=-1)  # (N,N)

        # avoid divide by zero only on diagonal
        r_safe = r + jnp.eye(n_pts)
        r_hat = v / r_safe[..., None]  # (N,N,3)

        # cosines
        cos_i = jnp.sum(Nrm[:, None, :] * r_hat, axis=-1)  # (N,N)
        cos_j = -jnp.sum(Nrm[None, :, :] * r_hat, axis=-1)  # (N,N)

        # physical clipping
        cos_i = jnp.maximum(0.0, cos_i)
        cos_j = jnp.maximum(0.0, cos_j)

        # kernel
        F_ij = (cos_i * cos_j) / (jnp.pi * r_safe**2)  # 3D Formula

        # apply visibility
        F_ij = F_ij * VM

        # total view factor from i
        F = jnp.sum(F_ij * ds[None, :], axis=1)

        return F

    @staticmethod
    @jax.jit
    def get_view_factor_2d(P, VM, Nrm, ds):

        n_pts = P.shape[0]

        v = P[None, :, :] - P[:, None, :]  # (N,N,2), x_j - x_i
        r = jnp.linalg.norm(v, axis=-1)  # (N,N)

        # avoid divide by zero only on diagonal
        r_safe = r + jnp.eye(n_pts)
        r_hat = v / r_safe[..., None]  # (N,N,2)

        # cosines
        cos_i = jnp.sum(Nrm[:, None, :] * r_hat, axis=-1)  # (N,N)
        cos_j = -jnp.sum(Nrm[None, :, :] * r_hat, axis=-1)  # (N,N)

        # physical clipping
        cos_i = jnp.maximum(0.0, cos_i)
        cos_j = jnp.maximum(0.0, cos_j)

        # kernel
        F_ij = (cos_i * cos_j) / (2.0 * r_safe)  # 2D formula

        # apply visibility
        F_ij = F_ij * VM

        # total view factor from i
        F = jnp.sum(F_ij * ds[None, :], axis=1)

        return F

    @staticmethod
    def precompute_p1_line_geometry(points, elements):
        """
        Precompute P1 line element geometry (lengths and shape function gradients).

        Parameters
        ----------
        points : ndarray of shape (n_points, 1)
            Node coordinates
        elements : ndarray of shape (n_elements, 2)
            Line element connectivity (node indices)

        Returns
        -------
        length : ndarray of shape (n_elements,)
            Length of each line element
        grad_phi : ndarray of shape (n_elements, 2)
            Gradient of each shape function on each element
            grad_phi[e, i] = d(phi_i)/dx on element e
        """
        n_elements = elements.shape[0]

        # Get coordinates of element vertices
        x0 = points[elements[:, 0], 0]  # First node x-coordinate
        x1 = points[elements[:, 1], 0]  # Second node x-coordinate

        # Compute element lengths
        length = np.abs(x1 - x0)

        # For P1 elements in 1D:
        # phi_0(x) = (x1 - x) / L  =>  d(phi_0)/dx = -1/L
        # phi_1(x) = (x - x0) / L  =>  d(phi_1)/dx = +1/L
        # Note: Sign depends on orientation (x1 > x0 or x1 < x0)

        grad_phi = np.zeros((n_elements, 2))

        # Handle orientation: gradient sign depends on element direction
        dx = x1 - x0
        grad_phi[:, 0] = -1.0 / dx  # d(phi_0)/dx
        grad_phi[:, 1] = 1.0 / dx  # d(phi_1)/dx

        return length, grad_phi

    @staticmethod
    @jax.jit
    def precompute_p1_triangle_geometry(points: jnp.ndarray, triangles: jnp.ndarray):
        """
        points: (N,2)
        triangles: (T,3) int
        Returns:
        area: (T,)
        grad_phi: (T,3,2) where grad_phi[t,a,:] = ∇φ_a on triangle t
        """
        tri = triangles.astype(jnp.int32)
        p0 = points[tri[:, 0], :]  # (T,2)
        p1 = points[tri[:, 1], :]
        p2 = points[tri[:, 2], :]

        x0, y0 = p0[:, 0], p0[:, 1]
        x1, y1 = p1[:, 0], p1[:, 1]
        x2, y2 = p2[:, 0], p2[:, 1]

        # Twice signed area (Jacobian determinant)
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)  # (T,)
        area = 0.5 * jnp.abs(det)  # (T,)
        det_safe = jnp.where(jnp.abs(det) < 1e-12, 1e-12, det)

        # Gradients of barycentric basis functions on a triangle:
        # ∇φ0 = [ (y1 - y2), (x2 - x1) ] / det
        # ∇φ1 = [ (y2 - y0), (x0 - x2) ] / det
        # ∇φ2 = [ (y0 - y1), (x1 - x0) ] / det
        g0 = jnp.stack([(y1 - y2) / det_safe, (x2 - x1) / det_safe], axis=-1)  # (T,2)
        g1 = jnp.stack([(y2 - y0) / det_safe, (x0 - x2) / det_safe], axis=-1)
        g2 = jnp.stack([(y0 - y1) / det_safe, (x1 - x0) / det_safe], axis=-1)

        grad_phi = jnp.stack([g0, g1, g2], axis=1)  # (T,3,2)
        return area, grad_phi


class domain(MeshUtils, Geometries):
    """
    Mesh-based domain class for defining computational domains and sampling collocation points.

    Supports:
    - Rectangle, circle, and custom geometries via PyGmsh
    - Loading meshes from files
    - Sampling interior and boundary points
    - Time-dependent problems

    Attributes:
        points_by_tag: Dictionary mapping region names to point coordinates
        sampled_points: Currently sampled points as contiguous array
        context: Dictionary of sampled arrays for training
    """

    def __init__(self, constructor: Union[Callable, str] = None, algorithm: int = 6, time: Optional[Tuple[float, float]] = None, compute_mesh_connectivity: bool = True):
        """
        Initialize the domain.

        Args:
            constructor: Function accepting a pygmsh.geo.Geometry object or a path to a meshfile
            algorithm: Gmsh meshing algorithm
            time: Tuple of (start, end) for time-dependent problems
            mesh_connectivity: Wether or not to compute the some hyperparameters about the mesh (needed for finite_difference methods)
        """
        super().__init__()
        self.log: Logger = get_logger()

        # Storage
        self.compute_mesh_connectivity = compute_mesh_connectivity
        self.points_by_tag: Dict[str, np.ndarray] = {}
        self.sampled_points: Dict[str, np.ndarray] = {}
        self.context: Dict[str, np.ndarray] = {}
        self.normals_by_tag: Dict[str, np.ndarray] = {}

        # Neural operator storage
        self.parameters: Dict[str, Union[float, int]] = {}
        self.arrays: Dict[str, np.ndarray] = {}
        self.tag_indices: Dict[str, np.ndarray] = {}
        self.avaiable_mesh_tags: List[str] = []  # names of the tags from the mesh generator

        # Tensor tags for parametric PDEs (shape (1, ...) or (B, ...))
        self.tensor_tags: Dict[str, np.ndarray] = {}

        # Resampling support
        self._mesh_points: Dict[str, np.ndarray] = {}  # Full mesh points for resampling
        self._resampling_strategies: Dict[str, Any] = {}  # Tag -> ResamplingStrategy

        # Configuration
        self.dimension: int = 2
        self.total_samples: int = 1
        self.time = time
        self._is_time_dependent = time is not None
        self._verbose = True
        self.same_domain = False

        # Tracking
        self.index_tags: List[str] = []
        self.normal_tags: List[str] = []
        self.reference_solutions: List[Callable] = []
        self.sample_dict: List = []

        # Meshio mesh
        self.mesh = None

        # Generate or load mesh
        if isinstance(constructor, str):
            self._load_mesh(constructor)
            self.log.info(f"Loaded mesh from the constructor function: {len(self.mesh.points)} points")
        elif isinstance(constructor, Callable):
            self._generate_mesh(constructor, algorithm)
            self.log.info(f"Loaded mesh from {constructor}: {len(self.mesh.points)} points")
        else:
            raise ValueError("Must provide either geometry_func or mesh_file")

        boundary_indices = self._extract_points_from_mesh(self.mesh)

        # Preprocess mesh for finite differences
        if self.mesh is not None and self.compute_mesh_connectivity:
            self.mesh_connectivity, msg = self._preprocess_mesh_connectivity(self.mesh, self.dimension, boundary_indices)
            self.log.info(msg)

        # Add time dimension if needed
        if self._is_time_dependent:
            self._add_time_dimension(time[0], time[1], time[2])

        # Set up independent variable names
        spatial_dims = self.dimension - (1 if self._is_time_dependent else 0)
        default_spatial = ["x", "y", "z"][:spatial_dims]
        default_indep = default_spatial + (["t"] if self._is_time_dependent else [])

        self.indep = default_indep
        if self._is_time_dependent:
            self.spatial = [i for i in self.indep if i != "t"]
        else:
            self.spatial = list(self.indep)

        user_spatial_dims = len(self.spatial)
        if user_spatial_dims < spatial_dims:
            self.dimension = user_spatial_dims + (1 if self._is_time_dependent else 0)
            for tag, pts in self.points_by_tag.items():
                if pts.shape[1] > self.dimension:
                    self.points_by_tag[tag] = pts[..., : self.dimension]

    def __lt__(self, other: Tuple[str, Any]) -> "domain":
        """Attach parameters or arrays using < operator."""
        if not isinstance(other, tuple) or len(other) != 2:
            raise ValueError("Use: domain < (name, value)")

        name, value = other

        if not isinstance(name, str):
            raise ValueError("Name must be a string")

        if isinstance(value, (int, float)):
            self.parameters[name] = float(value)
            self.log.info(f"Attached parameter '{name}' = {value}")
        elif isinstance(value, np.ndarray):
            self.arrays[name] = value.astype(np.float32)
            self.log.info(f"Attached array '{name}' with shape {value.shape}")
        elif isinstance(value, (list, tuple)):
            arr = np.array(value, dtype=np.float32)
            self.arrays[name] = arr
            self.log.info(f"Attached array '{name}' with shape {arr.shape}")
        else:
            raise ValueError(f"Value must be scalar or array, got {type(value)}")

        return self

    def __rmul__(self, n: int) -> "domain":
        """Batch the domain n times: 2 * domain samples 2x independently.

        When sample() is called, it will sample n times and concatenate results.

        Example:
            domain = 10 * domain.from_mesh(domain.rect, {...}, 0.05)
            domain.sample({"interior": (100, None)})  # Results in 1000 interior points
        """
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"Batch count must be positive integer, got {n}")
        self._batch_count = n
        self.total_samples = n
        self.same_domain = True
        return self

    def __mul__(self, n: int) -> "domain":
        """Batch the domain n times: domain * 2 samples 2x independently."""
        return self.__rmul__(n)

    def __add__(self, other: Tuple[str, np.ndarray]) -> "domain":
        """Merge another domain into this one (stacks along batch dimension)."""
        for tag, points in other.points_by_tag.items():
            if tag in self.points_by_tag:
                self.points_by_tag[tag] = np.vstack([self.points_by_tag[tag], points])
            else:
                self.points_by_tag[tag] = points

        for tag, points in other.sampled_points.items():
            if tag in self.sampled_points:
                self.sampled_points[tag] = np.concatenate([self.sampled_points[tag], points], axis=0)
            else:
                self.sampled_points[tag] = points

        if not hasattr(self, "_parameter_list"):
            self._parameter_list = {k: [v] for k, v in self.parameters.items()}
        for name, value in other.parameters.items():
            if name in self._parameter_list:
                self._parameter_list[name].append(value)
            else:
                self._parameter_list[name] = [value]

        for name, values in self._parameter_list.items():
            self.parameters[name] = np.array(values, dtype=np.float32)

        return self

    # Generators

    def _generate_mesh(self, geometry_func: Callable, algorithm: int):
        """Generate mesh using PyGmsh."""
        import pygmsh

        explicit_dim = None

        with pygmsh.geo.Geometry() as geo:
            mesh, explicit_dim, ds = geometry_func(geo)

            if not isinstance(mesh, meshio.Mesh):
                mesh = geo.generate_mesh(dim=explicit_dim, algorithm=algorithm, verbose=True)

        self.mesh = mesh
        self.dimension = explicit_dim
        self.ds = ds

    def _load_mesh(self, mesh_file: str):
        """Load mesh from file using Meshio."""
        import meshio

        from pathlib import Path

        if not Path(mesh_file).exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        self.mesh = meshio.read(mesh_file)

        points = self.mesh.points
        if points.shape[1] == 3 and np.allclose(points[:, 2], 0):
            self.dimension = 2
        else:
            self.dimension = points.shape[1]

    def _extract_points_from_mesh(self, mesh):
        """Extract points and normals from mesh and organize by tag."""
        # 1. Pre-compute global normals for the entire mesh boundary
        # This returns an array of shape (N_total_points, 3)
        index_to_normal_pos = {}
        points = mesh.points[:, : self.dimension]
        self.points = points
        self.points_by_tag = {}

        if self.dimension > 1:
            boundary_normals, boundary_indices = self.get_boundary_normals(mesh)
            boundary_normals = boundary_normals[:, : self.dimension]
            index_to_normal_pos = {int(idx): int(pos) for pos, idx in enumerate(boundary_indices)}
        else:
            left_boundary = np.where(points[:, 0] == np.min(points[:, 0]))[0]
            right_boundary = np.where(points[:, 0] == np.max(points[:, 0]))[0]

            boundary_indices = np.stack([left_boundary, right_boundary]).flatten()

        if hasattr(mesh, "cell_sets") and mesh.cell_sets:
            for name, cell_data in mesh.cell_sets.items():
                if name.startswith("gmsh:"):
                    continue

                self.avaiable_mesh_tags.append(name)

                tag_points = set()

                if isinstance(cell_data, dict):
                    for cell_type, indices in cell_data.items():
                        if len(indices) > 0:
                            for cell_block in mesh.cells:
                                if cell_block.type == cell_type:
                                    for idx in indices:
                                        if idx < len(cell_block.data):
                                            tag_points.update(cell_block.data[idx].flatten())
                else:
                    for block_idx, indices in enumerate(cell_data):
                        if indices is None:
                            continue
                        if block_idx < len(mesh.cells) and len(indices) > 0:
                            cell_block = mesh.cells[block_idx]
                            for idx in indices:
                                if idx < len(cell_block.data):
                                    tag_points.update(cell_block.data[idx].flatten())

                if tag_points:
                    # Convert set to sorted list for consistent indexing
                    indices_list = np.array(list(tag_points)).flatten()

                    # Store points
                    self.points_by_tag[name] = points[indices_list]

                    # Map indices_list to positions in boundary_normals
                    # if all indices in indices_list are also in boundary_indices
                    missing = set(indices_list) - set(index_to_normal_pos.keys())
                    if not missing and self.dimension > 1:
                        normal_positions = np.array([index_to_normal_pos[i] for i in indices_list])
                        self.normals_by_tag[name] = boundary_normals[normal_positions]

        return boundary_indices

    def _add_time_dimension(self, t_start: float, t_end: float, n_time: int = 100):
        """Add time dimension to all point sets."""
        t_points = np.linspace(t_start, t_end, n_time)
        new_points_by_tag = {}
        for tag, points in self.points_by_tag.items():
            n_spatial = len(points)

            if tag == "interior":
                initial_interior = points.copy()
                new_points_by_tag["initial"] = np.concat([initial_interior, np.zeros((initial_interior.shape[0], 1))], axis=-1)

            repeated_points = np.repeat(points, n_time, axis=0)
            tiled_time = np.tile(t_points, n_spatial).reshape(-1, 1)
            new_points_by_tag[tag] = np.hstack([repeated_points, tiled_time])

        self.points_by_tag = new_points_by_tag
        self.dimension += 1

        # t_points = np.linspace(t_start, t_end, n_time)
        # new_points_by_tag = {}
        # for tag, points in self.points_by_tag.items()
        return None

    def add_tensor_tag(self, name: str, tensor: Union[np.ndarray, jnp.ndarray]) -> "domain":
        """Attach a tensor to this domain for parametric PDEs.

        Tensor tags allow parameters to vary across batched domains.
        The first dimension is the batch dimension and must match the
        domain's batch count exactly.

        Args:
            name: Name for this tensor (used in vars(name))
            tensor: Array with shape (B, ...) where B equals the domain's batch count.

        Returns:
            Self for method chaining.

        Example:
            domain = 2 * domain.from_mesh(...)
            domain.add_tensor_tag('diffusivity', jnp.array([[1.0], [2.0]]))  # shape (2, 1)
            a = domain.variable('diffusivity')  # Returns TensorTag
        """
        tensor = jnp.asarray(tensor)
        if tensor.ndim < 1:
            tensor = tensor.reshape(1, 1)

        # Validate batch dimension - must match exactly
        batch_count = getattr(self, "_batch_count", 1)
        tensor_batch = tensor.shape[0]
        if tensor_batch != batch_count:
            self.log.warning(f"Tensor '{name}' has batch dimension {tensor_batch}, but domain has " f"batch count {batch_count}. Was this intended?")

        self.tensor_tags[name] = tensor
        return self

    def variable(
        self,
        tag: str,
        sample: Tuple[Optional[int], Optional[Callable]] = (None, None),
        resampling_strategy=None,
        normals: bool = False,
        view_factor: bool = False,
        point_data: bool = False,
        split: bool = False,
        return_indices=False,
    ) -> Tuple[Variable, ...]:
        """Create Variable placeholders for a tagged point set or tensor.

        Args:
            tag: Name of the point set (e.g., 'interior', 'boundary')
                 or tensor tag (e.g., 'diffusivity')
            sample: Optional sampling specification for this tag:
                    - (n_samples, sampler) tuple to trigger sampling
                    - jax.numpy array to register a tensor tag

            resampling_strategy: Optional ResamplingStrategy for adaptive point selection
            normals: If True, also compute and return normal vectors for this tag
            return_indices: Wether or not to return the indices of the sampled points

        Returns:
            For point sets: Tuple of Variable placeholders, one per dimension.
            For point sets with normals=True: Tuple of variables + normal variables.
            For tensor tags: Single TensorTag placeholder.
        """

        # Optional sampling / tensor-tag attachment
        if sample is not None:
            if isinstance(sample, jnp.ndarray) or isinstance(sample, np.ndarray):
                # Attach as tensor tag (parameter field)
                if point_data:
                    self.sampled_points[tag] = sample
                else:
                    self.add_tensor_tag(tag, sample)

        if tag in self.points_by_tag.keys() and isinstance(sample, tuple) and len(sample) > 0 and isinstance(sample[0], (int, type(None))):
            # Sample points for this tag on demand
            # Save sample dict for inference
            self.sample_dict.append([tag, (None, None), resampling_strategy, normals, view_factor])
            points, idx, tag = self.sample({tag: sample}, normals, return_indices)

        # Store resampling strategy if provided
        if resampling_strategy is not None:
            self._resampling_strategies[tag] = resampling_strategy

        # Check if it's a tensor tag first
        if tag in self.tensor_tags:
            if split:
                return tuple([TensorTag(tag=tag, dim_index=i, domain=self) for i in range(sample.shape[-1])])
            else:
                return TensorTag(tag=tag, domain=self)

        if point_data:
            if split:
                (Variable(tag=tag, dim=[i, i + 1], domain=self) for i in range(sample.shape[-1]))
            else:
                return Variable(tag=tag, dim=[0, None], domain=self)

        if tag not in self.sampled_points:
            available = list(self.sampled_points.keys()) + list(self.tensor_tags.keys())
            raise ValueError(f"Tag '{tag}' not found. Did you call sample() first? Available: {available}")

        # Create Variable placeholder for each dimension
        coord_vars = [Variable(tag=tag, dim=[i, i + 1], domain=self) for i in range(self.dimension)]

        if normals:
            coord_vars += [Variable(tag=f"n_{tag}", dim=[i, i + 1], domain=self) for i in range(len(self.spatial))]

        if view_factor and hasattr(self, "mesh_connectivity"):

            # Only take the first batch index
            Nrm = -self.sampled_points[f"n_{tag}"][0, ...]  # Reverse the normals
            P = points[0, ...]

            ds = self.mesh_connectivity["nodal_ds"][self.mesh_connectivity["boundary_indices"]]

            if ds.shape[0] != P.shape[0]:
                ds = self.ds * np.ones(P.shape[0])
                self.log.warning("Size of elements is constant due to mismatch in boundary array.")

            all_bp = self.mesh_connectivity["boundary_points"]
            all_VM = self.mesh_connectivity["VM"]
            subset_bp = P
            point_to_idx = {tuple(pt): i for i, pt in enumerate(all_bp)}
            subset_indices = np.array([point_to_idx[tuple(pt)] for pt in subset_bp])
            subset_VM = all_VM[np.ix_(subset_indices, subset_indices)]

            if self.dimension == 2:
                VF = self.get_view_factor_2d(P, subset_VM, Nrm, ds)
            elif self.dimension == 3:
                VF = self.get_view_factor_3d(P, subset_VM, Nrm, ds)

            import matplotlib.pyplot as plt

            plt.figure()
            plt.scatter(P[:, 0], P[:, 1], c=VF)
            plt.savefig("b.png")

            # TODO: Fix if used for multiple domains !
            # self.sampled_points[f"v_{tag}"] = VM
            # self.sampled_points[f"f_{tag}"] = VF
            self.tensor_tags[f"v_{tag}"] = subset_VM[None, ...]
            self.tensor_tags[f"f_{tag}"] = VF[None, ...]
            # coord_vars += [TensorTag(tag=f"v_{tag}", domain=self)]
            coord_vars += [TensorTag(tag=f"f_{tag}", domain=self)]

        # if view_factor and not hasattr(self, "mesh_connectivity"):
        #    self.log.error("In order to calcuate the view factor please set compute_mesh_connectivity in the domain initialization to true.")

        if return_indices:
            coord_vars += [idx]

        return tuple(coord_vars)

    def __getitem__(self, tag: str) -> Tuple[Variable, ...]:
        """Shorthand for domain.variable(tag).

        Example:
            x, y = domain['interior']
        """
        return self.variable(tag)

    def sample(self, sample_spec: Dict[str, Tuple[int, Optional[Callable]]], normals: bool = False, return_indices: bool = False):
        """
        Sample points from the domain.

        Args:
            sample_spec: Dictionary mapping tag names to (n_samples, optional_sampler)
                        Example: {"interior": (2000, None), "boundary": (500, None)}

                        For time-dependent problems, use "initial" to sample points at t=0:
                        Example: {"interior": (2000, None), "initial": (500, None)}

        If domain was batched (e.g., 10 * domain), samples n_samples for each
        batch independently and concatenates results.
        """

        batch_count = getattr(self, "_batch_count", 1)

        for tag, (n_samples, sampler) in sample_spec.items():
            # Handle special "initial" tag for time-dependent problems
            if tag not in self.points_by_tag:
                available = list(self.points_by_tag.keys())
                self.log.error(f"Tag '{tag}' not found. Available: {available}")

            normals_avaiable = tag in self.normals_by_tag

            available_points = self.points_by_tag[tag]
            if normals_avaiable and normals:
                available_normals = self.normals_by_tag[tag]  # Pull the normals for this tag
            n_available = len(available_points)

            ii = 0
            og_tag = tag
            while tag in self.sampled_points:
                tag = og_tag + f"_{ii}"
                ii += 1

            # Store full mesh points for resampling
            if tag not in self._mesh_points:
                self._mesh_points[tag] = available_points.copy()

            if n_samples is None:
                n_samples = n_available

            if n_samples > n_available:
                self.log.warning(f"Requested {n_samples} samples for '{tag}' but only {n_available} available. Using all points.")
                n_samples = n_available

            all_samples = []
            all_normals = []

            if not self.same_domain:
                for _ in range(batch_count):
                    if sampler is not None:
                        # If your sampler needs the points, pass them,
                        # but ensure it returns the chosen INDICES.
                        if isinstance(sampler, Callable):
                            idx = sampler(available_points, n_samples)
                        elif isinstance(sampler, np.ndarray):
                            idx = sampler
                    else:
                        if n_available != n_samples:
                            idx = np.random.choice(n_available, size=n_samples, replace=False)
                        else:
                            idx = np.arange(n_available)

                    all_samples.append(available_points[idx])
                    if normals_avaiable and normals:
                        all_normals.append(available_normals[idx])

                # 3. Stack both
                self.sampled_points[tag] = np.stack(all_samples, axis=0)  # Shape (B, N, D)
                if normals_avaiable and normals:
                    self.sampled_points[f"n_{tag}"] = np.stack(all_normals, axis=0)  # Shape (B, N, D)

            else:
                # Sample once -> broadcast to all batches
                if sampler is not None:
                    idx = sampler(available_points, n_samples)
                else:
                    if n_available != n_samples:
                        idx = np.random.choice(n_available, size=n_samples, replace=False)
                    else:
                        idx = np.arange(n_available)

                sampled_pts = available_points[idx]  # Shape (N, D)
                # Broadcast to (B, N, D)
                self.sampled_points[tag] = np.broadcast_to(sampled_pts[np.newaxis, :, :], (batch_count, *sampled_pts.shape))  # Add batch dim: (1, N, D)  # Target: (B, N, D)
                if normals_avaiable and normals:
                    sampled_nrm = available_normals[idx]  # Shape (N, D)
                    self.sampled_points[f"n_{tag}"] = np.broadcast_to(sampled_nrm[np.newaxis, :, :], (batch_count, *sampled_nrm.shape))

            if self._verbose:
                if batch_count > 1:
                    self.log.info(f"Sampled {n_samples} x {batch_count} = {batch_count * n_samples} points for '{tag}' with shape {self.sampled_points[tag].shape}")
                else:
                    self.log.info(f"Sampled {n_samples} points for '{tag}'")

        if return_indices:
            return self.sampled_points[tag], idx, tag
        else:
            return self.sampled_points[tag], None, tag

    # Utilities

    def plot(self, save_path: str = "./runs/domain.png", figsize: Tuple[int, int] = (10, 8), show_normals: bool = True, arrow_scale: float = 0.05):
        """Plot the sampled points and normals.

        Args:
            name: Base name for saved figure
            figsize: Figure size (width, height)
            show_normals: Whether to display normal vectors as arrows
            arrow_scale: Scale factor for normal vector arrows
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        if not self.sampled_points:
            self.log.warning("No sampled points to plot")
            return

        # Get spatial dimension (exclude time)
        spatial_dim = len(self.spatial)

        # Create figure
        if spatial_dim == 3:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10.colors

        # Plot points by tag
        for i, (tag, points) in enumerate(self.sampled_points.items()):
            # Skip normal tags
            if tag.startswith("n_"):
                continue

            color = colors[i % len(colors)]
            n_points = points.shape[1]

            if spatial_dim == 1:
                # 1D: plot as points on a line
                if self._is_time_dependent:
                    ax.scatter(points[0, :, 0], points[0, :, 1], c=[color], s=10, alpha=0.7, label=f"{tag} ({n_points})")
                else:
                    ax.scatter(points[0, :, 0], np.zeros(n_points), c=[color], s=10, alpha=0.7, label=f"{tag} ({n_points})")

                # Plot normals if available
                if show_normals and f"n_{tag}" in self.sampled_points:
                    normals = self.sampled_points[f"n_{tag}"][0]  # (N, 1)
                    for j in range(n_points):
                        if self._is_time_dependent:
                            ax.arrow(
                                points[0, j, 0],
                                points[0, j, 1],
                                normals[j, 0] * arrow_scale,
                                0,
                                head_width=0.02,
                                head_length=0.01,
                                fc=color,
                                ec=color,
                                alpha=0.8,
                                linewidth=1.5,
                            )
                        else:
                            ax.arrow(
                                points[0, j, 0],
                                0,
                                normals[j, 0] * arrow_scale,
                                0,
                                head_width=0.02,
                                head_length=0.01,
                                fc=color,
                                ec=color,
                                alpha=0.8,
                                linewidth=1.5,
                            )

            elif spatial_dim == 2:
                # 2D: scatter plot
                ax.scatter(points[0, :, 0], points[0, :, 1], c=[color], s=10, alpha=0.7, label=f"{tag} ({n_points})")

                # Plot normals if available
                if show_normals and f"n_{tag}" in self.sampled_points:
                    normals = self.sampled_points[f"n_{tag}"][0]  # (N, 2)
                    ax.quiver(
                        points[0, :, 0],
                        points[0, :, 1],
                        normals[:, 0],
                        normals[:, 1],
                        color=color,
                        alpha=0.6,
                        scale=1 / arrow_scale,
                        width=0.003,
                        label=f"{tag} normals",
                    )

            elif spatial_dim == 3:
                # 3D: scatter plot
                ax.scatter(
                    points[0, :, 0],
                    points[0, :, 1],
                    points[0, :, 2],
                    c=[color],
                    s=10,
                    alpha=0.7,
                    label=f"{tag} ({n_points})",
                )

                # Plot normals if available
                if show_normals and f"n_{tag}" in self.sampled_points:
                    normals = self.sampled_points[f"n_{tag}"][0]  # (N, 3)
                    ax.quiver(
                        points[0, :, 0],
                        points[0, :, 1],
                        points[0, :, 2],
                        normals[:, 0],
                        normals[:, 1],
                        normals[:, 2],
                        color=color,
                        alpha=0.6,
                        length=arrow_scale,
                        normalize=True,
                        label=f"{tag} normals",
                    )

        # Set labels
        if spatial_dim == 1:
            ax.set_xlabel(self.spatial[0])
            if self._is_time_dependent:
                ax.set_ylabel("time")
            else:
                ax.set_ylabel("(placeholder)")
                ax.set_ylim(-0.1, 0.1)
        elif spatial_dim == 2:
            ax.set_xlabel(self.spatial[0])
            ax.set_ylabel(self.spatial[1])
            ax.set_aspect("equal")
        elif spatial_dim == 3:
            ax.set_xlabel(self.spatial[0])
            ax.set_ylabel(self.spatial[1])
            ax.set_zlabel(self.spatial[2])

        # Time info
        time_info = ""
        if self._is_time_dependent and spatial_dim > 1:
            if "t" in self.context:
                t_vals = self.context["t"]
                if t_vals.ndim == 3:
                    t_vals = t_vals[0, :, 0]
                elif t_vals.ndim == 2:
                    t_vals = t_vals[0, :]
                time_info = f" (t ∈ [{t_vals.min():.3f}, {t_vals.max():.3f}])"

        ax.set_title(f"Sampled Points: {time_info}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        import matplotlib.pyplot as plt  # already imported above, but safe

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.log.info(f"Saved domain plot to {save_path}")

    def save(self, filepath: str):
        """Save the trained core model to a file.

        Saves all trained parameters, layer info, operations, constraints,
        domain, and training history in a single file using dill for serialization.

        Args:
            filepath: Path to save file (e.g., "model.pkl" or "solution.dill")

        Example:
            sol = pino.solve(...)
            sol.save("trained_model.pkl")
        """
        import cloudpickle

        with open(filepath, "wb") as f:
            cloudpickle.dump(self, f)

        self.log.info(f"Model saved to: {filepath}")

        return None

    @classmethod
    def load(cls, filepath: str) -> "domain":
        """Load a trained core model from a file.

        Restores all trained parameters, operations, domain, and history.

        Args:
            filepath: Path to saved model file

        Returns:
            domain instance with trained parameters

        Example:
            sol = domain.load("trained_model.pkl")
        """
        import cloudpickle

        with open(filepath, "rb") as f:
            domain = cloudpickle.load(f)
        return domain
