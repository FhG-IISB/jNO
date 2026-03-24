from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.spatial import KDTree
import jax
import jax.numpy as jnp
import meshio


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
        # Use raytrace-based visibility for multi-connected domains (holes),
        # fall back to ordered method for simple single-loop boundaries.
        if dimension == 2 and "triangle" in mesh.cells_dict:
            bpe_global = MeshUtils.extract_boundary_edges(mesh.cells_dict["triangle"], len(bp))
            bpe_global = np.asarray(bpe_global)

            # Re-map edge indices from full-mesh space to boundary-only space
            global_to_local = {int(gi): li for li, gi in enumerate(boundary_indices)}
            bpe_local = np.array([[global_to_local[int(e[0])], global_to_local[int(e[1])]] for e in bpe_global if int(e[0]) in global_to_local and int(e[1]) in global_to_local])

            mesh_connectivity["boundary_edges"] = bpe_local
            mesh_connectivity["VM"] = MeshUtils.get_visibility_matrix_raytrace(bp, bpe_local, _bp[0], n_ray_samples=20)
        elif dimension <= 2:
            # 1-D domains: boundary is just 2 points; ordered visibility still works.
            mesh_connectivity["VM"] = MeshUtils.get_visibility_matrix_ordered(bp, _bp[0])
        else:
            # 3-D (and higher): the 2-D ordered visibility algorithm does not
            # generalise to higher-dimensional boundaries.  Store a trivial
            # all-visible placeholder so the rest of the pipeline keeps working.
            n_bp = len(bp)
            mesh_connectivity["VM"] = np.ones((n_bp, n_bp), dtype=np.float32) - np.eye(n_bp, dtype=np.float32)

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
    def get_visibility_matrix_raytrace(boundary_points, boundary_edges, interior_point=None, n_ray_samples: int = 3) -> jnp.ndarray:
        """
        Compute visibility matrix via segment–edge intersection tests.

        Two boundary points see each other if and only if the straight line
        between them does **not** cross any boundary edge (excluding the
        edges adjacent to the two endpoints).  This is exact for any closed
        2-D enclosure and avoids the fragile point-in-polygon sampling that
        the previous implementation relied on.

        The computation is fully vectorized over target points for each
        source point, giving O(N · E) work per source row.

        Parameters
        ----------
        boundary_points : array-like, shape (N, 2)
            Coordinates of the boundary discretisation points.
        boundary_edges : array-like, shape (E, 2)
            Index pairs into *boundary_points* defining the boundary segments.
        interior_point : ignored (kept for API compatibility)
        n_ray_samples : ignored (kept for API compatibility)

        Returns
        -------
        jnp.ndarray, shape (N, N)
            Binary visibility matrix (float32).  ``VM[i, j] = 1`` means
            point *i* can see point *j*.
        """
        import numpy as np
        import time

        P = np.asarray(boundary_points, dtype=np.float64)
        edges = np.asarray(boundary_edges, dtype=np.int32)
        n_bnd = P.shape[0]
        n_edges = edges.shape[0]

        E0 = P[edges[:, 0]]  # (n_edges, 2)
        E1 = P[edges[:, 1]]  # (n_edges, 2)

        t0 = time.time()

        # ==================================================================
        # Build adjacency: adj_mask[j, k] = True if edge k touches point j
        # ==================================================================
        adj_mask = np.zeros((n_bnd, n_edges), dtype=bool)
        for k in range(n_edges):
            adj_mask[edges[k, 0], k] = True
            adj_mask[edges[k, 1], k] = True

        # ==================================================================
        # Precompute edge directions and 2-D cross-product helper
        # ==================================================================
        edge_dir = E1 - E0  # (n_edges, 2)

        def cross2d(a, b):
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

        # ==================================================================
        # For each source point, test all target segments against all edges
        # ==================================================================
        VM = np.zeros((n_bnd, n_bnd), dtype=np.float32)

        for i in range(n_bnd):
            A = P[i]  # (2,)

            AB = P - A  # (n_bnd, 2)  — direction vectors to every target
            AB_exp = AB[:, None, :]  # (n_bnd, 1, 2)
            edge_exp = edge_dir[None, :, :]  # (1, n_edges, 2)
            diff_row = (E0 - A)[None, :, :]  # (1, n_edges, 2)

            denom = cross2d(AB_exp, edge_exp)  # (n_bnd, n_edges)
            parallel = np.abs(denom) < 1e-12
            denom_safe = np.where(parallel, 1.0, denom)

            t_seg = cross2d(diff_row, edge_exp) / denom_safe  # param on A→B
            t_edge = cross2d(diff_row, AB_exp) / denom_safe  # param on edge

            eps = 1e-10
            crossings = (~parallel) & (t_seg > eps) & (t_seg < 1 - eps) & (t_edge > eps) & (t_edge < 1 - eps)

            # Ignore edges that share an endpoint with source or target
            crossings[:, adj_mask[i]] = False  # edges touching source i
            crossings &= ~adj_mask  # edges touching each target j

            any_crossing = np.any(crossings, axis=1)  # (n_bnd,)

            visible = ~any_crossing
            visible[i] = False  # no self-visibility

            VM[i, :] = visible.astype(np.float32)

        elapsed = time.time() - t0

        return jnp.array(VM)

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

        triangles_np = np.asarray(triangles)

        # Collect all edges (sorted to make undirected)
        edges = []
        for tri in triangles_np:
            for k in range(3):
                e = tuple(sorted([tri[k], tri[(k + 1) % 3]]))
                edges.append(e)

        # Count occurrences
        from collections import Counter

        edge_count = Counter(edges)

        # Boundary edges appear exactly once
        boundary_edges = [list(e) for e, c in edge_count.items() if c == 1]  # type: ignore[misc]

        return jnp.array(boundary_edges)  # type: ignore[return-value]

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

        v = P[None, :, :] - P[:, None, :]
        r = jnp.linalg.norm(v, axis=-1)

        r_safe = r + jnp.eye(n_pts)
        r_hat = v / r_safe[..., None]

        cos_i = jnp.sum(Nrm[:, None, :] * r_hat, axis=-1)
        cos_j = -jnp.sum(Nrm[None, :, :] * r_hat, axis=-1)

        cos_i = jnp.maximum(0.0, cos_i)
        cos_j = jnp.maximum(0.0, cos_j)

        F_ij = (cos_i * cos_j) / (2.0 * r_safe)
        F_ij = F_ij * VM
        F_ij = F_ij * (1 - jnp.eye(n_pts))

        # include quadrature weights
        F_op = F_ij * ds[None, :]

        # enforce row sum = 1
        # row_sum = jnp.sum(F_op, axis=1, keepdims=True)
        F_op = F_op  # / row_sum

        return F_op

    @staticmethod
    def get_view_factor_1d(P, VM, Nrm, ds):
        n_pts = P.shape[0]
        return jnp.ones(n_pts)

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
