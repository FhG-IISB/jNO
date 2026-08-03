from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import KDTree


class MeshConnectivity(dict):
    """The mesh-connectivity mapping, with entries that may be computed on **first read**.

    An ordinary ``dict`` in every respect a caller can observe -- ``mc["VM"]``, ``mc.get("VM")`` and
    ``"VM" in mc`` all behave exactly as when the value was precomputed -- except that a registered
    entry costs nothing until something actually asks for it.

    This exists for ``VM``, the boundary-to-boundary visibility matrix. Preprocessing built it for
    every 2-D domain, while exactly one consumer reads it: the radiation view-factor path
    (``domain_class._compute_view_factors``, guarded by ``view_factor=``). It is not cheap and it is
    not linear -- it raytraces each of the ``n_b^2`` boundary pairs against every boundary edge, so
    the cost grows roughly as ``n_b^3``. Measured on a 2-D unit square, as a share of the whole
    domain build::

        boundary pts     80    160    320    432
        VM               10     30    185    519  ms
        share of build    4%    12%    27%    36%

    Doubling the boundary again would put it past 70%. The 3-D branch has the same problem in
    memory rather than time: its all-visible placeholder is a dense ``(n_b, n_b)`` array, which is
    hundreds of MB on a real 3-D mesh and, being a placeholder, is usually never read at all.

    Deferring rather than deleting is deliberate: a domain that *does* radiate still gets the same
    matrix, computed identically, at the moment it is needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._deferred = {}

    def defer(self, key, thunk):
        """Register ``key`` to be produced by ``thunk()`` the first time it is read."""
        self._deferred[key] = thunk

    def __missing__(self, key):  # only called by __getitem__, and only when the key is absent
        thunk = self._deferred.pop(key, None)
        if thunk is None:
            raise KeyError(key)
        self[key] = value = thunk()
        return value

    def __contains__(self, key):
        return super().__contains__(key) or key in self._deferred

    def get(self, key, default=None):  # dict.get does NOT consult __missing__
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return (super().keys() | self._deferred.keys()) if self._deferred else super().keys()


class MeshUtils:
    #: Memory budget, in float64 elements, for one row-block of the axisymmetric ring kernel's
    #: (M, M, n_phi) intermediate. Lower it if the view-factor build OOMs on a small GPU; it changes
    #: only how the work is split, never the result.
    _kernel_block_doubles = 2**24

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

        # Build neighbor lists by SORTING the edge list once, then slicing each vertex's block out of
        # it. The obvious loop -- `mask = edges[:, 0] == i` per vertex -- rescans every edge for every
        # point, which is O(n_points x n_edges): a 3-D tet mesh has 12 directed edges per cell, so at
        # 30932 points / 168277 cells that is ~6e10 comparisons and it measured 103.6 s, against 1.3 s
        # for gmsh to produce the mesh in the first place. Sorting is O(E log E) and the per-vertex
        # work then touches only that vertex's own edges.
        order = np.argsort(edges[:, 0], kind="stable")
        src_sorted = edges[order, 0]
        dst_sorted = edges[order, 1]
        # one boundary scan for all vertices, rather than a search per vertex
        bounds = np.searchsorted(src_sorted, np.arange(n_points + 1))
        neighbors = {
            i: np.unique(dst_sorted[bounds[i] : bounds[i + 1]]).tolist() if bounds[i + 1] > bounds[i] else []
            for i in range(n_points)
        }

        # Store connectivity info
        mesh_connectivity = MeshConnectivity(
            {
                "points": points,
                element_type: elements,
                "neighbors": neighbors,
                "n_points": n_points,
                "dimension": dimension,
            }
        )

        if dimension == 2:
            mesh_connectivity["p1_area"] = np.array(area)
            mesh_connectivity["p1_grad_phi"] = np.array(grad_phi)

        msg = f"Preprocessed mesh connectivity: {n_points} points, {len(elements)} {element_type}"

        mesh_connectivity["nodal_ds"] = MeshUtils.compute_nodal_ds(mesh_connectivity)
        mesh_connectivity["nodal_volumes"] = MeshUtils.compute_nodal_volumes(mesh_connectivity)
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
            bpe_local = np.array(
                [
                    [global_to_local[int(e[0])], global_to_local[int(e[1])]]
                    for e in bpe_global
                    if int(e[0]) in global_to_local and int(e[1]) in global_to_local
                ]
            )

            mesh_connectivity["boundary_edges"] = bpe_local
            # Deferred: ~n_b^3 raytracing, read only by the view-factor path. See MeshConnectivity.
            mesh_connectivity.defer(
                "VM", lambda: MeshUtils.get_visibility_matrix_raytrace(bp, bpe_local, _bp[0], n_ray_samples=20)
            )
        elif dimension <= 2:
            # 1-D domains: boundary is just 2 points; ordered visibility still works.
            mesh_connectivity.defer("VM", lambda: MeshUtils.get_visibility_matrix_ordered(bp, _bp[0]))
        else:
            # 3-D (and higher): the 2-D ordered visibility algorithm does not
            # generalise to higher-dimensional boundaries.  Store a trivial
            # all-visible placeholder so the rest of the pipeline keeps working.
            # Deferred because the placeholder is dense (n_b, n_b) -- hundreds of MB on a real 3-D
            # mesh, allocated for every domain, and read by nothing unless radiation is in play.
            n_bp = len(bp)
            mesh_connectivity.defer("VM", lambda: np.ones((n_bp, n_bp), dtype=np.float32) - np.eye(n_bp, dtype=np.float32))

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
        dimension = mesh_connectivity["dimension"]
        points = mesh_connectivity["points"]
        n_points = mesh_connectivity["n_points"]

        def scatter(nodes, measure):
            """Split each element's measure equally over its own nodes and accumulate."""
            per_node = np.repeat(measure / nodes.shape[1], nodes.shape[1])
            return np.bincount(nodes.ravel(), weights=per_node, minlength=n_points)

        if dimension == 1:
            # Boundary = endpoints, ds = half of adjacent line element
            segments = np.asarray(mesh_connectivity["lines"])
            lengths = np.linalg.norm(points[segments[:, 1]] - points[segments[:, 0]], axis=-1)
            ds = scatter(segments, lengths)

        elif dimension == 2:
            # Boundary = edges that appear once in the triangulation
            edges = MeshUtils._get_boundary_elements(np.asarray(mesh_connectivity["triangles"]), "triangle")
            lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=-1)
            ds = scatter(edges, lengths)

        elif dimension == 3:
            # Boundary = faces that appear once in the tetrahedralization
            faces = MeshUtils._get_boundary_elements(np.asarray(mesh_connectivity["tetrahedra"]), "tetra")
            p0, p1, p2 = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
            areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=-1)
            ds = scatter(faces, areas)

        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        if boundary_indices is not None:
            return ds[boundary_indices]

        return ds

    @staticmethod
    def compute_nodal_volumes(mesh_connectivity: dict) -> np.ndarray:
        """Per-node volume weights for interior integration.

        Mirrors ``compute_nodal_ds`` for boundary, but distributes element
        volume/area/length to all interior nodes:

        - 1D: ½ of adjacent segment lengths (trapezoidal rule)
        - 2D: ⅓ of incident triangle areas
        - 3D: ¼ of incident tetrahedron volumes

        Called once during mesh preprocessing and stored as
        ``mesh_connectivity["nodal_volumes"]``.
        """
        dimension = mesh_connectivity["dimension"]
        points = mesh_connectivity["points"]
        n_points = mesh_connectivity["n_points"]

        if dimension == 1:
            cells = np.asarray(mesh_connectivity["lines"])
            measure = np.linalg.norm(points[cells[:, 1]] - points[cells[:, 0]], axis=-1)

        elif dimension == 2:
            cells = np.asarray(mesh_connectivity["triangles"])
            a, b, c = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
            ba, ca = b - a, c - a
            measure = 0.5 * np.abs(ba[:, 0] * ca[:, 1] - ba[:, 1] * ca[:, 0])

        elif dimension == 3:
            cells = np.asarray(mesh_connectivity["tetrahedra"])
            a, b, c, d = (points[cells[:, i]] for i in range(4))
            measure = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0

        else:
            raise ValueError(f"Unsupported dimension: {dimension}")

        # each of an element's nodes takes an equal share of its measure
        n_local = cells.shape[1]
        return np.bincount(cells.ravel(), weights=np.repeat(measure / n_local, n_local), minlength=n_points)

    @staticmethod
    def get_boundary_normals(mesh, k=8):
        points = mesh.points
        if "tetra" in mesh.cells_dict:
            bfaces, bapex = MeshUtils._boundary_faces_with_apex(mesh.cells_dict["tetra"])
            return MeshUtils._compute_normals_from_boundary_faces(points, bfaces, apex_points=points[bapex])
        elif "triangle" in mesh.cells_dict:
            boundary_elements = MeshUtils._get_boundary_elements(mesh.cells_dict["triangle"], "triangle")
            actual_dim = 2
        else:
            raise ValueError("Unsupported mesh type.")

        boundary_indices = np.unique(boundary_elements)
        return MeshUtils._compute_normals_pca(points, boundary_indices, actual_dim, k, mesh=mesh)

    @staticmethod
    def _boundary_faces_with_apex(tetra_cells):
        """Boundary faces of a tetrahedral mesh, each with the apex of its owning element.

        A boundary face is shared by exactly one tetrahedron; that tet's fourth vertex (the
        one not on the face) is the ``apex``. Returns ``(faces, apex)`` where ``faces`` are
        node triples and ``apex`` the per-face opposite-vertex node index -- enough to orient
        each face outward exactly (away from the apex), for any geometry incl. concave ones.
        """
        cells = np.asarray(tetra_cells, dtype=np.int64)
        specs = ([0, 1, 2], 3), ([0, 1, 3], 2), ([0, 2, 3], 1), ([1, 2, 3], 0)
        faces = np.vstack([cells[:, list(tri)] for tri, _ in specs])
        apex = np.concatenate([cells[:, a] for _, a in specs])
        keys = np.sort(faces, axis=1)
        _uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        boundary = counts[inv.ravel()] == 1
        return faces[boundary], apex[boundary]

    @staticmethod
    def _compute_normals_from_boundary_faces(points, boundary_faces, apex_points=None):
        """Compute robust 3D boundary vertex normals from boundary triangle faces.

        When ``apex_points`` is given -- one point per face, the opposite vertex of the single
        volume element that owns that face -- each face normal is oriented to point *away* from
        that vertex. This is exact for any geometry, including concave boundaries (a roll
        contact arc, an L-shape, a T-junction). Without it, faces are oriented by the global
        mesh centroid, which is valid only for convex / star-shaped domains and is kept as a
        fallback for callers that have no volume connectivity. Normals are accumulated per
        vertex (area-weighted via unnormalized face normals).
        """
        pts = np.asarray(points[:, :3], dtype=np.float64)
        faces = np.asarray(boundary_faces, dtype=np.int64)
        if faces.size == 0:
            return np.zeros((0, 3), dtype=np.float64), np.array([], dtype=np.int64)

        apex = None if apex_points is None else np.asarray(apex_points[:, :3], dtype=np.float64)
        centroid = np.mean(pts, axis=0)
        vnorm = np.zeros_like(pts)
        eps = 1e-20

        for k, f in enumerate(faces):
            i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
            p0, p1, p2 = pts[i0], pts[i1], pts[i2]

            # Unnormalized normal magnitude is 2*face_area.
            n = np.cross(p1 - p0, p2 - p0)
            nlen = np.linalg.norm(n)
            if nlen < eps:
                continue

            fc = (p0 + p1 + p2) / 3.0
            # Outward = away from the owning element's apex (exact); else away from centroid.
            ref = centroid if apex is None else apex[k]
            if np.dot(n, fc - ref) < 0.0:
                n = -n

            vnorm[i0] += n
            vnorm[i1] += n
            vnorm[i2] += n

        boundary_indices = np.unique(faces.ravel())
        out = vnorm[boundary_indices]
        lens = np.linalg.norm(out, axis=1, keepdims=True)

        # Fallback for numerically degenerate vertices.
        bad = lens[:, 0] < eps
        if np.any(bad):
            radial = pts[boundary_indices[bad]] - centroid
            radial_len = np.linalg.norm(radial, axis=1, keepdims=True)
            radial_len[radial_len < eps] = 1.0
            out[bad] = radial / radial_len
            lens[bad] = 1.0

        out = out / lens
        return out, boundary_indices

    @staticmethod
    def _get_boundary_elements(cells, cell_type):
        """Finds elements (lines/triangles) that appear only once."""
        if cell_type == "tetra":
            faces = np.sort(
                np.vstack(
                    [
                        cells[:, [0, 1, 2]],
                        cells[:, [0, 1, 3]],
                        cells[:, [0, 2, 3]],
                        cells[:, [1, 2, 3]],
                    ]
                ),
                axis=1,
            )
        else:  # triangle
            faces = np.sort(
                np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]),
                axis=1,
            )

        unique_elements, counts = np.unique(faces, axis=0, return_counts=True)
        return unique_elements[counts == 1]

    @staticmethod
    def _points_in_polygon_2d(pts, edges, all_points, block=1 << 22):
        """Ray-casting point-in-polygon test, every point against every edge at once.

        Crossing-number rule (Shimrat, "Algorithm 112: Position of point relative to polygon",
        Comm. ACM 5(8), 1962): a point is inside when a ray cast along +x crosses the boundary an
        odd number of times.

        Parameters
        ----------
        pts : (m, 2) array of query points.
        edges : (n_edges, 2) array of indices into `all_points`, or None.
        all_points : (n, >=2) array holding the edge endpoints.
        block : int, cap on the (points x edges) temporary, processed in chunks of this many pairs.

        Returns
        -------
        (m,) bool array, True where the point is inside. All-True when `edges` is None, matching
        the caller's "no boundary information, assume inside" fallback.
        """
        pts = np.atleast_2d(pts)
        if edges is None or len(edges) == 0:
            return np.ones(len(pts), dtype=bool)

        edges = np.asarray(edges)
        x0, y0 = all_points[edges[:, 0], 0], all_points[edges[:, 0], 1]
        x1, y1 = all_points[edges[:, 1], 0], all_points[edges[:, 1], 1]
        dy = y1 - y0
        # a horizontal edge can never be crossed; keep it out of the divisor rather than
        # dividing by zero and masking the nan afterwards
        safe_dy = np.where(dy == 0.0, 1.0, dy)
        crossable = dy != 0.0

        rows = max(1, block // max(len(edges), 1))
        inside = np.empty(len(pts), dtype=bool)
        for lo in range(0, len(pts), rows):
            x = pts[lo : lo + rows, 0:1]
            y = pts[lo : lo + rows, 1:2]
            straddles = (y0 > y) != (y1 > y)
            x_int = x0 + (x1 - x0) * (y - y0) / safe_dy
            crossings = np.count_nonzero(straddles & crossable & (x < x_int), axis=1)
            inside[lo : lo + rows] = crossings % 2 == 1
        return inside

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
        neighbors = neighbors.reshape(len(coords), -1)  # k=1 drops the neighbour axis

        mesh_centroid = np.mean(points[:, :dim], axis=0)

        # Get boundary edges for point-in-polygon test
        boundary_edges = None
        if dim == 2 and mesh is not None and "triangle" in mesh.cells_dict:
            boundary_edges = MeshUtils._get_boundary_elements(mesh.cells_dict["triangle"], "triangle")

        # --- normals: one batched SVD over every neighbourhood patch ---------------------------
        # KDTree returns a fixed k, so the patches stack into a single (n_boundary, k, dim) array
        # and the decomposition batches. The last right-singular vector is the direction of least
        # variance, i.e. the surface normal.
        patches = coords[neighbors]
        _, _, vh = np.linalg.svd(patches - patches.mean(axis=1, keepdims=True))
        v_normals = vh[:, -1, :]
        v_normals = v_normals / (np.linalg.norm(v_normals, axis=1, keepdims=True) + 1e-12)

        # --- orientation: a normal points OUT, i.e. away from the material ---------------------
        # Step a short way along each normal and against it, and keep the direction that lands
        # outside the domain.
        step_size = 1e-4 * np.max(np.abs(coords.max(axis=0) - coords.min(axis=0)))
        if dim == 2:
            inside_positive = MeshUtils._points_in_polygon_2d(
                coords + step_size * v_normals, boundary_edges, points[:, :dim]
            )
            inside_negative = MeshUtils._points_in_polygon_2d(
                coords - step_size * v_normals, boundary_edges, points[:, :dim]
            )
            flip = inside_positive & ~inside_negative
            # both sides inside, or both outside -- the test is inconclusive here
            undecided = inside_positive == inside_negative
        else:
            flip = np.zeros(len(coords), dtype=bool)
            undecided = np.ones(len(coords), dtype=bool)

        # 3D, no boundary edges, and inconclusive 2D points fall back to the centroid heuristic
        outward = np.einsum("ij,ij->i", v_normals, coords - mesh_centroid) < 0
        flip = np.where(undecided, outward, flip)

        return np.where(flip[:, None], -v_normals, v_normals), boundary_indices

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

                    visible_ij = jax.lax.cond(
                        is_same,
                        lambda: False,
                        lambda: jax.lax.cond(is_adjacent_point, lambda: True, lambda: seg_visible(i, j)),
                    )  # Diagonal is always 0 (can't see itself)  # Adjacent boundary points are always visible
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
    def get_visibility_matrix_raytrace(
        boundary_points, boundary_edges, interior_point=None, n_ray_samples: int = 3
    ) -> jnp.ndarray:
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

        P = np.asarray(boundary_points, dtype=np.float64)
        edges = np.asarray(boundary_edges, dtype=np.int32)
        n_bnd = P.shape[0]
        n_edges = edges.shape[0]

        E0 = P[edges[:, 0]]  # (n_edges, 2)
        E1 = P[edges[:, 1]]  # (n_edges, 2)

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
            Unused; kept because callers pass it positionally.

        Returns
        -------
        jnp.ndarray
            (n_boundary_edges, 2) boundary edge indices, each sorted low-to-high. The edges come
            back in lexicographic order rather than first-encountered order -- they describe an
            unordered edge set, and the callers either re-index or chain them into loops.
        """
        return jnp.array(MeshUtils._get_boundary_elements(np.asarray(triangles), "triangle"))

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
    def get_view_factor_axisymmetric(P, VM, Nrm, ds, n_phi: int = 16):
        r"""Axisymmetric (cylindrical) point-to-point view-factor matrix ``F_op[i, j]``.

        For a body of revolution the enclosure is described in the meridional ``(r, z)`` half-plane
        (``P[:, 0] = r``, ``P[:, 1] = z``); by rotational symmetry the receiver ``i`` is fixed at
        azimuth ``phi = 0`` and the source ``j`` is a full ring, integrated over the azimuthal angle:

        .. math::
            F_{ij} \approx r_j \, \mathrm{ds}_j \, \frac{1}{n_\phi}
                \sum_{m} \frac{\cos\theta_i(\phi_m)\,\cos\theta_j(\phi_m)}{\pi R(\phi_m)^2}

        a midpoint quadrature (``n_phi`` uniform samples) of the diffuse point-to-ring kernel. The
        factor ``r_j`` is the cylindrical Jacobian of the ring. Only the self-pair (diagonal) is
        removed; same-surface concave self-view is left to the visibility matrix ``VM``.

        Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4 (view factors;
        bodies of revolution / the crossed-strings and ring-integration constructions).
        """
        n = P.shape[0]
        r = P[:, 0]
        z = P[:, 1]
        nr = Nrm[:, 0]
        nz = Nrm[:, 1]

        phi = jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, endpoint=False)  # (M,)

        # Shape expansions: (N_i, N_j, M)
        r_i, z_i, nr_i, nz_i = (a[:, None, None] for a in (r, z, nr, nz))
        r_j, z_j, nr_j, nz_j = (a[None, :, None] for a in (r, z, nr, nz))
        phi_m = phi[None, None, :]

        # Displacement from receiver i (at phi=0) to source j (at angle phi_m)
        dx = r_j * jnp.cos(phi_m) - r_i  # (N_i, N_j, M)
        dy = r_j * jnp.sin(phi_m)
        dz = z_j - z_i

        R2 = dx**2 + dy**2 + dz**2
        R = jnp.sqrt(R2 + 1e-30)

        # cos theta_i: n_i (2D, in the r-z plane at phi=0) dotted with the 3D direction
        cos_i = (nr_i * dx + nz_i * dz) / R
        # cos theta_j: n_j rotated into 3D = (nr_j cos phi, nr_j sin phi, nz_j); take negative dot
        dot_j = nr_j * jnp.cos(phi_m) * dx + nr_j * jnp.sin(phi_m) * dy + nz_j * dz
        cos_j = -dot_j / R

        cos_i = jnp.maximum(0.0, cos_i)
        cos_j = jnp.maximum(0.0, cos_j)

        # Azimuthal integral of the diffuse kernel: int_0^2pi (...) dphi ~= (2pi/n_phi) * sum_m (...).
        # (A plain mean would compute the ring AVERAGE and underestimate the view factor by 2*pi.)
        dphi = 2.0 * jnp.pi / n_phi
        kernel = dphi * jnp.sum(cos_i * cos_j / (jnp.pi * R2 + 1e-30), axis=-1)  # (N_i, N_j)
        F_op = kernel * r[None, :] * VM * ds[None, :]
        F_op = F_op * (1 - jnp.eye(n))
        return F_op

    @staticmethod
    def get_view_factor_1d(P, VM, Nrm, ds):
        n_pts = P.shape[0]
        return jnp.ones(n_pts)

    @staticmethod
    def get_view_factor_2d_element(E0, E1, Nrm, VM, n_quad: int = 3):
        r"""Element-based 2D view-factor matrix ``F[i, j]`` by double-area Gauss quadrature.

        Each radiating boundary element is a straight segment ``[E0_k, E1_k]`` with constant outward
        normal ``Nrm_k`` (pointing into the enclosure). The diffuse exchange factor between elements is

        .. math::
            F_{ij} = \frac{1}{L_i} \int_{e_i}\!\int_{e_j}
                      \frac{\cos\theta_i\,\cos\theta_j}{2\,r}\; \mathrm{d}s_j\,\mathrm{d}s_i ,

        evaluated with ``n_quad`` Gauss-Legendre points per element (``n_quad=1`` reduces to the
        midpoint/point kernel). Integrating over the element extent (rather than a single point) is what
        makes the near-field self-view of concave surfaces accurate. ``VM`` is the element-to-element
        visibility (0/1) carrying occlusion; only the self-pair (diagonal) is otherwise removed.

        Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4 (diffuse view factors).
        """
        E0 = jnp.asarray(E0)
        E1 = jnp.asarray(E1)
        Nrm = jnp.asarray(Nrm)
        m = E0.shape[0]

        gx, gw = np.polynomial.legendre.leggauss(int(n_quad))  # nodes/weights on [-1, 1]
        s = jnp.asarray((gx + 1.0) * 0.5)  # -> [0, 1]   (n_quad,)
        wq = jnp.asarray(gw * 0.5)  # weights sum to 1 on [0, 1]

        length = jnp.linalg.norm(E1 - E0, axis=-1)  # (m,)
        qp = E0[:, None, :] + s[None, :, None] * (E1 - E0)[:, None, :]  # (m, n_quad, 2)
        qw = wq[None, :] * length[:, None]  # (m, n_quad) -> sums to L_i over the element
        qp = qp.reshape(m * int(n_quad), 2)
        qw = qw.reshape(-1)  # (M,)
        qn = jnp.repeat(Nrm, int(n_quad), axis=0)  # (M, 2) element normal per quad point

        mq = qp.shape[0]
        v = qp[None, :, :] - qp[:, None, :]  # (M, M, 2)
        r = jnp.linalg.norm(v, axis=-1)
        r_safe = r + jnp.eye(mq)
        rh = v / r_safe[..., None]
        cos_i = jnp.maximum(0.0, jnp.sum(qn[:, None, :] * rh, axis=-1))
        cos_j = jnp.maximum(0.0, -jnp.sum(qn[None, :, :] * rh, axis=-1))
        kernel = cos_i * cos_j / (2.0 * r_safe)  # (M, M)

        g = qw[:, None] * qw[None, :] * kernel  # (M, M) quad-pair contributions
        # Block-sum quad pairs back to elements (quads are grouped contiguously per element).
        F = g.reshape(m, int(n_quad), m, int(n_quad)).sum(axis=(1, 3)) / length[:, None]  # (m, m)
        F = F * jnp.asarray(VM) * (1.0 - jnp.eye(m))
        return F

    @staticmethod
    def meridional_quad_points(E0, E1, n_quad: int = 3):
        """Gauss-Legendre quadrature points along each meridional element -> ``(m*n_quad, 2)``.

        The exact points :meth:`get_view_factor_axisymmetric_element` integrates on (grouped
        contiguously per element, so ``reshape(m, n_quad, ...)`` recovers the element blocks). Exposed so
        an occlusion test can be evaluated at the SAME points the kernel uses, rather than once per
        element at its midpoint -- element-level visibility makes a partially-shadowed element
        all-or-nothing, which converges only at O(h)."""
        E0, E1 = np.asarray(E0), np.asarray(E1)
        gx, _ = np.polynomial.legendre.leggauss(int(n_quad))
        s = (gx + 1.0) * 0.5  # -> [0, 1]
        m = E0.shape[0]
        return (E0[:, None, :] + s[None, :, None] * (E1 - E0)[:, None, :]).reshape(m * int(n_quad), 2)

    @staticmethod
    def get_view_factor_axisymmetric_element(E0, E1, Nrm, VM, n_quad: int = 3, n_phi: int = 16, r_min: float = 0.0):
        r"""Element-based axisymmetric view-factor matrix ``F[i, j]`` for a body of revolution.

        ``r_min`` softens the near-field ``1/R^2`` singularity (``R^2 -> R^2 + r_min^2``); set it to a
        fraction of the element size to keep close/coincident element pairs finite (else they blow up).

        Each element is a meridional segment ``[E0_k, E1_k]`` in the ``(r, z)`` half-plane (a frustum
        ring when revolved), with constant normal ``Nrm_k`` pointing into the enclosure. The exchange
        factor combines **meridional element quadrature** (``n_quad`` Gauss points along each element,
        as in :meth:`get_view_factor_2d_element`) with **azimuthal integration** of the diffuse
        point-to-ring kernel (``n_phi`` samples; the integral is ``(2π/n_phi)·Σ``, *not* a mean):

        .. math::
            F_{ij} = \frac{\sum_{q\in e_i} r_q w_q \big[\sum_{p\in e_j} r_p w_p
                      \,(2\pi/n_\phi)\textstyle\sum_\phi \cos\theta_q \cos\theta_p/(\pi R^2)\big]}
                     {\sum_{q\in e_i} r_q w_q}

        i.e. the source-ring view factor (area-weighted by ``r_p w_p``) averaged over the receiver
        element's rings (weighted by ``r_q w_q``). ``n_quad=1`` recovers the single-point-per-ring kernel.

        ``VM`` is the visibility (occlusion) mask and accepts three shapes:

        * ``(m, m)`` — a single azimuth-blind flag per element pair (the historical behaviour): applied
          as one constant mask *after* the azimuthal sum. This is only exact at ``phi = 0``, i.e. it
          silently assumes that whatever occludes (or doesn't) the same-meridian chord occludes (or
          doesn't) the chord at *every* azimuthal offset — not true in general for a solid of revolution,
          since the true 3-D chord's distance from the axis is strictly smaller than the flat
          ``(r, z)``-plane projection for any nonzero azimuth (it "cuts the corner" toward the axis).
        * ``(m, m, n_phi)`` — per-azimuth visibility per element PAIR, applied **inside** the azimuthal
          sum before it is reduced: occlusion is checked at the same azimuth the exchange is integrated
          over. Correct, but decided once per element (at its midpoint), so a partially-shadowed element
          is all-or-nothing and the shadow boundary is only resolved to O(element size).
        * ``(m*n_quad, m*n_quad, n_phi)`` — per-azimuth visibility at the **quadrature points** (see
          :meth:`meridional_quad_points`). Same as above but resolves the shadow boundary within an
          element, so a partially-shadowed element is partially shadowed. Preferred.

        Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4 (bodies of revolution).
        """
        E0 = jnp.asarray(E0)
        E1 = jnp.asarray(E1)
        Nrm = jnp.asarray(Nrm)
        VM = jnp.asarray(VM)
        phi_resolved = VM.ndim == 3
        m = E0.shape[0]
        nq = int(n_quad)

        gx, gw = np.polynomial.legendre.leggauss(nq)
        s = jnp.asarray((gx + 1.0) * 0.5)
        wq = jnp.asarray(gw * 0.5)
        length = jnp.linalg.norm(E1 - E0, axis=-1)  # (m,) meridional length
        qp = (E0[:, None, :] + s[None, :, None] * (E1 - E0)[:, None, :]).reshape(m * nq, 2)  # (M, 2)
        w_merid = (wq[None, :] * length[:, None]).reshape(-1)  # (M,) meridional ds per quad
        qn = jnp.repeat(Nrm, nq, axis=0)  # (M, 2)

        r = qp[:, 0]
        z = qp[:, 1]
        nr = qn[:, 0]
        nz = qn[:, 1]
        phi = jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, endpoint=False)
        dphi = 2.0 * jnp.pi / n_phi

        r_j, z_j, nr_j, nz_j = (a[None, :, None] for a in (r, z, nr, nz))
        phi_m = phi[None, None, :]
        cos_phi, sin_phi = jnp.cos(phi_m), jnp.sin(phi_m)
        if phi_resolved:
            # Occlusion checked AT the azimuth being integrated, not just phi=0. Quadrature-resolved
            # visibility is used as-is; element-level visibility is broadcast up to the quad points.
            vm_full = VM if VM.shape[0] == m * nq else jnp.repeat(jnp.repeat(VM, nq, axis=0), nq, axis=1)

        # The point-to-point kernel is (M, M, n_phi) -- 816 MB at M = 1785, n_phi = 32, which OOMs an
        # 8 GB GPU before the azimuthal sum ever reduces it away. It is only ever needed one receiver
        # row-block at a time (the reduction is over the azimuth and the source), so build it in
        # blocks: same arithmetic per element, memory bounded by `chunk` instead of M. Not bit-identical
        # to the unblocked build -- the azimuthal reduction runs over a differently-shaped array and XLA
        # reassociates it, measured at 3 ULP (see the row-chunking test), which is rounding, not drift.
        M = r.shape[0]
        budget = int(getattr(MeshUtils, "_kernel_block_doubles", 2**24))  # ~16M doubles = 128 MB
        chunk = max(1, min(M, budget // max(M * n_phi, 1)))
        blocks = []
        for a0 in range(0, M, chunk):
            b0 = min(a0 + chunk, M)
            r_i, z_i, nr_i, nz_i = (a[a0:b0, None, None] for a in (r, z, nr, nz))
            dx = r_j * cos_phi - r_i
            dy = r_j * sin_phi
            dz = z_j - z_i
            R2 = dx**2 + dy**2 + dz**2 + r_min**2
            R = jnp.sqrt(R2 + 1e-30)
            cos_i = jnp.maximum(0.0, (nr_i * dx + nz_i * dz) / R)
            cos_j = jnp.maximum(0.0, -(nr_j * cos_phi * dx + nr_j * sin_phi * dy + nz_j * dz) / R)
            kern = cos_i * cos_j / (jnp.pi * R2 + 1e-30)
            if phi_resolved:
                kern = kern * vm_full[a0:b0]
            blocks.append(dphi * jnp.sum(kern, axis=-1))
        ring = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=0)  # (M, M)

        # Differential view factor ring_q -> ring_p (source weighted by its ring area r_p * w_p):
        fqq = ring * (r[None, :] * w_merid[None, :])  # (M, M)
        # Aggregate quads -> elements: area-weighted over receiver rings (r_q w_q), summed over source.
        a_q = r * w_merid  # (M,) ring-area weight (2*pi cancels in the ratio)
        num = (a_q[:, None] * fqq).reshape(m, nq, m, nq).sum(axis=(1, 3))  # (m, m)
        a_elem = a_q.reshape(m, nq).sum(axis=1)  # (m,)
        F = num / a_elem[:, None]
        if phi_resolved:
            # Occlusion already applied per-azimuth above. The diagonal is KEPT: unlike a flat 2-D
            # segment (which cannot see itself), a RING sees itself around the azimuth, and on a concave
            # surface that is real exchange -- zeroing it discards energy and is precisely what makes the
            # raw row sums fall short of 1 (confirmed against a 3-D Monte-Carlo ray trace). Its phi = 0
            # term is harmless: the chord degenerates and the cosines vanish.
            pass
        else:
            F = F * VM * (1.0 - jnp.eye(m))
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
