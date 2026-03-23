"""Mesh-based finite-difference differential operators.

Provides :class:`DifferentialOperators`, a collection of static methods that
compute gradients, Laplacians and Hessians on 1-D line meshes, 2-D triangular
meshes, and 3-D tetrahedral meshes.

Gradient methods
----------------
Three averaging strategies are available through the ``method`` argument on
every ``compute_fd_gradient_*`` function:

``"area_weighted"`` *(default)*
    Gradient computed per element, then averaged to nodes weighted by
    element area / volume.  Fast and robust.

``"uniform"``
    Same as ``"area_weighted"`` but with equal weight for every incident
    element.  Slightly cheaper; good for nearly uniform meshes.

``"inverse_distance"``
    Weight each incident element by ``1 / dist(node, centroid)``.
    Useful when element sizes vary strongly (graded meshes).

``"least_squares"``
    Per-node weighted least-squares fit to the local patch of element
    centroids.  Solves a **2×2** (2-D) or **3×3** (3-D) system at each
    node via Cramer's rule using only JAX scatter operations.  Most
    accurate on irregular meshes.

Laplacian methods (2-D)
-----------------------
``"gradient_of_gradient"`` *(default)*
    Apply the gradient operator twice.  Simple and works in all
    dimensions, first-order on irregular meshes.

``"cotangent"`` *(2-D triangular meshes only)*
    The cotangent-weight (Laplace–Beltrami) formula — the gold standard
    for triangle meshes.  Second-order accurate and isotropic on
    irregular meshes.

``"lsq_of_gradient"``
    Apply the LSQ gradient twice.

Usage from the trace API
------------------------
Select the sub-method by appending it to the scheme string with ``:``:

.. code-block:: python

    du_dx = u.d(x, scheme="finite_difference:least_squares")
    lap_u = u.laplacian(x, y, scheme="finite_difference:cotangent")

The default ``"finite_difference"`` (no suffix) keeps the existing
``"area_weighted"`` / ``"gradient_of_gradient"`` behaviour.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


class DifferentialOperators:
    """Static collection of mesh-based FD operators (1-D, 2-D, 3-D).

    All public methods are *static* — the class is used purely as a namespace.
    See the module docstring for full method descriptions.
    """

    # ══════════════════════════════════════════════════════════════════
    # 1-D line mesh
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_fd_gradient_1d_simple(
        u_values: jnp.ndarray,
        points: jnp.ndarray,
        lines,
        method: str = "area_weighted",
    ) -> jnp.ndarray:
        """Gradient on a 1-D line mesh.

        Args:
            u_values: Function values at mesh points, shape ``(N,)``.
            points:   Mesh point coordinates, shape ``(N, 1)`` or ``(N,)``.
            lines:    Line element connectivity, shape ``(M, 2)``.
            method:   Weighting strategy — one of ``"area_weighted"``
                      (default), ``"uniform"``, ``"inverse_distance"``,
                      ``"least_squares"``.

        Returns:
            ``du/dx`` at each point, shape ``(N,)``.
        """
        if method == "least_squares":
            return DifferentialOperators._gradient_1d_lsq(u_values, points, lines)

        n_points = len(u_values)
        lines = jnp.array(lines)
        i_idx, j_idx = lines[:, 0], lines[:, 1]

        if points.ndim == 2:
            x0, x1 = points[i_idx, 0], points[j_idx, 0]
        else:
            x0, x1 = points[i_idx], points[j_idx]

        u0, u1 = u_values[i_idx], u_values[j_idx]
        dx = x1 - x0
        lengths = jnp.abs(dx)
        grads = (u1 - u0) / (dx + 1e-12)
        grads = jnp.where(lengths > 1e-12, grads, 0.0)

        if method == "uniform":
            w_elem = jnp.where(lengths > 1e-12, 1.0, 0.0)
        elif method == "inverse_distance":
            half = lengths / 2.0
            w_elem = jnp.where(lengths > 1e-12, 1.0 / (half + 1e-12), 0.0)
        else:  # area_weighted
            w_elem = jnp.where(lengths > 1e-12, lengths, 0.0)

        contributions = grads * w_elem
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(w_elem).at[j_idx].add(w_elem)
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def _gradient_1d_lsq(
        u_values: jnp.ndarray,
        points: jnp.ndarray,
        lines: np.ndarray,
    ) -> jnp.ndarray:
        """Least-squares gradient on a 1-D line mesh (internal helper)."""
        # In 1-D the WLS system reduces to a single weighted average of (du/dx)_element.
        # We use length-weighted WLS which is equivalent to the Green-Gauss approach
        # on a 1-D mesh, so this is exactly the area_weighted result but spelled out
        # as an explicit least-squares derivation.
        return DifferentialOperators.compute_fd_gradient_1d_simple(u_values, points, lines, method="area_weighted")

    @staticmethod
    def compute_fd_laplacian_1d_simple(
        u_values,
        points,
        lines,
        method: str = "gradient_of_gradient",
    ):
        """Laplacian on a 1-D line mesh.

        Args:
            u_values: Function values, shape ``(N,)``.
            points:   Coordinates, shape ``(N, 1)`` or ``(N,)``.
            lines:    Line connectivity, shape ``(M, 2)``.
            method:   ``"gradient_of_gradient"`` (only option in 1-D).

        Returns:
            ``d²u/dx²`` at each point, shape ``(N,)``.
        """
        grad = DifferentialOperators.compute_fd_gradient_1d_simple(u_values, points, lines)
        return DifferentialOperators.compute_fd_gradient_1d_simple(grad, points, lines)

    @staticmethod
    def compute_fd_hessian_1d_simple(u_values, points, lines, var_dims=None):
        """Hessian (= d²u/dx²) on a 1-D line mesh.

        Args:
            u_values: Function values, shape ``(N,)``.
            points:   Coordinates, shape ``(N, 1)`` or ``(N,)``.
            lines:    Line connectivity, shape ``(M, 2)``.
            var_dims: Optional ``[(i, vi_dim, j, vj_dim), …]``.

        Returns:
            Hessian, shape ``(N, 1, 1)``.
        """
        N = len(u_values)
        grad_x = DifferentialOperators.compute_fd_gradient_1d_simple(u_values, points, lines)
        d2u_dx2 = DifferentialOperators.compute_fd_gradient_1d_simple(grad_x, points, lines)
        hess_full = jnp.zeros((N, 1, 1)).at[:, 0, 0].set(d2u_dx2)

        if var_dims is not None:
            n_vars = int(jnp.sqrt(len(var_dims)))
            result = jnp.zeros((N, n_vars, n_vars))
            for i, vi_dim, j, vj_dim in var_dims:
                result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])
            return result
        return hess_full

    # ══════════════════════════════════════════════════════════════════
    # 2-D triangular mesh
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_fd_gradient_2d_simple(
        u_values: jnp.ndarray,
        points: jnp.ndarray,
        triangles,
        dim: int,
        method: str = "area_weighted",
    ) -> jnp.ndarray:
        """Gradient on a 2-D triangular mesh.

        Args:
            u_values:  Function values at mesh points, shape ``(N,)``.
            points:    Mesh point coordinates, shape ``(N, 2)``.
            triangles: Triangle connectivity, shape ``(M, 3)``.
            dim:       Spatial dimension to differentiate (0 = x, 1 = y).
            method:    ``"area_weighted"`` (default), ``"uniform"``,
                       ``"inverse_distance"``, ``"least_squares"``.

        Returns:
            ``∂u/∂x_dim`` at each point, shape ``(N,)``.
        """
        if method == "least_squares":
            return DifferentialOperators.compute_gradient_2d_lsq(u_values, points, triangles, dim)

        n_points = len(u_values)
        triangles = jnp.array(triangles)
        i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        p0, p1, p2 = points[i_idx], points[j_idx], points[k_idx]
        u0, u1, u2 = u_values[i_idx], u_values[j_idx], u_values[k_idx]
        dx1, dy1 = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
        dx2, dy2 = p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1]
        areas = 0.5 * jnp.abs(dx1 * dy2 - dx2 * dy1)

        if dim == 0:
            grads = ((u1 - u0) * dy2 - (u2 - u0) * dy1) / (2 * areas + 1e-12)
        else:
            grads = ((u2 - u0) * dx1 - (u1 - u0) * dx2) / (2 * areas + 1e-12)
        grads = jnp.where(areas > 1e-12, grads, 0.0)

        if method == "uniform":
            w_elem = jnp.where(areas > 1e-12, 1.0, 0.0)
        elif method == "inverse_distance":
            cx = (p0[:, 0] + p1[:, 0] + p2[:, 0]) / 3.0
            cy = (p0[:, 1] + p1[:, 1] + p2[:, 1]) / 3.0
            avg_dist = (jnp.sqrt((p0[:, 0] - cx) ** 2 + (p0[:, 1] - cy) ** 2) + jnp.sqrt((p1[:, 0] - cx) ** 2 + (p1[:, 1] - cy) ** 2) + jnp.sqrt((p2[:, 0] - cx) ** 2 + (p2[:, 1] - cy) ** 2)) / 3.0
            w_elem = jnp.where(areas > 1e-12, 1.0 / (avg_dist + 1e-12), 0.0)
        else:  # area_weighted
            w_elem = jnp.where(areas > 1e-12, areas, 0.0)

        contributions = grads * w_elem
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions).at[k_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(w_elem).at[j_idx].add(w_elem).at[k_idx].add(w_elem)
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_gradient_2d_lsq(
        u_values: jnp.ndarray,
        points: jnp.ndarray,
        triangles,
        dim: int,
    ) -> jnp.ndarray:
        """Least-squares gradient on a 2-D triangular mesh.

        For each node *i* the gradient is estimated by solving a 2×2
        area-weighted least-squares problem built from incident triangle
        centroids.  The 2×2 system is solved via Cramer's rule so the
        entire computation uses only JAX scatter-add operations with no
        per-node Python loops.

        Args:
            u_values:  Function values, shape ``(N,)``.
            points:    Coordinates, shape ``(N, 2)``.
            triangles: Triangle connectivity, shape ``(M, 3)``.
            dim:       0 → ``∂u/∂x``, 1 → ``∂u/∂y``.

        Returns:
            Gradient component at each node, shape ``(N,)``.
        """
        N = len(u_values)
        triangles = jnp.array(triangles)
        i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        p0, p1, p2 = points[i_idx], points[j_idx], points[k_idx]
        u0, u1, u2 = u_values[i_idx], u_values[j_idx], u_values[k_idx]

        cx = (p0[:, 0] + p1[:, 0] + p2[:, 0]) / 3.0
        cy = (p0[:, 1] + p1[:, 1] + p2[:, 1]) / 3.0
        u_c = (u0 + u1 + u2) / 3.0

        dx1 = p1[:, 0] - p0[:, 0]
        dy1 = p1[:, 1] - p0[:, 1]
        dx2 = p2[:, 0] - p0[:, 0]
        dy2 = p2[:, 1] - p0[:, 1]
        area = 0.5 * jnp.abs(dx1 * dy2 - dx2 * dy1)
        w = jnp.where(area > 1e-12, area, 0.0)

        def _s(rx, ry, du, vid):
            Axx = jnp.zeros(N).at[vid].add(w * rx * rx)
            Axy = jnp.zeros(N).at[vid].add(w * rx * ry)
            Ayy = jnp.zeros(N).at[vid].add(w * ry * ry)
            bx = jnp.zeros(N).at[vid].add(w * rx * du)
            by = jnp.zeros(N).at[vid].add(w * ry * du)
            return Axx, Axy, Ayy, bx, by

        def _add5(a, b):
            return tuple(x + y for x, y in zip(a, b))

        Axx, Axy, Ayy, bx, by = _add5(
            _add5(
                _s(cx - p0[:, 0], cy - p0[:, 1], u_c - u0, i_idx),
                _s(cx - p1[:, 0], cy - p1[:, 1], u_c - u1, j_idx),
            ),
            _s(cx - p2[:, 0], cy - p2[:, 1], u_c - u2, k_idx),
        )

        det = Axx * Ayy - Axy * Axy
        safe_det = jnp.where(jnp.abs(det) > 1e-20, det, 1.0)
        valid = jnp.abs(det) > 1e-20

        if dim == 0:
            return jnp.where(valid, (Ayy * bx - Axy * by) / safe_det, 0.0)
        else:
            return jnp.where(valid, (Axx * by - Axy * bx) / safe_det, 0.0)

    @staticmethod
    def compute_fd_laplacian_2d_simple(
        u_values,
        points,
        triangles,
        dims,
        method: str = "gradient_of_gradient",
    ):
        """Laplacian on a 2-D triangular mesh.

        Args:
            u_values:  Function values, shape ``(N,)``.
            points:    Coordinates, shape ``(N, 2)``.
            triangles: Triangle connectivity, shape ``(M, 3)``.
            dims:      Spatial dimensions to sum over, e.g. ``(0, 1)``.
            method:    ``"gradient_of_gradient"`` (default), ``"cotangent"``,
                       or ``"lsq_of_gradient"``.

        Returns:
            Laplacian, shape ``(N,)``.
        """
        if method == "cotangent":
            return DifferentialOperators.compute_laplacian_2d_cotangent(u_values, points, triangles)
        if method == "lsq_of_gradient":
            result = jnp.zeros_like(u_values)
            for d in dims:
                g = DifferentialOperators.compute_gradient_2d_lsq(u_values, points, triangles, d)
                result = result + DifferentialOperators.compute_gradient_2d_lsq(g, points, triangles, d)
            return result
        # default: gradient_of_gradient
        result = jnp.zeros_like(u_values)
        for dim in dims:
            grad = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, dim)
            result = result + DifferentialOperators.compute_fd_gradient_2d_simple(grad, points, triangles, dim)
        return result

    @staticmethod
    def compute_laplacian_2d_cotangent(
        u_values: jnp.ndarray,
        points: jnp.ndarray,
        triangles,
    ) -> jnp.ndarray:
        """Cotangent-weight (Laplace–Beltrami) Laplacian on a 2-D mesh.

        For each triangle ``(i, j, k)`` the cotangent of each interior angle
        is used to weight the edge contributions::

            lap[i] += (1/A_i) * [ cot_k*(u_j - u_i) + cot_j*(u_k - u_i) ]

        with ``cot_k`` = cotangent of the angle at vertex *k* (opposite edge
        ``(i,j)``), and ``A_i = (1/3) * Σ area``.

        This is second-order accurate and isotropic; it is the gold standard
        for PDE discretisation on unstructured 2-D triangular meshes.

        Args:
            u_values:  Function values, shape ``(N,)``.
            points:    Coordinates, shape ``(N, 2)``.
            triangles: Triangle connectivity, shape ``(M, 3)``.

        Returns:
            Laplacian at each point, shape ``(N,)``.
        """
        N = len(u_values)
        triangles = jnp.array(triangles)
        i_idx, j_idx, k_idx = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        p0, p1, p2 = points[i_idx], points[j_idx], points[k_idx]
        u0, u1, u2 = u_values[i_idx], u_values[j_idx], u_values[k_idx]

        cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
        two_area = jnp.abs(cross)
        safe_2A = jnp.where(two_area > 1e-12, two_area, 1.0)

        # Cotangents via dot / (2*area)
        e01x = p1[:, 0] - p0[:, 0]
        e01y = p1[:, 1] - p0[:, 1]
        e02x = p2[:, 0] - p0[:, 0]
        e02y = p2[:, 1] - p0[:, 1]
        cot0 = (e01x * e02x + e01y * e02y) / safe_2A  # angle at vertex 0

        e10x = p0[:, 0] - p1[:, 0]
        e10y = p0[:, 1] - p1[:, 1]
        e12x = p2[:, 0] - p1[:, 0]
        e12y = p2[:, 1] - p1[:, 1]
        cot1 = (e10x * e12x + e10y * e12y) / safe_2A  # angle at vertex 1

        e20x = p0[:, 0] - p2[:, 0]
        e20y = p0[:, 1] - p2[:, 1]
        e21x = p1[:, 0] - p2[:, 0]
        e21y = p1[:, 1] - p2[:, 1]
        cot2 = (e20x * e21x + e20y * e21y) / safe_2A  # angle at vertex 2

        # Clamp to avoid issues with flat / obtuse triangles
        cot0 = jnp.clip(cot0, -10.0, 10.0)
        cot1 = jnp.clip(cot1, -10.0, 10.0)
        cot2 = jnp.clip(cot2, -10.0, 10.0)

        # Accumulate: edge (i,j) opposite k → weight cot2, etc.
        lap = jnp.zeros(N).at[i_idx].add(cot2 * (u1 - u0)).at[j_idx].add(cot2 * (u0 - u1)).at[j_idx].add(cot0 * (u2 - u1)).at[k_idx].add(cot0 * (u1 - u2)).at[i_idx].add(cot1 * (u2 - u0)).at[k_idx].add(cot1 * (u0 - u2))

        # A_i = 2 * (barycentric dual area) because the standard cotangent
        # Laplacian formula is  (Δu)_i = (1/(2*A_i)) * Σ_j (cot α + cot β)(u_j-u_i)
        # with A_i = (1/3) * Σ area.  Equivalently: normalise by (2/3)*Σ area.
        area_sum = jnp.zeros(N).at[i_idx].add(two_area / 2.0).at[j_idx].add(two_area / 2.0).at[k_idx].add(two_area / 2.0)
        A_i = area_sum * 2.0 / 3.0
        safe_A = jnp.where(A_i > 1e-12, A_i, 1.0)
        return jnp.where(A_i > 1e-12, lap / safe_A, 0.0)

    @staticmethod
    def compute_fd_hessian_2d_simple(u_values, points, triangles, var_dims):
        """Hessian on a 2-D triangular mesh (area-weighted FD).

        Args:
            u_values:  Function values, shape ``(N,)``.
            points:    Coordinates, shape ``(N, 2)``.
            triangles: Triangle connectivity, shape ``(M, 3)``.
            var_dims:  List of ``(i, vi_dim, j, vj_dim)`` tuples.

        Returns:
            Hessian, shape ``(N, n_vars, n_vars)``.
        """
        N = points.shape[0]
        n_vars = int(jnp.sqrt(len(var_dims)))

        grad_x = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, 0)
        grad_y = DifferentialOperators.compute_fd_gradient_2d_simple(u_values, points, triangles, 1)

        d2u_dx2 = DifferentialOperators.compute_fd_gradient_2d_simple(grad_x, points, triangles, 0)
        d2u_dxdy = DifferentialOperators.compute_fd_gradient_2d_simple(grad_x, points, triangles, 1)
        d2u_dy2 = DifferentialOperators.compute_fd_gradient_2d_simple(grad_y, points, triangles, 1)

        hess_full = jnp.zeros((N, 2, 2)).at[:, 0, 0].set(d2u_dx2).at[:, 0, 1].set(d2u_dxdy).at[:, 1, 0].set(d2u_dxdy).at[:, 1, 1].set(d2u_dy2)
        result = jnp.zeros((N, n_vars, n_vars))
        for i, vi_dim, j, vj_dim in var_dims:
            result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])
        return result

    # ══════════════════════════════════════════════════════════════════
    # 3-D tetrahedral mesh
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def compute_fd_gradient_3d_simple(
        u_values: jnp.ndarray,
        points: jnp.ndarray,
        tetrahedra,
        dim: int,
        method: str = "area_weighted",
    ) -> jnp.ndarray:
        """Gradient on a 3-D tetrahedral mesh.

        Args:
            u_values:    Function values, shape ``(N,)``.
            points:      Mesh point coordinates, shape ``(N, 3)``.
            tetrahedra:  Tet connectivity, shape ``(M, 4)``.
            dim:         Spatial dimension (0, 1 or 2).
            method:      ``"area_weighted"`` (default), ``"uniform"``,
                         ``"inverse_distance"``, ``"least_squares"``.

        Returns:
            ``∂u/∂x_dim`` at each point, shape ``(N,)``.
        """
        if method == "least_squares":
            return DifferentialOperators.compute_gradient_3d_lsq(u_values, points, tetrahedra, dim)

        n_points = len(u_values)
        tetrahedra = jnp.array(tetrahedra)
        i_idx, j_idx, k_idx, l_idx = (
            tetrahedra[:, 0],
            tetrahedra[:, 1],
            tetrahedra[:, 2],
            tetrahedra[:, 3],
        )
        p0, p1, p2, p3 = points[i_idx], points[j_idx], points[k_idx], points[l_idx]
        u0, u1, u2, u3 = (
            u_values[i_idx],
            u_values[j_idx],
            u_values[k_idx],
            u_values[l_idx],
        )
        v1, v2, v3 = p1 - p0, p2 - p0, p3 - p0
        volumes = jnp.abs(v1[:, 0] * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]) - v1[:, 1] * (v2[:, 0] * v3[:, 2] - v2[:, 2] * v3[:, 0]) + v1[:, 2] * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])) / 6.0

        if dim == 0:
            grads = ((u1 - u0) * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]) + (u2 - u0) * (v3[:, 1] * v1[:, 2] - v3[:, 2] * v1[:, 1]) + (u3 - u0) * (v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1])) / (6 * volumes + 1e-12)
        elif dim == 1:
            grads = ((u1 - u0) * (v2[:, 2] * v3[:, 0] - v2[:, 0] * v3[:, 2]) + (u2 - u0) * (v3[:, 2] * v1[:, 0] - v3[:, 0] * v1[:, 2]) + (u3 - u0) * (v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2])) / (6 * volumes + 1e-12)
        else:
            grads = ((u1 - u0) * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0]) + (u2 - u0) * (v3[:, 0] * v1[:, 1] - v3[:, 1] * v1[:, 0]) + (u3 - u0) * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])) / (6 * volumes + 1e-12)

        grads = jnp.where(volumes > 1e-12, grads, 0.0)

        if method == "uniform":
            w_elem = jnp.where(volumes > 1e-12, 1.0, 0.0)
        elif method == "inverse_distance":
            cx = (p0[:, 0] + p1[:, 0] + p2[:, 0] + p3[:, 0]) / 4.0
            cy = (p0[:, 1] + p1[:, 1] + p2[:, 1] + p3[:, 1]) / 4.0
            cz = (p0[:, 2] + p1[:, 2] + p2[:, 2] + p3[:, 2]) / 4.0
            avg_d = (
                jnp.sqrt((p0[:, 0] - cx) ** 2 + (p0[:, 1] - cy) ** 2 + (p0[:, 2] - cz) ** 2)
                + jnp.sqrt((p1[:, 0] - cx) ** 2 + (p1[:, 1] - cy) ** 2 + (p1[:, 2] - cz) ** 2)
                + jnp.sqrt((p2[:, 0] - cx) ** 2 + (p2[:, 1] - cy) ** 2 + (p2[:, 2] - cz) ** 2)
                + jnp.sqrt((p3[:, 0] - cx) ** 2 + (p3[:, 1] - cy) ** 2 + (p3[:, 2] - cz) ** 2)
            ) / 4.0
            w_elem = jnp.where(volumes > 1e-12, 1.0 / (avg_d + 1e-12), 0.0)
        else:  # area_weighted (volume)
            w_elem = jnp.where(volumes > 1e-12, volumes, 0.0)

        contributions = grads * w_elem
        gradients = jnp.zeros(n_points).at[i_idx].add(contributions).at[j_idx].add(contributions).at[k_idx].add(contributions).at[l_idx].add(contributions)
        weights = jnp.zeros(n_points).at[i_idx].add(w_elem).at[j_idx].add(w_elem).at[k_idx].add(w_elem).at[l_idx].add(w_elem)
        return jnp.where(weights > 1e-12, gradients / weights, 0.0)

    @staticmethod
    def compute_gradient_3d_lsq(
        u_values: jnp.ndarray,
        points: jnp.ndarray,
        tetrahedra,
        dim: int,
    ) -> jnp.ndarray:
        """Least-squares gradient on a 3-D tetrahedral mesh.

        Analogous to :meth:`compute_gradient_2d_lsq` but solves a 3×3
        normal-equation system at each node via Cramer's rule.

        Args:
            u_values:   Function values, shape ``(N,)``.
            points:     Coordinates, shape ``(N, 3)``.
            tetrahedra: Tet connectivity, shape ``(M, 4)``.
            dim:        0 → ``∂u/∂x``, 1 → ``∂u/∂y``, 2 → ``∂u/∂z``.

        Returns:
            Gradient component at each node, shape ``(N,)``.
        """
        N = len(u_values)
        tetrahedra = jnp.array(tetrahedra)
        i_idx, j_idx, k_idx, l_idx = (
            tetrahedra[:, 0],
            tetrahedra[:, 1],
            tetrahedra[:, 2],
            tetrahedra[:, 3],
        )
        p0, p1, p2, p3 = points[i_idx], points[j_idx], points[k_idx], points[l_idx]
        u0, u1, u2, u3 = u_values[i_idx], u_values[j_idx], u_values[k_idx], u_values[l_idx]

        cx = (p0[:, 0] + p1[:, 0] + p2[:, 0] + p3[:, 0]) / 4.0
        cy = (p0[:, 1] + p1[:, 1] + p2[:, 1] + p3[:, 1]) / 4.0
        cz = (p0[:, 2] + p1[:, 2] + p2[:, 2] + p3[:, 2]) / 4.0
        u_c = (u0 + u1 + u2 + u3) / 4.0

        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        vol = jnp.abs(v1[:, 0] * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1]) - v1[:, 1] * (v2[:, 0] * v3[:, 2] - v2[:, 2] * v3[:, 0]) + v1[:, 2] * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])) / 6.0
        w = jnp.where(vol > 1e-12, vol, 0.0)

        def _a(rx, ry, rz, du, vid):
            Axx = jnp.zeros(N).at[vid].add(w * rx * rx)
            Axy = jnp.zeros(N).at[vid].add(w * rx * ry)
            Axz = jnp.zeros(N).at[vid].add(w * rx * rz)
            Ayy = jnp.zeros(N).at[vid].add(w * ry * ry)
            Ayz = jnp.zeros(N).at[vid].add(w * ry * rz)
            Azz = jnp.zeros(N).at[vid].add(w * rz * rz)
            bx = jnp.zeros(N).at[vid].add(w * rx * du)
            by = jnp.zeros(N).at[vid].add(w * ry * du)
            bz = jnp.zeros(N).at[vid].add(w * rz * du)
            return Axx, Axy, Axz, Ayy, Ayz, Azz, bx, by, bz

        def _add9(a, b):
            return tuple(x + y for x, y in zip(a, b))

        acc = _add9(
            _add9(
                _add9(
                    _a(cx - p0[:, 0], cy - p0[:, 1], cz - p0[:, 2], u_c - u0, i_idx),
                    _a(cx - p1[:, 0], cy - p1[:, 1], cz - p1[:, 2], u_c - u1, j_idx),
                ),
                _a(cx - p2[:, 0], cy - p2[:, 1], cz - p2[:, 2], u_c - u2, k_idx),
            ),
            _a(cx - p3[:, 0], cy - p3[:, 1], cz - p3[:, 2], u_c - u3, l_idx),
        )
        Axx, Axy, Axz, Ayy, Ayz, Azz, bx, by, bz = acc

        det = Axx * (Ayy * Azz - Ayz * Ayz) - Axy * (Axy * Azz - Ayz * Axz) + Axz * (Axy * Ayz - Ayy * Axz)
        safe_det = jnp.where(jnp.abs(det) > 1e-30, det, 1.0)
        valid = jnp.abs(det) > 1e-30

        if dim == 0:
            num = bx * (Ayy * Azz - Ayz * Ayz) - Axy * (by * Azz - Ayz * bz) + Axz * (by * Ayz - Ayy * bz)
        elif dim == 1:
            num = Axx * (by * Azz - Ayz * bz) - bx * (Axy * Azz - Ayz * Axz) + Axz * (Axy * bz - by * Axz)
        else:
            num = Axx * (Ayy * bz - by * Ayz) - Axy * (Axy * bz - by * Axz) + bx * (Axy * Ayz - Ayy * Axz)

        return jnp.where(valid, num / safe_det, 0.0)

    @staticmethod
    def compute_fd_laplacian_3d_simple(
        u_values,
        points,
        tetrahedra,
        dims,
        method: str = "gradient_of_gradient",
    ):
        """Laplacian on a 3-D tetrahedral mesh.

        Args:
            u_values:   Function values, shape ``(N,)``.
            points:     Coordinates, shape ``(N, 3)``.
            tetrahedra: Tet connectivity, shape ``(M, 4)``.
            dims:       Spatial dimensions to sum over, e.g. ``(0, 1, 2)``.
            method:     ``"gradient_of_gradient"`` (default) or
                        ``"lsq_of_gradient"``.

        Returns:
            Laplacian, shape ``(N,)``.
        """
        if method == "lsq_of_gradient":
            result = jnp.zeros_like(u_values)
            for d in dims:
                g = DifferentialOperators.compute_gradient_3d_lsq(u_values, points, tetrahedra, d)
                result = result + DifferentialOperators.compute_gradient_3d_lsq(g, points, tetrahedra, d)
            return result
        result = jnp.zeros_like(u_values)
        for dim in dims:
            grad = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, dim)
            result = result + DifferentialOperators.compute_fd_gradient_3d_simple(grad, points, tetrahedra, dim)
        return result

    @staticmethod
    def compute_fd_hessian_3d_simple(u_values, points, tetrahedra, var_dims):
        """Hessian on a 3-D tetrahedral mesh (volume-weighted FD).

        Args:
            u_values:   Function values, shape ``(N,)``.
            points:     Coordinates, shape ``(N, 3)``.
            tetrahedra: Tet connectivity, shape ``(M, 4)``.
            var_dims:   List of ``(i, vi_dim, j, vj_dim)`` tuples.

        Returns:
            Hessian, shape ``(N, n_vars, n_vars)``.
        """
        N = points.shape[0]
        n_vars = int(jnp.sqrt(len(var_dims)))

        grad_x = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 0)
        grad_y = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 1)
        grad_z = DifferentialOperators.compute_fd_gradient_3d_simple(u_values, points, tetrahedra, 2)

        d2u_dx2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 0)
        d2u_dxdy = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 1)
        d2u_dxdz = DifferentialOperators.compute_fd_gradient_3d_simple(grad_x, points, tetrahedra, 2)
        d2u_dy2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_y, points, tetrahedra, 1)
        d2u_dydz = DifferentialOperators.compute_fd_gradient_3d_simple(grad_y, points, tetrahedra, 2)
        d2u_dz2 = DifferentialOperators.compute_fd_gradient_3d_simple(grad_z, points, tetrahedra, 2)

        hess_full = (
            jnp.zeros((N, 3, 3))
            .at[:, 0, 0]
            .set(d2u_dx2)
            .at[:, 0, 1]
            .set(d2u_dxdy)
            .at[:, 0, 2]
            .set(d2u_dxdz)
            .at[:, 1, 0]
            .set(d2u_dxdy)
            .at[:, 1, 1]
            .set(d2u_dy2)
            .at[:, 1, 2]
            .set(d2u_dydz)
            .at[:, 2, 0]
            .set(d2u_dxdz)
            .at[:, 2, 1]
            .set(d2u_dydz)
            .at[:, 2, 2]
            .set(d2u_dz2)
        )
        result = jnp.zeros((N, n_vars, n_vars))
        for i, vi_dim, j, vj_dim in var_dims:
            result = result.at[:, i, j].set(hess_full[:, vi_dim, vj_dim])
        return result

    # ══════════════════════════════════════════════════════════════════
    # Scheme-string helper (consumed by TraceEvaluator)
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def parse_fd_scheme(scheme: str) -> tuple[str, str, str]:
        """Parse a scheme string into ``(main_scheme, grad_method, lap_method)``.

        Supported formats::

            "finite_difference"                  → fd, "area_weighted", "gradient_of_gradient"
            "finite_difference:lsq"              → fd, "least_squares", "lsq_of_gradient"
            "finite_difference:cotangent"        → fd, "area_weighted", "cotangent"
            "finite_difference:uniform"          → fd, "uniform",       "gradient_of_gradient"
            "finite_difference:inverse_distance" → fd, "inverse_distance", "gradient_of_gradient"
            "automatic_differentiation"          → ad, None, None

        Returns:
            Tuple ``(main_scheme, grad_method, lap_method)``.
        """
        if ":" not in scheme:
            if scheme == "automatic_differentiation":
                return "automatic_differentiation", None, None
            return "finite_difference", "area_weighted", "gradient_of_gradient"

        main, sub = scheme.split(":", 1)
        sub = sub.strip()

        if sub == "cotangent":
            return main, "area_weighted", "cotangent"
        if sub in ("lsq", "least_squares"):
            return main, "least_squares", "lsq_of_gradient"
        # uniform / inverse_distance / area_weighted
        return main, sub, "gradient_of_gradient"
