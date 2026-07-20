"""Mesh-based numerical integration operators.

Mirrors :class:`DifferentialOperators` but for integration rather than
differentiation.  All methods are static and work on plain NumPy arrays
derived from ``mesh_connectivity`` — no domain object dependency.
"""

from __future__ import annotations

import numpy as np


class IntegrationOperators:
    """Static namespace for mesh-based numerical integration.

    Works on the ``mesh_connectivity`` dict produced by the domain class.
    Boundary weights (``nodal_ds``) are already stored there; this class
    adds volume weights (``nodal_volumes``) computed on the fly.
    """

    @staticmethod
    def nodal_volumes(mesh_connectivity: dict) -> np.ndarray:
        """Per-node volume weights for interior integration.

        Returns ``mesh_connectivity["nodal_volumes"]`` if it was precomputed
        during domain setup (the normal path).  Otherwise computes on the fly
        (fallback for manually constructed mesh_connectivity dicts).

        Each node receives a share of surrounding element volumes:

        - 1-D: ½ × sum of adjacent segment lengths (trapezoidal rule)
        - 2-D: ⅓ × sum of incident triangle areas
        - 3-D: ¼ × sum of incident tetrahedron volumes

        Parameters
        ----------
        mesh_connectivity : dict
            Preprocessed mesh connectivity from the domain class.

        Returns
        -------
        vols : ndarray of shape (n_points,)
        """
        if "nodal_volumes" in mesh_connectivity:
            return np.asarray(mesh_connectivity["nodal_volumes"])

        dim = mesh_connectivity["dimension"]
        points = mesh_connectivity["points"]
        N = mesh_connectivity["n_points"]
        vols = np.zeros(N)

        if dim == 1:
            for seg in mesh_connectivity["lines"]:
                L = np.linalg.norm(points[seg[1]] - points[seg[0]])
                vols[seg[0]] += 0.5 * L
                vols[seg[1]] += 0.5 * L

        elif dim == 2:
            for tri in mesh_connectivity["triangles"]:
                a = points[tri[0]]
                b = points[tri[1]]
                c = points[tri[2]]
                ba, ca = b - a, c - a
                area = 0.5 * abs(float(ba[0] * ca[1] - ba[1] * ca[0]))
                for n in tri:
                    vols[n] += area / 3.0

        elif dim == 3:
            for tet in mesh_connectivity["tetrahedra"]:
                a, b, c, d = (points[tet[i]] for i in range(4))
                vol = abs(float(np.dot(b - a, np.cross(c - a, d - a)))) / 6.0
                for n in tet:
                    vols[n] += vol / 4.0

        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        return vols

    @staticmethod
    def gauss_points_and_weights(mesh_connectivity: dict, degree: int = 4, cells: np.ndarray | None = None):
        """Element **Gauss** quadrature over the volume mesh: physical points + ``JxW`` weights.

        Unlike :meth:`nodal_volumes` (a vertex rule that samples only at mesh nodes), this maps a
        reference-cell Gauss rule of the requested ``degree`` into every element and returns the
        physical quadrature points together with their ``w · |det J|`` weights, so
        ``sum(f(points) * weights)`` is the higher-order Gauss approximation of ``∫ f dx``. Exact
        for polynomials up to ``degree``; many points per element make it far harder for an
        expressive integrand (e.g. a network) to alias the rule than the vertex rule can.

        Parameters
        ----------
        mesh_connectivity : dict
            Preprocessed mesh connectivity from the domain class (``points`` + ``triangles`` /
            ``tetrahedra``).
        degree : int
            Polynomial degree the rule integrates exactly (basix quadrature degree).
        cells : ndarray, optional
            Element connectivity subset (e.g. a sub-region's triangles); defaults to all cells.

        Returns
        -------
        (points, JxW) : tuple[ndarray, ndarray]
            ``points`` of shape ``(n_cell * n_qp, dim)`` and ``JxW`` of shape ``(n_cell * n_qp,)``.
        """
        import basix
        from basix import CellType

        dim = mesh_connectivity["dimension"]
        points = np.asarray(mesh_connectivity["points"], dtype=np.float64)[:, :dim]

        if dim == 2:
            cells = mesh_connectivity["triangles"] if cells is None else cells
            qp, qw = basix.make_quadrature(CellType.triangle, degree)  # ref pts (nq,2), wts sum to 1/2
        elif dim == 3:
            cells = mesh_connectivity["tetrahedra"] if cells is None else cells
            qp, qw = basix.make_quadrature(CellType.tetrahedron, degree)  # ref pts (nq,3), wts sum to 1/6
        else:
            raise ValueError(f"Gauss integration unsupported for dimension {dim}")

        cells = np.asarray(cells)
        qp = np.asarray(qp, dtype=np.float64)
        qw = np.asarray(qw, dtype=np.float64)

        a = points[cells[:, 0]]  # (ncell, dim) — reference-vertex origin of the affine map
        edges = np.stack([points[cells[:, k + 1]] - a for k in range(dim)], axis=1)  # (ncell, dim, dim)

        # |det J| of the affine reference→physical map (= 2·area in 2-D, 6·volume in 3-D)
        detJ = np.abs(np.linalg.det(edges))  # (ncell,)

        # physical quadrature points: a + Σ_k ξ_k · edge_k
        phys = a[:, None, :] + np.einsum("qk,ckd->cqd", qp, edges)  # (ncell, nq, dim)
        JxW = qw[None, :] * detJ[:, None]  # (ncell, nq)

        return phys.reshape(-1, dim), JxW.reshape(-1)
