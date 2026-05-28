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
                area = 0.5 * abs(float(np.cross(b - a, c - a)))
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
