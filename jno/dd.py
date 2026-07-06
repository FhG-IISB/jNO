"""Heterogeneous domain-decomposition coupling — overlapping Schwarz over subdomain solves.

On one shared mesh, each subdomain problem (a ``jno.fdm([...])`` / ``jno.fem([...])``) solves its own
region; the driver pins each subdomain's **complement** to the neighbours' current field (Dirichlet
exchange) and iterates the fixed point to a tolerance on the overlap. Flux continuity emerges at
convergence, so value-only (Dirichlet) exchange is complete for overlapping subdomains.

This is the internal coupled-solve engine; the user-facing surface is ``jno.core([...])``, which detects
subdomain solve terms and calls this. See ``plans/heterogeneous-domain-decomposition.md``.
"""

from __future__ import annotations

import numpy as np


def _region_mask(pts, geom):
    """Boolean mask of the mesh nodes ``pts`` inside a shapely region ``geom``."""
    from shapely.geometry import Point

    g = geom.buffer(1e-9)
    try:
        import shapely  # vectorized (shapely >= 2.0.2)

        return np.asarray(shapely.contains_xy(g, np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]))
    except (ImportError, AttributeError):
        return np.array([g.contains(Point(float(q[0]), float(q[1]))) for q in np.asarray(pts)])


class _Coupled:
    """A coupled domain-decomposition problem: subdomains + their regions, solved by overlapping Schwarz."""

    def __init__(self, subdomains):
        if len(subdomains) != 2:
            raise NotImplementedError(
                "jno.dd: only 2 subdomains are supported for now (the complement of one is the other); "
                "N-subdomain coupling pins each complement to the combined field of all the others."
            )
        self._subdomains = list(subdomains)

    def solve(self, *, tol: float = 1e-6, max_iter: int = 100, return_info: bool = False):
        """Run the overlapping-Schwarz fixed point to ``tol`` on the overlap; return the combined field."""
        probs = [p for p, _ in self._subdomains]
        geoms = [g for _, g in self._subdomains]
        dom = probs[0].domain
        dim = int(getattr(dom, "dimension", 2))
        pts = np.asarray(dom.mesh_connectivity["points"])[:, :dim]
        n = pts.shape[0]

        masks = [_region_mask(pts, g) for g in geoms]  # nodes owned by each subdomain
        complements = [np.where(~m)[0].astype(int) for m in masks]  # nodes to pin from the neighbour
        overlap = masks[0] & masks[1]
        sols = [np.zeros(n), np.zeros(n)]

        # Build each subdomain's pinned solver ONCE (JIT reused across iterations — see pinned_solver).
        solvers = [probs[i].pinned_solver(complements[i]) for i in range(2)]

        jump = np.inf
        iters = 0
        for iters in range(1, max_iter + 1):
            for i in range(2):
                neighbour = sols[1 - i]  # 2 subdomains: the complement of i is (a subset of) the neighbour
                sols[i] = np.asarray(solvers[i](neighbour[complements[i]])).reshape(-1)
            jump = float(np.max(np.abs(sols[0][overlap] - sols[1][overlap]))) if overlap.any() else 0.0
            if jump < tol:
                break

        combined = np.where(masks[0], sols[0], sols[1])  # the two agree on the overlap at convergence
        if return_info:
            return combined, {"iterations": iters, "overlap_jump": jump}
        return combined


def couple(subdomains):
    """Couple subdomain problems by overlapping Schwarz on a shared mesh.

    ``subdomains``: a list of ``(problem, region)`` pairs, where ``problem`` is a subdomain solve
    (``jno.fdm([...])`` / ``jno.fem([...])``) authored with its PDE + outer boundary conditions, and
    ``region`` is the shapely geometry it owns. ``.solve()`` returns the combined nodal field."""
    return _Coupled(subdomains)
