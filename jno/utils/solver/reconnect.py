"""Alpha-shape reconnection of a moved node set -- the remeshing step of the Particle Finite Element Method.

A body meshed with its own nodes, moved by its own flow, changes shape but not topology: two drops that
touch stay two meshes. PFEM (Idelsohn, Oñate & Del Pin, IJNME 61 (2004) 964-989) recovers the domain
from the NODES instead: re-triangulate them (Delaunay) and keep only the simplices whose circumradius is
below ``alpha * h`` -- the alpha shape (Edelsbrunner & Mücke, ACM TOG 13 (1994) 43-72). A triangle
bridging a gap narrower than about ``2 alpha h`` passes the filter, so two bodies that come that close
become one; a triangle spanning open space does not, so a concave surface stays concave.

The nodes are KEPT, in their order, so a P1 state carries across by identity. A node left in no
triangle (a PFEM "free particle") is refused: the mesh hygiene pass would drop it and renumber the rest,
which would silently permute a state carried by identity.

Private: no public spelling exists for this yet.
"""

from __future__ import annotations

import numpy as np


def _circumradius_2d(P: np.ndarray) -> np.ndarray:
    """Circumradius of each triangle ``P[k] = (3, 2)``: ``abc / (4 area)``."""
    a = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)
    b = np.linalg.norm(P[:, 0] - P[:, 2], axis=1)
    c = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)
    p, q = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area = 0.5 * np.abs(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])
    return a * b * c / (4.0 * np.maximum(area, 1e-300))


def alpha_reconnect(
    points: np.ndarray,
    h: float,
    alpha: float = 1.2,
    *,
    previous: np.ndarray | None = None,
    hysteresis: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-triangulate ``points`` and keep the alpha shape: ``(cells (n_cells, 3), boundary edges (n_b, 2))``.

    ``h`` is the mesh's length scale (its mean edge length) and ``alpha`` the filter: a triangle is kept
    when its circumradius is below ``alpha * h``. A regular triangle of side ``h`` has circumradius
    ``h / sqrt(3) = 0.58 h`` and a right isosceles one ``0.71 h``, so ``alpha`` near 1.2 keeps a sound
    mesh and removes what spans open space. Cells come out counter-clockwise; indices refer to ``points``
    unchanged.

    ``previous`` (the triangulation now in use) switches on HYSTERESIS, and a march wants it. A single
    threshold makes the filter FLICKER: a triangle whose circumradius sits near ``alpha * h`` drops out on
    one reconnection and returns on the next, so the free surface loses and regains wedges from step to
    step. Measured on a coalescing drop reconnecting every step, with one threshold: the perimeter jumped
    +22.0 % at one step and recovered -18.0 % two steps later, then +25.0 % and -19.4 %, while the
    interior stayed valid -- the shape was wrong in whichever frame one happened to look at. A triangle
    that already exists is therefore kept until it exceeds ``hysteresis * alpha * h``, which is the usual
    remedy for a thresholded set that is re-decided every step. New triangles still face ``alpha`` alone,
    so the gap at which two bodies merge is unchanged.
    """
    from scipy.spatial import Delaunay

    from .fem_adapt import _boundary_edges_from_triangles

    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise NotImplementedError(
            f"alpha reconnection is 2-D only; got points of shape {X.shape}. In 3-D the Delaunay + alpha "
            "filter leaves sliver tetrahedra, a known PFEM problem that needs its own treatment."
        )
    if not (h > 0.0 and alpha > 0.0):
        raise ValueError(f"alpha reconnection needs h > 0 and alpha > 0; got h={h}, alpha={alpha}.")
    if hysteresis < 1.0:
        raise ValueError(
            f"alpha reconnection: hysteresis widens the threshold for existing cells, so it must be >= 1; got {hysteresis}."
        )
    cells = Delaunay(X).simplices
    radius = _circumradius_2d(X[cells])
    keep = radius < alpha * h
    if previous is not None and np.asarray(previous).size:
        # A cell already in use survives up to the wider threshold -- see the docstring on flicker.
        held = {tuple(c) for c in np.sort(np.asarray(previous, dtype=np.int64), axis=1)}
        existing = np.fromiter((tuple(c) in held for c in np.sort(cells, axis=1)), dtype=bool, count=cells.shape[0])
        keep |= existing & (radius < hysteresis * alpha * h)
    cells = cells[keep]
    t = X[cells]
    p, q = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    flip = (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]) < 0.0
    cells[flip] = cells[flip][:, [0, 2, 1]]
    used = np.zeros(X.shape[0], dtype=bool)
    used[cells.reshape(-1)] = True
    if not used.all():
        lost = np.flatnonzero(~used)
        raise ValueError(
            f"alpha reconnection: {lost.size} node(s) are in no triangle after the filter (first: {lost[:5].tolist()}) "
            f"-- free particles, further than about {alpha} h from the rest. They would be dropped and every "
            "other node renumbered, silently permuting the state. Raise alpha, or refine where the body thins."
        )
    return cells.astype(np.int64), np.asarray(_boundary_edges_from_triangles(cells), dtype=np.int64)


def n_components(n_points: int, cells: np.ndarray) -> int:
    """Number of connected bodies a triangulation of ``n_points`` nodes forms."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    c = np.asarray(cells)
    r = np.concatenate([c[:, 0], c[:, 1], c[:, 2]])
    s = np.concatenate([c[:, 1], c[:, 2], c[:, 0]])
    A = coo_matrix((np.ones(r.size), (r, s)), shape=(n_points, n_points))
    used = np.zeros(n_points, dtype=bool)
    used[c.reshape(-1)] = True
    _k, lab = connected_components(A, directed=False)
    return int(np.unique(lab[used]).size)
