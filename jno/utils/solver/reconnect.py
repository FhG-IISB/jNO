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


def _manage_nodes(
    X: np.ndarray, cells: np.ndarray, h: float, *, long_f: float, short_f: float, max_growth: float
) -> tuple[np.ndarray, int, int]:
    """PFEM node management: ``(points, n_inserted, n_removed)``.

    Re-triangulating cannot fix a bad point DISTRIBUTION -- Delaunay already maximises the minimum angle
    for the points it is given. A body that stretches thins its nodes, and one that merges is left with
    whatever spacing the two surfaces happened to have, so the mesh degrades however often it is
    reconnected. Measured on a coalescing drop with reconnection alone: the smallest angle fell from 41.5
    to 8.3 degrees and the largest cell grew to 57x the smallest, with the slivers lined up along the
    plane where the two bodies joined. So nodes are added and dropped, as PFEM does:

    * an edge longer than ``long_f * h`` gains its MIDPOINT. On a boundary edge that midpoint lies on the
      straight edge itself, so the polygon -- and the liquid area -- is unchanged;
    * an INTERIOR node closer than ``short_f * h`` to another node is dropped. Boundary nodes are never
      dropped: removing one cuts a corner off the body and loses liquid.

    ``max_growth`` caps insertion per call (as a fraction of the node count) so a stretching body cannot
    grow its mesh without bound; the longest edges are served first.
    """
    from scipy.spatial import cKDTree

    from .fem_adapt import _boundary_edges_from_triangles

    on_bnd = np.zeros(X.shape[0], dtype=bool)
    on_bnd[np.asarray(_boundary_edges_from_triangles(cells)).reshape(-1)] = True

    edges = np.unique(np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1), axis=0)
    length = np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)
    hv = np.broadcast_to(np.asarray(h, dtype=float).reshape(-1), (X.shape[0],)) if np.asarray(h).ndim else None
    he = 0.5 * (hv[edges[:, 0]] + hv[edges[:, 1]]) if hv is not None else float(h)
    too_long = np.flatnonzero(length > long_f * he)
    if too_long.size:
        budget = max(1, int(max_growth * X.shape[0]))
        if too_long.size > budget:  # serve the worst offenders first
            too_long = too_long[np.argsort(-length[too_long])[:budget]]
    fresh = 0.5 * (X[edges[too_long, 0]] + X[edges[too_long, 1]])

    drop: set[int] = set()
    if short_f > 0.0:
        _hmax = float(np.max(hv)) if hv is not None else float(h)
        for i, j in cKDTree(X).query_pairs(short_f * _hmax, output_type="ndarray"):
            if i in drop or j in drop:
                continue
            if hv is not None and np.linalg.norm(X[i] - X[j]) > short_f * 0.5 * (hv[i] + hv[j]):
                continue  # the KD-tree query used the GLOBAL max; re-test against the local scale
            if not on_bnd[i]:
                drop.add(int(i))
            elif not on_bnd[j]:
                drop.add(int(j))  # both on the boundary: keep them, or the surface would move
    keep = np.ones(X.shape[0], dtype=bool)
    if drop:
        keep[list(drop)] = False
    out = np.concatenate([X[keep], fresh]) if fresh.size else X[keep]
    # The length scale rides the nodes: a kept node keeps its own, and an inserted MIDPOINT takes the
    # mean of the two it splits. Without this the field is stale the moment management fires, and the
    # filter indexes it with the new cell array -- an IndexError at best, the wrong threshold at worst.
    if hv is None:
        h_out = h
    else:
        h_new = 0.5 * (hv[edges[too_long, 0]] + hv[edges[too_long, 1]]) if fresh.size else hv[:0]
        h_out = np.concatenate([hv[keep], h_new]) if fresh.size else hv[keep]
    return out, int(fresh.shape[0]), len(drop), h_out


def _h_at(h, pts: np.ndarray, cells: np.ndarray):
    """Per-cell length scale: the mean of its vertices' sizes, or the scalar if ``h`` is one.

    A GRADED mesh has no single length scale, and forcing one breaks the filter at both ends: with the
    mean, coarse-region cells exceed ``alpha*h`` and their nodes are reported as free particles; with
    the coarse value, the fine region fuses surfaces that are genuinely apart. Both thresholds the
    filter applies are per-cell or per-edge quantities already, so they take a per-node ``h`` directly.
    """
    h = np.asarray(h, dtype=float)
    if h.ndim == 0:
        return float(h)
    return h[cells].mean(axis=1)


def alpha_reconnect(
    points: np.ndarray,
    h: float,
    alpha: float = 1.2,
    *,
    previous: np.ndarray | None = None,
    hysteresis: float = 1.5,
    manage: bool = True,
    long_f: float = 1.5,
    short_f: float = 0.55,
    max_growth: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-triangulate ``points`` and keep the alpha shape: ``(points, cells (n, 3), boundary edges (n_b, 2))``.

    ``points`` comes back because ``manage=True`` (the default) also INSERTS and DROPS nodes -- see
    :func:`_manage_nodes`, which is what keeps the elements usable rather than merely optimally connected.
    When the node set is unchanged the array is returned as it came in, and a caller carrying a P1 state
    by identity can check that; when it changes, the state has to be interpolated from the old mesh.

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
    h = np.asarray(h, dtype=float)
    if h.ndim not in (0, 1) or (h.ndim == 1 and h.shape[0] != X.shape[0]):
        raise ValueError(
            f"alpha reconnection: h must be a scalar or one value PER POINT ({X.shape[0]}); got shape {h.shape}."
        )
    if not (np.all(h > 0.0) and alpha > 0.0):
        raise ValueError(f"alpha reconnection needs h > 0 and alpha > 0; got h={h}, alpha={alpha}.")
    if hysteresis < 1.0:
        raise ValueError(
            f"alpha reconnection: hysteresis widens the threshold for existing cells, so it must be >= 1; got {hysteresis}."
        )

    def _filter(pts: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        cells = Delaunay(pts).simplices
        radius = _circumradius_2d(pts[cells])
        hc = _h_at(h, pts, cells)  # per-cell length scale: a GRADED mesh has no single one
        keep = radius < alpha * hc
        if prev is not None and np.asarray(prev).size:
            # A cell already in use survives up to the wider threshold -- see the docstring on flicker.
            held = {tuple(c) for c in np.sort(np.asarray(prev, dtype=np.int64), axis=1)}
            existing = np.fromiter((tuple(c) in held for c in np.sort(cells, axis=1)), dtype=bool, count=cells.shape[0])
            keep |= existing & (radius < hysteresis * alpha * hc)
        return cells[keep]

    cells = _filter(X, previous)
    if manage:
        moved, n_new, n_gone, h = _manage_nodes(X, cells, h, long_f=long_f, short_f=short_f, max_growth=max_growth)
        if n_new or n_gone:
            # The node numbering has changed, so `previous` no longer names the same cells: this pass runs
            # on the plain threshold. Node management is occasional, so the hysteresis that steadies the
            # ordinary steps is not lost in practice.
            X, cells = moved, _filter(moved, None)
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
    return X, cells.astype(np.int64), np.asarray(_boundary_edges_from_triangles(cells), dtype=np.int64)


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
