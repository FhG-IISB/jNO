"""Pairing two surfaces by proximity, one query point at a time.

The interface machinery used to answer "where does this secondary point land on the main surface?" by
fitting **one** tangent plane to the whole main face (:func:`~.fem_utils._interface_frame`), flattening
everything onto it, and locating the query in that coordinate. That is exact for a flat, already-paired
interface and degrades with curvature, because the flattened coordinate is not arc length: two points
that are far apart along a curved surface can share a coordinate, and the facet under a query is then
not the facet nearest to it.

Measured against a brute-force oracle -- ``n . (closest point on the main polyline - x)``, the very
quantity being computed -- on two concentric arcs whose true separation is 0.3 everywhere:

    main arc      flattened      this module
      5 deg       1.240e-04       0.000e+00
     20 deg       2.069e-03       0.000e+00
     60 deg       2.904e-02       0.000e+00
    100 deg       2.949e-01       0.000e+00
    140 deg       1.117e+00       0.000e+00

At 140 degrees the flattened projection reports a gap of 0.43 against a true 0.30. It over-reports
separation, so a contact pressure ``max(0, -g)`` never activates and free interpenetration is an exact
root of the residual -- the failure is silent. In 3-D, on a triangulated spherical cap, the same
comparison gives 6.3e-02 against 5.6e-17.

**A flat interface must not move**, and does not: the trace ``u_m . Phi`` agrees with the flattened
path to 1.4e-15 over random main fields, and ``g0`` is bit-identical. Note that the *weights* may
differ there while the *trace* does not -- a query landing exactly on a facet edge is contained by two
facets, and the two paths may pick either, but the weight on the non-shared node is exactly zero. Test
the trace, never the weights.

Host/NumPy by design. Locating a point on a facet is a discrete search, the same eager-setup exception
the rest of the interface machinery takes (structural work may live on the host; the *values* computed
from it must stay differentiable). The result is a plain gather, so a field read through it is
differentiable in the DOF values -- though not yet in the mesh coordinates.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .fem_utils import _edge_shape, _tri_shape

#: Reported for a query with no facet inside the capture distance: wide open, so a contact pressure
#: ``max(0, -g)`` yields exactly zero. Chosen over ``inf`` so a stray multiplication cannot make a NaN.
OPEN_GAP = 1.0e30

#: Chunk size for the unbounded all-pairs narrow phase, so a large interface cannot blow up memory.
_CHUNK = 4096


def facet_geometry(facets: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-facet vertex coordinates, centroid and radius.

    The leading columns are the **vertices**, and that is not the same as the nodes: a P2 edge carries
    3 nodes but 2 vertices, a P2 triangle 6 and 3. Letting a midside node into the centroid would bias
    it off the facet's own centre and, through the broad phase's radius, silently shrink the region a
    query is allowed to see.
    """
    pts = np.asarray(points, dtype=float)
    dim = int(pts.shape[1])
    nv = 2 if dim == 2 else 3
    V = pts[np.asarray(facets, dtype=int)[:, :nv]]  # (n_f, nv, dim)
    cent = V.mean(axis=1)
    rad = np.linalg.norm(V - cent[:, None, :], axis=2).max(axis=1)
    return V, cent, rad


def closest_point_on_segment(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point on segment ``a-b`` to each ``x``, and its parameter in ``[0, 1]``.

    Batched over the leading axis; ``x``, ``a``, ``b`` are all ``(n, dim)``.
    """
    ab = b - a
    dd = np.einsum("nd,nd->n", ab, ab)
    t = np.clip(np.einsum("nd,nd->n", x - a, ab) / np.where(dd < 1e-300, 1.0, dd), 0.0, 1.0)
    return a + t[:, None] * ab, t


def closest_point_on_triangle(x: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Closest point on each triangle ``V`` (``(n, 3, 3)``) to each ``x``, plus its barycentrics.

    The in-plane projection when it lands inside; otherwise the nearest of the three edges (Ericson,
    *Real-Time Collision Detection*, 5.1.5). Clamping to the edge is what makes a query beyond the
    surface's rim pair with the rim rather than with a facet it does not overlap.
    """
    a, b, c = V[:, 0], V[:, 1], V[:, 2]
    ab, ac = b - a, c - a
    nrm = np.cross(ab, ac)
    n2 = np.einsum("nd,nd->n", nrm, nrm)
    n2s = np.where(n2 < 1e-300, 1.0, n2)
    ap = x - a
    l2 = np.einsum("nd,nd->n", np.cross(ab, ap), nrm) / n2s
    l1 = np.einsum("nd,nd->n", np.cross(ap, ac), nrm) / n2s
    bary = np.stack([1.0 - l1 - l2, l1, l2], axis=1)
    inside = (bary >= -1e-12).all(axis=1)
    p = np.einsum("nk,nkd->nd", bary, V)
    if not inside.all():
        out = ~inside
        best_d = best_p = best_b = None
        for i, j in ((0, 1), (1, 2), (2, 0)):
            q, t = closest_point_on_segment(x[out], V[out, i], V[out, j])
            d = np.linalg.norm(q - x[out], axis=1)
            bb = np.zeros((int(out.sum()), 3))
            bb[:, i], bb[:, j] = 1.0 - t, t
            if best_d is None:
                best_d, best_p, best_b = d, q, bb
            else:
                take = d < best_d
                best_d = np.where(take, d, best_d)
                best_p = np.where(take[:, None], q, best_p)
                best_b = np.where(take[:, None], bb, best_b)
        p[out], bary[out] = best_p, best_b
    return p, bary


def _narrow(x: np.ndarray, V: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest point, its facet-local coordinates, and the distance -- for one query against many facets."""
    if dim == 2:
        p, t = closest_point_on_segment(x, V[:, 0], V[:, 1])
        loc = t[:, None]
    else:
        p, loc = closest_point_on_triangle(x, V)
    return p, loc, np.linalg.norm(p - x, axis=1)


def project_points(
    query: np.ndarray,
    m_facets: np.ndarray,
    points: np.ndarray,
    secondary_normals: np.ndarray,
    *,
    capture: Optional[float] = None,
    exclude_nodes: Optional[Sequence[Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair each query point with the nearest main facet: ``(ids, w, g0, active)``.

    ``ids`` and ``w`` are ``(n_q, k)`` -- the main nodes each query reads and their shape values, so
    ``u_m . Phi`` is a plain weighted sum. ``sum(w, axis=1) == 1`` on an active query, so a constant
    main field is reproduced exactly. ``g0`` is the along-normal separation ``n . (Phi(x) - x)`` with
    ``n`` the **secondary** outward normal, which points at the main body: positive is open, negative
    is penetrating. Handing it the opposite normal returns ``-g0`` and every downstream sign follows.

    ``capture=None`` means **unbounded** -- every query pairs with its nearest facet however far away,
    which is what the build-time gap tables have always done (they clamp to the nearest facet rather
    than dropping a query). A finite ``capture`` is what a search wants: beyond it a query is
    **inactive**, and keeps its slot with ``w = 0`` and ``g0 = OPEN_GAP``. That is deliberate. The
    tables stay a fixed ``(n_q, k)`` shape however the bodies move, so re-pairing changes only their
    values and never forces a retrace.

    ``exclude_nodes`` is a per-query iterable of node ids the pairing may not use -- a secondary
    facet's own adjacency ring, which is what lets a surface search against *itself* without every
    facet trivially contacting its own neighbour.
    """
    pts = np.asarray(points, dtype=float)
    facets = np.asarray(m_facets, dtype=int)
    dim = int(pts.shape[1])
    q = np.asarray(query, dtype=float).reshape(-1, dim)
    k = int(facets.shape[1])
    if q.shape[0] == 0 or facets.shape[0] == 0:
        return (np.zeros((0, k), int), np.zeros((0, k)), np.zeros(0), np.zeros(0, bool))

    V, cent, rad = facet_geometry(facets, pts)
    best_i = np.zeros(len(q), dtype=int)
    best_d = np.full(len(q), np.inf)
    best_p = np.zeros((len(q), dim))
    best_loc = np.zeros((len(q), 3 if dim == 3 else 1))

    if capture is None and exclude_nodes is None:
        # Unbounded and unfiltered: every facet is a candidate, so scan them all. Same O(n_q * n_f) the
        # flattened path already paid, just chunked so a large interface cannot exhaust memory.
        for lo in range(0, len(q), _CHUNK):
            qs = q[lo : lo + _CHUNK]
            d = np.empty((len(qs), len(facets)))
            ps = np.empty((len(qs), len(facets), dim))
            ls = np.empty((len(qs), len(facets), best_loc.shape[1]))
            for f in range(len(facets)):
                ps[:, f], ls[:, f], d[:, f] = _narrow(qs, np.repeat(V[f][None], len(qs), 0), dim)
            j = np.argmin(d, axis=1)
            r = np.arange(len(qs))
            best_i[lo : lo + len(qs)] = j
            best_d[lo : lo + len(qs)] = d[r, j]
            best_p[lo : lo + len(qs)] = ps[r, j]
            best_loc[lo : lo + len(qs)] = ls[r, j]
    else:
        from scipy.spatial import cKDTree  # local import: the pattern the rest of jNO uses for scipy

        tree = cKDTree(cent)
        # BROAD PHASE. The radius must cover the capture distance PLUS the largest facet: a query
        # sitting off the end of a long facet is far from that facet's centroid but close to the facet.
        reach = (float(capture) if capture is not None else float(np.linalg.norm(pts.max(0) - pts.min(0)))) + float(
            rad.max()
        )
        cand = tree.query_ball_point(q, reach)
        for n, cs in enumerate(cand):
            if not len(cs):
                continue
            cs = np.asarray(cs, dtype=int)
            if exclude_nodes is not None:
                ex = np.asarray(list(exclude_nodes[n]), dtype=int)
                if ex.size:
                    cs = cs[~np.isin(facets[cs], ex).any(axis=1)]
                if not len(cs):
                    continue
            p, loc, d = _narrow(np.repeat(q[n][None, :], len(cs), axis=0), V[cs], dim)
            j = int(np.argmin(d))
            best_i[n], best_d[n], best_p[n], best_loc[n] = cs[j], d[j], p[j], loc[j]

    active = best_d <= (np.inf if capture is None else float(capture))
    ids = facets[best_i]
    w = _edge_shape(best_loc[:, 0], k) if dim == 2 else _tri_shape(best_loc, k)
    nrm = np.asarray(secondary_normals, dtype=float).reshape(-1, dim)
    # Measured from the secondary TOWARD the main (``proj - x``), because ``n`` points that way.
    g0 = np.einsum("qd,qd->q", nrm, best_p - q)
    w = np.where(active[:, None], w, 0.0)
    g0 = np.where(active, g0, OPEN_GAP)
    return ids, w, g0, active


# ----------------------------------------------------------------------------------------------
# Arc length along an interface chain -- the parametrisation a CURVED tie needs
# ----------------------------------------------------------------------------------------------
def chain_order(facets: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Order edge facets into a connected chain: ``(facet_order, is_closed)``.

    The mortar parametrises a 2-D interface by ONE scalar and clips secondary against main edges in
    it. Projecting onto a fitted tangent plane supplies that scalar only while the interface is flat:
    a corner folds its two arms onto the same interval, and a closed loop folds everything. Arc length
    along the chain is the coordinate that cannot fold, because it is monotone along the interface by
    construction.

    Returns the facet indices in traversal order. ``is_closed`` distinguishes a loop -- an annular air
    gap between a rotor and a stator -- from an open chain, which matters because a loop's coordinate
    is periodic and a facet may straddle the seam.

    Raises if the facets do not form a single chain: a branching or disjoint interface has no arc
    length, and silently picking one branch is exactly the class of error this module exists to end.
    """
    f = np.asarray(facets, dtype=int)
    ends = f[:, :2]
    adj: dict = {}
    for i, (a, b) in enumerate(ends):
        adj.setdefault(int(a), []).append(i)
        adj.setdefault(int(b), []).append(i)
    deg = {v: len(e) for v, e in adj.items()}
    if any(d > 2 for d in deg.values()):
        raise ValueError(
            f"interface facets do not form a simple chain: {sum(d > 2 for d in deg.values())} node(s) "
            "join three or more edges. Arc length is undefined on a branching interface."
        )
    tips = [v for v, d in deg.items() if d == 1]
    closed = len(tips) == 0
    if not closed and len(tips) != 2:
        raise ValueError(f"interface has {len(tips)} loose end(s); expected 0 (a loop) or 2 (an open chain).")

    start_v = min(tips) if tips else int(ends[0, 0])
    order, seen_f, v = [], set(), start_v
    while True:
        nxt = [i for i in adj[v] if i not in seen_f]
        if not nxt:
            break
        i = nxt[0]
        seen_f.add(i)
        order.append(i)
        a, b = int(ends[i, 0]), int(ends[i, 1])
        v = b if a == v else a
    if len(order) != len(f):
        raise ValueError(
            f"interface facets are disjoint: walked {len(order)} of {len(f)} from one end, so this is "
            "more than one connected piece. Tie each piece as its own interface."
        )
    return np.asarray(order, dtype=int), closed


def chain_arclength(facets: np.ndarray, points: np.ndarray):
    """Arc-length coordinate of an interface chain: ``(order, closed, starts, lengths, total)``.

    ``starts[k]`` is the arc length at the first endpoint of the ``k``-th facet in traversal order, so
    a point at local parameter ``t`` on that facet sits at ``starts[k] + t * lengths[k]``.
    """
    f = np.asarray(facets, dtype=int)
    pts = np.asarray(points, dtype=float)
    order, closed = chain_order(f, pts)
    # walk again to get each facet's traversal ORIENTATION, so `t` runs the same way along the chain
    ends = f[order][:, :2]
    v = int(ends[0, 0]) if len(order) == 1 else (
        int(ends[0, 0]) if int(ends[0, 0]) not in ends[1, :2].tolist() else int(ends[0, 1]))
    a_list, b_list = [], []
    for a, b in ends:
        a, b = int(a), int(b)
        if a == v:
            a_list.append(a); b_list.append(b); v = b
        else:
            a_list.append(b); b_list.append(a); v = a
    A, B = np.asarray(a_list), np.asarray(b_list)
    lengths = np.linalg.norm(pts[B] - pts[A], axis=1)
    starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
    return order, closed, A, B, starts, lengths, float(lengths.sum())


def arclength_of(query: np.ndarray, facets: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Arc-length coordinate of arbitrary points, by closest point on the chain.

    Both sides of a tie must be measured on the SAME ruler, so the main chain supplies it and every
    node -- main or secondary -- is located on it. Reuses the closest-point pairing above, which is
    why this lives here rather than in the tie module.
    """
    _o, closed, A, B, starts, lengths, total = chain_arclength(facets, points)
    pts = np.asarray(points, dtype=float)
    q = np.asarray(query, dtype=float).reshape(-1, pts.shape[1])
    a, b = pts[A], pts[B]
    best_s = np.zeros(len(q))
    best_d = np.full(len(q), np.inf)
    for k in range(len(A)):  # chains are short (an interface, not a mesh); clarity over vectorising
        p, t = closest_point_on_segment(q, np.repeat(a[k][None], len(q), 0), np.repeat(b[k][None], len(q), 0))
        d = np.linalg.norm(p - q, axis=1)
        take = d < best_d
        best_d = np.where(take, d, best_d)
        best_s = np.where(take, starts[k] + t * lengths[k], best_s)
    return best_s, total, closed
