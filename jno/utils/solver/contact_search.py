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

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .fem_utils import _edge_shape, _log, _tri_shape

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
    main_normals: Optional[np.ndarray] = None,
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

    ``main_normals`` (one outward normal per main facet) restricts the pairing to facets that FACE the
    query, ``n_s . n_m < 0``. Adjacency exclusion alone is not enough for self-contact: two facets a
    few rings apart on a flat stretch are collinear, so ``Phi(x) - x`` lies along the surface, ``g0``
    comes out ~0, and the surface reads as touching itself everywhere. Their normals are *parallel*,
    not opposed, which is what separates them from the two sides of a genuine fold. Omitted (the
    default) nothing is filtered, so two-body pairing is unchanged.
    """
    pts = np.asarray(points, dtype=float)
    facets = np.asarray(m_facets, dtype=int)
    dim = int(pts.shape[1])
    q = np.asarray(query, dtype=float).reshape(-1, dim)
    k = int(facets.shape[1])
    if q.shape[0] == 0 or facets.shape[0] == 0:
        return (np.zeros((0, k), int), np.zeros((0, k)), np.zeros(0), np.zeros(0, bool))

    V, cent, rad = facet_geometry(facets, pts)
    nrm_q = np.asarray(secondary_normals, dtype=float).reshape(-1, dim)
    best_i = np.zeros(len(q), dtype=int)
    best_d = np.full(len(q), np.inf)
    best_p = np.zeros((len(q), dim))
    best_loc = np.zeros((len(q), 3 if dim == 3 else 1))

    mn = None if main_normals is None else np.asarray(main_normals, dtype=float).reshape(-1, dim)
    if mn is not None and len(mn) != len(facets):
        raise ValueError(f"project_points: main_normals has {len(mn)} rows for {len(facets)} main facets.")
    if capture is None and exclude_nodes is None and mn is None:
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
            if mn is not None:
                cs = cs[mn[cs] @ nrm_q[n] < 0.0]  # keep only facets whose outward normal faces the query
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


# ---------------------------------------------------------------------------------------------------
# The re-pairing driver -- `fem.solve(contact=...)`
# ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ContactSpec:
    """Controls for the contact-search loop (``FEM.solve(contact=...)``). See :func:`jno.solve.contact`."""

    capture: Optional[float] = None
    rounds: int = 12
    tol: float = 1e-4


def _pairing_moved(a, b):
    """How many quadrature-point slots changed which main nodes they read, between two payloads."""
    n = 0
    for k, tb in b.items():
        prev = (a or {}).get(k)
        if prev is None:  # the driver seeds a pairing before round 1, so there is always one to compare
            raise KeyError(f"contact pairing {k!r} has no previous state to compare against")
        ia, ib = np.asarray(prev["ids_full"]), np.asarray(tb["ids_full"])
        wa, wb = np.asarray(prev["w_full"]), np.asarray(tb["w_full"])
        # A slot counts as moved when it reads a different node OR when an ACTIVE/inactive flip happened
        # (an inactive slot keeps its ids and only zeroes its weights, so ids alone would miss it).
        moved = (ia != ib).any(axis=-1) | ((wa.sum(-1) > 0) != (wb.sum(-1) > 0))
        n += int(moved.sum())
    return n


def run_contact_solve(fem, spec, *, solve_fn=None, **kwargs):
    """Solve, re-pair from the deformed configuration, repeat until the pairing settles.

    The frozen pairing built at ``jno.fem(...)`` time is valid only while displacements are far below
    the element size. Past that a secondary point is still tied to the facet it faced in the REFERENCE
    configuration -- and nothing reports it, because the solve converges perfectly well, just for a
    contact configuration that is not the one being solved.

    The signature is that refining makes it WORSE. On a 12:20 involute gear pair, against the kinematic
    oracle ``|T_B/T_A| = z_B/z_A``, the same problem solved both ways as the rim mesh went 0.050 ->
    0.018::

        frozen pairing   1.97%   2.33%   2.66%   2.83%      <- grows as h falls
        re-paired        2.24%   2.27%   2.30%   2.30%      <- settles

    (The ~2.3% both share at that drive is the demo geometry, not the pairing: it is flat in h, GROWS
    with the penalty toward 3.6%, and does not move when the involute flank is sampled twice as finely.)
    Where the pairing genuinely goes stale the error is not subtle -- a flat-bottomed block slid 0.9
    across a disk of radius 1 reads a separation of 0.05, the value at its starting position, where the
    truth is 0.182.

    Each round solves the ordinary system with the current pairing threaded on ``args``, then calls the
    operator's host-side search at ``x + u``. Two things must hold to stop: the pairing is unchanged
    (no quadrature point moved to a different main facet, none flipped active) and the solution stopped
    moving. Exhausting ``rounds`` without both **raises**, naming how many slots are still oscillating
    -- a contact solve that quietly stops iterating is the classic plausible-wrong answer.
    """
    op = getattr(fem, "_op", None)
    repair = getattr(op, "repair_contact", None)
    if repair is None or not getattr(op, "contact_pairs", None):
        raise ValueError(
            "fem.solve(contact=...) but this form declares no contact pair. `contact=` re-runs the "
            "search behind `u.gap(secondary, main)` / `u.slide(...)`; without one there is nothing to "
            "re-pair. Add the gap to the term list, or drop `contact=`."
        )
    if kwargs.get("tau") is not None:
        raise NotImplementedError(
            "fem.solve(contact=..., tau=...) is not implemented. The load-path march compiles ONE step "
            "and replays it under `lax.scan` -- which is what keeps a load path reverse-mode "
            "differentiable end to end -- and a host-side search cannot run inside a scan. Solve the "
            "steady problem at each load level yourself with `contact=`, or accept the frozen pairing "
            "and drop `contact=`."
        )
    if getattr(kwargs.get("nonlinear"), "direct", False):
        raise NotImplementedError(
            "fem.solve(contact=..., nonlinear=jno.solve.newton(direct=True)) is not supported: the "
            "assembled tangent hoists the contact block's sparsity pattern once, from the pairing's "
            "concrete node ids, and re-pairing changes which main nodes each point reads. Use the "
            "matrix-free default `jno.solve.newton()`, whose tangent is `jax.linearize` of the residual."
        )
    rounds = int(spec.rounds)
    if rounds < 1:
        raise ValueError(f"fem.solve(contact=...): rounds must be at least 1, got {rounds}.")

    base_res, base_jac = op.residual, op.jacobian
    cell: dict = {"tb": None}

    def _inject(args):
        return {**(args or {}), "__gap_tables__": cell["tb"]} if cell["tb"] else args

    def _drop_compiled():
        """Discard the operator's compiled-solve cache.

        ``FemResidualOperator.solve`` caches a ``jax.jit`` of a closure that reads ``self.residual`` at
        TRACE time, keyed only on the solver identity and the argument shapes -- none of which change
        between rounds. Left alone it would replay round 1's pairing for every later round, and worse,
        an ordinary ``fem.solve()`` afterwards would silently get the searched answer. Re-pairing
        changes what the residual IS, so its compilation cannot be reused; that retrace is the real cost
        of the search on a runtime-parametric form.
        """
        op.__dict__.pop("_eager_solve_cache", None)

    try:
        fem._in_contact_loop = True  # the rounds re-enter `_solve_dispatch` with contact=None, by design
        op.residual = lambda u, args=None, _b=base_res: _b(u, _inject(args))
        if base_jac is not None:
            op.jacobian = lambda u, args=None, _b=base_jac: _b(u, _inject(args))

        # Round 1 must ALREADY use a searched pairing, not the frozen one. The build-time tables are
        # built unbounded -- every secondary point clamps to its nearest main facet however far away --
        # so on a closed body the points on the FAR side pair with a main surface that sits behind them,
        # `g0 = n . (Phi(x) - x)` comes out large and negative, and a penalty reads that as a huge
        # interpenetration. Measured on a gear pair: |u|max 5.9e-01 where the rigid drive is 6e-03, and
        # the torque ratio 0.82 against an exact 1.667. `capture` is what excludes those points, and a
        # caller who passed `contact=` asked for the search -- so it applies from the first solve.
        u_prev, moved, du, rel = None, None, float("inf"), float("inf")
        cell["tb"] = repair(np.zeros(int(op.size)), capture=spec.capture)
        for rnd in range(rounds):
            _drop_compiled()
            # Warm-start from the previous round: consecutive rounds differ only in which facets a few
            # quadrature points read, so the last solution is a far better guess than the caller's x0
            # and Newton reaches the new equilibrium in a fraction of the steps.
            kw = dict(kwargs) if u_prev is None else {**kwargs, "x0": u_prev}
            u = np.asarray(fem._solve_dispatch(solve_fn=solve_fn, **kw)).reshape(-1)
            new = repair(u, capture=spec.capture)
            moved = _pairing_moved(cell["tb"], new)
            du = np.inf if u_prev is None else float(np.abs(u - u_prev).max())
            # RELATIVE to the solution's own size. A floor of 1.0 here turned this into an absolute
            # 1e-6 test, which on a gear whose displacement is 6e-3 asked for four digits more than the
            # answer has -- the pairing had settled (0 slots moving for two rounds) and the loop still
            # raised. The floor exists only to keep 0/0 finite when nothing moved at all.
            scale = max(float(np.abs(u).max()), 1e-30)
            # One argument: `PrintFallback.info` (the no-logging-configured path) takes exactly one,
            # so %-style args would raise here and only here -- in the branch nobody runs under pytest.
            _log.info(
                f"contact round {rnd + 1}/{rounds}: "
                f"{moved} slot(s) re-paired, |du| = {du:.3e}"
            )
            if moved == 0 and du <= spec.tol * scale:
                fem.contact_rounds = rnd + 1
                return u
            cell["tb"], u_prev, rel = new, u, du / scale
    finally:
        fem._in_contact_loop = False
        op.residual, op.jacobian = base_res, base_jac
        _drop_compiled()

    if not np.isfinite(du):  # only ever true after a single round: nothing to difference against
        raise RuntimeError(
            f"fem.solve(contact=...): only {rounds} round was run, so the solution was never compared "
            "against a previous one -- a single round cannot show that the search has settled, however "
            "good its answer looks. Use rounds >= 2."
        )
    if moved == 0:
        raise RuntimeError(
            f"fem.solve(contact=...): after {rounds} round(s) the PAIRING has settled -- no quadrature "
            f"point changes which main facet it reads -- but the solution is still moving, by "
            f"{rel:.2e} relative on the last round against a tolerance of {spec.tol:.0e}. The gap keeps "
            "shifting inside the facets it already pairs with, which is an ordinary fixed point and "
            "contracts by roughly 0.2-0.3 per round: raise `rounds=`, or loosen `tol=` if that "
            "precision is more than the answer needs."
        )
    raise RuntimeError(
        f"fem.solve(contact=...): the pairing had not settled after {rounds} round(s) -- {moved} "
        f"quadrature slot(s) still change which main facet they read, and the solution last moved by "
        f"{du:.3e}. Either the search is still converging or it is oscillating between two pairings. "
        f"Raise `rounds=`, or widen `capture=` if points are flickering in and out of the search radius."
    )
