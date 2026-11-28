"""High-order quadrilaterals on a curved boundary: the nodes of a Q_k field placed on the true curve.

A straight-sided mesh keeps a polygonal domain whatever the degree: the geometry error is ``O(h^2)`` and caps
the spectral elements there (a Q8 Poisson on a disk stalls at the chord-to-arc gap). The cells' geometry in jNO
is isoparametric -- every cell maps through its field's own nodes -- so curving the domain means moving those
nodes. On each boundary edge the edge nodes are projected onto the boundary along the gradient of the shape's
signed distance (exact on the zero set for every analytic shape, booleans included), and the nodes inside the
cell follow by transfinite interpolation of the edge displacements (Gordon & Hall 1973), so the map stays a
smooth polynomial rather than one with a boundary layer of distortion; vertices and interior edges do not move.

Based on:
  Gordon, W. J. & Hall, C. A. (1973). Transfinite element methods: blending-function interpolation over
  arbitrary curved element domains. Numer. Math. 21, 109-129.
"""

from __future__ import annotations

import numpy as np

__all__ = ["curve_boundary_nodes"]

# basix vertex pairs of a quadrilateral's sides, vertices at (0,0), (1,0), (0,1), (1,1)
_SIDES = {"bottom": (0, 1), "top": (2, 3), "left": (0, 2), "right": (1, 3)}


def _project(sdf, P, scale, iters=40):
    """Points ``P`` moved onto the zero set of ``sdf`` by Newton steps along its gradient (central differences)."""
    P = np.array(P, dtype=float)
    if P.size == 0:
        return P
    h = 1e-7 * scale
    eye = np.eye(P.shape[1])
    for _ in range(iters):
        phi = np.asarray(sdf(P), dtype=float).reshape(-1)
        if np.abs(phi).max() <= 1e-14 * scale:
            break
        g = np.stack([(np.asarray(sdf(P + h * e)) - np.asarray(sdf(P - h * e))).reshape(-1) / (2 * h) for e in eye], 1)
        P = P - (phi / np.maximum((g * g).sum(axis=1), 1e-300))[:, None] * g
    phi = np.asarray(sdf(P), dtype=float).reshape(-1)
    if np.abs(phi).max() > 1e-10 * scale:
        raise ValueError(
            f"shape.quad().curved(): {int((np.abs(phi) > 1e-10 * scale).sum())} boundary node(s) could not be put on "
            "the shape's boundary (the signed distance does not vanish there). Mesh straight-sided (drop .curved())."
        )
    return P


def curve_boundary_nodes(shape, pts, cells_v, cells_f, order):
    """``pts`` with the nodes of every boundary cell of a Q_``order`` quadrilateral mesh moved onto ``shape``'s
    boundary. ``cells_v`` are the cells' vertices and ``cells_f`` their nodes, both in basix order; the vertices
    are ids ``0 .. nv-1`` of ``pts`` and stay where they are."""
    from basix import CellType

    from .fem_lagrange import _lagrange_basix, edge_node_parameters, lagrange_interp_points

    try:
        shape.sdf(np.zeros((1, 2)))
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"shape.quad().curved(): this shape has no signed distance to put boundary nodes on ({exc}). Mesh it "
            "straight-sided (drop .curved())."
        ) from None
    k = int(order)
    t = np.asarray(edge_node_parameters("quad", k), dtype=float)  # (k + 1,)
    ref = np.asarray(lagrange_interp_points(2, k, "quad"), dtype=float)  # (n_loc, 2), basix order
    ii = np.abs(ref[:, 0][:, None] - t[None, :]).argmin(axis=1)
    jj = np.abs(ref[:, 1][:, None] - t[None, :]).argmin(axis=1)
    lex = np.full((k + 1, k + 1), -1, dtype=np.int64)
    lex[ii, jj] = np.arange(ref.shape[0])
    if (lex < 0).any():
        raise RuntimeError("curve_boundary_nodes: the element's nodes are not a tensor grid of its edge nodes.")
    # the local nodes along each side, ordered by the side's own parameter
    along = {"bottom": lex[:, 0], "top": lex[:, k], "left": lex[0, :], "right": lex[k, :]}

    cells_v = np.asarray(cells_v, dtype=np.int64)
    cells_f = np.asarray(cells_f, dtype=np.int64)
    count = {}
    for name, (a, b) in _SIDES.items():
        for e in map(tuple, np.sort(cells_v[:, [a, b]], axis=1)):
            count[e] = count.get(e, 0) + 1
    on_boundary = {
        name: np.array([count[e] == 1 for e in map(tuple, np.sort(cells_v[:, [a, b]], axis=1))])
        for name, (a, b) in _SIDES.items()
    }
    curved_cells = np.flatnonzero(np.any(np.stack(list(on_boundary.values())), axis=0))
    if curved_cells.size == 0 or k < 2:
        return pts
    pts = np.array(pts, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    scale = float(np.linalg.norm(hi - lo)) or 1.0

    # project every boundary side's edge-interior nodes at once
    picks = [(name, c) for name in _SIDES for c in np.flatnonzero(on_boundary[name])]
    ids = np.concatenate([cells_f[c, along[name][1:k]] for name, c in picks]) if picks else np.zeros(0, int)
    moved = _project(shape.sdf, pts[ids], scale)
    delta_of = {}
    for n, (name, c) in enumerate(picks):
        d = np.zeros((k + 1, pts.shape[1]))
        d[1:k] = moved[n * (k - 1) : (n + 1) * (k - 1)] - pts[cells_f[c, along[name][1:k]]]
        delta_of[(name, c)] = d

    zero = np.zeros((k + 1, pts.shape[1]))
    new = pts.copy()
    for c in curved_cells:
        db, dt = delta_of.get(("bottom", c), zero), delta_of.get(("top", c), zero)
        dl, dr = delta_of.get(("left", c), zero), delta_of.get(("right", c), zero)
        # transfinite blend of the side displacements (the corners do not move, so no correction term)
        D = (
            (1.0 - t)[None, :, None] * db[:, None, :]
            + t[None, :, None] * dt[:, None, :]
            + (1.0 - t)[:, None, None] * dl[None, :, :]
            + t[:, None, None] * dr[None, :, :]
        )  # (k+1, k+1, dim): node (i, j)
        node = cells_f[c][lex]
        new[node] = pts[node] + D

    # every moved cell must keep a positive Jacobian at its nodes
    elem = _lagrange_basix(CellType.quadrilateral, k)
    tab = np.asarray(elem.tabulate(1, ref))  # (3, n_loc, n_loc, 1): at the nodes
    X = new[cells_f[curved_cells]]  # (C, n_loc, 2)
    J = np.stack([np.einsum("qn,cnd->cqd", tab[1 + a, :, :, 0], X) for a in range(2)], axis=-1)  # (C, q, d, a)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    X0 = pts[cells_f[curved_cells]]
    J0 = np.stack([np.einsum("qn,cnd->cqd", tab[1 + a, :, :, 0], X0) for a in range(2)], axis=-1)
    det0 = J0[..., 0, 0] * J0[..., 1, 1] - J0[..., 0, 1] * J0[..., 1, 0]
    bad = (det * np.sign(det0) <= 0).any(axis=1)
    if bad.any():
        sides = np.stack([on_boundary[name][curved_cells] for name in _SIDES], axis=1).sum(axis=1)
        corner = int((bad & (sides >= 2)).sum())
        why = (
            f"{corner} of them with two boundary edges meeting at a smooth point of the boundary: curved, the corner "
            "between them opens to 180 degrees and the cell degenerates there"
            if corner
            else "the boundary curves too much across one cell"
        )
        raise ValueError(
            f"shape.quad().curved(): putting the nodes on the boundary folds {int(bad.sum())} cell(s), {why}. Mesh "
            "finer near the curved boundary (a smaller size=), so that no cell spans that much of it."
        )
    return new
