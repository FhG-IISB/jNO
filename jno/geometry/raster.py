"""Exact area fractions of a region on a non-conforming triangle mesh (no polygon library).

``cell_fractions(mesh, region)`` answers "how much of each triangle lies inside ``region``" -- the input a
fixed-mesh (fictitious-domain / immersed) material model needs when the geometry changes from design to design
but the mesh must not (reduced-order models, design sweeps, surrogates). Optional ``normals=True`` also returns,
per cell, the direction of the region's outline inside it (for anisotropic interface homogenisation).

Exact path -- every leaf is an axis-aligned ``shape.rect`` combined with ``|``, ``-``, ``&`` or ``shape.regions``:
the rectangle edges define a non-uniform grid on which the region is EXACTLY a set of grid cells (membership of
each grid cell = ``shape.contains`` at its midpoint, the analytic CSG test). The grid cells are disjoint
rectangles, so ``area(tri & region) = sum_cells area(tri & cell)``, and the overlap of a triangle with a rectangle
has a closed form: insert the points where the triangle's edges cross the four rectangle lines, clamp to the
rectangle (the clamped detours are collinear and enclose no area), shoelace. Fixed array sizes, no
data-dependent branches: the same kernel runs under numpy or ``jax.numpy`` (jit/vmap-able).
``close=c`` applies a morphological closing by the square ``[-c, c]^2`` first (fills slits narrower than ``2c``
between touching pieces, e.g. rounding gaps in a CAD export); exact on the grid as well.

Fallback -- any other shape (disks, polygons, rotated rects): each triangle is split into ``4**levels`` congruent
sub-triangles and ``shape.contains`` is evaluated at their centroids. First-order in the sub-cell size; a warning
says so.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = ["cell_fractions", "rect_boxes", "union_boxes", "closing_boxes", "outline_segments", "tri_box_area"]


# ---------------------------------------------------------------------------------------------------- input forms
def _mesh_arrays(mesh):
    """jno domain | (points, triangles) -> (points (N, 2) float64, triangles (C, 3) int64)."""
    if isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        pts, tris = mesh
    else:
        m = mesh.built_mesh if hasattr(mesh, "built_mesh") else mesh.mesh
        pts, tris = m.points, m.cells_dict["triangle"]
    pts = np.asarray(pts, np.float64)[:, :2]
    return pts, np.asarray(tris, np.int64)


def _rect_leaves(node, out):
    """Collect the Rect leaves of a CSG tree; False if the tree has anything the exact path cannot take."""
    from .primitives import Rect

    kind = node[0]
    if kind == "leaf":
        if not isinstance(node[1], Rect):
            return False
        out.append(node[1])
        return True
    if kind in ("fuse", "cut", "inter"):
        return _rect_leaves(node[1]._node, out) and _rect_leaves(node[2]._node, out)
    if kind == "regions":
        return all(_rect_leaves(sub._node, out) for _n, sub in node[1])
    return False


def rect_boxes(region, tol=1e-12):
    """Disjoint boxes ``(n, 4)`` [x0, y0, x1, y1] covering ``region`` exactly, or ``None`` if ``region`` is not
    built from axis-aligned rectangles. ``region``: a 2-D ``jno.shape`` or an ``(n, 4)`` array of (possibly
    overlapping) boxes, read as their union."""
    if not hasattr(region, "_node"):
        return union_boxes(region, tol)
    leaves = []
    if region.dim != 2 or not _rect_leaves(region._node, leaves):
        return None
    xs = _uniq(np.array([v for r in leaves for v in (r.x0, r.x1)], float), tol)
    ys = _uniq(np.array([v for r in leaves for v in (r.y0, r.y1)], float), tol)
    xm, ym = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    X, Y = np.meshgrid(xm, ym, indexing="ij")
    occ = np.asarray(region.contains(np.c_[X.ravel(), Y.ravel()], tol=0.0)).reshape(X.shape)
    return _boxes(xs, ys, occ)


# ---------------------------------------------------------------------------------------- rectangle-union algebra
def _uniq(v, tol):
    v = np.sort(np.asarray(v, float))
    return v[np.r_[True, np.diff(v) > tol]] if len(v) else v


def _grid(boxes, tol=1e-12):
    """(n, 4) boxes -> (xs, ys, occ): occ[i, j] = grid cell [xs_i, xs_i+1] x [ys_j, ys_j+1] lies in a box."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    xs = _uniq(np.r_[boxes[:, 0], boxes[:, 2]], tol)
    ys = _uniq(np.r_[boxes[:, 1], boxes[:, 3]], tol)
    return xs, ys, _cover(xs, ys, boxes)


def _cover(xs, ys, boxes):
    xm, ym = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    occ = np.zeros((len(xm), len(ym)), bool)
    for x0, y0, x1, y1 in boxes:
        occ |= ((xm > x0) & (xm < x1))[:, None] & ((ym > y0) & (ym < y1))[None, :]
    return occ


def _boxes(xs, ys, occ):
    """Occupied grid cells -> disjoint boxes (runs along y merged within each grid column)."""
    out = []
    for i in range(occ.shape[0]):
        col = np.r_[False, occ[i], False].astype(np.int8)
        d = np.diff(col)
        for j0, j1 in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            out.append((xs[i], ys[j0], xs[i + 1], ys[j1]))
    return np.array(out, float).reshape(-1, 4)


def union_boxes(boxes, tol=1e-12):
    """Disjoint boxes covering the union of ``boxes`` (n, 4)."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    return _boxes(*_grid(b, tol)) if len(b) else b


def closing_boxes(boxes, c):
    """Morphological closing (dilate, then erode) of a union of boxes by the square ``[-c, c]^2`` -- shapely's
    ``buffer(c, join_style="mitre").buffer(-c, join_style="mitre")`` for axis-aligned input. Exact: dilating a
    union of boxes = union of the dilated boxes, and erosion = complement(dilation(complement))."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    if c <= 0 or len(b) == 0:
        return union_boxes(b)
    grow = np.array([-c, -c, c, c])
    xs, ys, occ = _grid(b + grow)
    pad = 4 * c
    X, Y = np.r_[xs[0] - pad, xs, xs[-1] + pad], np.r_[ys[0] - pad, ys, ys[-1] + pad]
    empty = np.ones((len(X) - 1, len(Y) - 1), bool)
    empty[1:-1, 1:-1] = ~occ
    comp = _boxes(X, Y, empty) + grow  # dilated complement
    xs2, ys2, _ = _grid(np.r_[comp, [[X[0], Y[0], X[-1], Y[-1]]]])
    return _boxes(xs2, ys2, ~_cover(xs2, ys2, comp))


def outline_segments(boxes):
    """Outline of the union of ``boxes`` as axis-aligned pieces: ``(vertical (n, 3) [x, y0, y1], horizontal (m, 3)
    [y, x0, x1])`` -- grid edges with an occupied cell on one side and an empty one on the other."""
    xs, ys, occ = _grid(boxes)
    P = np.zeros((occ.shape[0] + 2, occ.shape[1] + 2), bool)
    P[1:-1, 1:-1] = occ
    iv, jv = np.nonzero(P[1:, 1:-1] != P[:-1, 1:-1])
    ih, jh = np.nonzero(P[1:-1, 1:] != P[1:-1, :-1])
    return np.c_[xs[iv], ys[jv], ys[jv + 1]], np.c_[ys[jh], xs[ih], xs[ih + 1]]


# ------------------------------------------------------------------------------------------- triangle kernels
def _insert_crossings(P, axis, a, xp):
    """P (n, K, 2) closed polygons -> (n, 2K, 2): after every vertex, the point where the edge to the next vertex
    crosses the line coord[axis] = a (a duplicate of the vertex where it does not cross)."""
    Q = xp.roll(P, -1, axis=1)
    p, q, a = P[..., axis], Q[..., axis], a[:, None]
    den = q - p
    ok = xp.abs(den) > 0
    t = xp.clip(xp.where(ok, (a - p) / xp.where(ok, den, 1.0), 0.0), 0.0, 1.0)
    X = xp.where(((p - a) * (q - a) < 0)[..., None], P + t[..., None] * (Q - P), P)
    return xp.stack([P, X], axis=2).reshape(P.shape[0], -1, 2)


def tri_box_area(T, B, xp=np):
    """Overlap areas ``(n,)`` of triangles ``T (n, 3, 2)`` with boxes ``B (n, 4)`` [x0, y0, x1, y1].
    ``xp`` = numpy or jax.numpy (differentiable in both the vertices and the box corners)."""
    P = T
    for axis, lo, hi in ((0, B[:, 0], B[:, 2]), (1, B[:, 1], B[:, 3])):
        P = _insert_crossings(_insert_crossings(P, axis, lo, xp), axis, hi, xp)
        c = xp.clip(P[..., axis], lo[:, None], hi[:, None])
        P = xp.stack([c, P[..., 1]], -1) if axis == 0 else xp.stack([P[..., 0], c], -1)
    x, y = P[..., 0], P[..., 1]
    return 0.5 * xp.abs(xp.sum(x * xp.roll(y, -1, axis=1) - xp.roll(x, -1, axis=1) * y, axis=1))


def _segment_in_tri(T, axis, c, lo, hi):
    """Length of the axis-aligned segments {coord[axis] = c, other coord in [lo, hi]} inside triangles T (n, 3, 2);
    c, lo, hi (n,). A segment lying on a triangle edge counts for that triangle."""
    o = 1 - axis
    A, B = T, np.roll(T, -1, axis=1)
    p, q, c2 = A[..., axis], B[..., axis], c[:, None]
    den = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.nan_to_num((c2 - p) / den)
    hit = ((p - c2) * (q - c2) <= 0) & (np.abs(den) > 0)
    s = A[..., o] + np.clip(t, 0, 1) * (B[..., o] - A[..., o])
    on = (np.abs(p - c2) < 1e-15) & (np.abs(q - c2) < 1e-15)
    s_lo = np.minimum(np.where(hit, s, np.inf).min(1), np.where(on, np.minimum(A[..., o], B[..., o]), np.inf).min(1))
    s_hi = np.maximum(np.where(hit, s, -np.inf).max(1), np.where(on, np.maximum(A[..., o], B[..., o]), -np.inf).max(1))
    return np.maximum(np.minimum(s_hi, hi) - np.maximum(s_lo, lo), 0.0)


class _Locator:
    """Bounding-box candidate search over a fixed triangle set (sorted sweep in x, filter in y)."""

    def __init__(self, T):
        self.T = T
        self.lo, self.hi = T.min(1), T.max(1)
        self.order = np.argsort(self.lo[:, 0])
        self.lo_sorted = self.lo[self.order, 0]
        self.maxw = float((self.hi[:, 0] - self.lo[:, 0]).max()) if len(T) else 0.0

    def pairs(self, boxes, eps=0.0):
        """All (cell, box) pairs whose bounding boxes overlap (closed with eps)."""
        cs, bs = [], []
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            i0 = np.searchsorted(self.lo_sorted, x0 - eps - self.maxw, "left")
            i1 = np.searchsorted(self.lo_sorted, x1 + eps, "right")
            c = self.order[i0:i1]
            c = c[
                (self.hi[c, 0] >= x0 - eps)
                & (self.lo[c, 0] <= x1 + eps)
                & (self.hi[c, 1] >= y0 - eps)
                & (self.lo[c, 1] <= y1 + eps)
            ]
            cs.append(c)
            bs.append(np.full(len(c), k))
        if not cs:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(cs), np.concatenate(bs)


# ------------------------------------------------------------------------------------------------- public entry
def cell_fractions(
    mesh, region, *, close: float = 0.0, normals: bool = False, exclude_x: float | None = None, levels: int = 4
):
    """Area fraction of every triangle of ``mesh`` inside ``region``.

    Parameters
    ----------
    mesh
        A meshed 2-D ``jno.domain`` or ``(points (N, 2), triangles (C, 3))``.
    region
        A 2-D ``jno.shape`` (exact when built from ``shape.rect`` with ``|``/``-``/``&``/``regions``), or an
        ``(n, 4)`` array of boxes [x0, y0, x1, y1] read as their union.
    close
        Morphological closing radius applied to the region first (exact path only).
    normals
        Also return ``nx2`` (C,): per cell, the share of the region's outline length inside the cell that is
        VERTICAL (outline normal along x), i.e. the length-weighted <n_x^2>; 0.5 where no outline passes.
    exclude_x
        Outline pieces lying on the vertical line x = exclude_x are not interfaces (a symmetry axis); ignored
        for ``normals``.
    levels
        Fallback only: each triangle is split into ``4**levels`` sub-triangles for the membership quadrature.

    Returns ``frac (C,)`` in [0, 1] (exactly 0 / 1 within 1e-12), or ``(frac, nx2)`` with ``normals=True``.
    """
    pts, tris = _mesh_arrays(mesh)
    T = pts[tris]
    area = 0.5 * np.abs(
        (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
    )
    boxes = rect_boxes(region)
    if boxes is None:
        if close or normals:
            raise NotImplementedError("cell_fractions: close= and normals= need a region built from shape.rect.")
        warnings.warn(
            "cell_fractions: region is not a union of axis-aligned rectangles; using sub-cell quadrature "
            f"(4**{levels} points per cell), first-order accurate.",
            stacklevel=2,
        )
        return _fractions_quadrature(T, region, levels)
    if close:
        boxes = closing_boxes(boxes, close)
    loc = _Locator(T)
    frac = np.zeros(len(T))
    if len(boxes):
        c, b = loc.pairs(boxes)
        np.add.at(frac, c, tri_box_area(T[c], boxes[b]))
    frac = frac / area
    frac = np.where(frac < 1e-12, 0.0, np.where(frac > 1.0 - 1e-12, 1.0, frac))  # round-off at full/empty cells
    if not normals:
        return frac
    vert, hori = outline_segments(boxes) if len(boxes) else (np.zeros((0, 3)), np.zeros((0, 3)))
    if exclude_x is not None:
        vert = vert[np.abs(vert[:, 0] - exclude_x) > 1e-15]
    Sx, Sy = np.zeros(len(T)), np.zeros(len(T))
    for segs, axis, S in ((vert, 0, Sx), (hori, 1, Sy)):
        if not len(segs):
            continue
        bx = (
            np.c_[segs[:, 0], segs[:, 1], segs[:, 0], segs[:, 2]]
            if axis == 0
            else np.c_[segs[:, 1], segs[:, 0], segs[:, 2], segs[:, 0]]
        )
        c, k = loc.pairs(bx, eps=1e-15)
        np.add.at(S, c, _segment_in_tri(T[c], axis, segs[k, 0], segs[k, 1], segs[k, 2]))
    nx2 = np.full(len(T), 0.5)
    tot = Sx + Sy
    nx2[tot > 0] = Sx[tot > 0] / tot[tot > 0]
    return frac, nx2


def _fractions_quadrature(T, region, levels):
    # barycentric centroids of the 4**levels congruent sub-triangles of the reference triangle
    tri = [np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])]
    for _ in range(levels):
        nxt = []
        for a, b, c in tri:
            ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
            nxt += [np.array(v) for v in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
        tri = nxt
    lam = np.array([t.mean(0) for t in tri])  # (q, 3)
    Q = np.einsum("qk,ckd->cqd", lam, T).reshape(-1, 2)
    inside = np.asarray(region.contains(Q, tol=0.0)).reshape(len(T), len(lam))
    return inside.mean(1)
