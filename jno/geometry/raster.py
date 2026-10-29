"""``jno.geometry.raster`` — exact area fractions of a region on a triangle mesh that does not conform to it.

``cell_fractions(mesh, region)`` answers how much of each triangle lies inside ``region``. That is the input of a
fixed-mesh (fictitious-domain, immersed) material model, where the geometry changes from design to design but the
mesh must not: design sweeps, reduced-order models, surrogates. ``normals=True`` also returns, per cell, the
direction of the region's outline inside it, for an anisotropic interface homogenisation::

    frac = jno.geometry.cell_fractions(d, jno.shape.rect(0, 0, 0.3, 0.2) | jno.shape.rect(0.3, 0, 0.5, 0.1))

**Exact path.** Every leaf is an axis-aligned ``shape.rect`` combined with ``|``, ``-``, ``&`` or
``shape.regions``. The rectangle edges then define a non-uniform grid on which the region is exactly a set of
grid cells (each cell's membership is ``shape.contains`` at its midpoint, the analytic CSG test). The grid cells are
disjoint rectangles, so ``area(tri & region)`` is a sum over cells of ``area(tri & cell)``, and the overlap of a
triangle with a rectangle has a closed form: insert the points where the triangle's edges cross the four lines,
clamp to the rectangle (the clamped detours are collinear and enclose no area), shoelace. The kernel has fixed
array sizes and no data-dependent branch, so it runs under numpy or ``jax.numpy`` (``jit``, ``vmap``, ``grad``).
``close=c`` first applies a morphological closing by the square ``[-c, c]^2``: it fills slits narrower than ``2c``
between touching pieces (rounding gaps in a CAD export), exactly on the grid as well.

**Fallback.** Any other shape (disks, polygons, rotated rectangles): each triangle is split into ``4**levels``
congruent sub-triangles and ``shape.contains`` is evaluated at their centroids. First order in the sub-cell size;
a warning says so.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

__all__ = [
    "cell_fractions",
    "box_fractions",
    "segment_length",
    "CellMesh",
    "rect_boxes",
    "union_boxes",
    "closing_boxes",
    "outline_segments",
    "tri_box_area",
]


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


def rect_boxes(region, tol: float = 1e-12) -> np.ndarray | None:
    """Disjoint boxes covering a region built from axis-aligned rectangles.

    Args:
        region: A 2-D ``jno.shape``, or an ``(n, 4)`` array of possibly overlapping boxes read as their union.
        tol: Coordinates closer than this are one grid line.

    Returns:
        ``(k, 4)`` boxes ``[x0, y0, x1, y1]`` whose union is exactly ``region``, or ``None`` when ``region`` has a
        leaf that is not an axis-aligned rectangle.
    """
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


def union_boxes(boxes: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Disjoint boxes covering the union of ``boxes``.

    Args:
        boxes: ``(n, 4)`` boxes ``[x0, y0, x1, y1]``, possibly overlapping.
        tol: Coordinates closer than this are one grid line.

    Returns:
        ``(k, 4)`` disjoint boxes with the same union.
    """
    b = np.asarray(boxes, float).reshape(-1, 4)
    return _boxes(*_grid(b, tol)) if len(b) else b


def closing_boxes(boxes: np.ndarray, c: float) -> np.ndarray:
    """Morphological closing (dilate, then erode) of a union of boxes by the square ``[-c, c]^2``.

    The axis-aligned case of shapely's ``buffer(c, join_style="mitre").buffer(-c, join_style="mitre")``, and exact:
    dilating a union of boxes is the union of the dilated boxes, and an erosion is the complement of the dilated
    complement.

    Args:
        boxes: ``(n, 4)`` boxes ``[x0, y0, x1, y1]``.
        c: Half-width of the structuring square; slits narrower than ``2c`` close.

    Returns:
        ``(k, 4)`` disjoint boxes of the closed region.
    """
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


def outline_segments(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outline of the union of boxes as axis-aligned pieces: the grid edges with the region on one side only.

    Args:
        boxes: ``(n, 4)`` boxes ``[x0, y0, x1, y1]``.

    Returns:
        ``(vertical, horizontal)``: ``(p, 3)`` rows ``[x, y0, y1]`` and ``(q, 3)`` rows ``[y, x0, x1]``.
    """
    xs, ys, occ = _grid(boxes)
    P = np.zeros((occ.shape[0] + 2, occ.shape[1] + 2), bool)
    P[1:-1, 1:-1] = occ
    iv, jv = np.nonzero(P[1:, 1:-1] != P[:-1, 1:-1])
    ih, jh = np.nonzero(P[1:-1, 1:] != P[1:-1, :-1])
    return np.c_[xs[iv], ys[jv], ys[jv + 1]], np.c_[ys[jh], xs[ih], xs[ih + 1]]


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
    """Overlap areas of triangles with boxes, pair by pair, in closed form.

    Args:
        T: ``(n, 3, 2)`` triangle vertices.
        B: ``(n, 4)`` boxes ``[x0, y0, x1, y1]``.
        xp: ``numpy`` or ``jax.numpy``; with JAX the areas are differentiable in the vertices and box corners.

    Returns:
        ``(n,)`` overlap areas.
    """
    P = T
    for axis, lo, hi in ((0, B[:, 0], B[:, 2]), (1, B[:, 1], B[:, 3])):
        P = _insert_crossings(_insert_crossings(P, axis, lo, xp), axis, hi, xp)
        c = xp.clip(P[..., axis], lo[:, None], hi[:, None])
        P = xp.stack([c, P[..., 1]], -1) if axis == 0 else xp.stack([P[..., 0], c], -1)
    x, y = P[..., 0], P[..., 1]
    return 0.5 * xp.abs(xp.sum(x * xp.roll(y, -1, axis=1) - xp.roll(x, -1, axis=1) * y, axis=1))


def segment_length(T, axis: int, c, lo, hi, *, edges: bool = False, xp=np):
    """Length of axis-aligned segments inside triangles, pair by pair.

    Segment ``k`` lies on the line ``coord[axis] = c[k]`` and spans ``[lo[k], hi[k]]`` in the other coordinate;
    the result is the length of its part inside triangle ``T[k]``. Used to measure an outline per cell, e.g. the
    vertical and horizontal faces of a staircase for an anisotropic homogenisation.

    Args:
        T: ``(n, 3, 2)`` triangle vertices.
        axis: 0 for vertical segments (``x = c``), 1 for horizontal ones (``y = c``).
        c: ``(n,)`` position of each segment's line.
        lo: ``(n,)`` lower end of each segment along the other coordinate.
        hi: ``(n,)`` upper end.
        edges: Also count a segment lying on a triangle edge whose vertices miss the line by round-off (within
            1e-15). An edge exactly on the line is found through its vertices either way.
        xp: ``numpy`` or ``jax.numpy``; with JAX the lengths run under ``jit`` / ``vmap`` and are differentiable
            in the vertices and segment ends.

    Returns:
        ``(n,)`` lengths.
    """
    o = 1 - axis
    A, B = T, xp.roll(T, -1, axis=1)
    p, q, c2 = A[..., axis], B[..., axis], c[:, None]
    den = q - p
    ok = xp.abs(den) > 0
    t = xp.clip(xp.where(ok, (c2 - p) / xp.where(ok, den, 1.0), 0.0), 0.0, 1.0)
    hit = ((p - c2) * (q - c2) <= 0) & ok
    s = A[..., o] + t * (B[..., o] - A[..., o])
    s_lo = xp.where(hit, s, xp.inf).min(1)
    s_hi = xp.where(hit, s, -xp.inf).max(1)
    if edges:
        on = (xp.abs(p - c2) < 1e-15) & (xp.abs(q - c2) < 1e-15)
        s_lo = xp.minimum(s_lo, xp.where(on, xp.minimum(A[..., o], B[..., o]), xp.inf).min(1))
        s_hi = xp.maximum(s_hi, xp.where(on, xp.maximum(A[..., o], B[..., o]), -xp.inf).max(1))
    return xp.maximum(xp.minimum(s_hi, hi) - xp.maximum(s_lo, lo), 0.0)


def box_fractions(T, boxes, *, xp=np):
    """Area fraction of each triangle inside a union of disjoint boxes, in closed form.

    The traceable counterpart of :func:`cell_fractions` for a mesh that moves: no search structure, fixed array
    sizes, so it runs under ``jit`` and ``vmap`` with the vertices as inputs. The boxes are either shared by all
    triangles or given per triangle, e.g. a few candidates picked from boxes sorted along x. Padding boxes of zero
    width contribute nothing.

    Args:
        T: ``(n, 3, 2)`` triangle vertices.
        boxes: ``(k, 4)`` boxes ``[x0, y0, x1, y1]`` for every triangle, or ``(n, k, 4)`` per triangle. Must not
            overlap each other.
        xp: ``numpy`` or ``jax.numpy``.

    Returns:
        ``(n,)`` fractions in [0, 1], snapped to exactly 0 or 1 within 1e-12.
    """
    n = T.shape[0]
    if boxes.ndim == 2:
        boxes = xp.broadcast_to(boxes[None], (n,) + boxes.shape)
    k = boxes.shape[1]
    area = tri_box_area(xp.repeat(T, k, axis=0), boxes.reshape(-1, 4), xp).reshape(n, k).sum(1)
    det = (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
    frac = area / (0.5 * xp.abs(det))
    return xp.where(frac < 1e-12, 0.0, xp.where(frac > 1 - 1e-12, 1.0, frac))


class _Locator:
    """Bounding-box candidate search over a fixed triangle set (sorted sweep in x, filter in y)."""

    def __init__(self, T):
        self.T = T
        self.lo, self.hi = T.min(1), T.max(1)
        self.order = np.argsort(self.lo[:, 0])
        self.lo_sorted = self.lo[self.order, 0]
        self.maxw = float((self.hi[:, 0] - self.lo[:, 0]).max()) if len(T) else 0.0

    def pairs(self, boxes, eps=0.0):
        """All (cell, box) pairs whose bounding boxes overlap (closed with eps). The sweep runs over the cells
        that overlap the bounding box of ALL boxes, so large far-field cells do not widen every search window."""
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if not len(boxes):
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        X0, Y0 = boxes[:, 0].min() - eps, boxes[:, 1].min() - eps
        X1, Y1 = boxes[:, 2].max() + eps, boxes[:, 3].max() + eps
        sub = np.nonzero((self.hi[:, 0] >= X0) & (self.lo[:, 0] <= X1) & (self.hi[:, 1] >= Y0) & (self.lo[:, 1] <= Y1))[0]
        lo, hi = self.lo[sub], self.hi[sub]
        order = np.argsort(lo[:, 0])
        lo_sorted = lo[order, 0]
        maxw = float((hi[:, 0] - lo[:, 0]).max()) if len(sub) else 0.0
        cs, bs = [], []
        for k, (x0, y0, x1, y1) in enumerate(boxes):
            i0 = np.searchsorted(lo_sorted, x0 - eps - maxw, "left")
            i1 = np.searchsorted(lo_sorted, x1 + eps, "right")
            c = order[i0:i1]
            c = c[(hi[c, 0] >= x0 - eps) & (lo[c, 0] <= x1 + eps) & (hi[c, 1] >= y0 - eps) & (lo[c, 1] <= y1 + eps)]
            cs.append(sub[c])
            bs.append(np.full(len(c), k))
        return np.concatenate(cs), np.concatenate(bs)


class CellMesh:
    """A triangle mesh prepared once (vertices, areas, search index) for repeated :func:`cell_fractions` calls.

    Pass it instead of the domain when rasterising many regions on the same mesh.

    Args:
        mesh: A meshed 2-D ``jno.domain`` or ``(points (N, 2), triangles (C, 3))``.
    """

    def __init__(self, mesh):
        pts, tris = _mesh_arrays(mesh)
        self.T = pts[tris]
        T = self.T
        self.area = 0.5 * np.abs(
            (T[:, 1, 0] - T[:, 0, 0]) * (T[:, 2, 1] - T[:, 0, 1]) - (T[:, 2, 0] - T[:, 0, 0]) * (T[:, 1, 1] - T[:, 0, 1])
        )
        self.loc = _Locator(T)


def cell_fractions(
    mesh: Any,
    region: Any,
    *,
    close: float = 0.0,
    normals: bool = False,
    exclude_x: float | None = None,
    levels: int = 4,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Area fraction of every triangle of ``mesh`` inside ``region``.

    Args:
        mesh: A meshed 2-D ``jno.domain``, ``(points (N, 2), triangles (C, 3))``, or a :class:`CellMesh` prepared
            once for many calls on the same mesh.
        region: A 2-D ``jno.shape`` (exact when built from ``shape.rect`` with ``|``, ``-``, ``&``, ``regions``),
            or an ``(n, 4)`` array of boxes ``[x0, y0, x1, y1]`` read as their union.
        close: Closing radius applied to the region first (exact path only).
        normals: Also return ``nx2``: per cell, the share of the region's outline length inside it that is
            vertical (normal along x), the length-weighted ``<n_x^2>``; 0.5 where no outline passes.
        exclude_x: Outline pieces on the vertical line ``x = exclude_x`` (a symmetry axis) are not interfaces and
            do not count for ``normals``.
        levels: Fallback only: each triangle is split into ``4**levels`` sub-triangles.

    Returns:
        ``frac`` ``(C,)`` in [0, 1] (snapped to exactly 0 or 1 within 1e-12), or ``(frac, nx2)`` with
        ``normals=True``.

    Raises:
        NotImplementedError: ``close`` or ``normals`` with a region that is not built from ``shape.rect``.
    """
    cm = mesh if isinstance(mesh, CellMesh) else CellMesh(mesh)
    T, area, loc = cm.T, cm.area, cm.loc
    boxes = rect_boxes(region)
    if boxes is None:
        if close or normals:
            raise NotImplementedError(
                "jno.geometry.cell_fractions: close= and normals= need a region built from shape.rect."
            )
        warnings.warn(
            "jno.geometry.cell_fractions: the region is not a union of axis-aligned rectangles; using sub-cell quadrature "
            f"(4**{levels} points per cell), first-order accurate.",
            stacklevel=2,
        )
        return _fractions_quadrature(T, region, levels)
    if close:
        boxes = closing_boxes(boxes, close)
    frac = np.zeros(len(T))
    if len(boxes):
        c, b = loc.pairs(boxes)
        bb = boxes[b]
        full = (
            (loc.lo[c, 0] >= bb[:, 0])
            & (loc.hi[c, 0] <= bb[:, 2])
            & (loc.lo[c, 1] >= bb[:, 1])
            & (loc.hi[c, 1] <= bb[:, 3])
        )  # cell inside the box: no clipping
        np.add.at(frac, c[full], area[c[full]])
        cut = ~full
        np.add.at(frac, c[cut], tri_box_area(T[c[cut]], bb[cut]))
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
        np.add.at(S, c, segment_length(T[c], axis, segs[k, 0], segs[k, 1], segs[k, 2], edges=True))
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
