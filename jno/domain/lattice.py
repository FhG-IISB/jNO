"""Rectilinear lattices for ``.structured()`` plans built from several axis-aligned rects/boxes.

A plan made only of axis-aligned rectangles or boxes (combined by ``+`` regions and ``|``/``-``/``&``) is
constant on every cell of the grid spanned by its primitives' faces -- the *breakpoint* grid. So a
lattice whose planes include every breakpoint resolves every material interface exactly, and the only
choice left is the spacing inside each slab between two neighbouring breakpoints. A slab runs through
the whole cell along the other axes, so it takes the finest size of anything it crosses: a thin film
refines its own z-slab, not the slabs above and below it. Cells are voxels, or each voxel's six Kuhn
tets (two triangles in 2-D) sharing one diagonal, so the mesh is conforming and opposite faces match
node for node.
"""

from __future__ import annotations

import numpy as np

from ..utils.sets import unique_values

_AXIS_NAMES = {2: (("left", "right"), ("bottom", "top")), 3: (("left", "right"), ("front", "back"), ("bottom", "top"))}
# facets of each cell kind, as local vertex indices
_FACETS = {
    "tetra": ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)),
    "hexahedron": ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    "triangle": ((1, 2), (2, 0), (0, 1)),
    "quad": ((0, 1), (1, 2), (2, 3), (3, 0)),
}
_FACET_BLOCK = {"tetra": "triangle", "hexahedron": "quad", "triangle": "line", "quad": "line"}


def _bounds(prim):
    if prim.dim == 3:
        return np.array([[prim.x0, prim.x1], [prim.y0, prim.y1], [prim.z0, prim.z1]], dtype=float)
    return np.array([[prim.x0, prim.x1], [prim.y0, prim.y1]], dtype=float)


def check_plan(plan):
    """Refuse, by name, a plan a rectilinear lattice cannot mesh exactly."""
    from ..geometry.primitives import Box, Rect

    def walk(node):
        kind = node[0]
        if kind == "leaf":
            if not isinstance(node[1], (Rect, Box)):
                raise NotImplementedError(
                    f"shape.structured() needs a single axis-aligned rect/box, or rects/boxes combined by +, |, - "
                    f"and &; this plan has a {type(node[1]).__name__.lower()}. Mesh it unstructured (drop "
                    ".structured())."
                )
        elif kind in ("cut", "fuse", "inter"):
            walk(node[1]._node)
            walk(node[2]._node)
        elif kind == "regions":
            if len(node) > 2 and not node[2]:
                raise NotImplementedError(
                    "shape.structured() on regions(conforming=False): a lattice is conforming by construction, so "
                    "it cannot build the two non-matching sides of a mortar interface. Drop conforming=False, or "
                    "drop .structured()."
                )
            for _name, sub in node[1]:
                walk(sub._node)
        else:
            raise NotImplementedError(
                f"shape.structured() needs axis-aligned rects/boxes; a {kind!r} can leave them off the axes. "
                "Mesh it unstructured (drop .structured())."
            )

    walk(plan._node)


def _graded_nodes(cuts, h_slab, r):
    """Node coordinates along one axis with the spacing ``g(x) = min_j(h_j + (r - 1) dist(x, slab j))``:
    geometric growth at ratio ``r`` away from every slab, capped by each slab's own size. Each slab is
    split into ``ceil(int 1/g)`` cells placed at equal steps of ``int 1/g``, so the cuts stay nodes."""
    fin = np.isfinite(h_slab)
    lo_j, hi_j, h_j = cuts[:-1][fin], cuts[1:][fin], h_slab[fin]
    pieces = [cuts[:1]]
    for i in range(len(h_slab)):
        lo, hi = cuts[i], cuts[i + 1]
        if not np.isfinite(h_slab[i]):
            pieces.append(np.array([hi]))
            continue
        x = np.linspace(lo, hi, 2049)
        dist = np.maximum(0.0, np.maximum(lo_j[None, :] - x[:, None], x[:, None] - hi_j[None, :]))
        g = np.minimum(h_slab[i], (h_j[None, :] + (r - 1.0) * dist).min(axis=1))
        inv = 1.0 / g
        F = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(x))])
        n = max(1, int(np.ceil(F[-1] - 1e-9)))
        pieces.append(np.interp(np.arange(1, n + 1) * F[-1] / n, F, x))
        pieces[-1][-1] = hi
    return np.concatenate(pieces)


def lattice_axes(plan, items, default_h, grading=None):
    """The node coordinates along each axis: every primitive face is a plane, and each slab between
    two planes is split evenly at no more than the finest size of the region cells it crosses, or,
    with ``grading``, blended across slabs (:func:`_graded_nodes`).

    ``items`` are the ``(name, shape)`` regions in priority order. A region's size is the size of its
    own primitive containing the point, inherited from enclosing ``.size()``/``.sized()`` calls as the
    gmsh path does; ``default_h`` stands in where none is set. Returns ``(axes, owner)``, ``owner``
    being the region index of each breakpoint cell (``-1`` outside the plan)."""
    dim = int(plan.dim)
    leaves = plan.leaves()
    cuts = []
    for a in range(dim):
        v = unique_values(np.concatenate([_bounds(p)[a] for p, _s, _k in leaves]))
        span = float(v[-1] - v[0]) or 1.0
        cuts.append(v[np.concatenate([[True], np.diff(v) > 1e-9 * span])])
    mids = [0.5 * (c[1:] + c[:-1]) for c in cuts]
    C = np.stack(np.meshgrid(*mids, indexing="ij"), axis=-1).reshape(-1, dim)
    owner = np.full(len(C), -1, dtype=np.int64)
    size = np.full(len(C), np.inf)
    for r, (_name, sub) in enumerate(items):
        take = np.asarray(sub.contains(C), dtype=bool) & (owner < 0)
        owner[take] = r
        for prim, s, _k in sub.leaves(plan._size if sub is not plan else None):
            if callable(s):
                raise NotImplementedError(
                    "shape.structured() takes its spacing from each region's size=, and a graded size=<callable> "
                    "has no single spacing per slab. Give the regions numeric sizes, or drop .structured()."
                )
            h = float(s) if s is not None else float(default_h)
            b = _bounds(prim)
            inside = np.all((C >= b[:, 0] - 1e-12) & (C <= b[:, 1] + 1e-12), axis=1) & take
            size[inside] = np.minimum(size[inside], h)
    size[(owner >= 0) & ~np.isfinite(size)] = float(default_h)
    owner = owner.reshape(tuple(len(m) for m in mids))
    size = size.reshape(owner.shape)
    axes = []
    for a in range(dim):
        other = tuple(k for k in range(dim) if k != a)
        h_slab = np.where(owner >= 0, size, np.inf).min(axis=other)
        if grading is not None:
            axes.append(_graded_nodes(cuts[a], h_slab, float(grading)))
            continue
        pieces = [cuts[a][:1]]
        for i in range(len(mids[a])):
            lo, hi = cuts[a][i], cuts[a][i + 1]
            n = 1 if not np.isfinite(h_slab[i]) else max(1, int(np.ceil((hi - lo) / h_slab[i] - 1e-9)))
            pieces.append(np.linspace(lo, hi, n + 1)[1:])
        axes.append(np.concatenate(pieces))
    return axes, owner


def lattice_mesh(axes, items, cell):
    """The ``meshio.Mesh`` of the rectilinear lattice over ``axes``, keeping the cells whose centroid
    lies in a region (the first in priority order wins).

    ``cell_sets``: ``interior``; one volume set per named region; ``boundary`` (the outer facets only);
    the outer facets by their outward axis direction (``left/right``, ``front/back``, ``bottom/top``, as
    :class:`jno.shape.box` names its faces); and one facet set per material interface, ``"a|b"`` with
    the names sorted (plus ``"a|b.0"``, ``"a|b.1"`` ... when one pair meets on disjoint patches)."""
    import meshio

    from ..geometry.emit import _facet_components

    dim = len(axes)
    shape = tuple(len(x) for x in axes)
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
    stride = np.cumprod((1,) + shape[::-1][:-1])[::-1]  # C-order node index
    vox = np.stack(np.meshgrid(*[np.arange(n - 1) for n in shape], indexing="ij"), axis=-1).reshape(-1, dim)
    corner = np.stack(
        [((vox + np.array([(c >> (dim - 1 - a)) & 1 for a in range(dim)])) * stride).sum(1) for c in range(2**dim)], 1
    )
    if dim == 3:
        if cell == "hex":
            kind, cells = "hexahedron", corner[:, [0, 4, 6, 2, 1, 5, 7, 3]]
        else:  # Kuhn: six tets on the voxel's main diagonal 000 -> 111
            kuhn = np.asarray([(0, 4, 6, 7), (0, 4, 5, 7), (0, 2, 6, 7), (0, 2, 3, 7), (0, 1, 5, 7), (0, 1, 3, 7)])
            kind, cells = "tetra", corner[:, kuhn].reshape(-1, 4)
    else:  # corner c = (di, dj) as bits (c>>1, c&1)
        if cell == "quad":
            kind, cells = "quad", corner[:, [0, 2, 3, 1]]
        else:
            kind, cells = "triangle", corner[:, np.asarray([(0, 2, 3), (0, 3, 1)])].reshape(-1, 3)
    if kind in ("tetra", "triangle"):  # positive orientation, as the assembler's Jacobians expect
        V = P[cells]
        J = np.stack([V[:, k] - V[:, 0] for k in range(1, dim + 1)], axis=-1)
        neg = np.linalg.det(J) < 0
        cells[neg] = cells[neg][:, [1, 0] + list(range(2, dim + 1))]
    cent = P[cells].mean(axis=1)
    owner = np.full(len(cells), -1, dtype=np.int64)
    for r, (_name, sub) in enumerate(items):
        owner[np.asarray(sub.contains(cent), dtype=bool) & (owner < 0)] = r
    keep = owner >= 0
    cells, owner, cent = cells[keep], owner[keep], cent[keep]
    used = unique_values(cells)
    remap = np.full(len(P), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    P, cells = P[used], remap[cells]

    # facets: shared by two cells (interior / interface) or owned by one (outer boundary)
    loc = np.asarray(_FACETS[kind])
    F = cells[:, loc].reshape(-1, loc.shape[1])
    who = np.repeat(np.arange(len(cells)), len(loc))
    _u, inv, cnt = np.unique(np.sort(F, axis=1), axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    first = np.full(len(cnt), -1, dtype=np.int64)
    second = np.full(len(cnt), -1, dtype=np.int64)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(cnt)))
    first = order[starts]
    has2 = cnt == 2
    second[has2] = order[starts[has2] + 1]
    outer = first[cnt == 1]
    iface = np.flatnonzero(has2)
    iface = iface[owner[who[first[iface]]] != owner[who[second[iface]]]]

    rows = F[outer].copy()
    fc = P[rows].mean(axis=1)
    out_dir = fc - cent[who[outer]]
    flat = np.ptp(P[rows], axis=1) < 1e-12 * float(np.ptp(P, axis=0).max())
    ax = np.argmax(flat, axis=1)
    side = (out_dir[np.arange(len(rows)), ax] > 0).astype(int)
    # orient each outer facet outward (the convention of the gmsh path's boundary facets)
    if dim == 3:
        V = P[rows]
        n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
        flip = np.einsum("ij,ij->i", n, out_dir) < 0
        rows[flip] = rows[flip][:, ::-1]
    else:
        t = P[rows[:, 1]] - P[rows[:, 0]]
        flip = (t[:, 1] * out_dir[:, 0] - t[:, 0] * out_dir[:, 1]) < 0  # outward normal (ty, -tx)
        rows[flip] = rows[flip][:, ::-1]
    irows = F[first[iface]]
    facets = np.concatenate([rows, irows]).astype(np.int64)
    n_out = len(rows)

    empty = np.array([], dtype=np.int64)
    sets = {"interior": [np.arange(len(cells), dtype=np.int64), empty.copy()]}
    sets["boundary"] = [empty.copy(), np.arange(n_out, dtype=np.int64)]
    for a, names in enumerate(_AXIS_NAMES[dim]):
        for s, name in enumerate(names):
            sets[name] = [empty.copy(), np.flatnonzero((ax == a) & (side == s)).astype(np.int64)]
    names = [n for n, _s in items]
    pair_of = np.sort(np.stack([owner[who[first[iface]]], owner[who[second[iface]]]], 1), axis=1)
    for a, b in {tuple(p) for p in pair_of.tolist()}:
        if names[a] is None or names[b] is None:
            continue
        key = "|".join(sorted((names[a], names[b])))
        idx = n_out + np.flatnonzero((pair_of[:, 0] == a) & (pair_of[:, 1] == b))
        sets[key] = [empty.copy(), idx.astype(np.int64)]
        comps = _facet_components(idx, facets[idx])
        if len(comps) > 1:
            for k, comp in enumerate(comps):
                sets[f"{key}.{k}"] = [empty.copy(), np.asarray(comp, dtype=np.int64)]
    for r, name in enumerate(names):
        if name is not None:
            sets[name] = [np.flatnonzero(owner == r).astype(np.int64), empty.copy()]
    pts3 = P if dim == 3 else np.column_stack([P, np.zeros(len(P))])
    mesh = meshio.Mesh(points=pts3, cells=[(kind, cells.astype(np.int64)), (_FACET_BLOCK[kind], facets)], cell_sets=sets)
    ds = float(min(np.diff(x).min() for x in axes))
    return mesh, dim, ds
