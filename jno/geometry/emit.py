"""Realize a :class:`~jno.geometry.shape.Shape` into a ``meshio.Mesh`` via gmsh-OCC.

Pipeline: build the OCC entities from the plan -> synchronize -> classify the boundary
(:mod:`jno.geometry.naming`) -> per-shape mesh-size fields -> generate -> assemble a
``meshio.Mesh`` in memory with the ``cell_sets`` contract ``jno.domain`` consumes
(block 0 = volume cells, block 1 = boundary facets; ``interior`` = all of block 0, each
auto-name + ``boundary`` index into block 1). Returns ``(mesh, dim, ds)`` -- the same
triple ``jno.domain``'s callable path expects.

Owns its gmsh session: initializes/finalizes only if it opened it, and never leaks the
model it added.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

from .naming import classify_boundary

_MODEL_SEQ = itertools.count()

# gmsh element type ids
_TRI = 2
_TET = 4
_LINE = 1
_POINT = 15  # 1-node point element (the boundary block of a 1-D domain)


def _emit_node(node, occ, split_full=False):
    """Recursively build the OCC entities for a plan node; return a list of dimtags.

    ``split_full`` builds a full (2pi) revolve as two fused half-sweeps -- used only as a retry
    when the single-sweep periodic surface cannot be meshed (see :func:`build`).
    """
    kind = node[0]
    if kind == "leaf":
        return [node[1].build(occ)]
    if kind == "regions":
        pieces: List[tuple] = []
        for _name, sub in node[1]:
            pieces.extend(_emit_node(sub._node, occ, split_full))
        # Fragment every piece against the others: shared interfaces become conforming
        # (element edges align), and each material region survives as its own volume entity.
        out, _ = occ.fragment(pieces[:1], pieces[1:])
        return out
    if kind in ("cut", "fuse", "inter"):
        a = _emit_node(node[1]._node, occ, split_full)
        b = _emit_node(node[2]._node, occ, split_full)
        op = {"cut": occ.cut, "fuse": occ.fuse, "inter": occ.intersect}[kind]
        out, _ = op(a, b)
        return out
    if kind == "extrude":
        base = _emit_node(node[1]._node, occ, split_full)
        ext = occ.extrude(base, 0.0, 0.0, node[2])
        return [dt for dt in ext if dt[0] == 3]
    if kind == "revolve":
        base = _emit_node(node[1]._node, occ, split_full)
        ap, ad, ang = node[2], node[3], node[4]
        if split_full and abs(ang - 2.0 * math.pi) < 1e-9:
            # A detached full solid of revolution makes a periodic surface gmsh cannot mesh;
            # build it as two half-sweeps fused (each non-periodic, shared seam removed).
            other = occ.copy(base)
            h1 = [dt for dt in occ.revolve(base, *ap, *ad, math.pi) if dt[0] == 3]
            h2 = [dt for dt in occ.revolve(other, *ap, *ad, -math.pi) if dt[0] == 3]
            out, _ = occ.fuse(h1, h2)
            return out
        rev = occ.revolve(base, *ap, *ad, ang)
        return [dt for dt in rev if dt[0] == 3]
    if kind == "translate":
        ent = _emit_node(node[1]._node, occ, split_full)
        occ.translate(ent, *node[2])
        return ent
    if kind == "rotate":
        ent = _emit_node(node[1]._node, occ, split_full)
        occ.rotate(ent, *node[2], *node[3], node[4])
        return ent
    if kind == "sweep":
        profile = _emit_node(node[1]._node, occ, split_full)  # [(2, tag)]
        wire = node[2]._wire(occ)
        pipe = occ.addPipe(profile, wire)
        return [dt for dt in pipe if dt[0] == 3]
    if kind == "fillet":
        import gmsh

        vols = _emit_node(node[1]._node, occ, split_full)
        radius, where = node[2], node[3]
        occ.synchronize()  # incremental sync so we can query the solid's edges
        faces = gmsh.model.getBoundary(vols, combined=False, oriented=False, recursive=False)
        edges = gmsh.model.getBoundary(faces, combined=False, oriented=False, recursive=False)
        etags = sorted({abs(t) for d, t in edges if d == 1})
        if where is not None:
            etags = [t for t in etags if where(*occ.getCenterOfMass(1, t))]
        if not etags:
            return vols
        out = occ.fillet([t for d, t in vols if d == 3], etags, [radius])
        return out
    raise ValueError(f"unknown node kind {kind!r}")


def _has_full_revolve(node) -> bool:
    kind = node[0]
    if kind == "revolve":
        return abs(node[4] - 2.0 * math.pi) < 1e-9 or _has_full_revolve(node[1]._node)
    if kind in ("cut", "fuse", "inter"):
        return _has_full_revolve(node[1]._node) or _has_full_revolve(node[2]._node)
    if kind in ("extrude", "translate", "rotate", "fillet", "sweep"):
        return _has_full_revolve(node[1]._node)
    return False


def _plan_sizes(shape):
    """Every ``size`` attached anywhere in the plan (dedup by identity)."""
    seen, out = set(), []

    def walk(sh):
        s = sh._size
        if s is not None and id(s) not in seen:
            seen.add(id(s))
            out.append(s)
        node = sh._node
        kind = node[0]
        if kind in ("cut", "fuse", "inter"):
            walk(node[1])
            walk(node[2])
        elif kind in ("extrude", "revolve", "translate", "rotate", "fillet", "sweep"):
            walk(node[1])
        elif kind == "regions":
            for _name, sub in node[1]:
                walk(sub)

    walk(shape)
    return out


def _apply_size_fields(dim: int, leaves, labels: Dict[int, Tuple[int, str]], shape) -> Optional[float]:
    """Turn per-shape ``size`` into gmsh mesh-size controls. Returns a representative ``ds``.

    Three kinds compose: a **scalar on a primitive** -> a Distance+Threshold field that refines the
    band around that shape's boundary; a **callable ``f(x,y,z)``** anywhere -> a ``setSizeCallback``
    that refines by position (the general 'denser here' knob); a **scalar on the whole shape**
    (``(a-b).sized(0.05)``) -> a global size cap. All combine via ``min``.
    """
    import gmsh

    field = gmsh.model.mesh.field
    sized = [(key, s) for _prim, s, key in leaves if isinstance(s, (int, float))]
    thresholds: List[int] = []
    if sized:
        background = max(s for _k, s in sized)
        key_prop = {1: "PointsList", 2: "CurvesList", 3: "SurfacesList"}[dim]
        for key, s in sized:
            ents = [float(tag) for tag, (k, _n) in labels.items() if k == key]
            if not ents:
                continue
            dist = field.add("Distance")
            field.setNumbers(dist, key_prop, ents)
            th = field.add("Threshold")
            field.setNumber(th, "InField", dist)
            field.setNumber(th, "SizeMin", float(s))
            field.setNumber(th, "SizeMax", float(background))
            field.setNumber(th, "DistMin", float(s))
            field.setNumber(th, "DistMax", float(10.0 * s))
            thresholds.append(th)
        if thresholds:
            mn = field.add("Min")
            field.setNumbers(mn, "FieldsList", [float(t) for t in thresholds])
            field.setAsBackgroundMesh(mn)

    callables = [s for s in _plan_sizes(shape) if callable(s)]
    if callables:
        fns = tuple(callables)

        def _size_cb(cdim, ctag, x, y, z, lc, _fns=fns):
            return float(min([lc] + [float(f(x, y, z)) for f in _fns]))

        gmsh.model.mesh.setSizeCallback(_size_cb)

    top = shape._size if isinstance(shape._size, (int, float)) else None
    if top is not None:
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(top))

    if thresholds or callables or top is not None:
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    scalars = [s for _k, s in sized] + ([top] if top is not None else [])
    return float(min(scalars)) if scalars else None


def _to_meshio(dim: int, labels: Dict[int, Tuple[int, str]], region_items=None):
    """Assemble the generated gmsh mesh into a ``meshio.Mesh`` with named ``cell_sets``.

    ``region_items`` (``((name, sub_shape), ...)``, when the plan is a :meth:`Shape.regions`
    multi-material domain) adds one volume ``cell_set`` per region — each cell assigned to the
    first region whose shape contains its centroid — and drops internal interface facets from the
    boundary block (a facet shared by two volumes is not a domain boundary).
    """
    import gmsh
    import meshio
    import numpy as np

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(node_tags)}

    if dim == 1:
        vtype, npv, vblock = _LINE, 2, "line"
        btype, npb, bblock = _POINT, 1, "vertex"  # boundary of a 1-D domain = its two endpoints
    else:
        vtype, npv, vblock = (_TET, 4, "tetra") if dim == 3 else (_TRI, 3, "triangle")
        btype, npb, bblock = (_TRI, 3, "triangle") if dim == 3 else (_LINE, 2, "line")

    _vtags, vnodes = gmsh.model.mesh.getElementsByType(vtype)
    vcells = np.asarray([index[int(t)] for t in vnodes], dtype=np.int64).reshape(-1, npv)

    bdim = dim - 1
    facet_rows: List[List[int]] = []
    by_name: Dict[str, List[np.ndarray]] = {}
    for _edim, etag in gmsh.model.getEntities(bdim):
        if region_items is not None and len(gmsh.model.getAdjacencies(bdim, etag)[0]) >= 2:
            continue  # facet shared by two volumes -> internal material interface, not boundary
        label = labels.get(etag)
        # A 1-D polyline's intermediate junctions are interior points (unnamed) -- they must not
        # land in the boundary block. Higher-dim unnamed facets stay boundary (just unnamed).
        if dim == 1 and label is None:
            continue
        etypes, _etags, enodes = gmsh.model.mesh.getElements(bdim, etag)
        for et, en in zip(etypes, enodes):
            if et != btype:
                continue
            rows = np.asarray(en, dtype=np.int64).reshape(-1, npb)
            start = len(facet_rows)
            facet_rows.extend([index[int(t)] for t in row] for row in rows)
            if label is not None:
                rng = np.arange(start, len(facet_rows), dtype=np.int64)
                by_name.setdefault(label[1], []).append(rng)

    facets = np.asarray(facet_rows, dtype=np.int64).reshape(-1, npb) if facet_rows else np.zeros((0, npb), dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    cells = [(vblock, vcells), (bblock, facets)]
    cell_sets: Dict[str, list] = {"interior": [np.arange(len(vcells), dtype=np.int64), empty.copy()]}
    for name, chunks in by_name.items():
        cell_sets[name] = [empty.copy(), np.concatenate(chunks) if chunks else empty.copy()]
    cell_sets["boundary"] = [empty.copy(), np.arange(len(facets), dtype=np.int64)]

    if region_items and len(vcells):
        centroids = coords[vcells].mean(axis=1)  # (M, 3) cell centroids
        assigned = np.full(len(vcells), -1, dtype=np.int64)
        for ri, (name, sub) in enumerate(region_items):
            take = np.asarray(sub.contains(centroids), dtype=bool) & (assigned < 0)  # first match wins
            assigned[take] = ri
        for ri, (name, _sub) in enumerate(region_items):
            cell_sets[name] = [np.where(assigned == ri)[0].astype(np.int64), empty.copy()]

    return meshio.Mesh(points=coords, cells=cells, cell_sets=cell_sets)


def _apply_periodic(dim, labels, periodic):
    """gmsh ``setPeriodic`` for each ``(master_name, slave_name)`` boundary-face pair: mesh the slave face
    as a *translated copy* of the master so opposite boundaries mesh identically (conforming) — required
    for edge-element (Nédélec) periodic ties, whose per-edge DOFs must line up one-to-one across the cell.
    The translation is read from the face bounding-box centroids; a pair whose faces aren't both present is
    skipped (nothing to tie)."""
    import gmsh
    import numpy as np

    bdim = dim - 1
    by_name: dict = {}
    for etag, lab in labels.items():
        if lab is not None:
            by_name.setdefault(lab[1], []).append(int(etag))

    def _centroid(tags):
        bb = np.array([gmsh.model.getBoundingBox(bdim, t) for t in tags])  # (n, 6): (xlo,ylo,zlo,xhi,yhi,zhi)
        return 0.5 * (bb[:, :3].min(axis=0) + bb[:, 3:].max(axis=0))

    for master, slave in periodic:
        m_tags, s_tags = by_name.get(master, []), by_name.get(slave, [])
        if not m_tags or not s_tags:
            continue
        t = _centroid(s_tags) - _centroid(m_tags)  # translation master -> slave
        affine = [1, 0, 0, t[0], 0, 1, 0, t[1], 0, 0, 1, t[2], 0, 0, 0, 1]  # row-major 4×4
        gmsh.model.mesh.setPeriodic(bdim, s_tags, m_tags, affine)


def _build_once(shape, split_full, periodic=None):
    import gmsh

    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"jno_shape_{next(_MODEL_SEQ)}")
    try:
        occ = gmsh.model.occ
        _emit_node(shape._node, occ, split_full)
        occ.synchronize()
        dim = shape.dim
        leaves = shape.leaves()
        labels = classify_boundary(dim, shape)
        ds = _apply_size_fields(dim, leaves, labels, shape)
        if periodic:  # make named opposite faces conform (edge-element periodic ties need it)
            _apply_periodic(dim, labels, periodic)
        gmsh.model.mesh.generate(dim)
        region_items = shape._node[1] if shape._node[0] == "regions" else None
        mesh = _to_meshio(dim, labels, region_items)
        return mesh, dim, ds
    finally:
        gmsh.model.remove()
        if started:
            gmsh.finalize()


def build(shape, periodic=None):
    """Mesh ``shape`` -> ``(meshio.Mesh, dim, ds)``.

    ``periodic`` is an optional list of ``(master_name, slave_name)`` boundary-face pairs meshed conforming
    (via :func:`_apply_periodic`) so opposite faces line up — needed for Nédélec edge periodic ties.

    A single-sweep full (2pi) revolve of a *detached* profile makes a periodic surface gmsh cannot mesh,
    while an axis-touching profile (a cone) meshes fine that way -- so we try the single sweep first and
    only fall back to the two-halves construction if meshing fails.
    """
    try:
        return _build_once(shape, split_full=False, periodic=periodic)
    except Exception as exc:  # noqa: BLE001 - narrow retry on the periodic-surface mesher failure
        if "periodic" in str(exc).lower() and _has_full_revolve(shape._node):
            return _build_once(shape, split_full=True, periodic=periodic)
        raise
