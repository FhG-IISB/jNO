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
from typing import Dict, List, Optional, Tuple

from .naming import classify_boundary

_MODEL_SEQ = itertools.count()

# gmsh element type ids
_TRI = 2
_TET = 4
_LINE = 1


def _emit_node(node, occ):
    """Recursively build the OCC entities for a plan node; return a list of dimtags."""
    kind = node[0]
    if kind == "leaf":
        return [node[1].build(occ)]
    if kind == "cut":
        a = _emit_node(node[1]._node, occ)
        b = _emit_node(node[2]._node, occ)
        out, _ = occ.cut(a, b)
        return out
    if kind == "fuse":
        a = _emit_node(node[1]._node, occ)
        b = _emit_node(node[2]._node, occ)
        out, _ = occ.fuse(a, b)
        return out
    if kind == "inter":
        a = _emit_node(node[1]._node, occ)
        b = _emit_node(node[2]._node, occ)
        out, _ = occ.intersect(a, b)
        return out
    if kind == "extrude":
        base = _emit_node(node[1]._node, occ)
        ext = occ.extrude(base, 0.0, 0.0, node[2])
        return [dt for dt in ext if dt[0] == 3]
    raise ValueError(f"unknown node kind {kind!r}")


def _apply_size_fields(dim: int, leaves, labels: Dict[int, Tuple[int, str]]) -> Optional[float]:
    """Per-shape size -> gmsh Distance+Threshold fields, combined via Min. Returns ``ds``.

    A leaf's ``size`` sizes the mesh near the boundary it owns (kept region *or* the arc a
    subtracted disk carves). Callable sizes are deferred; scalars only for now.
    """
    import gmsh

    sized = [(key, s) for _prim, s, key in leaves if isinstance(s, (int, float))]
    if not sized:
        return None
    background = max(s for _k, s in sized)
    field = gmsh.model.mesh.field
    key_prop = "CurvesList" if dim == 2 else "SurfacesList"
    thresholds: List[int] = []
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
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    return float(min(s for _k, s in sized))


def _to_meshio(dim: int, labels: Dict[int, Tuple[int, str]]):
    """Assemble the generated gmsh mesh into a ``meshio.Mesh`` with named ``cell_sets``."""
    import gmsh
    import meshio
    import numpy as np

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(node_tags)}

    vtype, npv, vblock = (_TET, 4, "tetra") if dim == 3 else (_TRI, 3, "triangle")
    btype, npb, bblock = (_TRI, 3, "triangle") if dim == 3 else (_LINE, 2, "line")

    _vtags, vnodes = gmsh.model.mesh.getElementsByType(vtype)
    vcells = np.asarray([index[int(t)] for t in vnodes], dtype=np.int64).reshape(-1, npv)

    bdim = dim - 1
    facet_rows: List[List[int]] = []
    by_name: Dict[str, List[np.ndarray]] = {}
    for _edim, etag in gmsh.model.getEntities(bdim):
        label = labels.get(etag)
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

    return meshio.Mesh(points=coords, cells=cells, cell_sets=cell_sets)


def build(shape):
    """Mesh ``shape`` -> ``(meshio.Mesh, dim, ds)``."""
    import gmsh

    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add(f"jno_shape_{next(_MODEL_SEQ)}")
    try:
        occ = gmsh.model.occ
        _emit_node(shape._node, occ)
        occ.synchronize()
        dim = shape.dim
        dz = shape._node[2] if shape._node[0] == "extrude" else None
        leaves = shape.leaves()
        labels = classify_boundary(dim, leaves, dz)
        ds = _apply_size_fields(dim, leaves, labels)
        gmsh.model.mesh.generate(dim)
        mesh = _to_meshio(dim, labels)
        return mesh, dim, ds
    finally:
        gmsh.model.remove()
        if started:
            gmsh.finalize()
