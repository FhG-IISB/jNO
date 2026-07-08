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


def _emit_node(node, occ, split_full=False):
    """Recursively build the OCC entities for a plan node; return a list of dimtags.

    ``split_full`` builds a full (2pi) revolve as two fused half-sweeps -- used only as a retry
    when the single-sweep periodic surface cannot be meshed (see :func:`build`).
    """
    kind = node[0]
    if kind == "leaf":
        return [node[1].build(occ)]
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
    raise ValueError(f"unknown node kind {kind!r}")


def _has_full_revolve(node) -> bool:
    kind = node[0]
    if kind == "revolve":
        return abs(node[4] - 2.0 * math.pi) < 1e-9 or _has_full_revolve(node[1]._node)
    if kind in ("cut", "fuse", "inter"):
        return _has_full_revolve(node[1]._node) or _has_full_revolve(node[2]._node)
    if kind in ("extrude", "translate", "rotate"):
        return _has_full_revolve(node[1]._node)
    return False


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


def _build_once(shape, split_full):
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
        ds = _apply_size_fields(dim, leaves, labels)
        gmsh.model.mesh.generate(dim)
        mesh = _to_meshio(dim, labels)
        return mesh, dim, ds
    finally:
        gmsh.model.remove()
        if started:
            gmsh.finalize()


def build(shape):
    """Mesh ``shape`` -> ``(meshio.Mesh, dim, ds)``.

    A single-sweep full (2pi) revolve of a *detached* profile makes a periodic surface gmsh
    cannot mesh, while an axis-touching profile (a cone) meshes fine that way -- so we try the
    single sweep first and only fall back to the two-halves construction if meshing fails.
    """
    try:
        return _build_once(shape, split_full=False)
    except Exception as exc:  # noqa: BLE001 - narrow retry on the periodic-surface mesher failure
        if "periodic" in str(exc).lower() and _has_full_revolve(shape._node):
            return _build_once(shape, split_full=True)
        raise
