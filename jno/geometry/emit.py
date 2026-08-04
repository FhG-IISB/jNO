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


def _facet_components(facet_idx, facet_nodes):
    """Group facets into connected components (share a node -> same component), returning a list of
    global-index arrays. A material interface spans many gmsh faces but is usually ONE connected
    patch; only genuinely disjoint patches (e.g. two separate inclusions) split into several."""
    from collections import defaultdict

    import numpy as np

    n = len(facet_idx)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    node_first: Dict[int, int] = {}
    for fi in range(n):
        for node in facet_nodes[fi]:
            node = int(node)
            if node in node_first:
                ri, rj = find(fi), find(node_first[node])
                if ri != rj:
                    parent[ri] = rj
            else:
                node_first[node] = fi
    groups = defaultdict(list)
    for fi in range(n):
        groups[find(fi)].append(fi)
    return [np.asarray(facet_idx)[g] for g in groups.values()]


def _to_meshio(dim: int, labels: Dict[int, Tuple[int, str]], region_items=None):
    """Assemble the generated gmsh mesh into a ``meshio.Mesh`` with named ``cell_sets``.

    ``region_items`` (``((name, sub_shape), ...)``, when the plan is a :meth:`Shape.regions`
    multi-material domain) adds one volume ``cell_set`` per region — each cell assigned to the first
    region whose shape contains its centroid — plus one facet ``cell_set`` per material **interface**,
    auto-named by the region pair it separates (``"a|b"`` = every facet between those two materials,
    however many gmsh faces that spans). Only *topologically disjoint* interfaces of the same pair
    (e.g. two separate inclusions) additionally split into connected components ``"a|b.0"`` / ``"a|b.1"``.
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

    # Which region each volume ENTITY belongs to (all its cells share one region -> classify one cell).
    vol_region: Dict[int, str] = {}
    if region_items is not None:
        for _d, vtag in gmsh.model.getEntities(dim):
            vt, _vt2, vn = gmsh.model.mesh.getElements(dim, vtag)
            for et, en in zip(vt, vn):
                if et != vtype:
                    continue
                nodes = [index[int(t)] for t in np.asarray(en[:npv], dtype=np.int64)]
                c = coords[nodes].mean(axis=0)[None, :]
                for name, sub in region_items:
                    if bool(np.asarray(sub.contains(c))[0]):
                        vol_region[int(vtag)] = name
                        break
                break

    bdim = dim - 1
    facet_rows: List[List[int]] = []
    by_name: Dict[str, List[np.ndarray]] = {}
    external_ranges: List[np.ndarray] = []
    iface_entities: Dict[str, List[np.ndarray]] = {}  # "a|b" -> per-entity facet index ranges
    for _edim, etag in gmsh.model.getEntities(bdim):
        adj = gmsh.model.getAdjacencies(bdim, etag)[0] if region_items is not None else ()
        interface_pair = None
        if len(adj) >= 2:
            regs = sorted({vol_region[int(v)] for v in adj if int(v) in vol_region})
            if len(regs) < 2:
                continue  # both sides are the same region -> not a material interface
            interface_pair = "|".join(regs)
        label = labels.get(etag)
        # A 1-D polyline's intermediate junctions are interior points (unnamed) -- they must not
        # land in the boundary block. Higher-dim unnamed facets stay boundary (just unnamed).
        if dim == 1 and label is None:
            continue
        etypes, _etags, enodes = gmsh.model.mesh.getElements(bdim, etag)
        start = len(facet_rows)
        for et, en in zip(etypes, enodes):
            if et != btype:
                continue
            rows = np.asarray(en, dtype=np.int64).reshape(-1, npb)
            facet_rows.extend([index[int(t)] for t in row] for row in rows)
        rng = np.arange(start, len(facet_rows), dtype=np.int64)
        if rng.size == 0:
            continue
        if interface_pair is not None:
            iface_entities.setdefault(interface_pair, []).append(rng)
        else:
            external_ranges.append(rng)
            if label is not None:
                by_name.setdefault(label[1], []).append(rng)

    facets = np.asarray(facet_rows, dtype=np.int64).reshape(-1, npb) if facet_rows else np.zeros((0, npb), dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    cells = [(vblock, vcells), (bblock, facets)]
    cell_sets: Dict[str, list] = {"interior": [np.arange(len(vcells), dtype=np.int64), empty.copy()]}
    for name, chunks in by_name.items():
        cell_sets[name] = [empty.copy(), np.concatenate(chunks) if chunks else empty.copy()]
    # "boundary" is the OUTER boundary only (internal interfaces are separate, named tags).
    ext = np.concatenate(external_ranges) if external_ranges else empty.copy()
    cell_sets["boundary"] = [empty.copy(), ext]

    for pair, ent_list in iface_entities.items():
        all_idx = np.concatenate(ent_list)  # every facet between this pair (across all its faces)
        cell_sets[pair] = [empty.copy(), all_idx]
        comps = _facet_components(all_idx, facets[all_idx])  # group by TOPOLOGY, not gmsh face
        if len(comps) > 1:  # genuinely disjoint interfaces of the SAME pair -> keep each separable
            for k, comp in enumerate(comps):
                cell_sets[f"{pair}.{k}"] = [empty.copy(), comp]

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


#: gmsh's 3-D mesher. 10 is HXT, a parallel Delaunay kernel; gmsh's own default is 1 (serial
#: Delaunay), which is what jNO used to get by not setting this at all. Measured on a unit cube at
#: mesh size 0.022: **3.91 s serial Delaunay vs 0.27 s HXT on 8 threads, 14.5x**, and the quality
#: does not pay for it -- HXT's WORST element was better (minSICN 0.343 vs 0.288) with mean quality
#: within 2% and no bad elements either way.
#:
#: HXT produces a different (slightly coarser) mesh for the same target size -- 28,068 nodes against
#: 32,773 on that cube -- so anything asserting an exact node count will see a change.
MESH_ALGORITHM_3D = 10

#: Threads for the 3-D mesher. Only HXT parallelises; serial algorithms ignore it.
#:
#: **1, not 8, because parallel HXT is NOT REPRODUCIBLE.** Measured: three builds of the same box at
#: 8 threads gave the same node COUNT but different node positions each time, while at 1 thread the
#: meshes were bit-identical across runs. A mesh that changes run to run means results that cannot
#: be reproduced, tests that pin anything about the mesh going flaky, and benchmark variance that
#: silently includes mesh variation.
#:
#: The speed that matters survives this: HXT at 1 thread is still 6.6x gmsh's serial Delaunay
#: (0.59 s vs 3.91 s at size 0.022); threading adds a further 2.2x (to 0.27 s) on a step that is now
#: only ~10% of the domain build. Pass ``threads=8`` explicitly to trade reproducibility for it.
MESH_THREADS = 1

#: gmsh's 2-D mesher, left at 6 (Frontal-Delaunay) DELIBERATELY. Algorithm 5 (Delaunay) is ~1.4x
#: faster per node but measurably worse: mean minSICN 0.952 against 0.999, worst element 0.665
#: against 0.874. Algorithm 6 produces near-perfect triangles, and 2-D meshing is not the
#: bottleneck the 3-D kernel was. Pass ``algorithm=5`` if speed matters more than conditioning.
MESH_ALGORITHM_2D = 6


def _build_once(shape, split_full, periodic=None, algorithm=None, threads=None):
    import gmsh

    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
    # Both kernels get jNO's default, then `algorithm` overrides the one for the shape's OWN
    # dimension. A 3-D shape still meshes its surfaces with the 2-D kernel first, so that one stays
    # on the quality default rather than being overridden by a 3-D algorithm number.
    gmsh.option.setNumber("Mesh.Algorithm", MESH_ALGORITHM_2D)
    gmsh.option.setNumber("Mesh.Algorithm3D", MESH_ALGORITHM_3D)
    if algorithm is not None:
        gmsh.option.setNumber("Mesh.Algorithm3D" if shape.dim == 3 else "Mesh.Algorithm", int(algorithm))
    n_threads = MESH_THREADS if threads is None else int(threads)
    gmsh.option.setNumber("General.NumThreads", n_threads)
    gmsh.option.setNumber("Mesh.MaxNumThreads3D", n_threads)
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


def build(shape, periodic=None, *, algorithm=None, threads=None):
    """Mesh ``shape`` -> ``(meshio.Mesh, dim, ds)``.

    ``algorithm`` selects gmsh's meshing kernel for the shape's own dimension -- there is no
    separate 2-D/3-D argument because ``shape.dim`` already says which applies. ``threads`` sets the
    thread count. ``None`` uses :data:`MESH_ALGORITHM_2D` / :data:`MESH_ALGORITHM_3D` /
    :data:`MESH_THREADS`, whose docstrings record the measurements behind each default.

    ``periodic`` is an optional list of ``(master_name, slave_name)`` boundary-face pairs meshed conforming
    (via :func:`_apply_periodic`) so opposite faces line up — needed for Nédélec edge periodic ties.

    A single-sweep full (2pi) revolve of a *detached* profile makes a periodic surface gmsh cannot mesh,
    while an axis-touching profile (a cone) meshes fine that way -- so we try the single sweep first and
    only fall back to the two-halves construction if meshing fails.
    """
    opts = dict(algorithm=algorithm, threads=threads)
    try:
        return _build_once(shape, split_full=False, periodic=periodic, **opts)
    except Exception as exc:  # noqa: BLE001 - narrow retry on the periodic-surface mesher failure
        if "periodic" in str(exc).lower() and _has_full_revolve(shape._node):
            return _build_once(shape, split_full=True, periodic=periodic, **opts)
        raise
