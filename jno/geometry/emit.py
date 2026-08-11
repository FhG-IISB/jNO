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
        # `conforming=False` skips it, so each piece is meshed on its own and two touching regions
        # get two coincident, NON-matching surfaces with duplicated nodes -- the configuration a
        # mortar tie glues. Measured on two boxes sharing a face at different mesh sizes: fragment
        # gives 44 interface nodes at 44 distinct coordinates (merged), no-fragment gives 56 at 52
        # (only the 4 corners coincide).
        if len(node) > 2 and not node[2]:
            return pieces
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


def _apply_region_sizes(dim: int, shape) -> Optional[float]:
    """Per-**region** mesh size for a ``conforming=False`` multi-material shape.

    The ordinary path (:func:`_apply_size_fields`) turns each ``size=`` into a Distance+Threshold
    *field*, which is a function of POSITION. Two coincident faces sit at the same position, so both
    bodies of a non-conforming interface get the same target size and gmsh meshes them identically --
    the two sides come out matching, the tie resolves node-to-node, and the mortar coupling is never
    reached. Measured: a 3x size ratio between two stacked blocks still gave 41 nodes on each side.

    Sizing each region's own OCC entities instead is what makes "coarse body, fine body" expressible.
    Entities are attributed to regions by centroid containment -- the same test ``_to_meshio`` uses for
    cells -- so no extra plumbing is needed from the emitter. Note this only makes sense when the
    pieces are NOT fragmented: a conforming interface shares its points between both regions, so a
    per-region size would just be whichever was written last.

    Returns a representative ``ds``, or ``None`` when it does not apply (leaving the field path in
    charge).
    """
    node = shape._node
    if node[0] != "regions" or (len(node) > 2 and node[2]):
        return None  # not a regions shape, or conforming -> shared entities, per-region size is moot
    items = node[1]
    sizes = {name: sub._size for name, sub in items if isinstance(sub._size, (int, float))}
    if not sizes:
        return None

    import gmsh
    import numpy as np

    applied: List[float] = []
    for edim, etag in gmsh.model.getEntities(dim):
        centre = np.asarray(gmsh.model.occ.getCenterOfMass(edim, etag), dtype=float).reshape(1, 3)
        for name, sub in items:
            if name not in sizes:
                continue
            try:
                inside = bool(np.asarray(sub.contains(centre[:, : sub.dim]))[0])
            except NotImplementedError:
                inside = False  # a swept/revolved region has no closed-form membership; skip it
            if inside:
                pts = [e for e in gmsh.model.getBoundary([(edim, etag)], oriented=False, recursive=True) if e[0] == 0]
                if pts:
                    gmsh.model.mesh.setSize(pts, float(sizes[name]))
                    applied.append(float(sizes[name]))
                break
    if not applied:
        return None
    # The field path disables point sizes; this one depends on them.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    return float(min(applied))


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


#: gmsh element types for the second-order (curved) simplices, keyed by (dim, n_nodes-per-cell).
_TRI6, _TET10, _LINE3 = 9, 11, 8


def _to_meshio(
    dim: int, labels: Dict[int, Tuple[int, str]], region_items=None, nonconforming: bool = False, order: int = 1
):
    """Assemble the generated gmsh mesh into a ``meshio.Mesh`` with named ``cell_sets``.

    ``region_items`` (``((name, sub_shape), ...)``, when the plan is a :meth:`Shape.regions`
    multi-material domain) adds one volume ``cell_set`` per region — each cell assigned to the first
    region whose shape contains its centroid — plus one facet ``cell_set`` per material **interface**,
    auto-named by the region pair it separates (``"a|b"`` = every facet between those two materials,
    however many gmsh faces that spans). Only *topologically disjoint* interfaces of the same pair
    (e.g. two separate inclusions) additionally split into connected components ``"a|b.0"`` / ``"a|b.1"``.

    ``nonconforming`` (``Shape.regions(conforming=False)``) changes how an interface is *found*. With
    the fragment skipped, the two sides of an interface are separate OCC entities each adjacent to a
    single volume, so the adjacency test above sees them as ordinary outer boundary. They are instead
    matched **geometrically** — two boundary faces of different regions occupying the same bounding
    box — and each side is tagged separately as ``"a|b.a"`` / ``"a|b.b"``, since the two are spatially
    coincident and no ``domain.tag`` predicate could tell them apart. Tie them in ``jno.fem`` with
    ``u("a|b.a") - u("a|b.b")``.
    """
    import gmsh
    import meshio
    import numpy as np

    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    index = {int(t): i for i, t in enumerate(node_tags)}

    curved = int(order) > 1
    if dim == 1:
        vtype, npv, vblock = (_LINE3, 3, "line3") if curved else (_LINE, 2, "line")
        btype, npb, bblock = _POINT, 1, "vertex"  # boundary of a 1-D domain = its two endpoints
    elif dim == 3:
        vtype, npv, vblock = (_TET10, 10, "tetra10") if curved else (_TET, 4, "tetra")
        btype, npb, bblock = (_TRI6, 6, "triangle6") if curved else (_TRI, 3, "triangle")
    else:
        vtype, npv, vblock = (_TRI6, 6, "triangle6") if curved else (_TRI, 3, "triangle")
        btype, npb, bblock = (_LINE3, 3, "line3") if curved else (_LINE, 2, "line")

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
    nonconf_faces: List[Tuple[tuple, str, np.ndarray]] = []  # (bbox, owning region, facet range)
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
            if nonconforming and len(adj) == 1 and int(adj[0]) in vol_region:
                # Remember this face's owner + extent; coincident faces of DIFFERENT regions are the
                # two sides of a non-conforming interface, matched after the loop.
                bbox = tuple(np.round(gmsh.model.getBoundingBox(bdim, etag), 9))
                nonconf_faces.append((bbox, vol_region[int(adj[0])], rng))

    facets = np.asarray(facet_rows, dtype=np.int64).reshape(-1, npb) if facet_rows else np.zeros((0, npb), dtype=np.int64)
    empty = np.array([], dtype=np.int64)
    cells = [(vblock, vcells), (bblock, facets)]
    cell_sets: Dict[str, list] = {"interior": [np.arange(len(vcells), dtype=np.int64), empty.copy()]}

    # Non-conforming interfaces: group the region-owned outer faces by extent; any bounding box shared
    # by two or more regions is an interface, and each region's side gets its own tag.
    by_bbox: Dict[tuple, Dict[str, List[np.ndarray]]] = {}
    for bbox, region, rng in nonconf_faces:
        by_bbox.setdefault(bbox, {}).setdefault(region, []).append(rng)
    nonconf_iface: List[np.ndarray] = []
    for sides in by_bbox.values():
        if len(sides) < 2:
            continue  # only one region on this footprint -> a genuine outer face, not an interface
        pair = "|".join(sorted(sides))
        for region, chunks in sides.items():
            idx = np.concatenate(chunks)
            cell_sets[f"{pair}.{region}"] = [empty.copy(), idx]
            nonconf_iface.append(idx)

    # These faces are INTERNAL to the assembly even though each is topologically its own body's outer
    # surface, so they must be withheld from "boundary" and from the auto face names exactly as a
    # conforming "a|b" interface is. Leaving them in makes a `u(boundary) - g` Dirichlet pin the
    # interface itself -- measured as a 24% drop in the peak solution before this was excluded.
    drop = np.concatenate(nonconf_iface) if nonconf_iface else empty.copy()
    _keep = (lambda a: a[~np.isin(a, drop)]) if drop.size else (lambda a: a)
    for name, chunks in by_name.items():
        cell_sets[name] = [empty.copy(), _keep(np.concatenate(chunks)) if chunks else empty.copy()]
    # "boundary" is the OUTER boundary only (internal interfaces are separate, named tags).
    cell_sets["boundary"] = [empty.copy(), _keep(np.concatenate(external_ranges) if external_ranges else empty.copy())]

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
    """gmsh ``setPeriodic`` for each ``(main_name, secondary_name)`` boundary-face pair: mesh the secondary face
    as a *translated copy* of the main so opposite boundaries mesh identically (conforming) — required
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

    for main, secondary in periodic:
        m_tags, s_tags = by_name.get(main, []), by_name.get(secondary, [])
        if not m_tags or not s_tags:
            continue
        t = _centroid(s_tags) - _centroid(m_tags)  # translation main -> secondary
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


def _build_once(shape, split_full, periodic=None, algorithm=None, threads=None, order=1):
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
        # A non-conforming multi-material shape sizes each region's own entities; the position-based
        # field path cannot distinguish two coincident faces and would mesh both bodies identically.
        ds = _apply_region_sizes(dim, shape) or _apply_size_fields(dim, leaves, labels, shape)
        if periodic:  # make named opposite faces conform (edge-element periodic ties need it)
            _apply_periodic(dim, labels, periodic)
        gmsh.model.mesh.generate(dim)
        if int(order) > 1:
            # CURVED (isoparametric) geometry. Without this the mesh is straight-sided and jNO
            # SYNTHESISES its higher-order nodes by interpolating reference points through the affine
            # map (`fem_native._get_mesh` -> `_promote_to_degree`), so a P2 midside node lands on the
            # straight-edge midpoint and the domain stays a polygon however high the basis order goes.
            # `setOrder` asks gmsh to place those nodes on the actual CAD surface instead; without
            # `HighOrderOptimize` gmsh curves only the boundary entities, which is exactly the part
            # that matters and keeps interior cells affine (so `detJ` cannot go negative on them).
            gmsh.model.mesh.setOrder(int(order))
        is_regions = shape._node[0] == "regions"
        region_items = shape._node[1] if is_regions else None
        nonconforming = is_regions and len(shape._node) > 2 and not shape._node[2]
        mesh = _to_meshio(dim, labels, region_items, nonconforming=nonconforming, order=int(order))
        return mesh, dim, ds
    finally:
        gmsh.model.remove()
        if started:
            gmsh.finalize()


def build(shape, periodic=None, *, algorithm=None, threads=None, order=1):
    """Mesh ``shape`` -> ``(meshio.Mesh, dim, ds)``.

    ``algorithm`` selects gmsh's meshing kernel for the shape's own dimension -- there is no
    separate 2-D/3-D argument because ``shape.dim`` already says which applies. ``threads`` sets the
    thread count. ``None`` uses :data:`MESH_ALGORITHM_2D` / :data:`MESH_ALGORITHM_3D` /
    :data:`MESH_THREADS`, whose docstrings record the measurements behind each default.

    ``periodic`` is an optional list of ``(main_name, secondary_name)`` boundary-face pairs meshed conforming
    (via :func:`_apply_periodic`) so opposite faces line up — needed for Nédélec edge periodic ties.

    ``order=2`` meshes **curved (isoparametric)** geometry: gmsh places the midside nodes on the actual
    CAD surface instead of jNO synthesising them at straight-edge midpoints, so a round boundary stays
    round. Without it the domain is a polygon at every basis order, which caps the discretisation at
    O(h^2) however high the element order goes, and leaves facet normals O(h) wrong. Emits second-order
    meshio blocks (``triangle6`` / ``tetra10`` / ``line3``).

    A single-sweep full (2pi) revolve of a *detached* profile makes a periodic surface gmsh cannot mesh,
    while an axis-touching profile (a cone) meshes fine that way -- so we try the single sweep first and
    only fall back to the two-halves construction if meshing fails.
    """
    opts = dict(algorithm=algorithm, threads=threads, order=order)
    try:
        return _build_once(shape, split_full=False, periodic=periodic, **opts)
    except Exception as exc:  # noqa: BLE001 - narrow retry on the periodic-surface mesher failure
        if "periodic" in str(exc).lower() and _has_full_revolve(shape._node):
            return _build_once(shape, split_full=True, periodic=periodic, **opts)
        raise
