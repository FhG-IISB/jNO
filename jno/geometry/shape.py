"""``Shape`` -- a friendly, immutable geometry build-plan over gmsh-OpenCASCADE.

A ``Shape`` records *what to build*, not a mesh: primitive leaves combined by boolean
operators (``-`` cut, ``|`` fuse, ``&`` intersect) and dimension transitions
(``.extrude``). It touches gmsh only inside :meth:`build` (delegated to
:mod:`jno.geometry.emit`), so authoring and the naming/selection algebra stay
pure-Python and testable without a mesher.

Mesh size rides on the shape it describes (``size=`` / :meth:`sized`) -- config on the
term it describes, not a global argument on ``jno.domain``. Primitives auto-name their
boundaries (``left/right/top/bottom``, ``arc``, extrude caps ``front/back``); you select
and merge afterwards with :meth:`edge` (by auto-name) and :meth:`edges_from` (by the
primitive a boundary came from), combining with ``|``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Callable, FrozenSet, Optional, Tuple, Union

import numpy as np

from .primitives import Box, Cylinder, Disk, Polygon, Rect, Sphere, _to3

# Nodes with no closed-form point membership. `sweep` follows an arbitrary path and `fillet`
# *removes* material near edges, so recursing to the child would silently answer for the
# un-filleted solid — a superset, i.e. a wrong-but-plausible mask. Both defer to a boundary
# tessellation instead (see ``Shape.tessellate``).
_NO_ANALYTIC_MEMBERSHIP = ("sweep", "fillet")


def _rotate_points(pts, axis_point, axis_dir, angle):
    """Rodrigues rotation of ``(N, 3)`` points about the axis — the vectorised twin of
    :func:`jno.geometry.naming._rotate_point`, which does the same for one point."""
    p = np.asarray(axis_point, dtype=float)
    u = np.asarray(axis_dir, dtype=float)
    u = u / (np.linalg.norm(u) or 1.0)
    v = pts - p
    c, s = math.cos(angle), math.sin(angle)
    return p + v * c + np.cross(u, v) * s + u * ((v @ u) * (1.0 - c))[:, None]


def _revolve_coords(pts, axis_point, axis_dir):
    """``(meridian_xy, azimuth)`` for revolved points: the 2-D profile coords each point came
    from, and how far round the sweep it sits. Mirrors :func:`naming._revolve_profile_coords`
    (same x-/y-axis-through-the-origin restriction) and adds the azimuth a partial sweep needs."""
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    ax, ay, az = axis_dir
    at_origin = all(abs(c) < 1e-9 for c in axis_point)
    if at_origin and abs(ay) > 1e-9 and abs(ax) < 1e-9 and abs(az) < 1e-9:  # y-axis
        # rotating the profile (z=0, x>0) by s sends it to (X cos s, Y, -X sin s)
        return np.stack([np.hypot(X, Z), Y], axis=1), np.mod(np.arctan2(-Z, X), 2.0 * math.pi)
    if at_origin and abs(ax) > 1e-9 and abs(ay) < 1e-9 and abs(az) < 1e-9:  # x-axis
        return np.stack([X, np.hypot(Y, Z)], axis=1), np.mod(np.arctan2(Z, Y), 2.0 * math.pi)
    raise NotImplementedError(
        f"revolve currently supports the x- or y-axis through the origin; got "
        f"axis_point={axis_point}, axis_dir={axis_dir}."
    )


def _node_contains(node, pts, tol):
    """Analytic point membership over the CSG tree — leaves, booleans and rigid/sweep transforms.

    Transforms are handled by mapping the *query point* into the child's own frame and recursing,
    which is the same strategy :func:`jno.geometry.naming._classify` uses for boundary naming.
    No gmsh.
    """
    kind = node[0]
    if kind == "leaf":
        return np.asarray(node[1].contains(pts, tol), dtype=bool)
    if kind == "cut":
        # The subtrahend is tested EXCLUSIVELY (-tol): `A - B` keeps the cut surface itself, which
        # is where gmsh puts nodes. Testing B inclusively rejected every node on a hole's boundary
        # -- 8% of an annulus mesh, 14% of a box with a spherical void, measured.
        return _node_contains(node[1]._node, pts, tol) & ~_node_contains(node[2]._node, pts, -tol)
    if kind == "fuse":
        return _node_contains(node[1]._node, pts, tol) | _node_contains(node[2]._node, pts, tol)
    if kind == "inter":
        return _node_contains(node[1]._node, pts, tol) & _node_contains(node[2]._node, pts, tol)
    if kind == "regions":
        mask = np.zeros(len(pts), dtype=bool)
        for _name, sub in node[1]:
            mask |= _node_contains(sub._node, pts, tol)
        return mask
    if kind == "translate":
        return _node_contains(node[1]._node, _to3(pts) - np.asarray(node[2], dtype=float), tol)
    if kind == "rotate":
        local = _rotate_points(_to3(pts), node[2], node[3], -node[4])  # undo the rotation
        return _node_contains(node[1]._node, local, tol)
    if kind == "extrude":
        p = _to3(pts)
        height = float(node[2])
        lo, hi = min(0.0, height), max(0.0, height)
        within = (p[:, 2] >= lo - tol) & (p[:, 2] <= hi + tol)
        base = p.copy()
        base[:, 2] = 0.0  # the height is consumed here; the base lives in its own z=0 plane
        return within & _node_contains(node[1]._node, base, tol)
    if kind == "revolve":
        axis_point, axis_dir, angle = node[2], node[3], node[4]
        meridian, azimuth = _revolve_coords(_to3(pts), axis_point, axis_dir)
        swept = np.ones(len(pts), dtype=bool)
        if abs(angle - 2.0 * math.pi) > 1e-9:  # a partial sweep only fills part of the turn
            span = abs(angle)
            atol = tol
            swept = (azimuth <= span + atol) | (azimuth >= 2.0 * math.pi - atol)
        return swept & _node_contains(node[1]._node, meridian, tol)
    raise NotImplementedError(
        f"Shape.contains has no closed-form point membership for a {kind!r} node. Supported "
        f"analytically: primitive leaves, '-'/'|'/'&' (cut/fuse/inter), regions, translate, "
        f"rotate, extrude and revolve. A {kind!r} shape is sampled through its boundary "
        f"tessellation instead — call Shape.sample_interior/sample_boundary rather than "
        f"Shape.contains, or tag that region another way."
    )


def _node_bounds(node):
    """Axis-aligned bounding box ``(lo, hi)`` of the CSG tree as two 3-vectors, no gmsh.

    Every case returns a **superset** of the true extent, never a subset: that is what lets it
    serve as the proposal box for rejection sampling. ``cut`` keeps the left operand's box (the
    difference can only shrink) and ``rotate``/``revolve`` bound the swept envelope rather than
    tracking it exactly.
    """
    kind = node[0]
    if kind == "leaf":
        lo, hi = node[1].bounds()
        return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if kind == "cut":
        return _node_bounds(node[1]._node)  # A - B is contained in A
    if kind in ("fuse", "inter"):
        alo, ahi = _node_bounds(node[1]._node)
        blo, bhi = _node_bounds(node[2]._node)
        if kind == "fuse":
            return np.minimum(alo, blo), np.maximum(ahi, bhi)
        return np.maximum(alo, blo), np.minimum(ahi, bhi)
    if kind == "regions":
        boxes = [_node_bounds(sub._node) for _name, sub in node[1]]
        return (
            np.min(np.stack([b[0] for b in boxes]), axis=0),
            np.max(np.stack([b[1] for b in boxes]), axis=0),
        )
    if kind == "translate":
        lo, hi = _node_bounds(node[1]._node)
        d = np.asarray(node[2], dtype=float)
        return lo + d, hi + d
    if kind == "rotate":
        lo, hi = _node_bounds(node[1]._node)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        moved = _rotate_points(corners, node[2], node[3], node[4])
        return moved.min(axis=0), moved.max(axis=0)
    if kind == "extrude":
        lo, hi = _node_bounds(node[1]._node)
        height = float(node[2])
        lo, hi = lo.copy(), hi.copy()
        lo[2], hi[2] = min(0.0, height), max(0.0, height)
        return lo, hi
    if kind == "revolve":
        lo, hi = _node_bounds(node[1]._node)  # the profile, in its own (x, y) meridian plane
        axis_point, axis_dir = node[2], node[3]
        _ax, ay, _az = axis_dir
        at_origin = all(abs(c) < 1e-9 for c in axis_point)
        if at_origin and abs(ay) > 1e-9:  # y-axis: profile x is the radius, profile y the height
            r = max(abs(lo[0]), abs(hi[0]))
            return np.array([-r, lo[1], -r]), np.array([r, hi[1], r])
        # x-axis: profile x runs along the axis, profile y is the radius
        r = max(abs(lo[1]), abs(hi[1]))
        return np.array([lo[0], -r, -r]), np.array([hi[0], r, r])
    raise NotImplementedError(
        f"Shape.bounds has no closed-form bounding box for a {kind!r} node; its extent comes "
        f"from the boundary tessellation instead (Shape.tessellate)."
    )


def _rotate_dirs(vecs, axis_dir, angle):
    """Rodrigues rotation of direction vectors — like :func:`_rotate_points` with no origin shift,
    so a normal rotates with its surface instead of being dragged by the axis offset."""
    return _rotate_points(vecs, (0.0, 0.0, 0.0), axis_dir, angle)


class _Moved:
    """A boundary source wrapped in a rigid transform: draws from ``inner`` and maps the points
    (and normals) into the parent frame, so a translated/rotated leaf still samples exactly."""

    def __init__(self, inner, translate=None, rotate=None):
        self._inner, self._translate, self._rotate = inner, translate, rotate

    def boundary_measure(self):
        return self._inner.boundary_measure()  # rigid motions preserve length and area

    def sample_boundary(self, n, rng):
        pts, nrm = self._inner.sample_boundary(n, rng)
        if self._rotate is not None:
            axis_point, axis_dir, angle = self._rotate
            pts = _rotate_points(pts, axis_point, axis_dir, angle)
            nrm = _rotate_dirs(nrm, axis_dir, angle)
        if self._translate is not None:
            pts = pts + np.asarray(self._translate, dtype=float)
        return pts, nrm


class _ExtrudedSurface:
    """The boundary of ``extrude(base, height)``: the swept lateral wall plus the two flat caps.

    The lateral wall is the base's own boundary carried to a uniform height (its 2-D outward normal
    is already the 3-D one); each cap is the base's *interior* at a fixed z with normal -/+ z. The
    cap-versus-wall split is weighted by the base area, which is estimated by Monte Carlo -- the
    only inexact step here, and it affects how the draws are *shared* between wall and cap, never
    where an individual point lands.
    """

    _AREA_SAMPLES = 20_000

    def __init__(self, base, height):
        self._base, self._h = base, float(height)
        lo, hi = (np.asarray(v, dtype=float) for v in base.bounds())
        span = hi - lo
        box = float(np.prod(span[span > 0.0])) if np.any(span > 0.0) else 0.0
        rng = np.random.default_rng(0)  # fixed: an area estimate must not wobble between calls
        free = span > 0.0
        cand = np.tile(lo, (self._AREA_SAMPLES, 1))
        cand[:, free] += rng.uniform(0.0, 1.0, size=(self._AREA_SAMPLES, int(free.sum()))) * span[free]
        self._area = box * float(np.mean(_node_contains(base._node, cand, 0.0)))
        self._perimeter = _node_boundary_measure(base._node)

    def boundary_measure(self):
        return self._perimeter * abs(self._h) + 2.0 * self._area

    def sample_boundary(self, n, rng):
        lateral = self._perimeter * abs(self._h)
        w = np.array([lateral, self._area, self._area], dtype=float)
        f = rng.choice(3, size=n, p=w / w.sum())
        pts = np.zeros((n, 3))
        nrm = np.zeros((n, 3))
        lo, hi = min(0.0, self._h), max(0.0, self._h)
        sel = f == 0
        if sel.any():
            k = int(sel.sum())
            wall, wall_n = self._base.sample_boundary(k, rng)
            wall = wall.copy()
            wall[:, 2] = rng.uniform(lo, hi, size=k)
            pts[sel], nrm[sel] = wall, wall_n
        for cap, z, sign in ((1, lo, -1.0), (2, hi, 1.0)):
            sel = f == cap
            if sel.any():
                k = int(sel.sum())
                face = self._base.sample_interior(k, rng)
                face = face.copy()
                face[:, 2] = z
                pts[sel] = face
                nrm[sel] = np.array([0.0, 0.0, sign])
        return pts, nrm


def _node_boundary_measure(node):
    """Total analytic boundary measure (perimeter in 2-D, surface area in 3-D) of the raw pieces."""
    return float(sum(m for _piece, m in _node_boundary_pieces(node)))


def _node_boundary_pieces(node):
    """``[(source, measure)]`` whose ``sample_boundary(n, rng)`` draws in **world** coordinates.

    These are *candidate* surfaces — a leaf contributes its whole boundary even where a later
    boolean cut it away. :meth:`Shape.sample_boundary` trims them with an exact membership probe,
    so this only has to enumerate and weight, never to resolve the booleans itself.
    """
    kind = node[0]
    if kind == "leaf":
        prim = node[1]
        if not hasattr(prim, "sample_boundary"):
            raise NotImplementedError(f"{type(prim).__name__} has no analytic boundary sampler.")
        return [(prim, prim.boundary_measure())]
    if kind in ("cut", "fuse", "inter"):
        return _node_boundary_pieces(node[1]._node) + _node_boundary_pieces(node[2]._node)
    if kind == "regions":
        out = []
        for _name, sub in node[1]:
            out += _node_boundary_pieces(sub._node)
        return out
    if kind == "translate":
        return [(_Moved(p, translate=node[2]), m) for p, m in _node_boundary_pieces(node[1]._node)]
    if kind == "rotate":
        rot = (node[2], node[3], node[4])
        return [(_Moved(p, rotate=rot), m) for p, m in _node_boundary_pieces(node[1]._node)]
    if kind == "extrude":
        surf = _ExtrudedSurface(node[1], node[2])
        return [(surf, surf.boundary_measure())]
    raise NotImplementedError(
        f"no analytic boundary sampler for a {kind!r} node; sample it through the boundary "
        f"tessellation instead (Shape.tessellate)."
    )


# Unique, identity-stable key per primitive leaf, for provenance (``edges_from``).
_LEAF_KEYS = itertools.count()

Size = Union[float, Callable[..., float], None]


@dataclass(frozen=True)
class Selector:
    """A set-valued reference to boundary entities: by auto-name and/or by provenance key.

    Returned by :meth:`Shape.edge` / :meth:`Shape.edges_from`; combined with ``|``.
    Resolved against a built mesh's classified boundary (name + originating leaf key).
    """

    names: FrozenSet[str] = frozenset()
    keys: FrozenSet[int] = frozenset()

    def __or__(self, other: "Selector") -> "Selector":
        return Selector(self.names | other.names, self.keys | other.keys)

    def matches(self, key: int, local_name: str) -> bool:
        return local_name in self.names or key in self.keys


@dataclass(frozen=True)
class Shape:
    """An immutable geometry build-plan. Operators return new shapes.

    **Derive with :func:`dataclasses.replace`, never by calling ``Shape(...)`` positionally.** Every
    method here returns a copy with one or two fields changed, and a positional constructor has to
    re-list every *other* field to carry it — so a field added later is silently dropped by whichever
    derivation forgets it, and the plan quietly meshes as something the caller did not ask for. That
    is not hypothetical: ``.quad().attach(k=1.0)`` used to erase the quadrilateral choice outright.
    """

    _node: tuple
    dim: int
    _size: Size = field(default=None, compare=False)
    _region_name: Optional[str] = field(default=None, compare=False)
    _mesh_order: int = field(default=1, compare=False)
    # Attached material properties, ``{name: value}``. ``compare=False`` keeps the dataclass hashable
    # (a dict is not) and keeps two geometrically identical shapes equal regardless of their materials.
    _attach: Optional[dict] = field(default=None, compare=False)
    # Volume cell to mesh with: None/"simplex" (triangles, tets) or "quad" (2-D quadrilaterals,
    # via gmsh recombination). Not part of equality for the same reason `_size` is not -- it
    # describes how to mesh the geometry, not what the geometry is.
    _cell: Optional[str] = field(default=None, compare=False)
    # Regular-lattice request from :meth:`structured`. ``None`` means "mesh it with gmsh"; a tuple of
    # per-axis CELL counts, or ``()`` for "derive them from ``size=``". Like ``_cell`` and ``_size``
    # this describes how to mesh the geometry, not what the geometry is, so it is not part of equality.
    _structured: Optional[tuple] = field(default=None, compare=False)

    # ----- primitive constructors ------------------------------------------------
    @classmethod
    def rect(cls, x0: float, y0: float, x1: float, y1: float, size: Size = None) -> "Shape":
        return cls(("leaf", Rect(x0, y0, x1, y1), next(_LEAF_KEYS)), 2, size)

    @classmethod
    def disk(cls, cx: float, cy: float, r: float, size: Size = None) -> "Shape":
        return cls(("leaf", Disk(cx, cy, r), next(_LEAF_KEYS)), 2, size)

    @classmethod
    def box(cls, x0: float, y0: float, z0: float, x1: float, y1: float, z1: float, size: Size = None) -> "Shape":
        return cls(("leaf", Box(x0, y0, z0, x1, y1, z1), next(_LEAF_KEYS)), 3, size)

    @classmethod
    def polygon(cls, points, size: Size = None) -> "Shape":
        """Arbitrary 2-D polygon from ordered ``(x, y)`` vertices; edges auto-named ``e0, e1, ...``."""
        pts = tuple((float(px), float(py)) for px, py in points)
        return cls(("leaf", Polygon(pts), next(_LEAF_KEYS)), 2, size)

    @classmethod
    def cylinder(
        cls, x: float, y: float, z: float, dx: float, dy: float, dz: float, r: float, size: Size = None
    ) -> "Shape":
        """Right cylinder: base centre ``(x,y,z)``, axis vector ``(dx,dy,dz)``, radius ``r``."""
        return cls(("leaf", Cylinder(x, y, z, dx, dy, dz, r), next(_LEAF_KEYS)), 3, size)

    @classmethod
    def sphere(cls, cx: float, cy: float, cz: float, r: float, size: Size = None) -> "Shape":
        """Sphere centred ``(cx,cy,cz)`` radius ``r``."""
        return cls(("leaf", Sphere(cx, cy, cz, r), next(_LEAF_KEYS)), 3, size)

    @classmethod
    def regions(cls, mapping=None, /, **named: "Shape") -> "Shape":
        """A multi-material domain: named sub-regions meshed **conforming** to their interfaces.

        Each keyword names a sub-region shape (all same ``dim``); the pieces are fragmented
        (``occ.fragment``) so element edges align exactly with every material interface, then each
        volume cell is assigned to the **first** region (keyword order = priority) whose shape
        contains its centroid — exact, because the mesh conforms. The realized domain exposes each
        region as its own variable set (``d.variable("core")``) alongside ``interior``/``boundary``,
        and the outer boundary keeps its auto-names; internal interface facets are not boundary.

        Regions may overlap: ``Shape.regions(inclusion=disk, matrix=plate)`` labels the disk
        ``inclusion`` (higher priority) and the remainder ``matrix``. Equivalent to combining named
        shapes with ``+`` — ``disk.name("inclusion") + plate.name("matrix")``. Must be the top-level
        shape (call ``.domain()`` on it); it is not composable with boolean operators/transforms.

        ``conforming=False`` skips the fragment: every piece is meshed **independently**, so two
        touching regions end up with two coincident but *non-matching* surfaces and duplicated nodes
        instead of one shared interface. Each side is tagged ``"a|b.a"`` / ``"a|b.b"``, and tying them
        with ``u("a|b.a") - u("a|b.b")`` in ``jno.fem`` glues them by a mortar coupling — which is how
        you join two bodies meshed at *different* resolutions. ``conforming`` is therefore a reserved
        region name.

        A positional ``{name: shape}`` **dict** is accepted for names that are not valid Python
        identifiers -- ``Shape.regions({"Quartz.1": q1, "Quartz.2": q2})``. Dict order is priority
        order, exactly as for keywords, and the two forms may be combined (dict entries first).
        """
        conforming = named.pop("conforming", True)
        if not isinstance(conforming, bool):
            raise TypeError(f"Shape.regions: `conforming` must be a bool, got {type(conforming).__name__}")
        items: Tuple[Tuple[str, "Shape"], ...] = ()
        if mapping is not None:
            if not isinstance(mapping, dict):
                raise TypeError(
                    f"Shape.regions: the positional argument must be a {{name: shape}} dict, got "
                    f"{type(mapping).__name__}. Pass shapes as keywords, or as one dict."
                )
            items += tuple((str(k), v) for k, v in mapping.items())
        return cls._from_region_items(items + tuple(named.items()), conforming=conforming)

    @classmethod
    def _from_region_items(cls, items, conforming: bool = True) -> "Shape":
        if len(items) < 2:
            raise ValueError("a multi-material domain needs at least two named regions")
        names = [n for n, _ in items]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate region name in {names}")
        dims = {sub.dim for _n, sub in items}
        if len(dims) != 1:
            raise ValueError(f"all regions must share one dimension, got {sorted(dims)}")
        # curving is a property of the whole mesh, so a multi-material domain is curved if ANY of its
        # regions asked for it -- otherwise `.curved()` on one part would be silently dropped
        order = max(int(getattr(sub, "_mesh_order", 1)) for _n, sub in items)
        return cls(("regions", tuple(items), bool(conforming)), items[0][1].dim, None, None, order)

    def name(self, name: str) -> "Shape":
        """Label this shape as a named material region, for combining with ``+`` (see :meth:`regions`).

        ``core.name("core") + clad.name("clad")`` builds a multi-material domain whose regions are
        ``core`` and ``clad``. Apply ``name`` last (a later transform drops the label)."""
        return replace(self, _region_name=str(name))

    def attach(self, **props) -> "Shape":
        """Attach material properties to this region: ``.attach(k=220.0, eps=0.794)``.

        The realized domain exposes each attached name as a **per-region coefficient** ready to drop
        into a weak form -- ``d.k`` is exactly ``d.by_region({"Kristall": 220.0, ...})`` assembled from
        every region that attached a ``k``::

            kri = Shape.polygon(v).name("Kristall").attach(k=220.0, eps=0.794)
            gas = Shape.polygon(w).name("Gas").attach(k=0.186, eps=1.0)
            d   = (kri + gas).domain()
            heat = d.k * (T.x*s.x + T.y*s.y) - d.q * s

        A value may be anything :meth:`domain.by_region` accepts -- a scalar, a symbolic expression,
        or a traced/trainable array -- so an attached property can be fitted or differentiated
        through. A plain **function** is also accepted and is called with the domain's spatial
        coordinates when the property is read (``.attach(k=lambda r, z: 2.0 + 0.5*z)``); it has to be
        deferred that way because a spatially varying coefficient is built from ``d.variable(...)``,
        which does not exist yet while the geometry plan is being written.

        Properties are declared, not typed: whether ``eps`` is a volume or a surface quantity is
        decided by the term that consumes it, not here.

        ``d.<name>`` raises if **any** region failed to attach that name, listing the ones that did
        not: a forgotten material surfaces at first use rather than as a region that silently
        conducts nothing. Repeated calls merge (last wins), so properties can be built up in stages.
        Apply after :meth:`name` -- like ``name``, a later transform drops the attachment."""
        merged = dict(self._attach or {})
        merged.update(props)
        return replace(self, _attach=merged)

    def size(self, size: Size) -> "Shape":
        """Alias for :meth:`sized` -- ``.size(h)`` reads better in a chain than ``.sized(h)``."""
        return self.sized(size)

    def _region_items(self):
        """This shape as ``((name, shape), ...)`` region items — its own group, or a single named leaf."""
        if self._node[0] == "regions":
            return self._node[1]
        if self._region_name is None:
            raise ValueError("combine regions with '+' only after naming each with .name('...')")
        return ((self._region_name, self),)

    def __add__(self, other: "Shape") -> "Shape":
        """Combine named regions into a conforming multi-material domain (sugar for :meth:`regions`).

        ``a.name("x") + b.name("y")`` keeps ``a`` and ``b`` as distinct materials with a conforming
        interface — unlike ``a | b`` (fuse), which merges them into one. Left-to-right order is region
        priority; composes n-ary (``a + b + c``)."""
        if not isinstance(other, Shape):
            return NotImplemented
        return Shape._from_region_items(self._region_items() + other._region_items())

    # ----- boolean operators -----------------------------------------------------
    def __sub__(self, other: "Shape") -> "Shape":
        return replace(
            self, _node=("cut", self, other), _region_name=None, _mesh_order=max(self._mesh_order, other._mesh_order)
        )

    def __or__(self, other: "Shape") -> "Shape":
        return replace(
            self, _node=("fuse", self, other), _region_name=None, _mesh_order=max(self._mesh_order, other._mesh_order)
        )

    def __and__(self, other: "Shape") -> "Shape":
        return replace(
            self, _node=("inter", self, other), _region_name=None, _mesh_order=max(self._mesh_order, other._mesh_order)
        )

    # ----- transforms ------------------------------------------------------------
    def extrude(self, height: float) -> "Shape":
        if self.dim != 2:
            raise ValueError("extrude requires a 2-D shape")
        return replace(self, _node=("extrude", self, float(height)), dim=3, _region_name=None)

    def revolve(self, axis_point, axis_dir, angle: float = 2.0 * math.pi) -> "Shape":
        """Sweep a 2-D shape around an axis by ``angle`` radians into a 3-D solid.

        ``angle == 2*pi`` gives a full solid of revolution; a partial angle gives a wedge
        or half-donut (revolve a disk offset from the axis by ``pi``). The profile lies in
        the z=0 plane on the positive-radius side; the axis must be the x- or y-axis through
        the origin (the common axisymmetric case) -- other axes raise at build time. Swept
        faces inherit the profile edge's auto-name (``arc``, ``e0``, ...); a partial sweep
        adds ``back``/``front`` end caps.
        """
        if self.dim != 2:
            raise ValueError("revolve requires a 2-D shape")
        ap = tuple(float(c) for c in axis_point)
        ad = tuple(float(c) for c in axis_dir)
        # Validate the axis up front: an unsupported axis (e.g. z, perpendicular to the profile)
        # is a degenerate zero-volume sweep that hangs OCC rather than erroring cleanly.
        at_origin = all(abs(c) < 1e-9 for c in ap)
        x_on, y_on, z_on = (abs(ad[0]) > 1e-9, abs(ad[1]) > 1e-9, abs(ad[2]) > 1e-9)
        if not (at_origin and not z_on and (x_on != y_on)):
            raise NotImplementedError(
                "revolve currently supports the x- or y-axis through the origin (axisymmetric); "
                f"got axis_point={ap}, axis_dir={ad}."
            )
        return replace(self, _node=("revolve", self, ap, ad, float(angle)), dim=3, _region_name=None)

    def translate(self, vector) -> "Shape":
        """Move the shape by ``vector`` (2- or 3-component). Boundary names are preserved."""
        v = tuple(float(c) for c in vector)
        if len(v) == 2:
            v = (v[0], v[1], 0.0)
        return replace(self, _node=("translate", self, v), _region_name=None)

    def rotate(self, axis_point, axis_dir, angle: float) -> "Shape":
        """Rotate ``angle`` radians about the axis through ``axis_point`` along ``axis_dir``."""
        ap = tuple(float(c) for c in axis_point)
        ad = tuple(float(c) for c in axis_dir)
        return replace(self, _node=("rotate", self, ap, ad, float(angle)), _region_name=None)

    def sweep(self, path) -> "Shape":
        """Sweep this 2-D profile along an open :class:`~jno.geometry.path.Path` trajectory.

        The path must be smooth (line and arc segments); a sharp line->line corner is rejected
        up front (it self-intersects the swept profile). Naming degrades to ``interior`` +
        ``boundary`` for a general sweep -- carve regions afterwards with ``d.tag(predicate)``.
        """
        if self.dim != 2:
            raise ValueError("sweep requires a 2-D profile")
        path._check_sweepable()
        h = path._as_extrude()
        if h is not None:
            return self.extrude(h)  # a straight vertical sweep IS an extrude -- reuse its rich naming
        return replace(self, _node=("sweep", self, path), dim=3, _region_name=None)

    def array(self, n: int, step=None, about=None, angle: float = 2.0 * math.pi) -> "Shape":
        """``n`` fused copies of this shape: a **linear** array (``step=`` vector between copies)
        or a **polar** array (``about=(axis_point, axis_dir)`` spread over ``angle``).

        Pure composition over translate/rotate/fuse -- e.g. a bolt-circle of holes is
        ``plate - Shape.disk(R, 0, r).array(8, about=((0,0,0),(0,0,1)))``.
        """
        if n < 1:
            raise ValueError("array needs n >= 1")
        if (step is None) == (about is None):
            raise ValueError("array needs exactly one of step= (linear) or about= (polar)")
        out = self
        for k in range(1, n):
            if step is not None:
                out = out | self.translate(tuple(k * float(c) for c in step))
            else:
                out = out | self.rotate(about[0], about[1], k * angle / n)
        return out

    def fillet(self, radius: float, where=None) -> "Shape":
        """Round the solid's edges by ``radius``.

        ``where=f(x, y, z)`` selects which edges to round by their midpoint (default: all
        edges). The rounded blend faces are unnamed (they fall into ``boundary``); the flat
        faces keep their names.
        """
        return replace(self, _node=("fillet", self, float(radius), where), _region_name=None)

    def sized(self, size: Size) -> "Shape":
        """Return a copy of this shape with its target mesh size set (scalar or ``f(x,y,z)``)."""
        return replace(self, _size=size)

    def curved(self, order: int = 2) -> "Shape":
        """Return a copy meshed with **curved (isoparametric)** geometry of the given order.

        By default jNO meshes straight-sided and *synthesises* any higher-order nodes at the
        straight-edge midpoints, so the domain stays a polygon however high the element order goes.
        That polygonal approximation carries an **O(h²)** domain error at every basis order — it is
        what caps P2/P3 at second order on a round boundary — and leaves facet normals O(h) wrong.
        ``curved()`` asks the CAD kernel to place those nodes on the true surface instead::

            d = jno.Shape.disk(0, 0, 1, size=0.1).curved().domain()

        Worth pairing with a matching element order (``jno.fem(..., order=2)``); a curved mesh under a
        P1 basis buys only the geometry, not the convergence rate. Meshing is a property of the shape,
        which is why this lives here beside :meth:`sized` rather than on the solve."""
        if int(order) not in (1, 2):
            raise ValueError(f"Shape.curved: only order 1 (straight) or 2 is supported, got {order!r}.")
        return replace(self, _mesh_order=int(order))

    def quad(self) -> "Shape":
        """Return a copy meshed with **quadrilaterals** — or, on a structured 3-D plan, hexahedra::

            d = jno.Shape.disk(0, 0, 1, size=0.1).quad().domain()                  # quads
            d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.1).structured().quad().domain()   # hexes

        In 2-D gmsh meshes triangles and then recombines them, which works on arbitrary geometry — a
        disk recombines to pure quads just as a rectangle does. Quadrilaterals cost fewer cells for
        the same nodes and are the element the topology-optimisation literature is written on; on
        bending-dominated elasticity they are markedly less stiff than linear triangles.

        **In 3-D it needs** :meth:`structured`. gmsh cannot hex-mesh general geometry: measured here,
        ``Recombine3DAll`` on a plain box returns 944 tetrahedra and no hexahedra at all. Hexes come
        from a regular lattice (or, later, from sweeping/transfinite meshing on a swept build plan),
        so an unstructured 3-D shape is refused and the message says to add ``.structured()``.

        Meshing is a property of the shape, which is why this lives beside :meth:`curved` and
        :meth:`sized` rather than on the solve.
        """
        # Checked at BUILD time, not here: `.structured()` and `.quad()` must compose in either order,
        # and only the finished plan knows whether a 3-D quad request has a lattice under it.
        return replace(self, _cell="quad")

    def tri(self) -> "Shape":
        """Return a copy meshed with **simplices** — triangles in 2-D, tetrahedra in 3-D.

        Simplices are the default, so this is the explicit *opposite* of :meth:`quad`: it turns
        recombination back off for a shape that inherited it from an enclosing plan::

            plate.quad() - hole.tri()      # (once mixed-cell meshes land, see below)
            (plate.quad()).tri()           # today: cancels the .quad()

        A cell choice rides on the shape it describes, so each shape in a plan already carries its
        own. What is not built yet is *honouring* two different choices in one mesh: gmsh can do it
        (``setRecombine`` is per-surface, and a quad region and a triangle region meeting at a shared
        edge conform node-for-node, since both have 2-node edges there), but jNO's assembler is built
        around one element table and one cell array. Until that lands, a plan that asks for two
        different cells is refused rather than silently meshed with one of them.
        """
        return replace(self, _cell="simplex")

    def structured(self, n=None) -> "Shape":
        """Return a copy meshed as a **regular lattice** instead of by gmsh::

            jno.Shape.rect(0, 0, 1, 1, size=0.1).structured().domain()
            jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.1).structured().quad().domain()   # hexes
            jno.Shape.box(...).structured(n=(32, 16, 16)).domain()

        Three things follow from a lattice that a gmsh mesh cannot give:

        * **Hexahedra.** gmsh cannot hex-mesh general geometry, which is why :meth:`quad` refuses a
          3-D shape — but a structured box is exactly the plan that *can* be hex-meshed, so
          ``.structured().quad()`` is how a hex mesh is spelled.
        * **A grid descriptor** on ``domain.grid``, which is what lets ``jno.fdm`` take its
          assembly-free 5-/7-point stencils instead of the cotangent operator, and what a nodal field
          reshapes against for operator-learning work.
        * **Exactly matched opposite faces**, so whole-domain periodic ties collapse onto one DOF
          rather than holding to a tolerance.

        ``n`` counts **cells**, so a lattice has ``n + 1`` nodes per axis and ``domain.grid["shape"]``
        is ``n + 1`` — consistent with the ``nx``/``ny``/``nz`` of every other grid in jNO. Pass a
        scalar for every axis, a tuple per axis, or nothing at all to derive it from the shape's own
        ``size=`` (``n = max(2, round(extent / h))``), which keeps one resolution concept.

        Refuses by name -- a CSG plan, a non-rectangular primitive, a graded ``size=`` callable --
        rather than falling back to gmsh: a caller who then reads ``domain.grid`` or expects hexes
        would fail somewhere else instead, having silently solved on a different mesh.

        Meshing is a property of the shape, which is why this lives beside :meth:`quad` and
        :meth:`curved` rather than on the domain -- by the time a domain exists it holds a mesh and
        everything derived from it, and a mesh loaded from a file has no geometry to rebuild from.
        """
        if n is None:
            counts: tuple = ()
        else:
            axes = (n,) * int(self.dim) if isinstance(n, (int, np.integer)) else tuple(n)
            if len(axes) != int(self.dim):
                raise ValueError(
                    f"Shape.structured(n={n!r}): expected {self.dim} cell counts for a {self.dim}-D "
                    f"shape (or one scalar for all axes), got {len(axes)}."
                )
            counts = tuple(int(a) for a in axes)
            if any(a < 1 for a in counts):
                raise ValueError(f"Shape.structured(n={n!r}): every axis needs at least one cell.")
        return replace(self, _structured=counts)

    def cell_choices(self) -> FrozenSet[str]:
        """Every explicit cell choice (:meth:`quad` / :meth:`tri`) anywhere in this build plan.

        More than one means a **mixed-cell** mesh was asked for. gmsh can produce one — recombination
        is per-surface, and in 2-D a quad region and a triangle region sharing an edge conform
        node-for-node because both have 2-node edges there — but jNO's assembler is built around a
        single element table and a single cell array, so the mesh could be built and not assembled.
        :func:`emit.build` refuses on that rather than quietly meshing everything one way.
        """
        seen: set = set()

        def walk(shape):
            if getattr(shape, "_cell", None) is not None:
                seen.add(shape._cell)
            node = shape._node
            kind = node[0]
            if kind in ("cut", "fuse", "inter"):
                walk(node[1])
                walk(node[2])
            elif kind == "regions":
                for _name, sub in node[1]:
                    walk(sub)
            elif kind in ("extrude", "revolve", "sweep", "fillet", "translate", "rotate"):
                walk(node[1])

        walk(self)
        return frozenset(seen)

    # ----- introspection (pure; used by emit + selection) ------------------------
    def leaves(self, _inherit: Size = None) -> Tuple[Tuple[object, Size, int], ...]:
        """Flat ``(primitive, size, key)`` list of every primitive in the plan.
        A ``size`` set on a compound shape (``(a - b).sized(h)``, ``core.name("core").sized(h)``)
        is inherited by every primitive leaf underneath it that has no size of its own — so
        per-region ``.sized()`` on a CSG region actually reaches the mesher's per-primitive
        Distance/Threshold fields instead of being silently dropped. A leaf's own size wins."""
        size = self._size if self._size is not None else _inherit
        node = self._node
        kind = node[0]
        if kind == "leaf":
            prim, key = node[1], node[2]
            return ((prim, size, key),)
        if kind in ("cut", "fuse", "inter"):
            return node[1].leaves(size) + node[2].leaves(size)
        if kind in ("extrude", "revolve", "translate", "rotate", "fillet", "sweep"):
            return node[1].leaves(size)
        if kind == "regions":
            out: Tuple[Tuple[object, Size, int], ...] = ()
            for _name, sub in node[1]:
                out += sub.leaves(size)
            return out
        raise ValueError(f"unknown node kind {kind!r}")

    def keys(self) -> FrozenSet[int]:
        return frozenset(k for _, _, k in self.leaves())

    def contains(self, points, tol: float = 1e-9):
        """Boolean mask over ``points`` (shape ``(N, dim)``): which lie inside this shape.

        Analytic CSG membership evaluated host-side with numpy — primitive leaves combined by ``-``
        (cut), ``|`` (fuse), ``&`` (inter) — so it needs no gmsh and works in 2-D and 3-D alike. This is
        the shapely-free point-in-region test that resolves a geometric ``domain.region(name, shape)`` to
        a mesh-node subset for a subdomain / domain-decomposition solve. Inclusive within ``tol`` (the
        analogue of shapely's ``buffer(1e-9)``), so nodes exactly on a face count as inside.
        ``translate``/``rotate``/``extrude``/``revolve`` are handled by mapping the query point into
        the child's frame; ``sweep`` and ``fillet`` have no closed form and raise
        :class:`NotImplementedError` (sample those through :meth:`sample_interior`, which falls back
        to the boundary tessellation)."""
        pts = np.asarray(points, dtype=float)
        return _node_contains(self._node, pts, float(tol))

    def bounds(self):
        """Axis-aligned bounding box as ``(lo, hi)``, two 3-vectors — no gmsh.

        Always a **superset** of the true extent, which is what makes it a valid proposal box for
        the rejection sampler in :meth:`sample_interior`. Raises for a ``sweep``/``fillet`` plan,
        whose extent is only known from its tessellation.
        """
        lo, hi = _node_bounds(self._node)
        return tuple(float(v) for v in lo), tuple(float(v) for v in hi)

    def is_analytic(self) -> bool:
        """Whether this plan can be sampled with no gmsh at all.

        All three of extent, membership and a boundary sampler have to be closed-form: a shape that
        knows where it is but cannot produce a point on its own surface is no use to a PINN, which
        needs boundary collocation as much as interior. A plan that fails any of them keeps the
        eager meshing path rather than being half-served.
        """
        try:
            _node_bounds(self._node)
            _node_contains(self._node, np.zeros((1, 3)), 0.0)
            _node_boundary_pieces(self._node)
        except NotImplementedError:
            return False
        return True

    def sample_interior(self, n: int, rng=None, tol: float = 0.0, max_rounds: int = 10_000):
        """``(n, 3)`` points drawn uniformly from the interior — the mesh-free collocation draw.

        Rejection sampling in the analytic bounding box, so points are *continuous*: they are not
        drawn from any fixed node set and two calls give different points. A 1-D shape has no
        volume to reject into and is parametrised along its arclength instead (see
        :meth:`jno.geometry.primitives.Curve.sample_interior`).

        Falls back to the boundary tessellation for a ``sweep``/``fillet`` plan; that path logs
        what it did rather than pretending it stayed analytic.
        """
        rng = np.random.default_rng() if rng is None else rng
        n = int(n)
        if n <= 0:
            return np.zeros((0, 3))
        if self.dim == 1:
            leaves = [prim for prim, _size, _key in self.leaves()]
            if len(leaves) != 1 or not hasattr(leaves[0], "sample_interior"):
                raise NotImplementedError(
                    f"a 1-D shape is sampled along its own arclength, which needs exactly one "
                    f"curve primitive; this plan has {len(leaves)}. Build it with jno.Path(...)."
                )
            return leaves[0].sample_interior(n, rng)
        lo, hi = (np.asarray(v, dtype=float) for v in self.bounds())
        span = hi - lo
        free = span > 0.0  # a 2-D shape is flat in z: propose only in the axes that have extent
        out = []
        have = 0
        for _ in range(max_rounds):
            batch = max(4 * (n - have), 256)
            cand = np.tile(lo, (batch, 1))
            cand[:, free] += rng.uniform(0.0, 1.0, size=(batch, int(free.sum()))) * span[free]
            keep = cand[_node_contains(self._node, cand, tol)]
            if len(keep):
                out.append(keep)
                have += len(keep)
            if have >= n:
                return np.concatenate(out, axis=0)[:n]
        raise RuntimeError(
            f"sample_interior drew {max_rounds} rounds without reaching {n} points (got {have}). "
            f"The shape occupies a vanishing fraction of its bounding box {self.bounds()}, or it "
            f"is empty."
        )

    def sample_boundary(self, n: int, rng=None, max_rounds: int = 10_000):
        """``(points (n,3), normals (n,3))`` uniform by measure on the boundary, normals outward.

        Points land **exactly** on the analytic boundary — on the true circle, not on a chord — and
        the normal is the closed-form one, so a disk's normal is exactly radial rather than
        piecewise-constant per facet.

        Candidates come from each primitive's own boundary, weighted by measure; a candidate
        survives only if the shape actually changes across it (``contains`` differs a hair either
        side), which is what trims the parts of a leaf's boundary that a boolean cut away and what
        orients the normal outward — no winding convention is assumed.
        """
        rng = np.random.default_rng() if rng is None else rng
        n = int(n)
        if n <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        pieces = _node_boundary_pieces(self._node)
        weights = np.array([m for _prim, m in pieces], dtype=float)
        if not len(pieces) or weights.sum() <= 0.0:
            raise NotImplementedError(
                f"no analytic boundary is available for this plan; its boundary comes from the "
                f"tessellation instead (Shape.tessellate)."
            )
        lo, hi = (np.asarray(v, dtype=float) for v in self.bounds())
        eps = 1e-9 * max(float(np.max(hi - lo)), 1.0)
        p_out, n_out, have = [], [], 0
        for _ in range(max_rounds):
            batch = max(4 * (n - have), 256)
            idx = rng.choice(len(pieces), size=batch, p=weights / weights.sum())
            pts = np.zeros((batch, 3))
            nrm = np.zeros((batch, 3))
            for i, (prim, _m) in enumerate(pieces):
                sel = idx == i
                if sel.any():
                    pts[sel], nrm[sel] = prim.sample_boundary(int(sel.sum()), rng)
            inside_pos = _node_contains(self._node, pts + eps * nrm, 0.0)
            inside_neg = _node_contains(self._node, pts - eps * nrm, 0.0)
            on_surface = inside_pos ^ inside_neg  # the shape changes across a real boundary point
            flip = np.where(inside_pos & on_surface, -1.0, 1.0)[:, None]
            if on_surface.any():
                p_out.append(pts[on_surface])
                n_out.append((nrm * flip)[on_surface])
                have += int(on_surface.sum())
            if have >= n:
                return np.concatenate(p_out)[:n], np.concatenate(n_out)[:n]
        raise RuntimeError(
            f"sample_boundary drew {max_rounds} rounds without reaching {n} points (got {have})."
        )

    # ----- boundary selection ----------------------------------------------------
    def edge(self, name: str) -> Selector:
        """Select boundary entities carrying auto-name ``name`` (``"top"``, ``"left"``, ...)."""
        return Selector(names=frozenset({name}))

    def edges_from(self, sub: "Shape") -> Selector:
        """Select boundary entities that originate from the primitive(s) in ``sub``."""
        return Selector(keys=sub.keys())

    # ----- realization -----------------------------------------------------------
    def build(self, *, algorithm=None, threads=None):
        """Mesh this shape -> ``(meshio.Mesh, dim, ds)``. Imports gmsh lazily.

        ``algorithm`` selects gmsh's meshing kernel and ``threads`` its thread count. Which kernel
        it applies to follows from this shape's own dimension, so there is no separate 2-D/3-D
        argument. ``None`` keeps jNO's defaults -- see :mod:`jno.geometry.emit`, which records why
        each was chosen; the 3-D default is HXT on 8 threads, measured 14.5x faster than gmsh's
        serial Delaunay at equal-or-better element quality.
        """
        from .emit import build as _build

        return _build(self, algorithm=algorithm, threads=threads, order=self._mesh_order)

    def __call__(self, geo=None):
        # Callable-constructor compatibility (jno.domain runs constructor()).
        return self.build()

    def domain(self, **kwargs):
        """Build a ``jno.domain`` from this shape as a one-liner.

        Forwards the *domain* keyword arguments (``time=``, ``sample=``, ``name=``, ...) to
        ``jno.domain`` -- but **not** a constructor or ``mesh_size``: the mesh size lives on the
        shape itself (via ``size=`` / :meth:`sized`). So ``Shape.rect(0, 0, 1, 1, size=0.1).domain()``
        replaces ``jno.domain(Shape.rect(0, 0, 1, 1, size=0.1))``, and batching still composes as
        ``B * Shape.rect(...).domain()``.
        """
        import jno

        return jno.domain(self, **kwargs)
