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
from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Optional, Tuple, Union

import numpy as np

from .primitives import Box, Cylinder, Disk, Polygon, Rect, Sphere


def _node_contains(node, pts, tol):
    """Analytic point membership over the CSG tree — leaves + cut/fuse/inter, no gmsh."""
    kind = node[0]
    if kind == "leaf":
        return np.asarray(node[1].contains(pts, tol), dtype=bool)
    if kind == "cut":
        return _node_contains(node[1]._node, pts, tol) & ~_node_contains(node[2]._node, pts, tol)
    if kind == "fuse":
        return _node_contains(node[1]._node, pts, tol) | _node_contains(node[2]._node, pts, tol)
    if kind == "inter":
        return _node_contains(node[1]._node, pts, tol) & _node_contains(node[2]._node, pts, tol)
    if kind == "regions":
        mask = np.zeros(len(pts), dtype=bool)
        for _name, sub in node[1]:
            mask |= _node_contains(sub._node, pts, tol)
        return mask
    raise NotImplementedError(
        f"Shape.contains supports the analytic CSG subset — primitive leaves combined by "
        f"'-'/'|'/'&' (cut/fuse/inter); a {kind!r} solid (extrude/revolve/sweep/fillet/translate/"
        f"rotate) has no closed-form point membership. Tag that region another way, or add its "
        f"inverse transform here."
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
    """An immutable geometry build-plan. Operators return new shapes."""

    _node: tuple
    dim: int
    _size: Size = field(default=None, compare=False)
    _region_name: Optional[str] = field(default=None, compare=False)
    _mesh_order: int = field(default=1, compare=False)
    # Attached material properties, ``{name: value}``. ``compare=False`` keeps the dataclass hashable
    # (a dict is not) and keeps two geometrically identical shapes equal regardless of their materials.
    _attach: Optional[dict] = field(default=None, compare=False)

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
        return Shape(self._node, self.dim, self._size, str(name), self._mesh_order, self._attach)

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
        return Shape(self._node, self.dim, self._size, self._region_name, self._mesh_order, merged)

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
        return Shape(
            ("cut", self, other), self.dim, self._size, None, max(self._mesh_order, other._mesh_order), self._attach
        )

    def __or__(self, other: "Shape") -> "Shape":
        return Shape(
            ("fuse", self, other), self.dim, self._size, None, max(self._mesh_order, other._mesh_order), self._attach
        )

    def __and__(self, other: "Shape") -> "Shape":
        return Shape(
            ("inter", self, other), self.dim, self._size, None, max(self._mesh_order, other._mesh_order), self._attach
        )

    # ----- transforms ------------------------------------------------------------
    def extrude(self, height: float) -> "Shape":
        if self.dim != 2:
            raise ValueError("extrude requires a 2-D shape")
        return Shape(("extrude", self, float(height)), 3, self._size, None, self._mesh_order, self._attach)

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
        return Shape(("revolve", self, ap, ad, float(angle)), 3, self._size, None, self._mesh_order, self._attach)

    def translate(self, vector) -> "Shape":
        """Move the shape by ``vector`` (2- or 3-component). Boundary names are preserved."""
        v = tuple(float(c) for c in vector)
        if len(v) == 2:
            v = (v[0], v[1], 0.0)
        return Shape(("translate", self, v), self.dim, self._size, None, self._mesh_order, self._attach)

    def rotate(self, axis_point, axis_dir, angle: float) -> "Shape":
        """Rotate ``angle`` radians about the axis through ``axis_point`` along ``axis_dir``."""
        ap = tuple(float(c) for c in axis_point)
        ad = tuple(float(c) for c in axis_dir)
        return Shape(("rotate", self, ap, ad, float(angle)), self.dim, self._size, None, self._mesh_order, self._attach)

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
        return Shape(("sweep", self, path), 3, self._size, None, self._mesh_order, self._attach)

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
        return Shape(("fillet", self, float(radius), where), self.dim, self._size, None, self._mesh_order, self._attach)

    def sized(self, size: Size) -> "Shape":
        """Return a copy of this shape with its target mesh size set (scalar or ``f(x,y,z)``)."""
        return Shape(self._node, self.dim, size, self._region_name, self._mesh_order, self._attach)

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
        return Shape(self._node, self.dim, self._size, self._region_name, int(order), self._attach)

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
        analogue of shapely's ``buffer(1e-9)``), so nodes exactly on a face count as inside. A shape
        carrying a non-CSG transform (``extrude``/``revolve``/``sweep``/``fillet``/``translate``/
        ``rotate``) has no closed-form membership and raises :class:`NotImplementedError`."""
        pts = np.asarray(points, dtype=float)
        return _node_contains(self._node, pts, float(tol))

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
