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
from typing import Callable, FrozenSet, Tuple, Union

from .primitives import Box, Cylinder, Disk, Polygon, Rect, Sphere

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

    # ----- boolean operators -----------------------------------------------------
    def __sub__(self, other: "Shape") -> "Shape":
        return Shape(("cut", self, other), self.dim, self._size)

    def __or__(self, other: "Shape") -> "Shape":
        return Shape(("fuse", self, other), self.dim, self._size)

    def __and__(self, other: "Shape") -> "Shape":
        return Shape(("inter", self, other), self.dim, self._size)

    # ----- transforms ------------------------------------------------------------
    def extrude(self, height: float) -> "Shape":
        if self.dim != 2:
            raise ValueError("extrude requires a 2-D shape")
        return Shape(("extrude", self, float(height)), 3, self._size)

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
        return Shape(("revolve", self, ap, ad, float(angle)), 3, self._size)

    def translate(self, vector) -> "Shape":
        """Move the shape by ``vector`` (2- or 3-component). Boundary names are preserved."""
        v = tuple(float(c) for c in vector)
        if len(v) == 2:
            v = (v[0], v[1], 0.0)
        return Shape(("translate", self, v), self.dim, self._size)

    def rotate(self, axis_point, axis_dir, angle: float) -> "Shape":
        """Rotate ``angle`` radians about the axis through ``axis_point`` along ``axis_dir``."""
        ap = tuple(float(c) for c in axis_point)
        ad = tuple(float(c) for c in axis_dir)
        return Shape(("rotate", self, ap, ad, float(angle)), self.dim, self._size)

    def sweep(self, path) -> "Shape":
        """Sweep this 2-D profile along an open :class:`~jno.geometry.path.Path` trajectory.

        The path must be smooth (line and arc segments); a sharp line->line corner is rejected
        up front (it self-intersects the swept profile). Naming degrades to ``interior`` +
        ``boundary`` for a general sweep -- carve regions afterwards with ``d.tag(predicate)``.
        """
        if self.dim != 2:
            raise ValueError("sweep requires a 2-D profile")
        path._check_sweepable()
        return Shape(("sweep", self, path), 3, self._size)

    def fillet(self, radius: float, where=None) -> "Shape":
        """Round the solid's edges by ``radius``.

        ``where=f(x, y, z)`` selects which edges to round by their midpoint (default: all
        edges). The rounded blend faces are unnamed (they fall into ``boundary``); the flat
        faces keep their names.
        """
        return Shape(("fillet", self, float(radius), where), self.dim, self._size)

    def sized(self, size: Size) -> "Shape":
        """Return a copy of this shape with its target mesh size set (scalar or ``f(x,y,z)``)."""
        return Shape(self._node, self.dim, size)

    # ----- introspection (pure; used by emit + selection) ------------------------
    def leaves(self) -> Tuple[Tuple[object, Size, int], ...]:
        """Flat ``(primitive, size, key)`` list of every primitive in the plan."""
        node = self._node
        kind = node[0]
        if kind == "leaf":
            prim, key = node[1], node[2]
            return ((prim, self._size, key),)
        if kind in ("cut", "fuse", "inter"):
            return node[1].leaves() + node[2].leaves()
        if kind in ("extrude", "revolve", "translate", "rotate", "fillet", "sweep"):
            return node[1].leaves()
        raise ValueError(f"unknown node kind {kind!r}")

    def keys(self) -> FrozenSet[int]:
        return frozenset(k for _, _, k in self.leaves())

    # ----- boundary selection ----------------------------------------------------
    def edge(self, name: str) -> Selector:
        """Select boundary entities carrying auto-name ``name`` (``"top"``, ``"left"``, ...)."""
        return Selector(names=frozenset({name}))

    def edges_from(self, sub: "Shape") -> Selector:
        """Select boundary entities that originate from the primitive(s) in ``sub``."""
        return Selector(keys=sub.keys())

    # ----- realization -----------------------------------------------------------
    def build(self):
        """Mesh this shape -> ``(meshio.Mesh, dim, ds)``. Imports gmsh lazily."""
        from .emit import build as _build

        return _build(self)

    def __call__(self, geo=None):
        # Callable-constructor compatibility (jno.domain runs constructor()).
        return self.build()
