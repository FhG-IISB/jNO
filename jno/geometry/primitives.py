"""Analytic primitives for the gmsh-OCC :class:`~jno.geometry.shape.Shape` layer.

Each primitive knows two things:

1. ``build(occ)`` -- how to instantiate itself in an OpenCASCADE model (returns a
   ``(dim, tag)`` dimtag).
2. ``classify(x, y, z)`` -- given a point that lies on *some* boundary entity of a
   built model, which of this primitive's named sub-boundaries (if any) does it lie
   on.

(2) is the *support-inheritance* test that survives boolean operations: cutting a
disk from a rectangle splits the flat ``"top"`` edge into two segments and inserts an
``"arc"`` on the disk's circle, and every resulting boundary curve is recovered by
testing a point on it against these analytic predicates -- never by trusting gmsh's
entity ordering or an edge-level ``outDimTagsMap`` (validated empirically before this
layer was written).

Pure Python: this module imports no gmsh (the ``occ`` handle is passed in), so the
naming/selection logic is unit-testable without a mesher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

# How close a boundary point must be to an analytic curve/surface to lie "on" it.
TOL = 1e-6


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle ``[x0, x1] x [y0, y1]``; edges ``left/right/bottom/top``."""

    x0: float
    y0: float
    x1: float
    y1: float
    dim: ClassVar[int] = 2

    def build(self, occ):
        return (2, occ.addRectangle(self.x0, self.y0, 0.0, self.x1 - self.x0, self.y1 - self.y0))

    def classify(self, x: float, y: float, z: float = 0.0, tol: float = TOL) -> Optional[str]:
        if abs(x - self.x0) < tol:
            return "left"
        if abs(x - self.x1) < tol:
            return "right"
        if abs(y - self.y0) < tol:
            return "bottom"
        if abs(y - self.y1) < tol:
            return "top"
        return None


@dataclass(frozen=True)
class Disk:
    """Filled disk centred ``(cx, cy)`` radius ``r``; boundary ``arc``."""

    cx: float
    cy: float
    r: float
    dim: ClassVar[int] = 2

    def build(self, occ):
        return (2, occ.addDisk(self.cx, self.cy, 0.0, self.r, self.r))

    def classify(self, x: float, y: float, z: float = 0.0, tol: float = TOL) -> Optional[str]:
        if abs(math.hypot(x - self.cx, y - self.cy) - self.r) < tol:
            return "arc"
        return None


@dataclass(frozen=True)
class Box:
    """Axis-aligned box; faces ``left/right/front/back/bottom/top``."""

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    dim: ClassVar[int] = 3

    def build(self, occ):
        return (
            3,
            occ.addBox(self.x0, self.y0, self.z0, self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0),
        )

    def classify(self, x: float, y: float, z: float, tol: float = TOL) -> Optional[str]:
        if abs(x - self.x0) < tol:
            return "left"
        if abs(x - self.x1) < tol:
            return "right"
        if abs(y - self.y0) < tol:
            return "front"
        if abs(y - self.y1) < tol:
            return "back"
        if abs(z - self.z0) < tol:
            return "bottom"
        if abs(z - self.z1) < tol:
            return "top"
        return None


@dataclass(frozen=True)
class Polygon:
    """Arbitrary 2-D polygon from ordered vertices; edges auto-named ``e0, e1, ...``."""

    points: Tuple[Tuple[float, float], ...]
    dim: ClassVar[int] = 2

    def build(self, occ):
        tags = [occ.addPoint(px, py, 0.0) for px, py in self.points]
        n = len(tags)
        lines = [occ.addLine(tags[i], tags[(i + 1) % n]) for i in range(n)]
        loop = occ.addCurveLoop(lines)
        return (2, occ.addPlaneSurface([loop]))

    def classify(self, x: float, y: float, z: float = 0.0, tol: float = TOL) -> Optional[str]:
        pts = self.points
        n = len(pts)
        for i in range(n):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % n]
            vx, vy = bx - ax, by - ay
            wx, wy = x - ax, y - ay
            seglen = math.hypot(vx, vy)
            if seglen < tol:
                continue
            # perpendicular distance to the edge line, then parameter along it
            if abs(vx * wy - vy * wx) / seglen < tol:
                t = (wx * vx + wy * vy) / (seglen * seglen)
                if -tol <= t <= 1.0 + tol:
                    return f"e{i}"
        return None


@dataclass(frozen=True)
class Cylinder:
    """Right circular cylinder: base centre ``(x,y,z)``, axis vector ``(dx,dy,dz)``, radius ``r``.

    Faces: lateral ``side``, and the two flat caps ``bottom`` (base) / ``top`` (axis end).
    """

    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    r: float
    dim: ClassVar[int] = 3

    def build(self, occ):
        return (3, occ.addCylinder(self.x, self.y, self.z, self.dx, self.dy, self.dz, self.r))

    def classify(self, x: float, y: float, z: float, tol: float = TOL) -> Optional[str]:
        alen = math.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        ux, uy, uz = self.dx / alen, self.dy / alen, self.dz / alen
        wx, wy, wz = x - self.x, y - self.y, z - self.z
        t = wx * ux + wy * uy + wz * uz  # axial coordinate in [0, alen]
        perp = math.sqrt(max(wx * wx + wy * wy + wz * wz - t * t, 0.0))
        if abs(t) < tol and perp < self.r + tol:
            return "bottom"
        if abs(t - alen) < tol and perp < self.r + tol:
            return "top"
        if abs(perp - self.r) < tol:
            return "side"
        return None


@dataclass(frozen=True)
class Sphere:
    """Sphere centred ``(cx,cy,cz)`` radius ``r``; single face ``surface``."""

    cx: float
    cy: float
    cz: float
    r: float
    dim: ClassVar[int] = 3

    def build(self, occ):
        return (3, occ.addSphere(self.cx, self.cy, self.cz, self.r))

    def classify(self, x: float, y: float, z: float, tol: float = TOL) -> Optional[str]:
        if abs(math.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2 + (z - self.cz) ** 2) - self.r) < tol:
            return "surface"
        return None


def _on_line(a, b, x, y, tol):
    """True if (x, y) lies on the segment a->b."""
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    seglen = math.hypot(vx, vy)
    if seglen < tol:
        return False
    wx, wy = x - ax, y - ay
    if abs(vx * wy - vy * wx) / seglen > tol:  # perpendicular distance to the line
        return False
    t = (wx * vx + wy * vy) / (seglen * seglen)  # parameter along the segment
    return -tol <= t <= 1.0 + tol


def _circumcenter(a, b, c):
    """Centre of the circle through three 2-D points, or None if collinear."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-14:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return (ux, uy)


def _on_arc(a, through, b, x, y, tol):
    """True if (x, y) lies on the circular arc a->through->b (3-point arc)."""
    centre = _circumcenter(a, through, b)
    if centre is None:
        return False
    cx, cy = centre
    r = math.hypot(a[0] - cx, a[1] - cy)
    if abs(math.hypot(x - cx, y - cy) - r) > tol:  # on the supporting circle?
        return False
    two_pi = 2.0 * math.pi

    def rel(px, py):  # angle from the centre, measured from a, in [0, 2pi)
        return (math.atan2(py - cy, px - cx) - math.atan2(a[1] - cy, a[0] - cx)) % two_pi

    d_e, d_t, d_q = rel(*b), rel(*through), rel(x, y)
    atol = 1e-6
    # if `through` is reached before `b` going CCW, the arc is [a, b] CCW; else its complement
    return (d_q <= d_e + atol) if d_t <= d_e else (d_q >= d_e - atol)


@dataclass(frozen=True)
class Curve:
    """An open 1-D contour of line/arc segments -- a 1-D domain (an interval or polyline).

    The 1-D sibling of :class:`Contour`: where ``Contour`` closes into a 2-D region, ``Curve``
    stays open as a 1-D manifold. ``segments`` walks from ``start`` (3-D points); the two overall
    ends are the boundary (auto-named ``left`` = start, ``right`` = final end), while any
    intermediate junctions are *interior* to the 1-D manifold. A single ``("line", ...)`` gives a
    plain interval; ``arc`` segments give a curved 1-D manifold.
    """

    start: Tuple[float, float, float]
    segments: Tuple[tuple, ...]
    dim: ClassVar[int] = 1

    def _end(self):
        return self.segments[-1][1]

    def build(self, occ):
        p0 = occ.addPoint(*self.start)
        prev = p0
        curves = []
        for seg in self.segments:
            end = seg[1]
            pe = occ.addPoint(*end)
            if seg[0] == "line":
                curves.append((1, occ.addLine(prev, pe)))
            else:  # ("arc", end, through, name) -- 3-point circular arc
                pt = occ.addPoint(*seg[2])
                curves.append((1, occ.addCircleArc(prev, pt, pe, center=False)))
            prev = pe
        # 1-D uses no booleans, so returning the first curve suffices: the remaining curves are
        # already in the model (chained through shared points) and get meshed by generate(1).
        return curves[0]

    def classify(self, x: float, y: float, z: float = 0.0, tol: float = TOL) -> Optional[str]:
        s, e = self.start, self._end()
        if math.dist((x, y, z), s) < tol:
            return "left"
        if math.dist((x, y, z), e) < tol:
            return "right"
        return None  # an intermediate junction of a polyline is interior, not boundary


@dataclass(frozen=True)
class Contour:
    """A closed 2-D contour of line and arc segments (generalises :class:`Polygon`).

    ``segments`` is a tuple of ``("line", (ex, ey), name)`` and ``("arc", (ex, ey), (tx, ty),
    name)`` entries walked from ``start``; it auto-closes back to ``start``. Each segment's
    boundary auto-names to ``name`` (or ``e0, e1, ...``). Revolving a contour reuses these
    predicates in meridian coords, so e.g. a diameter + semicircular arc revolves into a sphere.
    """

    start: Tuple[float, float]
    segments: Tuple[tuple, ...]
    dim: ClassVar[int] = 2

    def _effective(self):
        segs = list(self.segments)
        last = segs[-1][1] if segs else self.start
        if math.hypot(last[0] - self.start[0], last[1] - self.start[1]) > 1e-9:
            segs.append(("line", self.start, None))  # auto-close
        return segs

    def build(self, occ):
        p_start = occ.addPoint(self.start[0], self.start[1], 0.0)
        prev = p_start
        curves = []
        for seg in self._effective():
            end = seg[1]
            closing = math.hypot(end[0] - self.start[0], end[1] - self.start[1]) < 1e-9
            pe = p_start if closing else occ.addPoint(end[0], end[1], 0.0)
            if seg[0] == "line":
                curves.append(occ.addLine(prev, pe))
            else:
                thru = seg[2]
                pt = occ.addPoint(thru[0], thru[1], 0.0)
                curves.append(occ.addCircleArc(prev, pt, pe, center=False))
            prev = pe
        loop = occ.addCurveLoop(curves)
        return (2, occ.addPlaneSurface([loop]))

    def classify(self, x: float, y: float, z: float = 0.0, tol: float = TOL) -> Optional[str]:
        prev = self.start
        for i, seg in enumerate(self._effective()):
            end = seg[1]
            name = seg[-1] if isinstance(seg[-1], str) else f"e{i}"
            hit = _on_line(prev, end, x, y, tol) if seg[0] == "line" else _on_arc(prev, seg[2], end, x, y, tol)
            if hit:
                return name
            prev = end
        return None
