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
