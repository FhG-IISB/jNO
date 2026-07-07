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
from typing import ClassVar, Optional

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
