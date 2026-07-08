"""``Path`` -- a fluent contour/trajectory builder of line and arc segments.

Chain ``.line_to`` / ``.arc_to`` from a start point. Used two ways:

* **closed** -> ``.face()`` turns the contour into a 2-D :class:`~jno.geometry.shape.Shape`
  region (a diameter + arc becomes a half-disk; all-line is a polygon).
* **open** -> passed to ``profile.sweep(path)`` as a 3-D sweep trajectory.

Points are 3-D (``z`` defaults to 0, so 2-D contours read naturally). Segments can be named
(``name=``, default ``e0..eN``). Immutable: each call returns a new ``Path``.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


def _pad3(p):
    p = tuple(float(c) for c in p)
    return p if len(p) == 3 else (p[0], p[1], 0.0)


class Path:
    """A contour/trajectory under construction. Start at a point, then chain segments."""

    __slots__ = ("_start", "_segs")

    def __init__(self, x: float, y: float, z: float = 0.0):
        self._start: Tuple[float, float, float] = (float(x), float(y), float(z))
        self._segs: Tuple[tuple, ...] = ()

    @classmethod
    def _extend(cls, start, segs) -> "Path":
        p = cls.__new__(cls)
        p._start = start
        p._segs = segs
        return p

    def line_to(self, x: float, y: float, z: float = 0.0, name: Optional[str] = None) -> "Path":
        """Straight segment to ``(x, y, z)``."""
        return Path._extend(self._start, self._segs + (("line", (float(x), float(y), float(z)), name),))

    def arc_to(self, x: float, y: float, z: float = 0.0, through=None, name: Optional[str] = None) -> "Path":
        """Circular arc to ``(x, y, z)`` passing through the point ``through`` (a 3-point arc)."""
        if through is None:
            raise ValueError("arc_to requires through=<point on the arc>")
        seg = ("arc", (float(x), float(y), float(z)), _pad3(through), name)
        return Path._extend(self._start, self._segs + (seg,))

    def face(self, name: str = "interior", size=None):
        """Close the contour (using its x,y) and return it as a 2-D :class:`~jno.geometry.shape.Shape`."""
        from . import shape as _shape
        from .primitives import Contour

        start2 = (self._start[0], self._start[1])
        segs2 = tuple(
            (s[0], (s[1][0], s[1][1]), s[2]) if s[0] == "line" else (s[0], (s[1][0], s[1][1]), (s[2][0], s[2][1]), s[3])
            for s in self._segs
        )
        return _shape.Shape(("leaf", Contour(start2, segs2), next(_shape._LEAF_KEYS)), 2, size)

    # ----- trajectory use (for profile.sweep) -----------------------------------
    def _wire(self, occ):
        """Build the open OCC wire of this path's segments (for sweeping)."""
        p_start = occ.addPoint(*self._start)
        prev = p_start
        curves = []
        for seg in self._segs:
            end = occ.addPoint(*seg[1])
            if seg[0] == "line":
                curves.append(occ.addLine(prev, end))
            else:
                thru = occ.addPoint(*seg[2])
                curves.append(occ.addCircleArc(prev, thru, end, center=False))
            prev = end
        return occ.addWire(curves)

    def _as_extrude(self):
        """If this path is a single straight vertical line from z=0, return its height (== an
        extrude); else None. Lets ``sweep`` reuse the extrude engine's rich face-naming."""
        if len(self._segs) != 1 or self._segs[0][0] != "line":
            return None
        s, end = self._start, self._segs[0][1]
        if abs(s[2]) < 1e-9 and abs(end[0] - s[0]) < 1e-9 and abs(end[1] - s[1]) < 1e-9 and end[2] > 1e-9:
            return end[2]
        return None

    def _check_sweepable(self):
        """Reject a sharp line->line corner: sweeping a profile round it self-intersects and hangs."""
        if not self._segs:
            raise ValueError("sweep path is empty")
        pts = [self._start] + [s[1] for s in self._segs]
        kinds = [s[0] for s in self._segs]
        for i in range(1, len(kinds)):
            if kinds[i - 1] == "line" and kinds[i] == "line":
                d0 = _unit(pts[i - 1], pts[i])
                d1 = _unit(pts[i], pts[i + 1])
                if d0 is not None and d1 is not None and _dot(d0, d1) < 0.999:
                    raise ValueError(
                        "sweep path has a sharp line->line corner; round it with arc_to "
                        "(a sharp corner self-intersects the swept profile and cannot be meshed)."
                    )


def _unit(a, b):
    v = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return None if n < 1e-12 else (v[0] / n, v[1] / n, v[2] / n)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
