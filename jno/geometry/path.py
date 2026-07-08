"""``Path`` -- a fluent 2-D contour builder of line and arc segments.

Chain ``.line_to`` / ``.arc_to`` from a start point, then ``.face()`` to turn the closed
contour into a 2-D :class:`~jno.geometry.shape.Shape` region. Segments can be named as you
draw them (``name=``, default ``e0, e1, ...``). This is the general 2-D generator: ``polygon``
is an all-line contour, and a diameter + semicircular arc revolves into a sphere.

Immutable: each call returns a new ``Path`` (build plans stay shareable and side-effect free).
"""

from __future__ import annotations

from typing import Optional, Tuple


class Path:
    """A 2-D contour under construction. Start at a point, then chain segments."""

    __slots__ = ("_start", "_segs")

    def __init__(self, x: float, y: float):
        self._start: Tuple[float, float] = (float(x), float(y))
        self._segs: Tuple[tuple, ...] = ()

    @classmethod
    def _extend(cls, start, segs) -> "Path":
        p = cls.__new__(cls)
        p._start = start
        p._segs = segs
        return p

    def line_to(self, x: float, y: float, name: Optional[str] = None) -> "Path":
        """Straight segment to ``(x, y)``."""
        return Path._extend(self._start, self._segs + (("line", (float(x), float(y)), name),))

    def arc_to(self, x: float, y: float, through, name: Optional[str] = None) -> "Path":
        """Circular arc to ``(x, y)`` passing through the point ``through`` (a 3-point arc)."""
        seg = ("arc", (float(x), float(y)), (float(through[0]), float(through[1])), name)
        return Path._extend(self._start, self._segs + (seg,))

    def face(self, name: str = "interior", size=None):
        """Close the contour and return it as a 2-D :class:`~jno.geometry.shape.Shape`."""
        from . import shape as _shape
        from .primitives import Contour

        contour = Contour(self._start, self._segs)
        return _shape.Shape(("leaf", contour, next(_shape._LEAF_KEYS)), 2, size)
