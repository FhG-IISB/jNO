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

import numpy as np

# How close a boundary point must be to an analytic curve/surface to lie "on" it.
TOL = 1e-6


def _to3(a):
    """Right-pad an ``(N, k)`` array (k <= 3) with zeros to ``(N, 3)`` — the ambient frame everything
    analytic works in, so a 2-D primitive and a 3-D one compose under the same transforms."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] >= 3:
        return a[:, :3]
    out = np.zeros((a.shape[0], 3), dtype=float)
    out[:, : a.shape[1]] = a
    return out


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

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) — which rows of ``pts`` (N, ≥2) lie inside, inclusive within ``tol``."""
        x, y = pts[:, 0], pts[:, 1]
        return (x >= self.x0 - tol) & (x <= self.x1 + tol) & (y >= self.y0 - tol) & (y <= self.y1 + tol)

    def bounds(self):
        """Axis-aligned bounding box ``(lo, hi)``, both 3-vectors (z is 0 for a 2-D primitive)."""
        return (self.x0, self.y0, 0.0), (self.x1, self.y1, 0.0)

    def boundary_measure(self) -> float:
        """Perimeter (2-D) / surface area (3-D) — the weight for drawing across a composite."""
        return 2.0 * ((self.x1 - self.x0) + (self.y1 - self.y0))

    def sample_boundary(self, n: int, rng):
        """``(points (n,3), normals (n,3))`` drawn uniformly by arclength, exactly on the edges."""
        w, h = self.x1 - self.x0, self.y1 - self.y0
        # edge order matches `classify`: bottom, right, top, left
        lengths = np.array([w, h, w, h], dtype=float)
        origins = np.array([[self.x0, self.y0], [self.x1, self.y0], [self.x1, self.y1], [self.x0, self.y1]])
        tangents = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
        normals = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        e = rng.choice(4, size=n, p=lengths / lengths.sum())
        t = rng.uniform(0.0, 1.0, size=n)[:, None]
        xy = origins[e] + t * (tangents[e] * lengths[e][:, None])
        return _to3(xy), _to3(normals[e])


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

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) — which rows of ``pts`` (N, ≥2) lie inside, inclusive within ``tol``."""
        return (pts[:, 0] - self.cx) ** 2 + (pts[:, 1] - self.cy) ** 2 <= (self.r + tol) ** 2

    def bounds(self):
        """Axis-aligned bounding box ``(lo, hi)``, both 3-vectors (z is 0 for a 2-D primitive)."""
        return (self.cx - self.r, self.cy - self.r, 0.0), (self.cx + self.r, self.cy + self.r, 0.0)

    def boundary_measure(self) -> float:
        """Circumference — the weight for drawing across a composite boundary."""
        return 2.0 * math.pi * self.r

    def sample_boundary(self, n: int, rng):
        """``(points, normals)`` exactly on the circle: uniform in angle, normal = radial."""
        th = rng.uniform(0.0, 2.0 * math.pi, size=n)
        nrm = np.stack([np.cos(th), np.sin(th)], axis=1)
        return _to3(np.array([self.cx, self.cy]) + self.r * nrm), _to3(nrm)


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

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) — which rows of ``pts`` (N, ≥3) lie inside, inclusive within ``tol``."""
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return (
            (x >= self.x0 - tol)
            & (x <= self.x1 + tol)
            & (y >= self.y0 - tol)
            & (y <= self.y1 + tol)
            & (z >= self.z0 - tol)
            & (z <= self.z1 + tol)
        )

    def bounds(self):
        """Axis-aligned bounding box ``(lo, hi)``, both 3-vectors."""
        return (self.x0, self.y0, self.z0), (self.x1, self.y1, self.z1)

    def _extent(self):
        return self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0

    def boundary_measure(self) -> float:
        """Total surface area — the weight for drawing across a composite boundary."""
        w, d, h = self._extent()
        return 2.0 * (w * d + w * h + d * h)

    def sample_boundary(self, n: int, rng):
        """``(points, normals)`` uniform by area over the six faces, exactly on each plane."""
        w, d, h = self._extent()
        lo = np.array([self.x0, self.y0, self.z0])
        # face order matches `classify`: left, right, front, back, bottom, top
        axis = np.array([0, 0, 1, 1, 2, 2])
        at_hi = np.array([False, True, False, True, False, True])
        areas = np.array([d * h, d * h, w * h, w * h, w * d, w * d], dtype=float)
        f = rng.choice(6, size=n, p=areas / areas.sum())
        ext = np.array([w, d, h], dtype=float)
        pts = lo + rng.uniform(0.0, 1.0, size=(n, 3)) * ext  # uniform in the box, then pin one axis
        ax = axis[f]
        rows = np.arange(n)
        pts[rows, ax] = np.where(at_hi[f], lo[ax] + ext[ax], lo[ax])
        nrm = np.zeros((n, 3))
        nrm[rows, ax] = np.where(at_hi[f], 1.0, -1.0)
        return pts, nrm


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

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) — inside test by the even-odd crossing rule (ray-cast). Edge-exact points
        are ambiguous (the crossing rule is exclusive on edges); ``tol`` is unused here."""
        x, y = pts[:, 0], pts[:, 1]
        verts = self.points
        n = len(verts)
        inside = np.zeros(x.shape, dtype=bool)
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi)
            inside ^= crosses
            j = i
        return inside

    def bounds(self):
        """Axis-aligned bounding box ``(lo, hi)``, both 3-vectors (z is 0 for a 2-D primitive)."""
        p = np.asarray(self.points, dtype=float)
        return (p[:, 0].min(), p[:, 1].min(), 0.0), (p[:, 0].max(), p[:, 1].max(), 0.0)

    def _edges(self):
        """``(a, b, lengths, outward_normals)`` for the closed edge loop, outward by winding."""
        p = np.asarray(self.points, dtype=float)
        a, b = p, np.roll(p, -1, axis=0)
        v = b - a
        lengths = np.linalg.norm(v, axis=1)
        t = v / np.maximum(lengths, 1e-300)[:, None]
        # Shoelace: positive area == counter-clockwise, whose outward normal is (ty, -tx).
        signed2 = float(np.sum(a[:, 0] * b[:, 1] - b[:, 0] * a[:, 1]))
        sign = 1.0 if signed2 >= 0.0 else -1.0
        nrm = sign * np.stack([t[:, 1], -t[:, 0]], axis=1)
        return a, b, lengths, nrm

    def boundary_measure(self) -> float:
        """Perimeter — the weight for drawing across a composite boundary."""
        return float(self._edges()[2].sum())

    def sample_boundary(self, n: int, rng):
        """``(points, normals)`` uniform by arclength over the edge loop, exactly on each edge."""
        a, b, lengths, nrm = self._edges()
        keep = lengths > 0.0
        a, b, lengths, nrm = a[keep], b[keep], lengths[keep], nrm[keep]
        e = rng.choice(len(lengths), size=n, p=lengths / lengths.sum())
        t = rng.uniform(0.0, 1.0, size=n)[:, None]
        return _to3(a[e] + t * (b[e] - a[e])), _to3(nrm[e])


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

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) — inside the axial extent [0, |axis|] and within radius ``r`` (± ``tol``)."""
        axis = np.array([self.dx, self.dy, self.dz])
        alen = float(np.linalg.norm(axis))
        u = axis / alen
        w = pts[:, :3] - np.array([self.x, self.y, self.z])
        t = w @ u
        perp2 = np.sum(w * w, axis=1) - t * t
        return (t >= -tol) & (t <= alen + tol) & (perp2 <= (self.r + tol) ** 2)

    def _frame(self):
        """``(base, axis_unit, length, e1, e2)`` — an orthonormal frame with e1, e2 across the axis."""
        base = np.array([self.x, self.y, self.z], dtype=float)
        axis = np.array([self.dx, self.dy, self.dz], dtype=float)
        alen = float(np.linalg.norm(axis))
        u = axis / alen
        # any vector not parallel to u; the smallest component of u is the safest seed
        seed = np.zeros(3)
        seed[int(np.argmin(np.abs(u)))] = 1.0
        e1 = np.cross(u, seed)
        e1 /= np.linalg.norm(e1)
        return base, u, alen, e1, np.cross(u, e1)

    def bounds(self):
        """Exact AABB: the caps' centres widened by the disc's projected radius on each axis."""
        base, u, alen, _, _ = self._frame()
        tip = base + alen * u
        pad = self.r * np.sqrt(np.maximum(1.0 - u**2, 0.0))  # radius of the disc projected per axis
        lo = np.minimum(base, tip) - pad
        hi = np.maximum(base, tip) + pad
        return tuple(lo), tuple(hi)

    def boundary_measure(self) -> float:
        """Lateral area plus both caps."""
        _, _, alen, _, _ = self._frame()
        return 2.0 * math.pi * self.r * alen + 2.0 * math.pi * self.r**2

    def sample_boundary(self, n: int, rng):
        """``(points, normals)`` uniform by area over the side and the two caps."""
        base, u, alen, e1, e2 = self._frame()
        side, cap = 2.0 * math.pi * self.r * alen, math.pi * self.r**2
        areas = np.array([side, cap, cap])  # side, bottom, top
        f = rng.choice(3, size=n, p=areas / areas.sum())
        th = rng.uniform(0.0, 2.0 * math.pi, size=n)
        radial = np.cos(th)[:, None] * e1 + np.sin(th)[:, None] * e2
        # side: full radius, free axial position. caps: sqrt-warped radius for area-uniformity.
        r_cap = self.r * np.sqrt(rng.uniform(0.0, 1.0, size=n))
        rad = np.where(f[:, None] == 0, self.r * radial, r_cap[:, None] * radial)
        axial = np.where(f == 0, rng.uniform(0.0, alen, size=n), np.where(f == 1, 0.0, alen))
        pts = base + axial[:, None] * u + rad
        nrm = np.where(f[:, None] == 0, radial, np.where(f[:, None] == 1, -u, u))
        return pts, nrm


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

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) — which rows of ``pts`` (N, ≥3) lie inside, inclusive within ``tol``."""
        c = np.array([self.cx, self.cy, self.cz])
        return np.sum((pts[:, :3] - c) ** 2, axis=1) <= (self.r + tol) ** 2

    def bounds(self):
        """Axis-aligned bounding box ``(lo, hi)``, both 3-vectors."""
        c, r = np.array([self.cx, self.cy, self.cz]), self.r
        return tuple(c - r), tuple(c + r)

    def boundary_measure(self) -> float:
        """Surface area — the weight for drawing across a composite boundary."""
        return 4.0 * math.pi * self.r**2

    def sample_boundary(self, n: int, rng):
        """``(points, normals)`` uniform on the sphere (normalised Gaussians — Marsaglia 1972)."""
        g = rng.normal(size=(n, 3))
        nrm = g / np.linalg.norm(g, axis=1, keepdims=True)
        return np.array([self.cx, self.cy, self.cz]) + self.r * nrm, nrm


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


def _arc_circle(a, through, b):
    """``(centre, radius, plane_normal, e1, e2, span)`` for the 3-point circular arc ``a->through->b``.

    ``e1`` points from the centre to ``a``, so the arc runs over angles ``[0, span]`` in the
    ``(e1, e2)`` frame -- the same "is ``through`` reached before ``b``" rule :func:`_on_arc` uses,
    lifted to 3-D so it serves arcs of a ``Curve`` as well as a planar ``Contour``.
    Returns ``None`` when the three points are collinear (no circle).
    """
    a, through, b = (np.asarray(p, dtype=float) for p in (a, through, b))
    u, v = through - a, b - a
    w = np.cross(u, v)
    w2 = float(w @ w)
    if w2 < 1e-24:
        return None  # collinear
    centre = a + (float(u @ u) * np.cross(v, w) + float(v @ v) * np.cross(w, u)) / (2.0 * w2)
    radius = float(np.linalg.norm(a - centre))
    normal = w / math.sqrt(w2)
    e1 = (a - centre) / radius
    e2 = np.cross(normal, e1)

    def ang(p):
        d = np.asarray(p, dtype=float) - centre
        return math.atan2(float(d @ e2), float(d @ e1)) % (2.0 * math.pi)

    a_t, a_b = ang(through), ang(b)
    # `through` reached before `b` going CCW => the arc is [0, a_b]; otherwise it is the complement,
    # which we represent by flipping the frame so the span is again measured CCW from `a`.
    if a_t > a_b:
        e2, a_b = -e2, 2.0 * math.pi - a_b
    return centre, radius, normal, e1, e2, a_b


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

    def _pieces(self):
        """``[(kind, payload, length)]`` per segment: ``("line", (a, b), L)`` or ``("arc", circ, L)``."""
        out = []
        prev = np.asarray(self.start, dtype=float)
        for seg in self.segments:
            end = np.asarray(seg[1], dtype=float)
            if seg[0] == "line":
                out.append(("line", (prev, end), float(np.linalg.norm(end - prev))))
            else:
                circ = _arc_circle(prev, seg[2], end)
                if circ is None:  # three collinear points describe a straight segment
                    out.append(("line", (prev, end), float(np.linalg.norm(end - prev))))
                else:
                    out.append(("arc", circ, circ[1] * circ[5]))
            prev = end
        return out

    def measure(self) -> float:
        """Total arclength — a 1-D domain's ``volume``."""
        return float(sum(L for _, _, L in self._pieces()))

    def bounds(self):
        """AABB. An arc contributes its **full circle's** box, which is a superset of the arc's
        own extent -- never a subset, so a rejection proposal built on it stays valid."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for kind, payload, _ in self._pieces():
            if kind == "line":
                pts = np.stack(payload)
            else:
                centre, radius, normal, _, _, _ = payload
                pad = radius * np.sqrt(np.maximum(1.0 - normal**2, 0.0))
                pts = np.stack([centre - pad, centre + pad])
            lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
        return tuple(lo), tuple(hi)

    def _axis_only(self, k: int, tol: float):
        """True if the curve lies in the subspace spanned by the first ``k`` coordinates, so a
        query giving only those ``k`` columns is exact rather than quietly mismatched."""
        lo, hi = self.bounds()
        return all(abs(lo[i]) < tol and abs(hi[i]) < tol for i in range(k, 3))

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) — which points lie **on** the 1-D manifold (it has no volume)."""
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] < 3 and not self._axis_only(pts.shape[1], max(tol, 1e-9)):
            raise NotImplementedError(
                f"Curve.contains got {pts.shape[1]}-column points but this curve leaves that "
                f"subspace (bounds {self.bounds()}). A curved or off-axis 1-D manifold cannot be "
                f"tested from its first {pts.shape[1]} coordinate(s); pass full 3-D points."
            )
        p = _to3(pts)
        hit = np.zeros(len(p), dtype=bool)
        for kind, payload, _ in self._pieces():
            if kind == "line":
                a, b = payload
                v = b - a
                L2 = float(v @ v)
                if L2 <= 0.0:
                    continue
                t = np.clip(((p - a) @ v) / L2, 0.0, 1.0)
                hit |= np.linalg.norm(p - (a + t[:, None] * v), axis=1) <= tol
            else:
                centre, radius, normal, e1, e2, span = payload
                d = p - centre
                on_plane = np.abs(d @ normal) <= tol
                c1, c2 = d @ e1, d @ e2
                on_circle = np.abs(np.hypot(c1, c2) - radius) <= tol
                ang = np.mod(np.arctan2(c2, c1), 2.0 * math.pi)
                atol = tol / max(radius, tol)  # angular slack matching the positional tolerance
                hit |= on_plane & on_circle & ((ang <= span + atol) | (ang >= 2.0 * math.pi - atol))
        return hit

    def sample_interior(self, n: int, rng):
        """``(n, 3)`` points drawn uniformly by **arclength** along the manifold.

        A 1-D domain has no volume to reject into, so this parametrises rather than rejects --
        which also makes it exact, with every point lying on the curve to machine precision.
        """
        pieces = self._pieces()
        w = np.array([L for _, _, L in pieces], dtype=float)
        idx = rng.choice(len(pieces), size=n, p=w / w.sum())
        t = rng.uniform(0.0, 1.0, size=n)
        out = np.zeros((n, 3))
        for i, (kind, payload, _) in enumerate(pieces):
            sel = idx == i
            if not sel.any():
                continue
            if kind == "line":
                a, b = payload
                out[sel] = a + t[sel, None] * (b - a)
            else:
                centre, radius, _, e1, e2, span = payload
                ang = t[sel] * span
                out[sel] = centre + radius * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
        return out

    def boundary_measure(self) -> float:
        """A 1-D manifold's boundary is its two endpoints — counted, not measured."""
        return 2.0

    def sample_boundary(self, n: int, rng):
        """``(points, normals)`` over the two ends; the normal is the outward unit tangent (-1/+1
        for the usual straight interval)."""
        pieces = self._pieces()

        def tangent(kind, payload, at_start):
            if kind == "line":
                a, b = payload
                v = b - a
                return v / np.linalg.norm(v)
            centre, radius, _, e1, e2, span = payload
            ang = 0.0 if at_start else span
            return -np.sin(ang) * e1 + np.cos(ang) * e2  # d/dang of the arc parametrisation

        ends = np.stack([np.asarray(self.start, dtype=float), np.asarray(self._end(), dtype=float)])
        nrm = np.stack([-tangent(*pieces[0][:2], True), tangent(*pieces[-1][:2], False)])
        pick = rng.integers(0, 2, size=n)
        return ends[pick], nrm[pick]


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

    def _pieces(self):
        """``[(kind, payload, length, name)]`` around the closed loop, lifted to the z=0 plane."""
        out = []
        prev = (self.start[0], self.start[1], 0.0)
        for i, seg in enumerate(self._effective()):
            end = (seg[1][0], seg[1][1], 0.0)
            name = seg[-1] if isinstance(seg[-1], str) else f"e{i}"
            if seg[0] == "line":
                a, b = np.asarray(prev), np.asarray(end)
                out.append(("line", (a, b), float(np.linalg.norm(b - a)), name))
            else:
                circ = _arc_circle(prev, (seg[2][0], seg[2][1], 0.0), end)
                if circ is None:
                    a, b = np.asarray(prev), np.asarray(end)
                    out.append(("line", (a, b), float(np.linalg.norm(b - a)), name))
                else:
                    out.append(("arc", circ, circ[1] * circ[5], name))
            prev = end
        return out

    def bounds(self):
        """AABB; an arc contributes its full circle's box, a superset of the arc's own extent."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for kind, payload, _, _ in self._pieces():
            if kind == "line":
                pts = np.stack(payload)
            else:
                centre, radius, normal, _, _, _ = payload
                pad = radius * np.sqrt(np.maximum(1.0 - normal**2, 0.0))
                pts = np.stack([centre - pad, centre + pad])
            lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
        lo[2] = hi[2] = 0.0
        return tuple(lo), tuple(hi)

    def contains(self, pts, tol: float = TOL):
        """Boolean mask (N,) by even-odd ray casting along +x, with arcs crossed **analytically**
        (circle-line roots restricted to the arc's angular span) rather than polygonised."""
        p = _to3(pts)
        x, y = p[:, 0], p[:, 1]
        crossings = np.zeros(len(p), dtype=np.int64)
        for kind, payload, _, _ in self._pieces():
            if kind == "line":
                a, b = payload
                (ax, ay), (bx, by) = a[:2], b[:2]
                straddles = (ay > y) != (by > y)
                with np.errstate(divide="ignore", invalid="ignore"):
                    xint = (bx - ax) * (y - ay) / (by - ay) + ax
                crossings += (straddles & (x < xint)).astype(np.int64)
            else:
                centre, radius, _, e1, e2, span = payload
                dy = y - centre[1]
                disc = radius**2 - dy**2
                ok = disc >= 0.0
                root = np.sqrt(np.maximum(disc, 0.0))
                for sgn in (+1.0, -1.0):
                    xi = centre[0] + sgn * root
                    d = np.stack([xi - centre[0], dy, np.zeros_like(dy)], axis=1)
                    ang = np.mod(np.arctan2(d @ e2, d @ e1), 2.0 * math.pi)
                    # half-open in angle so the shared endpoint of two pieces is counted once
                    crossings += (ok & (x < xi) & (ang < span)).astype(np.int64)
        return (crossings % 2) == 1

    def boundary_measure(self) -> float:
        """Total arclength of the closed loop."""
        return float(sum(L for _, _, L, _ in self._pieces()))

    def sample_boundary(self, n: int, rng):
        """``(points, normals)`` uniform by arclength, exactly on each line/arc, normals outward.

        Orientation is settled by probing ``contains`` a hair either side rather than by trusting a
        winding convention, so it holds for arcs and for either traversal direction.
        """
        pieces = self._pieces()
        w = np.array([L for _, _, L, _ in pieces], dtype=float)
        idx = rng.choice(len(pieces), size=n, p=w / w.sum())
        t = rng.uniform(0.0, 1.0, size=n)
        pts = np.zeros((n, 3))
        nrm = np.zeros((n, 3))
        for i, (kind, payload, _, _) in enumerate(pieces):
            sel = idx == i
            if not sel.any():
                continue
            if kind == "line":
                a, b = payload
                v = b - a
                pts[sel] = a + t[sel, None] * v
                tang = v / np.linalg.norm(v)
                nrm[sel] = np.array([tang[1], -tang[0], 0.0])
            else:
                centre, radius, _, e1, e2, span = payload
                ang = t[sel] * span
                ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
                pts[sel] = centre + radius * (ca * e1 + sa * e2)
                tang = -sa * e1 + ca * e2
                nrm[sel] = np.stack([tang[:, 1], -tang[:, 0], np.zeros(len(tang))], axis=1)
        lo, hi = self.bounds()
        eps = 1e-7 * max(float(np.max(np.asarray(hi) - np.asarray(lo))), 1.0)
        if np.mean(self.contains(pts + eps * nrm, tol=0.0)) > 0.5:  # pointing in -> flip the loop
            nrm = -nrm
        return pts, nrm
