"""``jno.fd(...)``: a finite-difference stencil described by its mathematics, for ``scheme=``.

One function covers every stencil, passed wherever a ``scheme=`` is accepted (``u.d(x, scheme=...)``,
``u.d2(...)``, ``u.bind(x=x, y=y, scheme=...)`` for all of a binding's ``.x`` / ``.xx``)::

    jno.fd(order=4)                       # central, 4th order
    jno.fd(points=(-2, -1, 0, 1))         # any offsets; weights by Fornberg's algorithm
    jno.fd(weights={-1: -0.5, 1: 0.5})    # explicit weights on offsets (first derivatives)
    jno.fd(order=4, boundary=2)           # the order of the one-sided stencil at the edges

A stencil is fully defined by the derivative it approximates and the points it reads: the weights that are
exact on polynomials of the highest possible degree follow from B. Fornberg, "Generation of finite
difference formulas on arbitrarily spaced grids", *Math. Comp.* 51 (1988) 699, which :func:`fornberg`
implements. Where the interior stencil does not fit near a boundary, a one-sided stencil of order
``boundary`` (default: the interior order) is used, with the same number of points. A periodic axis wraps
instead. Structured grids only for now: on an unstructured mesh these options raise.

The spec is a ``str`` whose value is its canonical form (``"finite_difference:order=4,boundary=4"``), so
it travels through every path a scheme string does, and a kernel reads its options from the attributes.
"""

from __future__ import annotations

import numpy as np

_FAMILY = "finite_difference"


def fornberg(offsets, deriv: int) -> np.ndarray:
    """Weights ``w`` with ``Σ w_k f(x + s_k h) ≈ h^deriv f^(deriv)(x)``, exact for polynomials of degree
    ``len(offsets) − 1`` (Fornberg 1988, the algorithm of Table 3 with the expansion point at 0)."""
    s = np.asarray(offsets, dtype=float)
    n, m = len(s), int(deriv)
    if n <= m:
        raise ValueError(f"jno.fd: a derivative of order {m} needs at least {m + 1} points; got offsets {tuple(offsets)}.")
    if len(set(s.tolist())) != n:
        raise ValueError(f"jno.fd: repeated offsets {tuple(offsets)}.")
    c = np.zeros((n, m + 1))
    c[0, 0] = 1.0
    c1, c4 = 1.0, s[0]
    for i in range(1, n):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, s[i]
        for j in range(i):
            c3 = s[i] - s[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


class FDStencil(str):
    """A finite-difference stencil spec (see :func:`fd`). A ``str``, so it is a valid ``scheme=``."""

    order: int | None
    points: tuple | None
    weights: dict | None
    boundary: int | None
    average: str | None
    upwind: object

    def __new__(cls, *, order=None, points=None, weights=None, boundary=None, average=None, upwind=None):
        parts = []
        if upwind is not None:  # an expression; its identity keeps two different winds apart
            parts.append(f"upwind=#{id(upwind)}")
        if order is not None:
            parts.append(f"order={int(order)}")
        if points is not None:
            parts.append("points=" + "|".join(str(int(p)) for p in points))
        if weights is not None:
            parts.append("weights=" + "|".join(f"{int(k)}:{float(v)!r}" for k, v in sorted(weights.items())))
        if boundary is not None:
            parts.append(f"boundary={int(boundary)}")
        if average is not None:
            parts.append(f"average={average}")
        self = super().__new__(cls, f"{_FAMILY}:" + ",".join(parts))
        self.order = None if order is None else int(order)
        self.points = None if points is None else tuple(int(p) for p in points)
        self.weights = None if weights is None else {int(k): float(v) for k, v in weights.items()}
        self.boundary = None if boundary is None else int(boundary)
        self.average = average
        self.upwind = upwind
        return self

    def upwind_offsets(self):
        """``(offsets for a positive wind, offsets for a negative wind)`` of the order-``order`` upwind-biased
        first-derivative stencil: (−1, 0) for order 1, (−2, −1, 0) for 2, (−2, −1, 0, 1) for 3, … and the
        mirror image for a wind in the negative direction."""
        k = self.order or 1
        m = (k + 2) // 2
        pos = tuple(range(-m, k - m + 1))
        return pos, tuple(sorted(-o for o in pos))

    def stencils(self, deriv: int, n: int, periodic: bool = False):
        """``[(offsets, weights)]`` for the ``n`` nodes along one axis (weights scaled for h = 1).

        Interior nodes share one stencil; the nodes too close to an end for it get a one-sided stencil of
        order ``boundary`` with as many points, shifted inward. A periodic axis uses the interior stencil
        everywhere (the caller wraps)."""
        if self.weights is not None:
            if deriv != 1:
                raise ValueError(
                    "jno.fd(weights=...) gives a first-derivative stencil; for a second derivative, give "
                    "`points=` (weights by Fornberg) or write the second derivative as the math."
                )
            offs = tuple(sorted(self.weights))
            interior = (offs, np.array([self.weights[o] for o in offs]))
        else:
            offs = self.points if self.points is not None else self._central_offsets(deriv, self.order or 2)
            interior = (tuple(offs), fornberg(offs, deriv))
        if periodic:
            return [interior] * n
        lo, hi = min(interior[0]), max(interior[0])
        width = len(interior[0])
        b_order = self.boundary if self.boundary is not None else (self.order or max(width - deriv, 1))
        b_width = max(b_order + deriv, deriv + 1)
        if b_width > n:
            raise ValueError(
                f"jno.fd: the one-sided boundary stencil needs {b_width} points along the axis, which has {n}."
            )
        out = []
        for i in range(n):
            if i + lo >= 0 and i + hi <= n - 1:
                out.append(interior)
                continue
            start = 0 if i + lo < 0 else n - b_width  # the b_width nodes at the near end
            offs_i = tuple(range(start - i, start - i + b_width))
            out.append((offs_i, fornberg(offs_i, deriv)))
        return out

    @staticmethod
    def _central_offsets(deriv, order):
        if order < 1 or order % 2:
            raise ValueError(f"jno.fd(order={order}): a central stencil has an even order (2, 4, 6, …).")
        m = (deriv + order - 1) // 2
        return tuple(range(-m, m + 1))


def fd(
    *,
    order: int | None = None,
    points=None,
    weights=None,
    boundary: int | None = None,
    average: str | None = None,
    upwind=None,
) -> FDStencil:
    """A finite-difference stencil, for ``scheme=``: see the module docstring.

    Args:
        order: accuracy order of a central stencil (even). Default 2.
        points: integer offsets the stencil reads, e.g. ``(-2, -1, 0, 1)``; weights by Fornberg.
        weights: ``{offset: weight}`` of a first-derivative stencil for unit spacing (divided by h).
        boundary: order of the one-sided stencil used where the interior one does not fit. Default: the
            interior order.
        upwind: the wind (an expression: a number, a formula, a field, the unknown itself; a vector wind gives
            its component along each axis). The first derivative is then upwind-biased per node, of order
            ``order`` (default 1): it reads the side the wind comes from.
        average: the coefficient between nodes in the conservative form of ``(κ·u.x).x``: ``"exact"`` (κ at
            the half-points; the default when κ reads no stored field), ``"arithmetic"`` (the default
            otherwise) or ``"harmonic"``.
    """
    if average is not None and average not in ("exact", "arithmetic", "harmonic"):
        raise ValueError(f"jno.fd(average={average!r}): use 'exact', 'arithmetic' or 'harmonic'.")
    given = [k for k, v in (("order", order), ("points", points), ("weights", weights)) if v is not None]
    if len(given) > 1:
        raise ValueError(f"jno.fd: give one of order=, points=, weights= (got {', '.join(given)}).")
    if upwind is not None and (points is not None or weights is not None):
        raise ValueError(
            "jno.fd(upwind=...): the upwind stencil is set by its order; do not also give points= or weights=."
        )
    if upwind is None and order is not None and (int(order) < 2 or int(order) % 2):
        raise ValueError(f"jno.fd(order={order}): a central stencil has an even order (2, 4, 6, …).")
    if boundary is not None and int(boundary) < 1:
        raise ValueError(f"jno.fd(boundary={boundary}): the one-sided order must be at least 1.")
    if points is not None and 0 not in tuple(points) and len(tuple(points)) < 2:
        raise ValueError(f"jno.fd(points={points}): a stencil needs at least two points.")
    if weights is not None and abs(sum(weights.values())) > 1e-12:
        raise ValueError(
            f"jno.fd(weights={weights}): the weights of a derivative stencil sum to zero (a constant has zero "
            f"derivative); these sum to {sum(weights.values()):g}."
        )
    if upwind is not None and order is not None and int(order) < 1:
        raise ValueError(f"jno.fd(upwind=..., order={order}): the order must be at least 1.")
    return FDStencil(order=order, points=points, weights=weights, boundary=boundary, average=average, upwind=upwind)
