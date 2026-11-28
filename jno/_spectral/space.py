"""The spectral trial space of one unknown: what ``U.basis`` returns.

``dom.unknown(space="spectral", order=(64, 65))`` fixes the box and the number of nodes per axis. The family
of each axis is fixed later, by the problem that solves for the unknown: an axis with a written periodic tie
is Fourier, any other Chebyshev (unless ``space=("fourier", "legendre")`` named them). The node COUNT does not
depend on the family, so the DOF vector and every point set have their final sizes from the start; only the
positions wait for the problem. Once a problem has fixed the families they are frozen: a second problem
that would read them differently raises.
"""

from __future__ import annotations

import numpy as np

from .bases import Fourier, axis_from_spec
from .grid import Grid

__all__ = ["SpectralSpace", "FAMILIES", "parse_space"]

FAMILIES = ("fourier", "chebyshev", "legendre")
DEFAULT_FAMILY = "chebyshev"


def _split_rule(s):
    """``"chebyshev:gauss"`` -> ``("chebyshev", "gauss")``; a bare name has no rule."""
    name, _, rule = str(s).lower().partition(":")
    if rule and rule not in ("gauss", "lobatto"):
        raise ValueError(f"jno.spectral: rule {rule!r} in space {s!r}; use ':gauss' or ':lobatto'.")
    return name, (rule or None)


def parse_rules(space, ndim: int):
    """The node rule per axis named in ``space`` (``"spectral:gauss"``, ``("fourier", "legendre:gauss")``)."""
    names = (space,) * ndim if isinstance(space, str) else tuple(space)
    return tuple(_split_rule(s)[1] for s in names)


def parse_space(space, ndim: int):
    """``"spectral"`` -> ``(None,) * ndim`` (inferred); a family name or a tuple of them -> explicit per axis.
    A ``":gauss"`` / ``":lobatto"`` suffix picks the node rule of a polynomial axis (see :func:`parse_rules`).

    Returns ``None`` when ``space`` is not a spectral space at all.
    """
    if isinstance(space, str):
        space = _split_rule(space)[0]
    elif isinstance(space, (tuple, list)):
        space = tuple(_split_rule(s)[0] if isinstance(s, str) else s for s in space)
    if isinstance(space, str):
        s = space.lower()
        if s == "spectral":
            return (None,) * ndim
        if s in FAMILIES:
            return (s,) * ndim
        return None
    if isinstance(space, (tuple, list)) and space and all(isinstance(s, str) for s in space):
        fams = tuple(None if s.lower() == "spectral" else s.lower() for s in space)
        bad = [s for s in fams if s is not None and s not in FAMILIES]
        if bad:
            raise ValueError(f"jno.spectral: unknown basis families {bad}; use 'spectral' or one of {FAMILIES}.")
        if len(fams) != ndim:
            raise ValueError(f"jno.spectral: space={tuple(space)!r} names {len(fams)} axes for a {ndim}-D domain.")
        return fams
    return None


def parse_orders(order, ndim: int) -> tuple:
    orders = (order,) * ndim if np.isscalar(order) else tuple(order)
    if len(orders) != ndim:
        raise ValueError(f"jno.spectral: order={order!r} gives {len(orders)} sizes for a {ndim}-D domain.")
    out = []
    for n in orders:
        if int(n) != n or int(n) < 2:
            raise ValueError(f"jno.spectral: every order (nodes per axis) must be an integer >= 2, got {order!r}.")
        out.append(int(n))
    return tuple(out)


class SpectralSpace:
    """Box, nodes per axis, per-axis families (``None`` until a problem fixes them) and value shape."""

    def __init__(self, box, orders, families=None, value_shape=(), rules=None):
        self.box = tuple((float(a), float(b)) for a, b in box)
        self.ndim = len(self.box)
        self.orders = parse_orders(orders, self.ndim)
        fams = (None,) * self.ndim if families is None else tuple(families)
        if len(fams) != self.ndim:
            raise ValueError(f"jno.spectral: {len(fams)} families for a {self.ndim}-D space.")
        self._explicit = fams
        self._families = fams if all(f is not None for f in fams) else None
        self.value_shape = tuple(value_shape)
        self.rules = tuple(rules) if rules is not None else (None,) * self.ndim
        self._grid = None

    def __repr__(self):
        fams = self._families or tuple(f or "?" for f in self._explicit)
        return f"SpectralSpace(orders={self.orders}, families={fams}, box={self.box})"

    # families ---------------------------------------------------------------------------------------------
    @property
    def explicit(self) -> tuple:
        return self._explicit

    @property
    def resolved(self) -> bool:
        return self._families is not None

    @property
    def families(self) -> tuple:
        if self._families is None:
            raise RuntimeError(
                "jno.spectral: the basis families of this unknown are fixed by the problem that solves for it "
                "(a written periodic tie makes an axis Fourier); build and solve the problem first, or name them: "
                "dom.unknown(space=('fourier', 'chebyshev'), ...)."
            )
        return self._families

    def resolve(self, tied_axes) -> Grid:
        """Fix the families from the set of tied axes; returns the grid. Idempotent; a conflict raises."""
        tied = set(int(i) for i in tied_axes)
        fams = []
        for i, f in enumerate(self._explicit):
            if i in tied and f not in (None, "fourier"):
                raise ValueError(
                    f"jno.spectral: axis {i} has a periodic tie written for it but its basis is named {f!r}; "
                    "a tied axis is Fourier. Drop the tie or name the axis 'fourier'."
                )
            if i not in tied and f == "fourier":
                raise ValueError(
                    f"jno.spectral: axis {i} is named 'fourier' but no periodic tie is written for it. Write the "
                    "tie (e.g. U(xl, yl) - U(xr, yr)) or name another family."
                )
            fams.append("fourier" if i in tied else (f or DEFAULT_FAMILY))
        fams = tuple(fams)
        if self._families is not None and self._families != fams:
            raise ValueError(
                f"jno.spectral: this unknown was already solved with families {self._families}; this problem's ties "
                f"give {fams}. One unknown has one basis."
            )
        self._families = fams
        self._grid = None
        return self.grid

    @property
    def grid(self) -> Grid:
        if self._grid is None:
            self._grid = self.grid_for(self.families)
        return self._grid

    def grid_for(self, families) -> Grid:
        return Grid(
            axis_from_spec(f, n, a, b, rule=None if f == "fourier" else r)
            for f, n, (a, b), r in zip(families, self.orders, self.box, self.rules)
        )

    @property
    def provisional_grid(self) -> Grid:
        """The grid with unresolved axes taken as the default family: right sizes, provisional positions."""
        if self.resolved:
            return self.grid
        return self.grid_for(tuple(f or DEFAULT_FAMILY for f in self._explicit))

    # sizes ------------------------------------------------------------------------------------------------
    @property
    def size(self) -> int:
        return int(np.prod(self.orders))

    # what users call --------------------------------------------------------------------------------------
    @property
    def nodes(self) -> np.ndarray:
        """``(size, ndim)`` node coordinates, C order: the points the nodal values belong to."""
        return self.grid.points

    def coeffs(self, values):
        """Spectral coefficients of a nodal field, ``orders + value_shape``: complex Fourier coefficients
        (``fftfreq`` order, ``u = Σ ĉ_k e^{i k (x - a)}``) on Fourier axes, Chebyshev ``T_k`` / Legendre ``P_k``
        coefficients on polynomial axes. Works on a single field ``(size,) + value_shape`` or a batch / trajectory
        with leading axes."""
        return self._batched(values, self.grid.forward, lead_out=self.orders)

    def values(self, coeffs, *, real: bool = True):
        """Nodal values from :meth:`coeffs` (inverse transform); ``real`` drops imaginary round-off."""
        c = np.asarray(coeffs) if not hasattr(coeffs, "shape") else coeffs
        nd = self.ndim
        lead = c.shape[: c.ndim - nd - len(self.value_shape)]
        if lead:
            import jax

            flat = c.reshape((-1,) + c.shape[len(lead) :])
            out = jax.vmap(lambda ci: self.grid.backward(ci, real=real))(flat)
            return out.reshape(lead + out.shape[1:])
        return self.grid.backward(c, real=real)

    def __call__(self, values, *coords):
        """The interpolant of ``values`` at points: ``basis(u, x, y)`` with coordinate arrays (broadcast), or
        ``basis(u, pts)`` with ``pts`` of shape ``(m, ndim)``. Spectrally accurate, not nearest-node."""
        if len(coords) == 1 and np.ndim(coords[0]) == 2 and np.shape(coords[0])[1] == self.ndim:
            pts = np.asarray(coords[0], dtype=float)
            shape = (pts.shape[0],)
        elif len(coords) == self.ndim:
            arrs = np.broadcast_arrays(*[np.asarray(c, dtype=float) for c in coords])
            shape = arrs[0].shape
            pts = np.stack([a.reshape(-1) for a in arrs], axis=1)
        else:
            raise TypeError(f"jno.spectral: give {self.ndim} coordinate arrays or one (m, {self.ndim}) array.")
        out = self.grid.at(values, pts)
        return out.reshape(shape + out.shape[1:])

    def refine(self, values, order):
        """Nodal values on the same families and box with ``order`` nodes per axis (padding or truncating Fourier
        spectra, re-interpolating polynomial axes): a warm start on a finer space or a coarser view."""
        return self.grid.to(values, self.grid.with_orders(parse_orders(order, self.ndim)))

    def integrate(self, values):
        """``∫_box u`` by the basis quadrature (exact for the space's polynomials / trig polynomials)."""
        return self.grid.integrate(values)

    def mean(self, values):
        return self.grid.integrate(values) / self.grid.volume

    def wavenumbers(self, axis: int) -> np.ndarray:
        """Angular wavenumbers of a Fourier axis in ``fftfreq`` order (the order of :meth:`coeffs`)."""
        a = self.grid.axes[axis]
        if not isinstance(a, Fourier):
            raise ValueError(f"jno.spectral: axis {axis} is {a.family}, not Fourier; its modes are polynomial degrees.")
        return a.wavenumbers

    def _batched(self, values, fn, lead_out):
        import jax
        import jax.numpy as jnp

        u = jnp.asarray(values)
        lead = u.shape[: u.ndim - 1 - len(self.value_shape)]
        if lead:
            flat = u.reshape((-1,) + u.shape[len(lead) :])
            out = jax.vmap(fn)(flat)
            return out.reshape(lead + out.shape[1:])
        return fn(u)
