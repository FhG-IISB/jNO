"""Tensor-product grids of spectral axes on a box.

A field on a :class:`Grid` is stored flat, ``(size,) + components``, in C order over the axes (the layout of a
nodal unknown, ``(N,) + value_shape``). Every operator acts one axis at a time -- an FFT or a dense
``(m x n)`` matrix along that axis -- so a tensor operator costs ``O(size · Σ n_k)`` (dense axes) or
``O(size · Σ log n_k)`` (FFT axes) and is never formed as a Kronecker matrix.
"""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np

from .bases import HIGHEST, Axis, Fourier, apply_axis

__all__ = ["Grid"]


class Grid:
    """The tensor product of ``axes`` (one :class:`~jno._spectral.bases.Axis` per coordinate)."""

    def __init__(self, axes):
        axes = tuple(axes)
        if not axes or not all(isinstance(a, Axis) for a in axes):
            raise TypeError(f"jno.spectral: a Grid needs one or more spectral axes, got {axes!r}.")
        self.axes = axes

    # identity ---------------------------------------------------------------------------------------------
    @property
    def key(self):
        return tuple(a.key for a in self.axes)

    def __eq__(self, other):
        return isinstance(other, Grid) and self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f"Grid({', '.join(map(repr, self.axes))})"

    # sizes ------------------------------------------------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple:
        return tuple(a.n for a in self.axes)

    @property
    def size(self) -> int:
        return int(math.prod(self.shape))

    @property
    def box(self) -> tuple:
        return tuple((a.a, a.b) for a in self.axes)

    @property
    def periodic(self) -> tuple:
        return tuple(a.periodic for a in self.axes)

    def with_orders(self, orders) -> "Grid":
        """The same families and box with ``orders`` nodes per axis (an int applies to every axis)."""
        orders = (orders,) * self.ndim if np.isscalar(orders) else tuple(orders)
        if len(orders) != self.ndim:
            raise ValueError(f"jno.spectral: {len(orders)} orders for a {self.ndim}-D grid.")
        return Grid(a.with_n(int(n)) for a, n in zip(self.axes, orders))

    # points -----------------------------------------------------------------------------------------------
    @functools.cached_property
    def points(self) -> np.ndarray:
        """``(size, ndim)`` node coordinates in C order."""
        mesh = np.meshgrid(*[a.nodes for a in self.axes], indexing="ij")
        return np.stack([m.reshape(-1) for m in mesh], axis=1)

    def face(self, axis: int, side: int) -> np.ndarray:
        """Flat indices of the nodes on the ``side`` (0 = low, 1 = high) face of ``axis``.

        A Fourier axis has no node on its high end (it is the periodic image of the low one): asking for it
        raises, because a row there would duplicate the low face's.
        """
        ax = self.axes[axis]
        if ax.periodic and side == 1:
            raise ValueError(
                f"jno.spectral: axis {axis} is periodic ({ax}); its high face is the image of the low one and "
                "has no nodes of its own."
            )
        if side == 0 and not np.isclose(ax.nodes[0], ax.a):
            raise ValueError(f"jno.spectral: axis {axis} ({ax}) has no node on its low face (a Gauss rule).")
        if side == 1 and not np.isclose(ax.nodes[-1], ax.b):
            raise ValueError(f"jno.spectral: axis {axis} ({ax}) has no node on its high face (a Gauss rule).")
        idx = np.arange(self.size).reshape(self.shape)
        return np.ascontiguousarray(np.take(idx, 0 if side == 0 else ax.n - 1, axis=axis).reshape(-1))

    @functools.cached_property
    def boundary(self) -> np.ndarray:
        """Flat indices of every node on a face of a non-periodic axis (sorted, each once)."""
        faces = [
            self.face(i, s)
            for i, a in enumerate(self.axes)
            if not a.periodic
            for s in (0, 1)
            if np.isclose(a.nodes[0 if s == 0 else -1], a.a if s == 0 else a.b)
        ]
        return np.unique(np.concatenate(faces)) if faces else np.empty(0, dtype=int)

    # layout -----------------------------------------------------------------------------------------------
    def as_grid(self, values):
        """``(size,) + comp`` -> ``shape + comp``."""
        values = jnp.asarray(values)
        if values.shape[: self.ndim] == self.shape:
            return values
        if values.shape[0] != self.size:
            raise ValueError(f"jno.spectral: a field of shape {values.shape} does not live on {self} ({self.size} nodes).")
        return values.reshape(self.shape + values.shape[1:])

    def as_flat(self, arr):
        """``shape + comp`` -> ``(size,) + comp``."""
        return arr.reshape((self.size,) + arr.shape[self.ndim :])

    # operators --------------------------------------------------------------------------------------------
    def diff(self, values, alpha):
        """``∂^alpha`` of a flat field (``alpha`` = derivative order per axis), at the same nodes."""
        alpha = tuple(int(k) for k in alpha)
        if len(alpha) != self.ndim:
            raise ValueError(f"jno.spectral: derivative orders {alpha} for a {self.ndim}-D grid.")
        u = self.as_grid(values)
        for i, k in enumerate(alpha):
            if k:
                u = self.axes[i].diff(u, k, axis=i)
        return self.as_flat(u)

    def to(self, values, other: "Grid"):
        """Nodal values on ``other`` (same box, any axes) of the interpolant of a flat field on this grid."""
        if other == self:
            return jnp.asarray(values)
        if other.ndim != self.ndim:
            raise ValueError(f"jno.spectral: cannot move a field from {self} to {other}.")
        u = self.as_grid(values)
        for i, (src, dst) in enumerate(zip(self.axes, other.axes)):
            u = src.to(u, dst, axis=i)
        return other.as_flat(u)

    def at(self, values, points, *, batch: int = 4096):
        """The interpolant at scattered ``points`` ``(m, ndim)``: ``(m,) + comp``.

        Contracts one axis at a time with that axis's ``(m x n_k)`` cardinal matrix, so the work is
        ``m · size`` and the largest temporary ``batch · size / n_0``; points are processed in chunks of
        ``batch``.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, self.ndim)
        u = self.as_grid(values)
        comp = u.shape[self.ndim :]
        outs = []
        for lo in range(0, pts.shape[0], int(batch)):
            p = pts[lo : lo + int(batch)]
            A = jnp.tensordot(
                jnp.asarray(self.axes[0].interp_matrix(p[:, 0]), dtype=_real(u)), u, axes=(1, 0), precision=HIGHEST
            )
            for i in range(1, self.ndim):
                L = jnp.asarray(self.axes[i].interp_matrix(p[:, i]), dtype=_real(u))
                A = jnp.einsum("mj...,mj->m...", A, L, precision=HIGHEST)
            outs.append(A)
        out = jnp.concatenate(outs, axis=0) if outs else jnp.zeros((0,) + comp, dtype=u.dtype)
        return out

    # quadrature -------------------------------------------------------------------------------------------
    @functools.cached_property
    def weights(self) -> np.ndarray:
        """``(size,)`` tensor quadrature weights: ``∫ f ≈ Σ w f(x)``."""
        w = self.axes[0].weights
        for a in self.axes[1:]:
            w = np.multiply.outer(w, a.weights)
        return np.asarray(w).reshape(-1)

    def integrate(self, values):
        """``∫_box u`` of a flat field (per component)."""
        u = jnp.asarray(values)
        w = jnp.asarray(self.weights, dtype=_real(u))
        return jnp.tensordot(w, u, axes=(0, 0), precision=HIGHEST)

    @property
    def volume(self) -> float:
        return float(np.prod([a.length for a in self.axes]))

    # modal transforms -------------------------------------------------------------------------------------
    def forward(self, values):
        """Coefficients, ``shape + comp``: complex Fourier coefficients (``fftfreq`` order) on Fourier axes, modal
        coefficients on polynomial axes (real when no axis is Fourier)."""
        u = self.as_grid(values)
        for i, a in enumerate(self.axes):
            u = a.forward(u, axis=i)
        return u

    def backward(self, coeffs, *, real: bool = True):
        """Flat nodal values from :meth:`forward`'s coefficients."""
        c = jnp.asarray(coeffs)
        for i, a in reversed(list(enumerate(self.axes))):
            c = a.backward(c, axis=i, real=False) if isinstance(a, Fourier) else a.backward(c, axis=i)
        if real and jnp.iscomplexobj(c):
            c = c.real
        return self.as_flat(c)

    def apply(self, values, ops):
        """Apply one ``(m_k x n_k)`` matrix (or ``None`` = identity) per axis; returns ``shape' + comp``."""
        u = self.as_grid(values)
        for i, M in enumerate(ops):
            if M is not None:
                u = apply_axis(M, u, i)
        return u


def _real(u):
    return jnp.finfo(jnp.asarray(u).dtype).dtype if jnp.issubdtype(jnp.asarray(u).dtype, jnp.inexact) else float
