"""Grids whose every axis is Fourier: one ``rfftn`` per field, derivatives and resizing in coefficient space.

On such a grid a derivative leaf of an unknown is ``irfftn(pad(Π (i k_a)^α_a · û))``: one inverse transform per
leaf, the forward transform shared by every leaf of the unknown (cached per evaluation), and the padding of a
dealiased equation folded into the same inverse transform. This is the transform count of a hand-written
pseudo-spectral code.

Coefficients are normalised by the number of nodes: ``û = rfftn(u) / N``, so a resize keeps the amplitudes.
The last axis is the ``rfft`` (half-spectrum) axis.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
import numpy as np

__all__ = ["all_fourier", "coeffs", "values", "symbol", "resize", "wavenumber_grids"]


def all_fourier(grid) -> bool:
    from .bases import Fourier

    return all(isinstance(a, Fourier) for a in grid.axes)


def coeffs(grid, values):
    """``rfftn`` over the grid axes of a flat field ``(size,) + comp``, normalised: ``shape' + comp``."""
    u = grid.as_grid(values)
    axes = tuple(range(grid.ndim))
    return jnp.fft.rfftn(u, axes=axes) / grid.size


def values(grid, c):
    """Flat nodal values ``(size,) + comp`` from :func:`coeffs`-normalised coefficients on ``grid``."""
    axes = tuple(range(grid.ndim))
    u = jnp.fft.irfftn(c * grid.size, s=grid.shape, axes=axes)
    return grid.as_flat(u)


@functools.lru_cache(maxsize=64)
def _k(n, length, real):
    f = np.fft.rfftfreq(n, d=1.0 / n) if real else np.fft.fftfreq(n, d=1.0 / n)
    return 2.0 * np.pi * f / length


def wavenumber_grids(grid):
    """Per axis, the angular wavenumbers of the coefficient layout (last axis half-spectrum), broadcastable."""
    nd = grid.ndim
    out = []
    for i, a in enumerate(grid.axes):
        k = _k(a.n, a.length, i == nd - 1)
        out.append(k.reshape([k.size if j == i else 1 for j in range(nd)]))
    return out


def symbol(grid, alpha):
    """``Π_a (i k_a)^α_a`` on the coefficient layout of ``grid`` (Nyquist zeroed for odd orders, even n)."""
    nd = grid.ndim
    s = np.ones([1] * nd, dtype=complex)
    for i, (a, k) in enumerate(zip(grid.axes, wavenumber_grids(grid))):
        p = int(alpha[i])
        if not p:
            continue
        f = (1j * k) ** p
        if p % 2 == 1 and a.n % 2 == 0:
            f = f.copy()
            idx = [slice(None)] * nd
            idx[i] = a.n // 2
            f[tuple(idx)] = 0.0
        s = s * f
    return s


def _zeros(c, n):
    return jnp.zeros((n,) + c.shape[1:], dtype=c.dtype)


def _resize_full(c, axis, n0, n1):
    """Full-spectrum axis ``n0 -> n1`` (fftfreq order): pad with zeros or truncate; the Nyquist mode of the
    smaller even size is split in halves (padding) or folded (truncation). Static slices only (a slice update is
    done in place; an index scatter of complex128 is slow on a GPU)."""
    if n0 == n1:
        return c
    c = jnp.moveaxis(c, axis, 0)
    if n1 > n0:
        out = _zeros(c, n1)
        if n0 % 2:
            p, q = (n0 + 1) // 2, (n0 - 1) // 2
            out = out.at[:p].set(c[:p]).at[n1 - q :].set(c[n0 - q :])
        else:
            h = n0 // 2
            nyq = 0.5 * c[h : h + 1]
            out = out.at[:h].set(c[:h]).at[h : h + 1].set(nyq).at[n1 - h : n1 - h + 1].set(nyq)
            if h > 1:
                out = out.at[n1 - h + 1 :].set(c[h + 1 :])
    else:
        if n1 % 2:
            p, q = (n1 + 1) // 2, (n1 - 1) // 2
            out = jnp.concatenate([c[:p], c[n0 - q :]], axis=0) if q else c[:p]
        else:
            h = n1 // 2
            parts = [c[:h], c[h : h + 1] + c[n0 - h : n0 - h + 1]]
            if h > 1:
                parts.append(c[n0 - h + 1 :])
            out = jnp.concatenate(parts, axis=0)
    return jnp.moveaxis(out, 0, axis)


def _resize_half(c, axis, n0, n1):
    """Half-spectrum (rfft) axis ``n0 -> n1``."""
    if n0 == n1:
        return c
    c = jnp.moveaxis(c, axis, 0)
    m0, m1 = n0 // 2 + 1, n1 // 2 + 1
    if n1 > n0:
        out = _zeros(c, m1).at[:m0].set(c)
        if n0 % 2 == 0:
            out = out.at[m0 - 1 : m0].multiply(0.5)
    else:
        out = c[:m1]
        if n1 % 2 == 0:
            out = out.at[m1 - 1 : m1].set((2.0 * out[m1 - 1 : m1].real).astype(out.dtype))
    return jnp.moveaxis(out, 0, axis)


def resize(c, src, dst):
    """Coefficients of ``src`` (an all-Fourier grid) on the layout of ``dst`` (same box, any sizes)."""
    nd = src.ndim
    for i, (a, b) in enumerate(zip(src.axes, dst.axes)):
        c = _resize_half(c, i, a.n, b.n) if i == nd - 1 else _resize_full(c, i, a.n, b.n)
    return c
