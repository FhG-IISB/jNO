"""Fourier coefficients of a sampled unit cell, and the convolution matrices built from them.

A grid of ``Nx x Ny`` samples ``f_ij`` covers one cell. With *centred* coordinates the sample ``(i, j)`` sits at
the pixel centre ``r_ij = ((i + 1/2)/Nx) a1 + ((j + 1/2)/Ny) a2``, the natural reading of a rasterised design;
with corner coordinates at ``(i/Nx) a1 + (j/Ny) a2``. The Fourier coefficient of order ``(m, n)`` is

    f_mn = (1 / (Nx Ny)) sum_ij f_ij exp(-2 pi i (m (i + s)/Nx + n (j + s)/Ny)),   s = 1/2 (centred) or 0,

that is the plain DFT times the phase ``exp(-pi i (m/Nx + n/Ny))`` in the centred case. The phase is evaluated in
the working precision of the data (a single-precision phase would cap a float64 solve at 1e-7).

Laurent's rule represents multiplication by ``f`` in the truncated basis by the Toeplitz matrix
``[[f]]_{ij} = f_{G_i - G_j}`` (Li, JOSA A 13, 1870 (1996); Liu & Fan, Comput. Phys. Commun. 183, 2233 (2012),
eq. 8): :func:`fourier_convolution_matrix`.
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as onp

from . import basis


def _real_dtype(x) -> jnp.dtype:
    return jnp.real(jnp.zeros((), jnp.result_type(x, jnp.float32))).dtype


def _centring_phase(shape: Tuple[int, int], sign: float, dtype) -> jnp.ndarray:
    """``exp(sign pi i (m/Nx + n/Ny))`` on the DFT frequency grid (``m``, ``n`` in DFT index order)."""
    fx = jnp.fft.fftfreq(shape[0]).astype(dtype)[:, None]
    fy = jnp.fft.fftfreq(shape[1]).astype(dtype)[None, :]
    return jnp.exp(sign * 1j * jnp.pi * (fx + fy))


def _coefficients(x: jnp.ndarray, axes: Tuple[int, int], centered: bool) -> jnp.ndarray:
    """All DFT coefficients ``f_mn`` of ``x`` over ``axes`` (normalised by ``Nx Ny``), in DFT index order."""
    y = jnp.fft.fft2(x, axes=axes, norm="forward")
    if centered:
        shape = (x.shape[axes[0]], x.shape[axes[1]])
        phase = _centring_phase(shape, -1.0, _real_dtype(x))
        y = y * phase.reshape(phase.shape + (1,) * (x.ndim - axes[1] - 1))
    return y


def _absolute_axes(axes, ndim: int) -> Tuple[int, ...]:
    out = tuple(a % ndim for a in axes)
    if len(set(out)) != len(out) or any(not -ndim <= a < ndim for a in axes):
        raise ValueError(f"invalid axes {axes} for an array of rank {ndim}.")
    return out


def fourier_convolution_matrix(x: jnp.ndarray, expansion: basis.Expansion) -> jnp.ndarray:
    """Laurent's-rule matrix ``[[x]]_{ij} = x_{G_i - G_j}`` of the grid ``x`` (last two axes spatial, centred).

    Returns shape ``x.shape[:-2] + (num_terms, num_terms)``. The difference orders index the DFT grid modulo its
    size (negative orders wrap), so the grid only needs to resolve the orders themselves; differences beyond
    that alias, which is the usual RCWA convention.
    """
    basis.validate_shape_for_expansion(x.shape, expansion)
    c = onp.asarray(expansion.basis_coefficients)
    d = c[:, None, :] - c[None, :, :]  # (n, n, 2) difference orders
    return _coefficients(x, (x.ndim - 2, x.ndim - 1), centered=True)[..., d[..., 0], d[..., 1]]


def fft(
    x: jnp.ndarray,
    expansion: basis.Expansion,
    axes: Tuple[int, int] = (-2, -1),
    centered_coordinates: bool = True,
) -> jnp.ndarray:
    """The coefficients ``f_mn`` of ``x`` at the expansion's orders; the two spatial ``axes`` collapse into one
    order axis at the position of the first."""
    axes = _absolute_axes(axes, x.ndim)
    basis.validate_shape_for_expansion(tuple(x.shape[a] for a in axes), expansion)
    y = _coefficients(x, axes, centered_coordinates)
    c = expansion.basis_coefficients
    index = [slice(None)] * axes[0] + [c[:, 0], c[:, 1]] + [slice(None)] * (x.ndim - axes[1] - 1)
    return y[tuple(index)]


def ifft(
    y: jnp.ndarray,
    expansion: basis.Expansion,
    shape: Tuple[int, int],
    axis: int = -1,
    centered_coordinates: bool = True,
) -> jnp.ndarray:
    """The grid ``x`` of the given spatial ``shape`` whose coefficients at the expansion's orders are ``y`` (along
    ``axis``) and zero elsewhere: the inverse of :func:`fft`. The order axis expands into the two spatial axes."""
    (axis,) = _absolute_axes((axis,), y.ndim)
    if y.shape[axis] != expansion.num_terms:
        raise ValueError(
            f"`y` has {y.shape[axis]} coefficients on its axis {axis}, but the expansion has {expansion.num_terms} terms."
        )
    basis.validate_shape_for_expansion(shape, expansion)
    c = expansion.basis_coefficients
    full = jnp.zeros(y.shape[:axis] + tuple(shape) + y.shape[axis + 1 :], y.dtype)
    index = [slice(None)] * axis + [c[:, 0], c[:, 1]] + [slice(None)] * (y.ndim - axis - 1)
    full = full.at[tuple(index)].set(y)
    if centered_coordinates:
        phase = _centring_phase(tuple(shape), 1.0, _real_dtype(y))
        full = full * phase.reshape(phase.shape + (1,) * (y.ndim - axis - 1))
    return jnp.fft.ifft2(full, axes=(axis, axis + 1), norm="forward")
