"""Fourier-domain transforms for the RCWA backend.

Owned implementation of the fmmax-convention Fourier-convolution matrix and
the expansion-aware forward/inverse FFT pair used by the RCWA seam. The
centered-coordinate phase convention, the normalization and the gather order
match fmmax 1.7.1 so that downstream matrices agree bit-for-bit; numerical
parity is pinned by ``tests/test_rcwa_backend_basis.py``.

"Centered coordinates" means the physical sample ``(0, 0)`` sits at the center
of array element ``(0, 0)`` (i.e. at ``du/2 + dv/2``), so the transform
carries an extra linear phase ``exp(+/- 2 pi i (ki/2 + kj/2))``.
"""

from typing import Tuple

import jax.numpy as jnp

from . import basis, utils


def fourier_convolution_matrix(
    x: jnp.ndarray,
    expansion: basis.Expansion,
) -> jnp.ndarray:
    """The Fourier convolution (Toeplitz) matrix for the real-space field `x`.

    Entry ``(i, j)`` is the Fourier coefficient of the lattice vector
    ``G_i - G_j`` (equation 8 of [2012 Liu]); the row/column ordering follows
    the expansion ordering. Computed as an FFT of the real-space grid followed
    by a gather at the Toeplitz index pairs.

    Args:
        x: The real-space field, with spatial dimensions as its final two
            axes, at least `min_array_shape_for_expansion(expansion)` large.
        expansion: The expansion describing the coefficient ordering.

    Returns:
        The convolution matrix, shape ``x.shape[:-2] + (nt, nt)``.
    """
    basis.validate_shape_for_expansion(x.shape, expansion)

    x_fft = _fft2(x, axes=(-2, -1), norm="backward", centered_coordinates=True)
    # Normalize by the grid size. The divisor is an exact integer, so the real
    # (vs complex) dtype choice here is numerically inert; using the real dtype
    # avoids a float64->float32 truncation warning from ``astype(complex)`` when
    # x64 is disabled and `x` is float64.
    x_fft = x_fft / jnp.prod(jnp.asarray(x.shape[-2:])).astype(jnp.real(x).dtype)
    idx = _standard_toeplitz_indices(expansion)
    return x_fft[..., idx[..., 0], idx[..., 1]]


def fft(
    x: jnp.ndarray,
    expansion: basis.Expansion,
    axes: Tuple[int, int] = (-2, -1),
    centered_coordinates: bool = True,
) -> jnp.ndarray:
    """The expansion-aware 2-D Fourier transform of `x`.

    Returns the Fourier coefficients gathered at the expansion's lattice
    vectors, so the spatial axes shrink to the expansion size.

    Args:
        x: The array to transform, with spatial axes large enough for
            `expansion`.
        expansion: The expansion specifying which coefficients to keep.
        axes: The two axes to transform, default ``(-2, -1)``.
        centered_coordinates: Whether the coordinates lie at element centers
            (default) or at element corners.

    Returns:
        The gathered coefficients, shape ``x.shape[:a0] + (nt,) + x.shape[a1+1:]``
        (the two spatial axes collapse into the term axis).
    """
    axes = utils.absolute_axes(axes, x.ndim)
    basis.validate_shape_for_expansion(tuple(x.shape[a] for a in axes), expansion)

    x_fft = _fft2(
        x,
        axes=axes,
        norm="forward",
        centered_coordinates=centered_coordinates,
    )

    leading_dims = len(x.shape[: axes[0]])
    trailing_dims = len(x.shape[axes[1] + 1 :])
    slices = (
        [slice(None)] * leading_dims
        + [expansion.basis_coefficients[:, 0], expansion.basis_coefficients[:, 1]]
        + [slice(None)] * trailing_dims
    )
    return x_fft[tuple(slices)]


def ifft(
    y: jnp.ndarray,
    expansion: basis.Expansion,
    shape: Tuple[int, int],
    axis: int = -1,
    centered_coordinates: bool = True,
) -> jnp.ndarray:
    """The expansion-aware 2-D inverse Fourier transform of `y`.

    Scatters the Fourier coefficients back onto the full grid at the lattice
    vectors given by `expansion`, then inverse-transforms.

    Args:
        y: The Fourier coefficients, with the term axis of length
            ``expansion.num_terms``.
        expansion: The expansion describing where the coefficients sit on the
            grid.
        shape: The output spatial shape ``(nx, ny)``, large enough for
            `expansion`.
        axis: The axis of `y` holding the coefficients, default ``-1``.
        centered_coordinates: As in `fft`.

    Returns:
        The real-space array of shape ``y.shape[:axis] + shape + y.shape[axis+1:]``.
    """
    (axis,) = utils.absolute_axes((axis,), y.ndim)
    if y.shape[axis] != expansion.num_terms:
        raise ValueError(
            f"`y` has {y.shape[axis]} coefficients on its axis {axis}, but the expansion has {expansion.num_terms} terms."
        )
    x_shape = y.shape[:axis] + tuple(shape) + y.shape[axis + 1 :]

    basis.validate_shape_for_expansion(shape, expansion)

    leading_dims = len(y.shape[:axis])
    trailing_dims = len(y.shape[axis + 1 :])
    slices = (
        [slice(None)] * leading_dims
        + [expansion.basis_coefficients[:, 0], expansion.basis_coefficients[:, 1]]
        + [slice(None)] * trailing_dims
    )

    x = jnp.zeros(x_shape, y.dtype)
    x = x.at[tuple(slices)].set(y)
    return _ifft2(
        x,
        axes=(leading_dims, leading_dims + 1),
        norm="forward",
        centered_coordinates=centered_coordinates,
    )


def _fft2(
    x: jnp.ndarray,
    axes: Tuple[int, int] = (-2, -1),
    norm: str = "forward",
    centered_coordinates: bool = True,
) -> jnp.ndarray:
    """The 2-D FFT, with the centered-coordinate phase applied when requested."""
    y = jnp.fft.fft2(x, axes=axes, norm=norm)
    if centered_coordinates:
        axes = utils.absolute_axes(axes, ndim=x.ndim)
        ki = 0.5 * jnp.fft.fftfreq(x.shape[axes[0]], dtype=jnp.float32)[:, jnp.newaxis]
        kj = 0.5 * jnp.fft.fftfreq(x.shape[axes[1]], dtype=jnp.float32)[jnp.newaxis, :]
        phase = jnp.exp(-1j * 2 * jnp.pi * (ki + kj))
        phase = phase.reshape(phase.shape + (1,) * (x.ndim - axes[1] - 1))
        y = y * phase
    return y


def _ifft2(
    x: jnp.ndarray,
    axes: Tuple[int, int] = (-2, -1),
    norm: str = "forward",
    centered_coordinates: bool = True,
) -> jnp.ndarray:
    """The 2-D IFFT, with the centered-coordinate phase applied when requested."""
    if centered_coordinates:
        axes = utils.absolute_axes(axes, ndim=x.ndim)
        ki = 0.5 * jnp.fft.fftfreq(x.shape[axes[0]], dtype=jnp.float32)[:, jnp.newaxis]
        kj = 0.5 * jnp.fft.fftfreq(x.shape[axes[1]], dtype=jnp.float32)[jnp.newaxis, :]
        phase = jnp.exp(1j * 2 * jnp.pi * (ki + kj))
        phase = phase.reshape(phase.shape + (1,) * (x.ndim - axes[1] - 1))
        x = x * phase
    return jnp.fft.ifft2(x, axes=axes, norm=norm)


def _standard_toeplitz_indices(expansion: basis.Expansion) -> jnp.ndarray:
    """Index pairs ``(nt, nt, 2)`` of ``G_i - G_j`` for the Toeplitz gather.

    The difference indices index into a standard FFT grid, where negative
    values wrap around (frequency-major ordering), giving the convolution
    matrix of Laurent's rule.
    """
    i, j = jnp.meshgrid(
        jnp.arange(expansion.num_terms),
        jnp.arange(expansion.num_terms),
        indexing="ij",
    )
    basis_coefficients = jnp.asarray(expansion.basis_coefficients)
    idx = basis_coefficients[i, :] - basis_coefficients[j, :]
    return idx
