"""Small numerical helpers for the jNO RCWA backend.

These reproduce the subset of fmmax's ``utils``/``misc`` helpers that the RCWA
seam consumes (``angular_frequency_for_wavelength``, ``absolute_axes``,
batch-diagonal and matrix-adjoint helpers), so that the owned backend needs
no fmmax import. Conventions (``c == 1``, trailing-axis operator ordering)
match fmmax 1.7.1; numerical parity is pinned by
``tests/test_rcwa_backend_basis.py``.
"""

import jax.numpy as jnp


def angular_frequency_for_wavelength(wavelength: jnp.ndarray) -> jnp.ndarray:
    """The angular frequency ``2 * pi / wavelength`` for the given wavelength.

    By the RCWA convention the speed of light is ``c == 1``, so the free-space
    wavevector is just the angular frequency.
    """
    return 2 * jnp.pi / wavelength


def absolute_axes(axes, ndim: int):
    """Normalizes relative `axes` against negative indices for an `ndim` array.

    Args:
        axes: Relative axes, each in ``range(-ndim, ndim)``.
        ndim: Number of dimensions of the target array.

    Returns:
        The normalized absolute axes as a tuple.

    Raises:
        ValueError: If an axis is out of range, or two axes collide after
            normalization.
    """
    if not all(a in list(range(-ndim, ndim)) for a in axes):
        raise ValueError(f"All elements of `axes` must be in the range ({-ndim}, {ndim - 1}) but got {axes}.")
    normalized = tuple(a % ndim for a in axes)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Found duplicates in `axes`; normalized axes are {normalized}.")
    return normalized


def batch_diag(x: jnp.ndarray) -> jnp.ndarray:
    """Maps a batch of vectors to a batch of diagonal matrices.

    `x` of shape ``(..., n)`` becomes an array of shape ``(..., n, n)`` with
    `x` as the diagonal of each matrix -- the trailing-axis generalization of
    `numpy.diag` restricted to square output.
    """
    shape = x.shape + (x.shape[-1],)
    y = jnp.zeros(shape, x.dtype)
    i = jnp.arange(x.shape[-1])
    return y.at[..., i, i].set(x)


def matrix_adjoint(x: jnp.ndarray) -> jnp.ndarray:
    """Computes the conjugate transpose of a batch of matrices."""
    axes = tuple(range(x.ndim - 2)) + (x.ndim - 1, x.ndim - 2)
    return jnp.conj(jnp.transpose(x, axes=axes))
