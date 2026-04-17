"""Shared building blocks for architecture modules.

Centralises BatchNorm, activation lookup, Fourier-mode computation, and
NHWC convolution helpers so they aren't duplicated across architecture files.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


# ---------------------------------------------------------------------------
# BatchNorm  (stateless / instance-norm style)
# ---------------------------------------------------------------------------


class BatchNorm(eqx.Module):
    """Simple stateless batch normalization (always uses per-batch statistics)."""

    weight: jnp.ndarray
    bias: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, num_features: int, eps: float = 1e-5, **kwargs):
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.eps = eps

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        axes = tuple(range(x.ndim - 1))
        mean = jnp.mean(x, axis=axes, keepdims=True)
        var = jnp.var(x, axis=axes, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# Activation lookup
# ---------------------------------------------------------------------------


def get_activation(name: str) -> Optional[Callable]:
    """Return a JAX activation function by name.

    Supported: ``gelu``, ``relu``, ``tanh``, ``elu``, ``leaky_relu``,
    ``sigmoid``, ``silu`` / ``swish``, ``none``.
    """
    activations = {
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "elu": jax.nn.elu,
        "leaky_relu": jax.nn.leaky_relu,
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,
        "none": None,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(activations.keys())}")
    return activations[key]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Flax → Equinox shims  (DEPRECATED — kept only for old checkpoint compat)
# ---------------------------------------------------------------------------
# FlaxModelWrapper and FlaxNNXWrapper have been removed from the runtime.
# All foundation models in foundax are now Equinox-native.
# If you need to load a legacy Flax checkpoint, convert it offline first.


# ---------------------------------------------------------------------------
# Fourier-mode computation (used by GeoFNO and PCNO)
# ---------------------------------------------------------------------------


def compute_Fourier_modes(ndims: int, nks: Sequence[int], Ls: Sequence[float]) -> np.ndarray:
    """Compute Fourier mode wave-vectors ``k``.

    Fourier bases are ``cos(k·x)``, ``sin(k·x)``, ``1``.
    We keep only one of each ``±k`` pair.

    Args:
        ndims: Number of spatial dimensions (1, 2, or 3).
        nks: Number of modes per dimension.
        Ls: Domain lengths per dimension.

    Returns:
        ``k_pairs`` of shape ``(nmodes, ndims)`` sorted by magnitude.
    """
    if ndims == 1:
        nk = nks[0]
        Lx = Ls[0]
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nk + 1):
            k_pairs[i, :] = 2 * np.pi / Lx * kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2 * nx * ny + nx + ny
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if ky == 0 and kx <= 0:
                    continue
                k_pairs[i, :] = 2 * np.pi / Lx * kx, 2 * np.pi / Ly * ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4 * nx * ny * nz + 2 * (nx * ny + nx * nz + ny * nz) + nx + ny + nz
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if kz == 0 and (ky < 0 or (ky == 0 and kx <= 0)):
                        continue
                    k_pairs[i, :] = (
                        2 * np.pi / Lx * kx,
                        2 * np.pi / Ly * ky,
                        2 * np.pi / Lz * kz,
                    )
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")

    k_pairs = k_pairs[np.argsort(k_pair_mag, kind="stable"), :]
    return k_pairs


# ---------------------------------------------------------------------------
# NHWC 2-D Convolution helpers  (used by CNO, MGNO, U-Net, …)
# ---------------------------------------------------------------------------


class Conv2d(eqx.Module):
    """2-D convolution operating on NHWC data."""

    weight: jnp.ndarray  # (kH, kW, in_ch, out_ch)
    bias: Optional[jnp.ndarray]
    padding: str = eqx.field(static=True)
    strides: Tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        strides: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        use_bias: bool = True,
        *,
        key: jax.Array,
    ):
        kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if not isinstance(kernel_size, int):
            kw = kernel_size[1]
        fan_in = in_channels * kh * kw
        std = 1.0 / jnp.sqrt(fan_in)
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (kh, kw, in_channels, out_channels)) * std
        self.bias = jnp.zeros(out_channels) if use_bias else None
        self.padding = padding
        self.strides = strides

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x[None]
        y = jax.lax.conv_general_dilated(
            x,
            self.weight,
            self.strides,
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias is not None:
            y = y + self.bias
        if was_unbatched:
            y = y[0]
        return y


class ConvTranspose2d(eqx.Module):
    """Transpose 2-D convolution in NHWC layout."""

    weight: jnp.ndarray  # (kH, kW, out_ch, in_ch)
    bias: Optional[jnp.ndarray]
    strides: Tuple[int, int] = eqx.field(static=True)
    padding: str = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        strides: Tuple[int, int] = (2, 2),
        padding: str = "SAME",
        use_bias: bool = False,
        *,
        key: jax.Array,
    ):
        kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if not isinstance(kernel_size, int):
            kw = kernel_size[1]
        fan_in = in_channels * kh * kw
        std = 1.0 / jnp.sqrt(fan_in)
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (kh, kw, out_channels, in_channels)) * std
        self.bias = jnp.zeros(out_channels) if use_bias else None
        self.strides = strides
        self.padding = padding

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x[None]
        y = jax.lax.conv_transpose(
            x,
            self.weight,
            self.strides,
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias is not None:
            y = y + self.bias
        if was_unbatched:
            y = y[0]
        return y
