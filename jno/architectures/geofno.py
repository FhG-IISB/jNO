# geofno.py - JAX/Flax implementation

from typing import Callable, List, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def _get_act(act: str) -> Optional[Callable]:
    """Get activation function by name."""
    activations = {
        "tanh": nn.tanh,
        "gelu": nn.gelu,
        "relu": nn.relu,
        "elu": nn.elu,
        "leaky_relu": nn.leaky_relu,
        "none": None,
    }
    if act not in activations:
        raise ValueError(f"{act} is not supported")
    return activations[act]


def compute_Fourier_modes(ndims: int, nks: Sequence[int], Ls: Sequence[float]) -> np.ndarray:
    """
    Compute Fourier modes number k.
    Fourier bases are cos(kx), sin(kx), 1.
    We cannot have both k and -k.

    Args:
        ndims: Number of spatial dimensions
        nks: Number of modes per dimension
        Ls: Domain lengths per dimension

    Returns:
        k_pairs: float[nmodes, ndims] - Fourier mode vectors
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

    # Sort by magnitude
    k_pairs = k_pairs[np.argsort(k_pair_mag, kind="stable"), :]
    return k_pairs


def compute_Fourier_bases(nodes: jnp.ndarray, modes: jnp.ndarray, node_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute Fourier bases.
    Fourier bases are cos(kx), sin(kx), 1.

    Args:
        nodes: float[batch_size, nnodes, ndims] - Node coordinates
        modes: float[nmodes, ndims] - Fourier mode vectors
        node_mask: float[batch_size, nnodes, 1] - Mask (1 for valid nodes)

    Returns:
        bases_c, bases_s: float[batch_size, nnodes, nmodes] - Cosine and sine bases
        bases_0: float[batch_size, nnodes, 1] - Constant basis
    """

    # temp: float[batch_size, nnodes, nmodes]
    temp = jnp.einsum("bxd,kd->bxk", nodes, modes)

    bases_c = jnp.cos(temp) * node_mask
    bases_s = jnp.sin(temp) * node_mask
    bases_0 = node_mask

    return bases_c, bases_s, bases_0


class SpectralConvGeo(nn.Module):
    """
    Spectral convolution layer for GeoFNO.

    Operates in Fourier space using precomputed bases for arbitrary geometries.
    """

    in_channels: int
    out_channels: int
    nmodes: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        bases_c: jnp.ndarray,
        bases_s: jnp.ndarray,
        bases_0: jnp.ndarray,
        wbases_c: jnp.ndarray,
        wbases_s: jnp.ndarray,
        wbases_0: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute Fourier neural layer.

        Args:
            x: float[batch_size, in_channels, nnodes]
            bases_c, bases_s: float[batch_size, nnodes, nmodes]
            bases_0: float[batch_size, nnodes, 1]
            wbases_c, wbases_s: float[batch_size, nnodes, nmodes] - Weighted bases
            wbases_0: float[batch_size, nnodes, 1]

        Returns:
            x: float[batch_size, out_channels, nnodes]
        """
        scale = 1.0 / (self.in_channels * self.out_channels)

        # Learnable weights for Fourier coefficients
        weights_c = self.param("weights_c", nn.initializers.uniform(scale), (self.in_channels, self.out_channels, self.nmodes))
        weights_s = self.param("weights_s", nn.initializers.uniform(scale), (self.in_channels, self.out_channels, self.nmodes))
        weights_0 = self.param("weights_0", nn.initializers.uniform(scale), (self.in_channels, self.out_channels, 1))

        # Forward Fourier transform (projection onto bases)
        x_c_hat = jnp.einsum("bix,bxk->bik", x, wbases_c)
        x_s_hat = -jnp.einsum("bix,bxk->bik", x, wbases_s)
        x_0_hat = jnp.einsum("bix,bxk->bik", x, wbases_0)

        # Apply weights in Fourier space (complex multiplication)
        f_c_hat = jnp.einsum("bik,iok->bok", x_c_hat, weights_c) - jnp.einsum("bik,iok->bok", x_s_hat, weights_s)
        f_s_hat = jnp.einsum("bik,iok->bok", x_s_hat, weights_c) + jnp.einsum("bik,iok->bok", x_c_hat, weights_s)
        f_0_hat = jnp.einsum("bik,iok->bok", x_0_hat, weights_0)

        # Inverse Fourier transform
        x = jnp.einsum("bok,bxk->box", f_0_hat, bases_0) + 2 * jnp.einsum("bok,bxk->box", f_c_hat, bases_c) - 2 * jnp.einsum("bok,bxk->box", f_s_hat, bases_s)

        return x


class GeoFNO(nn.Module):
    """
    Geometry-aware Fourier Neural Operator.

    The network architecture:
    1. Lift the input to the desired channel dimension via fc0.
    2. len(layers)-1 layers of Fourier neural layers: u' = (W + K)(u)
       - W: linear functions (self.ws)
       - K: integral operator (self.sp_convs)
    3. Project from channel space to output space via fc1 and fc2.

    Args:
        ndims: Dimensionality of the problem (1, 2, or 3)
        modes: float[nmodes, ndims] - Fourier mode vectors
        layers: List of channel dimensions for each layer
        fc_dim: Hidden dimension for projection (0 for no hidden layer)
        in_dim: Number of input channels
        out_dim: Number of output channels
        act: Activation function name
    """

    ndims: int
    modes: jnp.ndarray  # float[nmodes, ndims]
    layers: Sequence[int]
    fc_dim: int = 128
    in_dim: int = 3
    out_dim: int = 1
    act: str = "gelu"

    def setup(self):
        self.activation = _get_act(self.act)
        self.nmodes = self.modes.shape[0]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        node_mask: jnp.ndarray,
        nodes: jnp.ndarray,
        node_weights: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: float[batch_size, max_nnodes, in_dim] - Input data
            node_mask: float[batch_size, max_nnodes, 1] - 1 for valid node, 0 for padding
            nodes: float[batch_size, max_nnodes, ndims] - Node coordinates
            node_weights: float[batch_size, max_nnodes, 1] - Integration weights (rho(x)dx)
            training: Whether in training mode

        Returns:
            output: float[batch_size, max_nnodes, out_dim]
        """
        x = x[None, ...]
        node_mask = node_mask[None, ...]
        nodes = nodes[None, ...]
        node_weights = node_weights[None, ...]

        length = len(self.layers) - 1

        # Compute Fourier bases
        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, self.modes, node_mask)

        # Weight bases by integration weights
        wbases_c = bases_c * node_weights
        wbases_s = bases_s * node_weights
        wbases_0 = bases_0 * node_weights

        # Lifting layer
        x = nn.Dense(features=self.layers[0], name="fc0")(x)
        x = jnp.transpose(x, (0, 2, 1))  # [batch, channels, nodes]

        # Fourier layers
        for i in range(length):
            in_ch = self.layers[i]
            out_ch = self.layers[i + 1]

            # Spectral convolution (integral operator K)
            x1 = SpectralConvGeo(in_channels=in_ch, out_channels=out_ch, nmodes=self.nmodes, name=f"sp_conv_{i}")(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)

            # Linear transform (W) - 1x1 convolution
            x2 = nn.Conv(features=out_ch, kernel_size=(1,), use_bias=True, name=f"w_{i}")(jnp.transpose(x, (0, 2, 1)))
            x2 = jnp.transpose(x2, (0, 2, 1))

            x = x1 + x2

            # Activation (except last layer)
            if self.activation is not None and i != length - 1:
                x = self.activation(x)

        x = jnp.transpose(x, (0, 2, 1))  # [batch, nodes, channels]

        # Projection layers
        if self.fc_dim > 0:
            x = nn.Dense(features=self.fc_dim, name="fc1")(x)
            if self.activation is not None:
                x = self.activation(x)

        x = nn.Dense(features=self.out_dim, name="fc2")(x)

        # Apply mask to output
        masked = x * node_mask
        return jnp.squeeze(masked)
