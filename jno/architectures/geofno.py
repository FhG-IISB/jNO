# geofno.py - JAX/Equinox implementation

from typing import Callable, List, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from .linear import Linear
from .common import get_activation as _get_act, compute_Fourier_modes


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


class SpectralConvGeo(eqx.Module):
    """
    Spectral convolution layer for GeoFNO.

    Operates in Fourier space using precomputed bases for arbitrary geometries.
    """

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    nmodes: int = eqx.field(static=True)
    weights_c: jnp.ndarray
    weights_s: jnp.ndarray
    weights_0: jnp.ndarray

    def __init__(self, in_channels: int, out_channels: int, nmodes: int, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nmodes = nmodes

        scale = 1.0 / (in_channels * out_channels)
        k1, k2, k3 = jax.random.split(key, 3)
        self.weights_c = jax.random.uniform(k1, (in_channels, out_channels, nmodes), minval=-scale, maxval=scale)
        self.weights_s = jax.random.uniform(k2, (in_channels, out_channels, nmodes), minval=-scale, maxval=scale)
        self.weights_0 = jax.random.uniform(k3, (in_channels, out_channels, 1), minval=-scale, maxval=scale)

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
        # Forward Fourier transform (projection onto bases)
        x_c_hat = jnp.einsum("bix,bxk->bik", x, wbases_c)
        x_s_hat = -jnp.einsum("bix,bxk->bik", x, wbases_s)
        x_0_hat = jnp.einsum("bix,bxk->bik", x, wbases_0)

        # Apply weights in Fourier space (complex multiplication)
        f_c_hat = jnp.einsum("bik,iok->bok", x_c_hat, self.weights_c) - jnp.einsum("bik,iok->bok", x_s_hat, self.weights_s)
        f_s_hat = jnp.einsum("bik,iok->bok", x_s_hat, self.weights_c) + jnp.einsum("bik,iok->bok", x_c_hat, self.weights_s)
        f_0_hat = jnp.einsum("bik,iok->bok", x_0_hat, self.weights_0)

        # Inverse Fourier transform
        x = jnp.einsum("bok,bxk->box", f_0_hat, bases_0) + 2 * jnp.einsum("bok,bxk->box", f_c_hat, bases_c) - 2 * jnp.einsum("bok,bxk->box", f_s_hat, bases_s)

        return x


class GeoFNO(eqx.Module):
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
        key: PRNG key for parameter initialization
    """

    ndims: int = eqx.field(static=True)
    modes: jnp.ndarray  # float[nmodes, ndims] — non-trainable
    layers: Sequence[int] = eqx.field(static=True)
    fc_dim: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    act: str = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    nmodes: int = eqx.field(static=True)

    fc0: Linear
    sp_convs: list
    ws: list
    fc1: Optional[Linear]
    fc2: Linear

    def __init__(
        self,
        ndims: int,
        modes: jnp.ndarray,
        layers: Sequence[int],
        fc_dim: int = 128,
        in_dim: int = 3,
        out_dim: int = 1,
        act: str = "gelu",
        *,
        key,
    ):
        self.ndims = ndims
        self.modes = modes
        self.layers = list(layers)
        self.fc_dim = fc_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.act = act
        self.activation = _get_act(act)
        self.nmodes = modes.shape[0]

        length = len(layers) - 1

        # Split keys for all sub-modules
        keys = jax.random.split(key, 2 * length + 3)
        key_idx = 0

        # Lifting layer: Dense(in_dim -> layers[0])
        self.fc0 = Linear(in_dim, layers[0], key=keys[key_idx])
        key_idx += 1

        # Spectral convolution layers and linear (1x1 conv) layers
        sp_convs = []
        ws = []
        for i in range(length):
            in_ch = layers[i]
            out_ch = layers[i + 1]
            sp_convs.append(SpectralConvGeo(in_channels=in_ch, out_channels=out_ch, nmodes=self.nmodes, key=keys[key_idx]))
            key_idx += 1
            ws.append(Linear(in_ch, out_ch, key=keys[key_idx]))
            key_idx += 1
        self.sp_convs = sp_convs
        self.ws = ws

        # Projection layers
        if fc_dim > 0:
            self.fc1 = Linear(layers[-1], fc_dim, key=keys[key_idx])
            key_idx += 1
            self.fc2 = Linear(fc_dim, out_dim, key=keys[key_idx])
        else:
            self.fc1 = None
            self.fc2 = Linear(layers[-1], out_dim, key=keys[key_idx])

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

        # Lifting layer: apply fc0 on last dim [batch, nodes, in_dim] -> [batch, nodes, layers[0]]
        x = jax.vmap(jax.vmap(self.fc0))(x)
        x = jnp.transpose(x, (0, 2, 1))  # [batch, channels, nodes]

        # Fourier layers
        for i in range(length):
            # Spectral convolution (integral operator K)
            x1 = self.sp_convs[i](x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)

            # Linear transform (W) - replaces 1x1 conv
            # x: [batch, channels, nodes] -> transpose to [batch, nodes, channels],
            # apply linear on last dim, transpose back
            x_t = jnp.transpose(x, (0, 2, 1))  # [batch, nodes, channels]
            x2 = jax.vmap(jax.vmap(self.ws[i]))(x_t)  # [batch, nodes, out_ch]
            x2 = jnp.transpose(x2, (0, 2, 1))  # [batch, out_ch, nodes]

            x = x1 + x2

            # Activation (except last layer)
            if self.activation is not None and i != length - 1:
                x = self.activation(x)

        x = jnp.transpose(x, (0, 2, 1))  # [batch, nodes, channels]

        # Projection layers
        if self.fc1 is not None:
            x = jax.vmap(jax.vmap(self.fc1))(x)
            if self.activation is not None:
                x = self.activation(x)

        x = jax.vmap(jax.vmap(self.fc2))(x)

        # Apply mask to output
        masked = x * node_mask
        return jnp.squeeze(masked)
