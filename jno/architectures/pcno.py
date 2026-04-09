# pcno.py - JAX/Equinox implementation - https://github.com/PKU-CMEGroup/NeuralOperator/tree/main

from typing import Callable, List, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from .linear import Linear
from .common import get_activation as _get_act
from .common import compute_Fourier_modes as compute_Fourier_modes_helper


def scaled_sigmoid(x: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
    """Applies sigmoid scaled to [min_val, max_val]."""
    return min_val + (max_val - min_val) * jax.nn.sigmoid(x)


def scaled_logit(y: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
    """Inverse of scaled_sigmoid."""
    return jnp.log((y - min_val) / (max_val - y))


def compute_Fourier_modes(ndims: int, nks: Sequence[int], Ls: Sequence[float]) -> np.ndarray:
    """
    Compute nmeasures sets of Fourier modes.

    Returns:
        k_pairs: float[nmodes, ndims, nmeasures]
    """
    assert len(nks) == len(Ls)
    nmeasures = len(nks) // ndims
    k_pairs = np.stack([compute_Fourier_modes_helper(ndims, nks[i * ndims : (i + 1) * ndims], Ls[i * ndims : (i + 1) * ndims]) for i in range(nmeasures)], axis=-1)
    return k_pairs


def compute_Fourier_bases(nodes: jnp.ndarray, modes: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute Fourier bases for the whole space.

    Args:
        nodes: float[batch_size, nnodes, ndims]
        modes: float[nmodes, ndims, nmeasures]

    Returns:
        bases_c, bases_s: float[batch_size, nnodes, nmodes, nmeasures]
        bases_0: float[batch_size, nnodes, 1, nmeasures]
    """
    # temp: float[batch_size, nnodes, nmodes, nmeasures]
    temp = jnp.einsum("bxd,kdw->bxkw", nodes, modes)

    bases_c = jnp.cos(temp)
    bases_s = jnp.sin(temp)
    batch_size, nnodes, _, nmeasures = temp.shape
    bases_0 = jnp.ones((batch_size, nnodes, 1, nmeasures), dtype=temp.dtype)
    return bases_c, bases_s, bases_0


def compute_gradient(f: jnp.ndarray, directed_edges: jnp.ndarray, edge_gradient_weights: jnp.ndarray) -> jnp.ndarray:
    """
    Compute gradient of field f at each node using least squares.

    Args:
        f: float[batch_size, in_channels, nnodes]
        directed_edges: int[batch_size, max_nedges, 2]
        edge_gradient_weights: float[batch_size, max_nedges, ndims]

    Returns:
        f_gradients: float[batch_size, in_channels*ndims, max_nnodes]
    """
    f = jnp.transpose(f, (0, 2, 1))  # [batch_size, nnodes, in_channels]
    batch_size, max_nnodes, in_channels = f.shape
    _, max_nedges, ndims = edge_gradient_weights.shape

    target = directed_edges[..., 0]  # [batch_size, max_nedges]
    source = directed_edges[..., 1]  # [batch_size, max_nedges]

    # Gather source and target values
    batch_indices = jnp.arange(batch_size)[:, None]  # [batch_size, 1]
    f_source = f[batch_indices, source]  # [batch_size, max_nedges, in_channels]
    f_target = f[batch_indices, target]  # [batch_size, max_nedges, in_channels]

    # Compute message: edge_gradient_weights * (f_source - f_target)
    diff = f_source - f_target  # [batch_size, max_nedges, in_channels]
    message = jnp.einsum("bed,bec->becd", edge_gradient_weights, diff)
    message = message.reshape(batch_size, max_nedges, in_channels * ndims)

    # Scatter add to accumulate gradients
    f_gradients = jnp.zeros((batch_size, max_nnodes, in_channels * ndims), dtype=message.dtype)

    # Use segment_sum with sorted indices for scatter_add equivalent
    for b in range(batch_size):
        f_gradients = f_gradients.at[b].add(jax.ops.segment_sum(message[b], target[b], num_segments=max_nnodes))

    return jnp.transpose(f_gradients, (0, 2, 1))


class SpectralConv(eqx.Module):
    """Spectral convolution layer for PCNO."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    nmodes: int = eqx.field(static=True)
    nmeasures: int = eqx.field(static=True)
    weights_c: jnp.ndarray
    weights_s: jnp.ndarray
    weights_0: jnp.ndarray

    def __init__(self, in_channels: int, out_channels: int, nmodes: int, nmeasures: int, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nmodes = nmodes
        self.nmeasures = nmeasures

        scale = 1.0 / (in_channels * out_channels)
        k1, k2, k3 = jax.random.split(key, 3)
        self.weights_c = jax.random.uniform(k1, (in_channels, out_channels, nmodes, nmeasures), minval=-scale, maxval=scale)
        self.weights_s = jax.random.uniform(k2, (in_channels, out_channels, nmodes, nmeasures), minval=-scale, maxval=scale)
        self.weights_0 = jax.random.uniform(k3, (in_channels, out_channels, 1, nmeasures), minval=-scale, maxval=scale)

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
            bases_c, bases_s: float[batch_size, nnodes, nmodes, nmeasures]
            bases_0: float[batch_size, nnodes, 1, nmeasures]
            wbases_c, wbases_s: float[batch_size, nnodes, nmodes, nmeasures]
            wbases_0: float[batch_size, nnodes, 1, nmeasures]

        Returns:
            x: float[batch_size, out_channels, nnodes]
        """
        # Forward Fourier transform
        x_c_hat = jnp.einsum("bix,bxkw->bikw", x, wbases_c)
        x_s_hat = -jnp.einsum("bix,bxkw->bikw", x, wbases_s)
        x_0_hat = jnp.einsum("bix,bxkw->bikw", x, wbases_0)

        # Apply weights in Fourier space
        f_c_hat = jnp.einsum("bikw,iokw->bokw", x_c_hat, self.weights_c) - jnp.einsum("bikw,iokw->bokw", x_s_hat, self.weights_s)
        f_s_hat = jnp.einsum("bikw,iokw->bokw", x_s_hat, self.weights_c) + jnp.einsum("bikw,iokw->bokw", x_c_hat, self.weights_s)
        f_0_hat = jnp.einsum("bikw,iokw->bokw", x_0_hat, self.weights_0)

        # Inverse Fourier transform
        x = jnp.einsum("bokw,bxkw->box", f_0_hat, bases_0) + 2 * jnp.einsum("bokw,bxkw->box", f_c_hat, bases_c) - 2 * jnp.einsum("bokw,bxkw->box", f_s_hat, bases_s)

        return x


class PCNO(eqx.Module):
    """
    Point Cloud Neural Operator.

    The network architecture:
    1. Lift input to desired channel dimension via fc0
    2. Multiple layers of point cloud neural layers: u' = (W + K + D)(u)
       - W: linear functions (self.ws)
       - K: integral operator (self.sp_convs)
       - D: differential operator (self.gws)
    3. Project to output space via fc1 and fc2

    Args:
        ndims: Dimensionality of the problem (1, 2, or 3)
        modes: float[nmodes, ndims, nmeasures] - Fourier mode vectors
        nmeasures: Number of measures
        layers: List of channel dimensions for each layer
        fc_dim: Hidden dimension for projection (0 for no hidden layer)
        in_dim: Number of input channels
        out_dim: Number of output channels
        inv_L_scale_min: Minimum value for inverse length scale
        inv_L_scale_max: Maximum value for inverse length scale
        train_inv_L_scale: Whether to train the inverse length scale
        act: Activation function name
        key: PRNG key for parameter initialization
    """

    ndims: int = eqx.field(static=True)
    modes: jnp.ndarray  # float[nmodes, ndims, nmeasures] — non-trainable
    nmeasures: int = eqx.field(static=True)
    layers: Sequence[int] = eqx.field(static=True)
    fc_dim: int = eqx.field(static=True)
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    inv_L_scale_min: float = eqx.field(static=True)
    inv_L_scale_max: float = eqx.field(static=True)
    train_inv_L_scale: bool = eqx.field(static=True)
    act: str = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    nmodes: int = eqx.field(static=True)

    inv_L_scale_latent: jnp.ndarray  # trainable parameter
    fc0: Linear
    sp_convs: list
    ws: list
    gws: list
    fc1: Optional[Linear]
    fc2: Linear

    def __init__(
        self,
        ndims: int,
        modes: jnp.ndarray,
        nmeasures: int,
        layers: Sequence[int],
        fc_dim: int = 128,
        in_dim: int = 3,
        out_dim: int = 1,
        inv_L_scale_min: float = 0.5,
        inv_L_scale_max: float = 2.0,
        train_inv_L_scale: bool = True,
        act: str = "gelu",
        *,
        key,
    ):
        self.ndims = ndims
        self.modes = modes
        self.nmeasures = nmeasures
        self.layers = list(layers)
        self.fc_dim = fc_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inv_L_scale_min = inv_L_scale_min
        self.inv_L_scale_max = inv_L_scale_max
        self.train_inv_L_scale = train_inv_L_scale
        self.act = act
        self.activation = _get_act(act)
        self.nmodes = modes.shape[0]

        length = len(layers) - 1

        # Split keys for all sub-modules:
        # 1 inv_L_scale_latent + 1 fc0 + length sp_convs + length ws + length gws + up to 2 fc
        num_keys = 1 + 1 + 3 * length + 2
        keys = jax.random.split(key, num_keys)
        key_idx = 0

        # Learnable length scale parameter
        self.inv_L_scale_latent = jnp.full(
            (ndims, nmeasures),
            scaled_logit(jnp.array(1.0), inv_L_scale_min, inv_L_scale_max),
        )
        key_idx += 1

        # Lifting layer: Dense(in_dim -> layers[0])
        self.fc0 = Linear(in_dim, layers[0], key=keys[key_idx])
        key_idx += 1

        # Spectral convolution layers, linear (W) layers, and gradient (D) layers
        sp_convs = []
        ws = []
        gws = []
        for i in range(length):
            in_ch = layers[i]
            out_ch = layers[i + 1]
            sp_convs.append(SpectralConv(in_channels=in_ch, out_channels=out_ch, nmodes=self.nmodes, nmeasures=nmeasures, key=keys[key_idx]))
            key_idx += 1
            ws.append(Linear(in_ch, out_ch, key=keys[key_idx]))
            key_idx += 1
            gws.append(Linear(in_ch * ndims, out_ch, key=keys[key_idx]))
            key_idx += 1
        self.sp_convs = sp_convs
        self.ws = ws
        self.gws = gws

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
        directed_edges: jnp.ndarray,
        edge_gradient_weights: jnp.ndarray,
        training: bool = True,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: float[batch_size, max_nnodes, in_dim] - Input data
            node_mask: int[batch_size, max_nnodes, 1] - 1 for node, 0 for padding
            nodes: float[batch_size, max_nnodes, ndim] - Nodal coordinates
            node_weights: float[batch_size, max_nnodes, nmeasures] - Integration weights
            directed_edges: int[batch_size, max_nedges, 2] - Edge pairs
            edge_gradient_weights: float[batch_size, max_nedges, ndim] - Gradient weights
            training: Whether in training mode

        Returns:
            output: float[batch_size, max_nnodes, out_dim]
        """
        length = len(self.layers) - 1

        # Scale the modes
        inv_L_scale = scaled_sigmoid(self.inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)
        scaled_modes = self.modes * inv_L_scale

        # Compute Fourier bases
        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, scaled_modes)

        # Weight bases by node weights
        wbases_c = jnp.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = jnp.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = jnp.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        # Lifting layer: apply fc0 on last dim [batch, nodes, in_dim] -> [batch, nodes, layers[0]]
        x = jax.vmap(jax.vmap(self.fc0))(x)
        x = jnp.transpose(x, (0, 2, 1))  # [batch, channels, nodes]

        # Main layers
        for i in range(length):
            # Spectral convolution (integral operator K)
            x1 = self.sp_convs[i](x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)

            # Linear transform (W) - replaces 1x1 conv
            # x: [batch, channels, nodes] -> transpose to [batch, nodes, channels],
            # apply linear on last dim, transpose back
            x_for_w = jnp.transpose(x, (0, 2, 1))  # [batch, nodes, channels]
            x2 = jax.vmap(jax.vmap(self.ws[i]))(x_for_w)  # [batch, nodes, out_ch]
            x2 = jnp.transpose(x2, (0, 2, 1))  # [batch, out_ch, nodes]

            # Gradient operator (D)
            grad = compute_gradient(x, directed_edges, edge_gradient_weights)
            grad = jax.nn.soft_sign(grad)
            grad = jnp.transpose(grad, (0, 2, 1))  # [batch, nodes, channels*ndims]
            x3 = jax.vmap(jax.vmap(self.gws[i]))(grad)  # [batch, nodes, out_ch]
            x3 = jnp.transpose(x3, (0, 2, 1))  # [batch, out_ch, nodes]

            x = x1 + x2 + x3

            if self.activation is not None and i != length - 1:
                x = self.activation(x)

        x = jnp.transpose(x, (0, 2, 1))  # [batch, nodes, channels]

        # Projection layers
        if self.fc1 is not None:
            x = jax.vmap(jax.vmap(self.fc1))(x)
            if self.activation is not None:
                x = self.activation(x)

        x = jax.vmap(jax.vmap(self.fc2))(x)

        return x * node_mask
