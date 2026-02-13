# pcno.py - JAX/Flax implementation - https://github.com/PKU-CMEGroup/NeuralOperator/tree/main

from typing import Callable, Optional, Sequence, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np


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


def scaled_sigmoid(x: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
    """Applies sigmoid scaled to [min_val, max_val]."""
    return min_val + (max_val - min_val) * jax.nn.sigmoid(x)


def scaled_logit(y: jnp.ndarray, min_val: float, max_val: float) -> jnp.ndarray:
    """Inverse of scaled_sigmoid."""
    return jnp.log((y - min_val) / (max_val - y))


def compute_Fourier_modes_helper(ndims: int, nks: Sequence[int], Ls: Sequence[float]) -> np.ndarray:
    """
    Compute Fourier modes k for bases cos(kx), sin(kx), 1.
    We cannot have both k and -k, cannot have 0.
    """
    assert len(nks) == len(Ls) == ndims

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
                    k_pairs[i, :] = 2 * np.pi / Lx * kx, 2 * np.pi / Ly * ky, 2 * np.pi / Lz * kz
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")

    k_pairs = k_pairs[np.argsort(k_pair_mag, kind="stable"), :]
    return k_pairs


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


class SpectralConv(nn.Module):
    """Spectral convolution layer for PCNO."""

    in_channels: int
    out_channels: int
    nmodes: int
    nmeasures: int

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
            bases_c, bases_s: float[batch_size, nnodes, nmodes, nmeasures]
            bases_0: float[batch_size, nnodes, 1, nmeasures]
            wbases_c, wbases_s: float[batch_size, nnodes, nmodes, nmeasures]
            wbases_0: float[batch_size, nnodes, 1, nmeasures]

        Returns:
            x: float[batch_size, out_channels, nnodes]
        """
        scale = 1.0 / (self.in_channels * self.out_channels)

        weights_c = self.param("weights_c", nn.initializers.uniform(scale), (self.in_channels, self.out_channels, self.nmodes, self.nmeasures))
        weights_s = self.param("weights_s", nn.initializers.uniform(scale), (self.in_channels, self.out_channels, self.nmodes, self.nmeasures))
        weights_0 = self.param("weights_0", nn.initializers.uniform(scale), (self.in_channels, self.out_channels, 1, self.nmeasures))

        # Forward Fourier transform
        x_c_hat = jnp.einsum("bix,bxkw->bikw", x, wbases_c)
        x_s_hat = -jnp.einsum("bix,bxkw->bikw", x, wbases_s)
        x_0_hat = jnp.einsum("bix,bxkw->bikw", x, wbases_0)

        # Apply weights in Fourier space
        f_c_hat = jnp.einsum("bikw,iokw->bokw", x_c_hat, weights_c) - jnp.einsum("bikw,iokw->bokw", x_s_hat, weights_s)
        f_s_hat = jnp.einsum("bikw,iokw->bokw", x_s_hat, weights_c) + jnp.einsum("bikw,iokw->bokw", x_c_hat, weights_s)
        f_0_hat = jnp.einsum("bikw,iokw->bokw", x_0_hat, weights_0)

        # Inverse Fourier transform
        x = jnp.einsum("bokw,bxkw->box", f_0_hat, bases_0) + 2 * jnp.einsum("bokw,bxkw->box", f_c_hat, bases_c) - 2 * jnp.einsum("bokw,bxkw->box", f_s_hat, bases_s)

        return x


class PCNO(nn.Module):
    """
    Point Cloud Neural Operator.

    The network architecture:
    1. Lift input to desired channel dimension via fc0
    2. Multiple layers of point cloud neural layers: u' = (W + K + D)(u)
       - W: linear functions (self.ws)
       - K: integral operator (self.sp_convs)
       - D: differential operator (self.gws)
    3. Project to output space via fc1 and fc2

    """

    ndims: int
    modes: jnp.ndarray  # float[nmodes, ndims, nmeasures]
    nmeasures: int
    layers: Sequence[int]
    fc_dim: int = 128
    in_dim: int = 3
    out_dim: int = 1
    inv_L_scale_min: float = 0.5
    inv_L_scale_max: float = 2.0
    train_inv_L_scale: bool = True
    act: str = "gelu"

    def setup(self):
        self.activation = _get_act(self.act)
        nmodes = self.modes.shape[0]

        # Spectral convolution layers
        self.sp_convs = [
            SpectralConv(
                in_channels=in_size,
                out_channels=out_size,
                nmodes=nmodes,
                nmeasures=self.nmeasures,
            )
            for in_size, out_size in zip(self.layers[:-1], self.layers[1:])
        ]

        # Linear layers (W)
        self.ws = [nn.Conv(features=out_size, kernel_size=(1,)) for out_size in self.layers[1:]]

        # Gradient layers (D)
        self.gws = [nn.Conv(features=out_size, kernel_size=(1,)) for in_size, out_size in zip(self.layers[:-1], self.layers[1:])]

    @nn.compact
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

        # Learnable length scale parameter
        inv_L_scale_latent = self.param("inv_L_scale_latent", lambda rng, shape: jnp.full(shape, scaled_logit(jnp.array(1.0), self.inv_L_scale_min, self.inv_L_scale_max)), (self.ndims, self.nmeasures))

        # Scale the modes
        inv_L_scale = scaled_sigmoid(inv_L_scale_latent, self.inv_L_scale_min, self.inv_L_scale_max)
        scaled_modes = self.modes * inv_L_scale

        # Compute Fourier bases
        bases_c, bases_s, bases_0 = compute_Fourier_bases(nodes, scaled_modes)

        # Weight bases by node weights
        wbases_c = jnp.einsum("bxkw,bxw->bxkw", bases_c, node_weights)
        wbases_s = jnp.einsum("bxkw,bxw->bxkw", bases_s, node_weights)
        wbases_0 = jnp.einsum("bxkw,bxw->bxkw", bases_0, node_weights)

        # Lifting layer
        x = nn.Dense(features=self.layers[0], name="fc0")(x)
        x = jnp.transpose(x, (0, 2, 1))  # [batch, channels, nodes]

        # Main layers
        for i, (speconv, w, gw) in enumerate(zip(self.sp_convs, self.ws, self.gws)):
            # Spectral convolution (integral operator K)
            x1 = speconv(x, bases_c, bases_s, bases_0, wbases_c, wbases_s, wbases_0)

            # Linear transform (W)
            x_for_w = jnp.transpose(x, (0, 2, 1))  # [batch, nodes, channels]
            x2 = w(x_for_w)
            x2 = jnp.transpose(x2, (0, 2, 1))  # [batch, channels, nodes]

            # Gradient operator (D)
            grad = compute_gradient(x, directed_edges, edge_gradient_weights)
            grad = jax.nn.soft_sign(grad)
            grad = jnp.transpose(grad, (0, 2, 1))  # [batch, nodes, channels*ndims]
            x3 = gw(grad)
            x3 = jnp.transpose(x3, (0, 2, 1))  # [batch, channels, nodes]

            x = x1 + x2 + x3

            if self.activation is not None and i != length - 1:
                x = self.activation(x)

        x = jnp.transpose(x, (0, 2, 1))  # [batch, nodes, channels]

        # Projection layers
        if self.fc_dim > 0:
            x = nn.Dense(features=self.fc_dim, name="fc1")(x)
            if self.activation is not None:
                x = self.activation(x)

        x = nn.Dense(features=self.out_dim, name="fc2")(x)

        return x * node_mask
