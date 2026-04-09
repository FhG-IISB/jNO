# pit.py - JAX/Equinox implementation

from typing import List, Sequence, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from .linear import Linear


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


def pairwise_dist(res1x: int, res1y: int, res2x: int, res2y: int) -> jnp.ndarray:
    """
    Compute pairwise squared distances between two grids.

    Args:
        res1x, res1y: Resolution of first grid
        res2x, res2y: Resolution of second grid

    Returns:
        dist: float[res1x*res1y, res2x*res2y] - Scaled squared distances
    """
    # Create first grid
    gridx1 = jnp.linspace(0, 1, res1x + 1)[:-1]
    gridy1 = jnp.linspace(0, 1, res1y + 1)[:-1]
    grid1x, grid1y = jnp.meshgrid(gridx1, gridy1, indexing="xy")
    grid1 = jnp.stack([grid1x.flatten(), grid1y.flatten()], axis=-1)  # [res1x*res1y, 2]

    # Create second grid
    gridx2 = jnp.linspace(0, 1, res2x + 1)[:-1]
    gridy2 = jnp.linspace(0, 1, res2y + 1)[:-1]
    grid2x, grid2y = jnp.meshgrid(gridx2, gridy2, indexing="xy")
    grid2 = jnp.stack([grid2x.flatten(), grid2y.flatten()], axis=-1)  # [res2x*res2y, 2]

    # Compute pairwise distances
    # grid1: [N1, 2], grid2: [N2, 2]
    # diff: [N1, N2, 2]
    diff = grid1[:, None, :] - grid2[None, :, :]
    dist = jnp.sum(diff**2, axis=-1)  # [N1, N2]

    return (dist / 2.0).astype(_default_float_dtype())


def pairwise_dist_from_coords(coords1: jnp.ndarray, coords2: jnp.ndarray) -> jnp.ndarray:
    """
    Compute pairwise squared distances between two sets of coordinates.

    Args:
        coords1: float[N1, D] - First set of coordinates
        coords2: float[N2, D] - Second set of coordinates

    Returns:
        dist: float[N1, N2] - Scaled squared distances
    """
    diff = coords1[:, None, :] - coords2[None, :, :]
    dist = jnp.sum(diff**2, axis=-1)
    return (dist / 2.0).astype(_default_float_dtype())


class PiTMLP(eqx.Module):
    """
    A two-layer MLP with GELU activation for PiT.
    """

    fc1: Linear
    fc2: Linear
    in_channels: int = eqx.field(static=True)
    hid_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, *, key):
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        key1, key2 = jax.random.split(key)
        self.fc1 = Linear(in_channels, hid_channels, key=key1)
        self.fc2 = Linear(hid_channels, out_channels, key=key2)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = jax.vmap(jax.vmap(self.fc1))(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.fc2))(x)
        return x


class MultiHeadPosAtt(eqx.Module):
    """
    Multi-head position-attention mechanism.

    Supports global, local, and cross variants through the locality parameter.
    Attention weights are computed based on spatial distances rather than
    learned query-key products.
    """

    r: jnp.ndarray
    weight: jnp.ndarray
    n_head: int = eqx.field(static=True)
    hid_channels: int = eqx.field(static=True)
    locality: float = eqx.field(static=True)

    def __init__(self, n_head: int, hid_channels: int, locality: float, *, key):
        self.n_head = n_head
        self.hid_channels = hid_channels
        self.locality = locality

        v_dim = hid_channels // n_head
        key1, key2 = jax.random.split(key)

        # Learnable scaling parameter for distances
        self.r = jax.random.normal(key1, (n_head, 1, 1))

        # Learnable value projection weights
        self.weight = jax.random.normal(key2, (n_head, hid_channels, v_dim)) * (1.0 / jnp.sqrt(hid_channels))

    def __call__(self, m_dist: jnp.ndarray, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Apply position-based attention.

        Args:
            m_dist: float[n_head, N_query, N_key] - Pairwise distances
            x: float[batch_size, N_key, hid_channels] - Input features

        Returns:
            output: float[batch_size, N_query, hid_channels]
        """
        # Scale distances with learnable parameter
        # tan(0.25 * pi * (1 - eps) * (1 + sin(r))) maps r to positive scaling
        scale_factor = jnp.tan(0.25 * jnp.pi * (1 - 1e-7) * (1 + jnp.sin(self.r)))
        scaled_dist = m_dist * scale_factor  # [n_head, N_query, N_key]

        # Apply locality mask if needed
        if self.locality <= 100:
            # Compute threshold as quantile of distances
            threshold = jnp.percentile(scaled_dist, self.locality, axis=-1, keepdims=True)
            # Mask out distant positions
            scaled_dist = jnp.where(scaled_dist <= threshold, scaled_dist, jnp.inf)

        # Compute attention weights (softmax over negative distances)
        att = jax.nn.softmax(-scaled_dist, axis=-1)  # [n_head, N_query, N_key]

        # Project input to values
        # x: [batch, N_key, hid_channels]
        # weight: [n_head, hid_channels, v_dim]
        value = jnp.einsum("bnj,hjk->bhnk", x, self.weight)  # [batch, n_head, N_key, v_dim]

        # Apply attention
        # att: [n_head, N_query, N_key]
        # value: [batch, n_head, N_key, v_dim]
        out = jnp.einsum("hjn,bhnk->bhjk", att, value)  # [batch, n_head, N_query, v_dim]

        # Reshape: [batch, n_head, N_query, v_dim] -> [batch, N_query, n_head, v_dim]
        out = jnp.transpose(out, (0, 2, 1, 3))

        # Concatenate heads: [batch, N_query, n_head * v_dim]
        batch_size, n_query = out.shape[0], out.shape[1]
        out = out.reshape(batch_size, n_query, -1)

        return jax.nn.gelu(out)


class PiT(eqx.Module):
    """
    Position-induced Transformer.

    Built upon the multi-head position-attention mechanism where attention
    weights are derived from spatial distances rather than learned query-key
    products.

    Architecture:
    1. Encoder: Linear projection + downsampling attention
    2. Processor: Multiple blocks of position-attention + MLP
    3. Decoder: Upsampling attention + linear projection

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        hid_channels: Hidden dimension
        n_head: Number of attention heads
        localities: List of locality percentages [encoder, *processor, decoder]
        m_dists: List of precomputed distance matrices for each attention layer
    """

    en_fc1: Linear
    down: MultiHeadPosAtt
    PA_blocks: List[MultiHeadPosAtt]
    MLP_blocks: List[PiTMLP]
    W_blocks: List[Linear]
    up: MultiHeadPosAtt
    de_fc1: Linear
    de_fc2: Linear
    m_dists: Sequence[jnp.ndarray]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hid_channels: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    localities: Sequence[float] = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: int,
        n_head: int,
        localities: Sequence[float],
        m_dists: Sequence[jnp.ndarray],
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels
        self.n_head = n_head
        self.localities = tuple(localities)
        self.m_dists = m_dists
        self.n_blocks = len(localities) - 2

        en_locality = localities[0]
        de_locality = localities[-1]
        proc_localities = localities[1:-1]

        # Split keys for all sub-modules
        keys = jax.random.split(key, 3 + 3 * self.n_blocks + 2)
        idx = 0

        # Encoder
        self.en_fc1 = Linear(in_channels, hid_channels, key=keys[idx])
        idx += 1
        self.down = MultiHeadPosAtt(n_head=n_head, hid_channels=hid_channels, locality=en_locality, key=keys[idx])
        idx += 1

        # Processor
        PA_blocks = []
        MLP_blocks = []
        W_blocks = []
        for i in range(self.n_blocks):
            PA_blocks.append(MultiHeadPosAtt(n_head=n_head, hid_channels=hid_channels, locality=proc_localities[i], key=keys[idx]))
            idx += 1
            MLP_blocks.append(PiTMLP(in_channels=hid_channels, hid_channels=hid_channels, out_channels=hid_channels, key=keys[idx]))
            idx += 1
            W_blocks.append(Linear(hid_channels, hid_channels, key=keys[idx]))
            idx += 1
        self.PA_blocks = PA_blocks
        self.MLP_blocks = MLP_blocks
        self.W_blocks = W_blocks

        # Decoder
        self.up = MultiHeadPosAtt(n_head=n_head, hid_channels=hid_channels, locality=de_locality, key=keys[idx])
        idx += 1
        self.de_fc1 = Linear(hid_channels, hid_channels, key=keys[idx])
        idx += 1
        self.de_fc2 = Linear(hid_channels, out_channels, key=keys[idx])
        idx += 1

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: float[batch_size, n_points, in_channels] - Input features

        Returns:
            output: float[batch_size, n_points, out_channels]
        """
        # Handle unbatched input
        squeeze_batch = False
        if x.ndim == 2:
            x = x[None, ...]
            squeeze_batch = True

        # ============ Encoder ============
        x = jax.vmap(jax.vmap(self.en_fc1))(x)
        x = jax.nn.gelu(x)

        # Downsampling attention
        x = self.down(self.m_dists[0], x)

        # ============ Processor ============
        for i in range(self.n_blocks):
            # Position attention
            pa_out = self.PA_blocks[i](self.m_dists[i + 1], x)

            # MLP
            mlp_out = self.MLP_blocks[i](pa_out)

            # Residual connection
            residual = jax.vmap(jax.vmap(self.W_blocks[i]))(x)

            x = mlp_out + residual
            x = jax.nn.gelu(x)

        # ============ Decoder ============
        # Upsampling attention
        x = self.up(self.m_dists[-1], x)

        x = jax.vmap(jax.vmap(self.de_fc1))(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.de_fc2))(x)

        if squeeze_batch:
            x = x[0]

        return x


class PiTWithCoords(eqx.Module):
    """
    PiT variant that computes distance matrices from coordinates on-the-fly.

    Useful when coordinates are provided as input rather than precomputed.
    """

    en_fc1: Linear
    down: MultiHeadPosAtt
    PA_blocks: List[MultiHeadPosAtt]
    MLP_blocks: List[PiTMLP]
    W_blocks: List[Linear]
    up: MultiHeadPosAtt
    de_fc1: Linear
    de_fc2: Linear
    m_dist_down: jnp.ndarray
    m_dist_proc: jnp.ndarray
    m_dist_up: jnp.ndarray
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    hid_channels: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    localities: Sequence[float] = eqx.field(static=True)
    input_res: Tuple[int, int] = eqx.field(static=True)
    latent_res: Tuple[int, int] = eqx.field(static=True)
    output_res: Tuple[int, int] = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: int,
        n_head: int,
        localities: Sequence[float],
        input_res: Tuple[int, int] = (64, 64),
        latent_res: Tuple[int, int] = (16, 16),
        output_res: Tuple[int, int] = (64, 64),
        *,
        key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hid_channels = hid_channels
        self.n_head = n_head
        self.localities = tuple(localities)
        self.input_res = input_res
        self.latent_res = latent_res
        self.output_res = output_res
        self.n_blocks = len(localities) - 2

        en_locality = localities[0]
        de_locality = localities[-1]
        proc_localities = localities[1:-1]

        # Precompute distance matrices for regular grids
        # Encoder: input -> latent
        self.m_dist_down = pairwise_dist(latent_res[0], latent_res[1], input_res[0], input_res[1])
        # Processor: latent -> latent
        self.m_dist_proc = pairwise_dist(latent_res[0], latent_res[1], latent_res[0], latent_res[1])
        # Decoder: latent -> output
        self.m_dist_up = pairwise_dist(output_res[0], output_res[1], latent_res[0], latent_res[1])

        # Split keys for all sub-modules
        keys = jax.random.split(key, 3 + 3 * self.n_blocks + 2)
        idx = 0

        # Encoder
        self.en_fc1 = Linear(in_channels, hid_channels, key=keys[idx])
        idx += 1
        self.down = MultiHeadPosAtt(n_head=n_head, hid_channels=hid_channels, locality=en_locality, key=keys[idx])
        idx += 1

        # Processor
        PA_blocks = []
        MLP_blocks = []
        W_blocks = []
        for i in range(self.n_blocks):
            PA_blocks.append(MultiHeadPosAtt(n_head=n_head, hid_channels=hid_channels, locality=proc_localities[i], key=keys[idx]))
            idx += 1
            MLP_blocks.append(PiTMLP(in_channels=hid_channels, hid_channels=hid_channels, out_channels=hid_channels, key=keys[idx]))
            idx += 1
            W_blocks.append(Linear(hid_channels, hid_channels, key=keys[idx]))
            idx += 1
        self.PA_blocks = PA_blocks
        self.MLP_blocks = MLP_blocks
        self.W_blocks = W_blocks

        # Decoder
        self.up = MultiHeadPosAtt(n_head=n_head, hid_channels=hid_channels, locality=de_locality, key=keys[idx])
        idx += 1
        self.de_fc1 = Linear(hid_channels, hid_channels, key=keys[idx])
        idx += 1
        self.de_fc2 = Linear(hid_channels, out_channels, key=keys[idx])
        idx += 1

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Forward pass with internally computed distance matrices.

        Args:
            x: float[batch_size, n_points, in_channels]

        Returns:
            output: float[batch_size, n_points, out_channels]
        """

        squeeze_batch = False
        if x.ndim == 2:
            x = x[None, ...]
            squeeze_batch = True

        en_locality = self.localities[0]
        de_locality = self.localities[-1]
        proc_localities = self.localities[1:-1]

        # Broadcast distance matrices for multi-head attention
        # Shape: [n_head, N_query, N_key]
        m_dist_down = jnp.broadcast_to(self.m_dist_down[None, ...], (self.n_head,) + self.m_dist_down.shape)
        m_dist_proc = jnp.broadcast_to(self.m_dist_proc[None, ...], (self.n_head,) + self.m_dist_proc.shape)
        m_dist_up = jnp.broadcast_to(self.m_dist_up[None, ...], (self.n_head,) + self.m_dist_up.shape)

        # ============ Encoder ============
        x = jax.vmap(jax.vmap(self.en_fc1))(x)
        x = jax.nn.gelu(x)

        x = self.down(m_dist_down, x)

        # ============ Processor ============
        for i in range(self.n_blocks):
            pa_out = self.PA_blocks[i](m_dist_proc, x)

            mlp_out = self.MLP_blocks[i](pa_out)

            residual = jax.vmap(jax.vmap(self.W_blocks[i]))(x)
            x = mlp_out + residual
            x = jax.nn.gelu(x)

        # ============ Decoder ============
        x = self.up(m_dist_up, x)

        x = jax.vmap(jax.vmap(self.de_fc1))(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(jax.vmap(self.de_fc2))(x)

        if squeeze_batch:
            x = x[0]

        return x
