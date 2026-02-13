"""
General Neural Operator Transformer (GNOT) - JAX/Flax Implementation
=====================================================================

This module implements GNOT and its variants for learning operators on
arbitrary geometries using cross-attention mechanisms.

GNOT is designed for problems with:
- Irregular/unstructured meshes
- Multiple input functions (multi-physics)
- Variable-size point clouds
- Complex domain geometries

Architecture Overview
---------------------

GNOT uses a cross-attention mechanism between:
- **Trunk (Query)**: Output query points with coordinates
- **Branch (Key/Value)**: Input function(s) sampled at sensor locations

The attention mechanism is "linear" (O(n) complexity) using kernel
approximations, making it scalable to large point clouds.

Variants
--------
- **CGPTNO**: Basic cross-attention GPT neural operator
- **GNOT**: GNOT with Mixture-of-Experts (MoE) FFN layers
- **MoEGPTNO**: Single-input variant with MoE

Key Features
------------
1. **Linear Attention**: O(n) complexity via kernel approximation
2. **Multi-Input Support**: Handle multiple input functions
3. **Mixture of Experts**: Position-dependent FFN routing
4. **Fourier Embedding**: Optional horizontal Fourier features

References
----------
.. [1] Hao et al. "GNOT: A General Neural Operator Transformer for
       Operator Learning" ICML 2023. https://arxiv.org/abs/2302.14376

Example Usage
-------------
>>> import pino as pnp
>>>
>>> # Basic GNOT for 2D problem
>>> model = pnp.nn.gnot(
...     trunk_size=4,  # (x, y) + 2 params
...     branch_sizes=[3, 2],  # Two input functions
...     output_size=1,
...     n_layers=3,
...     n_hidden=128,
... )
>>>
>>> # Single-input variant
>>> model = pnp.nn.cgptno(
...     trunk_size=3,
...     branch_sizes=[3],
...     output_size=1,
... )
"""

from typing import Callable, List, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as ln


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class GPTConfig:
    """
    Configuration for basic GPT-style neural operator.

    Attributes:
        attn_type: Attention type ('linear', 'softmax').
        embd_pdrop: Embedding dropout probability.
        resid_pdrop: Residual connection dropout probability.
        attn_pdrop: Attention dropout probability.
        n_embd: Embedding/hidden dimension.
        n_head: Number of attention heads.
        n_layer: Number of transformer layers.
        block_size: Maximum sequence length (unused, for compatibility).
        n_inner: FFN inner dimension (default: 4 * n_embd).
        act: Activation function name.
        branch_sizes: List of input branch sizes.
        n_inputs: Number of input branches.
    """

    attn_type: str = "linear"
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    n_embd: int = 128
    n_head: int = 1
    n_layer: int = 3
    block_size: int = 128
    n_inner: int = 512
    act: str = "gelu"
    branch_sizes: Optional[List[int]] = None
    n_inputs: int = 1

    def __post_init__(self):
        if self.n_inner == 512:  # Default value
            self.n_inner = 4 * self.n_embd


@dataclass
class MoEGPTConfig(GPTConfig):
    """
    Configuration for Mixture-of-Experts GPT neural operator.

    Additional Attributes:
        n_experts: Number of expert networks in MoE layers.
        space_dim: Spatial dimension for position-based gating.
    """

    n_experts: int = 2
    space_dim: int = 1


# =============================================================================

# Utility Functions

# =============================================================================


def get_activation(name: str) -> Callable:
    """
    Get activation function by name.

    Args:
        name: Activation name ('gelu', 'relu', 'tanh', 'sigmoid', 'silu').

    Returns:
        Callable: JAX activation function.

    Raises:
        ValueError: If activation name is not recognized.
    """
    activations = {
        "gelu": ln.gelu,
        "relu": ln.relu,
        "tanh": ln.tanh,
        "sigmoid": ln.sigmoid,
        "silu": ln.silu,
        "swish": ln.swish,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation '{name}'. " f"Available: {list(activations.keys())}")
    return activations[name.lower()]


def horizontal_fourier_embedding(x: jnp.ndarray, n: int = 3) -> jnp.ndarray:
    """
    Apply horizontal Fourier feature embedding.

    Expands input features using sinusoidal functions at multiple frequencies,
    enabling the network to learn high-frequency patterns.

    Transform: x → [x, cos(2^{-n}x), sin(2^{-n}x), ..., cos(2^n x), sin(2^n x)]

    Args:
        x: Input tensor of shape [batch, seq_len, features].
        n: Number of frequency octaves. Total frequencies = 2n + 1.

    Returns:
        Embedded tensor of shape [batch, seq_len, features * (4n + 3)].

    Example:
        >>> x = jnp.ones((2, 100, 3))  # [batch, points, features]
        >>> x_emb = horizontal_fourier_embedding(x, n=3)
        >>> x_emb.shape  # (2, 100, 45) = 3 * (4*3 + 3)
    """
    # Frequencies: 2^{-n}, 2^{-n+1}, ..., 2^{n-1}, 2^n
    freqs = 2.0 ** jnp.linspace(-n, n, 2 * n + 1)  # [2n+1]
    freqs = freqs[None, None, None, :]  # [1, 1, 1, 2n+1]

    # Expand x for broadcasting
    x_expanded = x[..., None]  # [B, T, C, 1]

    # Compute cos and sin at all frequencies
    x_cos = jnp.cos(freqs * x_expanded)  # [B, T, C, 2n+1]
    x_sin = jnp.sin(freqs * x_expanded)  # [B, T, C, 2n+1]

    # Concatenate: [original, cos, sin]
    x_embedded = jnp.concatenate([x_expanded, x_cos, x_sin], axis=-1)  # [B, T, C, 4n+3]

    # Flatten last two dimensions
    batch, seq_len, features, freq_features = x_embedded.shape
    return x_embedded.reshape(batch, seq_len, features * freq_features)


# =============================================================================

# MLP Module

# =============================================================================


class MLP(ln.Module):
    """
    Multi-Layer Perceptron with configurable depth and activation.

    Attributes:
        in_dim: Input dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output dimension.
        n_layers: Number of layers (minimum 2).
        act: Activation function name.
    """

    in_dim: int
    hidden_dim: int
    out_dim: int
    n_layers: int = 2
    act: str = "gelu"

    @ln.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [..., in_dim].

        Returns:
            Output tensor of shape [..., out_dim].
        """
        activation = get_activation(self.act)

        # First layer
        x = ln.Dense(self.hidden_dim, name="fc_in")(x)
        x = activation(x)

        # Hidden layers
        for i in range(self.n_layers - 2):
            x = ln.Dense(self.hidden_dim, name=f"fc_{i}")(x)
            x = activation(x)

        # Output layer
        x = ln.Dense(self.out_dim, name="fc_out")(x)
        return x


# =============================================================================

# Attention Modules

# =============================================================================


class LinearAttention(ln.Module):
    """
    Linear (O(n)) self-attention using kernel approximation.

    Instead of computing the full N×N attention matrix, linear attention
    uses the associativity of matrix multiplication:

        Attention(Q, K, V) = softmax(QK^T)V ≈ φ(Q)(φ(K)^T V)

    where φ is a feature map (here, softmax normalization).

    This reduces complexity from O(n²) to O(n).

    Attributes:
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        attn_pdrop: Attention dropout probability.
        attn_type: Attention variant ('l1', 'l2', 'galerkin').
    """

    n_embd: int
    n_head: int
    attn_pdrop: float = 0.0
    attn_type: str = "l1"
    deterministic: bool = True

    @ln.compact
    def __call__(
        self,
        x: jnp.ndarray,
        y: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Compute linear attention.

        Args:
            x: Query tensor [batch, seq_q, n_embd].
            y: Key/Value tensor [batch, seq_kv, n_embd]. If None, uses x (self-attention).

        Returns:
            Output tensor [batch, seq_q, n_embd].
        """
        if y is None:
            y = x

        B, T1, C = x.shape
        _, T2, _ = y.shape
        head_dim = C // self.n_head

        # Project to Q, K, V
        q = ln.Dense(C, name="query")(x)
        k = ln.Dense(C, name="key")(y)
        v = ln.Dense(C, name="value")(y)

        # Reshape for multi-head: [B, T, C] -> [B, n_head, T, head_dim]
        q = q.reshape(B, T1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)

        # Apply kernel feature map based on attention type
        if self.attn_type == "l1":
            # L1-normalized softmax kernel
            q = jax.nn.softmax(q, axis=-1)
            k = jax.nn.softmax(k, axis=-1)
            k_sum = k.sum(axis=-2, keepdims=True)
            D_inv = 1.0 / (q * k_sum).sum(axis=-1, keepdims=True)

        elif self.attn_type == "galerkin":
            # Galerkin-style: uniform normalization
            q = jax.nn.softmax(q, axis=-1)
            k = jax.nn.softmax(k, axis=-1)
            D_inv = 1.0 / T2

        elif self.attn_type == "l2":
            # L2-normalized kernel
            q = q / (jnp.linalg.norm(q, ord=1, axis=-1, keepdims=True) + 1e-8)
            k = k / (jnp.linalg.norm(k, ord=1, axis=-1, keepdims=True) + 1e-8)
            k_sum = k.sum(axis=-2, keepdims=True)
            D_inv = 1.0 / (jnp.abs(q * k_sum).sum(axis=-1, keepdims=True) + 1e-8)

        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")

        # Linear attention: O(n) complexity
        # context = K^T @ V: [B, n_head, head_dim, head_dim]
        context = jnp.einsum("bhnd,bhnv->bhdv", k, v)
        # output = Q @ context: [B, n_head, T1, head_dim]
        out = jnp.einsum("bhqd,bhdv->bhqv", q, context)

        # Apply normalization and residual
        out = out * D_inv + q

        # Optional dropout
        if not self.deterministic and self.attn_pdrop > 0:
            out = ln.Dropout(rate=self.attn_pdrop)(out, deterministic=False)

        # Reshape back: [B, n_head, T1, head_dim] -> [B, T1, C]
        out = out.transpose(0, 2, 1, 3).reshape(B, T1, C)

        # Output projection
        out = ln.Dense(C, name="proj")(out)
        return out


class LinearCrossAttention(ln.Module):
    """
    Linear cross-attention for multiple input branches.

    Computes attention from query points to multiple key-value sources,
    aggregating information from all input functions.

    For each input branch i:
        out += Q @ (K_i^T @ V_i) / normalization

    Attributes:
        n_embd: Embedding dimension.
        n_head: Number of attention heads.
        n_inputs: Number of input branches.
        attn_pdrop: Attention dropout probability.
    """

    n_embd: int
    n_head: int
    n_inputs: int
    attn_pdrop: float = 0.0
    deterministic: bool = True

    @ln.compact
    def __call__(
        self,
        x: jnp.ndarray,
        ys: List[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Compute cross-attention to multiple inputs.

        Args:
            x: Query tensor [batch, seq_q, n_embd].
            ys: List of key/value tensors, each [batch, seq_kv_i, n_embd].

        Returns:
            Output tensor [batch, seq_q, n_embd].
        """
        B, T1, C = x.shape
        head_dim = C // self.n_head

        # Query projection (shared across all inputs)
        q = ln.Dense(C, name="query")(x)
        q = q.reshape(B, T1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        q = jax.nn.softmax(q, axis=-1)

        # Start with query as residual
        out = q

        # Process each input branch
        for i in range(self.n_inputs):
            y_i = ys[i]
            _, T2, _ = y_i.shape

            # Per-branch key and value projections
            k = ln.Dense(C, name=f"key_{i}")(y_i)
            v = ln.Dense(C, name=f"value_{i}")(y_i)

            k = k.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)

            # Normalize keys
            k = jax.nn.softmax(k, axis=-1)

            # Compute normalization
            k_sum = k.sum(axis=-2, keepdims=True)
            D_inv = 1.0 / ((q * k_sum).sum(axis=-1, keepdims=True) + 1e-8)

            # Linear attention contribution
            context = jnp.einsum("bhnd,bhnv->bhdv", k, v)
            attn_out = jnp.einsum("bhqd,bhdv->bhqv", q, context)
            out = out + attn_out * D_inv

        # Optional dropout
        if not self.deterministic and self.attn_pdrop > 0:
            out = ln.Dropout(rate=self.attn_pdrop)(out, deterministic=False)

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, T1, C)
        out = ln.Dense(C, name="proj")(out)
        return out


# =============================================================================

# Transformer Blocks

# =============================================================================


class CrossAttentionBlock(ln.Module):
    """
    Cross-attention block for CGPTNO.

    Structure:
        x = x + CrossAttn(LN(x), LN(ys))
        x = x + FFN(LN(x))
        x = x + SelfAttn(LN(x))
        x = x + FFN(LN(x))

    Attributes:
        config: GPTConfig with model hyperparameters.
        deterministic: Whether to disable dropout.
    """

    config: GPTConfig
    deterministic: bool = True

    @ln.compact
    def __call__(
        self,
        x: jnp.ndarray,
        ys: List[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x: Query tensor [batch, seq_q, n_embd].
            ys: List of input tensors.

        Returns:
            Output tensor [batch, seq_q, n_embd].
        """
        cfg = self.config
        activation = get_activation(cfg.act)

        # Cross-attention
        x_norm = ln.LayerNorm(name="ln1")(x)
        ys_norm = [ln.LayerNorm(name=f"ln2_branch_{i}")(y) for i, y in enumerate(ys)]

        cross_attn = LinearCrossAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_inputs=cfg.n_inputs,
            attn_pdrop=cfg.attn_pdrop,
            deterministic=self.deterministic,
            name="cross_attn",
        )
        x = x + ln.Dropout(rate=cfg.resid_pdrop, deterministic=self.deterministic)(cross_attn(x_norm, ys_norm))

        # FFN 1
        x_norm = ln.LayerNorm(name="ln3")(x)
        ffn1 = ln.Sequential(
            [
                ln.Dense(cfg.n_inner),
                activation,
                ln.Dense(cfg.n_embd),
            ],
            name="ffn1",
        )
        x = x + ffn1(x_norm)

        # Self-attention
        x_norm = ln.LayerNorm(name="ln4")(x)
        self_attn = LinearAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            attn_pdrop=cfg.attn_pdrop,
            deterministic=self.deterministic,
            name="self_attn",
        )
        x = x + ln.Dropout(rate=cfg.resid_pdrop, deterministic=self.deterministic)(self_attn(x_norm))

        # FFN 2
        x_norm = ln.LayerNorm(name="ln5")(x)
        ffn2 = ln.Sequential(
            [
                ln.Dense(cfg.n_inner),
                activation,
                ln.Dense(cfg.n_embd),
            ],
            name="ffn2",
        )
        x = x + ffn2(x_norm)

        return x


class MoECrossAttentionBlock(ln.Module):
    """
    Cross-attention block with Mixture-of-Experts FFN.

    The FFN layers are replaced with position-dependent MoE:
        - A gating network routes based on spatial position
        - Multiple expert FFNs process the input
        - Outputs are weighted by gate scores

    This allows the model to learn position-specific transformations,
    useful for problems with spatially varying behavior.

    Attributes:
        config: MoEGPTConfig with model hyperparameters.
        deterministic: Whether to disable dropout.
    """

    config: MoEGPTConfig
    deterministic: bool = True

    @ln.compact
    def __call__(
        self,
        x: jnp.ndarray,
        ys: List[jnp.ndarray],
        pos: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Forward pass with position-dependent MoE.

        Args:
            x: Query tensor [batch, seq_q, n_embd].
            ys: List of input tensors.
            pos: Position tensor [batch, seq_q, space_dim] for gating.

        Returns:
            Output tensor [batch, seq_q, n_embd].
        """
        cfg = self.config
        activation = get_activation(cfg.act)

        # Gating network: position -> expert weights
        gate_net = ln.Sequential(
            [
                ln.Dense(cfg.n_inner),
                activation,
                ln.Dense(cfg.n_inner),
                activation,
                ln.Dense(cfg.n_experts),
            ],
            name="gate_net",
        )
        gate_scores = jax.nn.softmax(gate_net(pos), axis=-1)  # [B, T, n_experts]
        gate_scores = gate_scores[..., None, :]  # [B, T, 1, n_experts]

        # Cross-attention
        x_norm = ln.LayerNorm(name="ln1")(x)
        ys_norm = [ln.LayerNorm(name=f"ln2_branch_{i}")(y) for i, y in enumerate(ys)]

        cross_attn = LinearCrossAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            n_inputs=cfg.n_inputs,
            attn_pdrop=cfg.attn_pdrop,
            deterministic=self.deterministic,
            name="cross_attn",
        )
        x = x + ln.Dropout(rate=cfg.resid_pdrop, deterministic=self.deterministic)(cross_attn(x_norm, ys_norm))

        # MoE FFN 1
        expert_outputs_1 = []
        for i in range(cfg.n_experts):
            expert = ln.Sequential(
                [
                    ln.Dense(cfg.n_inner),
                    activation,
                    ln.Dense(cfg.n_embd),
                ],
                name=f"moe_ffn1_expert_{i}",
            )
            expert_outputs_1.append(expert(x))

        # Stack and weight by gate: [B, T, C, n_experts]
        x_moe1 = jnp.stack(expert_outputs_1, axis=-1)
        x_moe1 = (gate_scores * x_moe1).sum(axis=-1)
        x = x + ln.LayerNorm(name="ln3")(x_moe1)

        # Self-attention
        x_norm = ln.LayerNorm(name="ln4")(x)
        self_attn = LinearAttention(
            n_embd=cfg.n_embd,
            n_head=cfg.n_head,
            attn_pdrop=cfg.attn_pdrop,
            deterministic=self.deterministic,
            name="self_attn",
        )
        x = x + ln.Dropout(rate=cfg.resid_pdrop, deterministic=self.deterministic)(self_attn(x_norm))

        # MoE FFN 2
        expert_outputs_2 = []
        for i in range(cfg.n_experts):
            expert = ln.Sequential(
                [
                    ln.Dense(cfg.n_inner),
                    activation,
                    ln.Dense(cfg.n_embd),
                ],
                name=f"moe_ffn2_expert_{i}",
            )
            expert_outputs_2.append(expert(x))

        x_moe2 = jnp.stack(expert_outputs_2, axis=-1)
        x_moe2 = (gate_scores * x_moe2).sum(axis=-1)
        x = x + ln.LayerNorm(name="ln5")(x_moe2)

        return x


# =============================================================================

# Main Model Classes

# =============================================================================


class CGPTNO(ln.Module):
    """
    Cross-attention GPT Neural Operator.

    A transformer-based neural operator that uses cross-attention to map
    from input functions (sampled at sensor points) to output functions
    (evaluated at query points).

    Architecture:
        1. Trunk MLP: Embed query points (coordinates + parameters)
        2. Branch MLPs: Embed each input function
        3. Cross-Attention Blocks: Query attends to all inputs
        4. Output MLP: Project to output dimension

    Attributes:
        trunk_size: Input dimension for trunk (query points).
        branch_sizes: List of input dimensions for each branch.
        output_size: Output dimension.
        n_layers: Number of transformer layers.
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        n_inner: FFN inner dimension multiplier.
        mlp_layers: Number of layers in embedding MLPs.
        attn_type: Attention type ('linear').
        act: Activation function name.
        ffn_dropout: FFN dropout rate.
        attn_dropout: Attention dropout rate.
        horiz_fourier_dim: Fourier embedding dimension (0 = disabled).
        deterministic: Whether to disable dropout.

    Example:
        >>> model = CGPTNO(
        ...     trunk_size=4,  # (x, y, param1, param2)
        ...     branch_sizes=[3],  # One input function with 3 features
        ...     output_size=1,
        ...     n_layers=3,
        ...     n_hidden=128,
        ... )
        >>> # x_trunk: [batch, n_query, 4]
        >>> # x_branch: [[batch, n_sensors, 3]]
        >>> y = model.apply(params, x_trunk, x_branch)  # [batch, n_query, 1]
    """

    trunk_size: int
    branch_sizes: Optional[List[int]] = None
    output_size: int = 1
    n_layers: int = 2
    n_hidden: int = 64
    n_head: int = 1
    n_inner: int = 4
    mlp_layers: int = 2
    attn_type: str = "linear"
    act: str = "gelu"
    ffn_dropout: float = 0.0
    attn_dropout: float = 0.0
    horiz_fourier_dim: int = 0
    deterministic: bool = True

    def setup(self):
        """Initialize model components."""
        # Compute effective sizes with Fourier embedding
        if self.horiz_fourier_dim > 0:
            fourier_mult = 4 * self.horiz_fourier_dim + 3
            self._trunk_size = self.trunk_size * fourier_mult
            self._branch_sizes = [bs * fourier_mult for bs in (self.branch_sizes or [])]
        else:
            self._trunk_size = self.trunk_size
            self._branch_sizes = self.branch_sizes or []

        self._n_inputs = len(self._branch_sizes)

        # Build config
        self._config = GPTConfig(
            attn_type=self.attn_type,
            embd_pdrop=self.ffn_dropout,
            resid_pdrop=self.ffn_dropout,
            attn_pdrop=self.attn_dropout,
            n_embd=self.n_hidden,
            n_head=self.n_head,
            n_layer=self.n_layers,
            n_inner=self.n_inner * self.n_hidden,
            act=self.act,
            branch_sizes=self._branch_sizes,
            n_inputs=max(self._n_inputs, 1),  # At least 1 for self-attention
        )

    @ln.compact
    def __call__(
        self,
        x_trunk: jnp.ndarray,
        x_branches: Optional[List[jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x_trunk: Query points [batch, n_query, trunk_size].
                Contains coordinates and optional parameters.
            x_branches: List of input function samples.
                Each element: [batch, n_sensors_i, branch_size_i].
                If None, uses self-attention only.

        Returns:
            Output values at query points [batch, n_query, output_size].
        """
        # Apply Fourier embedding if enabled
        if self.horiz_fourier_dim > 0:
            x_trunk = horizontal_fourier_embedding(x_trunk, self.horiz_fourier_dim)
            if x_branches is not None:
                x_branches = [horizontal_fourier_embedding(xb, self.horiz_fourier_dim) for xb in x_branches]

        # Trunk embedding
        trunk_mlp = MLP(
            in_dim=self._trunk_size,
            hidden_dim=self.n_hidden,
            out_dim=self.n_hidden,
            n_layers=self.mlp_layers,
            act=self.act,
            name="trunk_mlp",
        )
        x = trunk_mlp(x_trunk)

        # Branch embeddings
        if x_branches is not None and len(x_branches) > 0:
            z_list = []
            for i, xb in enumerate(x_branches):
                branch_mlp = MLP(
                    in_dim=self._branch_sizes[i],
                    hidden_dim=self.n_hidden,
                    out_dim=self.n_hidden,
                    n_layers=self.mlp_layers,
                    act=self.act,
                    name=f"branch_mlp_{i}",
                )
                z_list.append(branch_mlp(xb))
        else:
            # Self-attention only
            z_list = [x]

        # Transformer blocks
        for i in range(self.n_layers):
            block = CrossAttentionBlock(
                config=self._config,
                deterministic=self.deterministic,
                name=f"block_{i}",
            )
            x = block(x, z_list)

        # Output projection
        out_mlp = MLP(
            in_dim=self.n_hidden,
            hidden_dim=self.n_hidden,
            out_dim=self.output_size,
            n_layers=self.mlp_layers,
            act=self.act,
            name="out_mlp",
        )
        x = out_mlp(x)

        return x


class GNOT(ln.Module):
    """
    General Neural Operator Transformer with Mixture-of-Experts.

    Extends CGPTNO with position-dependent MoE layers, allowing the model
    to learn spatially-varying transformations. The gating network routes
    inputs to different expert networks based on spatial position.

    Key Differences from CGPTNO:
        - FFN layers replaced with MoE
        - Requires position input for gating
        - Better for problems with spatially varying behavior

    Attributes:
        trunk_size: Input dimension for trunk.
        branch_sizes: List of input dimensions for branches.
        space_dim: Spatial dimension for position-based gating.
        output_size: Output dimension.
        n_layers: Number of transformer layers.
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        n_experts: Number of expert networks.
        n_inner: FFN inner dimension multiplier.
        mlp_layers: Layers in embedding MLPs.
        attn_type: Attention type.
        act: Activation function.
        ffn_dropout: FFN dropout rate.
        attn_dropout: Attention dropout rate.
        horiz_fourier_dim: Fourier embedding dimension.
        deterministic: Whether to disable dropout.

    Example:
        >>> model = GNOT(
        ...     trunk_size=4,
        ...     branch_sizes=[3, 2],
        ...     space_dim=2,
        ...     output_size=1,
        ...     n_experts=4,
        ... )
        >>> # x_trunk: [batch, n_query, 4] where first 2 dims are (x, y)
        >>> # x_branches: list of input tensors
        >>> y = model.apply(params, x_trunk, x_branches)
    """

    trunk_size: int
    branch_sizes: List[int]
    space_dim: int = 2
    output_size: int = 1
    n_layers: int = 2
    n_hidden: int = 64
    n_head: int = 1
    n_experts: int = 2
    n_inner: int = 4
    mlp_layers: int = 2
    attn_type: str = "linear"
    act: str = "gelu"
    ffn_dropout: float = 0.0
    attn_dropout: float = 0.0
    horiz_fourier_dim: int = 0
    deterministic: bool = True

    def setup(self):
        """Initialize model components."""
        # Compute effective sizes
        if self.horiz_fourier_dim > 0:
            fourier_mult = 4 * self.horiz_fourier_dim + 3
            self._trunk_size = self.trunk_size * fourier_mult
            self._branch_sizes = [bs * fourier_mult for bs in self.branch_sizes]
        else:
            self._trunk_size = self.trunk_size
            self._branch_sizes = self.branch_sizes

        self._n_inputs = len(self._branch_sizes)

        # Build config
        self._config = MoEGPTConfig(
            attn_type=self.attn_type,
            embd_pdrop=self.ffn_dropout,
            resid_pdrop=self.ffn_dropout,
            attn_pdrop=self.attn_dropout,
            n_embd=self.n_hidden,
            n_head=self.n_head,
            n_layer=self.n_layers,
            n_inner=self.n_inner * self.n_hidden,
            act=self.act,
            n_experts=self.n_experts,
            space_dim=self.space_dim,
            branch_sizes=self._branch_sizes,
            n_inputs=self._n_inputs,
        )

    @ln.compact
    def __call__(
        self,
        x_trunk: jnp.ndarray,
        x_branches: List[jnp.ndarray],
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x_trunk: Query points [batch, n_query, trunk_size].
                First `space_dim` columns are spatial coordinates for gating.
            x_branches: List of input function samples.

        Returns:
            Output values [batch, n_query, output_size].
        """
        # Extract spatial positions for gating (before any embedding)
        pos = x_trunk[..., : self.space_dim]

        # Apply Fourier embedding if enabled
        if self.horiz_fourier_dim > 0:
            x_trunk = horizontal_fourier_embedding(x_trunk, self.horiz_fourier_dim)
            x_branches = [horizontal_fourier_embedding(xb, self.horiz_fourier_dim) for xb in x_branches]

        # Trunk embedding
        trunk_mlp = MLP(
            in_dim=self._trunk_size,
            hidden_dim=self.n_hidden,
            out_dim=self.n_hidden,
            n_layers=self.mlp_layers,
            act=self.act,
            name="trunk_mlp",
        )
        x = trunk_mlp(x_trunk)

        # Branch embeddings
        z_list = []
        for i, xb in enumerate(x_branches):
            branch_mlp = MLP(
                in_dim=self._branch_sizes[i],
                hidden_dim=self.n_hidden,
                out_dim=self.n_hidden,
                n_layers=self.mlp_layers,
                act=self.act,
                name=f"branch_mlp_{i}",
            )
            z_list.append(branch_mlp(xb))

        # MoE Transformer blocks
        for i in range(self.n_layers):
            block = MoECrossAttentionBlock(
                config=self._config,
                deterministic=self.deterministic,
                name=f"block_{i}",
            )
            x = block(x, z_list, pos)

        # Output projection
        out_mlp = MLP(
            in_dim=self.n_hidden,
            hidden_dim=self.n_hidden,
            out_dim=self.output_size,
            n_layers=self.mlp_layers,
            act=self.act,
            name="out_mlp",
        )
        x = out_mlp(x)

        return x


class MoEGPTNO(ln.Module):
    """
    Single-input Mixture-of-Experts GPT Neural Operator.

    Simplified variant of GNOT for problems with a single input function.
    Uses standard cross-attention (not multi-input) with MoE FFN layers.

    Attributes:
        trunk_size: Input dimension for trunk.
        branch_size: Input dimension for single branch.
        space_dim: Spatial dimension for gating.
        output_size: Output dimension.
        n_layers: Number of transformer layers.
        n_hidden: Hidden dimension.
        n_head: Number of attention heads.
        n_experts: Number of expert networks.
        mlp_layers: Layers in embedding MLPs.
        attn_type: Attention type.
        act: Activation function.
        ffn_dropout: FFN dropout rate.
        attn_dropout: Attention dropout rate.
        horiz_fourier_dim: Fourier embedding dimension.
        deterministic: Whether to disable dropout.
    """

    trunk_size: int
    branch_size: int
    space_dim: int = 2
    output_size: int = 1
    n_layers: int = 2
    n_hidden: int = 64
    n_head: int = 1
    n_experts: int = 2
    mlp_layers: int = 2
    attn_type: str = "linear"
    act: str = "gelu"
    ffn_dropout: float = 0.0
    attn_dropout: float = 0.0
    horiz_fourier_dim: int = 0
    deterministic: bool = True

    @ln.compact
    def __call__(
        self,
        x_trunk: jnp.ndarray,
        x_branch: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x_trunk: Query points [batch, n_query, trunk_size].
            x_branch: Input function [batch, n_sensors, branch_size].

        Returns:
            Output values [batch, n_query, output_size].
        """
        # Delegate to GNOT with single branch
        gnot = GNOT(
            trunk_size=self.trunk_size,
            branch_sizes=[self.branch_size],
            space_dim=self.space_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            n_head=self.n_head,
            n_experts=self.n_experts,
            n_inner=4,
            mlp_layers=self.mlp_layers,
            attn_type=self.attn_type,
            act=self.act,
            ffn_dropout=self.ffn_dropout,
            attn_dropout=self.attn_dropout,
            horiz_fourier_dim=self.horiz_fourier_dim,
            deterministic=self.deterministic,
            name="gnot_inner",
        )
        return gnot(x_trunk, [x_branch])


# =============================================================================

# Factory Functions (for integration with models.py)

# =============================================================================


def create_cgptno(
    trunk_size: int,
    branch_sizes: Optional[List[int]] = None,
    output_size: int = 1,
    n_layers: int = 2,
    n_hidden: int = 64,
    n_head: int = 1,
    n_inner: int = 4,
    mlp_layers: int = 2,
    attn_type: str = "linear",
    act: str = "gelu",
    ffn_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    horiz_fourier_dim: int = 0,
    deterministic: bool = True,
) -> CGPTNO:
    """
    Factory function for CGPTNO.

    See CGPTNO class for parameter descriptions.
    """
    return CGPTNO(
        trunk_size=trunk_size,
        branch_sizes=branch_sizes,
        output_size=output_size,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_head=n_head,
        n_inner=n_inner,
        mlp_layers=mlp_layers,
        attn_type=attn_type,
        act=act,
        ffn_dropout=ffn_dropout,
        attn_dropout=attn_dropout,
        horiz_fourier_dim=horiz_fourier_dim,
        deterministic=deterministic,
    )


def create_gnot(
    trunk_size: int,
    branch_sizes: List[int],
    space_dim: int = 2,
    output_size: int = 1,
    n_layers: int = 2,
    n_hidden: int = 64,
    n_head: int = 1,
    n_experts: int = 2,
    n_inner: int = 4,
    mlp_layers: int = 2,
    attn_type: str = "linear",
    act: str = "gelu",
    ffn_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    horiz_fourier_dim: int = 0,
    deterministic: bool = True,
) -> GNOT:
    """
    Factory function for GNOT.

    See GNOT class for parameter descriptions.
    """
    return GNOT(
        trunk_size=trunk_size,
        branch_sizes=branch_sizes,
        space_dim=space_dim,
        output_size=output_size,
        n_layers=n_layers,
        n_hidden=n_hidden,
        n_head=n_head,
        n_experts=n_experts,
        n_inner=n_inner,
        mlp_layers=mlp_layers,
        attn_type=attn_type,
        act=act,
        ffn_dropout=ffn_dropout,
        attn_dropout=attn_dropout,
        horiz_fourier_dim=horiz_fourier_dim,
        deterministic=deterministic,
    )
