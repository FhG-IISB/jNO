"""
General Neural Operator Transformer (GNOT) - JAX/Equinox Implementation
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
"""

from typing import Callable, List, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx
from .linear import Linear
from .common import get_activation


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


def horizontal_fourier_embedding(x: jnp.ndarray, n: int = 3) -> jnp.ndarray:
    """
    Apply horizontal Fourier feature embedding.

    Expands input features using sinusoidal functions at multiple frequencies,
    enabling the network to learn high-frequency patterns.

    Transform: x -> [x, cos(2^{-n}x), sin(2^{-n}x), ..., cos(2^n x), sin(2^n x)]

    Args:
        x: Input tensor of shape [batch, seq_len, features].
        n: Number of frequency octaves. Total frequencies = 2n + 1.

    Returns:
        Embedded tensor of shape [batch, seq_len, features * (4n + 3)].
    """
    freqs = 2.0 ** jnp.linspace(-n, n, 2 * n + 1)
    freqs = freqs[None, None, None, :]

    x_expanded = x[..., None]

    x_cos = jnp.cos(freqs * x_expanded)
    x_sin = jnp.sin(freqs * x_expanded)

    x_embedded = jnp.concatenate([x_expanded, x_cos, x_sin], axis=-1)

    batch, seq_len, features, freq_features = x_embedded.shape
    return x_embedded.reshape(batch, seq_len, features * freq_features)


# =============================================================================
# MLP Module
# =============================================================================


class MLP(eqx.Module):
    """
    Multi-Layer Perceptron with configurable depth and activation.

    Attributes:
        layers: List of Linear layers.
        activation: Activation function.
    """

    layers: list
    activation: Callable = eqx.field(static=True)

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2, act: str = "gelu", *, key, **kwargs):
        activation = get_activation(act)
        self.activation = activation

        keys = jax.random.split(key, n_layers)
        self.layers = []

        # First layer
        self.layers.append(Linear(in_dim, hidden_dim, key=keys[0]))

        # Hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim, key=keys[i]))

        # Output layer
        self.layers.append(Linear(hidden_dim, out_dim, key=keys[-1]))

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


# =============================================================================
# Attention Modules
# =============================================================================


class LinearAttention(eqx.Module):
    """
    Linear (O(n)) self-attention using kernel approximation.

    Instead of computing the full N*N attention matrix, linear attention
    uses the associativity of matrix multiplication:

        Attention(Q, K, V) = softmax(QK^T)V ~ phi(Q)(phi(K)^T V)

    where phi is a feature map (here, softmax normalization).

    This reduces complexity from O(n^2) to O(n).
    """

    query: Linear
    key: Linear
    value: Linear
    proj: Linear
    n_embd: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    attn_pdrop: float = eqx.field(static=True)
    attn_type: str = eqx.field(static=True)

    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float = 0.0, attn_type: str = "l1", *, key, **kwargs):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.query = Linear(n_embd, n_embd, key=k1)
        self.key = Linear(n_embd, n_embd, key=k2)
        self.value = Linear(n_embd, n_embd, key=k3)
        self.proj = Linear(n_embd, n_embd, key=k4)
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.attn_type = attn_type

    def __call__(self, x: jnp.ndarray, y: Optional[jnp.ndarray] = None, *, key=None, **kwargs) -> jnp.ndarray:
        if y is None:
            y = x

        B, T1, C = x.shape
        _, T2, _ = y.shape
        head_dim = C // self.n_head

        # Project Q, K, V
        q = jax.vmap(jax.vmap(self.query))(x)
        k = jax.vmap(jax.vmap(self.key))(y)
        v = jax.vmap(jax.vmap(self.value))(y)

        # Reshape for multi-head: [B, T, C] -> [B, n_head, T, head_dim]
        q = q.reshape(B, T1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)

        # Apply kernel feature map based on attention type
        if self.attn_type == "l1":
            q = jax.nn.softmax(q, axis=-1)
            k = jax.nn.softmax(k, axis=-1)
            k_sum = k.sum(axis=-2, keepdims=True)
            D_inv = 1.0 / (q * k_sum).sum(axis=-1, keepdims=True)

        elif self.attn_type == "galerkin":
            q = jax.nn.softmax(q, axis=-1)
            k = jax.nn.softmax(k, axis=-1)
            D_inv = 1.0 / T2

        elif self.attn_type == "l2":
            q = q / (jnp.linalg.norm(q, ord=1, axis=-1, keepdims=True) + 1e-8)
            k = k / (jnp.linalg.norm(k, ord=1, axis=-1, keepdims=True) + 1e-8)
            k_sum = k.sum(axis=-2, keepdims=True)
            D_inv = 1.0 / (jnp.abs(q * k_sum).sum(axis=-1, keepdims=True) + 1e-8)

        else:
            raise ValueError(f"Unknown attention type: {self.attn_type}")

        # Linear attention: O(n) complexity
        context = jnp.einsum("bhnd,bhnv->bhdv", k, v)
        out = jnp.einsum("bhqd,bhdv->bhqv", q, context)

        # Apply normalization and residual
        out = out * D_inv + q

        # Optional dropout
        if self.attn_pdrop > 0 and key is not None:
            key, subkey = jax.random.split(key)
            out = eqx.nn.Dropout(p=self.attn_pdrop)(out, key=subkey)

        # Reshape back: [B, n_head, T1, head_dim] -> [B, T1, C]
        out = out.transpose(0, 2, 1, 3).reshape(B, T1, C)

        # Output projection
        out = jax.vmap(jax.vmap(self.proj))(out)
        return out


class LinearCrossAttention(eqx.Module):
    """
    Linear cross-attention for multiple input branches.

    Computes attention from query points to multiple key-value sources,
    aggregating information from all input functions.
    """

    query_proj: Linear
    key_projs: list
    value_projs: list
    proj: Linear
    n_embd: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    n_inputs: int = eqx.field(static=True)
    attn_pdrop: float = eqx.field(static=True)

    def __init__(self, n_embd: int, n_head: int, n_inputs: int, attn_pdrop: float = 0.0, *, key, **kwargs):
        keys = jax.random.split(key, 2 * n_inputs + 2)
        self.query_proj = Linear(n_embd, n_embd, key=keys[0])
        self.key_projs = [Linear(n_embd, n_embd, key=keys[1 + i]) for i in range(n_inputs)]
        self.value_projs = [Linear(n_embd, n_embd, key=keys[1 + n_inputs + i]) for i in range(n_inputs)]
        self.proj = Linear(n_embd, n_embd, key=keys[-1])
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_inputs = n_inputs
        self.attn_pdrop = attn_pdrop

    def __call__(self, x: jnp.ndarray, ys: List[jnp.ndarray], *, key=None, **kwargs) -> jnp.ndarray:
        B, T1, C = x.shape
        head_dim = C // self.n_head

        # Query projection
        q = jax.vmap(jax.vmap(self.query_proj))(x)
        q = q.reshape(B, T1, self.n_head, head_dim).transpose(0, 2, 1, 3)
        q = jax.nn.softmax(q, axis=-1)

        # Start with query as residual
        out = q

        # Process each input branch
        for i in range(self.n_inputs):
            y_i = ys[i]
            _, T2, _ = y_i.shape

            k = jax.vmap(jax.vmap(self.key_projs[i]))(y_i)
            v = jax.vmap(jax.vmap(self.value_projs[i]))(y_i)

            k = k.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)
            v = v.reshape(B, T2, self.n_head, head_dim).transpose(0, 2, 1, 3)

            k = jax.nn.softmax(k, axis=-1)

            k_sum = k.sum(axis=-2, keepdims=True)
            D_inv = 1.0 / ((q * k_sum).sum(axis=-1, keepdims=True) + 1e-8)

            context = jnp.einsum("bhnd,bhnv->bhdv", k, v)
            attn_out = jnp.einsum("bhqd,bhdv->bhqv", q, context)
            out = out + attn_out * D_inv

        # Optional dropout
        if self.attn_pdrop > 0 and key is not None:
            key, subkey = jax.random.split(key)
            out = eqx.nn.Dropout(p=self.attn_pdrop)(out, key=subkey)

        # Reshape and project
        out = out.transpose(0, 2, 1, 3).reshape(B, T1, C)
        out = jax.vmap(jax.vmap(self.proj))(out)
        return out


# =============================================================================
# Transformer Blocks
# =============================================================================


class FFN(eqx.Module):
    """Simple 2-layer feed-forward network."""

    fc1: Linear
    fc2: Linear
    activation: Callable = eqx.field(static=True)

    def __init__(self, in_dim, inner_dim, out_dim, act="gelu", *, key, **kwargs):
        k1, k2 = jax.random.split(key)
        self.fc1 = Linear(in_dim, inner_dim, key=k1)
        self.fc2 = Linear(inner_dim, out_dim, key=k2)
        self.activation = get_activation(act)

    def __call__(self, x, **kwargs):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class CrossAttentionBlock(eqx.Module):
    """
    Cross-attention block for CGPTNO.

    Structure:
        x = x + CrossAttn(LN(x), LN(ys))
        x = x + FFN(LN(x))
        x = x + SelfAttn(LN(x))
        x = x + FFN(LN(x))
    """

    ln1: eqx.nn.LayerNorm
    ln2_branches: list
    ln3: eqx.nn.LayerNorm
    ln4: eqx.nn.LayerNorm
    ln5: eqx.nn.LayerNorm
    cross_attn: LinearCrossAttention
    self_attn: LinearAttention
    ffn1: FFN
    ffn2: FFN
    resid_pdrop: float = eqx.field(static=True)

    def __init__(self, config: GPTConfig, *, key, **kwargs):
        keys = jax.random.split(key, 4)
        cfg = config
        n_embd = cfg.n_embd

        self.ln1 = eqx.nn.LayerNorm(n_embd)
        self.ln2_branches = [eqx.nn.LayerNorm(n_embd) for _ in range(cfg.n_inputs)]
        self.ln3 = eqx.nn.LayerNorm(n_embd)
        self.ln4 = eqx.nn.LayerNorm(n_embd)
        self.ln5 = eqx.nn.LayerNorm(n_embd)

        self.cross_attn = LinearCrossAttention(n_embd=n_embd, n_head=cfg.n_head, n_inputs=cfg.n_inputs, attn_pdrop=cfg.attn_pdrop, key=keys[0])
        self.self_attn = LinearAttention(n_embd=n_embd, n_head=cfg.n_head, attn_pdrop=cfg.attn_pdrop, key=keys[1])
        self.ffn1 = FFN(n_embd, cfg.n_inner, n_embd, act=cfg.act, key=keys[2])
        self.ffn2 = FFN(n_embd, cfg.n_inner, n_embd, act=cfg.act, key=keys[3])
        self.resid_pdrop = cfg.resid_pdrop

    def __call__(self, x: jnp.ndarray, ys: List[jnp.ndarray], *, key=None, **kwargs) -> jnp.ndarray:
        # Cross-attention
        x_norm = jax.vmap(jax.vmap(self.ln1))(x)
        ys_norm = [jax.vmap(jax.vmap(self.ln2_branches[i]))(y) for i, y in enumerate(ys)]

        ca_out = self.cross_attn(x_norm, ys_norm, key=key)
        if self.resid_pdrop > 0 and key is not None:
            key, subkey = jax.random.split(key)
            ca_out = eqx.nn.Dropout(p=self.resid_pdrop)(ca_out, key=subkey)
        x = x + ca_out

        # FFN 1
        x_norm = jax.vmap(jax.vmap(self.ln3))(x)
        x = x + jax.vmap(jax.vmap(self.ffn1))(x_norm)

        # Self-attention
        x_norm = jax.vmap(jax.vmap(self.ln4))(x)
        sa_out = self.self_attn(x_norm, key=key)
        if self.resid_pdrop > 0 and key is not None:
            key, subkey = jax.random.split(key)
            sa_out = eqx.nn.Dropout(p=self.resid_pdrop)(sa_out, key=subkey)
        x = x + sa_out

        # FFN 2
        x_norm = jax.vmap(jax.vmap(self.ln5))(x)
        x = x + jax.vmap(jax.vmap(self.ffn2))(x_norm)

        return x


class MoECrossAttentionBlock(eqx.Module):
    """
    Cross-attention block with Mixture-of-Experts FFN.

    The FFN layers are replaced with position-dependent MoE:
        - A gating network routes based on spatial position
        - Multiple expert FFNs process the input
        - Outputs are weighted by gate scores
    """

    ln1: eqx.nn.LayerNorm
    ln2_branches: list
    ln3: eqx.nn.LayerNorm
    ln4: eqx.nn.LayerNorm
    ln5: eqx.nn.LayerNorm
    cross_attn: LinearCrossAttention
    self_attn: LinearAttention
    gate_net: list  # list of Linear layers for gate MLP
    moe_ffn1_experts: list  # list of FFN
    moe_ffn2_experts: list  # list of FFN
    gate_activation: Callable = eqx.field(static=True)
    resid_pdrop: float = eqx.field(static=True)
    n_experts: int = eqx.field(static=True)

    def __init__(self, config: MoEGPTConfig, *, key, **kwargs):
        cfg = config
        n_embd = cfg.n_embd
        keys = jax.random.split(key, 4 + 2 * cfg.n_experts + 3)

        self.ln1 = eqx.nn.LayerNorm(n_embd)
        self.ln2_branches = [eqx.nn.LayerNorm(n_embd) for _ in range(cfg.n_inputs)]
        self.ln3 = eqx.nn.LayerNorm(n_embd)
        self.ln4 = eqx.nn.LayerNorm(n_embd)
        self.ln5 = eqx.nn.LayerNorm(n_embd)

        self.cross_attn = LinearCrossAttention(n_embd=n_embd, n_head=cfg.n_head, n_inputs=cfg.n_inputs, attn_pdrop=cfg.attn_pdrop, key=keys[0])
        self.self_attn = LinearAttention(n_embd=n_embd, n_head=cfg.n_head, attn_pdrop=cfg.attn_pdrop, key=keys[1])

        activation = get_activation(cfg.act)
        self.gate_activation = activation

        # Gate network: 3 Dense layers
        self.gate_net = [
            Linear(cfg.space_dim, cfg.n_inner, key=keys[2]),
            Linear(cfg.n_inner, cfg.n_inner, key=keys[3]),
            Linear(cfg.n_inner, cfg.n_experts, key=keys[4]),
        ]

        # MoE experts
        self.moe_ffn1_experts = [FFN(n_embd, cfg.n_inner, n_embd, act=cfg.act, key=keys[5 + i]) for i in range(cfg.n_experts)]
        self.moe_ffn2_experts = [FFN(n_embd, cfg.n_inner, n_embd, act=cfg.act, key=keys[5 + cfg.n_experts + i]) for i in range(cfg.n_experts)]

        self.resid_pdrop = cfg.resid_pdrop
        self.n_experts = cfg.n_experts

    def _apply_gate(self, pos):
        """pos: [B, T, space_dim] -> [B, T, 1, n_experts]"""
        x = pos
        for i, layer in enumerate(self.gate_net[:-1]):
            x = jax.vmap(jax.vmap(layer))(x)
            x = self.gate_activation(x)
        x = jax.vmap(jax.vmap(self.gate_net[-1]))(x)
        gate_scores = jax.nn.softmax(x, axis=-1)
        return gate_scores[..., None, :]  # [B, T, 1, n_experts]

    def __call__(self, x: jnp.ndarray, ys: List[jnp.ndarray], pos: jnp.ndarray, *, key=None, **kwargs) -> jnp.ndarray:
        gate_scores = self._apply_gate(pos)

        # Cross-attention
        x_norm = jax.vmap(jax.vmap(self.ln1))(x)
        ys_norm = [jax.vmap(jax.vmap(self.ln2_branches[i]))(y) for i, y in enumerate(ys)]

        ca_out = self.cross_attn(x_norm, ys_norm, key=key)
        if self.resid_pdrop > 0 and key is not None:
            key, subkey = jax.random.split(key)
            ca_out = eqx.nn.Dropout(p=self.resid_pdrop)(ca_out, key=subkey)
        x = x + ca_out

        # MoE FFN 1
        expert_outputs_1 = [jax.vmap(jax.vmap(expert))(x) for expert in self.moe_ffn1_experts]
        x_moe1 = jnp.stack(expert_outputs_1, axis=-1)
        x_moe1 = (gate_scores * x_moe1).sum(axis=-1)
        x = x + jax.vmap(jax.vmap(self.ln3))(x_moe1)

        # Self-attention
        x_norm = jax.vmap(jax.vmap(self.ln4))(x)
        sa_out = self.self_attn(x_norm, key=key)
        if self.resid_pdrop > 0 and key is not None:
            key, subkey = jax.random.split(key)
            sa_out = eqx.nn.Dropout(p=self.resid_pdrop)(sa_out, key=subkey)
        x = x + sa_out

        # MoE FFN 2
        expert_outputs_2 = [jax.vmap(jax.vmap(expert))(x) for expert in self.moe_ffn2_experts]
        x_moe2 = jnp.stack(expert_outputs_2, axis=-1)
        x_moe2 = (gate_scores * x_moe2).sum(axis=-1)
        x = x + jax.vmap(jax.vmap(self.ln5))(x_moe2)

        return x


# =============================================================================
# Main Model Classes
# =============================================================================


class CGPTNO(eqx.Module):
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
    """

    trunk_mlp: MLP
    branch_mlps: list
    blocks: list
    out_mlp: MLP
    horiz_fourier_dim: int = eqx.field(static=True)
    _trunk_size: int = eqx.field(static=True)
    _branch_sizes: list = eqx.field(static=True)
    _n_inputs: int = eqx.field(static=True)

    def __init__(
        self,
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
        *,
        key,
        **kwargs,
    ):
        self.horiz_fourier_dim = horiz_fourier_dim

        # Compute effective sizes with Fourier embedding
        if horiz_fourier_dim > 0:
            fourier_mult = 4 * horiz_fourier_dim + 3
            self._trunk_size = trunk_size * fourier_mult
            self._branch_sizes = [bs * fourier_mult for bs in (branch_sizes or [])]
        else:
            self._trunk_size = trunk_size
            self._branch_sizes = list(branch_sizes or [])

        self._n_inputs = len(self._branch_sizes)

        # Build config
        config = GPTConfig(
            attn_type=attn_type,
            embd_pdrop=ffn_dropout,
            resid_pdrop=ffn_dropout,
            attn_pdrop=attn_dropout,
            n_embd=n_hidden,
            n_head=n_head,
            n_layer=n_layers,
            n_inner=n_inner * n_hidden,
            act=act,
            branch_sizes=self._branch_sizes,
            n_inputs=max(self._n_inputs, 1),
        )

        # Split keys
        n_total = 2 + self._n_inputs + n_layers
        keys = jax.random.split(key, n_total)

        # Trunk MLP
        self.trunk_mlp = MLP(self._trunk_size, n_hidden, n_hidden, n_layers=mlp_layers, act=act, key=keys[0])

        # Branch MLPs
        self.branch_mlps = [MLP(self._branch_sizes[i], n_hidden, n_hidden, n_layers=mlp_layers, act=act, key=keys[1 + i]) for i in range(self._n_inputs)]

        # Transformer blocks
        self.blocks = [CrossAttentionBlock(config, key=keys[1 + self._n_inputs + i]) for i in range(n_layers)]

        # Output MLP
        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers, act=act, key=keys[-1])

    def __call__(
        self,
        x_trunk: jnp.ndarray,
        x_branches: Optional[List[jnp.ndarray]] = None,
        *,
        key=None,
        **kwargs,
    ) -> jnp.ndarray:
        """
        Forward pass.

        Args:
            x_trunk: Query points [batch, n_query, trunk_size].
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
        x = jax.vmap(jax.vmap(self.trunk_mlp))(x_trunk)

        # Branch embeddings
        if x_branches is not None and len(x_branches) > 0:
            z_list = [jax.vmap(jax.vmap(self.branch_mlps[i]))(xb) for i, xb in enumerate(x_branches)]
        else:
            z_list = [x]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, z_list, key=key)

        # Output projection
        x = jax.vmap(jax.vmap(self.out_mlp))(x)

        return x


class GNOT(eqx.Module):
    """
    General Neural Operator Transformer with Mixture-of-Experts.

    Extends CGPTNO with position-dependent MoE layers, allowing the model
    to learn spatially-varying transformations.

    Key Differences from CGPTNO:
        - FFN layers replaced with MoE
        - Requires position input for gating
        - Better for problems with spatially varying behavior
    """

    trunk_mlp: MLP
    branch_mlps: list
    blocks: list
    out_mlp: MLP
    space_dim: int = eqx.field(static=True)
    horiz_fourier_dim: int = eqx.field(static=True)
    _trunk_size: int = eqx.field(static=True)
    _branch_sizes: list = eqx.field(static=True)
    _n_inputs: int = eqx.field(static=True)

    def __init__(
        self,
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
        *,
        key,
        **kwargs,
    ):
        self.space_dim = space_dim
        self.horiz_fourier_dim = horiz_fourier_dim

        if horiz_fourier_dim > 0:
            fourier_mult = 4 * horiz_fourier_dim + 3
            self._trunk_size = trunk_size * fourier_mult
            self._branch_sizes = [bs * fourier_mult for bs in branch_sizes]
        else:
            self._trunk_size = trunk_size
            self._branch_sizes = list(branch_sizes)

        self._n_inputs = len(self._branch_sizes)

        config = MoEGPTConfig(
            attn_type=attn_type,
            embd_pdrop=ffn_dropout,
            resid_pdrop=ffn_dropout,
            attn_pdrop=attn_dropout,
            n_embd=n_hidden,
            n_head=n_head,
            n_layer=n_layers,
            n_inner=n_inner * n_hidden,
            act=act,
            n_experts=n_experts,
            space_dim=space_dim,
            branch_sizes=self._branch_sizes,
            n_inputs=self._n_inputs,
        )

        n_total = 2 + self._n_inputs + n_layers
        keys = jax.random.split(key, n_total)

        self.trunk_mlp = MLP(self._trunk_size, n_hidden, n_hidden, n_layers=mlp_layers, act=act, key=keys[0])

        self.branch_mlps = [MLP(self._branch_sizes[i], n_hidden, n_hidden, n_layers=mlp_layers, act=act, key=keys[1 + i]) for i in range(self._n_inputs)]

        self.blocks = [MoECrossAttentionBlock(config, key=keys[1 + self._n_inputs + i]) for i in range(n_layers)]

        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers, act=act, key=keys[-1])

    def __call__(
        self,
        x_trunk: jnp.ndarray,
        x_branches: List[jnp.ndarray],
        *,
        key=None,
        **kwargs,
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
        pos = x_trunk[..., : self.space_dim]

        if self.horiz_fourier_dim > 0:
            x_trunk = horizontal_fourier_embedding(x_trunk, self.horiz_fourier_dim)
            x_branches = [horizontal_fourier_embedding(xb, self.horiz_fourier_dim) for xb in x_branches]

        x = jax.vmap(jax.vmap(self.trunk_mlp))(x_trunk)

        z_list = [jax.vmap(jax.vmap(self.branch_mlps[i]))(xb) for i, xb in enumerate(x_branches)]

        for block in self.blocks:
            x = block(x, z_list, pos, key=key)

        x = jax.vmap(jax.vmap(self.out_mlp))(x)

        return x


class MoEGPTNO(eqx.Module):
    """
    Single-input Mixture-of-Experts GPT Neural Operator.

    Simplified variant of GNOT for problems with a single input function.
    """

    gnot_inner: GNOT

    def __init__(
        self,
        trunk_size: int,
        branch_size: int,
        space_dim: int = 2,
        output_size: int = 1,
        n_layers: int = 2,
        n_hidden: int = 64,
        n_head: int = 1,
        n_experts: int = 2,
        mlp_layers: int = 2,
        attn_type: str = "linear",
        act: str = "gelu",
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        horiz_fourier_dim: int = 0,
        *,
        key,
        **kwargs,
    ):
        self.gnot_inner = GNOT(
            trunk_size=trunk_size,
            branch_sizes=[branch_size],
            space_dim=space_dim,
            output_size=output_size,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            n_experts=n_experts,
            n_inner=4,
            mlp_layers=mlp_layers,
            attn_type=attn_type,
            act=act,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            horiz_fourier_dim=horiz_fourier_dim,
            key=key,
        )

    def __call__(
        self,
        x_trunk: jnp.ndarray,
        x_branch: jnp.ndarray,
        *,
        key=None,
        **kwargs,
    ) -> jnp.ndarray:
        return self.gnot_inner(x_trunk, [x_branch], key=key)


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
    *,
    key,
) -> CGPTNO:
    """Factory function for CGPTNO."""
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
        key=key,
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
    *,
    key,
) -> GNOT:
    """Factory function for GNOT."""
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
        key=key,
    )
