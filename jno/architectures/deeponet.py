# Generalized Deep Operator Network (DeepONet) - Equinox Implementation

import equinox as eqx
from .linear import Linear
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Literal
from einops import repeat
from .common import BatchNorm


# =============================================================================
# Building Blocks
# =============================================================================


class MLPBlock(eqx.Module):
    """Flexible MLP block with various normalization and regularization options."""

    dense: Linear
    norm_layer: object
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, in_features, features, activation=jax.nn.gelu, norm=None, dropout_rate=0.0, use_bias=True, *, key, **kwargs):
        self.dense = Linear(in_features, features, use_bias=use_bias, key=key)
        self.activation = activation
        self.norm = norm
        self.dropout_rate = dropout_rate

        if norm == "layer":
            self.norm_layer = eqx.nn.LayerNorm(features)
        elif norm == "batch":
            self.norm_layer = BatchNorm(features)
        elif norm == "group":
            self.norm_layer = eqx.nn.GroupNorm(min(32, features), features)
        elif norm == "rms":
            self.norm_layer = eqx.nn.LayerNorm(features, elementwise_affine=False)
        else:
            self.norm_layer = None

    def __call__(self, x, *, key=None, **kwargs):
        x = self.dense(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        x = self.activation(x)
        if self.dropout_rate > 0.0 and key is not None:
            x = eqx.nn.Dropout(p=self.dropout_rate)(x, key=key)
        return x


class DeepONetMLP(eqx.Module):
    """Multi-layer perceptron with flexible architecture (internal to DeepONet)."""

    blocks: list
    output_layer: Linear
    output_activation: Optional[Callable] = eqx.field(static=True)

    def __init__(self, in_features, hidden_dims, output_dim, activation=jax.nn.gelu, output_activation=None, norm=None, dropout_rate=0.0, use_bias=True, *, key, **kwargs):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = [in_features] + list(hidden_dims)
        self.blocks = [MLPBlock(dims[i], dims[i + 1], activation=activation, norm=norm, dropout_rate=dropout_rate, use_bias=use_bias, key=keys[i]) for i in range(len(hidden_dims))]
        self.output_layer = Linear(hidden_dims[-1] if hidden_dims else in_features, output_dim, use_bias=use_bias, key=keys[-1])
        self.output_activation = output_activation

    def __call__(self, x, *, key=None, **kwargs):
        for i, block in enumerate(self.blocks):
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x = block(x, key=subkey)
        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class ResidualBlock(eqx.Module):
    """Residual block for deeper networks."""

    mlp_block: MLPBlock
    dense: Linear
    proj: Optional[Linear]
    out_norm: object
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)

    def __init__(self, in_features, features, activation=jax.nn.gelu, norm="layer", dropout_rate=0.0, *, key, **kwargs):
        k1, k2, k3 = jax.random.split(key, 3)
        self.mlp_block = MLPBlock(in_features, features, activation=activation, norm=norm, dropout_rate=dropout_rate, key=k1)
        self.dense = Linear(features, features, key=k2)
        self.proj = Linear(in_features, features, use_bias=False, key=k3) if in_features != features else None
        self.activation = activation
        self.norm = norm

        if norm == "layer":
            self.out_norm = eqx.nn.LayerNorm(features)
        elif norm == "batch":
            self.out_norm = BatchNorm(features)
        else:
            self.out_norm = None

    def __call__(self, x, *, key=None, **kwargs):
        residual = x if self.proj is None else self.proj(x)
        x = self.mlp_block(x, key=key)
        x = self.dense(x)
        if self.out_norm is not None:
            x = self.out_norm(x)
        return self.activation(x + residual)


class ResMLP(eqx.Module):
    """MLP with residual connections."""

    initial_proj: Linear
    blocks: list
    output_proj: Linear
    activation: Callable = eqx.field(static=True)
    output_activation: Optional[Callable] = eqx.field(static=True)

    def __init__(self, in_features, hidden_dim, output_dim, n_blocks=4, activation=jax.nn.gelu, output_activation=None, norm="layer", dropout_rate=0.0, *, key, **kwargs):
        keys = jax.random.split(key, n_blocks + 2)
        self.initial_proj = Linear(in_features, hidden_dim, key=keys[0])
        self.blocks = [ResidualBlock(hidden_dim, hidden_dim, activation=activation, norm=norm, dropout_rate=dropout_rate, key=keys[i + 1]) for i in range(n_blocks)]
        self.output_proj = Linear(hidden_dim, output_dim, key=keys[-1])
        self.activation = activation
        self.output_activation = output_activation

    def __call__(self, x, *, key=None, **kwargs):
        x = self.activation(self.initial_proj(x))
        for block in self.blocks:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x = block(x, key=subkey)
        x = self.output_proj(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class FourierFeatures(eqx.Module):
    """Fourier feature embedding for improved coordinate representation."""

    B: jnp.ndarray
    learnable: bool = eqx.field(static=True)

    def __init__(self, input_dim, n_features=64, scale=1.0, learnable=False, *, key, **kwargs):
        self.learnable = learnable
        if learnable:
            self.B = jax.random.normal(key, (input_dim, n_features)) * scale
        else:
            self.B = jax.random.normal(jax.random.PRNGKey(0), (input_dim, n_features)) * scale

    def __call__(self, x, **kwargs):
        B = self.B if self.learnable else jax.lax.stop_gradient(self.B)
        x_proj = x @ B
        return jnp.concatenate([jnp.sin(2 * jnp.pi * x_proj), jnp.cos(2 * jnp.pi * x_proj)], axis=-1)


class PositionalEncoding(eqx.Module):
    """Sinusoidal positional encoding."""

    d_model: int = eqx.field(static=True)
    max_len: int = eqx.field(static=True)
    base: float = eqx.field(static=True)

    def __init__(self, d_model, max_len=10000, base=10000.0, **kwargs):
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

    def __call__(self, x, **kwargs):
        """x: coordinates of shape (..., input_dim)"""
        input_dim = x.shape[-1]
        d_per_dim = self.d_model // input_dim

        encodings = []
        for i in range(input_dim):
            pos = x[..., i : i + 1]
            div_term = jnp.exp(jnp.arange(0, d_per_dim, 2) * (-jnp.log(self.base) / d_per_dim))
            pe = jnp.zeros((*x.shape[:-1], d_per_dim))
            pe = pe.at[..., 0::2].set(jnp.sin(pos * div_term))
            pe = pe.at[..., 1::2].set(jnp.cos(pos * div_term))
            encodings.append(pe)

        return jnp.concatenate(encodings, axis=-1)


class AttentionBlock(eqx.Module):
    """Multi-head self-attention block."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, in_features, n_heads=8, head_dim=64, dropout_rate=0.0, *, key, **kwargs):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        qkv_features = n_heads * head_dim
        self.q_proj = Linear(in_features, qkv_features, key=k1)
        self.k_proj = Linear(in_features, qkv_features, key=k2)
        self.v_proj = Linear(in_features, qkv_features, key=k3)
        self.out_proj = Linear(qkv_features, in_features, key=k4)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.dropout_rate = dropout_rate

    def __call__(self, x, *, key=None, **kwargs):
        input_shape = x.shape
        if x.ndim == 2:
            x = x[jnp.newaxis, :, :]

        B, T, _ = x.shape
        vv = jax.vmap(jax.vmap(self.q_proj))
        q = vv(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = jax.vmap(jax.vmap(self.k_proj))(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = jax.vmap(jax.vmap(self.v_proj))(x).reshape(B, T, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.array(self.head_dim, dtype=x.dtype))
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) / scale
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        out = jax.vmap(jax.vmap(self.out_proj))(out)

        if len(input_shape) == 2:
            out = out[0]
        return out


class TransformerBlock(eqx.Module):
    """Transformer encoder block."""

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    attn: AttentionBlock
    ffn1: Linear
    ffn2: Linear
    activation: Callable = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(self, d_model, n_heads=8, mlp_ratio=4.0, dropout_rate=0.0, activation=jax.nn.gelu, norm="layer", *, key, **kwargs):
        k1, k2, k3 = jax.random.split(key, 3)
        self.d_model = d_model
        self.norm1 = eqx.nn.LayerNorm(d_model)
        self.norm2 = eqx.nn.LayerNorm(d_model)
        self.attn = AttentionBlock(d_model, n_heads=n_heads, head_dim=d_model // n_heads, dropout_rate=dropout_rate, key=k1)
        mlp_dim = int(d_model * mlp_ratio)
        self.ffn1 = Linear(d_model, mlp_dim, key=k2)
        self.ffn2 = Linear(mlp_dim, d_model, key=k3)
        self.activation = activation
        self.dropout_rate = dropout_rate

    def __call__(self, x, *, key=None, **kwargs):
        def apply_norm(norm, t):
            if t.ndim == 2:
                return jax.vmap(norm)(t)
            elif t.ndim == 3:
                return jax.vmap(jax.vmap(norm))(t)
            return norm(t)

        residual = x
        x_n = apply_norm(self.norm1, x)
        x = residual + self.attn(x_n, key=key)

        if self.dropout_rate > 0 and key is not None:
            key, subkey = jax.random.split(key)
            x = eqx.nn.Dropout(p=self.dropout_rate)(x, key=subkey)

        residual = x
        x_n = apply_norm(self.norm2, x)

        if x_n.ndim == 2:
            x_n = jax.vmap(self.ffn1)(x_n)
            x_n = self.activation(x_n)
            x_n = jax.vmap(self.ffn2)(x_n)
        elif x_n.ndim == 3:
            x_n = jax.vmap(jax.vmap(self.ffn1))(x_n)
            x_n = self.activation(x_n)
            x_n = jax.vmap(jax.vmap(self.ffn2))(x_n)
        else:
            x_n = self.ffn1(x_n)
            x_n = self.activation(x_n)
            x_n = self.ffn2(x_n)

        x = residual + x_n

        if self.dropout_rate > 0 and key is not None:
            key, subkey = jax.random.split(key)
            x = eqx.nn.Dropout(p=self.dropout_rate)(x, key=subkey)

        return x


# =============================================================================
# Branch Networks
# =============================================================================


class BranchMLP(eqx.Module):
    """MLP-based branch network."""

    mlp: DeepONetMLP

    def __init__(self, in_features, hidden_dims, output_dim, activation=jax.nn.gelu, norm=None, dropout_rate=0.0, *, key, **kwargs):
        self.mlp = DeepONetMLP(in_features, hidden_dims, output_dim, activation=activation, norm=norm, dropout_rate=dropout_rate, key=key)

    def __call__(self, u, *, key=None, **kwargs):
        x = u.reshape(-1)
        return self.mlp(x, key=key)


class BranchResMLP(eqx.Module):
    """Residual MLP-based branch network."""

    resmlp: ResMLP

    def __init__(self, in_features, hidden_dim, output_dim, n_blocks=4, activation=jax.nn.gelu, norm="layer", dropout_rate=0.0, *, key, **kwargs):
        self.resmlp = ResMLP(in_features, hidden_dim, output_dim, n_blocks=n_blocks, activation=activation, norm=norm, dropout_rate=dropout_rate, key=key)

    def __call__(self, u, *, key=None, **kwargs):
        x = u.reshape(-1)
        return self.resmlp(x, key=key)


class Conv1dCL(eqx.Module):
    """1D convolution in channels-last (L, C) format."""

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    kernel_size: int = eqx.field(static=True)
    padding: str = eqx.field(static=True)

    def __init__(self, in_channels, out_channels, kernel_size=3, padding="SAME", use_bias=True, *, key, **kwargs):
        k1, k2 = jax.random.split(key)
        fan_in = in_channels * kernel_size
        std = 1.0 / jnp.sqrt(fan_in)
        self.weight = jax.random.uniform(k1, (kernel_size, in_channels, out_channels), minval=-std, maxval=std)
        self.bias = jax.random.uniform(k2, (out_channels,), minval=-std, maxval=std) if use_bias else None
        self.kernel_size = kernel_size
        self.padding = padding

    def __call__(self, x, **kwargs):
        # x: (L, C) -> add batch dim
        needs_squeeze = x.ndim == 2
        if needs_squeeze:
            x = x[jnp.newaxis, :, :]
        out = jax.lax.conv_general_dilated(x, self.weight, window_strides=(1,), padding=self.padding, dimension_numbers=("NWC", "WIO", "NWC"))
        if self.bias is not None:
            out = out + self.bias
        if needs_squeeze:
            out = out[0]
        return out


def avg_pool_1d(x, window_shape, strides):
    """Average pooling for (L, C) format."""
    w = window_shape[0]
    s = strides[0]
    L, C = x.shape
    # Truncate to fit
    out_len = (L - w) // s + 1
    indices = jnp.arange(out_len) * s
    windows = jax.vmap(lambda i: jax.lax.dynamic_slice(x, (i, 0), (w, C)))(indices)
    return jnp.mean(windows, axis=1)


class BranchConv1D(eqx.Module):
    """1D CNN-based branch network for sequential sensor data."""

    conv_layers: list
    norm_layers: list
    output_dense: Linear
    attn_dense: Optional[Linear]
    activation: Callable = eqx.field(static=True)
    pool_type: str = eqx.field(static=True)

    def __init__(self, in_channels, channels, output_dim, kernel_size=3, activation=jax.nn.gelu, norm="batch", pool_type="avg", *, key, **kwargs):
        keys = jax.random.split(key, len(channels) + 2)
        dims = [in_channels] + list(channels)
        self.conv_layers = [Conv1dCL(dims[i], dims[i + 1], kernel_size=kernel_size, key=keys[i]) for i in range(len(channels))]
        self.norm_layers = []
        for ch in channels:
            if norm == "batch":
                self.norm_layers.append(BatchNorm(ch))
            elif norm == "layer":
                self.norm_layers.append(eqx.nn.LayerNorm(ch))
            else:
                self.norm_layers.append(None)

        last_ch = channels[-1] if channels else in_channels
        self.attn_dense = Linear(last_ch, 1, key=keys[-2]) if pool_type == "attention" else None
        self.output_dense = Linear(last_ch, output_dim, key=keys[-1])
        self.activation = activation
        self.pool_type = pool_type

    def __call__(self, u, *, key=None, **kwargs):
        if u.ndim == 1:
            u = u[:, jnp.newaxis]
        x = u

        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.norm_layers[i] is not None:
                x = self.norm_layers[i](x)
            x = self.activation(x)
            x = avg_pool_1d(x, window_shape=(2,), strides=(2,))

        if self.pool_type == "avg":
            x = jnp.mean(x, axis=0)
        elif self.pool_type == "max":
            x = jnp.max(x, axis=0)
        elif self.pool_type == "attention":
            attn_weights = jax.vmap(self.attn_dense)(x)
            attn_weights = jax.nn.softmax(attn_weights, axis=0)
            x = jnp.sum(x * attn_weights, axis=0)

        return self.output_dense(x)


class BranchTransformer(eqx.Module):
    """Transformer-based branch network."""

    input_proj: Linear
    cls_token: Optional[jnp.ndarray]
    pos_embedding: jnp.ndarray
    blocks: list
    output_proj: Linear
    pool_type: str = eqx.field(static=True)

    def __init__(self, in_features, n_sensors, d_model, output_dim, n_layers=4, n_heads=8, mlp_ratio=4.0, dropout_rate=0.0, activation=jax.nn.gelu, pool_type="cls", *, key, **kwargs):
        keys = jax.random.split(key, n_layers + 4)
        self.pool_type = pool_type
        self.input_proj = Linear(in_features, d_model, key=keys[0])

        seq_len = n_sensors + (1 if pool_type == "cls" else 0)
        self.cls_token = jax.random.normal(keys[1], (1, d_model)) * 0.02 if pool_type == "cls" else None
        self.pos_embedding = jax.random.normal(keys[2], (seq_len, d_model)) * 0.02

        self.blocks = [TransformerBlock(d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate, activation=activation, key=keys[i + 3]) for i in range(n_layers)]
        self.output_proj = Linear(d_model, output_dim, key=keys[-1])

    def __call__(self, u, *, key=None, **kwargs):
        if u.ndim == 1:
            u = u[:, jnp.newaxis]

        x = jax.vmap(self.input_proj)(u)

        if self.pool_type == "cls":
            x = jnp.concatenate([self.cls_token, x], axis=0)

        x = x + self.pos_embedding

        for block in self.blocks:
            if key is not None:
                key, subkey = jax.random.split(key)
            else:
                subkey = None
            x = block(x, key=subkey)

        if self.pool_type == "cls":
            x = x[0]
        elif self.pool_type == "avg":
            x = jnp.mean(x, axis=0)
        elif self.pool_type == "max":
            x = jnp.max(x, axis=0)

        return self.output_proj(x)


# =============================================================================
# Trunk Networks
# =============================================================================


class TrunkMLP(eqx.Module):
    """MLP-based trunk network."""

    embedding: object
    mlp: DeepONetMLP

    def __init__(self, coord_dim, hidden_dims, output_dim, activation=jax.nn.gelu, norm=None, dropout_rate=0.0, coord_embedding=None, coord_embedding_dim=64, coord_embedding_scale=1.0, *, key, **kwargs):
        k1, k2 = jax.random.split(key)
        effective_dim = coord_dim
        if coord_embedding == "fourier":
            self.embedding = FourierFeatures(coord_dim, n_features=coord_embedding_dim // 2, scale=coord_embedding_scale, key=k1)
            effective_dim = coord_dim + coord_embedding_dim
        elif coord_embedding == "positional":
            self.embedding = PositionalEncoding(d_model=coord_embedding_dim)
            effective_dim = coord_dim + coord_embedding_dim
        else:
            self.embedding = None

        self.mlp = DeepONetMLP(effective_dim, hidden_dims, output_dim, activation=activation, norm=norm, dropout_rate=dropout_rate, key=k2)

    def __call__(self, y, *, key=None, **kwargs):
        single_point = y.ndim == 1
        if single_point:
            y = y[jnp.newaxis, :]

        if self.embedding is not None:
            y_emb = self.embedding(y)
            y = jnp.concatenate([y, y_emb], axis=-1)

        x = jax.vmap(lambda yi: self.mlp(yi, key=key))(y)

        if single_point:
            x = x[0]
        return x


class TrunkResMLP(eqx.Module):
    """Residual MLP-based trunk network."""

    embedding: object
    resmlp: ResMLP

    def __init__(self, coord_dim, hidden_dim, output_dim, n_blocks=4, activation=jax.nn.gelu, norm="layer", dropout_rate=0.0, coord_embedding=None, coord_embedding_dim=64, coord_embedding_scale=1.0, *, key, **kwargs):
        k1, k2 = jax.random.split(key)
        effective_dim = coord_dim
        if coord_embedding == "fourier":
            self.embedding = FourierFeatures(coord_dim, n_features=coord_embedding_dim // 2, scale=coord_embedding_scale, key=k1)
            effective_dim = coord_dim + coord_embedding_dim
        elif coord_embedding == "positional":
            self.embedding = PositionalEncoding(d_model=coord_embedding_dim)
            effective_dim = coord_dim + coord_embedding_dim
        else:
            self.embedding = None

        self.resmlp = ResMLP(effective_dim, hidden_dim, output_dim, n_blocks=n_blocks, activation=activation, norm=norm, dropout_rate=dropout_rate, key=k2)

    def __call__(self, y, *, key=None, **kwargs):
        single_point = y.ndim == 1
        if single_point:
            y = y[jnp.newaxis, :]

        if self.embedding is not None:
            y_emb = self.embedding(y)
            y = jnp.concatenate([y, y_emb], axis=-1)

        x = jax.vmap(lambda yi: self.resmlp(yi, key=key))(y)

        if single_point:
            x = x[0]
        return x


class TrunkSIREN(eqx.Module):
    """SIREN (Sinusoidal Representation Network) trunk for implicit representations."""

    first_layer: Linear
    hidden_layers: list
    output_layer: Linear
    omega_0: float = eqx.field(static=True)
    omega_hidden: float = eqx.field(static=True)

    def __init__(self, coord_dim, hidden_dims, output_dim, omega_0=30.0, omega_hidden=30.0, *, key, **kwargs):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        s0 = 1.0 / coord_dim
        k_w, k_rest = jax.random.split(keys[0])
        self.first_layer = Linear(coord_dim, hidden_dims[0], key=keys[0])
        # Override weight with SIREN init
        self.first_layer = eqx.tree_at(
            lambda m: m.weight,
            self.first_layer,
            jax.random.uniform(k_w, self.first_layer.weight.shape, minval=-s0, maxval=s0),
        )

        self.hidden_layers = []
        for i in range(1, len(hidden_dims)):
            c = jnp.sqrt(6.0 / hidden_dims[i - 1]) / omega_hidden
            layer = Linear(hidden_dims[i - 1], hidden_dims[i], key=keys[i])
            k_w2, _ = jax.random.split(keys[i])
            layer = eqx.tree_at(
                lambda m: m.weight,
                layer,
                jax.random.uniform(k_w2, layer.weight.shape, minval=-c, maxval=c),
            )
            self.hidden_layers.append(layer)

        c_out = jnp.sqrt(6.0 / hidden_dims[-1]) / omega_hidden
        self.output_layer = Linear(hidden_dims[-1], output_dim, key=keys[-1])
        k_w3, _ = jax.random.split(keys[-1])
        self.output_layer = eqx.tree_at(
            lambda m: m.weight,
            self.output_layer,
            jax.random.uniform(k_w3, self.output_layer.weight.shape, minval=-c_out, maxval=c_out),
        )

        self.omega_0 = omega_0
        self.omega_hidden = omega_hidden

    def __call__(self, y, **kwargs):
        single_point = y.ndim == 1
        if single_point:
            y = y[jnp.newaxis, :]

        def forward_single(yi):
            x = jnp.sin(self.omega_0 * self.first_layer(yi))
            for layer in self.hidden_layers:
                x = jnp.sin(self.omega_hidden * layer(x))
            return self.output_layer(x)

        x = jax.vmap(forward_single)(y)

        if single_point:
            x = x[0]
        return x


# =============================================================================
# Combination Methods
# =============================================================================


class DotProductCombination(eqx.Module):
    """Standard dot product combination: sum(branch * trunk)."""

    def __call__(self, branch_out, trunk_out, **kwargs):
        if trunk_out.ndim == 1:
            return jnp.sum(branch_out * trunk_out)
        elif trunk_out.ndim == 2 and branch_out.ndim == 1:
            return jnp.sum(branch_out * trunk_out, axis=-1)
        elif trunk_out.ndim == 2 and branch_out.ndim == 2:
            return jnp.einsum("op,np->no", branch_out, trunk_out)
        else:
            return jnp.einsum("...p,...p->...", branch_out, trunk_out)


class BilinearCombination(eqx.Module):
    """Bilinear combination with learnable interaction matrix."""

    W: jnp.ndarray
    output_dim: int = eqx.field(static=True)

    def __init__(self, p, output_dim=1, *, key, **kwargs):
        fan_in = p
        std = 1.0 / jnp.sqrt(fan_in)
        self.W = jax.random.normal(key, (p, p, output_dim)) * std
        self.output_dim = output_dim

    def __call__(self, branch_out, trunk_out, **kwargs):
        if trunk_out.ndim == 1:
            return jnp.einsum("i,j,ijo->o", branch_out, trunk_out, self.W)
        else:
            return jnp.einsum("i,nj,ijo->no", branch_out, trunk_out, self.W)


class MLPCombination(eqx.Module):
    """MLP-based combination for more expressive interactions."""

    mlp: DeepONetMLP
    gamma_proj: Optional[Linear]
    beta_proj: Optional[Linear]
    combination_mode: str = eqx.field(static=True)

    def __init__(self, p_branch, p_trunk, hidden_dims=(128, 64), output_dim=1, activation=jax.nn.gelu, combination_mode="concat", *, key, **kwargs):
        k1, k2, k3 = jax.random.split(key, 3)
        self.combination_mode = combination_mode

        if combination_mode == "concat":
            in_features = p_branch + p_trunk
        elif combination_mode in ("add", "multiply"):
            in_features = p_branch
        elif combination_mode == "film":
            in_features = p_trunk
            self.gamma_proj = Linear(p_branch, p_trunk, key=k2)
            self.beta_proj = Linear(p_branch, p_trunk, key=k3)
        else:
            in_features = p_branch + p_trunk

        if combination_mode != "film":
            self.gamma_proj = None
            self.beta_proj = None

        self.mlp = DeepONetMLP(in_features, list(hidden_dims), output_dim, activation=activation, key=k1)

    def __call__(self, branch_out, trunk_out, *, key=None, **kwargs):
        single_point = trunk_out.ndim == 1
        if single_point:
            trunk_out = trunk_out[jnp.newaxis, :]

        n_points = trunk_out.shape[0]
        branch_expanded = repeat(branch_out, "p -> n p", n=n_points)

        if self.combination_mode == "concat":
            x = jnp.concatenate([branch_expanded, trunk_out], axis=-1)
        elif self.combination_mode == "add":
            x = branch_expanded + trunk_out
        elif self.combination_mode == "multiply":
            x = branch_expanded * trunk_out
        elif self.combination_mode == "film":
            gamma = jax.vmap(self.gamma_proj)(branch_expanded)
            beta = jax.vmap(self.beta_proj)(branch_expanded)
            x = gamma * trunk_out + beta

        x = jax.vmap(lambda xi: self.mlp(xi, key=key))(x)

        if single_point:
            x = x[0]
        return x


class CrossAttention(eqx.Module):
    """Cross-attention for combining branch and trunk outputs."""

    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    n_heads: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(self, in_features_query, in_features_kv, d_model, n_heads=4, *, key, **kwargs):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.q_proj = Linear(in_features_query, d_model, key=k1)
        self.k_proj = Linear(in_features_kv, d_model, key=k2)
        self.v_proj = Linear(in_features_kv, d_model, key=k3)
        self.out_proj = Linear(d_model, d_model, key=k4)
        self.n_heads = n_heads
        self.d_model = d_model

    def __call__(self, query, kv, **kwargs):
        head_dim = self.d_model // self.n_heads
        q = jax.vmap(self.q_proj)(query).reshape(-1, self.n_heads, head_dim).transpose(1, 0, 2)
        k = jax.vmap(self.k_proj)(kv).reshape(-1, self.n_heads, head_dim).transpose(1, 0, 2)
        v = jax.vmap(self.v_proj)(kv).reshape(-1, self.n_heads, head_dim).transpose(1, 0, 2)

        scale = jnp.sqrt(jnp.array(head_dim, dtype=query.dtype))
        attn = jnp.einsum("hqd,hkd->hqk", q, k) / scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("hqk,hkd->hqd", attn, v)
        out = out.transpose(1, 0, 2).reshape(-1, self.d_model)
        return jax.vmap(self.out_proj)(out)


class AttentionCombination(eqx.Module):
    """Cross-attention based combination."""

    branch_proj: Linear
    trunk_proj: Linear
    cross_attn: CrossAttention
    output_proj: Linear

    def __init__(self, p_branch, p_trunk, d_model=128, n_heads=4, output_dim=1, *, key, **kwargs):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.branch_proj = Linear(p_branch, d_model, key=k1)
        self.trunk_proj = Linear(p_trunk, d_model, key=k2)
        self.cross_attn = CrossAttention(d_model, d_model, d_model, n_heads=n_heads, key=k3)
        self.output_proj = Linear(d_model, output_dim, key=k4)

    def __call__(self, branch_out, trunk_out, **kwargs):
        single_point = trunk_out.ndim == 1
        if single_point:
            trunk_out = trunk_out[jnp.newaxis, :]

        branch_proj = self.branch_proj(branch_out)[jnp.newaxis, :]
        trunk_proj = jax.vmap(self.trunk_proj)(trunk_out)

        x = self.cross_attn(trunk_proj, branch_proj)
        x = jax.vmap(self.output_proj)(x)

        if single_point:
            x = x[0]
        return x


# =============================================================================
# Main DeepONet
# =============================================================================


class DeepONet(eqx.Module):
    """
    Generalized Deep Operator Network with extensive customization options.

    This implementation supports:
    - Multiple branch network architectures (MLP, ResMLP, Conv1D, Transformer)
    - Multiple trunk network architectures (MLP, ResMLP, SIREN)
    - Various combination methods (dot product, bilinear, MLP, attention)
    - Coordinate embeddings (Fourier features, positional encoding)
    - Multiple outputs and multi-physics problems
    - Flexible normalization and regularization
    """

    branch_net: object
    trunk_net: object
    combiner: object
    output_bias: Optional[jnp.ndarray]

    n_outputs: int = eqx.field(static=True)
    basis_functions: int = eqx.field(static=True)
    combination_type: str = eqx.field(static=True)
    use_output_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        branch_type="mlp",
        trunk_type="mlp",
        combination_type="dot",
        n_sensors=100,
        sensor_channels=1,
        coord_dim=1,
        n_outputs=1,
        basis_functions=128,
        branch_hidden_dims=(256, 256, 256),
        branch_hidden_dim=256,
        branch_n_blocks=4,
        branch_n_layers=4,
        branch_n_heads=8,
        branch_channels=(32, 64, 128),
        branch_pool_type="avg",
        trunk_hidden_dims=(256, 256, 256),
        trunk_hidden_dim=256,
        trunk_n_blocks=4,
        trunk_omega_0=30.0,
        trunk_omega_hidden=30.0,
        coord_embedding=None,
        coord_embedding_dim=64,
        coord_embedding_scale=1.0,
        combination_hidden_dims=(128, 64),
        combination_mode="concat",
        combination_d_model=128,
        combination_n_heads=4,
        activation=jax.nn.gelu,
        norm=None,
        dropout_rate=0.0,
        use_output_bias=True,
        *,
        key,
        **kwargs,
    ):
        self.n_outputs = n_outputs
        self.basis_functions = basis_functions
        self.combination_type = combination_type
        self.use_output_bias = use_output_bias

        p = basis_functions * n_outputs
        k1, k2, k3 = jax.random.split(key, 3)

        # Build branch network
        if branch_type == "mlp":
            branch_in = n_sensors * sensor_channels
            self.branch_net = BranchMLP(branch_in, branch_hidden_dims, p, activation=activation, norm=norm, dropout_rate=dropout_rate, key=k1)
        elif branch_type == "resmlp":
            branch_in = n_sensors * sensor_channels
            self.branch_net = BranchResMLP(branch_in, branch_hidden_dim, p, n_blocks=branch_n_blocks, activation=activation, norm=norm, dropout_rate=dropout_rate, key=k1)
        elif branch_type == "conv1d":
            self.branch_net = BranchConv1D(sensor_channels, branch_channels, p, activation=activation, norm=norm, pool_type=branch_pool_type, key=k1)
        elif branch_type == "transformer":
            self.branch_net = BranchTransformer(sensor_channels, n_sensors, branch_hidden_dim, p, n_layers=branch_n_layers, n_heads=branch_n_heads, dropout_rate=dropout_rate, activation=activation, pool_type=branch_pool_type, key=k1)
        else:
            raise ValueError(f"Unknown branch_type: {branch_type}")

        # Build trunk network
        if trunk_type == "mlp":
            self.trunk_net = TrunkMLP(
                coord_dim, trunk_hidden_dims, p, activation=activation, norm=norm, dropout_rate=dropout_rate, coord_embedding=coord_embedding, coord_embedding_dim=coord_embedding_dim, coord_embedding_scale=coord_embedding_scale, key=k2
            )
        elif trunk_type == "resmlp":
            self.trunk_net = TrunkResMLP(
                coord_dim,
                trunk_hidden_dim,
                p,
                n_blocks=trunk_n_blocks,
                activation=activation,
                norm=norm,
                dropout_rate=dropout_rate,
                coord_embedding=coord_embedding,
                coord_embedding_dim=coord_embedding_dim,
                coord_embedding_scale=coord_embedding_scale,
                key=k2,
            )
        elif trunk_type == "siren":
            self.trunk_net = TrunkSIREN(coord_dim, trunk_hidden_dims, p, omega_0=trunk_omega_0, omega_hidden=trunk_omega_hidden, key=k2)
        else:
            raise ValueError(f"Unknown trunk_type: {trunk_type}")

        # Build combination method
        if combination_type == "dot":
            self.combiner = DotProductCombination()
        elif combination_type == "bilinear":
            self.combiner = BilinearCombination(p, output_dim=n_outputs, key=k3)
        elif combination_type == "mlp":
            self.combiner = MLPCombination(p, p, hidden_dims=combination_hidden_dims, output_dim=n_outputs, activation=activation, combination_mode=combination_mode, key=k3)
        elif combination_type == "attention":
            self.combiner = AttentionCombination(p, p, d_model=combination_d_model, n_heads=combination_n_heads, output_dim=n_outputs, key=k3)
        else:
            raise ValueError(f"Unknown combination_type: {combination_type}")

        self.output_bias = jnp.zeros(n_outputs) if use_output_bias else None

    def __call__(self, u, y, *, key=None, **kwargs):
        """
        Forward pass of DeepONet.

        Args:
            u: Input function values at sensor points
               Shape: (n_sensors,) or (n_sensors, sensor_channels)
            y: Query coordinates
               Shape: (coord_dim,) or (n_points, coord_dim)

        Returns:
            Output values at query points.
        """
        if key is not None:
            k1, k2 = jax.random.split(key)
        else:
            k1 = k2 = None

        branch_out = self.branch_net(u, key=k1)
        trunk_out = self.trunk_net(y, key=k2)

        if self.combination_type == "dot" and self.n_outputs > 1:
            branch_out = branch_out.reshape(self.n_outputs, self.basis_functions)
            if trunk_out.ndim == 1:
                trunk_out = trunk_out.reshape(self.n_outputs, self.basis_functions)
            else:
                trunk_out = trunk_out.reshape(-1, self.n_outputs, self.basis_functions)

        output = self.combiner(branch_out, trunk_out)

        if self.output_bias is not None:
            output = output + self.output_bias

        if self.n_outputs == 1 and output.ndim > 0 and output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output

    def evaluate_batch(self, u, y, *, key=None, **kwargs):
        """
        Evaluate on a batch of input functions and query points.

        Args:
            u: Batch of input functions, shape (batch, n_sensors, ...) or (batch, n_sensors)
            y: Query points, shape (n_points, coord_dim) - shared across batch

        Returns:
            Output values, shape (batch, n_points) or (batch, n_points, n_outputs)
        """
        return jax.vmap(lambda u_i: self(u_i, y, key=key))(u)
