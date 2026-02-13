# Generalized Deep Operator Network (DeepONet)

from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Callable, Optional, Sequence, Literal
from flax.linen.initializers import lecun_normal, zeros
from einops import repeat


# =============================================================================
# Building Blocks
# =============================================================================


class MLPBlock(nn.Module):
    """Flexible MLP block with various normalization and regularization options."""

    features: int
    activation: Callable = nn.gelu
    norm: Optional[str] = None
    dropout_rate: float = 0.0
    use_bias: bool = True
    kernel_init: Callable = lecun_normal()
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
        )(x)

        if self.norm == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm == "batch":
            x = nn.BatchNorm(use_running_average=not self.training)(x)
        elif self.norm == "group":
            x = nn.GroupNorm(num_groups=min(32, self.features))(x)
        elif self.norm == "rms":
            x = nn.RMSNorm()(x)

        x = self.activation(x)

        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)

        return x


class MLP(nn.Module):
    """Multi-layer perceptron with flexible architecture."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.gelu
    output_activation: Optional[Callable] = None
    norm: Optional[str] = None
    dropout_rate: float = 0.0
    use_bias: bool = True
    kernel_init: Callable = lecun_normal()
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = MLPBlock(
                features=dim,
                activation=self.activation,
                norm=self.norm,
                dropout_rate=self.dropout_rate,
                use_bias=self.use_bias,
                kernel_init=self.kernel_init,
                training=self.training,
            )(x)

        # Output layer (no norm, no dropout)
        x = nn.Dense(
            features=self.output_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
        )(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""

    features: int
    activation: Callable = nn.gelu
    norm: Optional[str] = "layer"
    dropout_rate: float = 0.0
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x

        # Project residual if dimensions don't match
        if x.shape[-1] != self.features:
            residual = nn.Dense(features=self.features, use_bias=False)(x)

        x = MLPBlock(
            features=self.features,
            activation=self.activation,
            norm=self.norm,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )(x)

        x = nn.Dense(features=self.features)(x)

        if self.norm == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm == "batch":
            x = nn.BatchNorm(use_running_average=not self.training)(x)

        return self.activation(x + residual)


class ResMLP(nn.Module):
    """MLP with residual connections."""

    hidden_dim: int
    output_dim: int
    n_blocks: int = 4
    activation: Callable = nn.gelu
    output_activation: Optional[Callable] = None
    norm: Optional[str] = "layer"
    dropout_rate: float = 0.0
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Initial projection
        x = nn.Dense(features=self.hidden_dim)(x)
        x = self.activation(x)

        # Residual blocks
        for _ in range(self.n_blocks):
            x = ResidualBlock(
                features=self.hidden_dim,
                activation=self.activation,
                norm=self.norm,
                dropout_rate=self.dropout_rate,
                training=self.training,
            )(x)

        # Output projection
        x = nn.Dense(features=self.output_dim)(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


class FourierFeatures(nn.Module):
    """Fourier feature embedding for improved coordinate representation."""

    n_features: int = 64
    scale: float = 1.0
    learnable: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_dim = x.shape[-1]

        if self.learnable:
            B = self.param(
                "frequency_matrix",
                nn.initializers.normal(stddev=self.scale),
                (input_dim, self.n_features),
            )
        else:
            # Fixed random frequencies
            B = self.variable(
                "constants",
                "frequency_matrix",
                lambda: jax.random.normal(jax.random.PRNGKey(0), (input_dim, self.n_features)) * self.scale,
            ).value

        x_proj = x @ B
        return jnp.concatenate([jnp.sin(2 * jnp.pi * x_proj), jnp.cos(2 * jnp.pi * x_proj)], axis=-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    d_model: int
    max_len: int = 10000
    base: float = 10000.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: coordinates of shape (..., input_dim)"""
        input_dim = x.shape[-1]
        d_per_dim = self.d_model // input_dim

        encodings = []
        for i in range(input_dim):
            pos = x[..., i : i + 1]  # (..., 1)
            div_term = jnp.exp(jnp.arange(0, d_per_dim, 2) * (-jnp.log(self.base) / d_per_dim))

            pe = jnp.zeros((*x.shape[:-1], d_per_dim))
            pe = pe.at[..., 0::2].set(jnp.sin(pos * div_term))
            pe = pe.at[..., 1::2].set(jnp.cos(pos * div_term))
            encodings.append(pe)

        return jnp.concatenate(encodings, axis=-1)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    n_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.0
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape: (seq_len, features) or (batch, seq_len, features)
        input_shape = x.shape
        if x.ndim == 2:
            x = x[jnp.newaxis, :, :]  # Add batch dim

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.n_heads * self.head_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not self.training,
        )(x, x)

        if len(input_shape) == 2:
            x = x[0]  # Remove batch dim

        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    d_model: int
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu
    norm: str = "layer"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Pre-norm architecture
        residual = x

        if self.norm == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm == "rms":
            x = nn.RMSNorm()(x)

        x = AttentionBlock(
            n_heads=self.n_heads,
            head_dim=self.d_model // self.n_heads,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)

        x = x + residual

        # MLP
        residual = x

        if self.norm == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm == "rms":
            x = nn.RMSNorm()(x)

        mlp_dim = int(self.d_model * self.mlp_ratio)
        x = nn.Dense(features=mlp_dim)(x)
        x = self.activation(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)

        x = nn.Dense(features=self.d_model)(x)

        if self.dropout_rate > 0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)

        return x + residual


# =============================================================================
# Branch Networks
# =============================================================================


class BranchMLP(nn.Module):
    """MLP-based branch network."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.gelu
    norm: Optional[str] = None
    dropout_rate: float = 0.0
    training: bool = True

    @nn.compact
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            u: Input function values at sensor points, shape (n_sensors,) or (n_sensors, n_channels)

        Returns:
            Branch output of shape (output_dim,)
        """
        # Flatten input
        x = u.reshape(-1)

        return MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
            norm=self.norm,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )(x)


class BranchResMLP(nn.Module):
    """Residual MLP-based branch network."""

    hidden_dim: int
    output_dim: int
    n_blocks: int = 4
    activation: Callable = nn.gelu
    norm: Optional[str] = "layer"
    dropout_rate: float = 0.0
    training: bool = True

    @nn.compact
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        x = u.reshape(-1)

        return ResMLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_blocks=self.n_blocks,
            activation=self.activation,
            norm=self.norm,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )(x)


class BranchConv1D(nn.Module):
    """1D CNN-based branch network for sequential sensor data."""

    channels: Sequence[int]
    output_dim: int
    kernel_size: int = 3
    activation: Callable = nn.gelu
    norm: Optional[str] = "batch"
    pool_type: str = "avg"  # 'avg', 'max', or 'attention'
    training: bool = True

    @nn.compact
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            u: Shape (n_sensors,) or (n_sensors, n_channels)
        """
        if u.ndim == 1:
            u = u[:, jnp.newaxis]

        x = u

        for ch in self.channels:
            x = nn.Conv(features=ch, kernel_size=(self.kernel_size,), padding="SAME")(x)

            if self.norm == "batch":
                x = nn.BatchNorm(use_running_average=not self.training)(x)
            elif self.norm == "layer":
                x = nn.LayerNorm()(x)

            x = self.activation(x)
            x = nn.avg_pool(x, window_shape=(2,), strides=(2,), padding="SAME")

        # Global pooling
        if self.pool_type == "avg":
            x = jnp.mean(x, axis=0)
        elif self.pool_type == "max":
            x = jnp.max(x, axis=0)
        elif self.pool_type == "attention":
            # Attention pooling
            attn_weights = nn.Dense(features=1)(x)
            attn_weights = nn.softmax(attn_weights, axis=0)
            x = jnp.sum(x * attn_weights, axis=0)

        return nn.Dense(features=self.output_dim)(x)


class BranchTransformer(nn.Module):
    """Transformer-based branch network."""

    d_model: int
    output_dim: int
    n_layers: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    activation: Callable = nn.gelu
    pool_type: str = "cls"  # 'cls', 'avg', 'max'
    training: bool = True

    @nn.compact
    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            u: Shape (n_sensors,) or (n_sensors, n_channels)
        """
        if u.ndim == 1:
            u = u[:, jnp.newaxis]

        # Project to d_model
        x = nn.Dense(features=self.d_model)(u)

        # Add CLS token if using cls pooling
        if self.pool_type == "cls":
            cls_token = self.param("cls_token", nn.initializers.normal(stddev=0.02), (1, self.d_model))
            x = jnp.concatenate([cls_token, x], axis=0)

        # Add positional encoding
        seq_len = x.shape[0]
        pos_emb = self.param("pos_embedding", nn.initializers.normal(stddev=0.02), (seq_len, self.d_model))
        x = x + pos_emb

        # Transformer blocks
        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                training=self.training,
            )(x)

        # Pooling
        if self.pool_type == "cls":
            x = x[0]
        elif self.pool_type == "avg":
            x = jnp.mean(x, axis=0)
        elif self.pool_type == "max":
            x = jnp.max(x, axis=0)

        return nn.Dense(features=self.output_dim)(x)


# =============================================================================

# Trunk Networks

# =============================================================================


class TrunkMLP(nn.Module):
    """MLP-based trunk network."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = nn.gelu
    norm: Optional[str] = None
    dropout_rate: float = 0.0
    coord_embedding: Optional[str] = None  # 'fourier', 'positional', None
    coord_embedding_dim: int = 64
    coord_embedding_scale: float = 1.0
    training: bool = True

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            y: Query coordinates, shape (coord_dim,) or (n_points, coord_dim)

        Returns:
            Trunk output of shape (output_dim,) or (n_points, output_dim)
        """
        single_point = y.ndim == 1
        if single_point:
            y = y[jnp.newaxis, :]

        # Coordinate embedding
        if self.coord_embedding == "fourier":
            y_emb = FourierFeatures(n_features=self.coord_embedding_dim // 2, scale=self.coord_embedding_scale)(y)
            y = jnp.concatenate([y, y_emb], axis=-1)
        elif self.coord_embedding == "positional":
            y_emb = PositionalEncoding(d_model=self.coord_embedding_dim)(y)
            y = jnp.concatenate([y, y_emb], axis=-1)

        x = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation,
            norm=self.norm,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )(y)

        if single_point:
            x = x[0]

        return x


class TrunkResMLP(nn.Module):
    """Residual MLP-based trunk network."""

    hidden_dim: int
    output_dim: int
    n_blocks: int = 4
    activation: Callable = nn.gelu
    norm: Optional[str] = "layer"
    dropout_rate: float = 0.0
    coord_embedding: Optional[str] = None
    coord_embedding_dim: int = 64
    coord_embedding_scale: float = 1.0
    training: bool = True

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        single_point = y.ndim == 1
        if single_point:
            y = y[jnp.newaxis, :]

        # Coordinate embedding
        if self.coord_embedding == "fourier":
            y_emb = FourierFeatures(n_features=self.coord_embedding_dim // 2, scale=self.coord_embedding_scale)(y)
            y = jnp.concatenate([y, y_emb], axis=-1)
        elif self.coord_embedding == "positional":
            y_emb = PositionalEncoding(d_model=self.coord_embedding_dim)(y)
            y = jnp.concatenate([y, y_emb], axis=-1)

        x = ResMLP(
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_blocks=self.n_blocks,
            activation=self.activation,
            norm=self.norm,
            dropout_rate=self.dropout_rate,
            training=self.training,
        )(y)

        if single_point:
            x = x[0]

        return x


class TrunkSIREN(nn.Module):
    """SIREN (Sinusoidal Representation Network) trunk for implicit representations."""

    hidden_dims: Sequence[int]
    output_dim: int
    omega_0: float = 30.0  # First layer frequency
    omega_hidden: float = 30.0  # Hidden layer frequency

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        single_point = y.ndim == 1
        if single_point:
            y = y[jnp.newaxis, :]

        input_dim = y.shape[-1]

        # First layer with special initialization
        x = nn.Dense(
            features=self.hidden_dims[0],
            kernel_init=nn.initializers.uniform(scale=1.0 / input_dim),
        )(y)
        x = jnp.sin(self.omega_0 * x)

        # Hidden layers
        for dim in self.hidden_dims[1:]:
            c = jnp.sqrt(6.0 / dim) / self.omega_hidden
            x = nn.Dense(features=dim, kernel_init=nn.initializers.uniform(scale=c))(x)
            x = jnp.sin(self.omega_hidden * x)

        # Output layer
        c = jnp.sqrt(6.0 / self.hidden_dims[-1]) / self.omega_hidden
        x = nn.Dense(features=self.output_dim, kernel_init=nn.initializers.uniform(scale=c))(x)

        if single_point:
            x = x[0]

        return x


# =============================================================================

# Combination Methods

# =============================================================================


class DotProductCombination(nn.Module):
    """Standard dot product combination: sum(branch * trunk)."""

    @nn.compact
    def __call__(self, branch_out: jnp.ndarray, trunk_out: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            branch_out: (p,) or (n_outputs, p)
            trunk_out: (p,) or (n_points, p) or (n_points, n_outputs, p)
        """
        if trunk_out.ndim == 1:
            # Single point, single output
            return jnp.sum(branch_out * trunk_out)
        elif trunk_out.ndim == 2 and branch_out.ndim == 1:
            # Multiple points, single output
            return jnp.sum(branch_out * trunk_out, axis=-1)
        elif trunk_out.ndim == 2 and branch_out.ndim == 2:
            # Multiple points, multiple outputs (branch: n_outputs x p, trunk: n_points x p)
            return jnp.einsum("op,np->no", branch_out, trunk_out)
        else:
            # Most general case
            return jnp.einsum("...p,...p->...", branch_out, trunk_out)


class BilinearCombination(nn.Module):
    """Bilinear combination with learnable interaction matrix."""

    output_dim: int = 1

    @nn.compact
    def __call__(self, branch_out: jnp.ndarray, trunk_out: jnp.ndarray) -> jnp.ndarray:
        p = branch_out.shape[-1]

        W = self.param("interaction_matrix", lecun_normal(), (p, p, self.output_dim))

        if trunk_out.ndim == 1:
            return jnp.einsum("i,j,ijo->o", branch_out, trunk_out, W)
        else:
            return jnp.einsum("i,nj,ijo->no", branch_out, trunk_out, W)


class MLPCombination(nn.Module):
    """MLP-based combination for more expressive interactions."""

    hidden_dims: Sequence[int] = (128, 64)
    output_dim: int = 1
    activation: Callable = nn.gelu
    combination_mode: str = "concat"  # 'concat', 'add', 'multiply', 'film'
    training: bool = True

    @nn.compact
    def __call__(self, branch_out: jnp.ndarray, trunk_out: jnp.ndarray) -> jnp.ndarray:
        single_point = trunk_out.ndim == 1

        if single_point:
            trunk_out = trunk_out[jnp.newaxis, :]

        n_points = trunk_out.shape[0]

        # Expand branch to match trunk
        branch_expanded = repeat(branch_out, "p -> n p", n=n_points)

        # Combine
        if self.combination_mode == "concat":
            x = jnp.concatenate([branch_expanded, trunk_out], axis=-1)
        elif self.combination_mode == "add":
            x = branch_expanded + trunk_out
        elif self.combination_mode == "multiply":
            x = branch_expanded * trunk_out
        elif self.combination_mode == "film":
            # FiLM: Feature-wise Linear Modulation
            gamma = nn.Dense(features=trunk_out.shape[-1])(branch_expanded)
            beta = nn.Dense(features=trunk_out.shape[-1])(branch_expanded)
            x = gamma * trunk_out + beta

        x = MLP(
            hidden_dims=list(self.hidden_dims),
            output_dim=self.output_dim,
            activation=self.activation,
            training=self.training,
        )(x)

        if single_point:
            x = x[0]

        return x


class AttentionCombination(nn.Module):
    """Cross-attention based combination."""

    d_model: int = 128
    n_heads: int = 4
    output_dim: int = 1
    training: bool = True

    @nn.compact
    def __call__(self, branch_out: jnp.ndarray, trunk_out: jnp.ndarray) -> jnp.ndarray:
        single_point = trunk_out.ndim == 1

        if single_point:
            trunk_out = trunk_out[jnp.newaxis, :]

        # Project to d_model
        branch_proj = nn.Dense(features=self.d_model)(branch_out)
        trunk_proj = nn.Dense(features=self.d_model)(trunk_out)

        # Branch as key/value, trunk as query
        branch_proj = branch_proj[jnp.newaxis, :]  # (1, d_model)

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            deterministic=not self.training,
        )(trunk_proj[jnp.newaxis, :, :], branch_proj[jnp.newaxis, :, :])

        x = x[0]  # Remove batch dim
        x = nn.Dense(features=self.output_dim)(x)

        if single_point:
            x = x[0]

        return x


# =============================================================================

# Main DeepONet

# =============================================================================


class DeepONet(nn.Module):
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

    # Architecture choices
    branch_type: Literal["mlp", "resmlp", "conv1d", "transformer"] = "mlp"
    trunk_type: Literal["mlp", "resmlp", "siren"] = "mlp"
    combination_type: Literal["dot", "bilinear", "mlp", "attention"] = "dot"

    # Dimensions
    n_sensors: int = 100  # Number of sensor points for branch input
    sensor_channels: int = 1  # Channels per sensor
    coord_dim: int = 1  # Dimension of query coordinates
    n_outputs: int = 1  # Number of output fields
    basis_functions: int = 128  # Number of basis functions (p)

    # Branch network config
    branch_hidden_dims: Sequence[int] = (256, 256, 256)
    branch_hidden_dim: int = 256  # For ResMLP/Transformer
    branch_n_blocks: int = 4  # For ResMLP
    branch_n_layers: int = 4  # For Transformer
    branch_n_heads: int = 8  # For Transformer
    branch_channels: Sequence[int] = (32, 64, 128)  # For Conv1D
    branch_pool_type: str = "avg"  # For Conv1D/Transformer

    # Trunk network config
    trunk_hidden_dims: Sequence[int] = (256, 256, 256)
    trunk_hidden_dim: int = 256  # For ResMLP
    trunk_n_blocks: int = 4  # For ResMLP
    trunk_omega_0: float = 30.0  # For SIREN
    trunk_omega_hidden: float = 30.0  # For SIREN

    # Coordinate embedding
    coord_embedding: Optional[Literal["fourier", "positional"]] = None
    coord_embedding_dim: int = 64
    coord_embedding_scale: float = 1.0

    # Combination config (for MLP/attention combination)
    combination_hidden_dims: Sequence[int] = (128, 64)
    combination_mode: str = "concat"  # For MLP combination
    combination_d_model: int = 128  # For attention combination
    combination_n_heads: int = 4  # For attention combination

    # Regularization
    activation: Callable = nn.gelu
    norm: Optional[str] = None  # 'layer', 'batch', 'group', 'rms', None
    dropout_rate: float = 0.0

    # Training
    training: bool = True

    # Bias
    use_output_bias: bool = True

    def setup(self):
        # Effective output dimension for branch/trunk
        p = self.basis_functions * self.n_outputs

        # Build branch network
        if self.branch_type == "mlp":
            self.branch_net = BranchMLP(
                hidden_dims=self.branch_hidden_dims,
                output_dim=p,
                activation=self.activation,
                norm=self.norm,
                dropout_rate=self.dropout_rate,
                training=self.training,
            )
        elif self.branch_type == "resmlp":
            self.branch_net = BranchResMLP(
                hidden_dim=self.branch_hidden_dim,
                output_dim=p,
                n_blocks=self.branch_n_blocks,
                activation=self.activation,
                norm=self.norm,
                dropout_rate=self.dropout_rate,
                training=self.training,
            )
        elif self.branch_type == "conv1d":
            self.branch_net = BranchConv1D(
                channels=self.branch_channels,
                output_dim=p,
                activation=self.activation,
                norm=self.norm,
                pool_type=self.branch_pool_type,
                training=self.training,
            )
        elif self.branch_type == "transformer":
            self.branch_net = BranchTransformer(
                d_model=self.branch_hidden_dim,
                output_dim=p,
                n_layers=self.branch_n_layers,
                n_heads=self.branch_n_heads,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                pool_type=self.branch_pool_type,
                training=self.training,
            )

        # Build trunk network
        if self.trunk_type == "mlp":
            self.trunk_net = TrunkMLP(
                hidden_dims=self.trunk_hidden_dims,
                output_dim=p,
                activation=self.activation,
                norm=self.norm,
                dropout_rate=self.dropout_rate,
                coord_embedding=self.coord_embedding,
                coord_embedding_dim=self.coord_embedding_dim,
                coord_embedding_scale=self.coord_embedding_scale,
                training=self.training,
            )
        elif self.trunk_type == "resmlp":
            self.trunk_net = TrunkResMLP(
                hidden_dim=self.trunk_hidden_dim,
                output_dim=p,
                n_blocks=self.trunk_n_blocks,
                activation=self.activation,
                norm=self.norm,
                dropout_rate=self.dropout_rate,
                coord_embedding=self.coord_embedding,
                coord_embedding_dim=self.coord_embedding_dim,
                coord_embedding_scale=self.coord_embedding_scale,
                training=self.training,
            )
        elif self.trunk_type == "siren":
            self.trunk_net = TrunkSIREN(
                hidden_dims=self.trunk_hidden_dims,
                output_dim=p,
                omega_0=self.trunk_omega_0,
                omega_hidden=self.trunk_omega_hidden,
            )

        # Build combination method
        if self.combination_type == "dot":
            self.combiner = DotProductCombination()
        elif self.combination_type == "bilinear":
            self.combiner = BilinearCombination(output_dim=self.n_outputs)
        elif self.combination_type == "mlp":
            self.combiner = MLPCombination(
                hidden_dims=self.combination_hidden_dims,
                output_dim=self.n_outputs,
                activation=self.activation,
                combination_mode=self.combination_mode,
                training=self.training,
            )
        elif self.combination_type == "attention":
            self.combiner = AttentionCombination(
                d_model=self.combination_d_model,
                n_heads=self.combination_n_heads,
                output_dim=self.n_outputs,
                training=self.training,
            )

        # Output bias
        if self.use_output_bias:
            self.output_bias = self.param("output_bias", zeros, (self.n_outputs,))

    def __call__(self, u: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of DeepONet.

        Args:
            u: Input function values at sensor points
               Shape: (n_sensors,) or (n_sensors, sensor_channels)
            y: Query coordinates
               Shape: (coord_dim,) or (n_points, coord_dim)

        Returns:
            Output values at query points
            Shape: () or (n_outputs,) or (n_points,) or (n_points, n_outputs)
        """
        # Get branch output
        branch_out = self.branch_net(u)  # (p,) or (n_outputs * basis_functions,)

        # Get trunk output
        trunk_out = self.trunk_net(y)  # (p,) or (n_points, p)

        # Reshape for multiple outputs if using dot product
        if self.combination_type == "dot" and self.n_outputs > 1:
            branch_out = branch_out.reshape(self.n_outputs, self.basis_functions)
            if trunk_out.ndim == 1:
                trunk_out = trunk_out.reshape(self.n_outputs, self.basis_functions)
            else:
                trunk_out = trunk_out.reshape(-1, self.n_outputs, self.basis_functions)

        # Combine
        output = self.combiner(branch_out, trunk_out)

        # Add bias
        if self.use_output_bias:
            output = output + self.output_bias

        # Squeeze if single output
        if self.n_outputs == 1 and output.ndim > 0 and output.shape[-1] == 1:
            output = output.squeeze(-1)

        return output

    def evaluate_batch(self, u: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate on a batch of input functions and query points.

        Args:
            u: Batch of input functions, shape (batch, n_sensors, ...) or (batch, n_sensors)
            y: Query points, shape (n_points, coord_dim) - shared across batch

        Returns:
            Output values, shape (batch, n_points) or (batch, n_points, n_outputs)
        """
        return jax.vmap(lambda u_i: self(u_i, y))(u)
