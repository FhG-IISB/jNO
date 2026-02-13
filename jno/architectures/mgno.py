# mgno.py - JAX/Flax implementation of Multigrid Neural Operator

from typing import List, Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn


class MgIte(nn.Module):
    """Single multigrid iteration: u = u + S(f - A(u))"""

    num_channel_u: int
    num_channel_f: int
    padding_mode: str = "CIRCULAR"
    use_bias: bool = False

    @nn.compact
    def __call__(self, u: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            u: Current solution estimate [B, H, W, C_u]
            f: Right-hand side [B, H, W, C_f]
        Returns:
            Updated solution estimate [B, H, W, C_u]
        """
        # A: u -> f (residual operator)
        A_out = nn.Conv(features=self.num_channel_f, kernel_size=(3, 3), strides=(1, 1), padding=self.padding_mode, use_bias=self.use_bias, name="A")(u)

        # S: f -> u (smoothing operator)
        residual = f - A_out
        correction = nn.Conv(features=self.num_channel_u, kernel_size=(3, 3), strides=(1, 1), padding=self.padding_mode, use_bias=self.use_bias, name="S")(residual)

        return u + correction


class MgIteInit(nn.Module):
    """Initial multigrid iteration: u = S(f)"""

    num_channel_u: int
    num_channel_f: int
    padding_mode: str = "CIRCULAR"
    use_bias: bool = False

    @nn.compact
    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            f: Right-hand side [B, H, W, C_f]
        Returns:
            Initial solution estimate [B, H, W, C_u]
        """
        u = nn.Conv(features=self.num_channel_u, kernel_size=(3, 3), strides=(1, 1), padding=self.padding_mode, use_bias=self.use_bias, name="S")(f)
        return u


class Restrict(nn.Module):
    """Restriction operator: coarsens u and f to next level"""

    num_channel_u: int
    num_channel_f: int
    padding_mode: str = "CIRCULAR"

    @nn.compact
    def __call__(self, u: jnp.ndarray, f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            u: Solution [B, H, W, C_u]
            f: Right-hand side [B, H, W, C_f]
        Returns:
            Coarsened (u, f) at half resolution
        """
        # Pi: restrict u (stride 2)
        u_coarse = nn.Conv(features=self.num_channel_u, kernel_size=(3, 3), strides=(2, 2), padding=self.padding_mode, use_bias=False, name="Pi")(u)

        # R: restrict f (stride 2)
        f_coarse = nn.Conv(features=self.num_channel_f, kernel_size=(3, 3), strides=(2, 2), padding=self.padding_mode, use_bias=False, name="R")(f)

        return u_coarse, f_coarse


class Prolongate(nn.Module):
    """Prolongation operator: interpolates correction from coarse to fine"""

    num_channel_u: int
    kernel_size: Tuple[int, int]

    @nn.compact
    def __call__(self, u_coarse: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            u_coarse: Coarse solution [B, H/2, W/2, C_u]
        Returns:
            Interpolated solution [B, H, W, C_u]
        """
        u_fine = nn.ConvTranspose(features=self.num_channel_u, kernel_size=self.kernel_size, strides=(2, 2), padding="SAME", use_bias=False, name="RT")(u_coarse)
        return u_fine


class MgConvLevel(nn.Module):
    """Single level of multigrid V-cycle"""

    num_channel_u: int
    num_channel_f: int
    num_pre_smooth: int
    num_post_smooth: int
    is_first_level: bool
    padding_mode: str = "CIRCULAR"
    use_bias: bool = False

    @nn.compact
    def __call__(self, f: jnp.ndarray, u: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            f: Right-hand side [B, H, W, C_f]
            u: Initial solution (None for first level, first iteration)
        Returns:
            (u, f) after smoothing iterations
        """
        # Pre-smoothing iterations
        for i in range(self.num_pre_smooth):
            if self.is_first_level and i == 0 and u is None:
                # First iteration of first level: initialize u from f
                u = MgIteInit(num_channel_u=self.num_channel_u, num_channel_f=self.num_channel_f, padding_mode=self.padding_mode, use_bias=self.use_bias, name=f"init")(f)
            else:
                u = MgIte(num_channel_u=self.num_channel_u, num_channel_f=self.num_channel_f, padding_mode=self.padding_mode, use_bias=self.use_bias, name=f"pre_smooth_{i}")(u, f)

        return u, f

    def post_smooth(self, u: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        """Post-smoothing after coarse grid correction"""
        for i in range(self.num_post_smooth):
            u = MgIte(num_channel_u=self.num_channel_u, num_channel_f=self.num_channel_f, padding_mode=self.padding_mode, use_bias=self.use_bias, name=f"post_smooth_{i}")(u, f)
        return u


class MgConv(nn.Module):
    """
    Multigrid Convolution Block - implements V-cycle multigrid.

    This is the core building block of MGNO, performing:
    1. Pre-smoothing at each level
    2. Restriction to coarser levels
    3. Coarse grid solve
    4. Prolongation back to finer levels
    5. Post-smoothing at each level

    """

    input_shape: Tuple[int, int]  # (H, W)
    num_iteration: List[Tuple[int, int]]  # [(pre, post), ...] for each level
    num_channel_u: int
    num_channel_f: int
    padding_mode: str = "CIRCULAR"
    use_bias: bool = False

    def setup(self):
        """Compute kernel sizes for prolongation operators"""
        self.num_levels = len(self.num_iteration)

        # Compute kernel sizes for each prolongation (transpose conv)
        # kernel_size depends on whether dimension is odd or even
        kernel_sizes = []
        shape = list(self.input_shape)
        for j in range(self.num_levels - 1):
            # odd => 3, even => 4 (matches PyTorch: 4 - shape % 2)
            ks = (4 - shape[0] % 2, 4 - shape[1] % 2)
            kernel_sizes.append(ks)
            # Update shape for next level (after stride-2 restriction)
            shape = [(shape[0] + 2 - 1) // 2, (shape[1] + 2 - 1) // 2]
        self.kernel_sizes = kernel_sizes

    @nn.compact
    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            f: Input features [B, H, W, C_f]
        Returns:
            Output features [B, H, W, C_u]
        """
        # Storage for intermediate results at each level
        level_outputs = []

        # ============ Downward pass (restriction) ============
        current_f = f
        current_u = None

        for level in range(self.num_levels):
            num_pre, num_post = self.num_iteration[level]
            is_first = level == 0

            # Pre-smoothing at this level
            for i in range(num_pre):
                if is_first and i == 0 and current_u is None:
                    # Initialize u from f
                    current_u = MgIteInit(num_channel_u=self.num_channel_u, num_channel_f=self.num_channel_f, padding_mode=self.padding_mode, use_bias=self.use_bias, name=f"level{level}_init")(current_f)
                else:
                    current_u = MgIte(num_channel_u=self.num_channel_u, num_channel_f=self.num_channel_f, padding_mode=self.padding_mode, use_bias=self.use_bias, name=f"level{level}_pre{i}")(current_u, current_f)

            # Save state at this level for upward pass
            level_outputs.append((current_u, current_f, num_post))

            # Restrict to coarser level (if not at coarsest)
            if level < self.num_levels - 1:
                current_u, current_f = Restrict(num_channel_u=self.num_channel_u, num_channel_f=self.num_channel_f, padding_mode=self.padding_mode, name=f"restrict{level}")(current_u, current_f)

        # ============ Upward pass (prolongation) ============
        for level in range(self.num_levels - 2, -1, -1):
            u_fine, f_fine, num_post = level_outputs[level]
            u_coarse = level_outputs[level + 1][0]

            # Prolongate coarse correction
            u_correction = Prolongate(num_channel_u=self.num_channel_u, kernel_size=self.kernel_sizes[level], name=f"prolongate{level}")(u_coarse)

            # Add correction to fine level solution
            current_u = u_fine + u_correction

            # Post-smoothing
            for i in range(num_post):
                current_u = MgIte(num_channel_u=self.num_channel_u, num_channel_f=self.num_channel_f, padding_mode=self.padding_mode, use_bias=self.use_bias, name=f"level{level}_post{i}")(current_u, f_fine)

            # Update stored output for this level
            level_outputs[level] = (current_u, f_fine, num_post)

        # Return finest level solution
        return level_outputs[0][0]


class MgNO(nn.Module):
    """
    Multigrid Neural Operator.

    Architecture:
    1. Multiple MgConv layers with skip connections
    2. Each MgConv implements a V-cycle multigrid
    3. Final linear projection to output channels

    Args:
        input_shape: Spatial dimensions (H, W)
        num_layer: Number of MgConv layers
        num_channel_u: Channels for solution representation
        num_channel_f: Channels for input/forcing representation
        num_iteration: List of (pre_smooth, post_smooth) per multigrid level
        output_dim: Number of output channels
        activation: Activation function
        padding_mode: Padding mode for convolutions
    """

    input_shape: Tuple[int, int]
    num_layer: int = 5
    num_channel_u: int = 24
    num_channel_f: int = 3
    num_iteration: List[Tuple[int, int]] = None
    output_dim: int = 1
    activation: str = "gelu"
    padding_mode: str = "CIRCULAR"

    def setup(self):
        # Default iteration structure if not provided
        if self.num_iteration is None:
            self.num_iteration_list = [[1, 1]] * 5
        else:
            self.num_iteration_list = self.num_iteration

        # Activation function
        if self.activation == "relu":
            self.act = nn.relu
        elif self.activation == "gelu":
            self.act = nn.gelu
        elif self.activation == "tanh":
            self.act = nn.tanh
        elif self.activation == "silu":
            self.act = nn.silu
        else:
            self.act = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor [B, H, W, C] or [H, W, C]
            training: Whether in training mode

        Returns:
            Output tensor [B, H, W, output_dim] or [H, W, output_dim]
        """
        # Handle unbatched input
        squeeze_batch = False
        if x.ndim == 3:
            x = x[None, ...]
            squeeze_batch = True

        u = x

        for i in range(self.num_layer):
            # Determine input channels for this layer
            if i == 0:
                in_channels = self.num_channel_f
            else:
                in_channels = self.num_channel_u

            # MgConv layer
            mg_out = MgConv(input_shape=self.input_shape, num_iteration=self.num_iteration_list, num_channel_u=self.num_channel_u, num_channel_f=in_channels, padding_mode=self.padding_mode, name=f"mgconv_{i}")(u)

            # Linear skip connection (1x1 conv)
            linear_out = nn.Conv(features=self.num_channel_u, kernel_size=(1, 1), strides=(1, 1), padding="SAME", use_bias=True, name=f"linear_{i}")(u)

            # Combine with activation
            u = self.act(mg_out + linear_out)

        # Final projection to output channels
        output = nn.Conv(features=self.output_dim, kernel_size=(1, 1), strides=(1, 1), padding="SAME", use_bias=False, name="output_proj")(u)

        if squeeze_batch:
            output = output[0]

        return output


class MgNO1D(nn.Module):
    """
    1D Multigrid Neural Operator.

    Adapted for 1D problems (e.g., time series, 1D PDEs).
    """

    input_length: int
    num_layer: int = 5
    num_channel_u: int = 24
    num_channel_f: int = 3
    num_iteration: List[Tuple[int, int]] = None
    output_dim: int = 1
    activation: str = "gelu"
    padding_mode: str = "CIRCULAR"

    def setup(self):
        if self.num_iteration is None:
            self.num_iteration_list = [[1, 1]] * 5
        else:
            self.num_iteration_list = self.num_iteration

        if self.activation == "relu":
            self.act = nn.relu
        elif self.activation == "gelu":
            self.act = nn.gelu
        elif self.activation == "tanh":
            self.act = nn.tanh
        elif self.activation == "silu":
            self.act = nn.silu
        else:
            self.act = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tensor [B, L, C] or [L, C]
        Returns:
            Output tensor [B, L, output_dim] or [L, output_dim]
        """
        squeeze_batch = False
        if x.ndim == 2:
            x = x[None, ...]
            squeeze_batch = True

        # Reshape to 2D: [B, L, C] -> [B, L, 1, C]
        x = x[:, :, None, :]

        # Use 2D MgNO with height=1
        output = MgNO(
            input_shape=(self.input_length, 1),
            num_layer=self.num_layer,
            num_channel_u=self.num_channel_u,
            num_channel_f=self.num_channel_f,
            num_iteration=self.num_iteration_list,
            output_dim=self.output_dim,
            activation=self.activation,
            padding_mode=self.padding_mode,
            name="mgno_2d",
        )(x, training)

        # Reshape back: [B, L, 1, C] -> [B, L, C]
        output = output[:, :, 0, :]

        if squeeze_batch:
            output = output[0]

        return output
