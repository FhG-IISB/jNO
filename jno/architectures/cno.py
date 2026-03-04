"""
CNO2d JAX/Equinox Implementation (NHWC Format)

Convolutional Neural Operator for learning mappings between function spaces.
Based on the implementation from ETH Zurich course "AI in the Sciences and Engineering."

Reference:
    https://github.com/bogdanraonic3/AI_Science_Engineering

Input/Output format: (N, H, W, C) - NHWC
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple, List
from .common import BatchNorm, Conv2d


# ---------------------------------------------------------------------------
# Free helper function
# ---------------------------------------------------------------------------


def bicubic_resize(x: jnp.ndarray, target_size: Tuple[int, int]) -> jnp.ndarray:
    """
    Bicubic interpolation for 2D images in NHWC format.

    Args:
        x: Input array of shape (batch, height, width, channels) - NHWC format
        target_size: Target (height, width)

    Returns:
        Resized array in NHWC format
    """
    batch, h, w, channels = x.shape
    target_h, target_w = target_size

    x = jax.image.resize(
        x,
        shape=(batch, target_h, target_w, channels),
        method="bicubic",
        antialias=True,
    )
    return x


# ---------------------------------------------------------------------------
# CNO building blocks
# ---------------------------------------------------------------------------


class CNOLReLu(eqx.Module):
    """
    CNO LReLU activation with up/downsampling.
    Upsamples to 2x, applies LeakyReLU, then downsamples to output size.
    """

    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size: int, out_size: int, *, key: jax.Array):
        self.in_size = in_size
        self.out_size = out_size

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = bicubic_resize(x, (2 * self.in_size, 2 * self.in_size))
        x = jax.nn.leaky_relu(x)
        x = bicubic_resize(x, (self.out_size, self.out_size))
        return x


class CNOBlock(eqx.Module):
    """
    CNO Block: Conv2d -> BatchNorm (optional) -> CNO_LReLu Activation
    """

    conv: Conv2d
    bn: Optional[BatchNorm]
    act: CNOLReLu
    use_bn: bool = eqx.field(static=True)

    def __init__(self, in_channels: int, out_channels: int, in_size: int, out_size: int, use_bn: bool = True, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.conv = Conv2d(in_channels, out_channels, kernel_size=3, padding="SAME", key=k1)
        self.use_bn = use_bn
        self.bn = BatchNorm(out_channels) if use_bn else None
        self.act = CNOLReLu(in_size=in_size, out_size=out_size, key=k2)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.conv(x)
        if self.use_bn and self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        return x


class LiftProjectBlock(eqx.Module):
    """
    Lift/Project Block for embedding transformations.
    """

    inter_block: CNOBlock
    conv: Conv2d

    def __init__(self, in_channels: int, out_channels: int, size: int, latent_dim: int = 64, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        self.inter_block = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            use_bn=False,
            key=k1,
        )
        self.conv = Conv2d(latent_dim, out_channels, kernel_size=3, padding="SAME", key=k2)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.inter_block(x)
        x = self.conv(x)
        return x


class ResidualBlock(eqx.Module):
    """
    Residual Block: Conv -> BN -> Activation -> Conv -> BN -> Skip Connection
    """

    conv1: Conv2d
    conv2: Conv2d
    bn1: Optional[BatchNorm]
    bn2: Optional[BatchNorm]
    act: CNOLReLu
    use_bn: bool = eqx.field(static=True)

    def __init__(self, channels: int, size: int, use_bn: bool = True, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = Conv2d(channels, channels, kernel_size=3, padding="SAME", key=k1)
        self.conv2 = Conv2d(channels, channels, kernel_size=3, padding="SAME", key=k2)
        self.use_bn = use_bn
        self.bn1 = BatchNorm(channels) if use_bn else None
        self.bn2 = BatchNorm(channels) if use_bn else None
        self.act = CNOLReLu(in_size=size, out_size=size, key=k3)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        out = self.conv1(x)
        if self.use_bn and self.bn1 is not None:
            out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        if self.use_bn and self.bn2 is not None:
            out = self.bn2(out)
        return x + out


class ResNet(eqx.Module):
    """
    ResNet: Stack of Residual Blocks
    """

    blocks: List[ResidualBlock]

    def __init__(self, channels: int, size: int, num_blocks: int, use_bn: bool = True, *, key: jax.Array):
        keys = jax.random.split(key, num_blocks)
        self.blocks = [ResidualBlock(channels=channels, size=size, use_bn=use_bn, key=keys[i]) for i in range(num_blocks)]

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CNO2D(eqx.Module):
    """
    Convolutional Neural Operator 2D (CNO2d)

    A U-Net style architecture for learning operators between function spaces.
    Uses convolutions with bicubic interpolation for resolution-invariant
    learning.

    Architecture::

        Input -> Lift -> [Encoder + ResNet] x N -> Bottleneck -> [Decoder + Skip] x N -> Project -> Output

    Key Features:
        - Resolution-invariant through bicubic interpolation
        - Skip connections for multi-scale information flow
        - Batch normalization for training stability
        - Configurable depth and width

    Input format: (N, H, W, C) - NHWC
    Output format: (N, H, W, C) - NHWC

    Attributes:
        in_dim: Number of input channels
        out_dim: Number of output channels
        size: Input and output spatial size (assumes square)
        N_layers: Number of encoder/decoder blocks
        N_res: Number of residual blocks per level (except neck)
        N_res_neck: Number of residual blocks in the bottleneck
        channel_multiplier: Base channel multiplier for feature evolution
        use_bn: Whether to use batch normalization

    Example:
        >>> model = CNO2D(
        ...     in_dim=3,
        ...     out_dim=1,
        ...     size=64,
        ...     N_layers=3,
        ...     channel_multiplier=16,
        ...     key=jax.random.PRNGKey(0),
        ... )

    Reference:
        Raonic et al., "Convolutional Neural Operators for robust and accurate learning of PDEs"
        https://github.com/bogdanraonic3/AI_Science_Engineering
    """

    # --- static configuration ---
    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)
    size: int = eqx.field(static=True)
    N_layers: int = eqx.field(static=True)
    N_res: int = eqx.field(static=True)
    N_res_neck: int = eqx.field(static=True)
    channel_multiplier: int = eqx.field(static=True)
    use_bn: bool = eqx.field(static=True)

    encoder_features: List[int] = eqx.field(static=True)
    decoder_features_in: List[int] = eqx.field(static=True)
    decoder_features_out: List[int] = eqx.field(static=True)
    encoder_sizes: List[int] = eqx.field(static=True)
    decoder_sizes: List[int] = eqx.field(static=True)

    # --- learnable submodules ---
    lift: LiftProjectBlock
    project: LiftProjectBlock
    encoder: List[CNOBlock]
    ED_expansion: List[CNOBlock]
    decoder: List[CNOBlock]
    res_nets: List[ResNet]
    res_net_neck: ResNet

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        size: int,
        N_layers: int,
        N_res: int = 4,
        N_res_neck: int = 4,
        channel_multiplier: int = 16,
        use_bn: bool = True,
        *,
        key: jax.Array,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.size = size
        self.N_layers = N_layers
        self.N_res = N_res
        self.N_res_neck = N_res_neck
        self.channel_multiplier = channel_multiplier
        self.use_bn = use_bn

        lift_dim = channel_multiplier // 2

        # ----- feature / size bookkeeping (same logic as setup()) -----
        encoder_features: List[int] = [lift_dim]
        for i in range(N_layers):
            encoder_features.append(2**i * channel_multiplier)

        decoder_features_in_list = list(encoder_features[1:])
        decoder_features_in_list.reverse()
        decoder_features_out = list(encoder_features[:-1])
        decoder_features_out.reverse()

        decoder_features_in = decoder_features_in_list.copy()
        for i in range(1, N_layers):
            decoder_features_in[i] = 2 * decoder_features_in[i]

        encoder_sizes: List[int] = []
        decoder_sizes: List[int] = []
        for i in range(N_layers + 1):
            encoder_sizes.append(size // (2**i))
            decoder_sizes.append(size // (2 ** (N_layers - i)))

        # store bookkeeping as static fields
        self.encoder_features = encoder_features
        self.decoder_features_in = decoder_features_in
        self.decoder_features_out = decoder_features_out
        self.encoder_sizes = encoder_sizes
        self.decoder_sizes = decoder_sizes

        # ----- allocate submodule keys -----
        # We need keys for: lift, project, N_layers encoders, (N_layers+1) ED_expansions,
        # N_layers decoders, N_layers res_nets, 1 res_net_neck
        num_keys = 2 + N_layers + (N_layers + 1) + N_layers + N_layers + 1
        keys = jax.random.split(key, num_keys)
        idx = 0

        # Lift
        self.lift = LiftProjectBlock(
            in_channels=in_dim,
            out_channels=encoder_features[0],
            size=size,
            key=keys[idx],
        )
        idx += 1

        # Project (input channels = first encoder features + last decoder output)
        self.project = LiftProjectBlock(
            in_channels=encoder_features[0] + decoder_features_out[-1],
            out_channels=out_dim,
            size=size,
            key=keys[idx],
        )
        idx += 1

        # Encoder blocks
        self.encoder = []
        for i in range(N_layers):
            self.encoder.append(
                CNOBlock(
                    in_channels=encoder_features[i],
                    out_channels=encoder_features[i + 1],
                    in_size=encoder_sizes[i],
                    out_size=encoder_sizes[i + 1],
                    use_bn=use_bn,
                    key=keys[idx],
                )
            )
            idx += 1

        # ED expansion blocks
        self.ED_expansion = []
        for i in range(N_layers + 1):
            self.ED_expansion.append(
                CNOBlock(
                    in_channels=encoder_features[i],
                    out_channels=encoder_features[i],
                    in_size=encoder_sizes[i],
                    out_size=decoder_sizes[N_layers - i],
                    use_bn=use_bn,
                    key=keys[idx],
                )
            )
            idx += 1

        # Decoder blocks
        self.decoder = []
        for i in range(N_layers):
            self.decoder.append(
                CNOBlock(
                    in_channels=decoder_features_in[i],
                    out_channels=decoder_features_out[i],
                    in_size=decoder_sizes[i],
                    out_size=decoder_sizes[i + 1],
                    use_bn=use_bn,
                    key=keys[idx],
                )
            )
            idx += 1

        # ResNet blocks (one per encoder level, except neck)
        self.res_nets = []
        for l in range(N_layers):
            self.res_nets.append(
                ResNet(
                    channels=encoder_features[l],
                    size=encoder_sizes[l],
                    num_blocks=N_res,
                    use_bn=use_bn,
                    key=keys[idx],
                )
            )
            idx += 1

        # Bottleneck ResNet
        self.res_net_neck = ResNet(
            channels=encoder_features[N_layers],
            size=encoder_sizes[N_layers],
            num_blocks=N_res_neck,
            use_bn=use_bn,
            key=keys[idx],
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        """
        Forward pass of CNO2d.

        Args:
            x: Input tensor of shape (batch, height, width, in_dim) - NHWC format

        Returns:
            Output tensor of shape (batch, height, width, out_dim) - NHWC format
        """
        if x.ndim == 3:
            x = x[None, ...]

        # Lift
        x = self.lift(x)
        skip: List[jnp.ndarray] = []

        # Encoder
        for i in range(self.N_layers):
            y = self.res_nets[i](x)
            skip.append(y)
            x = self.encoder[i](x)

        # Bottleneck
        x = self.res_net_neck(x)

        # Decoder
        for i in range(self.N_layers):
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)
            else:
                x = jnp.concatenate(
                    [x, self.ED_expansion[self.N_layers - i](skip[-i])],
                    axis=-1,
                )
            x = self.decoder[i](x)

        # Final skip and project
        x = jnp.concatenate(
            [x, self.ED_expansion[0](skip[0])],
            axis=-1,
        )
        x = self.project(x)

        return x
