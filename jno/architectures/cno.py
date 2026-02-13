"""
CNO2d JAX/Flax Implementation (NHWC Format)

Convolutional Neural Operator for learning mappings between function spaces.
Based on the implementation from ETH Zurich course "AI in the Sciences and Engineering."

Reference:
    https://github.com/bogdanraonic3/AI_Science_Engineering

Input/Output format: (N, H, W, C) - NHWC
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Optional, Tuple, List


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

    x = jax.image.resize(x, shape=(batch, target_h, target_w, channels), method="bicubic", antialias=True)

    return x


class CNOLReLu(nn.Module):
    """
    CNO LReLU activation with up/downsampling.
    Upsamples to 2x, applies LeakyReLU, then downsamples to output size.
    """

    in_size: int
    out_size: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = bicubic_resize(x, (2 * self.in_size, 2 * self.in_size))
        x = nn.leaky_relu(x)
        x = bicubic_resize(x, (self.out_size, self.out_size))
        return x


class CNOBlock(nn.Module):
    """
    CNO Block: Conv2d -> BatchNorm (optional) -> CNO_LReLu Activation
    """

    in_channels: int
    out_channels: int
    in_size: int
    out_size: int
    use_bn: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME", name="convolution")(x)

        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train, axis=-1, name="batch_norm")(x)

        x = CNOLReLu(in_size=self.in_size, out_size=self.out_size, name="act")(x)

        return x


class LiftProjectBlock(nn.Module):
    """
    Lift/Project Block for embedding transformations.
    """

    in_channels: int
    out_channels: int
    size: int
    latent_dim: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        x = CNOBlock(in_channels=self.in_channels, out_channels=self.latent_dim, in_size=self.size, out_size=self.size, use_bn=False, name="inter_CNOBlock")(x, train=train)

        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME", name="convolution")(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual Block: Conv -> BN -> Activation -> Conv -> BN -> Skip Connection
    """

    channels: int
    size: int
    use_bn: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        residual = x

        out = nn.Conv(features=self.channels, kernel_size=(3, 3), padding="SAME", name="convolution1")(x)

        if self.use_bn:
            out = nn.BatchNorm(use_running_average=not train, axis=-1, name="batch_norm1")(out)

        out = CNOLReLu(in_size=self.size, out_size=self.size, name="act")(out)

        out = nn.Conv(features=self.channels, kernel_size=(3, 3), padding="SAME", name="convolution2")(out)

        if self.use_bn:
            out = nn.BatchNorm(use_running_average=not train, axis=-1, name="batch_norm2")(out)

        return x + out


class ResNet(nn.Module):
    """
    ResNet: Stack of Residual Blocks
    """

    channels: int
    size: int
    num_blocks: int
    use_bn: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for i in range(self.num_blocks):
            x = ResidualBlock(channels=self.channels, size=self.size, use_bn=self.use_bn, name=f"res_block_{i}")(x, train=train)
        return x


class CNO2D(nn.Module):
    """
    Convolutional Neural Operator 2D (CNO2d)

    A U-Net style architecture for learning operators between function spaces.
    Uses convolutions with bicubic interpolation for resolution-invariant
    learning.

    Architecture::

        Input → Lift → [Encoder + ResNet] × N → Bottleneck → [Decoder + Skip] × N → Project → Output

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
        training: Training mode flag (affects batch norm behavior)

    Example:
        >>> model = CNO2D(
        ...     in_dim=3,
        ...     out_dim=1,
        ...     size=64,
        ...     N_layers=3,
        ...     channel_multiplier=16,
        ... )

    Reference:
        Raonić et al., "Convolutional Neural Operators for robust and accurate learning of PDEs"
        https://github.com/bogdanraonic3/AI_Science_Engineering
    """

    in_dim: int
    out_dim: int
    size: int
    N_layers: int
    N_res: int = 4
    N_res_neck: int = 4
    channel_multiplier: int = 16
    use_bn: bool = True
    training: bool = True

    def setup(self):
        """Initialize architecture parameters and layers."""

        self.lift_dim = self.channel_multiplier // 2

        # Encoder features evolution
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2**i * self.channel_multiplier)

        # Decoder features evolution
        decoder_features_in = list(self.encoder_features[1:])
        decoder_features_in.reverse()
        self.decoder_features_out = list(self.encoder_features[:-1])
        self.decoder_features_out.reverse()

        self.decoder_features_in = decoder_features_in.copy()
        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]

        # Spatial sizes evolution
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(self.size // (2**i))
            self.decoder_sizes.append(self.size // (2 ** (self.N_layers - i)))

        # Lift and Project blocks
        self.lift = LiftProjectBlock(in_channels=self.in_dim, out_channels=self.encoder_features[0], size=self.size, name="lift")

        self.project = LiftProjectBlock(in_channels=self.encoder_features[0] + self.decoder_features_out[-1], out_channels=self.out_dim, size=self.size, name="project")

        # Encoder blocks
        self.encoder = [
            CNOBlock(in_channels=self.encoder_features[i], out_channels=self.encoder_features[i + 1], in_size=self.encoder_sizes[i], out_size=self.encoder_sizes[i + 1], use_bn=self.use_bn, name=f"encoder_{i}") for i in range(self.N_layers)
        ]

        # ED expansion blocks
        self.ED_expansion = [
            CNOBlock(in_channels=self.encoder_features[i], out_channels=self.encoder_features[i], in_size=self.encoder_sizes[i], out_size=self.decoder_sizes[self.N_layers - i], use_bn=self.use_bn, name=f"ED_expansion_{i}")
            for i in range(self.N_layers + 1)
        ]

        # Decoder blocks
        self.decoder = [
            CNOBlock(in_channels=self.decoder_features_in[i], out_channels=self.decoder_features_out[i], in_size=self.decoder_sizes[i], out_size=self.decoder_sizes[i + 1], use_bn=self.use_bn, name=f"decoder_{i}") for i in range(self.N_layers)
        ]

        # ResNet blocks
        self.res_nets = [ResNet(channels=self.encoder_features[l], size=self.encoder_sizes[l], num_blocks=self.N_res, use_bn=self.use_bn, name=f"res_net_{l}") for l in range(self.N_layers)]

        self.res_net_neck = ResNet(channels=self.encoder_features[self.N_layers], size=self.encoder_sizes[self.N_layers], num_blocks=self.N_res_neck, use_bn=self.use_bn, name="res_net_neck")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of CNO2d.

        Args:
            x: Input tensor of shape (batch, height, width, in_dim) - NHWC format

        Returns:
            Output tensor of shape (batch, height, width, out_dim) - NHWC format
        """
        train = self.training

        if x.ndim == 3:
            x = x[None, ...]

        # Lift
        x = self.lift(x, train=train)
        skip = []

        # Encoder
        for i in range(self.N_layers):
            y = self.res_nets[i](x, train=train)
            skip.append(y)
            x = self.encoder[i](x, train=train)

        # Bottleneck
        x = self.res_net_neck(x, train=train)

        # Decoder
        for i in range(self.N_layers):
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x, train=train)
            else:
                x = jnp.concatenate([x, self.ED_expansion[self.N_layers - i](skip[-i], train=train)], axis=-1)
            x = self.decoder[i](x, train=train)

        # Final skip and project
        x = jnp.concatenate([x, self.ED_expansion[0](skip[0], train=train)], axis=-1)
        x = self.project(x, train=train)

        return x
