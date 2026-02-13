from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Sequence, Optional, Tuple
from flax.linen.initializers import lecun_normal, zeros
import jax


# 1 Dimensional UNet


def pad_1d(x: jnp.ndarray, pad: int, mode: str = "circular") -> jnp.ndarray:
    """Apply padding to 1D spatial dimension. x shape: (L, C)"""
    pad_mode = "wrap" if mode == "circular" else "reflect"
    return jnp.pad(x, ((pad, pad), (0, 0)), mode=pad_mode)


class UNetConv1d(nn.Module):
    """Single 1D convolution block with optional normalization and activation."""

    out_channels: int
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Optional[Callable] = None
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Manual padding for circular/reflect modes
        pad = self.kernel_size // 2
        x = pad_1d(x, pad, self.padding_mode)

        # 1D Convolution (VALID padding since we already padded)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding="VALID",
            feature_group_count=self.groups,
            use_bias=True,
            kernel_init=lecun_normal(),
            bias_init=zeros,
        )(x)

        # Normalization
        if self.norm == "batch":
            x = nn.BatchNorm(use_running_average=not self.training)(x)
        elif self.norm == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm == "group":
            x = nn.GroupNorm(num_groups=min(32, self.out_channels))(x)

        # Activation
        if self.activation is not None:
            x = self.activation(x)

        return x


class UNetConvBlock1d(nn.Module):
    """Double 1D convolution block used in UNet."""

    out_channels: Tuple[int, int]
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # First conv with activation
        x = UNetConv1d(
            out_channels=self.out_channels[0],
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        # Second conv without activation
        x = UNetConv1d(
            out_channels=self.out_channels[1],
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=None,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        return x


class UNetDownBlock1d(nn.Module):
    """1D Downsampling block: ConvBlock + AvgPool."""

    out_channels: Tuple[int, int]
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Conv block
        skip = UNetConvBlock1d(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        # Average pooling for downsampling (factor of 2)
        x_down = nn.avg_pool(skip, window_shape=(2,), strides=(2,))

        return x_down, skip


class UNetUpBlock1d(nn.Module):
    """1D Upsampling block: Upsample + Concat + ConvBlock."""

    out_channels: Tuple[int, int]
    up_mode: str = "upconv"
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        in_channels = x.shape[-1]

        # Upsample
        if self.up_mode == "upconv":
            x = nn.ConvTranspose(
                features=in_channels,
                kernel_size=(2,),
                strides=(2,),
                padding="SAME",
                use_bias=False,
                kernel_init=lecun_normal(),
            )(x)
        else:
            # Linear upsampling + 1x1 conv
            L = x.shape[-2]
            new_shape = x.shape[:-2] + (L * 2, x.shape[-1])
            x = jax.image.resize(x, shape=new_shape, method="linear")
            x = nn.Conv(
                features=in_channels,
                kernel_size=(1,),
                use_bias=False,
                kernel_init=lecun_normal(),
            )(x)

        # Handle size mismatch (crop/pad x to match skip)
        if x.shape[-2] != skip.shape[-2]:
            target_l = skip.shape[-2]
            curr_l = x.shape[-2]

            if curr_l < target_l:
                # Pad x to match skip
                pad_l = target_l - curr_l
                x = jnp.pad(x, ((0, pad_l), (0, 0)), mode="edge")

            # Crop if needed
            x = x[..., :target_l, :]

        # Concatenate skip connection
        if self.groups == 1:
            x = jnp.concatenate([x, skip], axis=-1)
        else:
            # Interleaved concatenation for grouped convolutions
            channels = x.shape[-1]
            ch_per_group = channels // self.groups
            parts = []
            for g in range(self.groups):
                start = g * ch_per_group
                end = (g + 1) * ch_per_group
                parts.append(x[..., start:end])
                parts.append(skip[..., start:end])
            x = jnp.concatenate(parts, axis=-1)

        # Conv block
        x = UNetConvBlock1d(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        return x


class UNet1D(nn.Module):
    """1D UNet architecture for sequence-to-sequence tasks."""

    in_channels: int = 1
    out_channels: int = 1
    depth: int = 4
    wf: int = 6  # width factor: base channels = 2^wf
    norm: str = "batch"
    up_mode: str = "upconv"  # 'upconv' or 'upsample'
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of 1D UNet.

        Args:
            x: Input tensor of shape (L, C) or (L,)

        Returns:
            Output tensor of shape (L, out_channels) or (L,) if out_channels=1 and input was 1D
        """
        # Handle input dimensions
        input_ndim = x.ndim

        if x.ndim == 1:
            x = x[..., jnp.newaxis]  # (L,) -> (L, 1)

        skips = []

        # Encoder path
        for i in range(self.depth):
            ch = (2**self.wf) * (2**i)

            x, skip = UNetDownBlock1d(
                out_channels=(ch, ch),
                kernel_size=3,
                norm=self.norm,
                groups=self.groups,
                activation=self.activation,
                padding_mode=self.padding_mode,
                training=self.training,
                name=f"encoder_{i}",
            )(x)

            skips.append(skip)

        # Bottleneck
        bottleneck_ch = (2**self.wf) * (2 ** (self.depth - 1))
        x = UNetConvBlock1d(
            out_channels=(bottleneck_ch, bottleneck_ch),
            kernel_size=3,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
            name="bottleneck",
        )(x)

        # Decoder path
        for i in range(self.depth):
            depth_idx = self.depth - 1 - i
            ch_in = (2**self.wf) * (2**depth_idx)
            ch_out = (2**self.wf) * (2 ** max(0, depth_idx - 1))

            x = UNetUpBlock1d(
                out_channels=(ch_in, ch_out),
                up_mode=self.up_mode,
                kernel_size=3,
                norm=self.norm,
                groups=self.groups,
                activation=self.activation,
                padding_mode=self.padding_mode,
                training=self.training,
                name=f"decoder_{i}",
            )(x, skips[-(i + 1)])

        # Final 1x1 convolution
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(1,),
            padding="SAME",
            use_bias=False,
            kernel_init=lecun_normal(),
            name="final_conv",
        )(x)

        # Match output shape to input shape
        if input_ndim == 1 and self.out_channels == 1:
            x = x[..., 0]

        return x


def pad_2d(x: jnp.ndarray, pad: int, mode: str = "circular") -> jnp.ndarray:
    """Apply padding to 2D spatial dimensions. x shape: (H, W, C)"""
    pad_mode = "wrap" if mode == "circular" else "reflect"
    return jnp.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)


class UNetConv2d(nn.Module):
    """Single convolution block with optional normalization and activation."""

    out_channels: int
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Optional[Callable] = None
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Manual padding for circular/reflect modes
        pad = self.kernel_size // 2
        x = pad_2d(x, pad, self.padding_mode)

        # Convolution (VALID padding since we already padded)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding="VALID",
            feature_group_count=self.groups,
            use_bias=True,
            kernel_init=lecun_normal(),
            bias_init=zeros,
        )(x)

        # Normalization
        if self.norm == "batch":
            x = nn.BatchNorm(use_running_average=not self.training)(x)
        elif self.norm == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm == "group":
            x = nn.GroupNorm(num_groups=min(32, self.out_channels))(x)

        # Activation
        if self.activation is not None:
            x = self.activation(x)

        return x


class UNetConvBlock2d(nn.Module):
    """Double convolution block used in UNet."""

    out_channels: Tuple[int, int]
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # First conv with activation
        x = UNetConv2d(
            out_channels=self.out_channels[0],
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        # Second conv without activation
        x = UNetConv2d(
            out_channels=self.out_channels[1],
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=None,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        return x


class UNetDownBlock2d(nn.Module):
    """Downsampling block: ConvBlock + AvgPool."""

    out_channels: Tuple[int, int]
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Conv block
        skip = UNetConvBlock2d(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        # Average pooling for downsampling
        x_down = nn.avg_pool(skip, window_shape=(2, 2), strides=(2, 2))

        return x_down, skip


class UNetUpBlock2d(nn.Module):
    """Upsampling block: Upsample + Concat + ConvBlock."""

    out_channels: Tuple[int, int]
    up_mode: str = "upconv"
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        in_channels = x.shape[-1]

        # Upsample
        if self.up_mode == "upconv":
            x = nn.ConvTranspose(
                features=in_channels,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding="SAME",
                use_bias=False,
                kernel_init=lecun_normal(),
            )(x)
        else:
            # Bilinear upsampling + 1x1 conv
            H, W = x.shape[-3], x.shape[-2]
            new_shape = x.shape[:-3] + (H * 2, W * 2, x.shape[-1])
            x = jax.image.resize(x, shape=new_shape, method="bilinear")
            x = nn.Conv(
                features=in_channels,
                kernel_size=(1, 1),
                use_bias=False,
                kernel_init=lecun_normal(),
            )(x)

        # Handle size mismatch (crop x to match skip)
        if x.shape[-3] != skip.shape[-3] or x.shape[-2] != skip.shape[-2]:
            target_h, target_w = skip.shape[-3], skip.shape[-2]
            curr_h, curr_w = x.shape[-3], x.shape[-2]

            if curr_h < target_h or curr_w < target_w:
                # Pad x to match skip
                pad_h = target_h - curr_h
                pad_w = target_w - curr_w
                x = jnp.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

            # Crop if needed
            x = x[..., :target_h, :target_w, :]

        # Concatenate skip connection
        if self.groups == 1:
            x = jnp.concatenate([x, skip], axis=-1)
        else:
            # Interleaved concatenation for grouped convolutions
            channels = x.shape[-1]
            ch_per_group = channels // self.groups
            parts = []
            for g in range(self.groups):
                start = g * ch_per_group
                end = (g + 1) * ch_per_group
                parts.append(x[..., start:end])
                parts.append(skip[..., start:end])
            x = jnp.concatenate(parts, axis=-1)

        # Conv block
        x = UNetConvBlock2d(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        return x


class UNet2D(nn.Module):
    """2D UNet architecture."""

    in_channels: int = 1
    out_channels: int = 1
    depth: int = 4
    wf: int = 6
    norm: str = "batch"
    up_mode: str = "upconv"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of UNet.

        Args:
            x: Input tensor of shape (H, W, C) or (H, W)

        Returns:
            Output tensor of shape (H, W, out_channels)
        """
        # Handle input dimensions
        input_ndim = x.ndim

        if x.ndim == 2:
            x = x[..., jnp.newaxis]  # (H, W) -> (H, W, 1)

        skips = []

        # Encoder path
        for i in range(self.depth):
            ch = (2**self.wf) * (2**i)

            x, skip = UNetDownBlock2d(
                out_channels=(ch, ch),
                kernel_size=3,
                norm=self.norm,
                groups=self.groups,
                activation=self.activation,
                padding_mode=self.padding_mode,
                training=self.training,
                name=f"encoder_{i}",
            )(x)

            skips.append(skip)

        # Bottleneck
        bottleneck_ch = (2**self.wf) * (2 ** (self.depth - 1))
        x = UNetConvBlock2d(
            out_channels=(bottleneck_ch, bottleneck_ch),
            kernel_size=3,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
            name="bottleneck",
        )(x)

        # Decoder path
        for i in range(self.depth):
            depth_idx = self.depth - 1 - i
            ch_in = (2**self.wf) * (2**depth_idx)
            ch_out = (2**self.wf) * (2 ** max(0, depth_idx - 1))

            x = UNetUpBlock2d(
                out_channels=(ch_in, ch_out),
                up_mode=self.up_mode,
                kernel_size=3,
                norm=self.norm,
                groups=self.groups,
                activation=self.activation,
                padding_mode=self.padding_mode,
                training=self.training,
                name=f"decoder_{i}",
            )(x, skips[-(i + 1)])

        # Final 1x1 convolution
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=lecun_normal(),
            name="final_conv",
        )(x)

        # Match output shape to input shape
        if input_ndim == 2 and self.out_channels == 1:
            x = x[..., 0]

        return x


def circular_pad_3d(x: jnp.ndarray, pad: int) -> jnp.ndarray:
    """Apply circular (wrap) padding to 3D spatial dimensions."""
    # x shape: (D, H, W, C)
    return jnp.pad(x, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), mode="wrap")


def reflect_pad_3d(x: jnp.ndarray, pad: int) -> jnp.ndarray:
    """Apply reflect padding to 3D spatial dimensions."""
    return jnp.pad(x, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), mode="reflect")


def get_pad_fn(mode: str):
    """Get padding function from mode string."""
    if mode == "circular":
        return circular_pad_3d
    elif mode == "reflect":
        return reflect_pad_3d
    else:
        return reflect_pad_3d


def avg_pool_3d(x: jnp.ndarray, window_shape: Tuple[int, int, int] = (2, 2, 2)) -> jnp.ndarray:
    """3D average pooling."""
    # x shape: (D, H, W, C)
    D, H, W, C = x.shape
    new_D, new_H, new_W = D // window_shape[0], H // window_shape[1], W // window_shape[2]

    # Reshape and take mean
    x = x.reshape(new_D, window_shape[0], new_H, window_shape[1], new_W, window_shape[2], C)
    x = x.mean(axis=(1, 3, 5))
    return x


class UNetConv3d(nn.Module):
    """Single 3D convolution block with optional normalization and activation."""

    out_channels: int
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Optional[Callable] = None
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Apply padding manually
        pad = self.kernel_size // 2
        pad_fn = get_pad_fn(self.padding_mode)
        x = pad_fn(x, pad)

        # 3D Convolution
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size, self.kernel_size),
            padding="VALID",
            feature_group_count=self.groups,
            use_bias=True,
            kernel_init=lecun_normal(),
            bias_init=zeros,
        )(x)

        # Normalization
        if self.norm == "batch":
            x = nn.BatchNorm(use_running_average=not self.training)(x)
        elif self.norm == "layer":
            x = nn.LayerNorm()(x)
        elif self.norm == "group":
            x = nn.GroupNorm(num_groups=min(32, self.out_channels))(x)

        # Activation
        if self.activation is not None:
            x = self.activation(x)

        return x


class UNetConvBlock3d(nn.Module):
    """Double 3D convolution block."""

    out_channels: Sequence[int]
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = UNetConv3d(
            out_channels=self.out_channels[0],
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        x = UNetConv3d(
            out_channels=self.out_channels[1],
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=None,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        return x


class UNetDownBlock3d(nn.Module):
    """3D Downsampling block."""

    out_channels: Sequence[int]
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        skip = UNetConvBlock3d(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        # 3D Average pooling
        x_down = avg_pool_3d(skip, window_shape=(2, 2, 2))

        return x_down, skip


class UNetUpBlock3d(nn.Module):
    """3D Upsampling block."""

    out_channels: Sequence[int]
    up_mode: str = "upconv"
    kernel_size: int = 3
    norm: str = "batch"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        in_channels = x.shape[-1]

        if self.up_mode == "upconv":
            # 3D Transposed convolution
            x = nn.ConvTranspose(
                features=in_channels,
                kernel_size=(2, 2, 2),
                strides=(2, 2, 2),
                padding="SAME",
                use_bias=False,
                kernel_init=lecun_normal(),
            )(x)
        else:
            # Trilinear upsampling + 1x1x1 conv
            D, H, W = x.shape[-4], x.shape[-3], x.shape[-2]
            x = jax.image.resize(x, shape=(*x.shape[:-4], D * 2, H * 2, W * 2, x.shape[-1]), method="trilinear")
            x = nn.Conv(
                features=in_channels,
                kernel_size=(1, 1, 1),
                use_bias=False,
                kernel_init=lecun_normal(),
            )(x)

        # Handle size mismatch
        target_shape = skip.shape[:-1]
        if x.shape[:-1] != target_shape:
            x = x[..., : target_shape[-3], : target_shape[-2], : target_shape[-1], :]

        # Concatenate
        x = jnp.concatenate([x, skip], axis=-1)

        # Conv block
        x = UNetConvBlock3d(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
        )(x)

        return x


class UNet3D(nn.Module):
    """3D UNet architecture."""

    in_channels: int = 1
    out_channels: int = 2
    depth: int = 4
    wf: int = 6
    norm: str = "batch"
    up_mode: str = "upconv"
    groups: int = 1
    activation: Callable = nn.celu
    padding_mode: str = "circular"
    training: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of 3D UNet.

        Args:
            x: Input tensor of shape (D, H, W, C)

        Returns:
            Output tensor of shape (D, H, W, out_channels)
        """
        input_ndim = x.ndim
        if x.ndim == 3:
            # (D, H, W) -> (D, H, W, 1)
            x = x[..., jnp.newaxis]

        skips = []

        # Encoder
        for i in range(self.depth):
            ch = (2**self.wf) * (2**i)
            x, skip = UNetDownBlock3d(
                out_channels=[ch, ch],
                kernel_size=3,
                norm=self.norm,
                groups=self.groups,
                activation=self.activation,
                padding_mode=self.padding_mode,
                training=self.training,
                name=f"encoder_{i}",
            )(x)
            skips.append(skip)

        # Bottleneck
        bottleneck_ch = (2**self.wf) * (2 ** (self.depth - 1))
        x = UNetConvBlock3d(
            out_channels=[bottleneck_ch, bottleneck_ch],
            kernel_size=3,
            norm=self.norm,
            groups=self.groups,
            activation=self.activation,
            padding_mode=self.padding_mode,
            training=self.training,
            name="bottleneck",
        )(x)

        # Decoder
        for i in range(self.depth):
            depth_idx = self.depth - 1 - i
            ch_in = (2**self.wf) * (2**depth_idx)
            ch_out = (2**self.wf) * (2 ** max(0, depth_idx - 1))

            x = UNetUpBlock3d(
                out_channels=[ch_in, ch_out],
                up_mode=self.up_mode,
                kernel_size=3,
                norm=self.norm,
                groups=self.groups,
                activation=self.activation,
                padding_mode=self.padding_mode,
                training=self.training,
                name=f"decoder_{i}",
            )(x, skips[-(i + 1)])

        # Final convolution
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1, 1),
            padding="SAME",
            use_bias=False,
            kernel_init=lecun_normal(),
            name="final_conv",
        )(x)

        if input_ndim == 3 and self.out_channels == 1:
            x = x[..., 0]

        return x
