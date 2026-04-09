import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Sequence, Optional, Tuple
from .common import BatchNorm, Conv2d as _Conv2dBase, ConvTranspose2d


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


# ============================================================
# Helper functions and conv wrappers
# ============================================================


class Conv1dNHWC(eqx.Module):
    """1D convolution in (N, L, C) / (L, C) format."""

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    padding: str = eqx.field(static=True)
    strides: tuple = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size, strides=(1,), padding="SAME", use_bias=True, groups=1, *, key):
        fan_in = (in_ch // groups) * kernel_size
        std = 1.0 / jnp.sqrt(jnp.array(fan_in, dtype=_default_float_dtype()))
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (kernel_size, in_ch, out_ch)) * std
        self.bias = jnp.zeros(out_ch) if use_bias else None
        self.padding = padding
        self.strides = strides
        self.groups = groups

    def __call__(self, x, **kwargs):
        was_2d = x.ndim == 2
        if was_2d:
            x = x[None]
        y = jax.lax.conv_general_dilated(x, self.weight, self.strides, self.padding, dimension_numbers=("NWC", "WIO", "NWC"), feature_group_count=self.groups)
        if self.bias is not None:
            y = y + self.bias
        if was_2d:
            y = y[0]
        return y


class ConvTranspose1d(eqx.Module):
    """1D transposed conv in (N, L, C) / (L, C) format."""

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    strides: tuple = eqx.field(static=True)
    padding: str = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size, strides=(2,), padding="SAME", use_bias=False, *, key):
        fan_in = in_ch * kernel_size
        std = 1.0 / jnp.sqrt(jnp.array(fan_in, dtype=_default_float_dtype()))
        self.weight = jax.random.normal(key, (kernel_size, out_ch, in_ch)) * std
        self.bias = jnp.zeros(out_ch) if use_bias else None
        self.strides = strides
        self.padding = padding

    def __call__(self, x, **kwargs):
        was_2d = x.ndim == 2
        if was_2d:
            x = x[None]
        y = jax.lax.conv_transpose(x, self.weight, self.strides, self.padding, dimension_numbers=("NWC", "WIO", "NWC"))
        if self.bias is not None:
            y = y + self.bias
        if was_2d:
            y = y[0]
        return y


class Conv2dNHWC(eqx.Module):
    """2D convolution in (N, H, W, C) / (H, W, C) format."""

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    padding: str = eqx.field(static=True)
    strides: tuple = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size, strides=(1, 1), padding="SAME", use_bias=True, groups=1, *, key):
        kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        fan_in = (in_ch // groups) * kh * kw
        std = 1.0 / jnp.sqrt(jnp.array(fan_in, dtype=_default_float_dtype()))
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (kh, kw, in_ch, out_ch)) * std
        self.bias = jnp.zeros(out_ch) if use_bias else None
        self.padding = padding
        self.strides = strides
        self.groups = groups

    def __call__(self, x, **kwargs):
        was_3d = x.ndim == 3
        if was_3d:
            x = x[None]
        y = jax.lax.conv_general_dilated(x, self.weight, self.strides, self.padding, dimension_numbers=("NHWC", "HWIO", "NHWC"), feature_group_count=self.groups)
        if self.bias is not None:
            y = y + self.bias
        if was_3d:
            y = y[0]
        return y


class Conv3dNHWC(eqx.Module):
    """3D convolution in (N, D, H, W, C) / (D, H, W, C) format."""

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    padding: str = eqx.field(static=True)
    strides: tuple = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size, strides=(1, 1, 1), padding="SAME", use_bias=True, groups=1, *, key):
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        fan_in = (in_ch // groups) * ks**3
        std = 1.0 / jnp.sqrt(jnp.array(fan_in, dtype=_default_float_dtype()))
        self.weight = jax.random.normal(key, (ks, ks, ks, in_ch, out_ch)) * std
        self.bias = jnp.zeros(out_ch) if use_bias else None
        self.padding = padding
        self.strides = strides
        self.groups = groups

    def __call__(self, x, **kwargs):
        was_4d = x.ndim == 4
        if was_4d:
            x = x[None]
        y = jax.lax.conv_general_dilated(x, self.weight, self.strides, self.padding, dimension_numbers=("NDHWC", "DHWIO", "NDHWC"), feature_group_count=self.groups)
        if self.bias is not None:
            y = y + self.bias
        if was_4d:
            y = y[0]
        return y


class ConvTranspose3d(eqx.Module):
    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    strides: tuple = eqx.field(static=True)
    padding: str = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size, strides=(2, 2, 2), padding="SAME", use_bias=False, *, key):
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        fan_in = in_ch * ks**3
        std = 1.0 / jnp.sqrt(jnp.array(fan_in, dtype=_default_float_dtype()))
        self.weight = jax.random.normal(key, (ks, ks, ks, out_ch, in_ch)) * std
        self.bias = jnp.zeros(out_ch) if use_bias else None
        self.strides = strides
        self.padding = padding

    def __call__(self, x, **kwargs):
        was_4d = x.ndim == 4
        if was_4d:
            x = x[None]
        y = jax.lax.conv_transpose(x, self.weight, self.strides, self.padding, dimension_numbers=("NDHWC", "DHWIO", "NDHWC"))
        if self.bias is not None:
            y = y + self.bias
        if was_4d:
            y = y[0]
        return y


# ============================================================
# Padding helpers
# ============================================================


def pad_1d(x, pad, mode="circular"):
    pad_mode = "wrap" if mode == "circular" else "reflect"
    return jnp.pad(x, ((pad, pad), (0, 0)), mode=pad_mode)


def pad_2d(x, pad, mode="circular"):
    pad_mode = "wrap" if mode == "circular" else "reflect"
    return jnp.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)


def circular_pad_3d(x, pad):
    return jnp.pad(x, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), mode="wrap")


def reflect_pad_3d(x, pad):
    return jnp.pad(x, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), mode="reflect")


def get_pad_fn(mode):
    if mode == "circular":
        return circular_pad_3d
    return reflect_pad_3d


def avg_pool_1d(x, window=2, stride=2):
    L, C = x.shape[-2], x.shape[-1]
    nL = L // stride
    was_2d = x.ndim == 2
    if was_2d:
        x = x[None]
    x = x[:, : nL * stride, :].reshape(x.shape[0], nL, stride, C).mean(axis=2)
    if was_2d:
        x = x[0]
    return x


def avg_pool_2d(x, window=2, stride=2):
    was_3d = x.ndim == 3
    if was_3d:
        x = x[None]
    N, H, W, C = x.shape
    nH, nW = H // stride, W // stride
    x = x[:, : nH * stride, : nW * stride, :].reshape(N, nH, stride, nW, stride, C).mean(axis=(2, 4))
    if was_3d:
        x = x[0]
    return x


def avg_pool_3d(x, window_shape=(2, 2, 2)):
    D, H, W, C = x.shape
    nD = D // window_shape[0]
    nH = H // window_shape[1]
    nW = W // window_shape[2]
    x = x.reshape(nD, window_shape[0], nH, window_shape[1], nW, window_shape[2], C)
    return x.mean(axis=(1, 3, 5))


# ============================================================
# 1D UNet
# ============================================================


class VmapLayerNorm(eqx.Module):
    """LayerNorm vmapped over spatial dimensions."""

    norm: eqx.nn.LayerNorm
    spatial_ndim: int = eqx.field(static=True)

    def __init__(self, channels, spatial_ndim):
        self.norm = eqx.nn.LayerNorm(channels)
        self.spatial_ndim = spatial_ndim

    def __call__(self, x, **kwargs):
        fn = self.norm
        for _ in range(self.spatial_ndim):
            fn = jax.vmap(fn)
        return fn(x)


def _make_norm(norm_type, channels, spatial_ndim=1):
    if norm_type == "batch":
        return BatchNorm(channels)
    elif norm_type == "layer":
        return VmapLayerNorm(channels, spatial_ndim)
    elif norm_type == "group":
        return eqx.nn.GroupNorm(min(32, channels), channels)
    return None


class UNetConv1d(eqx.Module):
    conv: Conv1dNHWC
    norm_layer: Optional[eqx.Module]
    activation: Optional[Callable] = eqx.field(static=True)
    pad: int = eqx.field(static=True)
    padding_mode: str = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size=3, norm="batch", groups=1, activation=None, padding_mode="circular", *, key):
        self.conv = Conv1dNHWC(in_ch, out_ch, kernel_size, padding="VALID", groups=groups, key=key)
        self.norm_layer = _make_norm(norm, out_ch, spatial_ndim=1)
        self.activation = activation
        self.pad = kernel_size // 2
        self.padding_mode = padding_mode

    def __call__(self, x, **kwargs):
        x = pad_1d(x, self.pad, self.padding_mode)
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNetConvBlock1d(eqx.Module):
    conv1: UNetConv1d
    conv2: UNetConv1d

    def __init__(self, in_ch, out_channels, kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = UNetConv1d(in_ch, out_channels[0], kernel_size, norm, groups, activation, padding_mode, key=k1)
        self.conv2 = UNetConv1d(out_channels[0], out_channels[1], kernel_size, norm, groups, None, padding_mode, key=k2)

    def __call__(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNetDownBlock1d(eqx.Module):
    conv_block: UNetConvBlock1d

    def __init__(self, in_ch, out_channels, kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        self.conv_block = UNetConvBlock1d(in_ch, out_channels, kernel_size, norm, groups, activation, padding_mode, key=key)

    def __call__(self, x, **kwargs):
        skip = self.conv_block(x)
        x_down = avg_pool_1d(skip)
        return x_down, skip


class UNetUpBlock1d(eqx.Module):
    upsample: eqx.Module
    conv_block: UNetConvBlock1d
    up_mode: str = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    def __init__(self, in_ch, skip_ch, out_channels, up_mode="upconv", kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        k1, k2 = jax.random.split(key)
        self.up_mode = up_mode
        self.groups = groups
        if up_mode == "upconv":
            self.upsample = ConvTranspose1d(in_ch, in_ch, 2, strides=(2,), key=k1)
        else:
            self.upsample = Conv1dNHWC(in_ch, in_ch, 1, padding="SAME", key=k1)
        concat_ch = in_ch + skip_ch if groups == 1 else in_ch + skip_ch
        self.conv_block = UNetConvBlock1d(concat_ch, out_channels, kernel_size, norm, groups, activation, padding_mode, key=k2)

    def __call__(self, x, skip, **kwargs):
        if self.up_mode == "upconv":
            x = self.upsample(x)
        else:
            L = x.shape[-2]
            new_shape = x.shape[:-2] + (L * 2, x.shape[-1])
            x = jax.image.resize(x, shape=new_shape, method="linear")
            x = self.upsample(x)
        # Handle size mismatch
        target_l = skip.shape[-2]
        curr_l = x.shape[-2]
        if curr_l < target_l:
            pad_l = target_l - curr_l
            x = jnp.pad(x, ((0, pad_l), (0, 0)), mode="edge")
        x = x[..., :target_l, :]
        # Concatenate
        if self.groups == 1:
            x = jnp.concatenate([x, skip], axis=-1)
        else:
            ch = x.shape[-1]
            ch_per_g = ch // self.groups
            parts = []
            for g in range(self.groups):
                s, e = g * ch_per_g, (g + 1) * ch_per_g
                parts.extend([x[..., s:e], skip[..., s:e]])
            x = jnp.concatenate(parts, axis=-1)
        return self.conv_block(x)


class UNet1D(eqx.Module):
    encoders: list
    bottleneck: UNetConvBlock1d
    decoders: list
    final_conv: Conv1dNHWC
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(self, in_channels=1, out_channels=1, depth=4, wf=6, norm="batch", up_mode="upconv", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        keys = jax.random.split(key, 2 * depth + 2)
        ki = 0
        encoders = []
        enc_in = in_channels
        for i in range(depth):
            ch = (2**wf) * (2**i)
            encoders.append(UNetDownBlock1d(enc_in, (ch, ch), 3, norm, groups, activation, padding_mode, key=keys[ki]))
            enc_in = ch
            ki += 1
        self.encoders = encoders
        bneck_ch = (2**wf) * (2 ** (depth - 1))
        self.bottleneck = UNetConvBlock1d(enc_in, (bneck_ch, bneck_ch), 3, norm, groups, activation, padding_mode, key=keys[ki])
        ki += 1
        decoders = []
        dec_in = bneck_ch
        for i in range(depth):
            didx = depth - 1 - i
            ch_in = (2**wf) * (2**didx)
            ch_out = (2**wf) * (2 ** max(0, didx - 1))
            skip_ch = ch_in
            decoders.append(UNetUpBlock1d(dec_in, skip_ch, (ch_in, ch_out), up_mode, 3, norm, groups, activation, padding_mode, key=keys[ki]))
            dec_in = ch_out
            ki += 1
        self.decoders = decoders
        self.final_conv = Conv1dNHWC(dec_in, out_channels, 1, padding="SAME", use_bias=False, key=keys[ki])

    def __call__(self, x, **kwargs):
        input_ndim = x.ndim
        if x.ndim == 1:
            x = x[..., jnp.newaxis]
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
        x = self.final_conv(x)
        if input_ndim == 1 and self.out_channels == 1:
            x = x[..., 0]
        return x


# ============================================================
# 2D UNet
# ============================================================


class UNetConv2d(eqx.Module):
    conv: Conv2dNHWC
    norm_layer: Optional[eqx.Module]
    activation: Optional[Callable] = eqx.field(static=True)
    pad: int = eqx.field(static=True)
    padding_mode: str = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size=3, norm="batch", groups=1, activation=None, padding_mode="circular", *, key):
        self.conv = Conv2dNHWC(in_ch, out_ch, kernel_size, padding="VALID", groups=groups, key=key)
        self.norm_layer = _make_norm(norm, out_ch, spatial_ndim=2)
        self.activation = activation
        self.pad = kernel_size // 2
        self.padding_mode = padding_mode

    def __call__(self, x, **kwargs):
        x = pad_2d(x, self.pad, self.padding_mode)
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNetConvBlock2d(eqx.Module):
    conv1: UNetConv2d
    conv2: UNetConv2d

    def __init__(self, in_ch, out_channels, kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = UNetConv2d(in_ch, out_channels[0], kernel_size, norm, groups, activation, padding_mode, key=k1)
        self.conv2 = UNetConv2d(out_channels[0], out_channels[1], kernel_size, norm, groups, None, padding_mode, key=k2)

    def __call__(self, x, **kwargs):
        return self.conv2(self.conv1(x))


class UNetDownBlock2d(eqx.Module):
    conv_block: UNetConvBlock2d

    def __init__(self, in_ch, out_channels, kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        self.conv_block = UNetConvBlock2d(in_ch, out_channels, kernel_size, norm, groups, activation, padding_mode, key=key)

    def __call__(self, x, **kwargs):
        skip = self.conv_block(x)
        x_down = avg_pool_2d(skip)
        return x_down, skip


class UNetUpBlock2d(eqx.Module):
    upsample: eqx.Module
    conv_block: UNetConvBlock2d
    up_mode: str = eqx.field(static=True)
    groups: int = eqx.field(static=True)

    def __init__(self, in_ch, skip_ch, out_channels, up_mode="upconv", kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        k1, k2 = jax.random.split(key)
        self.up_mode = up_mode
        self.groups = groups
        if up_mode == "upconv":
            self.upsample = ConvTranspose2d(in_ch, in_ch, 2, strides=(2, 2), key=k1)
        else:
            self.upsample = Conv2dNHWC(in_ch, in_ch, 1, padding="SAME", key=k1)
        concat_ch = in_ch + skip_ch
        self.conv_block = UNetConvBlock2d(concat_ch, out_channels, kernel_size, norm, groups, activation, padding_mode, key=k2)

    def __call__(self, x, skip, **kwargs):
        if self.up_mode == "upconv":
            x = self.upsample(x)
        else:
            H, W = x.shape[-3], x.shape[-2]
            new_shape = x.shape[:-3] + (H * 2, W * 2, x.shape[-1])
            x = jax.image.resize(x, shape=new_shape, method="bilinear")
            x = self.upsample(x)
        target_h, target_w = skip.shape[-3], skip.shape[-2]
        curr_h, curr_w = x.shape[-3], x.shape[-2]
        if curr_h < target_h or curr_w < target_w:
            pad_h = max(0, target_h - curr_h)
            pad_w = max(0, target_w - curr_w)
            x = jnp.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        x = x[..., :target_h, :target_w, :]
        if self.groups == 1:
            x = jnp.concatenate([x, skip], axis=-1)
        else:
            ch = x.shape[-1]
            ch_per_g = ch // self.groups
            parts = []
            for g in range(self.groups):
                s, e = g * ch_per_g, (g + 1) * ch_per_g
                parts.extend([x[..., s:e], skip[..., s:e]])
            x = jnp.concatenate(parts, axis=-1)
        return self.conv_block(x)


class UNet2D(eqx.Module):
    encoders: list
    bottleneck: UNetConvBlock2d
    decoders: list
    final_conv: Conv2dNHWC
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(self, in_channels=1, out_channels=1, depth=4, wf=6, norm="batch", up_mode="upconv", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        keys = jax.random.split(key, 2 * depth + 2)
        ki = 0
        encoders = []
        enc_in = in_channels
        for i in range(depth):
            ch = (2**wf) * (2**i)
            encoders.append(UNetDownBlock2d(enc_in, (ch, ch), 3, norm, groups, activation, padding_mode, key=keys[ki]))
            enc_in = ch
            ki += 1
        self.encoders = encoders
        bneck_ch = (2**wf) * (2 ** (depth - 1))
        self.bottleneck = UNetConvBlock2d(enc_in, (bneck_ch, bneck_ch), 3, norm, groups, activation, padding_mode, key=keys[ki])
        ki += 1
        decoders = []
        dec_in = bneck_ch
        for i in range(depth):
            didx = depth - 1 - i
            ch_in = (2**wf) * (2**didx)
            ch_out = (2**wf) * (2 ** max(0, didx - 1))
            skip_ch = ch_in
            decoders.append(UNetUpBlock2d(dec_in, skip_ch, (ch_in, ch_out), up_mode, 3, norm, groups, activation, padding_mode, key=keys[ki]))
            dec_in = ch_out
            ki += 1
        self.decoders = decoders
        self.final_conv = Conv2dNHWC(dec_in, out_channels, 1, padding="SAME", use_bias=False, key=keys[ki])

    def __call__(self, x, **kwargs):
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
        x = self.final_conv(x)
        return x


# ============================================================
# 3D UNet
# ============================================================


class UNetConv3d(eqx.Module):
    conv: Conv3dNHWC
    norm_layer: Optional[eqx.Module]
    activation: Optional[Callable] = eqx.field(static=True)
    pad: int = eqx.field(static=True)
    pad_fn: Callable = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size=3, norm="batch", groups=1, activation=None, padding_mode="circular", *, key):
        self.conv = Conv3dNHWC(in_ch, out_ch, kernel_size, padding="VALID", groups=groups, key=key)
        self.norm_layer = _make_norm(norm, out_ch, spatial_ndim=3)
        self.activation = activation
        self.pad = kernel_size // 2
        self.pad_fn = get_pad_fn(padding_mode)

    def __call__(self, x, **kwargs):
        x = self.pad_fn(x, self.pad)
        x = self.conv(x)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class UNetConvBlock3d(eqx.Module):
    conv1: UNetConv3d
    conv2: UNetConv3d

    def __init__(self, in_ch, out_channels, kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        k1, k2 = jax.random.split(key)
        self.conv1 = UNetConv3d(in_ch, out_channels[0], kernel_size, norm, groups, activation, padding_mode, key=k1)
        self.conv2 = UNetConv3d(out_channels[0], out_channels[1], kernel_size, norm, groups, None, padding_mode, key=k2)

    def __call__(self, x, **kwargs):
        return self.conv2(self.conv1(x))


class UNetDownBlock3d(eqx.Module):
    conv_block: UNetConvBlock3d

    def __init__(self, in_ch, out_channels, kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        self.conv_block = UNetConvBlock3d(in_ch, out_channels, kernel_size, norm, groups, activation, padding_mode, key=key)

    def __call__(self, x, **kwargs):
        skip = self.conv_block(x)
        x_down = avg_pool_3d(skip)
        return x_down, skip


class UNetUpBlock3d(eqx.Module):
    upsample: eqx.Module
    conv_block: UNetConvBlock3d
    up_mode: str = eqx.field(static=True)

    def __init__(self, in_ch, skip_ch, out_channels, up_mode="upconv", kernel_size=3, norm="batch", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key):
        k1, k2 = jax.random.split(key)
        self.up_mode = up_mode
        if up_mode == "upconv":
            self.upsample = ConvTranspose3d(in_ch, in_ch, 2, strides=(2, 2, 2), key=k1)
        else:
            self.upsample = Conv3dNHWC(in_ch, in_ch, 1, padding="SAME", key=k1)
        concat_ch = in_ch + skip_ch
        self.conv_block = UNetConvBlock3d(concat_ch, out_channels, kernel_size, norm, groups, activation, padding_mode, key=k2)

    def __call__(self, x, skip, **kwargs):
        if self.up_mode == "upconv":
            x = self.upsample(x)
        else:
            D, H, W = x.shape[-4], x.shape[-3], x.shape[-2]
            x = jax.image.resize(x, shape=(*x.shape[:-4], D * 2, H * 2, W * 2, x.shape[-1]), method="trilinear")
            x = self.upsample(x)
        target_shape = skip.shape[:-1]
        x = x[..., : target_shape[-3], : target_shape[-2], : target_shape[-1], :]
        x = jnp.concatenate([x, skip], axis=-1)
        return self.conv_block(x)


class UNet3D(eqx.Module):
    encoders: list
    bottleneck: UNetConvBlock3d
    decoders: list
    final_conv: Conv3dNHWC
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(self, in_channels=1, out_channels=2, depth=4, wf=6, norm="batch", up_mode="upconv", groups=1, activation=jax.nn.celu, padding_mode="circular", *, key, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        keys = jax.random.split(key, 2 * depth + 2)
        ki = 0
        encoders = []
        enc_in = in_channels
        for i in range(depth):
            ch = (2**wf) * (2**i)
            encoders.append(UNetDownBlock3d(enc_in, (ch, ch), 3, norm, groups, activation, padding_mode, key=keys[ki]))
            enc_in = ch
            ki += 1
        self.encoders = encoders
        bneck_ch = (2**wf) * (2 ** (depth - 1))
        self.bottleneck = UNetConvBlock3d(enc_in, (bneck_ch, bneck_ch), 3, norm, groups, activation, padding_mode, key=keys[ki])
        ki += 1
        decoders = []
        dec_in = bneck_ch
        for i in range(depth):
            didx = depth - 1 - i
            ch_in = (2**wf) * (2**didx)
            ch_out = (2**wf) * (2 ** max(0, didx - 1))
            skip_ch = ch_in
            decoders.append(UNetUpBlock3d(dec_in, skip_ch, (ch_in, ch_out), up_mode, 3, norm, groups, activation, padding_mode, key=keys[ki]))
            dec_in = ch_out
            ki += 1
        self.decoders = decoders
        self.final_conv = Conv3dNHWC(dec_in, out_channels, 1, padding="SAME", use_bias=False, key=keys[ki])

    def __call__(self, x, **kwargs):
        input_ndim = x.ndim
        if x.ndim == 3:
            x = x[..., jnp.newaxis]
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
        x = self.final_conv(x)
        if input_ndim == 3 and self.out_channels == 1:
            x = x[..., 0]
        return x
