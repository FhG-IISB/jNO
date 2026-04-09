# FNO implementation taken from https://rodrigodzf.com/physmodjax/models/fno.html
import equinox as eqx
from .linear import Linear
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from einops import rearrange
from .common import BatchNorm


# 1 Dimensional Fourier Neural Operator


class SpectralConv1d(eqx.Module):
    """Spectral Convolution Layer for 1D inputs."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    weight_real: jnp.ndarray
    weight_imag: jnp.ndarray

    def __init__(self, in_channels: int, out_channels: int, n_modes: int, linear_conv: bool = True, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.linear_conv = linear_conv

        weight_shape = (in_channels, out_channels, n_modes)
        scale = 1 / (in_channels * out_channels)

        key1, key2 = jax.random.split(key)
        self.weight_real = jax.random.uniform(key1, weight_shape, minval=-scale, maxval=scale)
        self.weight_imag = jax.random.uniform(key2, weight_shape, minval=-scale, maxval=scale)

    def __call__(self, x: jnp.ndarray, **kwargs):  # (w, c)
        W, C = x.shape

        # Compute FFT length based on convolution type
        if self.linear_conv:
            fft_len = W * 2 - 1
        else:
            fft_len = W

        # Get the Fourier coefficients along the spatial dimension
        X = jnp.fft.rfft(x, n=fft_len, axis=0, norm="ortho")

        # Truncate to the first n_modes coefficients
        n_modes_actual = min(self.n_modes, X.shape[0])
        X_truncated = X[:n_modes_actual, :]

        # Create complex weight and multiply
        complex_weight = self.weight_real[:, :, :n_modes_actual] + 1j * self.weight_imag[:, :, :n_modes_actual]

        # Einsum: (modes, in_channels) x (in_channels, out_channels, modes) -> (modes, out_channels)
        X_out = jnp.einsum("mi,iom->mo", X_truncated, complex_weight)

        # Pad back to full frequency domain
        full_freq_len = fft_len // 2 + 1
        X_padded = jnp.zeros((full_freq_len, self.out_channels), dtype=X_out.dtype)
        X_padded = X_padded.at[:n_modes_actual, :].set(X_out)

        # Inverse FFT and truncate to original length
        x_out = jnp.fft.irfft(X_padded, n=fft_len, axis=0, norm="ortho")

        # Truncate to original spatial dimension
        x_out = x_out[:W, :]

        return x_out


class SpectralLayers1d(eqx.Module):
    """Stack of 1D Spectral Convolution Layers"""

    n_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)
    layers_conv: list
    layers_w: list
    norm_layers: Optional[list]

    def __init__(self, n_channels: int, n_modes: int, linear_conv: bool = True, n_layers: int = 4, activation: Callable = jax.nn.gelu, norm: Optional[str] = None, training: bool = True, *, key):
        self.n_channels = n_channels
        self.n_modes = n_modes
        self.linear_conv = linear_conv
        self.n_layers = n_layers
        self.activation = activation
        self.norm = norm

        keys = jax.random.split(key, n_layers * 2)

        self.layers_conv = [
            SpectralConv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                n_modes=n_modes,
                linear_conv=linear_conv,
                key=keys[i],
            )
            for i in range(n_layers)
        ]

        self.layers_w = [
            eqx.nn.Conv1d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=1,
                key=keys[n_layers + i],
            )
            for i in range(n_layers)
        ]

        # Setup normalization layers
        if norm == "layer":
            self.norm_layers = [eqx.nn.LayerNorm(n_channels) for _ in range(n_layers)]
        elif norm == "batch":
            self.norm_layers = [BatchNorm(n_channels) for _ in range(n_layers)]
        elif norm == "instance":
            self.norm_layers = [eqx.nn.LayerNorm(n_channels) for _ in range(n_layers)]
        else:
            self.norm_layers = None

    def __call__(self, x, **kwargs):  # (grid_points, channels)
        for i, (conv, w) in enumerate(zip(self.layers_conv, self.layers_w)):
            x1 = conv(x)
            # Conv1d expects (channels, spatial), so transpose
            x2 = w(rearrange(x, "w c -> c w"))
            x2 = rearrange(x2, "c w -> w c")
            x = x1 + x2

            # Apply normalization if specified
            if self.norm_layers is not None:
                # vmap LayerNorm over spatial dim (W,)
                x = jax.vmap(self.norm_layers[i])(x)

            # Apply activation
            x = self.activation(x)

        return x


class FNO1D(eqx.Module):
    hidden_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    d_vars: int = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    lift: Linear
    spectral_layers: SpectralLayers1d
    proj1: Linear
    proj2: Linear
    drop1: Optional[eqx.nn.Dropout]
    drop2: Optional[eqx.nn.Dropout]

    def __init__(
        self,
        in_features: int,
        hidden_channels: int,
        n_modes: int,
        d_vars: int = 1,
        linear_conv: bool = True,
        n_layers: int = 4,
        n_steps: int = 1,
        activation: Callable = jax.nn.gelu,
        norm: Optional[str] = None,
        training: bool = True,
        dropout_rate: float = 0.0,
        *,
        key,
    ):
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.d_vars = d_vars
        self.linear_conv = linear_conv
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.activation = activation
        self.norm = norm
        self.dropout_rate = dropout_rate

        k1, k2, k3, k4 = jax.random.split(key, 4)

        # Lift
        self.lift = Linear(in_features, hidden_channels, key=k1)

        # Spectral layers
        self.spectral_layers = SpectralLayers1d(
            n_channels=hidden_channels,
            n_modes=n_modes,
            linear_conv=linear_conv,
            n_layers=n_layers,
            activation=activation,
            norm=norm,
            training=training,
            key=k2,
        )

        # Project
        self.proj1 = Linear(hidden_channels, 128, key=k3)
        self.proj2 = Linear(128, d_vars * n_steps, key=k4)

        # Dropout
        if dropout_rate > 0.0:
            self.drop1 = eqx.nn.Dropout(p=dropout_rate)
            self.drop2 = eqx.nn.Dropout(p=dropout_rate)
        else:
            self.drop1 = None
            self.drop2 = None

    def __call__(self, x, key=None, **kwargs):  # input (T, W, C)
        """
        The input to the FNO1D model is a 1D signal of shape (t, w, c)
        where w is the spatial dimension and c is the number of channels.
        """
        if x.ndim == 1:
            x = x[None, :, None]
        elif x.ndim == 2:
            x = x[None, :, :]

        x = rearrange(x, "t w c -> w (t c)")

        # Lift the input to the hidden dimension
        h = jax.vmap(self.lift)(x)

        # Apply spectral layers
        h = self.spectral_layers(h)

        # Optional dropout
        if self.drop1 is not None and key is not None:
            key, subkey = jax.random.split(key)
            h = self.drop1(h, key=subkey)

        # Project down to output dimension using a small MLP
        y = jax.vmap(self.proj1)(h)
        y = self.activation(y)

        if self.drop2 is not None and key is not None:
            key, subkey = jax.random.split(key)
            y = self.drop2(y, key=subkey)

        y = jax.vmap(self.proj2)(y)

        # Rearrange output to (t, w, c) format
        y = rearrange(y, "w (t c) -> t w c", t=self.n_steps, c=self.d_vars)

        return y


# 2 Dimensional Fourier Neural Operator


def create_grid(height: int, width: int) -> jnp.ndarray:
    """Create a 2D positional grid of shape (H, W, 2). Values normalized to [0, 1]."""
    y = jnp.linspace(0, 1, height)
    x = jnp.linspace(0, 1, width)
    yy, xx = jnp.meshgrid(y, x, indexing="ij")
    return jnp.stack([yy, xx], axis=-1)


class SpectralConv2d(eqx.Module):
    """2D Spectral Convolution Layer - JIT compatible."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes1: int = eqx.field(static=True)
    n_modes2: int = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    weight_1_real: jnp.ndarray
    weight_1_imag: jnp.ndarray
    weight_2_real: jnp.ndarray
    weight_2_imag: jnp.ndarray

    def __init__(self, in_channels: int, out_channels: int, n_modes1: int, n_modes2: int, linear_conv: bool = True, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        self.linear_conv = linear_conv

        shape = (in_channels, out_channels, n_modes1, n_modes2)
        scale = 1 / (in_channels * out_channels)

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.weight_1_real = jax.random.uniform(k1, shape, minval=-scale, maxval=scale)
        self.weight_1_imag = jax.random.uniform(k2, shape, minval=-scale, maxval=scale)
        self.weight_2_real = jax.random.uniform(k3, shape, minval=-scale, maxval=scale)
        self.weight_2_imag = jax.random.uniform(k4, shape, minval=-scale, maxval=scale)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        H, W, C = x.shape

        w1 = self.weight_1_real + 1j * self.weight_1_imag
        w2 = self.weight_2_real + 1j * self.weight_2_imag

        # Static FFT sizes
        fft_h = H * 2 - 1 if self.linear_conv else H
        fft_w = W * 2 - 1 if self.linear_conv else W

        # 2D real FFT
        X = jnp.fft.rfft2(x, s=(fft_h, fft_w), axes=(0, 1), norm="ortho")
        freq_h, freq_w = X.shape[0], X.shape[1]

        # Static mode counts (use Python min, not jnp.minimum)
        n_modes1 = min(self.n_modes1, (freq_h + 1) // 2)
        n_modes2 = min(self.n_modes2, freq_w)

        # Process upper and lower frequencies with static slicing
        X_upper = X[:n_modes1, :n_modes2, :]
        X_lower = X[-n_modes1:, :n_modes2, :]

        w1_slice = w1[:, :, :n_modes1, :n_modes2]
        w2_slice = w2[:, :, :n_modes1, :n_modes2]

        out_upper = jnp.einsum("hwi,iohw->hwo", X_upper, w1_slice)
        out_lower = jnp.einsum("hwi,iohw->hwo", X_lower, w2_slice)

        # Build output in frequency domain
        out_ft = jnp.zeros((freq_h, freq_w, self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:n_modes1, :n_modes2, :].set(out_upper)
        out_ft = out_ft.at[-n_modes1:, :n_modes2, :].set(out_lower)

        # Inverse FFT and truncate
        return jnp.fft.irfft2(out_ft, s=(fft_h, fft_w), axes=(0, 1), norm="ortho")[:H, :W, :]


class SpectralLayers2d(eqx.Module):
    """Stack of 2D Spectral Convolution Layers."""

    n_channels: int = eqx.field(static=True)
    n_modes1: int = eqx.field(static=True)
    n_modes2: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    conv_layers: list
    w_layers: list
    norm_layers: Optional[list]

    def __init__(self, n_channels: int, n_modes1: int, n_modes2: int, n_layers: int = 4, activation: Callable = jax.nn.gelu, norm: Optional[str] = None, training: bool = True, linear_conv: bool = True, *, key):
        self.n_channels = n_channels
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        self.n_layers = n_layers
        self.activation = activation
        self.norm = norm
        self.linear_conv = linear_conv

        keys = jax.random.split(key, n_layers * 2)

        self.conv_layers = [
            SpectralConv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                n_modes1=n_modes1,
                n_modes2=n_modes2,
                linear_conv=linear_conv,
                key=keys[i],
            )
            for i in range(n_layers)
        ]

        self.w_layers = [
            eqx.nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=1,
                key=keys[n_layers + i],
            )
            for i in range(n_layers)
        ]

        if norm == "layer":
            self.norm_layers = [eqx.nn.LayerNorm(n_channels) for _ in range(n_layers)]
        elif norm == "batch":
            self.norm_layers = [BatchNorm(n_channels) for _ in range(n_layers)]
        else:
            self.norm_layers = None

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for i, (conv, w) in enumerate(zip(self.conv_layers, self.w_layers)):
            x1 = conv(x)
            # Conv2d expects (channels, h, w), so transpose
            x2 = w(rearrange(x, "h w c -> c h w"))
            x2 = rearrange(x2, "c h w -> h w c")
            x = x1 + x2
            if self.norm_layers is not None:
                # vmap LayerNorm over spatial dims (H, W)
                x = jax.vmap(jax.vmap(self.norm_layers[i]))(x)
            x = self.activation(x)
        return x


class FNO2D(eqx.Module):
    """2D Fourier Neural Operator - JIT compatible."""

    hidden_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    d_vars: int = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    d_model: Tuple[int, int] = eqx.field(static=True)
    use_positions: bool = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)
    P: Linear
    spectral_layers: SpectralLayers2d
    Q_layers: list
    grid: Optional[jnp.ndarray]

    def __init__(
        self,
        in_features: int,
        hidden_channels: int,
        n_modes: int,
        d_vars: int = 1,
        linear_conv: bool = True,
        n_layers: int = 4,
        n_steps: int = 1,
        activation: Callable = jax.nn.gelu,
        d_model: Tuple[int, int] = (64, 64),
        use_positions: bool = False,
        norm: Optional[str] = "layer",
        training: bool = True,
        *,
        key,
    ):
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.d_vars = d_vars
        self.linear_conv = linear_conv
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.activation = activation
        self.d_model = d_model
        self.use_positions = use_positions
        self.norm = norm

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.P = Linear(in_features, hidden_channels, key=k1)

        self.spectral_layers = SpectralLayers2d(
            n_channels=hidden_channels,
            n_modes1=n_modes,
            n_modes2=n_modes,
            n_layers=n_layers,
            activation=activation,
            norm=norm,
            training=training,
            linear_conv=linear_conv,
            key=k2,
        )

        self.Q_layers = [
            Linear(hidden_channels, 128, key=k3),
            Linear(128, d_vars * n_steps, key=k4),
        ]

        if use_positions:
            self.grid = create_grid(d_model[0], d_model[1])
        else:
            self.grid = None

    def _normalize_input(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """Normalize input to (T, H, W, C) format."""
        ndim = x.ndim
        if ndim == 2:
            x = x[:, :, jnp.newaxis][jnp.newaxis, :, :, :]
        elif ndim == 3:
            x = x[jnp.newaxis, :, :, :]
        return x, ndim

    def _denormalize_output(self, x: jnp.ndarray, original_ndim: int) -> jnp.ndarray:
        """Convert output back to original format."""
        if original_ndim == 2 and self.n_steps == 1 and self.d_vars == 1:
            return x[0, :, :, 0]
        elif original_ndim == 2 and self.n_steps == 1:
            return x[0, :, :, :]
        elif original_ndim == 3 and self.n_steps == 1:
            return x[0, :, :, :]
        return x

    def _apply_Q(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.Q_layers[0](x)
        x = self.activation(x)
        x = self.Q_layers[1](x)
        return x

    def advance(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_positions and self.grid is not None:
            x = jnp.concatenate([x, self.grid], axis=-1)
        # vmap Linear over spatial dims (H, W)
        x = jax.vmap(jax.vmap(self.P))(x)
        x = self.spectral_layers(x)
        x = jax.vmap(jax.vmap(self._apply_Q))(x)
        return x

    def __call__(self, x: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        x, original_ndim = self._normalize_input(x)
        x = rearrange(x, "t h w c -> h w (t c)")
        x = self.advance(x)
        x = rearrange(x, "h w (t c) -> t h w c", t=self.n_steps, c=self.d_vars)
        return self._denormalize_output(x, original_ndim)


# 3 Dimensional Fourier Neural Operator


def create_grid_3d(depth: int, height: int, width: int) -> jnp.ndarray:
    """Create a 3D positional grid of shape (D, H, W, 3). Values normalized to [0, 1]."""
    z = jnp.linspace(0, 1, depth)
    y = jnp.linspace(0, 1, height)
    x = jnp.linspace(0, 1, width)
    zz, yy, xx = jnp.meshgrid(z, y, x, indexing="ij")
    return jnp.stack([zz, yy, xx], axis=-1)


class SpectralConv3d(eqx.Module):
    """3D Spectral Convolution Layer - JIT compatible."""

    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    n_modes1: int = eqx.field(static=True)
    n_modes2: int = eqx.field(static=True)
    n_modes3: int = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    weight_1_real: jnp.ndarray
    weight_1_imag: jnp.ndarray
    weight_2_real: jnp.ndarray
    weight_2_imag: jnp.ndarray
    weight_3_real: jnp.ndarray
    weight_3_imag: jnp.ndarray
    weight_4_real: jnp.ndarray
    weight_4_imag: jnp.ndarray

    def __init__(self, in_channels: int, out_channels: int, n_modes1: int, n_modes2: int, n_modes3: int, linear_conv: bool = True, *, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        self.n_modes3 = n_modes3
        self.linear_conv = linear_conv

        shape = (in_channels, out_channels, n_modes1, n_modes2, n_modes3)
        scale = 1 / (in_channels * out_channels)

        keys = jax.random.split(key, 8)
        self.weight_1_real = jax.random.uniform(keys[0], shape, minval=-scale, maxval=scale)
        self.weight_1_imag = jax.random.uniform(keys[1], shape, minval=-scale, maxval=scale)
        self.weight_2_real = jax.random.uniform(keys[2], shape, minval=-scale, maxval=scale)
        self.weight_2_imag = jax.random.uniform(keys[3], shape, minval=-scale, maxval=scale)
        self.weight_3_real = jax.random.uniform(keys[4], shape, minval=-scale, maxval=scale)
        self.weight_3_imag = jax.random.uniform(keys[5], shape, minval=-scale, maxval=scale)
        self.weight_4_real = jax.random.uniform(keys[6], shape, minval=-scale, maxval=scale)
        self.weight_4_imag = jax.random.uniform(keys[7], shape, minval=-scale, maxval=scale)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        D, H, W, C = x.shape

        w1 = self.weight_1_real + 1j * self.weight_1_imag
        w2 = self.weight_2_real + 1j * self.weight_2_imag
        w3 = self.weight_3_real + 1j * self.weight_3_imag
        w4 = self.weight_4_real + 1j * self.weight_4_imag

        # Static FFT sizes
        fft_d = D * 2 - 1 if self.linear_conv else D
        fft_h = H * 2 - 1 if self.linear_conv else H
        fft_w = W * 2 - 1 if self.linear_conv else W

        # 3D real FFT
        X = jnp.fft.rfftn(x, s=(fft_d, fft_h, fft_w), axes=(0, 1, 2), norm="ortho")
        freq_d, freq_h, freq_w = X.shape[0], X.shape[1], X.shape[2]

        # Static mode counts
        n_modes1 = min(self.n_modes1, (freq_d + 1) // 2)
        n_modes2 = min(self.n_modes2, (freq_h + 1) // 2)
        n_modes3 = min(self.n_modes3, freq_w)

        # Slice weights to actual mode counts
        w1_slice = w1[:, :, :n_modes1, :n_modes2, :n_modes3]
        w2_slice = w2[:, :, :n_modes1, :n_modes2, :n_modes3]
        w3_slice = w3[:, :, :n_modes1, :n_modes2, :n_modes3]
        w4_slice = w4[:, :, :n_modes1, :n_modes2, :n_modes3]

        # Process 4 corners (combinations of upper/lower in depth and height dimensions)
        # Corner 1: upper depth, upper height
        X_corner1 = X[:n_modes1, :n_modes2, :n_modes3, :]
        out_corner1 = jnp.einsum("dhwi,iodhw->dhwo", X_corner1, w1_slice)

        # Corner 2: upper depth, lower height
        X_corner2 = X[:n_modes1, -n_modes2:, :n_modes3, :]
        out_corner2 = jnp.einsum("dhwi,iodhw->dhwo", X_corner2, w2_slice)

        # Corner 3: lower depth, upper height
        X_corner3 = X[-n_modes1:, :n_modes2, :n_modes3, :]
        out_corner3 = jnp.einsum("dhwi,iodhw->dhwo", X_corner3, w3_slice)

        # Corner 4: lower depth, lower height
        X_corner4 = X[-n_modes1:, -n_modes2:, :n_modes3, :]
        out_corner4 = jnp.einsum("dhwi,iodhw->dhwo", X_corner4, w4_slice)

        # Build output in frequency domain
        out_ft = jnp.zeros((freq_d, freq_h, freq_w, self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:n_modes1, :n_modes2, :n_modes3, :].set(out_corner1)
        out_ft = out_ft.at[:n_modes1, -n_modes2:, :n_modes3, :].set(out_corner2)
        out_ft = out_ft.at[-n_modes1:, :n_modes2, :n_modes3, :].set(out_corner3)
        out_ft = out_ft.at[-n_modes1:, -n_modes2:, :n_modes3, :].set(out_corner4)

        # Inverse FFT and truncate
        return jnp.fft.irfftn(out_ft, s=(fft_d, fft_h, fft_w), axes=(0, 1, 2), norm="ortho")[:D, :H, :W, :]


class SpectralLayers3d(eqx.Module):
    """Stack of 3D Spectral Convolution Layers."""

    n_channels: int = eqx.field(static=True)
    n_modes1: int = eqx.field(static=True)
    n_modes2: int = eqx.field(static=True)
    n_modes3: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    conv_layers: list
    w_layers: list
    norm_layers: Optional[list]

    def __init__(self, n_channels: int, n_modes1: int, n_modes2: int, n_modes3: int, n_layers: int = 4, activation: Callable = jax.nn.gelu, norm: Optional[str] = None, training: bool = True, linear_conv: bool = True, *, key):
        self.n_channels = n_channels
        self.n_modes1 = n_modes1
        self.n_modes2 = n_modes2
        self.n_modes3 = n_modes3
        self.n_layers = n_layers
        self.activation = activation
        self.norm = norm
        self.linear_conv = linear_conv

        keys = jax.random.split(key, n_layers * 2)

        self.conv_layers = [
            SpectralConv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                n_modes1=n_modes1,
                n_modes2=n_modes2,
                n_modes3=n_modes3,
                linear_conv=linear_conv,
                key=keys[i],
            )
            for i in range(n_layers)
        ]

        self.w_layers = [
            eqx.nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=1,
                key=keys[n_layers + i],
            )
            for i in range(n_layers)
        ]

        if norm == "layer":
            self.norm_layers = [eqx.nn.LayerNorm(n_channels) for _ in range(n_layers)]
        elif norm == "batch":
            self.norm_layers = [BatchNorm(n_channels) for _ in range(n_layers)]
        else:
            self.norm_layers = None

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        for i, (conv, w) in enumerate(zip(self.conv_layers, self.w_layers)):
            x1 = conv(x)
            # Conv3d expects (channels, d, h, w), so transpose
            x2 = w(rearrange(x, "d h w c -> c d h w"))
            x2 = rearrange(x2, "c d h w -> d h w c")
            x = x1 + x2
            if self.norm_layers is not None:
                # vmap LayerNorm over spatial dims (D, H, W)
                x = jax.vmap(jax.vmap(jax.vmap(self.norm_layers[i])))(x)
            x = self.activation(x)
        return x


class FNO3D(eqx.Module):
    """3D Fourier Neural Operator - JIT compatible."""

    hidden_channels: int = eqx.field(static=True)
    n_modes: int = eqx.field(static=True)
    d_vars: int = eqx.field(static=True)
    linear_conv: bool = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    activation: Callable = eqx.field(static=True)
    d_model: Tuple[int, int, int] = eqx.field(static=True)
    use_positions: bool = eqx.field(static=True)
    norm: Optional[str] = eqx.field(static=True)
    dropout_rate: float = eqx.field(static=True)
    P: Linear
    spectral_layers: SpectralLayers3d
    Q_layers: list
    grid: Optional[jnp.ndarray]
    dropout: Optional[eqx.nn.Dropout]

    def __init__(
        self,
        in_features: int,
        hidden_channels: int,
        n_modes: int,
        d_vars: int = 1,
        linear_conv: bool = True,
        n_layers: int = 4,
        n_steps: int = 1,
        activation: Callable = jax.nn.gelu,
        d_model: Tuple[int, int, int] = (32, 32, 32),
        use_positions: bool = False,
        norm: Optional[str] = "layer",
        training: bool = True,
        dropout_rate: float = 0.0,
        *,
        key,
    ):
        self.hidden_channels = hidden_channels
        self.n_modes = n_modes
        self.d_vars = d_vars
        self.linear_conv = linear_conv
        self.n_layers = n_layers
        self.n_steps = n_steps
        self.activation = activation
        self.d_model = d_model
        self.use_positions = use_positions
        self.norm = norm
        self.dropout_rate = dropout_rate

        k1, k2, k3, k4 = jax.random.split(key, 4)

        self.P = Linear(in_features, hidden_channels, key=k1)

        self.spectral_layers = SpectralLayers3d(
            n_channels=hidden_channels,
            n_modes1=n_modes,
            n_modes2=n_modes,
            n_modes3=n_modes,
            n_layers=n_layers,
            activation=activation,
            norm=norm,
            training=training,
            linear_conv=linear_conv,
            key=k2,
        )

        self.Q_layers = [
            Linear(hidden_channels, 128, key=k3),
            Linear(128, d_vars * n_steps, key=k4),
        ]

        if use_positions:
            self.grid = create_grid_3d(d_model[0], d_model[1], d_model[2])
        else:
            self.grid = None

        if dropout_rate > 0.0:
            self.dropout = eqx.nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def _normalize_input(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """Normalize input to (T, D, H, W, C) format."""
        ndim = x.ndim
        if ndim == 3:
            x = x[:, :, :, jnp.newaxis][jnp.newaxis, :, :, :, :]
        elif ndim == 4:
            x = x[jnp.newaxis, :, :, :, :]
        return x, ndim

    def _denormalize_output(self, x: jnp.ndarray, original_ndim: int) -> jnp.ndarray:
        """Convert output back to original format."""
        if original_ndim == 3 and self.n_steps == 1 and self.d_vars == 1:
            return x[0, :, :, :, 0]
        elif original_ndim == 3 and self.n_steps == 1:
            return x[0, :, :, :, :]
        elif original_ndim == 4 and self.n_steps == 1:
            return x[0, :, :, :, :]
        return x

    def _apply_Q(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.Q_layers[0](x)
        x = self.activation(x)
        x = self.Q_layers[1](x)
        return x

    def advance(self, x: jnp.ndarray, key=None) -> jnp.ndarray:
        if self.use_positions and self.grid is not None:
            x = jnp.concatenate([x, self.grid], axis=-1)
        # vmap Linear over spatial dims (D, H, W)
        x = jax.vmap(jax.vmap(jax.vmap(self.P)))(x)
        x = self.spectral_layers(x)

        # Optional dropout
        if self.dropout is not None and key is not None:
            key, subkey = jax.random.split(key)
            x = self.dropout(x, key=subkey)

        x = jax.vmap(jax.vmap(jax.vmap(self._apply_Q)))(x)
        return x

    def __call__(self, x: jnp.ndarray, key=None, **kwargs) -> jnp.ndarray:
        x, original_ndim = self._normalize_input(x)
        x = rearrange(x, "t d h w c -> d h w (t c)")
        x = self.advance(x, key=key)
        x = rearrange(x, "d h w (t c) -> t d h w c", t=self.n_steps, c=self.d_vars)
        return self._denormalize_output(x, original_ndim)
