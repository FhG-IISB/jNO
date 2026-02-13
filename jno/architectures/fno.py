# FNO implementation taken from https://rodrigodzf.com/physmodjax/models/fno.html
from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from flax.linen.initializers import uniform
from einops import rearrange


# 1 Dimensional Fourier Neural Operator


class SpectralConv1d(nn.Module):
    """Spectral Convolution Layer for 1D inputs."""

    in_channels: int
    out_channels: int
    n_modes: int
    linear_conv: bool = True

    def setup(self):
        weight_shape = (self.in_channels, self.out_channels, self.n_modes)
        scale = 1 / (self.in_channels * self.out_channels)

        self.weight_real = self.param(
            "weight_real",
            uniform(scale=scale),
            weight_shape,
        )
        self.weight_imag = self.param(
            "weight_imag",
            uniform(scale=scale),
            weight_shape,
        )

    def __call__(self, x: jnp.ndarray):  # (w, c)
        W, C = x.shape

        # Compute FFT length based on convolution type
        if self.linear_conv:
            # Linear convolution requires zero-padding
            fft_len = W * 2 - 1
        else:
            # Circular convolution uses original length
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


class SpectralLayers1d(nn.Module):
    """Stack of 1D Spectral Convolution Layers"""

    n_channels: int
    n_modes: int
    linear_conv: bool = True
    n_layers: int = 4
    activation: Callable = nn.gelu
    norm: Optional[str] = None
    training: bool = True

    def setup(self):
        self.layers_conv = [
            SpectralConv1d(
                in_channels=self.n_channels,
                out_channels=self.n_channels,
                n_modes=self.n_modes,
                linear_conv=self.linear_conv,
            )
            for _ in range(self.n_layers)
        ]

        self.layers_w = [nn.Conv(features=self.n_channels, kernel_size=(1,)) for _ in range(self.n_layers)]

        # Setup normalization layers
        if self.norm == "layer":
            self.norm_layers = [nn.LayerNorm() for _ in range(self.n_layers)]
        elif self.norm == "batch":
            self.norm_layers = [nn.BatchNorm(use_running_average=not self.training) for _ in range(self.n_layers)]
        elif self.norm == "instance":
            self.norm_layers = [nn.LayerNorm() for _ in range(self.n_layers)]  # Instance norm approximation
        else:
            self.norm_layers = None

    def __call__(self, x):  # (grid_points, channels)
        for i, (conv, w) in enumerate(zip(self.layers_conv, self.layers_w)):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2

            # Apply normalization if specified
            if self.norm_layers is not None:
                x = self.norm_layers[i](x)

            # Apply activation (skip on last layer optionally, but keeping it for now)
            x = self.activation(x)

        return x


class FNO1D(nn.Module):
    hidden_channels: int
    n_modes: int
    d_vars: int = 1
    linear_conv: bool = True
    n_layers: int = 4
    n_steps: int = 1
    activation: Callable = nn.gelu
    norm: Optional[str] = None
    training: bool = True
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x):  # input (T, W, C)
        """
        The input to the FNO1D model is a 1D signal of shape (t, w, c)
        where w is the spatial dimension and c is the number of channels.
        """

        # jax.debug.print("shape of input {inp}", inp=x.shape)
        # Rearrange: make time as channel dimension for spectral layers
        if x.ndim == 1:
            x = x[None, :, None]
        elif x.ndim == 2:
            x = x[None, :, :]

        x = rearrange(x, "t w c -> w (t c)")

        # Lift the input to the hidden dimension
        h = nn.Dense(features=self.hidden_channels)(x)

        # Apply spectral layers
        spectral_layers = SpectralLayers1d(
            n_channels=self.hidden_channels,
            n_modes=self.n_modes,
            linear_conv=self.linear_conv,
            n_layers=self.n_layers,
            activation=self.activation,
            norm=self.norm,
            training=self.training,
        )
        h = spectral_layers(h)

        # Optional dropout
        if self.dropout_rate > 0.0:
            h = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(h)

        # Project down to output dimension using a small MLP
        y = nn.Dense(features=128)(h)
        y = self.activation(y)

        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(y)

        y = nn.Dense(features=self.d_vars * self.n_steps)(y)

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


class SpectralConv2d(nn.Module):
    """2D Spectral Convolution Layer - JIT compatible."""

    in_channels: int
    out_channels: int
    n_modes1: int
    n_modes2: int
    linear_conv: bool = True

    def setup(self):
        shape = (self.in_channels, self.out_channels, self.n_modes1, self.n_modes2)
        scale = 1 / (self.in_channels * self.out_channels)

        self.weight_1_real = self.param("weight_1_real", uniform(scale=scale), shape)
        self.weight_1_imag = self.param("weight_1_imag", uniform(scale=scale), shape)
        self.weight_2_real = self.param("weight_2_real", uniform(scale=scale), shape)
        self.weight_2_imag = self.param("weight_2_imag", uniform(scale=scale), shape)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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


class SpectralLayers2d(nn.Module):
    """Stack of 2D Spectral Convolution Layers."""

    n_channels: int
    n_modes1: int
    n_modes2: int
    n_layers: int = 4
    activation: Callable = nn.gelu
    norm: Optional[str] = None
    training: bool = True
    linear_conv: bool = True

    def setup(self):
        self.conv_layers = [
            SpectralConv2d(
                in_channels=self.n_channels,
                out_channels=self.n_channels,
                n_modes1=self.n_modes1,
                n_modes2=self.n_modes2,
                linear_conv=self.linear_conv,
            )
            for _ in range(self.n_layers)
        ]

        self.w_layers = [nn.Conv(features=self.n_channels, kernel_size=(1, 1)) for _ in range(self.n_layers)]

        if self.norm == "layer":
            self.norm_layers = [nn.LayerNorm() for _ in range(self.n_layers)]
        elif self.norm == "batch":
            self.norm_layers = [nn.BatchNorm(use_running_average=not self.training) for _ in range(self.n_layers)]
        else:
            self.norm_layers = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, (conv, w) in enumerate(zip(self.conv_layers, self.w_layers)):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            if self.norm_layers is not None:
                x = self.norm_layers[i](x)
            x = self.activation(x)
        return x


class FNO2D(nn.Module):
    """2D Fourier Neural Operator - JIT compatible."""

    hidden_channels: int
    n_modes: int
    d_vars: int = 1
    linear_conv: bool = True
    n_layers: int = 4
    n_steps: int = 1
    activation: Callable = nn.gelu
    d_model: Tuple[int, int] = (64, 64)
    use_positions: bool = False
    norm: Optional[str] = "layer"
    training: bool = True

    def setup(self):
        self.P = nn.Dense(features=self.hidden_channels)

        self.spectral_layers = SpectralLayers2d(
            n_channels=self.hidden_channels,
            n_modes1=self.n_modes,
            n_modes2=self.n_modes,
            n_layers=self.n_layers,
            activation=self.activation,
            norm=self.norm,
            training=self.training,
            linear_conv=self.linear_conv,
        )

        self.Q = nn.Sequential(
            [
                nn.Dense(features=128),
                self.activation,
                nn.Dense(features=self.d_vars * self.n_steps),
            ]
        )

        if self.use_positions:
            self.grid = create_grid(self.d_model[0], self.d_model[1])

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

    def advance(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_positions:
            x = jnp.concatenate([x, self.grid], axis=-1)
        x = self.P(x)
        x = self.spectral_layers(x)
        x = self.Q(x)
        return x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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


class SpectralConv3d(nn.Module):
    """3D Spectral Convolution Layer - JIT compatible."""

    in_channels: int
    out_channels: int
    n_modes1: int  # depth modes
    n_modes2: int  # height modes
    n_modes3: int  # width modes
    linear_conv: bool = True

    def setup(self):
        shape = (self.in_channels, self.out_channels, self.n_modes1, self.n_modes2, self.n_modes3)
        scale = 1 / (self.in_channels * self.out_channels)

        # For 3D rfftn, we need 4 weight tensors to handle the 4 corners in the
        # first two frequency dimensions (depth and height), while width is half-spectrum
        self.weight_1_real = self.param("weight_1_real", uniform(scale=scale), shape)
        self.weight_1_imag = self.param("weight_1_imag", uniform(scale=scale), shape)
        self.weight_2_real = self.param("weight_2_real", uniform(scale=scale), shape)
        self.weight_2_imag = self.param("weight_2_imag", uniform(scale=scale), shape)
        self.weight_3_real = self.param("weight_3_real", uniform(scale=scale), shape)
        self.weight_3_imag = self.param("weight_3_imag", uniform(scale=scale), shape)
        self.weight_4_real = self.param("weight_4_real", uniform(scale=scale), shape)
        self.weight_4_imag = self.param("weight_4_imag", uniform(scale=scale), shape)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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


class SpectralLayers3d(nn.Module):
    """Stack of 3D Spectral Convolution Layers."""

    n_channels: int
    n_modes1: int
    n_modes2: int
    n_modes3: int
    n_layers: int = 4
    activation: Callable = nn.gelu
    norm: Optional[str] = None
    training: bool = True
    linear_conv: bool = True

    def setup(self):
        self.conv_layers = [
            SpectralConv3d(
                in_channels=self.n_channels,
                out_channels=self.n_channels,
                n_modes1=self.n_modes1,
                n_modes2=self.n_modes2,
                n_modes3=self.n_modes3,
                linear_conv=self.linear_conv,
            )
            for _ in range(self.n_layers)
        ]

        self.w_layers = [nn.Conv(features=self.n_channels, kernel_size=(1, 1, 1)) for _ in range(self.n_layers)]

        if self.norm == "layer":
            self.norm_layers = [nn.LayerNorm() for _ in range(self.n_layers)]
        elif self.norm == "batch":
            self.norm_layers = [nn.BatchNorm(use_running_average=not self.training) for _ in range(self.n_layers)]
        else:
            self.norm_layers = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, (conv, w) in enumerate(zip(self.conv_layers, self.w_layers)):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            if self.norm_layers is not None:
                x = self.norm_layers[i](x)
            x = self.activation(x)
        return x


class FNO3D(nn.Module):
    """3D Fourier Neural Operator - JIT compatible."""

    hidden_channels: int
    n_modes: int
    d_vars: int = 1
    linear_conv: bool = True
    n_layers: int = 4
    n_steps: int = 1
    activation: Callable = nn.gelu
    d_model: Tuple[int, int, int] = (32, 32, 32)
    use_positions: bool = False
    norm: Optional[str] = "layer"
    training: bool = True
    dropout_rate: float = 0.0

    def setup(self):
        self.P = nn.Dense(features=self.hidden_channels)

        self.spectral_layers = SpectralLayers3d(
            n_channels=self.hidden_channels,
            n_modes1=self.n_modes,
            n_modes2=self.n_modes,
            n_modes3=self.n_modes,
            n_layers=self.n_layers,
            activation=self.activation,
            norm=self.norm,
            training=self.training,
            linear_conv=self.linear_conv,
        )

        self.Q = nn.Sequential(
            [
                nn.Dense(features=128),
                self.activation,
                nn.Dense(features=self.d_vars * self.n_steps),
            ]
        )

        if self.use_positions:
            self.grid = create_grid_3d(self.d_model[0], self.d_model[1], self.d_model[2])

    def _normalize_input(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
        """Normalize input to (T, D, H, W, C) format."""
        ndim = x.ndim
        if ndim == 3:
            # (D, H, W) -> (1, D, H, W, 1)
            x = x[:, :, :, jnp.newaxis][jnp.newaxis, :, :, :, :]
        elif ndim == 4:
            # (D, H, W, C) -> (1, D, H, W, C)
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

    def advance(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_positions:
            x = jnp.concatenate([x, self.grid], axis=-1)
        x = self.P(x)
        x = self.spectral_layers(x)

        # Optional dropout
        if self.dropout_rate > 0.0:
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not self.training)(x)

        x = self.Q(x)
        return x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x, original_ndim = self._normalize_input(x)
        x = rearrange(x, "t d h w c -> d h w (t c)")
        x = self.advance(x)
        x = rearrange(x, "d h w (t c) -> t d h w c", t=self.n_steps, c=self.d_vars)
        return self._denormalize_output(x, original_ndim)
