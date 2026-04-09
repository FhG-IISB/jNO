# mgno.py - JAX/Equinox implementation of Multigrid Neural Operator

from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
from .common import Conv2d, ConvTranspose2d


# ============================================================
# Convolution helpers (NHWC format)
# ============================================================


def circular_pad_2d(x, pad):
    """Circular (wrap) padding for NHWC or HWC tensors."""
    was_3d = x.ndim == 3
    if was_3d:
        x = x[None]
    x = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="wrap")
    if was_3d:
        x = x[0]
    return x


class CircularConv2d(eqx.Module):
    """Conv2d with circular (wrap) padding."""

    conv: Conv2d
    pad: int = eqx.field(static=True)

    def __init__(self, in_ch, out_ch, kernel_size=3, strides=(1, 1), use_bias=False, *, key):
        self.pad = kernel_size // 2
        self.conv = Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            strides=strides,
            padding="VALID",
            use_bias=use_bias,
            key=key,
        )

    def __call__(self, x, **kwargs):
        x = circular_pad_2d(x, self.pad)
        return self.conv(x)


# ============================================================
# Multigrid building blocks
# ============================================================


class MgIte(eqx.Module):
    """Single multigrid iteration: u = u + S(f - A(u))"""

    A: CircularConv2d
    S: CircularConv2d

    def __init__(self, num_channel_u: int, num_channel_f: int, use_bias: bool = False, *, key):
        kA, kS = jax.random.split(key)
        self.A = CircularConv2d(num_channel_u, num_channel_f, 3, use_bias=use_bias, key=kA)
        self.S = CircularConv2d(num_channel_f, num_channel_u, 3, use_bias=use_bias, key=kS)

    def __call__(self, u: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
        residual = f - self.A(u)
        correction = self.S(residual)
        return u + correction


class MgIteInit(eqx.Module):
    """Initial multigrid iteration: u = S(f)"""

    S: CircularConv2d

    def __init__(self, num_channel_u: int, num_channel_f: int, use_bias: bool = False, *, key):
        self.S = CircularConv2d(num_channel_f, num_channel_u, 3, use_bias=use_bias, key=key)

    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        return self.S(f)


class Restrict(eqx.Module):
    """Restriction operator: coarsens u and f (stride-2 circular conv)."""

    Pi: CircularConv2d
    R: CircularConv2d

    def __init__(self, num_channel_u: int, num_channel_f: int, *, key):
        kPi, kR = jax.random.split(key)
        self.Pi = CircularConv2d(num_channel_u, num_channel_u, 3, strides=(2, 2), use_bias=False, key=kPi)
        self.R = CircularConv2d(num_channel_f, num_channel_f, 3, strides=(2, 2), use_bias=False, key=kR)

    def __call__(self, u: jnp.ndarray, f: jnp.ndarray):
        return self.Pi(u), self.R(f)


class Prolongate(eqx.Module):
    """Prolongation: transpose convolution (stride-2) from coarse to fine."""

    RT: ConvTranspose2d

    def __init__(self, num_channel_u: int, kernel_size: Tuple[int, int], *, key):
        self.RT = ConvTranspose2d(
            num_channel_u,
            num_channel_u,
            kernel_size,
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
            key=key,
        )

    def __call__(self, u_coarse: jnp.ndarray) -> jnp.ndarray:
        return self.RT(u_coarse)


# ============================================================
# MgConv – V-cycle multigrid convolution block
# ============================================================


class MgConv(eqx.Module):
    """
    Multigrid Convolution Block — implements a single V-cycle.

    Downward pass: init + pre-smoothing + restriction at each level.
    Upward pass:   prolongation + post-smoothing at each level.
    """

    num_levels: int = eqx.field(static=True)
    num_iteration: list = eqx.field(static=True)

    # Per-level layers stored as tuples (immutable, pytree-friendly)
    init_layers: list  # length = 1 (only level 0) or 0
    pre_smooth_layers: list  # list of lists of MgIte, one list per level
    restrict_layers: list  # length = num_levels - 1
    prolongate_layers: list  # length = num_levels - 1
    post_smooth_layers: list  # list of lists of MgIte, one list per level

    def __init__(self, input_shape: Tuple[int, int], num_iteration: List[Tuple[int, int]], num_channel_u: int, num_channel_f: int, use_bias: bool = False, *, key):
        num_levels = len(num_iteration)
        self.num_levels = num_levels
        self.num_iteration = [tuple(it) for it in num_iteration]

        # Pre-compute prolongation kernel sizes (same logic as Flax setup)
        kernel_sizes = []
        shape = list(input_shape)
        for j in range(num_levels - 1):
            ks = (4 - shape[0] % 2, 4 - shape[1] % 2)
            kernel_sizes.append(ks)
            shape = [(shape[0] + 2 - 1) // 2, (shape[1] + 2 - 1) // 2]

        # --- allocate all sub-layers ---
        init_layers = []
        pre_smooth_layers = []
        restrict_layers = []
        prolongate_layers = []
        post_smooth_layers = []

        for level in range(num_levels):
            num_pre, num_post = num_iteration[level]
            pre_list = []
            post_list = []

            for i in range(num_pre):
                key, subkey = jax.random.split(key)
                if level == 0 and i == 0:
                    # First pre-smooth at level 0 is MgIteInit
                    init_layers.append(MgIteInit(num_channel_u, num_channel_f, use_bias=use_bias, key=subkey))
                else:
                    pre_list.append(MgIte(num_channel_u, num_channel_f if level == 0 else num_channel_f, use_bias=use_bias, key=subkey))

            for i in range(num_post):
                key, subkey = jax.random.split(key)
                post_list.append(MgIte(num_channel_u, num_channel_f, use_bias=use_bias, key=subkey))

            pre_smooth_layers.append(tuple(pre_list))
            post_smooth_layers.append(tuple(post_list))

            if level < num_levels - 1:
                key, subkey = jax.random.split(key)
                restrict_layers.append(Restrict(num_channel_u, num_channel_f, key=subkey))
                key, subkey = jax.random.split(key)
                prolongate_layers.append(Prolongate(num_channel_u, kernel_sizes[level], key=subkey))

        self.init_layers = tuple(init_layers)  # type: ignore[assignment]
        self.pre_smooth_layers = tuple(pre_smooth_layers)  # type: ignore[assignment]
        self.restrict_layers = tuple(restrict_layers)  # type: ignore[assignment]
        self.prolongate_layers = tuple(prolongate_layers)  # type: ignore[assignment]
        self.post_smooth_layers = tuple(post_smooth_layers)  # type: ignore[assignment]

    def __call__(self, f: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            f: Input features [B, H, W, C_f] or [H, W, C_f]
        Returns:
            Output [B, H, W, C_u] or [H, W, C_u]
        """
        level_outputs = []

        # ===== Downward pass (restriction) =====
        current_f = f
        current_u = None

        for level in range(self.num_levels):
            num_pre, num_post = self.num_iteration[level]

            pre_idx = 0  # index into pre_smooth_layers[level]
            for i in range(num_pre):
                if level == 0 and i == 0 and current_u is None:
                    current_u = self.init_layers[0](current_f)
                else:
                    current_u = self.pre_smooth_layers[level][pre_idx](current_u, current_f)
                    pre_idx += 1

            level_outputs.append((current_u, current_f, num_post))

            if level < self.num_levels - 1:
                current_u, current_f = self.restrict_layers[level](current_u, current_f)

        # ===== Upward pass (prolongation) =====
        for level in range(self.num_levels - 2, -1, -1):
            u_fine, f_fine, num_post = level_outputs[level]
            u_coarse = level_outputs[level + 1][0]

            u_correction = self.prolongate_layers[level](u_coarse)
            current_u = u_fine + u_correction

            for i in range(num_post):
                current_u = self.post_smooth_layers[level][i](current_u, f_fine)

            level_outputs[level] = (current_u, f_fine, num_post)

        return level_outputs[0][0]


# ============================================================
# MgNO – full Multigrid Neural Operator
# ============================================================

_ACTIVATIONS = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
    "silu": jax.nn.silu,
}


class MgNO(eqx.Module):
    """
    Multigrid Neural Operator.

    Architecture:
        1. Multiple MgConv layers with 1x1 skip connections + activation.
        2. Each MgConv implements a V-cycle multigrid.
        3. Final 1x1 projection to *output_dim* channels.
    """

    mgconv_layers: tuple  # MgConv per layer
    linear_layers: tuple  # 1x1 Conv2d skip per layer
    output_proj: Conv2d
    activation: str = eqx.field(static=True)

    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_layer: int = 5,
        num_channel_u: int = 24,
        num_channel_f: int = 3,
        num_iteration: Optional[List[Tuple[int, int]]] = None,
        output_dim: int = 1,
        activation: str = "gelu",
        padding_mode: str = "CIRCULAR",
        *,
        key,
    ):
        if num_iteration is None:
            num_iteration = [(1, 1)] * 5

        self.activation = activation

        mg_list = []
        lin_list = []

        for i in range(num_layer):
            in_ch = num_channel_f if i == 0 else num_channel_u
            key, k1, k2 = jax.random.split(key, 3)
            mg_list.append(MgConv(input_shape=input_shape, num_iteration=num_iteration, num_channel_u=num_channel_u, num_channel_f=in_ch, use_bias=False, key=k1))
            lin_list.append(Conv2d(in_ch, num_channel_u, kernel_size=1, strides=(1, 1), padding="SAME", use_bias=True, key=k2))

        self.mgconv_layers = tuple(mg_list)
        self.linear_layers = tuple(lin_list)

        key, subkey = jax.random.split(key)
        self.output_proj = Conv2d(
            num_channel_u,
            output_dim,
            kernel_size=1,
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
            key=subkey,
        )

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: [B, H, W, C] or [H, W, C]
        Returns:
            [B, H, W, output_dim] or [H, W, output_dim]
        """
        squeeze_batch = False
        if x.ndim == 3:
            x = x[None]
            squeeze_batch = True

        act = _ACTIVATIONS.get(self.activation, jax.nn.gelu)
        u = x
        for mg, lin in zip(self.mgconv_layers, self.linear_layers):
            mg_out = mg(u)
            lin_out = lin(u)
            u = act(mg_out + lin_out)  # type: ignore[operator]

        output = self.output_proj(u)

        if squeeze_batch:
            output = output[0]
        return output


# ============================================================
# MgNO1D – 1-D wrapper
# ============================================================


class MgNO1D(eqx.Module):
    """
    1-D Multigrid Neural Operator.

    Reshapes [B, L, C] -> [B, L, 1, C], runs MgNO, then squeezes back.
    """

    mgno: MgNO

    def __init__(
        self,
        input_length: int,
        num_layer: int = 5,
        num_channel_u: int = 24,
        num_channel_f: int = 3,
        num_iteration: Optional[List[Tuple[int, int]]] = None,
        output_dim: int = 1,
        activation: str = "gelu",
        padding_mode: str = "CIRCULAR",
        *,
        key,
    ):
        if num_iteration is None:
            num_iteration = [(1, 1)] * 5
        self.mgno = MgNO(
            input_shape=(input_length, 1),
            num_layer=num_layer,
            num_channel_u=num_channel_u,
            num_channel_f=num_channel_f,
            num_iteration=num_iteration,
            output_dim=output_dim,
            activation=activation,
            padding_mode=padding_mode,
            key=key,
        )

    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        """
        Args:
            x: [B, L, C] or [L, C]
        Returns:
            [B, L, output_dim] or [L, output_dim]
        """
        squeeze_batch = False
        if x.ndim == 2:
            x = x[None]
            squeeze_batch = True

        # [B, L, C] -> [B, L, 1, C]
        x = x[:, :, None, :]
        output = self.mgno(x, training)
        # [B, L, 1, out] -> [B, L, out]
        output = output[:, :, 0, :]

        if squeeze_batch:
            output = output[0]
        return output
