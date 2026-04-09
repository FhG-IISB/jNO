"""Shared building blocks for architecture modules.

Centralises BatchNorm, activation lookup, Fourier-mode computation, and
NHWC convolution helpers so they aren't duplicated across architecture files.
"""

from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


# ---------------------------------------------------------------------------
# BatchNorm  (stateless / instance-norm style)
# ---------------------------------------------------------------------------


class BatchNorm(eqx.Module):
    """Simple stateless batch normalization (always uses per-batch statistics)."""

    weight: jnp.ndarray
    bias: jnp.ndarray
    eps: float = eqx.field(static=True)

    def __init__(self, num_features: int, eps: float = 1e-5, **kwargs):
        self.weight = jnp.ones(num_features)
        self.bias = jnp.zeros(num_features)
        self.eps = eps

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        axes = tuple(range(x.ndim - 1))
        mean = jnp.mean(x, axis=axes, keepdims=True)
        var = jnp.var(x, axis=axes, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# Activation lookup
# ---------------------------------------------------------------------------


def get_activation(name: str) -> Optional[Callable]:
    """Return a JAX activation function by name.

    Supported: ``gelu``, ``relu``, ``tanh``, ``elu``, ``leaky_relu``,
    ``sigmoid``, ``silu`` / ``swish``, ``none``.
    """
    activations = {
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "elu": jax.nn.elu,
        "leaky_relu": jax.nn.leaky_relu,
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,
        "none": None,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(activations.keys())}")
    return activations[key]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Flax → Equinox shim
# ---------------------------------------------------------------------------


class FlaxModelWrapper(eqx.Module):
    """Wrap a Flax ``nn.Module`` + its params as a single Equinox module.

    This shim lets any Flax model participate in the Equinox-based jNO
    training loop (``eqx.partition`` / ``eqx.combine``, ``optax`` updates,
    etc.) without rewriting the model.

    The ``params`` dict is stored as the trainable pytree – ``eqx.is_array``
    works on its leaves exactly as it would for native Equinox modules.
    The ``apply_fn`` (typically ``model.apply``) is static so it is never
    serialised or traced by JAX.

    Args:
        apply_fn: The ``model.apply`` callable (Flax-style: params first).
        params: A Flax parameter dict, e.g. the output of ``model.init(...)``.
        default_kwargs: Keyword arguments forwarded to ``apply_fn`` on every
            call (e.g. ``deterministic=True``).

    Example::

        from jax_poseidon import poseidonT, init_poseidon_with_weights
        model, params = poseidonT(rng=key, weight_path="poseidonT.msgpack")
        wrapped = FlaxModelWrapper(model.apply, params, deterministic=True)
        out = wrapped(pixel_values=x, time=t)   # same as model.apply(params, ...)
    """

    apply_fn: Any = eqx.field(static=True)
    params: Any  # trainable Flax param dict
    default_kwargs: dict = eqx.field(static=True)
    post_fn: Optional[Callable] = eqx.field(static=True)

    def __init__(self, apply_fn, params, post_fn=None, **default_kwargs):
        self.apply_fn = apply_fn
        self.params = params
        self.post_fn = post_fn
        self.default_kwargs = default_kwargs

    def __call__(self, *args, **kwargs):
        merged = {**self.default_kwargs, **kwargs}
        merged.pop("key", None)  # drop unused PRNG key from jNO evaluator

        # Auto-cast float inputs to match param dtype (e.g. bfloat16)
        param_dtype = self._param_dtype()
        if param_dtype is not None:
            args = tuple(a.astype(param_dtype) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) and a.dtype != param_dtype else a for a in args)

        result = self.apply_fn(self.params, *args, **merged)
        if self.post_fn is not None:
            result = self.post_fn(result)

        # Cast output back to JAX default float for stable downstream losses.
        out_dtype = _default_float_dtype()
        if param_dtype is not None and param_dtype != out_dtype:
            result = jax.tree_util.tree_map(
                lambda x: x.astype(out_dtype) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
                result,
            )
        return result

    def _param_dtype(self):
        """Return the dtype of the first floating-point parameter, or None."""
        leaves = jax.tree_util.tree_leaves(self.params)
        for leaf in leaves:
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
                return leaf.dtype
        return None


class FlaxNNXWrapper(eqx.Module):
    """Wrap a ``flax.nnx`` module as a single Equinox-compatible module.

    ``flax.nnx`` models carry their own parameters as stateful attributes.
    This wrapper splits the model into:

    * ``graphdef`` — the static computation graph (architecture, shapes,
      variable types).  Stored as an equinox *static* field so it is never
      traced by JAX.
    * ``state`` — an ``nnx.State`` pytree whose leaves are raw JAX arrays.
      ``eqx.is_array`` returns ``True`` for every leaf, so the full
      equinox partition / optimizer / LoRA / mask machinery works
      transparently.

    ``__call__`` reconstructs the live NNX model via ``nnx.merge`` and
    invokes it, keeping the wrapper itself immutable (as required by JAX).

    Args:
        model: A ``flax.nnx.Module`` instance.
        post_fn: Optional callable applied to the model output.
        default_kwargs: Keyword arguments forwarded to ``model.__call__``
            on every invocation.

    Example::

        from flax import nnx
        import jno.numpy as jnn

        class MyNNX(nnx.Module):
            def __init__(self, rngs):
                self.l1 = nnx.Linear(2, 64, rngs=rngs)
                self.l2 = nnx.Linear(64, 1, rngs=rngs)
            def __call__(self, x):
                return self.l2(nnx.relu(self.l1(x)))

        net = jnn.nn.wrap(MyNNX(nnx.Rngs(0)))      # auto-detected
        y   = net(x)

    Partial mask (train only l2, freeze l1)::

        all_false = jax.tree_util.tree_map(lambda _: False, net.module)
        mask = eqx.tree_at(
            lambda w: (
                jax.tree_util.tree_leaves(w.state['l2']['kernel'])[0],
                jax.tree_util.tree_leaves(w.state['l2']['bias'])[0],
            ),
            all_false, (True, True),
        )
        net.mask(mask).optimizer(optax.adam, lr=1e-3)

        # Discover state paths:
        # jax.tree_util.tree_map_with_path(lambda p, _: p, net.module.state)
    """

    graphdef: Any = eqx.field(static=True)
    state: Any  # flax.nnx.State — leaves are raw JAX arrays
    post_fn: Optional[Callable] = eqx.field(static=True)
    default_kwargs: dict = eqx.field(static=True)

    def __init__(self, model, post_fn=None, **default_kwargs):
        try:
            from flax import nnx
        except ImportError as exc:  # pragma: no cover
            raise ImportError("flax is required for FlaxNNXWrapper — install with: pip install flax") from exc
        self.graphdef, self.state = nnx.split(model)
        self.post_fn = post_fn
        self.default_kwargs = default_kwargs

    def __call__(self, *args, **kwargs):
        from flax import nnx

        merged = {**self.default_kwargs, **kwargs}
        merged.pop("key", None)

        model = nnx.merge(self.graphdef, self.state)
        result = model(*args, **merged)

        if self.post_fn is not None:
            result = self.post_fn(result)
        return result

    def _param_count(self) -> int:
        return sum(l.size for l in jax.tree_util.tree_leaves(self.state) if eqx.is_array(l))


# ---------------------------------------------------------------------------
# Fourier-mode computation (used by GeoFNO and PCNO)
# ---------------------------------------------------------------------------


def compute_Fourier_modes(ndims: int, nks: Sequence[int], Ls: Sequence[float]) -> np.ndarray:
    """Compute Fourier mode wave-vectors ``k``.

    Fourier bases are ``cos(k·x)``, ``sin(k·x)``, ``1``.
    We keep only one of each ``±k`` pair.

    Args:
        ndims: Number of spatial dimensions (1, 2, or 3).
        nks: Number of modes per dimension.
        Ls: Domain lengths per dimension.

    Returns:
        ``k_pairs`` of shape ``(nmodes, ndims)`` sorted by magnitude.
    """
    if ndims == 1:
        nk = nks[0]
        Lx = Ls[0]
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(1, nk + 1):
            k_pairs[i, :] = 2 * np.pi / Lx * kx
            k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
            i += 1

    elif ndims == 2:
        nx, ny = nks
        Lx, Ly = Ls
        nk = 2 * nx * ny + nx + ny
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(0, ny + 1):
                if ky == 0 and kx <= 0:
                    continue
                k_pairs[i, :] = 2 * np.pi / Lx * kx, 2 * np.pi / Ly * ky
                k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                i += 1

    elif ndims == 3:
        nx, ny, nz = nks
        Lx, Ly, Lz = Ls
        nk = 4 * nx * ny * nz + 2 * (nx * ny + nx * nz + ny * nz) + nx + ny + nz
        k_pairs = np.zeros((nk, ndims))
        k_pair_mag = np.zeros(nk)
        i = 0
        for kx in range(-nx, nx + 1):
            for ky in range(-ny, ny + 1):
                for kz in range(0, nz + 1):
                    if kz == 0 and (ky < 0 or (ky == 0 and kx <= 0)):
                        continue
                    k_pairs[i, :] = (
                        2 * np.pi / Lx * kx,
                        2 * np.pi / Ly * ky,
                        2 * np.pi / Lz * kz,
                    )
                    k_pair_mag[i] = np.linalg.norm(k_pairs[i, :])
                    i += 1
    else:
        raise ValueError(f"{ndims} in compute_Fourier_modes is not supported")

    k_pairs = k_pairs[np.argsort(k_pair_mag, kind="stable"), :]
    return k_pairs


# ---------------------------------------------------------------------------
# NHWC 2-D Convolution helpers  (used by CNO, MGNO, U-Net, …)
# ---------------------------------------------------------------------------


class Conv2d(eqx.Module):
    """2-D convolution operating on NHWC data."""

    weight: jnp.ndarray  # (kH, kW, in_ch, out_ch)
    bias: Optional[jnp.ndarray]
    padding: str = eqx.field(static=True)
    strides: Tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        strides: Tuple[int, int] = (1, 1),
        padding: str = "SAME",
        use_bias: bool = True,
        *,
        key: jax.Array,
    ):
        kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if not isinstance(kernel_size, int):
            kw = kernel_size[1]
        fan_in = in_channels * kh * kw
        std = 1.0 / jnp.sqrt(fan_in)
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (kh, kw, in_channels, out_channels)) * std
        self.bias = jnp.zeros(out_channels) if use_bias else None
        self.padding = padding
        self.strides = strides

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x[None]
        y = jax.lax.conv_general_dilated(
            x,
            self.weight,
            self.strides,
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias is not None:
            y = y + self.bias
        if was_unbatched:
            y = y[0]
        return y


class ConvTranspose2d(eqx.Module):
    """Transpose 2-D convolution in NHWC layout."""

    weight: jnp.ndarray  # (kH, kW, out_ch, in_ch)
    bias: Optional[jnp.ndarray]
    strides: Tuple[int, int] = eqx.field(static=True)
    padding: str = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        strides: Tuple[int, int] = (2, 2),
        padding: str = "SAME",
        use_bias: bool = False,
        *,
        key: jax.Array,
    ):
        kh = kw = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if not isinstance(kernel_size, int):
            kw = kernel_size[1]
        fan_in = in_channels * kh * kw
        std = 1.0 / jnp.sqrt(fan_in)
        k1, k2 = jax.random.split(key)
        self.weight = jax.random.normal(k1, (kh, kw, out_channels, in_channels)) * std
        self.bias = jnp.zeros(out_channels) if use_bias else None
        self.strides = strides
        self.padding = padding

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        was_unbatched = x.ndim == 3
        if was_unbatched:
            x = x[None]
        y = jax.lax.conv_transpose(
            x,
            self.weight,
            self.strides,
            self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        if self.bias is not None:
            y = y + self.bias
        if was_unbatched:
            y = y[0]
        return y
