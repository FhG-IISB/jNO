"""
Neural Operator Factory Module
==============================

This module provides factory methods for creating various neural operator architectures
for learning mappings between function spaces. Neural operators are particularly suited
for solving parametric PDEs, where the goal is to learn the solution operator that maps
input functions (e.g., initial conditions, boundary conditions, coefficients) to output
functions (e.g., PDE solutions).

Architecture Categories
-----------------------

**Spectral Methods** (Grid-based, efficient for regular domains):
    - FNO (1D/2D/3D): Fourier Neural Operator - spectral convolutions via FFT
    - GeoFNO: Geometry-aware FNO for irregular domains
    - PCNO: Point Cloud Neural Operator for unstructured data

**Convolutional Methods** (Multi-scale feature extraction):
    - UNet (1D/2D/3D): Encoder-decoder with skip connections
    - MgNO (1D/2D): Multigrid-inspired architecture

**Attention-Based Methods** (Flexible, handles irregular data):
    - Transformer: Standard attention mechanism
    - PiT: Position-induced Transformer with distance-based attention
    - ScOT: Scalable Operator Transformer (Swin-based)

**Branch-Trunk Methods** (Separate function and coordinate encoding):
    - DeepONet: Deep Operator Network with configurable branch/trunk

**Point-Based Methods** (Unstructured point clouds):
    - PointNet: Permutation-invariant point cloud processing

**Foundation Models** (Pretrained, transfer learning):
    - Poseidon (T/B/L): Pretrained ScOT variants for fluid dynamics

Quick Selection Guide
---------------------

+---------------------------+------------------------------------------+
| Use Case                  | Recommended Architecture                 |
+===========================+==========================================+
| Regular grid, periodic BC | FNO                                      |
| Regular grid, general BC  | UNet, MgNO                               |
| Irregular geometry        | GeoFNO, PCNO, PiT                        |
| Point cloud data          | PointNet, PCNO                           |
| Super-resolution          | FNO, UNet                                |
| Time-stepping             | FNO (n_steps>1), ScOT                    |
| Transfer learning         | Poseidon variants                        |
| Small data regime         | DeepONet, PiT                            |
+---------------------------+------------------------------------------+

Example Usage
-------------

>>> import pino as pnp
>>>
>>> # 2D Darcy flow with FNO
>>> model = pnp.nn.fno2d(
...     hidden_channels=32,
...     n_modes=12,
...     n_layers=4,
...     d_vars=1,
... )
>>>
>>> # Irregular domain with GeoFNO
>>> model = pnp.nn.geofno2d(
...     nks=(16, 16),
...     Ls=(1.0, 1.0),
...     in_dim=3,
...     out_dim=1,
... )
>>>
>>> # Custom Flax module
>>> model = pnp.nn.wrap(MyCustomModule())

References
----------
.. [1] Li et al. "Fourier Neural Operator for Parametric PDEs" (2020)
.. [2] Lu et al. "Learning nonlinear operators via DeepONet" (2021)
.. [3] Li et al. "Geometry-Informed Neural Operator" (2023)
.. [4] Herde et al. "Poseidon: Efficient Foundation Models for PDEs" (2024)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple, Sequence, Literal, List, Union, Any, overload

from .mlp import MLP
from .fno import FNO1D, FNO2D, FNO3D
from .unet import UNet1D, UNet2D, UNet3D
from .pointnet import PointNet
from .deeponet import DeepONet
from .transformer import Transformer
from .pcno import PCNO, compute_Fourier_modes as pcno_compute_Fourier_modes
from .mgno import MgNO, MgNO1D
from .geofno import GeoFNO, compute_Fourier_modes as geofno_compute_Fourier_modes
from .pit import PiT, PiTWithCoords
from .gnot import CGPTNO, GNOT, MoEGPTNO
from .cno import CNO2D
from .common import FlaxModelWrapper, FlaxNNXWrapper
from .linear import Linear
from .lora_linear import LoRALinear

from ..tuner import ArchSpace
from ..trace import Model, TunableModule
from ..utils.config import get_seed


_DEFAULT_NN_KEY: jax.Array | None = None


def set_default_rng_seed(seed: int | None) -> None:
    """Set (or clear) the default PRNG seed used by jno.nn factories."""
    global _DEFAULT_NN_KEY
    if seed is None:
        _DEFAULT_NN_KEY = None
    else:
        _DEFAULT_NN_KEY = jax.random.PRNGKey(int(seed))


def _resolve_key(key: jax.Array | None):
    """Resolve optional key; fall back to seeded global default if available."""
    global _DEFAULT_NN_KEY
    if key is not None:
        return key

    if _DEFAULT_NN_KEY is None:
        seed = get_seed()
        if seed is None:
            seed = 21   # make this consistent with jno.core
        _DEFAULT_NN_KEY = jax.random.PRNGKey(int(seed))

    if _DEFAULT_NN_KEY is None:
        raise ValueError("No PRNG key provided. Pass key=... explicitly, or set [jno].seed " "and call jno.setup(...) before creating models.")

    _DEFAULT_NN_KEY, key_out = jax.random.split(_DEFAULT_NN_KEY)
    return key_out


def parameter(shape: tuple, *, key: jax.Array | None = None, init: Callable = jax.nn.initializers.zeros, name: str = "value") -> Model:
    """
    Create a trainable parameter tensor.

    Useful for learning scalar or tensor constants during optimization,
    such as PDE coefficients or normalization factors.

    Args:
        shape: Shape of the parameter tensor.
        key: JAX PRNG key for initialization.
        init: Initializer function ``(key, shape) -> array``. Default: zeros.
        name: Name of the parameter (used for display). Default: "value".

    Returns:
        Model: A wrapped module that returns the parameter when called.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> # Learnable diffusion coefficient
        >>> D = parameter((1,), key=key, init=jax.nn.initializers.ones)
        >>>
        >>> # Learnable 2D tensor
        >>> K = parameter((3, 3), key=key)
    """

    class _Parameter(eqx.Module):
        value: jnp.ndarray

        def __call__(self):
            return self.value

    key = _resolve_key(key)
    return nn.wrap(_Parameter(value=init(key, shape)))


class nn:
    """
    Neural network factory class providing unified interface for model creation.

    This class contains class methods for instantiating various neural operator
    architectures with sensible defaults. All methods return a `Model`
    wrapper that integrates with the pino training pipeline.

    The factory pattern allows:
    - Consistent API across different architectures
    - Sensible defaults for common use cases
    - Easy hyperparameter tuning via keyword arguments
    - Automatic wrapping for pipeline integration

    Categories of Methods
    ---------------------
    - `wrap()`: Wrap arbitrary Flax modules
    - `mlp()`: Multi-layer perceptrons
    - `fno1d/2d/3d()`: Fourier Neural Operators
    - `unet1d/2d/3d()`: U-Net architectures
    - `geofno/geofno1d/2d/3d()`: Geometry-aware FNO
    - `pcno()`: Point Cloud Neural Operator
    - `mgno1d/2d()`: Multigrid Neural Operator
    - `pit()`: Position-induced Transformer
    - `deeponet()`: Deep Operator Network
    - `pointnet()`: PointNet for point clouds
    - `transformer()`: Standard transformer
    - `scot()`: Scalable Operator Transformer
    - `poseidonT/B/L()`: Pretrained foundation models

    """

    def __new__(cls, module=None, *args, **kwargs):
        """Alias ``nn(module, ...)`` to ``nn.wrap(module, ...)``.

        This keeps the fluent factory API while allowing direct callable usage.
        Calling ``nn()`` with no arguments still creates an instance.
        """
        if module is None and not args and not kwargs:
            return super().__new__(cls)
        return cls.wrap(module, *args, **kwargs)

    @staticmethod
    def linear(in_features: int, out_features: int, use_bias: bool = True, *, key: Any = None):
        return Linear(in_features, out_features, use_bias, key=_resolve_key(key))

    @staticmethod
    def _resolve_key(key):
        return _resolve_key(key)

    # =========================================================================
    # Core Wrapping Methods
    # =========================================================================

    @overload
    @classmethod
    def wrap(cls, module, space: None = ..., name: str = ..., weight_path: str = ...) -> "Model": ...
    @overload
    @classmethod
    def wrap(cls, module, space: "ArchSpace", name: str = ..., weight_path: str = ...) -> "TunableModule": ...

    @classmethod
    def wrap(cls, module, space: "ArchSpace" = None, name: str = "", weight_path: str = None) -> Union["Model", "TunableModule"]:
        """
        Wrap a module for use in the jno pipeline.

        This is the primary method for integrating custom architectures into
        the jno framework. It handles both standard wrapping and architecture
        search scenarios.

        Args:
            module: An ``eqx.Module`` instance (for standard use), a legacy
                Flax ``nn.Module``, or a class (for architecture search).
            space: Optional ``ArchSpace`` for hyperparameter tuning. When
                provided, ``module`` must be a class, not an instance.
            name: Optional display name.
            weight_path: Optional path to pretrained weights.

        Returns:
            Model: Standard wrapped module (when space=None).
            TunableModule: Tunable module for architecture search (when space provided).

        Raises:
            ValueError: If ``space`` is provided but ``module`` is an instance.

        Example:
            >>> # Wrap a custom equinox module
            >>> class MyOperator(eqx.Module):
            ...     linear: eqx.nn.Linear
            ...     def __init__(self, in_dim, out_dim, *, key):
            ...         self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)
            ...     def __call__(self, x):
            ...         return self.linear(x)
            >>>
            >>> model = nn.wrap(MyOperator(2, 1, key=jax.random.PRNGKey(0)))
        """
        if space is not None:
            if isinstance(module, type):
                return TunableModule(module_cls=module, space=space)
            else:
                raise ValueError("When space= is provided, module must be a CLASS (not instance). " "Use: pnp.nn.wrap(MLP, space=space) not pnp.nn.wrap(MLP(), space=space)")
        else:
            # Auto-detect flax.nnx modules and wrap in FlaxNNXWrapper so the
            # equinox partition / optimizer / LoRA / mask machinery works.
            try:
                from flax import nnx as _nnx

                if isinstance(module, _nnx.Module):
                    module = FlaxNNXWrapper(module)
            except ImportError:
                pass
            return Model(module, name, weight_path)

    @classmethod
    def flaxwrap(cls, module, input, key=None) -> Model:
        key = cls._resolve_key(key)
        params = module.init({"params": key}, *input)
        wrapped = FlaxModelWrapper(module.apply, params)
        return cls.wrap(wrapped)

    # =========================================================================
    # Basic Architectures
    # =========================================================================

    @classmethod
    def mlp(
        cls,
        in_features: int,
        output_dim: int = 1,
        activation: Callable = jnp.tanh,
        hidden_dims: Union[int, Sequence[int]] = 64,
        num_layers: int = 2,
        output_activation: Optional[Callable] = None,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        final_layer_bias: bool = True,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a Multi-Layer Perceptron (MLP).

        A fully-connected feedforward network suitable for learning point-wise
        mappings. Often used as a baseline or as components within larger
        architectures (e.g., projection layers in FNO).

        Architecture::

            Input → [Dense → Norm → Activation → Dropout] × N → Dense → Output

        Args:
            in_features: Number of input features (last axis of concatenated
                inputs).
            output_dim: Number of output features. Default: 1.
            activation: Activation function for hidden layers.
                Common choices: ``jax.nn.relu``, ``jax.nn.gelu``,
                ``jnp.tanh``, ``jax.nn.silu``.
                Default: ``jnp.tanh``.
            hidden_dims: Width of hidden layers. Can be:
                - ``int``: All layers have the same width.
                - ``Sequence[int]``: Specify width per layer.

                Default: 64.
            num_layers: Number of hidden layers. Ignored if ``hidden_dims`` is
                a sequence. Default: 2.
            output_activation: Optional activation for the output layer.
                Use ``jax.nn.sigmoid`` for [0,1] outputs, ``jax.nn.softplus``
                for positive outputs. Default: None (linear output).
            use_bias: Include bias in hidden layers. Default: True.
            dropout_rate: Dropout probability (0 = disabled). Default: 0.0.
            batch_norm: Apply batch normalization. Default: False.
            layer_norm: Apply layer normalization. Default: False.
            final_layer_bias: Include bias in output layer. Default: True.
            key: JAX PRNG key for weight initialization.

        Returns:
            Model: Wrapped MLP model.

        Raises:
            ValueError: If both `batch_norm` and `layer_norm` are True.

        Example:
            >>> key = jax.random.PRNGKey(0)
            >>> # Simple regression MLP
            >>> model = nn.mlp(2, output_dim=1, hidden_dims=64, num_layers=3, key=key)
            >>>
            >>> # Classification with softmax output
            >>> model = nn.mlp(
            ...     10,
            ...     output_dim=10,
            ...     activation=jax.nn.relu,
            ...     hidden_dims=[256, 128, 64],
            ...     output_activation=jax.nn.softmax,
            ...     layer_norm=True,
            ...     key=key,
            ... )
            >>>
            >>> # Positive output (e.g., variance prediction)
            >>> model = nn.mlp(2, output_dim=1, output_activation=jax.nn.softplus, key=key)

        Note:
            MLPs concatenate all inputs along the feature axis, making them
            suitable for conditional models where auxiliary information
            (e.g., parameters, coordinates) is provided alongside the main input.
        """
        if batch_norm and layer_norm:
            raise ValueError("Cannot use both batch_norm and layer_norm simultaneously.")

        if isinstance(hidden_dims, int):
            layer_widths = [hidden_dims] * num_layers
        else:
            layer_widths = list(hidden_dims)
            if len(layer_widths) != num_layers and num_layers != 2:
                raise ValueError(f"Length of hidden_dims ({len(layer_widths)}) must match " f"num_layers ({num_layers}) when both are specified.")

        inst = MLP(
            in_features,
            output_dim,
            activation,
            hidden_dims,
            num_layers,
            output_activation,
            use_bias,
            dropout_rate,
            batch_norm,
            layer_norm,
            final_layer_bias,
            key=cls._resolve_key(key),
        )

        return cls.wrap(inst)

    # =========================================================================
    # Convolutional Neural Operators
    # =========================================================================

    @classmethod
    def cno2d(
        cls,
        in_dim: int = 1,
        out_dim: int = 1,
        size: int = 64,
        N_layers: int = 3,
        N_res: int = 4,
        N_res_neck: int = 4,
        channel_multiplier: int = 16,
        use_bn: bool = True,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 2D Continuous Neural Operator (CNO).

        CNO is a U-Net style architecture that uses continuous convolutions
        with bicubic interpolation, enabling resolution-invariant learning
        of operators between function spaces.

        Architecture::

            Input → Lift → [Encoder + ResNet] × N → Bottleneck → [Decoder + Skip] × N → Project → Output

        Key Features:
            - Resolution-invariant through bicubic interpolation
            - Activation functions applied at 2x resolution then downsampled
            - Skip connections for multi-scale information flow
            - Configurable depth via N_layers
            - ResNet blocks at each encoder level

        The continuous activation is implemented as:
            1. Upsample to 2× resolution (bicubic)
            2. Apply LeakyReLU
            3. Downsample to target resolution (bicubic)

        This approach helps preserve high-frequency information that might
        otherwise be lost due to aliasing.

        Args:
            in_dim: Number of input channels. Default: 1.
            out_dim: Number of output channels. Default: 1.
            size: Spatial size of input/output (assumes square). Must be
                divisible by 2^N_layers. Default: 64.
            N_layers: Number of encoder/decoder levels. Controls the depth
                of the U-Net structure. Default: 3.
            N_res: Number of residual blocks per encoder level. Default: 4.
            N_res_neck: Number of residual blocks in the bottleneck. Default: 4.
            channel_multiplier: Base channel count. Channels at level i are
                2^i × channel_multiplier. Default: 16.
            use_bn: Whether to use batch normalization. Default: True.
            training: Training mode flag (affects batch norm). Default: True.

        Returns:
            Model: Wrapped CNO2D model.

        Example:
            >>> # Darcy flow: permeability → pressure
            >>> model = nn.cno2d(
            ...     in_dim=1,
            ...     out_dim=1,
            ...     size=64,
            ...     N_layers=3,
            ...     channel_multiplier=16,
            ... )
            >>>
            >>> # Multi-channel with deeper network
            >>> model = nn.cno2d(
            ...     in_dim=3,
            ...     out_dim=2,
            ...     size=128,
            ...     N_layers=4,
            ...     N_res=6,
            ...     channel_multiplier=32,
            ... )

        Note:
            - Input shape: (batch, H, W, in_dim) - NHWC format
            - Output shape: (batch, H, W, out_dim) - NHWC format
            - Spatial size must be divisible by 2^N_layers
            - Memory scales with channel_multiplier² and 2^N_layers

        Reference:
            Raonić et al., "Convolutional Neural Operators for robust and
            accurate learning of PDEs"
            https://github.com/bogdanraonic3/AI_Science_Engineering

        See Also:
            - `unet2d`: Similar architecture without continuous activations
            - `fno2d`: Spectral approach for periodic domains

        """
        model = CNO2D(
            in_dim=in_dim,
            out_dim=out_dim,
            size=size,
            N_layers=N_layers,
            N_res=N_res,
            N_res_neck=N_res_neck,
            channel_multiplier=channel_multiplier,
            use_bn=use_bn,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    # =========================================================================
    # Fourier Neural Operators
    # =========================================================================

    @classmethod
    def fno1d(
        cls,
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
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 1D Fourier Neural Operator.

        FNO learns operators in Fourier space, where convolutions become
        efficient element-wise multiplications. Particularly effective for
        problems with periodic boundary conditions or smooth solutions.

        Architecture::

            Input → Lift → [SpectralConv + Conv1D → Activation] × N → Project → Output

        Args:
            hidden_channels: Number of channels in spectral layers.
                Typical values: 32-128.
            n_modes: Number of Fourier modes to retain. Higher values capture
                finer details but increase computation. Typical: 16-64.
            d_vars: Number of output variables/channels. Default: 1.
            linear_conv: If True, use linear (non-circular) convolution by
                zero-padding. Set False for periodic domains. Default: True.
            n_layers: Number of spectral convolution layers. Default: 4.
            n_steps: Number of output time steps (for autoregressive rollout).
                Default: 1.
            activation: Activation function. Default: `jax.nn.gelu`.
            norm: Normalization type ('layer', 'batch', 'instance', None).
                Default: None.
            training: Training mode flag. Default: True.
            dropout_rate: Dropout probability. Default: 0.0.

        Returns:
            Model: Wrapped FNO1D model.

        Example:
            >>> # Burgers equation: u(x,0) → u(x,T)
            >>> model = nn.fno1d(
            ...     hidden_channels=64,
            ...     n_modes=16,
            ...     d_vars=1,
            ...     n_layers=4,
            ... )
            >>>
            >>> # Multi-step prediction
            >>> model = nn.fno1d(
            ...     hidden_channels=32,
            ...     n_modes=24,
            ...     n_steps=10,  # Predict 10 time steps
            ... )

        Note:
            - Input shape: (batch, length, channels)
            - Output shape: (batch, length, d_vars) or (batch, n_steps, length, d_vars)
            - For non-periodic BCs, set `linear_conv=True`

        """
        norm_type = norm[0] if isinstance(norm, tuple) else norm  # type: ignore[index]

        model_instance = FNO1D(
            in_features=in_features,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            d_vars=d_vars,
            linear_conv=linear_conv,
            n_layers=n_layers,
            n_steps=n_steps,
            activation=activation,
            norm=norm_type,
            training=training,
            dropout_rate=dropout_rate,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model_instance)

    @classmethod
    def fno2d(
        cls,
        in_features: int,
        hidden_channels: int = 32,
        n_modes: int = 16,
        d_vars: int = 1,
        n_layers: int = 4,
        n_steps: int = 1,
        d_model: Tuple[int, int] = (64, 64),
        activation: Callable = jax.nn.gelu,
        norm: Optional[str] = "layer",
        training: bool = True,
        use_positions: bool = False,
        linear_conv: bool = True,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 2D Fourier Neural Operator.

        The workhorse architecture for 2D PDE problems on regular grids.
        Efficiently captures global dependencies through spectral convolutions.

        Architecture::

            Input → Lift → [SpectralConv2D + Conv2D → Norm → Activation] × N → Project → Output

        Args:
            hidden_channels: Channels in spectral layers. Default: 32.
            n_modes: Fourier modes per dimension. Total modes = n_modes².
                Default: 16.
            d_vars: Output channels. Default: 1.
            n_layers: Number of spectral layers. Default: 4.
            n_steps: Output time steps. Default: 1.
            d_model: Spatial dimensions (H, W). Used for positional encoding.
                Default: (64, 64).
            activation: Activation function. Default: `jax.nn.gelu`.
            norm: Normalization ('layer', 'batch', 'instance', None).
                Default: 'layer'.
            training: Training mode. Default: True.
            use_positions: Concatenate coordinate grid to input. Default: False.
            linear_conv: Non-circular convolution (for non-periodic BC).
                Default: True.

        Returns:
            Model: Wrapped FNO2D model.

        Example:
            >>> # Darcy flow: permeability → pressure
            >>> model = nn.fno2d(
            ...     hidden_channels=32,
            ...     n_modes=12,
            ...     d_vars=1,
            ...     n_layers=4,
            ... )
            >>>
            >>> # Navier-Stokes: (u, v, p) → (u, v, p) at next time
            >>> model = nn.fno2d(
            ...     hidden_channels=64,
            ...     n_modes=20,
            ...     d_vars=3,
            ...     use_positions=True,
            ... )

        Note:
            - Input shape: (batch, H, W, channels)
            - Output shape: (batch, H, W, d_vars)
            - Memory scales as O(n_modes² × hidden_channels²)

        """
        fno = FNO2D(
            in_features=in_features,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            d_vars=d_vars,
            n_layers=n_layers,
            n_steps=n_steps,
            d_model=d_model,
            activation=activation,
            norm=norm,
            training=training,
            use_positions=use_positions,
            linear_conv=linear_conv,
            key=cls._resolve_key(key),
        )

        return cls.wrap(fno)

    @classmethod
    def fno3d(
        cls,
        in_features: int,
        hidden_channels: int = 32,
        n_modes: int = 12,
        d_vars: int = 1,
        n_layers: int = 4,
        n_steps: int = 1,
        d_model: Tuple[int, int, int] = (32, 32, 32),
        activation: Callable = jax.nn.gelu,
        norm: Optional[str] = "layer",
        training: bool = True,
        use_positions: bool = False,
        linear_conv: bool = True,
        dropout_rate: float = 0.0,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 3D Fourier Neural Operator.

        For spatiotemporal problems or 3D spatial domains. Note that memory
        requirements scale cubically with resolution.

        Args:
            hidden_channels: Channels in spectral layers. Default: 32.
            n_modes: Fourier modes per dimension. Default: 12.
            d_vars: Output channels. Default: 1.
            n_layers: Number of spectral layers. Default: 4.
            n_steps: Output time steps. Default: 1.
            d_model: Spatial dimensions (D, H, W). Default: (32, 32, 32).
            activation: Activation function. Default: `jax.nn.gelu`.
            norm: Normalization type. Default: 'layer'.
            training: Training mode. Default: True.
            use_positions: Add coordinate grid to input. Default: False.
            linear_conv: Non-circular convolution. Default: True.
            dropout_rate: Dropout probability. Default: 0.0.

        Returns:
            Model: Wrapped FNO3D model.

        Example:
            >>> # 3D elasticity
            >>> model = nn.fno3d(
            ...     hidden_channels=24,
            ...     n_modes=8,
            ...     d_vars=3,  # displacement (u, v, w)
            ... )

        Warning:
            Memory usage is O(n_modes³ × hidden_channels²). For large problems,
            consider using factorized variants or reducing n_modes.
        """
        fno = FNO3D(
            in_features=in_features,
            hidden_channels=hidden_channels,
            n_modes=n_modes,
            d_vars=d_vars,
            n_layers=n_layers,
            n_steps=n_steps,
            d_model=d_model,
            activation=activation,
            norm=norm,
            training=training,
            use_positions=use_positions,
            linear_conv=linear_conv,
            dropout_rate=dropout_rate,
            key=cls._resolve_key(key),
        )

        return cls.wrap(fno)

    # =========================================================================
    # Geometry-Aware Operators
    # =========================================================================

    @classmethod
    def geofno(
        cls,
        ndims: int,
        nks: Sequence[int],
        Ls: Sequence[float],
        layers: Sequence[int] = (64, 64, 64, 64),
        fc_dim: int = 128,
        in_dim: int = 3,
        out_dim: int = 1,
        act: str = "gelu",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a Geometry-aware Fourier Neural Operator.

        GeoFNO extends FNO to irregular geometries by computing explicit
        Fourier bases on mesh nodes rather than using FFT. This allows
        handling of complex domains, unstructured meshes, and varying
        node counts.

        Architecture::

            Input → Lift → [GeoSpectralConv → Activation] × N → Project → Output

        The spectral convolution uses explicit Fourier transforms:
            - Forward: F(u) = Σᵢ u(xᵢ) × exp(-i k·xᵢ) × wᵢ
            - Inverse: u(x) = Σₖ F̂(k) × exp(i k·x)

        Args:
            ndims: Spatial dimensionality (1, 2, or 3).
            nks: Fourier modes per dimension. For 2D: [nx, ny].
            Ls: Domain lengths per dimension. For 2D: [Lx, Ly].
            layers: Channel dimensions for each layer. Default: (64, 64, 64, 64).
            fc_dim: Hidden dimension for projection MLPs. 0 = no hidden layer.
                Default: 128.
            in_dim: Input channels. Typically ndims + 1 (field + coordinates).
                Default: 3.
            out_dim: Output channels. Default: 1.
            act: Activation function name. Default: 'gelu'.

        Returns:
            Model: Wrapped GeoFNO model.

        Example:
            >>> # 2D problem on irregular domain
            >>> model = nn.geofno(
            ...     ndims=2,
            ...     nks=[12, 12],
            ...     Ls=[1.0, 1.0],
            ...     in_dim=3,  # (field, x, y)
            ...     out_dim=1,
            ... )

        Note:
            The model expects auxiliary inputs during forward pass:
            - `node_mask`: [batch, max_nodes, 1] - Valid node indicator
            - `nodes`: [batch, max_nodes, ndims] - Node coordinates
            - `node_weights`: [batch, max_nodes, 1] - Integration weights

        See Also:
            - `geofno1d`, `geofno2d`, `geofno3d`: Dimension-specific shortcuts
            - `pcno`: Alternative for point cloud data

        """
        modes = geofno_compute_Fourier_modes(ndims, list(nks), list(Ls))
        modes = jnp.array(modes)  # type: ignore[assignment]

        model = GeoFNO(
            ndims=ndims,
            modes=modes,  # type: ignore[arg-type]
            layers=list(layers),
            fc_dim=fc_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            act=act,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    @classmethod
    def geofno1d(
        cls,
        nk: int = 16,
        L: float = 1.0,
        layers: Sequence[int] = (64, 64, 64, 64),
        fc_dim: int = 128,
        in_dim: int = 2,
        out_dim: int = 1,
        act: str = "gelu",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 1D Geometry-aware FNO.

        Convenience wrapper for `geofno()` with ndims=1.

        Args:
            nk: Number of Fourier modes. Default: 16.
            L: Domain length. Default: 1.0.
            layers: Channel dimensions. Default: (64, 64, 64, 64).
            fc_dim: Projection hidden dimension. Default: 128.
            in_dim: Input channels (typically field + x). Default: 2.
            out_dim: Output channels. Default: 1.
            act: Activation function. Default: 'gelu'.

        Returns:
            Model: Wrapped 1D GeoFNO model.
        """
        return cls.geofno(
            ndims=1,
            nks=[nk],
            Ls=[L],
            layers=layers,
            fc_dim=fc_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            act=act,
            key=cls._resolve_key(key),
        )

    @classmethod
    def geofno2d(
        cls,
        nks: Tuple[int, int] = (12, 12),
        Ls: Tuple[float, float] = (1.0, 1.0),
        layers: Sequence[int] = (64, 64, 64, 64),
        fc_dim: int = 128,
        in_dim: int = 3,
        out_dim: int = 1,
        act: str = "gelu",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 2D Geometry-aware FNO.

        Convenience wrapper for `geofno()` with ndims=2.

        Args:
            nks: Fourier modes (nx, ny). Default: (12, 12).
            Ls: Domain lengths (Lx, Ly). Default: (1.0, 1.0).
            layers: Channel dimensions. Default: (64, 64, 64, 64).
            fc_dim: Projection hidden dimension. Default: 128.
            in_dim: Input channels. Default: 3.
            out_dim: Output channels. Default: 1.
            act: Activation function. Default: 'gelu'.

        Returns:
            Model: Wrapped 2D GeoFNO model.

        Example:
            >>> # Flow around airfoil (irregular mesh)
            >>> model = nn.geofno2d(
            ...     nks=(16, 16),
            ...     Ls=(2.0, 1.0),  # Rectangular domain
            ...     in_dim=4,  # (rho, u, v, coordinates)
            ...     out_dim=3,  # (rho, u, v)
            ... )
        """
        return cls.geofno(
            ndims=2,
            nks=list(nks),
            Ls=list(Ls),
            layers=layers,
            fc_dim=fc_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            act=act,
            key=cls._resolve_key(key),
        )

    @classmethod
    def geofno3d(
        cls,
        nks: Tuple[int, int, int] = (8, 8, 8),
        Ls: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        layers: Sequence[int] = (64, 64, 64, 64),
        fc_dim: int = 128,
        in_dim: int = 4,
        out_dim: int = 1,
        act: str = "gelu",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 3D Geometry-aware FNO.

        Convenience wrapper for `geofno()` with ndims=3.

        Args:
            nks: Fourier modes (nx, ny, nz). Default: (8, 8, 8).
            Ls: Domain lengths (Lx, Ly, Lz). Default: (1.0, 1.0, 1.0).
            layers: Channel dimensions. Default: (64, 64, 64, 64).
            fc_dim: Projection hidden dimension. Default: 128.
            in_dim: Input channels. Default: 4.
            out_dim: Output channels. Default: 1.
            act: Activation function. Default: 'gelu'.

        Returns:
            Model: Wrapped 3D GeoFNO model.
        """
        return cls.geofno(
            ndims=3,
            nks=list(nks),
            Ls=list(Ls),
            layers=layers,
            fc_dim=fc_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            act=act,
            key=cls._resolve_key(key),
        )

    @classmethod
    def pcno(
        cls,
        ndims: int,
        nks: Sequence[int],
        Ls: Sequence[float],
        layers: Sequence[int] = (64, 64, 64, 64),
        fc_dim: int = 128,
        in_dim: int = 3,
        out_dim: int = 1,
        inv_L_scale_min: float = 0.5,
        inv_L_scale_max: float = 2.0,
        train_inv_L_scale: bool = True,
        act: str = "gelu",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a Point Cloud Neural Operator.

        PCNO is designed for unstructured point cloud data with varying
        point counts. It uses explicit Fourier bases with learnable
        length scales, making it adaptive to different domain sizes.

        Key Features:
            - Handles variable-size point clouds (with masking)
            - Learnable domain length scales
            - Multiple measures for multi-resolution processing

        Args:
            ndims: Spatial dimensionality (1, 2, or 3).
            nks: Fourier modes per dimension per measure.
                Length = ndims × nmeasures.
            Ls: Domain lengths per dimension per measure.
                Length = ndims × nmeasures.
            layers: Channel dimensions per layer. Default: (64, 64, 64, 64).
            fc_dim: Projection hidden dimension (0 = no hidden layer).
                Default: 128.
            in_dim: Input channels. Default: 3.
            out_dim: Output channels. Default: 1.
            inv_L_scale_min: Minimum learnable length scale. Default: 0.5.
            inv_L_scale_max: Maximum learnable length scale. Default: 2.0.
            train_inv_L_scale: Whether to learn length scales. Default: True.
            act: Activation function. Default: 'gelu'.

        Returns:
            Model: Wrapped PCNO model.

        Example:
            >>> # 2D point cloud with 16 modes
            >>> model = nn.pcno(
            ...     ndims=2,
            ...     nks=[16, 16],
            ...     Ls=[1.0, 1.0],
            ...     in_dim=3,  # (field, x, y)
            ...     out_dim=1,
            ... )
            >>>
            >>> # Multi-resolution with 2 measures
            >>> model = nn.pcno(
            ...     ndims=2,
            ...     nks=[8, 8, 16, 16],  # coarse + fine
            ...     Ls=[1.0, 1.0, 1.0, 1.0],
            ...     train_inv_L_scale=True,
            ... )

        Reference:
            "Point Cloud Neural Operator" - PKU-CMEGroup
            https://github.com/PKU-CMEGroup/NeuralOperator
        """
        modes = pcno_compute_Fourier_modes(ndims, nks, Ls)
        modes = jnp.array(modes)  # type: ignore[assignment]
        nmeasures = len(nks) // ndims

        model = PCNO(
            ndims=ndims,
            modes=modes,  # type: ignore[arg-type]
            nmeasures=nmeasures,
            layers=list(layers),
            fc_dim=fc_dim,
            in_dim=in_dim,
            out_dim=out_dim,
            inv_L_scale_min=inv_L_scale_min,
            inv_L_scale_max=inv_L_scale_max,
            train_inv_L_scale=train_inv_L_scale,
            act=act,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    # =========================================================================
    # General Neural Operator Transformer (GNOT)
    # =========================================================================

    @classmethod
    def cgptno(
        cls,
        trunk_size: int,
        branch_sizes: Optional[List[int]] = None,
        output_size: int = 1,
        n_layers: int = 2,
        n_hidden: int = 64,
        n_head: int = 1,
        n_inner: int = 4,
        mlp_layers: int = 2,
        attn_type: str = "linear",
        act: str = "gelu",
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        horiz_fourier_dim: int = 0,
        *,
        key: jax.Array | None = None,
    ) -> "Model":
        """
        Create a Cross-attention GPT Neural Operator (CGPTNO).

        CGPTNO uses linear cross-attention to map from input functions
        (sampled at sensor points) to output functions (at query points).
        It handles arbitrary geometries and multiple input functions.

        Architecture::

            Trunk (queries) ──→ MLP ──→ ┐
                                        ├──→ CrossAttn ──→ SelfAttn ──→ ... ──→ MLP ──→ Output
            Branch (inputs) ──→ MLP ──→ ┘

        Key Features:
            - Linear O(n) attention complexity
            - Multiple input function support
            - Arbitrary query/sensor point locations
            - Optional Fourier feature embedding

        Args:
            trunk_size: Input dimension for trunk (query points).
                Typically: space_dim + n_parameters.
            branch_sizes: List of input dimensions for each branch.
                Each branch represents a different input function.
                If None, uses self-attention only.
            output_size: Output dimension. Default: 1.
            n_layers: Number of transformer layers. Default: 2.
            n_hidden: Hidden dimension. Default: 64.
            n_head: Number of attention heads. Default: 1.
            n_inner: FFN inner dimension multiplier. Default: 4.
            mlp_layers: Layers in embedding MLPs. Default: 2.
            attn_type: Attention type ('linear'). Default: 'linear'.
            act: Activation function. Default: 'gelu'.
            ffn_dropout: FFN dropout rate. Default: 0.0.
            attn_dropout: Attention dropout rate. Default: 0.0.
            horiz_fourier_dim: Fourier embedding octaves (0 = disabled).
                When > 0, expands features using sin/cos at multiple frequencies.
                Default: 0.
            deterministic: Disable dropout (for inference). Default: True.

        Returns:
            Model: Wrapped CGPTNO model.

        Example:
            >>> # 2D Darcy flow with single input
            >>> model = nn.cgptno(
            ...     trunk_size=4,  # (x, y, param1, param2)
            ...     branch_sizes=[3],  # (a(x,y), x, y)
            ...     output_size=1,
            ...     n_layers=3,
            ...     n_hidden=128,
            ... )
            >>>
            >>> # Multi-physics with two inputs
            >>> model = nn.cgptno(
            ...     trunk_size=3,
            ...     branch_sizes=[3, 2],  # Two input functions
            ...     output_size=2,
            ... )

        Note:
            - Input x_trunk: [batch, n_query, trunk_size]
            - Input x_branches: List of [batch, n_sensors_i, branch_size_i]
            - Output: [batch, n_query, output_size]

        Reference:
            Hao et al. "GNOT: A General Neural Operator Transformer" (2023)
        """
        model = CGPTNO(
            trunk_size=trunk_size,
            branch_sizes=branch_sizes,
            output_size=output_size,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            n_inner=n_inner,
            mlp_layers=mlp_layers,
            attn_type=attn_type,
            act=act,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            horiz_fourier_dim=horiz_fourier_dim,
            key=cls._resolve_key(key),
        )
        return cls.wrap(model)

    @classmethod
    def gnot(
        cls,
        trunk_size: int,
        branch_sizes: List[int],
        space_dim: int = 2,
        output_size: int = 1,
        n_layers: int = 2,
        n_hidden: int = 64,
        n_head: int = 1,
        n_experts: int = 2,
        n_inner: int = 4,
        mlp_layers: int = 2,
        attn_type: str = "linear",
        act: str = "gelu",
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        horiz_fourier_dim: int = 0,
        *,
        key: jax.Array | None = None,
    ) -> "Model":
        """
        Create a General Neural Operator Transformer (GNOT).

        GNOT extends CGPTNO with Mixture-of-Experts (MoE) FFN layers,
        enabling position-dependent transformations. A gating network
        routes inputs to different expert networks based on spatial position.

        Architecture::

            Trunk ──→ MLP ──→ ┐
                              ├──→ CrossAttn ──→ MoE-FFN ──→ SelfAttn ──→ MoE-FFN ──→ ...
            Branch ──→ MLP ──→ ┘
                                        ↑
            Position ──→ GateNet ──→ Expert Weights

        Key Features:
            - Position-dependent expert routing
            - Learns spatially-varying transformations
            - Multiple input function support
            - Linear attention complexity

        Args:
            trunk_size: Input dimension for trunk.
            branch_sizes: List of input dimensions for each branch.
            space_dim: Spatial dimension for gating (first N columns of trunk).
                Default: 2.
            output_size: Output dimension. Default: 1.
            n_layers: Number of transformer layers. Default: 2.
            n_hidden: Hidden dimension. Default: 64.
            n_head: Number of attention heads. Default: 1.
            n_experts: Number of expert networks in MoE. Default: 2.
            n_inner: FFN inner dimension multiplier. Default: 4.
            mlp_layers: Layers in embedding MLPs. Default: 2.
            attn_type: Attention type. Default: 'linear'.
            act: Activation function. Default: 'gelu'.
            ffn_dropout: FFN dropout rate. Default: 0.0.
            attn_dropout: Attention dropout rate. Default: 0.0.
            horiz_fourier_dim: Fourier embedding octaves. Default: 0.
            deterministic: Disable dropout. Default: True.

        Returns:
            Model: Wrapped GNOT model.

        Example:
            >>> # 2D problem with MoE
            >>> model = nn.gnot(
            ...     trunk_size=4,  # (x, y, param1, param2)
            ...     branch_sizes=[3],
            ...     space_dim=2,  # Use (x, y) for gating
            ...     n_experts=4,
            ...     output_size=1,
            ... )
            >>>
            >>> # 3D multi-physics
            >>> model = nn.gnot(
            ...     trunk_size=6,  # (x, y, z, t, Re, Ma)
            ...     branch_sizes=[4, 3],  # velocity field, pressure field
            ...     space_dim=3,
            ...     n_experts=8,
            ...     output_size=4,
            ... )

        Note:
            The first `space_dim` columns of trunk input are used for
            gating. Ensure coordinates are in these positions.

        Reference:
            Hao et al. "GNOT: A General Neural Operator Transformer" (2023)
        """
        model = GNOT(
            trunk_size=trunk_size,
            branch_sizes=branch_sizes,
            space_dim=space_dim,
            output_size=output_size,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            n_experts=n_experts,
            n_inner=n_inner,
            mlp_layers=mlp_layers,
            attn_type=attn_type,
            act=act,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            horiz_fourier_dim=horiz_fourier_dim,
            key=cls._resolve_key(key),
        )
        return cls.wrap(model)

    @classmethod
    def moegptno(
        cls,
        trunk_size: int,
        branch_size: int,
        space_dim: int = 2,
        output_size: int = 1,
        n_layers: int = 2,
        n_hidden: int = 64,
        n_head: int = 1,
        n_experts: int = 2,
        mlp_layers: int = 2,
        attn_type: str = "linear",
        act: str = "gelu",
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        horiz_fourier_dim: int = 0,
        *,
        key: jax.Array | None = None,
    ) -> "Model":
        """
        Create a single-input MoE GPT Neural Operator.

        Simplified variant of GNOT for problems with a single input function.
        Uses MoE FFN layers with position-based gating.

        Args:
            trunk_size: Input dimension for trunk.
            branch_size: Input dimension for single branch.
            space_dim: Spatial dimension for gating. Default: 2.
            output_size: Output dimension. Default: 1.
            n_layers: Number of transformer layers. Default: 2.
            n_hidden: Hidden dimension. Default: 64.
            n_head: Number of attention heads. Default: 1.
            n_experts: Number of expert networks. Default: 2.
            mlp_layers: Layers in embedding MLPs. Default: 2.
            attn_type: Attention type. Default: 'linear'.
            act: Activation function. Default: 'gelu'.
            ffn_dropout: FFN dropout rate. Default: 0.0.
            attn_dropout: Attention dropout rate. Default: 0.0.
            horiz_fourier_dim: Fourier embedding octaves. Default: 0.
            deterministic: Disable dropout. Default: True.

        Returns:
            Model: Wrapped MoEGPTNO model.

        Example:
            >>> model = nn.moegptno(
            ...     trunk_size=4,
            ...     branch_size=3,
            ...     space_dim=2,
            ...     n_experts=4,
            ... )
        """
        model = MoEGPTNO(
            trunk_size=trunk_size,
            branch_size=branch_size,
            space_dim=space_dim,
            output_size=output_size,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            n_experts=n_experts,
            mlp_layers=mlp_layers,
            attn_type=attn_type,
            act=act,
            ffn_dropout=ffn_dropout,
            attn_dropout=attn_dropout,
            horiz_fourier_dim=horiz_fourier_dim,
            key=cls._resolve_key(key),
        )
        return cls.wrap(model)

    # =========================================================================
    # U-Net Architectures
    # =========================================================================

    @classmethod
    def unet1d(
        cls,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        wf: int = 6,
        norm: str = "batch",
        up_mode: str = "upconv",
        groups: int = 1,
        activation: Callable = jax.nn.celu,
        padding_mode: str = "circular",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 1D U-Net.

        U-Net is an encoder-decoder architecture with skip connections,
        effective for problems requiring multi-scale feature extraction.
        The skip connections preserve fine-grained information lost during
        downsampling.

        Architecture::

            Input
              ↓
            [Conv → Norm → Act] × 2  ──────────────────┐
              ↓ (downsample)                           │
            [Conv → Norm → Act] × 2  ────────────┐     │
              ↓ (downsample)                     │     │
            [Conv → Norm → Act] × 2  ──────┐     │     │
              ↓ (downsample)               │     │     │
            [Conv → Norm → Act] × 2        │     │     │ (skip connections)
              ↓ (upsample)                 │     │     │
            [Conv → Norm → Act] × 2  ←─────┘     │     │
              ↓ (upsample)                       │     │
            [Conv → Norm → Act] × 2  ←───────────┘     │
              ↓ (upsample)                             │
            [Conv → Norm → Act] × 2  ←─────────────────┘
              ↓
            Output

        Args:
            in_channels: Input channels. Default: 1.
            out_channels: Output channels. Default: 1.
            depth: Number of encoder/decoder levels. Default: 4.
            wf: Width factor. Base channels = 2^wf (wf=6 → 64). Default: 6.
            norm: Normalization ('batch', 'layer', 'group', None). Default: 'batch'.
            up_mode: Upsampling method:
                - 'upconv': Transposed convolution (learnable)
                - 'upsample': Interpolation + convolution

                Default: 'upconv'.
            groups: Groups for grouped convolutions. Default: 1.
            activation: Activation function. Default: `jax.nn.celu`.
            padding_mode: Padding ('circular', 'reflect'). Default: 'circular'.
            training: Training mode. Default: True.

        Returns:
            Model: Wrapped UNet1D model.

        Example:
            >>> # 1D signal denoising
            >>> model = nn.unet1d(
            ...     in_channels=1,
            ...     out_channels=1,
            ...     depth=4,
            ...     wf=5,  # 32 base channels
            ... )
        """
        unet = UNet1D(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            wf=wf,
            norm=norm,
            up_mode=up_mode,
            groups=groups,
            activation=activation,
            padding_mode=padding_mode,
            key=cls._resolve_key(key),
        )

        return cls.wrap(unet)

    @classmethod
    def unet2d(
        cls,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        wf: int = 6,
        norm: str = "layer",
        up_mode: str = "upconv",
        activation: Callable = jax.nn.gelu,
        padding_mode: str = "circular",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 2D U-Net.

        The standard U-Net architecture for 2D image-to-image tasks.
        Widely used for segmentation, super-resolution, and PDE solving.

        Args:
            in_channels: Input channels. Default: 1.
            out_channels: Output channels. Default: 1.
            depth: Number of levels. Default: 4.
            wf: Width factor (base channels = 2^wf). Default: 6.
            norm: Normalization type. Default: 'layer'.
            up_mode: Upsampling ('upconv' or 'upsample'). Default: 'upconv'.
            activation: Activation function. Default: `jax.nn.celu`.
            padding_mode: Padding mode. Default: 'circular'.
            training: Training mode. Default: True.

        Returns:
            Model: Wrapped UNet2D model.

        Example:
            >>> # Darcy flow
            >>> model = nn.unet2d(
            ...     in_channels=1,  # permeability
            ...     out_channels=1,  # pressure
            ...     depth=4,
            ...     wf=6,
            ... )
            >>>
            >>> # Multi-channel fluid dynamics
            >>> model = nn.unet2d(
            ...     in_channels=4,  # (rho, u, v, p)
            ...     out_channels=4,
            ...     depth=5,
            ...     norm='layer',
            ... )

        Note:
            - Input/output must have spatial dimensions divisible by 2^depth
            - For non-periodic BCs, use `padding_mode='reflect'`

        """
        model = UNet2D(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            wf=wf,
            norm=norm,
            up_mode=up_mode,
            groups=1,
            activation=activation,
            padding_mode=padding_mode,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    @classmethod
    def unet3d(
        cls,
        in_channels: int = 1,
        out_channels: int = 1,
        depth: int = 4,
        wf: int = 6,
        norm: str = "batch",
        up_mode: str = "upconv",
        activation: str = "celu",
        padding_mode: str = "circular",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 3D U-Net.

        For volumetric data or spatiotemporal problems.

        Args:
            in_channels: Input channels. Default: 1.
            out_channels: Output channels. Default: 1.
            depth: Number of levels. Default: 4.
            wf: Width factor. Default: 6.
            norm: Normalization type. Default: 'batch'.
            up_mode: Upsampling method. Default: 'upconv'.
            activation: Activation function. Default: 'celu'.
            padding_mode: Padding mode. Default: 'circular'.
            training: Training mode. Default: True.

        Returns:
            Model: Wrapped UNet3D model.

        Warning:
            Memory usage scales as O(2^(3×depth) × 2^(2×wf)). Consider
            reducing depth or wf for large volumes.
        """
        model = UNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            wf=wf,
            norm=norm,
            up_mode=up_mode,
            groups=1,
            activation=activation,
            padding_mode=padding_mode,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    # =========================================================================
    # Multigrid Neural Operators
    # =========================================================================

    @classmethod
    def mgno1d(
        cls,
        input_length: int,
        num_layer: int = 5,
        num_channel_u: int = 24,
        num_channel_f: int = 3,
        num_iteration: Optional[List[Tuple[int, int]]] = None,
        output_dim: int = 1,
        activation: str = "gelu",
        padding_mode: str = "CIRCULAR",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 1D Multigrid Neural Operator.

        MgNO uses a V-cycle multigrid structure inspired by classical
        multigrid methods for PDEs. This provides efficient multi-scale
        processing with linear complexity.

        V-Cycle Structure::

            Fine grid (smoothing)
                ↓ (restriction)
            Coarse grid (smoothing)
                ↓ (restriction)
            Coarsest grid (solve)
                ↓ (prolongation)
            Coarse grid (smoothing)
                ↓ (prolongation)
            Fine grid (smoothing)

        Args:
            input_length: Length of 1D input sequence.
            num_layer: Number of MgConv layers. Default: 5.
            num_channel_u: Channels for solution representation. Default: 24.
            num_channel_f: Input channels. Default: 3.
            num_iteration: List of (pre_smooth, post_smooth) per level.
                Default: [[1, 1]] × 5.
            output_dim: Output channels. Default: 1.
            activation: Activation function. Default: 'gelu'.
            padding_mode: Padding mode. Default: 'CIRCULAR'.

        Returns:
            Model: Wrapped MgNO1D model.

        Example:
            >>> # 1D Burgers equation
            >>> model = nn.mgno1d(
            ...     input_length=256,
            ...     num_layer=4,
            ...     num_channel_u=16,
            ...     num_channel_f=2,
            ... )
        """
        if num_iteration is None:
            num_iteration = [(1, 1)] * 5

        model = MgNO1D(
            input_length=input_length,
            num_layer=num_layer,
            num_channel_u=num_channel_u,
            num_channel_f=num_channel_f,
            num_iteration=num_iteration,
            output_dim=output_dim,
            activation=activation,
            padding_mode=padding_mode,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    @classmethod
    def mgno2d(
        cls,
        input_shape: Tuple[int, int],
        num_layer: int = 5,
        num_channel_u: int = 24,
        num_channel_f: int = 3,
        num_iteration: Optional[List[Tuple[int, int]]] = None,
        output_dim: int = 1,
        activation: str = "gelu",
        padding_mode: str = "CIRCULAR",
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a 2D Multigrid Neural Operator.

        MgNO provides efficient multi-scale learning through multigrid
        V-cycles. Each layer performs smoothing operations at multiple
        grid resolutions, enabling both local and global feature capture.

        Args:
            input_shape: Spatial dimensions (H, W).
            num_layer: Number of MgConv layers. Default: 5.
            num_channel_u: Solution representation channels. Default: 24.
            num_channel_f: Input feature channels. Default: 3.
            num_iteration: Smoothing iterations per level.
                Format: [[pre, post], ...] for each multigrid level.
                Default: [[1, 1]] × 5.
            output_dim: Output channels. Default: 1.
            activation: Activation ('gelu', 'relu', 'tanh', 'silu'). Default: 'gelu'.
            padding_mode: Padding ('CIRCULAR', 'SAME', 'VALID'). Default: 'CIRCULAR'.

        Returns:
            Model: Wrapped MgNO model.

        Example:
            >>> # 2D Darcy flow
            >>> model = nn.mgno2d(
            ...     input_shape=(64, 64),
            ...     num_layer=5,
            ...     num_channel_u=24,
            ...     num_channel_f=3,
            ...     output_dim=1,
            ... )
            >>>
            >>> # Navier-Stokes with more channels
            >>> model = nn.mgno2d(
            ...     input_shape=(128, 128),
            ...     num_layer=4,
            ...     num_channel_u=32,
            ...     num_channel_f=4,
            ...     output_dim=3,
            ...     activation='silu',
            ... )

        Note:
            The number of multigrid levels is determined by `len(num_iteration)`.
            Input dimensions should be divisible by 2^(num_levels-1).
        """
        if num_iteration is None:
            num_iteration = [(1, 1)] * 5

        model = MgNO(
            input_shape=input_shape,
            num_layer=num_layer,
            num_channel_u=num_channel_u,
            num_channel_f=num_channel_f,
            num_iteration=num_iteration,
            output_dim=output_dim,
            activation=activation,
            padding_mode=padding_mode,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    # =========================================================================
    # Attention-Based Operators
    # =========================================================================

    @classmethod
    def pit(
        cls,
        in_channels: int,
        out_channels: int,
        hid_channels: int = 256,
        n_head: int = 8,
        localities: Sequence[float] = (100, 50, 50, 50, 100),
        input_res: Optional[Tuple[int, int]] = (64, 64),
        latent_res: Optional[Tuple[int, int]] = (16, 16),
        output_res: Optional[Tuple[int, int]] = (64, 64),
        m_dists: Optional[Sequence[jnp.ndarray]] = None,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a Position-induced Transformer (PiT).

        PiT uses position-based attention where attention weights are derived
        from spatial distances rather than learned query-key products. This
        inductive bias makes it naturally suited for physical problems with
        spatial structure.

        Key Innovation:
            Instead of: Attention = softmax(QK^T / √d)
            PiT uses:   Attention = f(distance(pos_i, pos_j))

        This provides:
            - Built-in spatial awareness
            - Better generalization to different resolutions
            - Locality control via the `localities` parameter

        Architecture::

            Input (H×W) → Encoder → Latent (h×w) → Processor × N → Decoder → Output (H'×W')

        Two Modes of Operation:
            1. **Regular grids**: Provide resolution parameters; distances

               computed automatically from coordinates.
            2. **Custom distances**: Provide `m_dists` for irregular grids

               or custom metrics.

        Args:
            in_channels: Input feature channels.
            out_channels: Output feature channels.
            hid_channels: Hidden dimension. Default: 256.
            n_head: Number of attention heads. Default: 8.
            localities: Locality percentages for each attention layer.
                Format: [encoder, *processor_blocks, decoder].
                - 100 = global attention (all positions attend to all)
                - <100 = local attention (only nearest X% of positions)

                Default: (100, 50, 50, 50, 100).
            input_res: Input grid resolution (H, W). Ignored if m_dists provided.
                Default: (64, 64).
            latent_res: Latent/processor resolution. Default: (16, 16).
            output_res: Output grid resolution. Default: (64, 64).
            m_dists: Precomputed distance matrices for irregular grids.
                Length = len(localities). Each shape: [n_head, N_query, N_key].
                Default: None (compute from coordinates).

        Returns:
            Model: Wrapped PiT model.

        Example:
            >>> # Regular grid with local attention in processor
            >>> model = nn.pit(
            ...     in_channels=3,
            ...     out_channels=1,
            ...     localities=[100, 50, 50, 50, 100],
            ...     input_res=(64, 64),
            ...     latent_res=(16, 16),
            ...     output_res=(64, 64),
            ... )
            >>>
            >>> # Custom distances for irregular mesh
            >>> m_dists = compute_custom_distances(mesh)
            >>> model = nn.pit(
            ...     in_channels=3,
            ...     out_channels=1,
            ...     m_dists=m_dists,
            ... )

        Note:
            - Number of processor blocks = len(localities) - 2
            - localities[0] → encoder, localities[-1] → decoder
            - Lower locality values reduce computation but limit receptive field

        """
        if m_dists is not None:
            model = PiT(
                in_channels=in_channels,
                out_channels=out_channels,
                hid_channels=hid_channels,
                n_head=n_head,
                localities=list(localities),
                m_dists=m_dists,
                key=cls._resolve_key(key),
            )
        else:
            model = PiTWithCoords(  # type: ignore[assignment]
                in_channels=in_channels,
                out_channels=out_channels,
                hid_channels=hid_channels,
                n_head=n_head,
                localities=list(localities),
                input_res=input_res,
                latent_res=latent_res,
                output_res=output_res,
                key=cls._resolve_key(key),
            )

        return cls.wrap(model)

    @classmethod
    def transformer(
        cls,
        num_layers: int = 6,
        embed_dim: int = 512,
        num_heads: int = 8,
        mlp_features: int = 2048,
        dropout_rate: float = 0.1,
        vocab_size: int = 10000,
        max_len: int = 128,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a standard Transformer.

        Encoder-decoder transformer architecture for sequence-to-sequence
        tasks. Can be adapted for operator learning by treating spatial
        points as sequence elements.

        Args:
            num_layers: Layers in encoder and decoder. Default: 6.
            embed_dim: Embedding/hidden dimension. Default: 512.
            num_heads: Attention heads. Default: 8.
            mlp_features: FFN hidden dimension. Default: 2048.
            dropout_rate: Dropout probability. Default: 0.1.
            vocab_size: Vocabulary size (for token embeddings). Default: 10000.
            max_len: Maximum sequence length. Default: 128.
            key: JAX PRNG key for weight initialization.

        Returns:
            Model: Wrapped Transformer model.

        Note:
            For continuous operator learning, consider using PiT or ScOT
            which are specifically designed for spatial data.
        """
        model_inst = Transformer(
            encoder_num_layers=num_layers,
            decoder_num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            qkv_features=embed_dim,
            mlp_features=mlp_features,
            vocab_size=vocab_size,
            dropout_rate=dropout_rate,
            max_len=max_len,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model_inst)

    # =========================================================================
    # Branch-Trunk Architectures
    # =========================================================================

    @classmethod
    def deeponet(
        cls,
        branch_type: Literal["mlp", "resmlp", "conv1d", "transformer"] = "mlp",
        trunk_type: Literal["mlp", "resmlp", "siren"] = "mlp",
        combination_type: Literal["dot", "bilinear", "mlp", "attention"] = "dot",
        n_sensors: int = 100,
        sensor_channels: int = 1,
        coord_dim: int = 1,
        n_outputs: int = 1,
        basis_functions: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        coord_embedding: Optional[Literal["fourier", "positional"]] = None,
        coord_embedding_dim: int = 64,
        coord_embedding_scale: float = 1.0,
        activation: Callable = jax.nn.gelu,
        norm: Optional[str] = None,
        dropout_rate: float = 0.0,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a Deep Operator Network (DeepONet).

        DeepONet learns operators by decomposing them into:
            - **Branch network**: Encodes the input function (sampled at sensors)
            - **Trunk network**: Encodes query coordinates
            - **Combination**: Combines branch and trunk outputs

        Output: G(u)(y) = Σᵢ bᵢ(u) × tᵢ(y)

        Where bᵢ are branch outputs (basis coefficients) and tᵢ are trunk
        outputs (basis functions evaluated at query point y).

        Architecture Variants:
            - Branch: MLP, ResMLP, Conv1D, Transformer
            - Trunk: MLP, ResMLP, SIREN (for high-frequency details)
            - Combination: Dot product, Bilinear, MLP, Attention

        Args:
            branch_type: Branch network architecture. Default: 'mlp'.
            trunk_type: Trunk network architecture. Default: 'mlp'.
            combination_type: How to combine outputs. Default: 'dot'.
            n_sensors: Number of sensor points for input function. Default: 100.
            sensor_channels: Channels per sensor. Default: 1.
            coord_dim: Query coordinate dimension. Default: 1.
            n_outputs: Output field channels. Default: 1.
            basis_functions: Number of basis functions (p). Default: 128.
            hidden_dim: Hidden dimension for networks. Default: 256.
            n_layers: Number of layers/blocks. Default: 4.
            n_heads: Attention heads (for transformer/attention). Default: 8.
            coord_embedding: Coordinate embedding type. Default: None.
            coord_embedding_dim: Embedding dimension. Default: 64.
            coord_embedding_scale: Fourier feature scale. Default: 1.0.
            activation: Activation function. Default: `jax.nn.gelu`.
            norm: Normalization type. Default: None.
            dropout_rate: Dropout rate. Default: 0.0.
            training: Training mode. Default: True.

        Returns:
            Model: Wrapped DeepONet model.

        Example:
            >>> # Simple DeepONet for 1D problems
            >>> model = nn.deeponet(
            ...     n_sensors=100,
            ...     coord_dim=1,
            ...     basis_functions=64,
            ... )
            >>>
            >>> # Advanced DeepONet with Fourier features
            >>> model = nn.deeponet(
            ...     branch_type='transformer',
            ...     trunk_type='siren',
            ...     combination_type='attention',
            ...     n_sensors=256,
            ...     coord_dim=2,
            ...     coord_embedding='fourier',
            ...     coord_embedding_scale=10.0,
            ... )

        Note:
            - Input: (branch_input: [batch, n_sensors, sensor_channels],

                      trunk_input: [batch, n_query, coord_dim])
            - Output: [batch, n_query, n_outputs]

        """
        hidden_dims = tuple([hidden_dim] * n_layers)

        model = DeepONet(
            branch_type=branch_type,
            trunk_type=trunk_type,
            combination_type=combination_type,
            n_sensors=n_sensors,
            sensor_channels=sensor_channels,
            coord_dim=coord_dim,
            n_outputs=n_outputs,
            basis_functions=basis_functions,
            branch_hidden_dims=hidden_dims,
            branch_hidden_dim=hidden_dim,
            branch_n_blocks=n_layers,
            branch_n_layers=n_layers,
            branch_n_heads=n_heads,
            trunk_hidden_dims=hidden_dims,
            trunk_hidden_dim=hidden_dim,
            trunk_n_blocks=n_layers,
            coord_embedding=coord_embedding,
            coord_embedding_dim=coord_embedding_dim,
            coord_embedding_scale=coord_embedding_scale,
            combination_hidden_dims=(hidden_dim, hidden_dim // 2),
            combination_d_model=hidden_dim,
            combination_n_heads=n_heads,
            activation=activation,
            norm=norm,
            dropout_rate=dropout_rate,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    # =========================================================================
    # Point-Based Networks
    # =========================================================================

    @classmethod
    def pointnet(
        cls,
        in_features: int,
        output_dim: int,
        hidden_dims: List[int] = [32, 16, 8, 4, 2, 2, 4, 8, 8],
        dropout_rate: float = 0.0,
        feature_transform: Optional[Callable] = None,
        activation_function: Callable = jnp.tanh,
        use_bias: bool = True,
        *,
        key: jax.Array | None = None,
    ) -> Model:
        """
        Create a PointNet-style network.

        PointNet processes unordered point sets through shared MLPs and
        symmetric aggregation (max pooling), achieving permutation invariance.

        Architecture::

            Points → SharedMLP → MaxPool → GlobalFeature → MLP → Output
                         ↓
                    PointFeatures → Concat(GlobalFeature) → MLP → PerPointOutput

        Args:
            output_dim: Output features per point.
            hidden_dims: Sizes of each of the kernels (List of 9 integers)
            dropout_rate: Dropout probability. Default: 0.0.
            feature_transform: Optional input transformation. Default: None.
            activation_function: Activation function. Default: `jnp.tanh`.
            use_bias: Include bias terms. Default: True.

        Returns:
            Model: Wrapped PointNet model.

        Example:
            >>> # Point cloud regression
            >>> model = nn.pointnet(
            ...     output_dim=3,
            ...     conv_scale=1.0,
            ... )

        Note:
            - Input: [batch, n_points, in_features]
            - Output: [batch, n_points, output_dim]
            - Permutation invariant to point ordering

        """
        model = PointNet(
            in_features=in_features,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            feature_transform=feature_transform,
            act=activation_function,
            use_bias=use_bias,
            key=cls._resolve_key(key),
        )

        return cls.wrap(model)

    # =========================================================================
    # Scalable Operator Transformer (ScOT)
    # =========================================================================

    @classmethod
    def scot(
        cls,
        name: str,
        image_size: int,
        patch_size: int = 4,
        num_channels: int = 4,
        num_out_channels: int = 4,
        embed_dim: int = 48,
        depths: Tuple[int, int, int, int] = (4, 4, 4, 4),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        skip_connections: Tuple[int, int, int, int] = (2, 2, 2, 0),
        window_size: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        hidden_dropout_prob: float = 0.0,
        attention_probs_dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        hidden_act: str = "gelu",
        use_absolute_embeddings: bool = False,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-5,
        p: int = 1,
        channel_slice_list_normalized_loss: Optional[List[int]] = None,
        residual_model: str = "convnext",
        use_conditioning: bool = True,
        learn_residual: bool = False,
        pretrained_window_sizes: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Model:
        """
        Create a Scalable Operator Transformer (ScOT).

        ScOT combines Swin Transformer's efficient windowed attention with
        U-Net-style skip connections for multi-scale operator learning.
        It forms the backbone of the Poseidon foundation models.

        Architecture::

            Input → PatchEmbed → [SwinBlock × d₁] → Downsample → [SwinBlock × d₂] → ...
                                       ↓ (skip)                        ↓ (skip)
            Output ← PatchExpand ← [SwinBlock × d₁] ← Upsample ← [SwinBlock × d₂] ← ...

        Key Features:
            - Windowed self-attention for O(n) complexity
            - Shifted windows for cross-window connections
            - Skip connections for multi-scale information flow
            - Optional conditioning for time-dependent problems

        Args:
            name: Model identifier.
            image_size: Input image size (assumes square).
            patch_size: Patch size for tokenization. Default: 4.
            num_channels: Input channels. Default: 4.
            num_out_channels: Output channels. Default: 4.
            embed_dim: Base embedding dimension. Default: 48.
            depths: Blocks per stage. Default: (4, 4, 4, 4).
            num_heads: Attention heads per stage. Default: (3, 6, 12, 24).
            skip_connections: Skip connection frequency. Default: (2, 2, 2, 0).
            window_size: Attention window size. Default: 16.
            mlp_ratio: FFN expansion ratio. Default: 4.0.
            qkv_bias: Bias in QKV projections. Default: True.
            hidden_dropout_prob: Hidden layer dropout. Default: 0.0.
            attention_probs_dropout_prob: Attention dropout. Default: 0.0.
            drop_path_rate: Stochastic depth rate. Default: 0.0.
            hidden_act: Activation function. Default: 'gelu'.
            use_absolute_embeddings: Add absolute position embeddings. Default: False.
            initializer_range: Weight init std. Default: 0.02.
            layer_norm_eps: LayerNorm epsilon. Default: 1e-5.
            p: Patch merging factor. Default: 1.
            channel_slice_list_normalized_loss: Channels for normalized loss.
                Default: [0, 1, 3, 4].
            residual_model: Residual block type. Default: 'convnext'.
            use_conditioning: Enable time/parameter conditioning. Default: True.
            learn_residual: Learn residual instead of full output. Default: False.
            pretrained_window_sizes: For loading pretrained weights. Default: (0,0,0,0).

        Returns:
            Model: Wrapped ScOT model.

        Example:
            >>> # Custom ScOT for 128x128 images
            >>> model = nn.scot(
            ...     name="my_scot",
            ...     image_size=128,
            ...     num_channels=3,
            ...     num_out_channels=1,
            ...     embed_dim=96,
            ...     depths=(2, 2, 6, 2),
            ... )

        See Also:
            - `poseidonT/B/L`: Pretrained ScOT variants

        """
        if channel_slice_list_normalized_loss is None:
            channel_slice_list_normalized_loss = [0, 1, 3, 4]

        from jax_poseidon import ScOT, ScOTConfig

        config = ScOTConfig(
            name=name,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_out_channels=num_out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            skip_connections=skip_connections,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            use_absolute_embeddings=use_absolute_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            p=p,
            channel_slice_list_normalized_loss=channel_slice_list_normalized_loss,
            residual_model=residual_model,
            use_conditioning=use_conditioning,
            learn_residual=learn_residual,
            pretrained_window_sizes=pretrained_window_sizes,
        )

        flax_model = ScOT(config=config, use_conditioning=use_conditioning)

        rng = jax.random.PRNGKey(0)
        dummy_pv = jnp.ones((1, image_size, image_size, num_channels))
        dummy_t = jnp.zeros((1,))
        params = flax_model.init(
            {"params": rng, "dropout": rng},
            pixel_values=dummy_pv,
            time=dummy_t,
            deterministic=True,
        )
        wrapped = FlaxModelWrapper(
            flax_model.apply,
            params,
            post_fn=lambda x: x.output,
            deterministic=True,
        )
        return cls.wrap(wrapped, name="poseidon")

    # =========================================================================
    # Foundation Models (Pretrained)
    # =========================================================================

    @classmethod
    def _poseidon(
        cls,
        name: str,
        embed_dim: int,
        depths: Tuple[int, ...],
        num_in_channels: int,
        num_out_channels: int,
        compute_dtype=None,
    ) -> Model:
        """Internal helper that builds a fresh Poseidon ScOT model."""
        from jax_poseidon import ScOT, ScOTConfig

        config = ScOTConfig(
            name=name,
            image_size=128,
            patch_size=4,
            num_channels=num_in_channels,
            num_out_channels=num_out_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=(3, 6, 12, 24),
            skip_connections=(2, 2, 2, 0),
            window_size=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            drop_path_rate=0.0,
            hidden_act="gelu",
            use_absolute_embeddings=False,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            p=1,
            channel_slice_list_normalized_loss=[0, 1, 3, 4],
            residual_model="convnext",
            use_conditioning=True,
            learn_residual=False,
            pretrained_window_sizes=(0, 0, 0, 0),
            # compute_dtype=compute_dtype,  # None → float32 (or float64 if JAX_ENABLE_X64=1)
        )

        flax_model = ScOT(config=config, use_conditioning=True)

        rng = jax.random.PRNGKey(0)
        dummy_pv = jnp.ones((1, 128, 128, num_in_channels))
        dummy_t = jnp.ones((1, 1))
        fresh_params = flax_model.init(
            {"params": rng, "dropout": rng},
            pixel_values=dummy_pv,
            time=dummy_t,
            deterministic=True,
        )

        wrapped = FlaxModelWrapper(
            flax_model.apply,
            fresh_params,
            post_fn=lambda x: x.output,
            deterministic=True,
        )
        return cls.wrap(wrapped, name=name)

    @classmethod
    def poseidonT(
        cls,
        num_in_channels: int = 4,
        num_out_channels: int = 4,
        compute_dtype=None,
    ) -> Model:
        """
        Poseidon-T (Tiny) foundation model (~20.8M parameters).

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            num_in_channels: Number of input channels.  Default: 4.
            num_out_channels: Number of output channels.  Default: 4.

        Returns:
            Model wrapping the Poseidon ScOT model.

        Example:
            >>> u = nn.poseidonT(num_in_channels=1, num_out_channels=1)
            >>> u.initialize("poseidonT.msgpack")

        Reference:
            Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
        """
        return cls._poseidon("poseidonT", embed_dim=48, depths=(4, 4, 4, 4), num_in_channels=num_in_channels, num_out_channels=num_out_channels, compute_dtype=compute_dtype)

    @classmethod
    def poseidonB(
        cls,
        num_in_channels: int = 4,
        num_out_channels: int = 4,
        compute_dtype=None,
    ) -> Model:
        """
        Poseidon-B (Base) foundation model (~157.7M parameters).

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            num_in_channels: Number of input channels.  Default: 4.
            num_out_channels: Number of output channels.  Default: 4.

        Returns:
            Model wrapping the Poseidon ScOT model.

        Example:
            >>> u = nn.poseidonB(num_in_channels=1, num_out_channels=1)
            >>> u.initialize("poseidonB.msgpack")

        Reference:
            Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
        """
        return cls._poseidon("poseidonB", embed_dim=96, depths=(8, 8, 8, 8), num_in_channels=num_in_channels, num_out_channels=num_out_channels, compute_dtype=compute_dtype)

    @classmethod
    def poseidonL(
        cls,
        num_in_channels: int = 4,
        num_out_channels: int = 4,
        compute_dtype=None,
    ) -> Model:
        """
        Poseidon-L (Large) foundation model (~628.6M parameters).

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            num_in_channels: Number of input channels.  Default: 4.
            num_out_channels: Number of output channels.  Default: 4.

        Returns:
            Model wrapping the Poseidon ScOT model.

        Example:
            >>> u = nn.poseidonL(num_in_channels=1, num_out_channels=1)
            >>> u.initialize("poseidonL.msgpack")

        Reference:
            Herde et al., "Poseidon: Efficient Foundation Models for PDEs" (2024)
        """
        return cls._poseidon("poseidonL", embed_dim=192, depths=(8, 8, 8, 8), num_in_channels=num_in_channels, num_out_channels=num_out_channels, compute_dtype=compute_dtype)

    @classmethod
    def walrus(
        cls,
        input_shape: Tuple[int, ...],
        num_out_channels: int = 1,
        state_labels: Optional[jnp.ndarray] = None,
        bcs: Optional[List] = None,
        dim_key: int = 2,
        remat: bool = True,
    ) -> Model:
        """
        Initialize a Walrus foundation model (1.29 B parameters).

        Walrus uses a *SpaceBag* encoder that selects input channels at
        runtime via ``field_indices``, so the weight file is architecture-
        independent — the same checkpoint works for any PDE.

        The model expects channels-last input ``(B, T, H, [W, [D]], C)``
        and returns ``(B, T_out, H, [W, [D]], C_out)``.

        Call ``.initialize(weight_path)`` on the returned module to load
        pretrained weights (deferred until training starts).

        Args:
            input_shape: Shape of a single input tensor, e.g.
                ``(1, 1, 128, 128, 1)`` for a 2-D problem with one channel.
                Used to run a dummy forward pass that creates the parameter
                template.
            num_out_channels: Number of output channels.  ``state_labels``
                is built as ``jnp.arange(num_out_channels)`` when not
                given explicitly.
            state_labels: Explicit output-channel indices for the decoder.
                Overrides ``num_out_channels`` when provided.
            bcs: Boundary conditions per spatial dim, each
                ``[bc_left, bc_right]``.  ``2`` = periodic, ``0`` = open.
                Default: periodic in H & W, singleton in D
                (``[[2,2],[2,2],[0,0]]``).
            dim_key: Spatial dimensionality of the data (2 or 3).
                Default: 2.
            remat: If ``True`` (default), wrap each processor block with
                ``nn.remat`` (gradient checkpointing) to reduce peak
                memory at the cost of recomputing activations during
                the backward pass.

        Returns:
            Model wrapping the Walrus model.

        Example::

            u = nn.walrus((1, 1, 128, 128, 1), num_out_channels=1)
            u.initialize("walrus.msgpack")

        Reference:
            McCabe et al., "Multiple Physics Pretraining for Physical
            Surrogate Models" (2023)
        """

        from jax_walrus import IsotropicModel as WalrusModel

        if state_labels is None:
            state_labels = jnp.arange(num_out_channels, dtype=jnp.int32)
        if bcs is None:
            bcs = [[2, 2], [2, 2], [0, 0]]

        # Store as a plain Python tuple so it can live in a static eqx field
        # without the "JAX array set as static" warning.  The Flax model
        # converts it back to an array at the top of __call__.
        state_labels_tup = tuple(int(s) for s in state_labels)

        # ── Build model with pretrained hyperparameters ──
        flax_model = WalrusModel(
            hidden_dim=1408,
            intermediate_dim=352,
            n_states=67,
            processor_blocks=40,
            num_heads=16,
            groups=16,
            causal_in_time=True,
            bias_type="rel",
            base_kernel_size=((8, 4), (8, 4), (8, 4)),
            use_spacebag=True,
            use_silu=True,
            include_d=(2, 3),
            encoder_groups=16,
            remat=remat,
        )

        # ── Create fresh (random) params via a dummy forward pass ──
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.ones(input_shape)
        params = flax_model.init(
            {"params": rng, "dropout": rng, "drop_path": rng, "jitter": rng},
            x=dummy_x,
            state_labels=state_labels,
            bcs=bcs,
            dim_key=dim_key,
            deterministic=True,
        )

        wrapped = FlaxModelWrapper(
            flax_model.apply,
            params,
            state_labels=state_labels_tup,
            bcs=bcs,
            dim_key=dim_key,
            deterministic=True,
        )
        return cls.wrap(wrapped, name="walrus")

    # =========================================================================
    # MORPH Foundation Models
    # =========================================================================

    @classmethod
    def _morph(
        cls,
        name: str,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        max_ar: int,
        model_size: str,
        spatial_size: int = 128,
        conv_filter: int = 8,
        heads_xa: int = 32,
        max_components: int = 3,
        max_patches: int = 4096,
        max_fields: int = 3,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> Model:
        """Internal helper that builds a fresh MORPH model.

        jNO's vmap+scan pipeline delivers one sample at a time to the model.
        For a domain storing ``_f`` as ``(S, 1, 1, H, W, 1)`` (Poseidon layout),
        each sample arrives as ``(1, H, W, 1)`` after B is vmapped
        and T is scanned.  MORPH expects ``(B, t, F, C, D, H, W)``
        (native 7-D volume format).  A ``MorphAdapter`` wrapper reshapes
        in both directions so the jNO interface is identical to Walrus.

        Args:
            name: Display name (e.g. ``"morph_Ti"``).
            dim: Transformer embedding dimension.
            depth: Number of transformer blocks.
            heads: Number of attention heads.
            mlp_dim: MLP hidden dimension.
            max_ar: Maximum auto-regressive timesteps.
            model_size: Model variant string (``'Ti'``, ``'S'``, ``'M'``, ``'L'``).
            spatial_size: Spatial resolution H=W (must be divisible by 8).
                Used for the dummy init forward pass.  Default: 128.
            conv_filter: Conv feature extractor output channels. Default: 8.
            heads_xa: Number of heads for field cross-attention. Default: 32.
            max_components: Maximum input components. Default: 3.
            max_patches: Maximum number of patches. Default: 4096.
            max_fields: Maximum number of fields. Default: 3.
            dropout: Dropout rate. Default: 0.0.
            emb_dropout: Embedding dropout rate. Default: 0.0.
        """

        from jax_morph import ViT3DRegression as MorphModel

        flax_model = MorphModel(
            patch_size=8,
            dim=dim,
            depth=depth,
            heads=heads,
            heads_xa=heads_xa,
            mlp_dim=mlp_dim,
            max_components=max_components,
            conv_filter=conv_filter,
            max_ar=max_ar,
            max_patches=max_patches,
            max_fields=max_fields,
            dropout=dropout,
            emb_dropout=emb_dropout,
            model_size=model_size,
        )

        # Init params with the native MORPH shape (B, t, F, C, D, H, W).
        # D=1 turns a 2-D problem into the 3-D volume MORPH expects.
        S = spatial_size
        rng = jax.random.PRNGKey(0)
        dummy_vol = jnp.ones((1, 1, 1, 1, 1, S, S))  # (B=1, t=1, F=1, C=1, D=1, H, W)
        params = flax_model.init(rng, dummy_vol, deterministic=True)

        # ── Adapter: jNO ↔ MORPH shape bridge ──────────────────────────
        # jNO may deliver either a single frame ``(H, W, C)`` or a time window
        # ``(T, H, W, C)`` per sample, depending on temporal scanning.
        # MORPH expects ``(B, t, F, C, D, H, W)``.
        # For the common single-step case, output ``(B=1, F=1, C=1, D=1, H, W)``
        # is mapped back to ``(1, H, W, C)``.
        #
        # We bake the reshape into the apply_fn closure so the wrapped
        # object stays a FlaxModelWrapper — this is required for the
        # Flax msgpack weight-loading path in jNO's build_single_layer_params.
        _base_apply = flax_model.apply

        def morph_apply(params, x, **kwargs):
            # x is channels-last 2-D data with optional leading time axis.
            if x.ndim == 3:
                h, w, c = x.shape
                vol = jnp.transpose(x, (2, 0, 1))[None, None, None, :, None, :, :]
                time_steps = 1
            elif x.ndim == 4:
                time_steps, h, w, c = x.shape
                vol = jnp.transpose(x, (0, 3, 1, 2))[None, :, None, :, None, :, :]
            else:
                raise ValueError(f"MORPH adapter expected (H,W,C) or (T,H,W,C), got {x.shape}")

            _enc, _z, x_last = _base_apply(params, vol, **kwargs)

            # Current MORPH variants used here emit the latest frame only as
            # (B, F, C, D, H, W); map back to jNO's channels-last layout.
            if x_last.ndim != 6:
                raise ValueError(f"Unexpected MORPH output shape: {x_last.shape}")
            y = jnp.squeeze(x_last, axis=(0, 1, 3))  # (C, H, W)
            y = jnp.transpose(y, (1, 2, 0))  # (H, W, C)
            return y[None, ...] if time_steps == 1 else y[None, ...]

        wrapped = FlaxModelWrapper(
            morph_apply,
            params,
            deterministic=True,
        )
        return cls.wrap(wrapped, name=name)

    @classmethod
    def morphTi(
        cls,
        spatial_size: int = 128,
    ) -> Model:
        """
        MORPH-Ti (Tiny) foundation model.

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            spatial_size: Spatial resolution H=W of the input grid
                (must be divisible by 8).  Used for the dummy init
                forward pass.  Default: 128.

        Returns:
            Model wrapping the MORPH-Ti model.

        Example::

            u = nn.morphTi()
            u.initialize("morph-Ti.msgpack")

        Reference:
            Rautela et al., "MORPH: PDE Foundation Models with Arbitrary
            Data Modality" (2025)
        """
        return cls._morph(
            "morph_Ti",
            dim=256,
            depth=4,
            heads=4,
            mlp_dim=1024,
            max_ar=1,
            model_size="Ti",
            spatial_size=spatial_size,
        )

    @classmethod
    def morphS(
        cls,
        spatial_size: int = 128,
    ) -> Model:
        """
        MORPH-S (Small) foundation model.

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            spatial_size: Spatial resolution H=W (must be divisible by 8).
                Default: 128.

        Returns:
            Model wrapping the MORPH-S model.

        Example::

            u = nn.morphS()
            u.initialize("morph-S.msgpack")

        Reference:
            Rautela et al., "MORPH: PDE Foundation Models with Arbitrary
            Data Modality" (2025)
        """
        return cls._morph(
            "morph_S",
            dim=512,
            depth=4,
            heads=8,
            mlp_dim=2048,
            max_ar=1,
            model_size="S",
            spatial_size=spatial_size,
        )

    @classmethod
    def morphM(
        cls,
        spatial_size: int = 128,
    ) -> Model:
        """
        MORPH-M (Medium) foundation model.

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            spatial_size: Spatial resolution H=W (must be divisible by 8).
                Default: 128.

        Returns:
            Model wrapping the MORPH-M model.

        Example::

            u = nn.morphM()
            u.initialize("morph-M.msgpack")

        Reference:
            Rautela et al., "MORPH: PDE Foundation Models with Arbitrary
            Data Modality" (2025)
        """
        return cls._morph(
            "morph_M",
            dim=768,
            depth=8,
            heads=12,
            mlp_dim=3072,
            max_ar=1,
            model_size="M",
            spatial_size=spatial_size,
        )

    @classmethod
    def morphL(
        cls,
        spatial_size: int = 128,
    ) -> Model:
        """
        MORPH-L (Large) foundation model.

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            spatial_size: Spatial resolution H=W (must be divisible by 8).
                Default: 128.

        Returns:
            Model wrapping the MORPH-L model.

        Example::

            u = nn.morphL()
            u.initialize("morph-L.msgpack")

        Reference:
            Rautela et al., "MORPH: PDE Foundation Models with Arbitrary
            Data Modality" (2025)
        """
        return cls._morph(
            "morph_L",
            dim=1024,
            depth=16,
            heads=16,
            mlp_dim=4096,
            max_ar=16,
            model_size="L",
            spatial_size=spatial_size,
        )

    # =========================================================================
    # MPP Foundation Models
    # =========================================================================

    @classmethod
    def _mpp(
        cls,
        name: str,
        variant: str,
        spatial_size: int = 128,
        num_channels: int = 1,
    ) -> Model:
        """Internal helper that builds a fresh MPP/AViT model.

        jNO's vmap+scan pipeline delivers one sample at a time to the model.
        For a domain storing ``_f`` as ``(S, 1, 1, H, W, 1)`` (Poseidon layout),
        each sample arrives as ``(1, H, W, 1)`` after B is vmapped
        and T is scanned.  MPP/AViT expects ``(T, B, C, H, W)``
        (time-first, channels-first).  An adapter closure reshapes
        in both directions so the jNO interface is identical to the
        other foundation models.

        Args:
            name: Display name (e.g. ``"mpp_Ti"``).
            variant: One of ``'Ti'``, ``'S'``, ``'B'``, ``'L'``.
            spatial_size: Spatial resolution H=W. Default: 128.
            num_channels: Number of active state-variable channels.
                Default: 1 (suitable for Poisson, heat equation, etc.).
        """

        from jax_mpp import AViT as MPPModel, AVIT_CONFIGS

        cfg = AVIT_CONFIGS[variant]
        flax_model = MPPModel(**cfg)

        # Init params with matched dummy input.
        S = spatial_size
        rng = jax.random.PRNGKey(0)
        dummy_x = jnp.ones((1, 1, num_channels, S, S))  # (T=1, B=1, C, H, W)
        state_labels = jnp.arange(num_channels, dtype=jnp.int32)
        dummy_bcs = jnp.zeros((1, 2), dtype=jnp.int32)
        params = flax_model.init(
            {"params": rng, "drop_path": rng},
            dummy_x,
            state_labels,
            dummy_bcs,
            deterministic=True,
        )

        # Store state_labels as tuple for static eqx field.
        state_labels_tup = tuple(int(s) for s in state_labels)

        # ── Adapter: jNO ↔ MPP shape bridge ─────────────────────────
        _base_apply = flax_model.apply

        def mpp_apply(params, x, **kwargs):
            # x: (1, H, W, 1) — jNO per-sample input (channels-last 2-D)
            H, W = x.shape[-3], x.shape[-2]
            C = x.shape[-1]
            # → (T=1, B=1, C, H, W)  channels-first, time-first
            vol = x.reshape(1, 1, C, H, W)
            sl = jnp.array(state_labels_tup, dtype=jnp.int32)
            bcs = jnp.zeros((1, 2), dtype=jnp.int32)
            out = _base_apply(params, vol, sl, bcs, deterministic=True)
            # out: (B=1, C, H, W) → (1, H, W, C)
            return out[0].transpose(1, 2, 0).reshape(1, H, W, C)

        wrapped = FlaxModelWrapper(
            mpp_apply,
            params,
            deterministic=True,
        )
        return cls.wrap(wrapped, name=name)

    @classmethod
    def mppTi(
        cls,
        spatial_size: int = 128,
        num_channels: int = 1,
    ) -> Model:
        """
        MPP-Ti (Tiny) foundation model (~7.3 M parameters).

        Creates a fresh model. To load pretrained weights call
        ``model.initialize(weight_path)`` afterwards.

        Args:
            spatial_size: Spatial resolution H=W. Default: 128.
            num_channels: Number of active state channels. Default: 1.

        Returns:
            Model wrapping the MPP-Ti model.

        Example::

            u = nn.mppTi()
            u.initialize("mpp-Ti.msgpack")

        Reference:
            McCabe et al., "Multiple Physics Pretraining for Physical
            Surrogate Models" (NeurIPS 2024)
        """
        return cls._mpp("mpp_Ti", "Ti", spatial_size, num_channels)

    @classmethod
    def mppS(
        cls,
        spatial_size: int = 128,
        num_channels: int = 1,
    ) -> Model:
        """
        MPP-S (Small) foundation model.

        See :meth:`mppTi` for full documentation.
        """
        return cls._mpp("mpp_S", "S", spatial_size, num_channels)

    @classmethod
    def mppB(
        cls,
        spatial_size: int = 128,
        num_channels: int = 1,
    ) -> Model:
        """
        MPP-B (Base) foundation model.

        See :meth:`mppTi` for full documentation.
        """
        return cls._mpp("mpp_B", "B", spatial_size, num_channels)

    @classmethod
    def mppL(
        cls,
        spatial_size: int = 128,
        num_channels: int = 1,
    ) -> Model:
        """
        MPP-L (Large) foundation model.

        See :meth:`mppTi` for full documentation.
        """
        return cls._mpp("mpp_L", "L", spatial_size, num_channels)

    # =========================================================================
    # PDEformer-2 Foundation Models
    # =========================================================================

    @staticmethod
    def _pdeformer2_bake_dag(dag_inputs: dict) -> dict:
        """Normalise raw DAG arrays into batched ``jnp`` tensors.

        ``build_dag_inputs`` returns unbatched numpy arrays.  This helper
        converts them to ``jnp`` arrays and adds a leading ``n_graph=1``
        dimension when missing, so they can be stored in the ``apply_fn``
        closure as constants.
        """
        _expect_ndim = {
            "node_type": 3,  # (1, n_node, 1)
            "node_scalar": 3,  # (1, num_scalar, 1)
            "node_function": 4,  # (1, num_func, pts, 5)
            "in_degree": 2,  # (1, n_node)
            "out_degree": 2,  # (1, n_node)
            "attn_bias": 3,  # (1, n_node, n_node)
            "spatial_pos": 3,  # (1, n_node, n_node)
        }
        out = {}
        for k, target_ndim in _expect_ndim.items():
            v = jnp.asarray(dag_inputs[k])
            if v.ndim == target_ndim - 1:
                v = v[jnp.newaxis]
            out[k] = v
        return out

    @classmethod
    def _pdeformer2(
        cls,
        name: str,
        config: dict,
        dag_inputs: Optional[dict],
        num_points: int,
    ) -> Model:
        """Internal helper that builds a fresh PDEformer-2 model.

        Args:
            name: Display name (e.g. ``"pdeformer2_small"``).
            config: One of ``PDEFORMER_SMALL_CONFIG``,
                ``PDEFORMER_BASE_CONFIG``, ``PDEFORMER_FAST_CONFIG``.
            dag_inputs: Optional dict from ``build_dag_inputs``.  When
                provided the static DAG arrays are baked into the model
                and the user only needs to pass ``coordinate`` at call
                time.
            num_points: Number of query coordinates for the INR decoder
                (used only in the dummy init when *dag_inputs* is
                ``None``).
        """
        from jax_pdeformer2 import (
            create_pdeformer_from_config,
            PDEFORMER_SMALL_CONFIG,
            PDEFORMER_BASE_CONFIG,
            PDEFORMER_FAST_CONFIG,
        )
        from jax_pdeformer2.utils import create_dummy_inputs as _pdeformer2_dummy_inputs

        flax_model = create_pdeformer_from_config({"model": config})

        # Derive sizes from config
        func_cfg = config.get("function_encoder", {})
        num_branches = func_cfg.get("num_branches", 4)
        resolution = func_cfg.get("resolution", 128)

        dummy = _pdeformer2_dummy_inputs(
            n_graph=1,
            num_scalar=80,
            num_function=6,
            num_branches=num_branches,
            num_points_function=resolution**2,
            num_points=num_points,
        )

        rng = jax.random.PRNGKey(0)
        params = flax_model.init(rng, **dummy)

        if dag_inputs is not None:
            # Bake static DAG arrays into the apply closure so the user
            # only passes ``coordinate`` at call time.
            static_dag = cls._pdeformer2_bake_dag(dag_inputs)
            base_apply = flax_model.apply

            def dag_apply(params, coordinate, **kwargs):
                # jNO evaluator passes (N, 4); model expects (1, N, 4)
                coord = coordinate[None] if coordinate.ndim == 2 else coordinate
                out = base_apply(
                    params,
                    **static_dag,
                    coordinate=coord,
                    **kwargs,
                )
                return out[0]  # (1, N, 1) → (N, 1)

            wrapped = FlaxModelWrapper(dag_apply, params, deterministic=True)
        else:
            wrapped = FlaxModelWrapper(
                flax_model.apply,
                params,
                deterministic=True,
            )

        return cls.wrap(wrapped, name=name)

    @classmethod
    def pdeformer2_small(
        cls,
        dag_inputs: Optional[dict] = None,
        num_points: int = 1000,
    ) -> Model:
        """PDEformer-2-Small foundation model (~27 M parameters).

        Creates a fresh model.  Call ``.initialize(weight_path)`` to load
        pretrained weights from a ``.msgpack`` file.

        The model encodes a PDE as a directed acyclic graph (DAG) whose
        nodes carry scalar coefficients and function data (initial /
        boundary conditions).  A Graphormer encodes the DAG, and an
        implicit neural representation (INR) with hyper-networks decodes
        the solution at arbitrary query coordinates.

        Args:
            dag_inputs: Optional dict returned by ``build_dag_inputs``
                containing the 7 static DAG tensors (``node_type``,
                ``node_scalar``, ``node_function``, ``in_degree``,
                ``out_degree``, ``attn_bias``, ``spatial_pos``).
                When provided, the model only requires ``coordinate``
                as input at call time.  When ``None``, all 8 tensors
                must be passed as positional arguments.
            num_points: Number of query coordinates (used only for
                the dummy forward pass that creates the parameter
                template).  Default 1 000.

        Returns:
            Model wrapping the PDEformer-2-Small model.

        Example::

            # With baked-in DAG (recommended):
            dag = pde.gen_dag(uf_num_mod=11)
            u = nn.pdeformer2_small(dag_inputs=dag)
            u.initialize("pdeformer2-small.msgpack")
            result = u(coordinate)  # only coordinate needed

            # Without DAG (pass all 8 inputs):
            u = nn.pdeformer2_small()
            u.initialize("pdeformer2-small.msgpack")
            result = u(node_type, node_scalar, node_function,
                       in_degree, out_degree, attn_bias,
                       spatial_pos, coordinate)

        Reference:
            Hao et al., "PDEformer 2" (2024)
        """
        from jax_pdeformer2 import PDEFORMER_SMALL_CONFIG

        return cls._pdeformer2(
            "pdeformer2_small",
            PDEFORMER_SMALL_CONFIG,
            dag_inputs,
            num_points,
        )

    @classmethod
    def pdeformer2_base(
        cls,
        dag_inputs: Optional[dict] = None,
        num_points: int = 1000,
    ) -> Model:
        """PDEformer-2-Base foundation model (~82 M parameters).

        See :meth:`pdeformer2_small` for full documentation.

        Example::

            dag = pde.gen_dag(uf_num_mod=11)
            u = nn.pdeformer2_base(dag_inputs=dag)
            u.initialize("pdeformer2-base.msgpack")
        """
        from jax_pdeformer2 import PDEFORMER_BASE_CONFIG

        return cls._pdeformer2(
            "pdeformer2_base",
            PDEFORMER_BASE_CONFIG,
            dag_inputs,
            num_points,
        )

    @classmethod
    def pdeformer2_fast(
        cls,
        dag_inputs: Optional[dict] = None,
        num_points: int = 1000,
    ) -> Model:
        """PDEformer-2-Fast foundation model (~71 M parameters).

        See :meth:`pdeformer2_small` for full documentation.

        Example::

            dag = pde.gen_dag(uf_num_mod=11)
            u = nn.pdeformer2_fast(dag_inputs=dag)
            u.initialize("pdeformer2-fast.msgpack")
        """
        from jax_pdeformer2 import PDEFORMER_FAST_CONFIG

        return cls._pdeformer2(
            "pdeformer2_fast",
            PDEFORMER_FAST_CONFIG,
            dag_inputs,
            num_points,
        )
