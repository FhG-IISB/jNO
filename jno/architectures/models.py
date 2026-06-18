"""
Neural Network Wrapping Module
==============================

Provides ``nn.wrap()`` for integrating arbitrary Equinox / Flax modules into the
jno training pipeline, plus helpers ``parameter()`` and ``set_default_rng_seed()``.

Architecture factories (``mlp``, ``fno2d``, ``deeponet``, …) have moved to the
``foundax`` package.  Use ``jno.nn.wrap(foundax.mlp(...))`` instead of the former
``jno.nn.mlp(...)`` shorthand.
"""

from typing import Any, Union, overload

import equinox as eqx
import jax
import jax.numpy as jnp

from ..trace import Model, ModelCall, TunableModule
from ..tuner import ArchSpace
from ..utils.config import get_seed
from .linear import Linear

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
            seed = 21  # make this consistent with jno.core
        _DEFAULT_NN_KEY = jax.random.PRNGKey(int(seed))

    if _DEFAULT_NN_KEY is None:
        raise ValueError(
            "No PRNG key provided. Pass key=... explicitly, or set [jno].seed "
            "and call jno.setup(...) before creating models."
        )

    _DEFAULT_NN_KEY, key_out = jax.random.split(_DEFAULT_NN_KEY)
    return key_out


def parameter(shape: tuple, *, key: jax.Array | None = None, name: str = "value") -> ModelCall:
    """
    Create a trainable parameter tensor.

    Useful for learning scalar or tensor constants during optimization,
    such as PDE coefficients or normalization factors.

    Args:
        shape: Shape of the parameter tensor.
        key: Optional JAX PRNG key used as default when calling
            ``.initialize(<jax initializer>)`` later.
        name: Name of the parameter model.

    Returns:
        ModelCall: A scalar/tensor parameter expression that can be optimized by jno.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> # Learnable diffusion coefficient
        >>> D = parameter((1,), key=key).initialize(jax.nn.initializers.ones)

        >>> # Learnable 2D tensor
        >>> K = parameter((3, 3), key=key)
    """

    # Field parameter: pass a FEM symbol (``u``/``phi`` from ``fem_symbols()``)
    # instead of a shape tuple to get a **P1 nodal coefficient field** sized to that
    # symbol's FE space (one trainable value per mesh node). The symbol carries its
    # owning domain (see ``variational_symbols``), so the size is known up front.
    fem_field = None
    fem_field_key = None
    if hasattr(shape, "field_key") and getattr(shape, "_domain", None) is not None:
        sym = shape
        if int(getattr(sym, "order", 1)) != 1:
            raise NotImplementedError(
                "jno.np.parameter(<symbol>): nodal field parameters support P1 (order=1) "
                "symbols only for now (higher-order spaces add nodes during init_fem)."
            )
        mesh = getattr(sym._domain, "built_mesh", None)
        if mesh is None:
            raise ValueError("jno.np.parameter(<symbol>): the symbol's domain has no mesh yet.")
        shape = (int(mesh.points.shape[0]),)
        fem_field = "node"
        fem_field_key = getattr(sym, "field_key", None)

    class _Parameter(eqx.Module):
        value: jnp.ndarray

        def __call__(self):
            return self.value

    model = nn.wrap(_Parameter(value=jnp.zeros(shape, dtype=jnp.float32)), name=name)
    # Metadata used by the FEM-time lowering route to distinguish trainable
    # physical coefficients from ordinary neural-network ModelCall nodes.
    model._is_parameter = True
    model._parameter_name = str(name)
    # A nodal field coefficient is re-assembled (parameter inside the integrand,
    # interpolated to quad points), not factored into a constant basis.
    model._fem_field = fem_field
    model._fem_field_key = fem_field_key

    model._initializer_key = _resolve_key(key)
    # Mark this Model so .posterior_samples unwraps the single-leaf wrapper
    # into the plain array a user expects from `.parameter()` semantics.
    model._is_jno_scalar_parameter = True
    return model()


class nn:
    """
    Neural network wrapping class for integrating modules into the jno pipeline.

    Use ``nn.wrap(module)`` (or the shorthand ``nn(module)``) to wrap an
    Equinox, Flax Linen, or Flax NNX module into a ``Model`` that works with
    ``jno.core``.

    Architecture factories have moved to the ``foundax`` package::

        import foundax
        model = jno.nn.wrap(foundax.mlp(2, hidden_dims=64, key=key))
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
    def wrap(
        cls, module: Any, space: "ArchSpace" = None, name: str = "", weight_path: str = None
    ) -> Union["Model", "TunableModule"]:
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
            >>> import foundax
            >>> model = nn.wrap(foundax.mlp(2, output_dim=1, key=jax.random.PRNGKey(0)))
        """
        if space is not None:
            if isinstance(module, type):
                return TunableModule(module_cls=module, space=space)
            else:
                raise ValueError(
                    "When space= is provided, module must be a CLASS (not instance). "
                    "Use: pnp.nn.wrap(MLP, space=space) not pnp.nn.wrap(MLP(), space=space)"
                )
        else:
            if not isinstance(module, eqx.Module):
                raise TypeError(
                    f"nn.wrap() expects an eqx.Module instance, got {type(module).__name__}. "
                    f"Flax modules are no longer supported at runtime. "
                    f"Use the Equinox version of your model (e.g. foundax provides *_eqx variants)."
                )
            return Model(module, name, weight_path)
