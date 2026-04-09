"""Functional helpers for traced expressions: ``jno.fn.sin(u)``, ``jno.fn.mse(pred, target)``.

This module is callable — ``jno.fn(my_func, [arg1, arg2])`` wraps an
arbitrary function into the tracing graph (replaces ``jno.np.function``).

Sections
--------
- **Math**: sin, cos, exp, log, sqrt, abs, …
- **Losses**: mse, mae, rmse, huber, log_cosh, relative_l2
- **PDEs**: poisson, heat, wave, burgers_1d, navier_stokes_incompressible_2d, …

Examples
--------
>>> import jno
>>> pde = jno.fn.sin(u) + jno.fn.exp(-x)
>>> loss = jno.fn.mse(pred, target)
>>> custom = jno.fn(lambda a, b: a ** 2 + b, [u, v])
"""

from __future__ import annotations

import inspect
import types
from typing import Sequence, TYPE_CHECKING
import jax.numpy as jnp

from .trace import Placeholder, FunctionCall, Variable
from .utils.adaptive import lrscheduler as _adaptive_lrscheduler
from .utils.adaptive import weights as _adaptive_weights

# ---------------------------------------------------------------------------
# Make the module callable: jno.fn(func, args, ...)
# ---------------------------------------------------------------------------

_MODULE_NAME = __name__


def _module_call(fn, args: list = [], name: str = "", reduces_axis: int = None):
    """Wrap an arbitrary function into the tracing graph.

    Args:
        fn: Any callable ``(*args) -> array``.
        args: Traced placeholder arguments.
        name: Optional display name in the expression tree.
        reduces_axis: If the function reduces an axis, specify it here.

    Returns:
        ``FunctionCall`` placeholder node.

    Example::

        custom = jno.fn(lambda a, b: a ** 2 + b, [u, v], name="my_op")
    """
    return FunctionCall(fn, args, name, reduces_axis)


def _to_scalar_loss(x):
    """Reduce an expression output to a scalar loss value."""
    arr = jnp.asarray(x)
    return jnp.squeeze(jnp.mean(arr))


if TYPE_CHECKING:
    # Static type stubs — Pylance/pyright reads this branch to provide
    # hover info and autocompletion.  Each method combines the balancer
    # configuration parameters with the traced-loss bridge parameters
    # (losses, mode, eps) so users see everything in a single call.
    from jno.trace import Placeholder as _Placeholder

    class _AdaptiveNamespace:
        """Adaptive weight balancing and learning rate scheduling.

        Single-call syntax::

            w = jno.fn.adaptive.relobralo([pde, bcs], mode="minmax")
            w0, w1 = jno.fn.adaptive.ReLoBRaLo([pde, bcs], alpha=0.5)
        """

        # -- Weight balancers (class names) ------------------------------------

        @staticmethod
        def ReLoBRaLo(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            alpha: float = 0.99,
            tau: float = 0.1,
            expected_rho: float = 0.999,
            seed: int = 42,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Adaptive loss balancing via relative residual scaling.

            Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for
            Physics-Informed Deep Learning" (2021/2025).

            At each training step:
              1. Compute softmax over loss ratios (current/previous) → local weights.
              2. Bernoulli coin flip selects global (vs initial losses) or history.
              3. EMA blend of local and historical weights → final lambda vector.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                alpha: EMA smoothing factor (default 0.99).
                tau: Temperature for softmax over loss ratios (default 0.1).
                expected_rho: Probability of using historical vs global weights (default 0.999).
                seed: Random seed for the Bernoulli coin flip.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def relobralo(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            alpha: float = 0.99,
            tau: float = 0.1,
            expected_rho: float = 0.999,
            seed: int = 42,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Adaptive loss balancing via relative residual scaling.

            Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for
            Physics-Informed Deep Learning" (2021/2025).

            At each training step:
              1. Compute softmax over loss ratios (current/previous) → local weights.
              2. Bernoulli coin flip selects global (vs initial losses) or history.
              3. EMA blend of local and historical weights → final lambda vector.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                alpha: EMA smoothing factor (default 0.99).
                tau: Temperature for softmax over loss ratios (default 0.1).
                expected_rho: Probability of using historical vs global weights (default 0.999).
                seed: Random seed for the Bernoulli coin flip.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def LbPINNsLossBalancing(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            init_s: float = 0.0,
            lr_s: float = 1e-2,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps_adam: float = 1e-8,
            s_min: float = -20.0,
            s_max: float = 20.0,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Self-adaptive loss balancing via learnable log-variances.

            Reference: Xiang, Peng, Liu & Yao, "Self-adaptive loss balanced
            Physics-informed neural networks" (2022).

            Learnable log-variances ``s_j = log(epsilon_j^2)`` yield weights
            ``w_j = 0.5 * exp(-s_j)``, updated host-side via Adam.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                init_s: Initial value(s) for the log-variance parameters.
                lr_s: Learning rate for the internal Adam optimizer on ``s``.
                beta1: Adam first-moment decay.
                beta2: Adam second-moment decay.
                eps_adam: Adam epsilon for numerical stability.
                s_min: Lower clamp for ``s`` values.
                s_max: Upper clamp for ``s`` values.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def lbpinns_loss_balancing(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            init_s: float = 0.0,
            lr_s: float = 1e-2,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps_adam: float = 1e-8,
            s_min: float = -20.0,
            s_max: float = 20.0,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Self-adaptive loss balancing via learnable log-variances.

            Reference: Xiang, Peng, Liu & Yao, "Self-adaptive loss balanced
            Physics-informed neural networks" (2022).

            Learnable log-variances ``s_j = log(epsilon_j^2)`` yield weights
            ``w_j = 0.5 * exp(-s_j)``, updated host-side via Adam.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                init_s: Initial value(s) for the log-variance parameters.
                lr_s: Learning rate for the internal Adam optimizer on ``s``.
                beta1: Adam first-moment decay.
                beta2: Adam second-moment decay.
                eps_adam: Adam epsilon for numerical stability.
                s_min: Lower clamp for ``s`` values.
                s_max: Upper clamp for ``s`` values.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def SoftAdapt(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            beta: float = 0.1,
            loss_floor: float = 1e-12,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Adaptive loss balancing via softmax over loss ratios.

            Reference: Heydari, Narang & Zweig, "SoftAdapt: Techniques for
            Adaptive Loss Weighting of Neural Networks with Multi-Part Loss
            Functions" (2019).

            Weights are ``N * softmax(L(t)/L(t-1) / beta)``.
            Losses not decreasing fast enough get more weight.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                beta: Temperature for the softmax (default 0.1).
                loss_floor: Minimum loss value to avoid division by zero.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def softadapt(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            beta: float = 0.1,
            loss_floor: float = 1e-12,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Adaptive loss balancing via softmax over loss ratios.

            Reference: Heydari, Narang & Zweig, "SoftAdapt: Techniques for
            Adaptive Loss Weighting of Neural Networks with Multi-Part Loss
            Functions" (2019).

            Weights are ``N * softmax(L(t)/L(t-1) / beta)``.
            Losses not decreasing fast enough get more weight.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                beta: Temperature for the softmax (default 0.1).
                loss_floor: Minimum loss value to avoid division by zero.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def DWA(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            temperature: float = 2.0,
            loss_floor: float = 1e-12,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Dynamic Weight Average for multi-task / multi-loss training.

            Reference: Liu, Johns & Davison, "End-to-End Multi-Task Learning
            with Attention" (CVPR 2019).

            Weights are ``N * softmax(r / T)`` where ``r_k = L_k(t-1) / L_k(t-2)``.
            Tasks whose loss is increasing (ratio > 1) are up-weighted.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                temperature: Softmax temperature (default 2.0).
                loss_floor: Minimum loss value to avoid division by zero.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def dwa(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            temperature: float = 2.0,
            loss_floor: float = 1e-12,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Dynamic Weight Average for multi-task / multi-loss training.

            Reference: Liu, Johns & Davison, "End-to-End Multi-Task Learning
            with Attention" (CVPR 2019).

            Weights are ``N * softmax(r / T)`` where ``r_k = L_k(t-1) / L_k(t-2)``.
            Tasks whose loss is increasing (ratio > 1) are up-weighted.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                temperature: Softmax temperature (default 2.0).
                loss_floor: Minimum loss value to avoid division by zero.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def RLW(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            alpha: float = 1.0,
            seed: int = 42,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Random Loss Weighting via symmetric Dirichlet sampling.

            Reference: Lin, Ye, Xu, Shi & Zhang, "Reasonable Effectiveness
            of Random Weighting: A Litmus Test for Multi-Task Learning"
            (TMLR 2021).

            At each step, sample ``w ~ Dirichlet(alpha, ..., alpha)`` scaled by N.
            Surprisingly competitive with sophisticated adaptive methods.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                alpha: Dirichlet concentration parameter (default 1.0 = uniform).
                seed: Random seed.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        @staticmethod
        def rlw(
            losses: list[_Placeholder],
            *context_terms: _Placeholder,
            alpha: float = 1.0,
            seed: int = 42,
            mode: str = "raw",
            eps: float = 1e-12,
        ) -> tuple[_Placeholder, ...]:
            """Random Loss Weighting via symmetric Dirichlet sampling.

            Reference: Lin, Ye, Xu, Shi & Zhang, "Reasonable Effectiveness
            of Random Weighting: A Litmus Test for Multi-Task Learning"
            (TMLR 2021).

            At each step, sample ``w ~ Dirichlet(alpha, ..., alpha)`` scaled by N.
            Surprisingly competitive with sophisticated adaptive methods.

            Args:
                losses: List of traced loss placeholders, e.g. ``[pde, bcs]``.
                *context_terms: Additional traced variables (optional).
                alpha: Dirichlet concentration parameter (default 1.0 = uniform).
                seed: Random seed.
                mode: Loss preprocessing — ``"raw"``, ``"minmax"``, or ``"l2"``.
                eps: Floor value for numerical stability.

            Returns:
                Tuple of traced weight placeholders, one per loss.
            """
            ...

        # -- Non-weight pass-throughs -----------------------------------------

        WeightSchedule = _adaptive_weights.WeightSchedule

        # lrscheduler
        LearningRateSchedule = _adaptive_lrscheduler.LearningRateSchedule
        DLRS = _adaptive_lrscheduler.DLRS
        dlrs = staticmethod(_adaptive_lrscheduler.dlrs)

else:

    class _AdaptiveNamespace:
        """Dynamic bridge for symbols exported by adaptive weights/LR modules.

        Any symbol added to ``weights.py`` or ``lrscheduler.py`` and listed in
        that module's ``__all__`` is automatically available under
        ``jno.fn.adaptive.<Symbol>``.

        Weight balancers accept a list of traced losses as the first argument::

            w = jno.fn.adaptive.relobralo([pde, bcs], mode="minmax")
        """

        _TRACE_WEIGHT_BUILDERS_EXCLUDE = {"WeightSchedule"}

        def __init__(self):
            self._registry = {}
            self._source_kind = {}

            for mod, kind in ((_adaptive_weights, "weights"), (_adaptive_lrscheduler, "lrscheduler")):
                for name in getattr(mod, "__all__", []):
                    if hasattr(mod, name):
                        self._registry[name] = getattr(mod, name)
                        self._source_kind[name] = kind

        def __dir__(self):
            return sorted(self._registry.keys())

        def __getattr__(self, name):
            # Guard against recursion during unpickling / early access
            if name.startswith("_"):
                raise AttributeError(name)
            try:
                registry = object.__getattribute__(self, "_registry")
            except AttributeError:
                raise AttributeError(name)
            if name not in registry:
                raise AttributeError(f"jno.fn.adaptive has no attribute '{name}'")

            obj = registry[name]
            if not callable(obj):
                return obj

            return self._make_bridge(name, obj, self._source_kind[name])

        def _make_bridge(self, name: str, obj, source_kind: str):
            def bridge(*args, **kwargs):
                # Weight balancer: single call with losses list as first arg
                losses = args[0] if args else kwargs.get("losses")
                if source_kind == "weights" and name not in self._TRACE_WEIGHT_BUILDERS_EXCLUDE and losses is not None and isinstance(losses, (list, tuple)) and len(losses) > 0 and all(isinstance(x, Placeholder) for x in losses):
                    context_terms = args[1:]
                    mode = kwargs.pop("mode", "raw")
                    eps = float(kwargs.pop("eps", 1e-12))
                    kwargs.pop("losses", None)
                    n = len(losses)
                    balancer = self._instantiate_weight_balancer(obj, n, kwargs)

                    def _weights_fn(*vals):
                        loss_vals = vals[:n]
                        processed = self._stack_loss_values(loss_vals, mode=mode, eps=eps)
                        weight_tuple = balancer(processed)
                        return jnp.stack(weight_tuple)

                    weight_vec = FunctionCall(
                        _weights_fn,
                        list(losses) + list(context_terms),
                        name=f"adaptive_{name.lower()}",
                    )
                    return tuple(weight_vec[i] for i in range(n))

                # plain pass-through (LR schedulers, WeightSchedule, etc.)
                return obj(*args, **kwargs)

            bridge.__name__ = name
            bridge.__doc__ = getattr(obj, "__doc__", None)
            return bridge

        @staticmethod
        def _instantiate_weight_balancer(symbol, n_losses, kwargs: dict):
            ctor_kwargs = dict(kwargs)

            if inspect.isclass(symbol):
                try:
                    sig = inspect.signature(symbol.__init__)
                    if "num_losses" in sig.parameters and "num_losses" not in ctor_kwargs:
                        ctor_kwargs["num_losses"] = n_losses
                except (TypeError, ValueError):
                    pass
                return symbol(**ctor_kwargs)

            # Factory function → call it, returns an instance
            instance = symbol(**ctor_kwargs)

            if hasattr(instance, "num_losses") and getattr(instance, "num_losses", None) is None:
                try:
                    instance.num_losses = n_losses
                except Exception:
                    pass

            if hasattr(instance, "lambdas") and getattr(instance, "lambdas", None) is None:
                try:
                    import numpy as _np

                    instance.lambdas = _np.ones((n_losses,), dtype=_np.float32)
                except Exception:
                    pass
            return instance

        @staticmethod
        def _stack_loss_values(loss_values, mode: str, eps: float):
            vals = jnp.stack([_to_scalar_loss(v) for v in loss_values], axis=0)
            vals = jnp.maximum(vals, eps)

            mode_l = str(mode).lower()
            if mode_l in ("none", "raw"):
                return vals
            if mode_l == "minmax":
                vmin = jnp.min(vals)
                vmax = jnp.max(vals)
                return (vals - vmin) / jnp.maximum(vmax - vmin, eps)
            if mode_l == "l2":
                return vals / jnp.maximum(jnp.linalg.norm(vals), eps)
            raise ValueError(f"Unsupported adaptive loss mode '{mode}'.")


adaptive = _AdaptiveNamespace()


# ---------------------------------------------------------------------------
# Unary / binary factories (same as jnp_ops but local)
# ---------------------------------------------------------------------------


def _unary(jnp_fn):
    def wrapper(x):
        return FunctionCall(jnp_fn, [x])

    wrapper.__name__ = jnp_fn.__name__
    wrapper.__doc__ = f"Element-wise ``{jnp_fn.__name__}`` on a traced expression."
    return wrapper


def _binary(jnp_fn):
    def wrapper(x, y):
        return FunctionCall(jnp_fn, [x, y])

    wrapper.__name__ = jnp_fn.__name__
    wrapper.__doc__ = f"Element-wise ``{jnp_fn.__name__}`` on traced expressions."
    return wrapper


# ============================================================================
# Math — trigonometric
# ============================================================================
sin = _unary(jnp.sin)
cos = _unary(jnp.cos)
tan = _unary(jnp.tan)
arcsin = _unary(jnp.arcsin)
arccos = _unary(jnp.arccos)
arctan = _unary(jnp.arctan)
arctan2 = _binary(jnp.arctan2)
atan2 = arctan2

# ============================================================================
# Math — hyperbolic
# ============================================================================
sinh = _unary(jnp.sinh)
cosh = _unary(jnp.cosh)
tanh = _unary(jnp.tanh)
arcsinh = _unary(jnp.arcsinh)
arccosh = _unary(jnp.arccosh)
arctanh = _unary(jnp.arctanh)

# ============================================================================
# Math — exponential / logarithmic
# ============================================================================
exp = _unary(jnp.exp)
exp2 = _unary(jnp.exp2)
expm1 = _unary(jnp.expm1)
log = _unary(jnp.log)
log2 = _unary(jnp.log2)
log10 = _unary(jnp.log10)
log1p = _unary(jnp.log1p)

# ============================================================================
# Math — power / root / rounding
# ============================================================================
sqrt = _unary(jnp.sqrt)
cbrt = _unary(jnp.cbrt)
square = _unary(jnp.square)
power = _binary(jnp.power)
abs = _unary(jnp.abs)
floor = _unary(jnp.floor)
ceil = _unary(jnp.ceil)
round = _unary(jnp.round)
sign = _unary(jnp.sign)

# ============================================================================
# Math — comparison
# ============================================================================
maximum = _binary(jnp.maximum)
minimum = _binary(jnp.minimum)


# ============================================================================
# Loss functions
# ============================================================================


def mse(pred: Placeholder, target: Placeholder) -> FunctionCall:
    """Mean Squared Error: ``mean((pred - target)²)``."""
    return FunctionCall(
        lambda p, t: jnp.squeeze(jnp.mean(jnp.square(p - t))),
        [pred, target],
        name="mse",
        reduces_axis=True,
    )


def mae(pred: Placeholder, target: Placeholder) -> FunctionCall:
    """Mean Absolute Error: ``mean(|pred - target|)``."""
    return FunctionCall(
        lambda p, t: jnp.squeeze(jnp.mean(jnp.abs(p - t))),
        [pred, target],
        name="mae",
        reduces_axis=True,
    )


def rmse(pred: Placeholder, target: Placeholder) -> FunctionCall:
    """Root Mean Squared Error: ``sqrt(mean((pred - target)²))``."""
    return FunctionCall(
        lambda p, t: jnp.squeeze(jnp.sqrt(jnp.mean(jnp.square(p - t)))),
        [pred, target],
        name="rmse",
        reduces_axis=True,
    )


def huber(pred: Placeholder, target: Placeholder, delta: float = 1.0) -> FunctionCall:
    """Huber loss (smooth L1): quadratic for small errors, linear for large.

    Args:
        pred: Predicted values.
        target: Target values.
        delta: Threshold where loss transitions from quadratic to linear.
    """

    def _huber(p, t, _d=delta):
        r = jnp.abs(p - t)
        return jnp.squeeze(jnp.mean(jnp.where(r <= _d, 0.5 * r**2, _d * (r - 0.5 * _d))))

    return FunctionCall(_huber, [pred, target], name="huber", reduces_axis=True)


def log_cosh(pred: Placeholder, target: Placeholder) -> FunctionCall:
    """Log-cosh loss: ``mean(log(cosh(pred - target)))``."""
    return FunctionCall(
        lambda p, t: jnp.squeeze(jnp.mean(jnp.log(jnp.cosh(p - t)))),
        [pred, target],
        name="log_cosh",
        reduces_axis=True,
    )


def relative_l2(pred: Placeholder, target: Placeholder, eps: float = 1e-8) -> FunctionCall:
    """Relative L2 error: ``||pred - target||₂ / (||target||₂ + eps)``."""

    def _rel_l2(p, t, _eps=eps):
        num = jnp.sqrt(jnp.sum(jnp.square(p - t)))
        den = jnp.sqrt(jnp.sum(jnp.square(t))) + _eps
        return num / den

    return FunctionCall(_rel_l2, [pred, target], name="relative_l2", reduces_axis=True)


def relative_l1(pred: Placeholder, target: Placeholder, eps: float = 1e-8) -> FunctionCall:
    """Relative L1 error: ``||pred - target||₁ / (||target||₁ + eps)``."""

    def _rel_l1(p, t, _eps=eps):
        num = jnp.sum(jnp.abs(p - t))
        den = jnp.sum(jnp.abs(t)) + _eps
        return num / den

    return FunctionCall(_rel_l1, [pred, target], name="relative_l1", reduces_axis=True)


def nmse(pred: Placeholder, target: Placeholder, eps: float = 1e-8) -> FunctionCall:
    """Normalized MSE: ``mean((pred - target)²) / (var(target) + eps)``."""

    def _nmse(p, t, _eps=eps):
        return jnp.mean(jnp.square(p - t)) / (jnp.var(t) + _eps)

    return FunctionCall(_nmse, [pred, target], name="nmse", reduces_axis=True)


def mape(pred: Placeholder, target: Placeholder, eps: float = 1e-8) -> FunctionCall:
    """Mean Absolute Percentage Error: ``mean(|pred - target| / (|target| + eps))``."""

    def _mape(p, t, _eps=eps):
        return jnp.squeeze(jnp.mean(jnp.abs(p - t) / (jnp.abs(t) + _eps)))

    return FunctionCall(_mape, [pred, target], name="mape", reduces_axis=True)


def linf(pred: Placeholder, target: Placeholder) -> FunctionCall:
    """L-infinity (max absolute) error: ``max(|pred - target|)``."""
    return FunctionCall(
        lambda p, t: jnp.max(jnp.abs(p - t)),
        [pred, target],
        name="linf",
        reduces_axis=True,
    )


def quantile(pred: Placeholder, target: Placeholder, tau: float = 0.5) -> FunctionCall:
    """Quantile (pinball) loss: asymmetric penalty for under/over-prediction.

    Args:
        pred: Predicted values.
        target: Target values.
        tau: Quantile level in ``(0, 1)``.  ``0.5`` gives MAE.
    """

    def _quantile(p, t, _tau=tau):
        r = t - p
        return jnp.squeeze(jnp.mean(jnp.maximum(_tau * r, (_tau - 1.0) * r)))

    return FunctionCall(_quantile, [pred, target], name="quantile", reduces_axis=True)


def cauchy(pred: Placeholder, target: Placeholder, c: float = 1.0) -> FunctionCall:
    """Cauchy (Lorentzian) loss: robust to outliers, heavier tails than Huber.

    ``mean(log(1 + ((pred - target) / c)²))``

    Args:
        pred: Predicted values.
        target: Target values.
        c: Scale parameter controlling outlier sensitivity.
    """

    def _cauchy(p, t, _c=c):
        return jnp.squeeze(jnp.mean(jnp.log1p(jnp.square((p - t) / _c))))

    return FunctionCall(_cauchy, [pred, target], name="cauchy", reduces_axis=True)


def smooth_l1(pred: Placeholder, target: Placeholder, beta: float = 1.0) -> FunctionCall:
    """Smooth L1 loss (parameterized Huber variant used in object detection).

    Quadratic when ``|error| < beta``, linear otherwise.

    Args:
        pred: Predicted values.
        target: Target values.
        beta: Transition point.
    """

    def _smooth_l1(p, t, _b=beta):
        r = jnp.abs(p - t)
        return jnp.squeeze(jnp.mean(jnp.where(r < _b, 0.5 * r**2 / _b, r - 0.5 * _b)))

    return FunctionCall(_smooth_l1, [pred, target], name="smooth_l1", reduces_axis=True)


# ============================================================================
# PDE residual helpers
# ============================================================================


def _advection(u: Placeholder, velocity: Sequence[Placeholder | float], variables: Sequence[Variable], scheme: str) -> Placeholder:
    if len(velocity) != len(variables):
        raise ValueError("velocity and variables must have the same length")
    adv = velocity[0] * u.d(variables[0], scheme=scheme)
    for i in range(1, len(variables)):
        adv = adv + velocity[i] * u.d(variables[i], scheme=scheme)
    return adv


def poisson(
    u: Placeholder,
    forcing: Placeholder | float = 0.0,
    *variables: Variable,
    diffusion: float = 1.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Poisson residual: ``-diffusion * Δu - forcing``."""
    return -diffusion * u.laplacian(*variables, scheme=scheme) - forcing


def helmholtz(
    u: Placeholder,
    forcing: Placeholder | float = 0.0,
    *variables: Variable,
    wave_number: float = 1.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Helmholtz residual: ``-Δu - k²u - forcing``."""
    return -u.laplacian(*variables, scheme=scheme) - (wave_number**2) * u - forcing


def heat(
    u: Placeholder,
    t: Variable,
    *spatial: Variable,
    diffusivity: float = 1.0,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Heat-equation residual: ``u_t - diffusivity * Δu - source``."""
    return u.d(t, scheme=scheme) - diffusivity * u.laplacian(*spatial, scheme=scheme) - source


def diffusion(
    u: Placeholder,
    *spatial: Variable,
    t: Variable | None = None,
    diffusivity: float = 1.0,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Diffusion residual: ``u_t - diffusivity * Δu - source`` (or steady when ``t=None``)."""
    residual: Placeholder | float = 0.0
    if t is not None:
        residual = residual + u.d(t, scheme=scheme)
    return residual - diffusivity * u.laplacian(*spatial, scheme=scheme) - source


def advection(
    u: Placeholder,
    velocity: Sequence[Placeholder | float],
    *spatial: Variable,
    t: Variable | None = None,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Advection residual: ``u_t + v·∇u - source`` (or steady when ``t=None``)."""
    if len(spatial) == 0:
        raise ValueError("advection requires at least one spatial variable")

    residual: Placeholder | float = 0.0
    if t is not None:
        residual = residual + u.d(t, scheme=scheme)
    residual = residual + _advection(u, velocity, spatial, scheme=scheme)
    return residual - source


def wave(
    u: Placeholder,
    t: Variable,
    *spatial: Variable,
    speed: float = 1.0,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Wave-equation residual: ``u_tt - speed² * Δu - source``."""
    return u.d(t, scheme=scheme).d(t, scheme=scheme) - (speed**2) * u.laplacian(*spatial, scheme=scheme) - source


def burgers_1d(
    u: Placeholder,
    x: Variable,
    t: Variable,
    viscosity: float = 0.01 / jnp.pi,
    forcing: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """1D viscous Burgers residual: ``u_t + u u_x - viscosity * u_xx - forcing``."""
    return u.d(t, scheme=scheme) + u * u.d(x, scheme=scheme) - viscosity * u.d(x, scheme=scheme).d(x, scheme=scheme) - forcing


def burgers_inviscid_1d(
    u: Placeholder,
    x: Variable,
    t: Variable,
    forcing: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """1D inviscid Burgers (shock-forming) residual: ``u_t + ∂x(0.5 u²) - forcing``."""
    flux = 0.5 * u**2
    return u.d(t, scheme=scheme) + flux.d(x, scheme=scheme) - forcing


def biharmonic(
    u: Placeholder,
    forcing: Placeholder | float = 0.0,
    *variables: Variable,
    stiffness: float = 1.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Biharmonic residual: ``stiffness * Δ²u - forcing``."""
    lap_u = u.laplacian(*variables, scheme=scheme)
    return stiffness * lap_u.laplacian(*variables, scheme=scheme) - forcing


def advection_diffusion_reaction(
    u: Placeholder,
    velocity: Sequence[Placeholder | float],
    *spatial: Variable,
    t: Variable | None = None,
    diffusivity: float = 1.0,
    reaction_coeff: float = 0.0,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """ADR residual: ``u_t + v·∇u - diffusivity * Δu + reaction_coeff * u - source``."""
    if len(spatial) == 0:
        raise ValueError("advection_diffusion_reaction requires at least one spatial variable")

    residual: Placeholder | float = 0.0
    if t is not None:
        residual = residual + u.d(t, scheme=scheme)
    residual = residual + _advection(u, velocity, spatial, scheme=scheme)
    residual = residual - diffusivity * u.laplacian(*spatial, scheme=scheme)
    residual = residual + reaction_coeff * u
    return residual - source


def allen_cahn(
    u: Placeholder,
    t: Variable,
    *spatial: Variable,
    epsilon: float = 1e-2,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Allen-Cahn residual: ``u_t - epsilon² * Δu + u³ - u - source``."""
    return u.d(t, scheme=scheme) - (epsilon**2) * u.laplacian(*spatial, scheme=scheme) + u**3 - u - source


def fisher_kpp(
    u: Placeholder,
    t: Variable,
    *spatial: Variable,
    diffusivity: float = 1.0,
    growth_rate: float = 1.0,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Fisher-KPP residual: ``u_t - diffusivity * Δu - growth_rate * u * (1 - u) - source``."""
    return u.d(t, scheme=scheme) - diffusivity * u.laplacian(*spatial, scheme=scheme) - growth_rate * u * (1.0 - u) - source


def cahn_hilliard(
    u: Placeholder,
    t: Variable,
    *spatial: Variable,
    epsilon: float = 1e-2,
    mobility: float = 1.0,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Cahn-Hilliard residual: ``u_t - mobility * Δ(-epsilon² Δu + u³ - u) - source``."""
    chemical_potential = -((epsilon**2) * u.laplacian(*spatial, scheme=scheme)) + u**3 - u
    return u.d(t, scheme=scheme) - mobility * chemical_potential.laplacian(*spatial, scheme=scheme) - source


def navier_stokes_incompressible_2d(
    u: Placeholder,
    v: Placeholder,
    p: Placeholder,
    x: Variable,
    y: Variable,
    t: Variable | None = None,
    density: float = 1.0,
    viscosity: float = 1e-2,
    forcing_x: Placeholder | float = 0.0,
    forcing_y: Placeholder | float = 0.0,
    continuity_source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> tuple[Placeholder, Placeholder, Placeholder]:
    """2D incompressible Navier-Stokes residuals ``(mom_x, mom_y, continuity)``."""
    ut: Placeholder | float = 0.0
    vt: Placeholder | float = 0.0
    if t is not None:
        ut = u.d(t, scheme=scheme)
        vt = v.d(t, scheme=scheme)

    mom_x = density * (ut + u * u.d(x, scheme=scheme) + v * u.d(y, scheme=scheme))
    mom_x = mom_x + p.d(x, scheme=scheme) - viscosity * u.laplacian(x, y, scheme=scheme) - forcing_x

    mom_y = density * (vt + u * v.d(x, scheme=scheme) + v * v.d(y, scheme=scheme))
    mom_y = mom_y + p.d(y, scheme=scheme) - viscosity * v.laplacian(x, y, scheme=scheme) - forcing_y

    mass = continuity([u, v], x, y, source=continuity_source, scheme=scheme)
    return mom_x, mom_y, mass


def maxwell_3d(
    ex: Placeholder,
    ey: Placeholder,
    ez: Placeholder,
    bx: Placeholder,
    by: Placeholder,
    bz: Placeholder,
    x: Variable,
    y: Variable,
    z: Variable,
    t: Variable,
    current_x: Placeholder | float = 0.0,
    current_y: Placeholder | float = 0.0,
    current_z: Placeholder | float = 0.0,
    charge_density: Placeholder | float = 0.0,
    permittivity: float = 1.0,
    permeability: float = 1.0,
    scheme: str = "automatic_differentiation",
) -> tuple[Placeholder, Placeholder, Placeholder, Placeholder, Placeholder, Placeholder, Placeholder, Placeholder]:
    """3D Maxwell residuals: Faraday, Ampere-Maxwell, Gauss(E), Gauss(B).

    Returns:
        Tuple ``(faraday_x, faraday_y, faraday_z,
        ampere_x, ampere_y, ampere_z, gauss_e, gauss_b)``.
    """
    c2 = 1.0 / (permittivity * permeability)

    curl_e_x = ez.d(y, scheme=scheme) - ey.d(z, scheme=scheme)
    curl_e_y = ex.d(z, scheme=scheme) - ez.d(x, scheme=scheme)
    curl_e_z = ey.d(x, scheme=scheme) - ex.d(y, scheme=scheme)

    curl_b_x = bz.d(y, scheme=scheme) - by.d(z, scheme=scheme)
    curl_b_y = bx.d(z, scheme=scheme) - bz.d(x, scheme=scheme)
    curl_b_z = by.d(x, scheme=scheme) - bx.d(y, scheme=scheme)

    faraday_x = bx.d(t, scheme=scheme) + curl_e_x
    faraday_y = by.d(t, scheme=scheme) + curl_e_y
    faraday_z = bz.d(t, scheme=scheme) + curl_e_z

    ampere_x = ex.d(t, scheme=scheme) - c2 * curl_b_x + current_x / permittivity
    ampere_y = ey.d(t, scheme=scheme) - c2 * curl_b_y + current_y / permittivity
    ampere_z = ez.d(t, scheme=scheme) - c2 * curl_b_z + current_z / permittivity

    gauss_e = ex.d(x, scheme=scheme) + ey.d(y, scheme=scheme) + ez.d(z, scheme=scheme) - charge_density / permittivity
    gauss_b = bx.d(x, scheme=scheme) + by.d(y, scheme=scheme) + bz.d(z, scheme=scheme)

    return (
        faraday_x,
        faraday_y,
        faraday_z,
        ampere_x,
        ampere_y,
        ampere_z,
        gauss_e,
        gauss_b,
    )


def continuity(
    velocity_field: Sequence[Placeholder],
    *variables: Variable,
    source: Placeholder | float = 0.0,
    scheme: str = "automatic_differentiation",
) -> Placeholder:
    """Incompressible continuity residual: ``div(velocity_field) - source``."""
    if len(velocity_field) == 0:
        raise ValueError("velocity_field must contain at least one component")
    if len(velocity_field) != len(variables):
        raise ValueError("velocity_field and variables must have the same length")

    div_u = velocity_field[0].d(variables[0], scheme=scheme)
    for i in range(1, len(variables)):
        div_u = div_u + velocity_field[i].d(variables[i], scheme=scheme)
    return div_u - source


# ============================================================================
# Make this module callable
# ============================================================================


class _CallableModule(types.ModuleType):
    """Module wrapper that makes ``import jno.fn; jno.fn(...)`` work."""

    def __call__(self, fn, args=[], name="", reduces_axis=None):
        return _module_call(fn, args, name, reduces_axis)


import sys as _sys

_self = _sys.modules[_MODULE_NAME]
_mod = _CallableModule(_MODULE_NAME, __doc__)
_mod.__dict__.update({k: v for k, v in _self.__dict__.items() if not k.startswith("_")})
_mod.__dict__["__file__"] = __file__
_mod.__dict__["__package__"] = __package__
_mod.__dict__["_module_call"] = _module_call
_sys.modules[_MODULE_NAME] = _mod
