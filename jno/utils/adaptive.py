from typing import Callable, List, Union, Sequence
import jax.numpy as jnp


def _default_float_dtype():
    """Return JAX's current default floating dtype (float32 or float64)."""
    return jnp.asarray(0.0).dtype


LRFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
WeightFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class LearningRateSchedule:
    """
    Stateless learning rate schedule.

    Wraps either:
      - a constant scalar, or
      - a function (t, losses) -> scalar

    and clamps the result to [min_lr, max_lr].

    This is JAX- and jit-compatible as long as the underlying function
    uses only JAX operations.
    """

    def __init__(
        self,
        fn: Union[float, LRFunction],
        min_lr: float = 1e-10,
        max_lr: float = 1.0,
    ):
        if isinstance(fn, (int, float)):
            const = float(fn)

            def _fn(t, losses):
                return jnp.asarray(const)

            self.fn: LRFunction = _fn
        else:
            self.fn = fn

        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)

    def __call__(self, t, losses: jnp.ndarray) -> jnp.ndarray:
        lr = self.fn(t, losses)
        lr = jnp.asarray(lr)
        lr = jnp.clip(lr, self.min_lr, self.max_lr)
        return lr

    # -------------------------
    # Classmethod constructors
    # -------------------------

    @classmethod
    def constant(cls, lr: float, *, min_lr: float = 1e-10, max_lr: float = 1.0):
        """Constant learning rate."""
        return cls(float(lr), min_lr=min_lr, max_lr=max_lr)

    @classmethod
    def cosine(
        cls,
        total_steps: int,
        lr0: float,
        lr_end: float = 0.0,
        *,
        min_lr: float = 1e-10,
        max_lr: float = 1.0,
    ):
        """Cosine decay from lr0 -> lr_end over total_steps."""
        T = float(total_steps)
        lr0 = float(lr0)
        lr_end = float(lr_end)

        def fn(t, losses):
            t = jnp.asarray(t, dtype=_default_float_dtype())
            frac = jnp.clip(t / T, 0.0, 1.0)
            # cosine from 1 -> 0
            cos = 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
            return lr_end + (lr0 - lr_end) * cos

        return cls(fn, min_lr=min_lr, max_lr=max_lr)

    @classmethod
    def warmup_cosine(
        cls,
        total_steps: int,
        warmup_steps: int,
        lr0: float = 1e-3,
        lr_end: float = 1e-6,
        *,
        min_lr: float = 1e-10,
        max_lr: float = 1.0,
    ):
        """Linear warmup to lr0, then cosine decay to lr_end."""
        T = int(total_steps)
        W = int(warmup_steps)
        lr0 = float(lr0)
        lr_end = float(lr_end)

        def fn(t, losses):
            t = jnp.asarray(t, dtype=_default_float_dtype())

            warm = lr0 * (t + 1.0) / jnp.maximum(1.0, float(W))

            # cosine phase fraction in [0,1]
            denom = jnp.maximum(1.0, float(T - W))
            frac = jnp.clip((t - float(W)) / denom, 0.0, 1.0)
            cos = 0.5 * (1.0 + jnp.cos(jnp.pi * frac))
            decayed = lr_end + (lr0 - lr_end) * cos

            return jnp.where(t < float(W), warm, decayed)

        return cls(fn, min_lr=min_lr, max_lr=max_lr)

    @classmethod
    def exponential(
        cls,
        lr0: float,
        decay_rate: float,
        decay_steps: int,
        lr_end: float = 0.0,
        *,
        staircase: bool = False,
        min_lr: float = 1e-10,
        max_lr: float = 1.0,
    ):
        """
        Exponential decay:
          lr(t) = max(lr_end, lr0 * decay_rate^(t/decay_steps))
        If staircase=True uses floor(t/decay_steps).
        """
        lr0 = float(lr0)
        decay_rate = float(decay_rate)
        decay_steps = float(decay_steps)  # type: ignore[assignment]
        lr_end = float(lr_end)

        def fn(t, losses):
            t = jnp.asarray(t, dtype=_default_float_dtype())
            p = t / decay_steps
            if staircase:
                p = jnp.floor(p)
            lr = lr0 * (decay_rate**p)
            return jnp.maximum(lr_end, lr)

        return cls(fn, min_lr=min_lr, max_lr=max_lr)

    @classmethod
    def piecewise_constant(
        cls,
        boundaries: Sequence[int],
        values: Sequence[float],
        *,
        min_lr: float = 1e-10,
        max_lr: float = 1.0,
    ):
        """
        Piecewise constant schedule.

        boundaries: increasing step indices, e.g. [1000, 5000]
        values: len(boundaries)+1 values, e.g. [5e-4, 2e-4, 5e-5]
        """
        b = jnp.asarray(list(boundaries), dtype=jnp.int32)
        v = jnp.asarray(list(values), dtype=_default_float_dtype())
        if len(values) != len(boundaries) + 1:
            raise ValueError("values must have length len(boundaries)+1")

        def fn(t, losses):
            t = jnp.asarray(t, dtype=jnp.int32)
            idx = jnp.sum(t >= b)  # 0..len(boundaries)
            return v[idx]

        return cls(fn, min_lr=min_lr, max_lr=max_lr)


class WeightSchedule:
    """
    Stateless loss weight schedule.

    Wraps a list of:
      - constants, or
      - functions (t, losses) -> scalar

    into a single callable that returns a 1D vector of weights.

    This is JAX- and jit-compatible as long as the underlying functions
    use only JAX operations.
    """

    def __init__(
        self,
        weight_fns: Union[List, Callable],
        min_weight: float = 0.0,
        max_weight: float = 1e6,
    ):
        """
        Args:
            weight_fns: list of floats or functions (t, losses) -> weight
            min_weight: minimum weight (clamp)
            max_weight: maximum weight (clamp)
        """
        self.weight_fns = weight_fns if callable(weight_fns) else lambda t, _: jnp.array(weight_fns)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def __call__(self, t, losses: jnp.ndarray) -> jnp.ndarray:
        """
        Compute weights for given step and losses.

        Args:
            t: current step/epoch (scalar int or jnp.ndarray)
            losses: 1D jnp.ndarray of per-constraint losses

        Returns:
            1D jnp.ndarray of weights, clipped elementwise to [min_weight, max_weight].
        """
        # Evaluate weight function and ensure result is an array
        weights = self.weight_fns(t, losses)
        weights = jnp.asarray(weights)
        if weights.ndim == 0:
            weights = weights[jnp.newaxis]
        weights = jnp.clip(weights, self.min_weight, self.max_weight)
        return weights
