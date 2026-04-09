"""
jno/utils/adaptive/lrscheduler.py
==================================
Learning-rate scheduling utilities for PINNs.

Two classes:
  LearningRateSchedule  — stateless, JIT-safe schedule wrapper
  DLRS                  — Dynamic Learning Rate Scheduler (Dharanalakota et al. 2025)

Factory:
  dlrs(...)  ->  DLRS instance

Import:
    from jno.utils.adaptive.lrscheduler import LearningRateSchedule, DLRS, dlrs
"""

from typing import Callable, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

LRFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


# =============================================================================
# LearningRateSchedule
# =============================================================================

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

    def __call__(self, t: int, losses: jnp.ndarray) -> jnp.ndarray:
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
            t = jnp.asarray(t, dtype=jnp.float32)
            frac = jnp.clip(t / T, 0.0, 1.0)
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
            t = jnp.asarray(t, dtype=jnp.float32)

            warm = lr0 * (t + 1.0) / jnp.maximum(1.0, float(W))

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
        decay_steps = float(decay_steps)
        lr_end = float(lr_end)

        def fn(t, losses):
            t = jnp.asarray(t, dtype=jnp.float32)
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
        v = jnp.asarray(list(values), dtype=jnp.float32)
        if len(values) != len(boundaries) + 1:
            raise ValueError("values must have length len(boundaries)+1")

        def fn(t, losses):
            t = jnp.asarray(t, dtype=jnp.int32)
            idx = jnp.sum(t >= b)
            return v[idx]

        return cls(fn, min_lr=min_lr, max_lr=max_lr)


# =============================================================================
# DLRS: Dynamic Learning Rate Scheduler
# =============================================================================

class DLRS:
    """
    Loss-based Dynamic Learning Rate Scheduler (DLRS).

    Implements the paper\'s idea: adjust the learning rate based on a normalized
    loss-slope computed from a sequence of loss values, and scale the adjustment
    by the order-of-magnitude of the current learning rate.

    Reference: Dharanalakota, Raikar & Ghosh (2025).

    Practical note for this codebase:
      - The scheduler is called once per training iteration (epoch/step).
      - We treat the scalar "batch loss" as the total loss at that step,
        and keep a rolling window of recent total losses to emulate the
        paper\'s within-epoch batch-loss array.

    Update rule (host-side, stateful):
      mean  = average(loss_window)
      slope = (loss_last - loss_first) / mean
      k     = floor(log10(lr))
      case:
        slope > 1      -> decrease more  (decremental_factor)
        0 < slope <= 1 -> decrease gently (stagnation_factor)
        slope <= 0     -> increase        (increment_factor, applied as negative)
      lr_next = clip(lr - 10^k * case, min_lr, max_lr)

    Import:
        from jno.utils.adaptive.lrscheduler import DLRS
    """

    def __init__(
        self,
        lr0: float = 1e-3,
        window: int = 10,
        decremental_factor: float = 0.5,
        stagnation_factor: float = 0.1,
        increment_factor: float = 0.1,
        min_lr: float = 1e-10,
        max_lr: float = 1.0,
        eps: float = 1e-12,
    ):
        self.lr0 = float(lr0)
        self.window = int(window)

        self.decremental_factor = float(decremental_factor)
        self.stagnation_factor = float(stagnation_factor)
        self.increment_factor = float(increment_factor)

        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.eps = float(eps)

        # Host-side mutable state
        self.lr = np.float32(self.lr0)
        self.loss_hist = []
        self.initialized = False

    def _update_state_host(self, total_loss_np):
        total_loss = float(np.asarray(total_loss_np, dtype=np.float32))
        total_loss = max(total_loss, self.eps)

        if not self.initialized:
            self.loss_hist = [total_loss]
            self.lr = np.float32(np.clip(self.lr0, self.min_lr, self.max_lr))
            self.initialized = True
            return np.asarray(self.lr, dtype=np.float32)

        # Update rolling window
        self.loss_hist.append(total_loss)
        if len(self.loss_hist) > self.window:
            self.loss_hist = self.loss_hist[-self.window:]

        # Compute normalized slope
        if len(self.loss_hist) >= 2:
            mean_loss = float(np.mean(self.loss_hist))
            mean_loss = max(mean_loss, self.eps)
            slope = (self.loss_hist[-1] - self.loss_hist[0]) / mean_loss
        else:
            slope = 0.0

        lr_safe = float(np.clip(self.lr, self.min_lr, self.max_lr))
        k = float(np.floor(np.log10(max(lr_safe, self.min_lr))))
        base = 10.0 ** k

        # Case selection 
        if slope > 1.0:
            case = +self.decremental_factor
        elif slope > 0.0:
            case = +self.stagnation_factor
        else:
            case = -self.increment_factor

        delta = base * case
        lr_new = lr_safe - delta
        lr_new = float(np.clip(lr_new, self.min_lr, self.max_lr))

        self.lr = np.float32(lr_new)
        return np.asarray(self.lr, dtype=np.float32)

    def __call__(self, t: int, losses: jnp.ndarray) -> jnp.ndarray:
        # Convert per-constraint losses to a scalar total loss for DLRS logic
        losses = jnp.asarray(losses)
        total_loss = jnp.sum(losses) if losses.ndim > 0 else losses

        result_shape = jax.ShapeDtypeStruct((), jnp.float32)
        lr = jax.pure_callback(
            self._update_state_host,
            result_shape,
            total_loss
        )
        # Safety clamp also on device
        lr = jnp.clip(lr, self.min_lr, self.max_lr)
        return lr

    def reset(self):
        """Reset all host-side state to lr0."""
        self.lr = np.float32(self.lr0)
        self.loss_hist = []
        self.initialized = False


# =============================================================================
# Factory
# =============================================================================

def dlrs(
    lr0: float = 1e-3,
    window: int = 10,
    decremental_factor: float = 0.5,
    stagnation_factor: float = 0.1,
    increment_factor: float = 0.1,
    min_lr: float = 1e-10,
    max_lr: float = 1.0,
) -> DLRS:
    """Convenience factory for DLRS."""
    return DLRS(
        lr0=lr0,
        window=window,
        decremental_factor=decremental_factor,
        stagnation_factor=stagnation_factor,
        increment_factor=increment_factor,
        min_lr=min_lr,
        max_lr=max_lr,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LearningRateSchedule",
    "DLRS",
    "dlrs",
]


