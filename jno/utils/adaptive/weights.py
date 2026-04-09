"""
jno/utils/adaptive/weights.py
==============================
Loss-weight scheduling utilities for PINNs.

Classes:
  WeightSchedule         — stateless, JIT-safe weight wrapper
  ReLoBRaLo              — Relative Loss Balancing Residual Algorithm
                           (Bischof & Kraus, 2021/2025)
  LbPINNsLossBalancing   — Self-adaptive loss balancing via learnable
                           log-variances (Xiang et al., 2022)
  SoftAdapt              — Softmax over loss ratios
                           (Heydari, Narang & Zweig, 2019)
  DWA                    — Dynamic Weight Average
                           (Liu, Johns & Davison, CVPR 2019)
  RLW                    — Random Loss Weighting via Dirichlet sampling
                           (Lin et al., 2021)

Factories:
  relobralo(...)              ->  ReLoBRaLo instance
  lbpinns_loss_balancing(...) ->  LbPINNsLossBalancing instance
  softadapt(...)              ->  SoftAdapt instance
  dwa(...)                    ->  DWA instance
  rlw(...)                    ->  RLW instance
"""

from typing import Callable, List, Optional, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np

WeightFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _host_callback_nondiff(host_fn, result_shape, losses: jnp.ndarray) -> jnp.ndarray:
    """Run a host callback with a zero-JVP custom rule.

    ``jax.pure_callback`` has no built-in JVP rule. Adaptive weight callbacks
    are meant to act as non-differentiable control signals, so we explicitly
    set their tangent contribution to zero.
    """

    @jax.custom_jvp
    def _call(x):
        return jax.pure_callback(
            host_fn,
            result_shape,
            x,
            vmap_method="sequential",
        )

    @_call.defjvp
    def _call_jvp(primals, tangents):
        (x,), (_x_dot,) = primals, tangents
        y = _call(x)
        return y, jnp.zeros_like(y)

    return _call(losses)


def _normalize_losses(losses: tuple) -> jnp.ndarray:
    """Convert varargs / list / array to a 1-D losses vector."""
    if len(losses) == 1:
        arg = losses[0]
        if isinstance(arg, (list, tuple)):
            return jnp.stack(arg)
        return jnp.asarray(arg)
    return jnp.stack(losses)


# =============================================================================
# WeightSchedule
# =============================================================================


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
        self.weight_fns = weight_fns if not isinstance(weight_fns, List) else lambda t, _: weight_fns
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def __call__(self, t, losses: jnp.ndarray) -> jnp.ndarray:
        weights = self.weight_fns(t, losses)
        weights = jnp.stack(weights, axis=0)
        weights = jnp.clip(weights, self.min_weight, self.max_weight)
        return weights


# =============================================================================
# ReLoBRaLo: Relative Loss Balancing Residual Algorithm
# =============================================================================


class ReLoBRaLo:
    """
    Adaptive loss balancing via relative residual scaling.

    Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for
    Physics-Informed Deep Learning" (2021/2025).

    At each training step:
      1. Compute softmax over loss ratios (current/previous) -> local weights.
      2. Bernoulli coin flip selects global (vs initial losses) or history.
      3. EMA blend of local and historical weights -> final lambda vector.

    Callable interface: (losses) -> 1-D weight vector [jnp.float32]
    State is held host-side; bridged into JAX via jax.pure_callback.

    Import:
        from jno.utils.adaptive.weights import ReLoBRaLo
    """

    def __init__(
        self,
        num_losses: int = None,
        alpha: float = 0.99,
        tau: float = 0.1,
        expected_rho: float = 0.999,
        seed: int = 42,
    ):
        self.num_losses = num_losses
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.expected_rho = float(expected_rho)
        self.seed = seed

        self.rng = np.random.default_rng(seed)

        if num_losses is not None:
            self.lambdas = np.ones(num_losses, dtype=np.float32)
        else:
            self.lambdas = None

        self.initial_losses = None
        self.prev_losses = None
        self.initialized = False

    def _stable_softmax_weights(
        self,
        current_losses: np.ndarray,
        reference_losses: np.ndarray,
    ) -> np.ndarray:
        """Pure-numpy softmax: runs inside pure_callback, no JAX needed."""
        reference_losses = np.maximum(reference_losses, 1e-12)
        current_losses = np.maximum(current_losses, 1e-12)

        logits = current_losses / (self.tau * reference_losses)
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        weights_normalized = exp_logits / np.sum(exp_logits)

        return self.num_losses * weights_normalized

    def _update_state_host(self, losses_np):
        losses_np = np.asarray(losses_np, dtype=np.float32)

        if self.num_losses is None:
            self.num_losses = len(losses_np)
            self.lambdas = np.ones(self.num_losses, dtype=np.float32)

        if not self.initialized:
            self.initial_losses = losses_np.copy()
            self.prev_losses = losses_np.copy()
            self.initialized = True
            return self.lambdas

        losses_np = np.maximum(losses_np, 1e-12)

        # Pure numpy — no JAX inside the host callback
        lambda_bal_local = self._stable_softmax_weights(losses_np, self.prev_losses)

        # Bernoulli coin flip via numpy RNG (no jax.random here)
        rho = float(self.rng.random() < self.expected_rho)

        lambda_bal_global = self._stable_softmax_weights(losses_np, self.initial_losses)

        lambda_hist = rho * self.lambdas + (1.0 - rho) * lambda_bal_global

        new_lambdas = self.alpha * lambda_hist + (1.0 - self.alpha) * lambda_bal_local

        self.lambdas = np.asarray(new_lambdas, dtype=np.float32)
        self.prev_losses = losses_np.copy()

        return self.lambdas

    def __call__(self, *losses: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Return one weight per loss term.

        Accepts individual scalars, a list, or a 1-D array::

            w0, w1 = balancer(loss0, loss1)
            w0, w1 = balancer([loss0, loss1])
            w0, w1 = balancer(jnp.array([loss0, loss1]))
        """
        losses_arr = _normalize_losses(losses)
        if self.num_losses is None:
            self.num_losses = int(losses_arr.shape[0])
            self.lambdas = np.ones(self.num_losses, dtype=np.float32)

        result_shape = jax.ShapeDtypeStruct((self.num_losses,), jnp.float32)
        weights = _host_callback_nondiff(
            self._update_state_host,
            result_shape,
            losses_arr,
        )
        return tuple(weights[i] for i in range(self.num_losses))

    def reset(self):
        """Reset all host-side state to initial values."""
        if self.num_losses is not None:
            self.lambdas = np.ones(self.num_losses, dtype=np.float32)
        else:
            self.lambdas = None
        self.initial_losses = None
        self.prev_losses = None
        self.initialized = False
        self.rng = np.random.default_rng(self.seed)


def relobralo(
    alpha: float = 0.99,
    tau: float = 0.1,
    expected_rho: float = 0.999,
    seed: int = 42,
) -> ReLoBRaLo:
    """Adaptive loss balancing via relative residual scaling.

    Reference: Bischof & Kraus, "Multi-Objective Loss Balancing for
    Physics-Informed Deep Learning" (2021/2025).

    At each training step:
      1. Compute softmax over loss ratios (current/previous) -> local weights.
      2. Bernoulli coin flip selects global (vs initial losses) or history.
      3. EMA blend of local and historical weights -> final lambda vector.

    Args:
        alpha: EMA smoothing factor (default 0.99).
        tau: Temperature for softmax over loss ratios (default 0.1).
        expected_rho: Probability of using historical vs global weights (default 0.999).
        seed: Random seed for the Bernoulli coin flip.

    Returns:
        A ``ReLoBRaLo`` instance.  Callable as ``balancer(losses) -> weights``.
    """
    return ReLoBRaLo(
        num_losses=None,
        alpha=alpha,
        tau=tau,
        expected_rho=expected_rho,
        seed=seed,
    )


# =============================================================================
# LbPINNsLossBalancing: Self-adaptive loss balancing
# =============================================================================


class LbPINNsLossBalancing:
    """
    Self-adaptive loss balancing via learnable log-variances.

    Reference: Xiang, Peng, Liu & Yao, "Self-adaptive loss balanced
    Physics-informed neural networks" (2022).

    Implements Eq. (11): learnable log-variances s_j = log(epsilon_j^2).

    Joint objective:
        L_total(s) = sum_j 0.5 * exp(-s_j) * L_j  +  sum_j s_j

    Weights returned:
        w_j = 0.5 * exp(-s_j)   =>   weighted_sum = sum_j w_j * L_j

    Gradient of objective wrt s_j (analytic):
        g_j = 1 - w_j * L_j

    s is updated host-side via a dedicated Adam optimizer.

    Callable interface: (losses) -> 1-D weight vector [jnp.float32]
    State is held host-side; bridged into JAX via jax.pure_callback.

    Import:
        from jno.utils.adaptive.weights import LbPINNsLossBalancing
    """

    def __init__(
        self,
        num_losses: int = None,
        init_s: Union[float, Sequence[float]] = 0.0,
        lr_s: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        s_min: float = -20.0,
        s_max: float = 20.0,
        loss_floor: float = 1e-12,
    ):
        self.num_losses = num_losses

        self.init_s = init_s
        self.lr_s = float(lr_s)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps_adam = float(eps_adam)

        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.loss_floor = float(loss_floor)

        # Host-side mutable state
        self.s: Optional[np.ndarray] = None
        self.m: Optional[np.ndarray] = None  # Adam 1st moment
        self.v: Optional[np.ndarray] = None  # Adam 2nd moment
        self.t_adam = 0
        self.initialized = False

    def _init_params(self, L: np.ndarray):
        if self.num_losses is None:
            self.num_losses = int(L.shape[0])

        if isinstance(self.init_s, (int, float)):
            s0 = np.full((self.num_losses,), float(self.init_s), dtype=np.float32)
        else:
            s0 = np.asarray(self.init_s, dtype=np.float32)
            if s0.shape != (self.num_losses,):
                raise ValueError(f"init_s must have shape {(self.num_losses,)}")

        self.s = np.clip(s0, self.s_min, self.s_max).astype(np.float32)
        self.m = np.zeros_like(self.s, dtype=np.float32)
        self.v = np.zeros_like(self.s, dtype=np.float32)
        self.t_adam = 0
        self.initialized = True

    def _update_state_host(self, losses_np):
        L = np.asarray(losses_np, dtype=np.float32).reshape(-1)
        L = np.maximum(L, self.loss_floor)

        if not self.initialized:
            self._init_params(L)

        assert self.s is not None and self.m is not None and self.v is not None

        # Gradient wrt s_j:
        #   d/ds_j [0.5 * exp(-s_j) * L_j + s_j]
        # = -0.5 * exp(-s_j) * L_j + 1
        # = 1 - w_j * L_j,  where w_j = 0.5 * exp(-s_j)
        w = 0.5 * np.exp(-self.s).astype(np.float32)
        g = (1.0 - w * L).astype(np.float32)

        # Adam step on s
        self.t_adam += 1
        b1, b2 = self.beta1, self.beta2
        self.m = (b1 * self.m + (1.0 - b1) * g).astype(np.float32)
        self.v = (b2 * self.v + (1.0 - b2) * (g * g)).astype(np.float32)

        mhat = self.m / (1.0 - (b1**self.t_adam))
        vhat = self.v / (1.0 - (b2**self.t_adam))

        self.s = (self.s - self.lr_s * mhat / (np.sqrt(vhat) + self.eps_adam)).astype(np.float32)
        self.s = np.clip(self.s, self.s_min, self.s_max).astype(np.float32)

        # Return updated weights
        w_new = (0.5 * np.exp(-self.s)).astype(np.float32)
        return w_new

    def __call__(self, *losses: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Return one weight per loss term.

        Accepts individual scalars, a list, or a 1-D array::

            w0, w1 = balancer(loss0, loss1)
            w0, w1 = balancer([loss0, loss1])
            w0, w1 = balancer(jnp.array([loss0, loss1]))
        """
        losses_arr = _normalize_losses(losses)
        if self.num_losses is None:
            self.num_losses = int(losses_arr.shape[0])

        result_shape = jax.ShapeDtypeStruct((self.num_losses,), jnp.float32)
        weights = _host_callback_nondiff(
            self._update_state_host,
            result_shape,
            losses_arr,
        )
        return tuple(weights[i] for i in range(self.num_losses))

    def get_s(self) -> np.ndarray:
        """Host-side accessor (useful for logging)."""
        if self.s is None:
            return None
        return np.asarray(self.s)

    def reset(self):
        """Reset all host-side state to initial values."""
        self.s = None
        self.m = None
        self.v = None
        self.t_adam = 0
        self.initialized = False


def lbpinns_loss_balancing(
    init_s: Union[float, Sequence[float]] = 0.0,
    lr_s: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps_adam: float = 1e-8,
    s_min: float = -20.0,
    s_max: float = 20.0,
) -> LbPINNsLossBalancing:
    """Self-adaptive loss balancing via learnable log-variances.

    Reference: Xiang, Peng, Liu & Yao, "Self-adaptive loss balanced
    Physics-informed neural networks" (2022).

    Learnable log-variances ``s_j = log(epsilon_j^2)`` yield weights
    ``w_j = 0.5 * exp(-s_j)``, updated host-side via Adam.

    Args:
        init_s: Initial value(s) for the log-variance parameters.
        lr_s: Learning rate for the internal Adam optimizer on ``s``.
        beta1: Adam first-moment decay.
        beta2: Adam second-moment decay.
        eps_adam: Adam epsilon for numerical stability.
        s_min: Lower clamp for ``s`` values.
        s_max: Upper clamp for ``s`` values.

    Returns:
        A ``LbPINNsLossBalancing`` instance.  Callable as ``balancer(losses) -> weights``.
    """
    return LbPINNsLossBalancing(
        num_losses=None,
        init_s=init_s,
        lr_s=lr_s,
        beta1=beta1,
        beta2=beta2,
        eps_adam=eps_adam,
        s_min=s_min,
        s_max=s_max,
    )


# =============================================================================
# SoftAdapt (Heydari, Narang & Zweig, 2019)
# =============================================================================


class SoftAdapt:
    """
    Adaptive loss balancing via softmax over loss ratios.

    Reference: Heydari, Narang & Zweig, "SoftAdapt: Techniques for
    Adaptive Loss Weighting of Neural Networks with Multi-Part Loss
    Functions" (2019).

    At each training step:
      1. Compute loss ratios  s_k = L_k(t) / L_k(t-1)
      2. Weights = N * softmax(s / beta)

    Losses that are *not* decreasing fast enough get more weight.

    Callable interface: (losses) -> 1-D weight vector [jnp.float32]
    State is held host-side; bridged into JAX via jax.pure_callback.

    Import:
        from jno.utils.adaptive.weights import SoftAdapt
    """

    def __init__(
        self,
        num_losses: int = None,
        beta: float = 0.1,
        loss_floor: float = 1e-12,
    ):
        self.num_losses = num_losses
        self.beta = float(beta)
        self.loss_floor = float(loss_floor)

        self.prev_losses = None
        self.initialized = False

    def _update_state_host(self, losses_np):
        L = np.asarray(losses_np, dtype=np.float32)
        L = np.maximum(L, self.loss_floor)

        if self.num_losses is None:
            self.num_losses = len(L)

        if not self.initialized:
            self.prev_losses = L.copy()
            self.initialized = True
            return np.ones(self.num_losses, dtype=np.float32)

        # Loss ratios (slopes)
        ratios = L / np.maximum(self.prev_losses, self.loss_floor)

        # Stable softmax with temperature beta
        logits = ratios / self.beta
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = self.num_losses * exp_logits / np.sum(exp_logits)

        self.prev_losses = L.copy()
        return weights.astype(np.float32)

    def __call__(self, *losses: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Return one weight per loss term.

        Accepts individual scalars, a list, or a 1-D array::

            w0, w1 = balancer(loss0, loss1)
            w0, w1 = balancer([loss0, loss1])
            w0, w1 = balancer(jnp.array([loss0, loss1]))
        """
        losses_arr = _normalize_losses(losses)
        if self.num_losses is None:
            self.num_losses = int(losses_arr.shape[0])

        result_shape = jax.ShapeDtypeStruct((self.num_losses,), jnp.float32)
        weights = _host_callback_nondiff(
            self._update_state_host,
            result_shape,
            losses_arr,
        )
        return tuple(weights[i] for i in range(self.num_losses))

    def reset(self):
        """Reset all host-side state."""
        self.prev_losses = None
        self.initialized = False


def softadapt(
    beta: float = 0.1,
    loss_floor: float = 1e-12,
) -> SoftAdapt:
    """Adaptive loss balancing via softmax over loss ratios.

    Reference: Heydari, Narang & Zweig, "SoftAdapt: Techniques for
    Adaptive Loss Weighting of Neural Networks with Multi-Part Loss
    Functions" (2019).

    Weights are ``N * softmax(L(t)/L(t-1) / beta)``.
    Losses not decreasing fast enough get more weight.

    Args:
        beta: Temperature for the softmax (default 0.1).
        loss_floor: Minimum loss value to avoid division by zero.

    Returns:
        A ``SoftAdapt`` instance.  Callable as ``balancer(losses) -> weights``.
    """
    return SoftAdapt(num_losses=None, beta=beta, loss_floor=loss_floor)


# =============================================================================
# DWA: Dynamic Weight Average (Liu, Johns & Davison, CVPR 2019)
# =============================================================================


class DWA:
    """
    Dynamic Weight Average for multi-task / multi-loss training.

    Reference: Liu, Johns & Davison, "End-to-End Multi-Task Learning
    with Attention" (CVPR 2019).

    At each training step (once two history steps exist):
      1. Compute rate of descent  r_k = L_k(t-1) / L_k(t-2)
      2. Weights = N * softmax(r / T)

    Tasks whose loss is *increasing* (ratio > 1) are up-weighted.

    Callable interface: (losses) -> 1-D weight vector [jnp.float32]
    State is held host-side; bridged into JAX via jax.pure_callback.

    Import:
        from jno.utils.adaptive.weights import DWA
    """

    def __init__(
        self,
        num_losses: int = None,
        temperature: float = 2.0,
        loss_floor: float = 1e-12,
    ):
        self.num_losses = num_losses
        self.temperature = float(temperature)
        self.loss_floor = float(loss_floor)

        self.losses_t1 = None  # L(t-1)
        self.losses_t2 = None  # L(t-2)
        self.step = 0

    def _update_state_host(self, losses_np):
        L = np.asarray(losses_np, dtype=np.float32)
        L = np.maximum(L, self.loss_floor)

        if self.num_losses is None:
            self.num_losses = len(L)

        self.step += 1

        if self.step <= 2:
            # Need two history steps before computing weights
            self.losses_t2 = self.losses_t1.copy() if self.losses_t1 is not None else L.copy()
            self.losses_t1 = L.copy()
            return np.ones(self.num_losses, dtype=np.float32)

        # r_k = L_k(t-1) / L_k(t-2)
        ratios = self.losses_t1 / np.maximum(self.losses_t2, self.loss_floor)

        # Stable softmax with temperature
        logits = ratios / self.temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights = self.num_losses * exp_logits / np.sum(exp_logits)

        # Shift history
        self.losses_t2 = self.losses_t1.copy()
        self.losses_t1 = L.copy()

        return weights.astype(np.float32)

    def __call__(self, *losses: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Return one weight per loss term.

        Accepts individual scalars, a list, or a 1-D array::

            w0, w1 = balancer(loss0, loss1)
            w0, w1 = balancer([loss0, loss1])
            w0, w1 = balancer(jnp.array([loss0, loss1]))
        """
        losses_arr = _normalize_losses(losses)
        if self.num_losses is None:
            self.num_losses = int(losses_arr.shape[0])

        result_shape = jax.ShapeDtypeStruct((self.num_losses,), jnp.float32)
        weights = _host_callback_nondiff(
            self._update_state_host,
            result_shape,
            losses_arr,
        )
        return tuple(weights[i] for i in range(self.num_losses))

    def reset(self):
        """Reset all host-side state."""
        self.losses_t1 = None
        self.losses_t2 = None
        self.step = 0


def dwa(
    temperature: float = 2.0,
    loss_floor: float = 1e-12,
) -> DWA:
    """Dynamic Weight Average for multi-task / multi-loss training.

    Reference: Liu, Johns & Davison, "End-to-End Multi-Task Learning
    with Attention" (CVPR 2019).

    Weights are ``N * softmax(r / T)`` where ``r_k = L_k(t-1) / L_k(t-2)``.
    Tasks whose loss is increasing (ratio > 1) are up-weighted.

    Args:
        temperature: Softmax temperature (default 2.0).
        loss_floor: Minimum loss value to avoid division by zero.

    Returns:
        A ``DWA`` instance.  Callable as ``balancer(losses) -> weights``.
    """
    return DWA(num_losses=None, temperature=temperature, loss_floor=loss_floor)


# =============================================================================
# RLW: Random Loss Weighting (Lin et al., 2021)
# =============================================================================


class RLW:
    """
    Random Loss Weighting via symmetric Dirichlet sampling.

    Reference: Lin, Ye, Xu, Shi & Zhang, "Reasonable Effectiveness
    of Random Weighting: A Litmus Test for Multi-Task Learning"
    (TMLR 2021).

    At each training step:
      Sample w ~ Dirichlet(alpha, …, alpha), scale by N.

    Despite being purely stochastic, this is remarkably competitive
    with sophisticated adaptive methods, serving as a strong baseline.

    Callable interface: (losses) -> 1-D weight vector [jnp.float32]
    State is held host-side; bridged into JAX via jax.pure_callback.

    Import:
        from jno.utils.adaptive.weights import RLW
    """

    def __init__(
        self,
        num_losses: int = None,
        alpha: float = 1.0,
        seed: int = 42,
    ):
        self.num_losses = num_losses
        self.alpha = float(alpha)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _update_state_host(self, losses_np):
        L = np.asarray(losses_np, dtype=np.float32)
        if self.num_losses is None:
            self.num_losses = len(L)

        weights = self.rng.dirichlet(np.full(self.num_losses, self.alpha))
        return (self.num_losses * weights).astype(np.float32)

    def __call__(self, *losses: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Return one weight per loss term.

        Accepts individual scalars, a list, or a 1-D array::

            w0, w1 = balancer(loss0, loss1)
            w0, w1 = balancer([loss0, loss1])
            w0, w1 = balancer(jnp.array([loss0, loss1]))
        """
        losses_arr = _normalize_losses(losses)
        if self.num_losses is None:
            self.num_losses = int(losses_arr.shape[0])

        result_shape = jax.ShapeDtypeStruct((self.num_losses,), jnp.float32)
        weights = _host_callback_nondiff(
            self._update_state_host,
            result_shape,
            losses_arr,
        )
        return tuple(weights[i] for i in range(self.num_losses))

    def reset(self):
        """Reset RNG to initial seed."""
        self.rng = np.random.default_rng(self.seed)


def rlw(
    alpha: float = 1.0,
    seed: int = 42,
) -> RLW:
    """Random Loss Weighting via symmetric Dirichlet sampling.

    Reference: Lin, Ye, Xu, Shi & Zhang, "Reasonable Effectiveness
    of Random Weighting: A Litmus Test for Multi-Task Learning"
    (TMLR 2021).

    At each step, sample ``w ~ Dirichlet(alpha, ..., alpha)`` scaled by N.
    Surprisingly competitive with sophisticated adaptive methods.

    Args:
        alpha: Dirichlet concentration parameter (default 1.0 = uniform).
        seed: Random seed.

    Returns:
        An ``RLW`` instance.  Callable as ``balancer(losses) -> weights``.
    """
    return RLW(num_losses=None, alpha=alpha, seed=seed)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "WeightSchedule",
    "ReLoBRaLo",
    "relobralo",
    "LbPINNsLossBalancing",
    "lbpinns_loss_balancing",
    "SoftAdapt",
    "softadapt",
    "DWA",
    "dwa",
    "RLW",
    "rlw",
]
