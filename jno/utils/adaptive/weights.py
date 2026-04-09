"""
jno/utils/adaptive/weights.py
==============================
Loss-weight scheduling utilities for PINNs.

Three classes:
  WeightSchedule         — stateless, JIT-safe weight wrapper
  ReLoBRaLo              — Relative Loss Balancing Residual Algorithm
                           (Bischof & Kraus, 2021/2025)
  LbPINNsLossBalancing   — Self-adaptive loss balancing via learnable
                           log-variances (Xiang et al., 2022)

Factories:
  relobralo(...)              ->  ReLoBRaLo instance
  lbpinns_loss_balancing(...) ->  LbPINNsLossBalancing instance

Import:
    from jno.utils.adaptive.weights import (
        WeightSchedule,
        ReLoBRaLo, relobralo,
        LbPINNsLossBalancing, lbpinns_loss_balancing,
    )
"""

from typing import Callable, List, Union, Sequence

import jax
import jax.numpy as jnp
import numpy as np

WeightFunction = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


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

    Callable interface: (t, losses) -> 1-D weight vector [jnp.float32]
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
        self.alpha = jnp.float32(alpha)
        self.tau = jnp.float32(tau)
        self.expected_rho = jnp.float32(expected_rho)
        self.seed = seed

        self.rng = np.array(jax.random.PRNGKey(seed))

        if num_losses is not None:
            self.lambdas = np.ones(num_losses, dtype=np.float32)
        else:
            self.lambdas = None

        self.initial_losses = None
        self.prev_losses = None
        self.initialized = False

    def _stable_softmax_weights(
        self,
        current_losses: jnp.ndarray,
        reference_losses: jnp.ndarray,
    ) -> jnp.ndarray:
        reference_losses = jnp.maximum(reference_losses, 1e-12)
        current_losses = jnp.maximum(current_losses, 1e-12)

        logits = current_losses / (self.tau * reference_losses)
        logits_shifted = logits - jnp.max(logits)
        exp_logits = jnp.exp(logits_shifted)
        weights_normalized = exp_logits / jnp.sum(exp_logits)

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

        losses = jnp.array(losses_np)
        prev_losses = jnp.array(self.prev_losses)
        initial_losses = jnp.array(self.initial_losses)

        lambda_bal_local = self._stable_softmax_weights(losses, prev_losses)

        self.rng, rng_step = jax.random.split(
            jax.random.PRNGKey(int(self.rng[0]))
        )
        self.rng = np.array(self.rng)

        rho = float(jax.random.bernoulli(rng_step, p=self.expected_rho))

        lambda_bal_global = self._stable_softmax_weights(losses, initial_losses)

        lambda_hist = (rho * self.lambdas +
                       (1.0 - rho) * np.array(lambda_bal_global))

        new_lambdas = (float(self.alpha) * lambda_hist +
                       (1.0 - float(self.alpha)) * np.array(lambda_bal_local))

        self.lambdas = np.asarray(new_lambdas, dtype=np.float32)
        self.prev_losses = losses_np.copy()

        return self.lambdas

    def __call__(self, t: int, losses: jnp.ndarray) -> jnp.ndarray:
        if self.num_losses is None:
            self.num_losses = len(losses)
            self.lambdas = np.ones(self.num_losses, dtype=np.float32)

        result_shape = jax.ShapeDtypeStruct((self.num_losses,), jnp.float32)

        weights = jax.pure_callback(
            self._update_state_host,
            result_shape,
            losses
        )
        return weights

    def reset(self):
        """Reset all host-side state to initial values."""
        if self.num_losses is not None:
            self.lambdas = np.ones(self.num_losses, dtype=np.float32)
        else:
            self.lambdas = None
        self.initial_losses = None
        self.prev_losses = None
        self.initialized = False
        self.rng = np.array(jax.random.PRNGKey(self.seed))


def relobralo(
    alpha: float = 0.99,
    tau: float = 0.1,
    expected_rho: float = 0.999,
    seed: int = 42,
) -> ReLoBRaLo:
    """Convenience factory for ReLoBRaLo."""
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

    Callable interface: (t, losses) -> 1-D weight vector [jnp.float32]
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
        self.s = None          
        self.m = None          # Adam 1st moment
        self.v = None          # Adam 2nd moment
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

        mhat = self.m / (1.0 - (b1 ** self.t_adam))
        vhat = self.v / (1.0 - (b2 ** self.t_adam))

        self.s = (self.s - self.lr_s * mhat / (np.sqrt(vhat) + self.eps_adam)).astype(np.float32)
        self.s = np.clip(self.s, self.s_min, self.s_max).astype(np.float32)

        # Return updated weights
        w_new = (0.5 * np.exp(-self.s)).astype(np.float32)
        return w_new

    def __call__(self, t: int, losses: jnp.ndarray) -> jnp.ndarray:
        if self.num_losses is None:
            self.num_losses = int(losses.shape[0])

        result_shape = jax.ShapeDtypeStruct((self.num_losses,), jnp.float32)
        weights = jax.pure_callback(
            self._update_state_host,
            result_shape,
            losses
        )
        return weights

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
    """Convenience factory for LbPINNs-style self-adaptive loss balancing."""
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
# Exports
# =============================================================================

__all__ = [
    "WeightSchedule",
    "ReLoBRaLo",
    "relobralo",
    "LbPINNsLossBalancing",
    "lbpinns_loss_balancing",
]


