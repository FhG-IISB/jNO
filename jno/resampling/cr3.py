"""CR3: Causal Retain-Resample with time gating for time-dependent problems."""

import jax
import jax.numpy as jnp
from .base import ResamplingStrategy


class CR3(ResamplingStrategy):
    """Causal Retain-Resample (CR3) with time gating.

    Applies a causal time gate g(t; γ) to modulate residuals, encouraging
    the network to learn from earlier times first. The gate parameter γ
    evolves during training to progressively include later times.

    This is particularly effective for time-dependent PDEs where causality
    is important (e.g., wave equations, diffusion).

    Reference:
        "Causal PINN: Respecting causality in physics-informed neural networks"
        https://proceedings.mlr.press/v202/daw23a.html

    Algorithm:
        1. Compute residual rI at interior points
        2. Apply time gate: g(t) = 0.5 * (1 - tanh(α * (t_norm - γ)))
        3. Score F = rI * g(t), keep points where F > mean(F)
        4. Update γ causally: γ += η_g * min(exp(-ε * Lg), δ_max)
    """

    def __init__(
        self,
        resample_every: int = 100,
        resample_fraction: float = 0.5,
        start_epoch: int = 1000,
        t_index: int = -1,
        alpha: float = 5.0,
        gamma0: float = -0.5,
        eta_g: float = 1e-3,
        epsilon: float = 20.0,
        delta_max: float = 0.1,
        min_keep_frac: float = 0.1,
        max_keep_frac: float = 0.9,
    ):
        """Initialize CR3 resampling.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Target fraction to keep
            start_epoch: Start resampling after this many epochs
            t_index: Index of time column (default -1 for last column)
            alpha: Gate steepness (default 5.0)
            gamma0: Initial gate position (default -0.5)
            eta_g: Gate learning rate (default 1e-3)
            epsilon: Gate update damping (default 20.0)
            delta_max: Maximum gate step (default 0.1)
            min_keep_frac: Minimum fraction to keep (default 0.1)
            max_keep_frac: Maximum fraction to keep (default 0.9)
        """
        super().__init__(resample_every, resample_fraction, start_epoch)
        self.t_index = t_index
        self.alpha = alpha
        self.gamma = gamma0
        self.eta_g = eta_g
        self.epsilon = epsilon
        self.delta_max = delta_max
        self.min_keep_frac = min_keep_frac
        self.max_keep_frac = max_keep_frac
        self.gamma_history: list[float] = []

    def _compute_gate(self, t_values: jnp.ndarray) -> jnp.ndarray:
        """Compute time gate: g(t) = 0.5 * (1 - tanh(α * (t_norm - γ)))

        Returns values in [0, 1] where earlier times have higher weight.
        """
        t = jnp.asarray(t_values).flatten()

        # Normalize time to [0, 1]
        t_min, t_max = jnp.min(t), jnp.max(t)
        t_range = jnp.maximum(t_max - t_min, 1e-12)
        t_norm = (t - t_min) / t_range

        # Apply gate
        g = 0.5 * (1.0 - jnp.tanh(self.alpha * (t_norm - self.gamma)))
        return jnp.clip(g, 0.0, 1.0)

    def _update_gamma(self, residuals: jnp.ndarray, gate_values: jnp.ndarray):
        """Update gamma based on gated loss."""
        if len(residuals) == 0:
            return

        # Gated loss
        Lg = float(jnp.mean(residuals**2 * gate_values))

        # Update step
        step = self.eta_g * jnp.minimum(jnp.exp(-self.epsilon * Lg), self.delta_max)
        self.gamma = float(self.gamma + step)

        # Clip to reasonable bounds
        self.gamma = float(jnp.clip(self.gamma, -1.0, 2.0))
        self.gamma_history.append(self.gamma)

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Resample using causal time gating.

        Args:
            points: Current points (N, D)
            residuals: Residual magnitudes (N,) or (B, N)
            domain: domain object
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key

        Returns:
            New points (N, D)
        """
        # Handle batched residuals
        if residuals.ndim > 1:
            residuals = jnp.mean(jnp.abs(residuals), axis=0)
        else:
            residuals = jnp.abs(residuals)

        n_points = points.shape[0]

        if residuals.shape[0] != n_points:
            return points

        # Compute time gate
        if points.shape[1] > abs(self.t_index):
            t_values = points[:, self.t_index]
            gate_values = self._compute_gate(t_values)
        else:
            # No time dimension - use uniform gate
            gate_values = jnp.ones(n_points)

        # Compute gated score
        F = residuals * gate_values

        # Threshold selection
        threshold = jnp.mean(F)
        keep_mask = F > threshold
        n_keep = int(jnp.sum(keep_mask))

        # Apply min/max constraints
        n_min = max(1, int(self.min_keep_frac * n_points))
        n_max = min(n_points - 1, int(self.max_keep_frac * n_points))

        if n_keep < n_min:
            # Keep top n_min by gated score
            top_indices = jnp.argsort(F)[-n_min:]
            keep_mask = jnp.zeros(n_points, dtype=bool)
            keep_mask = keep_mask.at[top_indices].set(True)
            n_keep = n_min
        elif n_keep > n_max:
            # Keep only top n_max
            keep_indices = jnp.where(keep_mask)[0]
            top_indices = keep_indices[jnp.argsort(F[keep_indices])[-n_max:]]
            keep_mask = jnp.zeros(n_points, dtype=bool)
            keep_mask = keep_mask.at[top_indices].set(True)
            n_keep = n_max

        # Retain selected points
        retained_points = points[keep_mask]
        n_new = n_points - len(retained_points)

        # Fill with random samples
        if n_new > 0 and hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
            candidates = jnp.array(domain._mesh_points[tag])
            new_indices = jax.random.choice(rng_key, candidates.shape[0], shape=(n_new,), replace=True)
            new_points = candidates[new_indices]
            result = jnp.concatenate([retained_points, new_points], axis=0)
        else:
            result = retained_points

        # Update gamma for next iteration
        self._update_gamma(residuals, gate_values)

        # Ensure correct size
        if len(result) < n_points:
            extra_indices = jax.random.choice(rng_key, len(result), shape=(n_points - len(result),), replace=True)
            extra_points = result[extra_indices]
            result = jnp.concatenate([result, extra_points], axis=0)

        return result[:n_points]
