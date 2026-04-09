"""R3: Residual-based Refinement and Resampling strategy."""

import jax
import jax.numpy as jnp
from .base import ResamplingStrategy


class R3(ResamplingStrategy):
    """Residual-based Refinement and Resampling (R3).

    Keeps points with high residuals and replaces low-residual points
    with new samples from the candidate pool. This focuses collocation
    on regions where the PDE is poorly satisfied.

    Reference:
        "Residual-based adaptive sampling for physics-informed neural networks"
    """

    def __init__(
        self,
        resample_every: int = 100,
        resample_fraction: float = 0.7,
        start_epoch: int = 1000,
        threshold_mode: str = "mean",
        min_keep_frac: float = 0.3,
        max_keep_frac: float = 0.9,
    ):
        """Initialize R3 resampling.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Not used for R3 (uses threshold instead)
            start_epoch: Start resampling after this many epochs
            threshold_mode: Threshold for keeping points ("mean", "median", or float)
            min_keep_frac: Minimum fraction of points to keep (default 0.3)
            max_keep_frac: Maximum fraction of points to keep (default 0.9)
        """
        super().__init__(resample_every, resample_fraction, start_epoch)
        self.threshold_mode = threshold_mode
        self.min_keep_frac = min_keep_frac
        self.max_keep_frac = max_keep_frac

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Keep high-residual points, resample low-residual points.

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

        # Determine threshold
        if self.threshold_mode == "mean":
            threshold = jnp.mean(residuals)
        elif self.threshold_mode == "median":
            threshold = jnp.median(residuals)
        elif isinstance(self.threshold_mode, (int, float)):
            threshold = float(self.threshold_mode)
        else:
            threshold = jnp.mean(residuals)

        # Find points to keep (above threshold)
        keep_mask = residuals >= threshold
        n_keep = int(jnp.sum(keep_mask))

        # Apply min/max constraints
        n_min = int(self.min_keep_frac * n_points)
        n_max = int(self.max_keep_frac * n_points)

        if n_keep < n_min:
            # Keep more points based on sorted residuals
            top_indices = jnp.argsort(residuals)[-n_min:]
            keep_mask = jnp.zeros(n_points, dtype=bool)
            keep_mask = keep_mask.at[top_indices].set(True)
            n_keep = n_min
        elif n_keep > n_max:
            # Keep fewer points based on sorted residuals
            top_indices = jnp.argsort(residuals)[-n_max:]
            keep_mask = jnp.zeros(n_points, dtype=bool)
            keep_mask = keep_mask.at[top_indices].set(True)
            n_keep = n_max

        kept_points = points[keep_mask]
        n_new = n_points - len(kept_points)

        # Sample new points from candidates
        if n_new > 0 and hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
            candidates = jnp.array(domain._mesh_points[tag])
            new_indices = jax.random.choice(rng_key, candidates.shape[0], shape=(n_new,), replace=True)
            new_points = candidates[new_indices]
            result = jnp.concatenate([kept_points, new_points], axis=0)
        else:
            result = kept_points

        # Ensure correct size
        if len(result) < n_points:
            extra_indices = jax.random.choice(rng_key, len(result), shape=(n_points - len(result),), replace=True)
            extra_points = result[extra_indices]
            result = jnp.concatenate([result, extra_points], axis=0)
        elif len(result) > n_points:
            result = result[:n_points]

        return result
