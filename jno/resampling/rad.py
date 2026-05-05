"""RAD: Residual-based Adaptive Distribution resampling."""

import jax
import jax.numpy as jnp
from .base import ResamplingStrategy


class RAD(ResamplingStrategy):
    """Residual-based Adaptive Distribution (RAD) resampling.

    Resamples points based on residual magnitude - focuses on regions
    with high PDE residuals (high errors).

    Reference: Lu et al. "DeepXDE: A Deep Learning Library for Solving
    Differential Equations" (2021)
    """

    def __init__(
        self,
        resample_every: int = 100,
        resample_fraction: float = 0.1,
        start_epoch: int = 1000,
        k: int = 10,
    ):
        """Initialize RAD resampling.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to resample
            start_epoch: Start resampling after this many epochs
            k: Number of candidate points to sample per replacement
        """
        super().__init__(resample_every, resample_fraction, start_epoch)
        self.k = k

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Resample based on residual magnitude.

        Removes points with lowest residuals, adds new points near
        high-residual regions.

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
        # Handle batched residuals (B, N) -> (N,) by averaging
        if residuals.ndim > 1:
            residuals = jnp.mean(jnp.abs(residuals), axis=0)
        else:
            residuals = jnp.abs(residuals)

        n_points = points.shape[0]
        n_resample = int(n_points * self.resample_fraction)

        if n_resample == 0 or residuals.shape[0] != n_points:
            return points

        # Sort by residual (ascending)
        sorted_indices = jnp.argsort(residuals)

        # Remove lowest residual points
        keep_indices = sorted_indices[n_resample:]
        points_kept = points[keep_indices]

        # Sample new points from candidate pool
        if hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
            candidates = jnp.array(domain._mesh_points[tag])
        else:
            # Fallback: random perturbation of high-residual points
            high_res_indices = sorted_indices[-n_resample:]
            new_points = points[high_res_indices]
            key1, key2 = jax.random.split(rng_key)
            noise = jax.random.normal(key1, new_points.shape) * 0.01
            new_points = new_points + noise
            return jnp.concatenate([points_kept, new_points], axis=0)

        # Sample k candidates per new slot, pick the one nearest to a high-residual anchor.
        key1, _ = jax.random.split(rng_key)
        candidate_indices = jax.random.choice(key1, candidates.shape[0], shape=(n_resample * self.k,), replace=True)
        candidate_points = candidates[candidate_indices].reshape(n_resample, self.k, -1)  # (n_resample, k, D)

        # High-residual anchors: the n_resample current points with largest residuals.
        anchor_points = points[sorted_indices[-n_resample:]]  # (n_resample, D)

        # For each of the n_resample groups, pick the candidate closest to ANY anchor.
        # Pairwise squared distances of shape (n_resample, k, n_resample), min over anchors -> (n_resample, k).
        diffs = candidate_points[:, :, None, :] - anchor_points[None, None, :, :]
        sq_dists = jnp.sum(diffs * diffs, axis=-1)  # (n_resample, k, n_resample)
        min_dist_to_anchor = jnp.min(sq_dists, axis=-1)  # (n_resample, k)
        best_in_group = jnp.argmin(min_dist_to_anchor, axis=-1)  # (n_resample,)

        new_points = jnp.take_along_axis(candidate_points, best_in_group[:, None, None], axis=1)[:, 0, :]

        return jnp.concatenate([points_kept, new_points], axis=0)
