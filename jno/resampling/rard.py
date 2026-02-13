"""RARD: Residual-based Adaptive Refinement with Distribution resampling."""

import jax
import jax.numpy as jnp
from .base import ResamplingStrategy


class RARD(ResamplingStrategy):
    """Residual-based Adaptive Refinement with Distribution (RARD).

    Similar to RAD but uses importance sampling based on residual distribution.
    Samples new points proportionally to residual^p where p is a power parameter.
    """

    def __init__(
        self,
        resample_every: int = 100,
        resample_fraction: float = 0.1,
        start_epoch: int = 1000,
        power: float = 2.0,
    ):
        """Initialize RARD resampling.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to resample
            start_epoch: Start resampling after this many epochs
            power: Power for residual-based importance weighting
        """
        super().__init__(resample_every, resample_fraction, start_epoch)
        self.power = power

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Resample using residual-weighted importance sampling.

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
        n_resample = int(n_points * self.resample_fraction)

        if n_resample == 0 or residuals.shape[0] != n_points:
            return points

        # Compute importance weights: weight = residual^power
        weights = jnp.power(residuals + 1e-10, self.power)
        weights = weights / jnp.sum(weights)

        # Sort by weight (ascending) and remove lowest-weight points
        sorted_indices = jnp.argsort(weights)
        keep_indices = sorted_indices[n_resample:]
        points_kept = points[keep_indices]

        # Sample new points from candidates
        if hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
            candidates = jnp.array(domain._mesh_points[tag])

            # Randomly sample from candidates
            key1, key2 = jax.random.split(rng_key)
            new_indices = jax.random.choice(key1, candidates.shape[0], shape=(n_resample,), replace=True)
            new_points = candidates[new_indices]
        else:
            # Fallback: sample from current high-residual regions
            keep_residuals = residuals[keep_indices]
            keep_weights = jnp.power(keep_residuals + 1e-10, self.power)
            keep_weights = keep_weights / jnp.sum(keep_weights)

            sampled_indices = jax.random.choice(
                rng_key,
                keep_indices.shape[0],
                shape=(n_resample,),
                replace=True,
                p=keep_weights,
            )
            new_points = points_kept[sampled_indices]

        return jnp.concatenate([points_kept, new_points], axis=0)
