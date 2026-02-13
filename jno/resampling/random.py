"""Random resampling strategy - simple baseline."""

import jax
import jax.numpy as jnp
from .base import ResamplingStrategy


class RandomResampling(ResamplingStrategy):
    """Randomly resample a fraction of collocation points.

    Useful as a baseline to prevent overfitting to specific points.
    """

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Randomly replace a fraction of points with new samples from domain.

        Args:
            points: Current points (N, D)
            residuals: Unused for random resampling
            domain: domain object for sampling
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key

        Returns:
            New points with some randomly replaced
        """
        n_points = points.shape[0]
        n_resample = int(n_points * self.resample_fraction)

        if n_resample == 0:
            return points

        # Get all available candidate points from domain
        if hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
            candidates = jnp.array(domain._mesh_points[tag])
        else:
            # Fallback: use current points
            return points

        # Randomly select which points to replace
        key1, key2 = jax.random.split(rng_key)
        replace_indices = jax.random.choice(key1, n_points, shape=(n_resample,), replace=False)

        # Randomly sample new points from candidates
        new_points_indices = jax.random.choice(key2, candidates.shape[0], shape=(n_resample,), replace=True)
        new_points = candidates[new_points_indices]

        # Replace selected points
        points = points.at[replace_indices].set(new_points)

        return points
