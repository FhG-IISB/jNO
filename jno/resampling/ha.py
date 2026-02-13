"""HA: Hybrid Adaptive resampling strategy.

Alternates between random and adaptive phases to provide regularization
while focusing on high-error regions.
"""

import jax
import jax.numpy as jnp
from .base import ResamplingStrategy


class HA(ResamplingStrategy):
    """Hybrid Adaptive (HA) resampling strategy.

    Alternates between two phases:
    - Random: Fully random refresh of interior points
    - Adaptive: Retain high-residual points, fill remainder randomly

    This provides regularization through random phases while still
    focusing on high-error regions during adaptive phases.

    Reference:
        "Hybrid adaptive sampling for physics-informed neural networks"
        https://link.springer.com/article/10.1007/s10489-024-06195-2
    """

    def __init__(
        self,
        resample_every: int = 100,
        resample_fraction: float = 0.5,
        start_epoch: int = 1000,
        alternate: bool = True,
        random_first: bool = True,
    ):
        """Initialize HA resampling.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to retain in adaptive phase (beta)
            start_epoch: Start resampling after this many epochs
            alternate: Whether to alternate between random and adaptive phases
            random_first: Start with random phase if True
        """
        super().__init__(resample_every, resample_fraction, start_epoch)
        self.alternate = alternate
        self.random_first = random_first
        self._apply_count = 0

    def _decide_phase(self) -> str:
        """Decide current phase based on apply count."""
        if not self.alternate:
            return "adaptive"

        is_random = (self._apply_count % 2 == 0) if self.random_first else (self._apply_count % 2 == 1)
        return "random" if is_random else "adaptive"

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Hybrid adaptive resampling.

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
        phase = self._decide_phase()
        self._apply_count += 1

        n_points = points.shape[0]

        if phase == "random":
            # Fully random refresh
            if hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
                candidates = jnp.array(domain._mesh_points[tag])
                indices = jax.random.choice(rng_key, candidates.shape[0], shape=(n_points,), replace=True)
                return candidates[indices]
            return points
        else:
            # Adaptive phase: retain high-residual, fill with random
            if residuals.ndim > 1:
                residuals = jnp.mean(jnp.abs(residuals), axis=0)
            else:
                residuals = jnp.abs(residuals)

            if residuals.shape[0] != n_points:
                return points

            # Retain top fraction by residual
            n_retain = int(n_points * self.resample_fraction)
            n_retain = max(1, min(n_retain, n_points - 1))
            n_new = n_points - n_retain

            retain_indices = jnp.argsort(residuals)[-n_retain:]
            retained_points = points[retain_indices]

            # Fill remainder with random
            if n_new > 0 and hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
                candidates = jnp.array(domain._mesh_points[tag])
                new_indices = jax.random.choice(rng_key, candidates.shape[0], shape=(n_new,), replace=True)
                new_points = candidates[new_indices]
                result = jnp.concatenate([retained_points, new_points], axis=0)
                assert result.shape[0] == n_points, f"Expected {n_points}, got {result.shape[0]}"
                return result
            elif n_new == 0:
                return retained_points
            else:
                # Fallback: pad with retained points if no candidates
                pad_indices = jax.random.choice(rng_key, n_retain, shape=(n_new,), replace=True)
                result = jnp.concatenate([retained_points, retained_points[pad_indices]], axis=0)
                return result
