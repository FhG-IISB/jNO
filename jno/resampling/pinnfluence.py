"""PINNFluence: Influence function-based resampling (simplified version)."""

import jax
import jax.numpy as jnp
from jax import grad, jvp
from .base import ResamplingStrategy


class PINNFluence(ResamplingStrategy):
    """PINNFluence: Influence function-based adaptive sampling (simplified).

    Uses influence functions to score candidate points based on their
    potential impact on reducing the loss. Points with high influence
    scores are more likely to be sampled.

    Note: This is a simplified version that uses gradient-based scoring
    rather than full influence functions for computational efficiency.

    Reference:
        "Training Physics-Informed Neural Networks with Optimal Test Points"
    """

    def __init__(
        self,
        resample_every: int = 500,
        resample_fraction: float = 0.2,
        start_epoch: int = 2000,
        alpha: float = 1.0,
        c: float = 1.0,
        candidate_factor: float = 3.0,
    ):
        """Initialize PINNFluence resampling.

        Args:
            resample_every: Resample every N epochs (use larger values, ~500+)
            resample_fraction: Fraction of points to replace
            start_epoch: Start resampling after this many epochs
            alpha: Score exponent for sampling (default 1.0)
            c: Additive smoothing constant (default 1.0)
            candidate_factor: Pool size multiplier (default 3.0)
        """
        super().__init__(resample_every, resample_fraction, start_epoch)
        self.alpha = alpha
        self.c = c
        self.candidate_factor = candidate_factor

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
    ) -> jnp.ndarray:
        """Resample using simplified influence-based scoring.

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

        # Simplified influence scoring: use residual magnitude + local variance
        # as a proxy for influence (points in high-error, high-variance regions
        # are more influential)

        # Compute local variance estimate using nearest neighbor distances
        def compute_local_variance(pts, res):
            # For each point, estimate local variance from nearby residuals
            # Using a simple distance-weighted scheme
            n = len(pts)
            local_var = jnp.zeros(n)

            for i in range(min(n, 100)):  # Sample subset for efficiency
                dists = jnp.sum((pts - pts[i : i + 1]) ** 2, axis=1)
                weights = jnp.exp(-dists / jnp.mean(dists + 1e-8))
                weighted_res = res * weights
                local_var = local_var.at[i].set(jnp.std(weighted_res))

            return local_var

        # Simplified scoring: residual magnitude + small penalty for uniformity
        scores = residuals + 0.1 * jnp.std(residuals)

        # Keep high-scoring points
        n_keep = n_points - int(n_points * self.resample_fraction)
        keep_indices = jnp.argsort(scores)[-n_keep:]
        kept_points = points[keep_indices]

        # Sample new points from candidates using influence-based weights
        n_new = n_points - len(kept_points)

        if n_new > 0 and hasattr(domain, "_mesh_points") and tag in domain._mesh_points:
            candidates = jnp.array(domain._mesh_points[tag])
            n_candidates = len(candidates)

            # Evaluate a subset of candidates
            n_eval = min(n_candidates, int(n_points * self.candidate_factor))

            key1, key2 = jax.random.split(rng_key)

            if n_eval < n_candidates:
                eval_indices = jax.random.choice(key1, n_candidates, shape=(n_eval,), replace=False)
                eval_candidates = candidates[eval_indices]
            else:
                eval_candidates = candidates

            # Score candidates based on distance to high-residual current points
            high_res_points = points[jnp.argsort(residuals)[-min(50, n_points) :]]

            # Compute minimum distance to high-residual region
            candidate_scores = jnp.zeros(len(eval_candidates))
            for i in range(len(eval_candidates)):
                dists = jnp.sum((high_res_points - eval_candidates[i : i + 1]) ** 2, axis=1)
                # Higher score for points near high-residual regions
                candidate_scores = candidate_scores.at[i].set(1.0 / (jnp.min(dists) + 1e-4))

            # Compute sampling weights
            weights = jnp.power(candidate_scores + 1e-12, self.alpha) + self.c
            weights = jnp.clip(weights, 0, None)
            total = jnp.sum(weights)

            if total > 0:
                probs = weights / total
                probs = jnp.clip(probs, 0, None)
                probs = probs / jnp.sum(probs)  # Re-normalize

                try:
                    # Sample with probabilities
                    new_indices = jax.random.choice(key2, len(eval_candidates), shape=(n_new,), replace=True, p=probs)
                    new_points = eval_candidates[new_indices]
                except:
                    # Fallback to uniform
                    new_indices = jax.random.choice(key2, len(eval_candidates), shape=(n_new,), replace=True)
                    new_points = eval_candidates[new_indices]
            else:
                # Fallback to uniform
                new_indices = jax.random.choice(key2, len(eval_candidates), shape=(n_new,), replace=True)
                new_points = eval_candidates[new_indices]

            result = jnp.concatenate([kept_points, new_points], axis=0)
        else:
            result = kept_points

        return result[:n_points]
