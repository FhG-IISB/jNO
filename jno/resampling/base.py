"""Base class for resampling strategies."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, List, Tuple
import jax.numpy as jnp
import jax

from ..trace import get_primary_tag


class ResamplingStrategy(ABC):
    """Base class for collocation point resampling strategies.

    Resampling strategies adaptively update the training points during
    optimization to focus computation on regions with high errors or
    interesting dynamics.
    """

    def __init__(self, resample_every: int = 100, resample_fraction: float = 1.0, start_epoch: int = 0):
        """Initialize resampling strategy.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to resample (0.0 to 1.0)
            start_epoch: Start resampling after this many epochs
        """
        self.resample_every = resample_every
        self.resample_fraction = resample_fraction
        self.start_epoch = start_epoch
        self._last_resample_epoch = -1

    def should_resample(self, epoch: int) -> bool:
        """Check if resampling should occur at this epoch.

        Args:
            epoch: Current training epoch

        Returns:
            True if resampling should happen
        """
        if epoch < self.start_epoch:
            return False
        if (epoch - self._last_resample_epoch) >= self.resample_every:
            return True
        return False

    @abstractmethod
    def resample(self, points: jnp.ndarray, residuals: jnp.ndarray, domain, tag: str, epoch: int, rng_key: jnp.ndarray) -> jnp.ndarray:
        """Compute new sample points.

        Args:
            points: Current points (N, D)
            residuals: Residual values at current points (N,) or (B, N)
            domain: domain object for sampling new candidates
            tag: domain tag being resampled
            epoch: Current training epoch
            rng_key: JAX random key

        Returns:
            New points (N, D)
        """
        pass

    def update_epoch(self, epoch: int):
        """Update internal epoch tracking."""
        self._last_resample_epoch = epoch

    def apply_resampling(self, domain, constraints: List, tags: List[str], compiled: List[Callable], params: Dict, layer_info: Dict, context: Dict[str, jax.Array], epoch: int, rng: jax.Array) -> Tuple[Dict[str, jax.Array], jax.Array]:
        """Apply resampling strategies if configured."""
        import numpy as np

        if not hasattr(domain, "_resampling_strategies"):
            return context, rng

        for tag, strategy in domain._resampling_strategies.items():
            if not strategy.should_resample(epoch):
                continue

            # Compute residuals for constraints on this tag
            tag_residuals = self.compute_tag_residuals(tag, constraints, tags, compiled, params, context)

            if not tag_residuals:
                continue

            # Combine and resample
            combined_residuals = jnp.mean(jnp.stack(tag_residuals, axis=0), axis=0)
            current_points = context[tag]

            rng, resample_key = jax.random.split(rng)
            new_points = self.resample_all_batches(strategy, current_points, combined_residuals, domain, tag, epoch, resample_key)

            # Update domain
            domain.context[tag] = np.array(new_points)
            context[tag] = new_points

            strategy.update_epoch(epoch)
            self.log.info(f"Resampled {tag} points (epoch {epoch+1})")

        return context, rng

    def compute_tag_residuals(self, tag: str, constraints: List, tags: List[str], compiled: List[Callable], params: Dict, context: Dict[str, jax.Array]) -> List[jax.Array]:
        """Compute residuals for constraints evaluated on a specific tag."""
        tag_residuals = []
        expected_n_points = context[tag].shape[1]

        for i, expr in enumerate(constraints):
            expr_tag = get_primary_tag(expr)
            if expr_tag != tag:
                continue

            residual = compiled[i](params, context)

            if residual.shape[-1] == expected_n_points:
                tag_residuals.append(residual)

        return tag_residuals

    def resample_all_batches(self, strategy, current_points: jax.Array, residuals: jax.Array, domain, tag: str, epoch: int, rng: jax.Array) -> jax.Array:
        """Resample points for all batches."""
        new_batches = []

        for b in range(current_points.shape[0]):
            batch_key = jax.random.fold_in(rng, b)
            batch_points = current_points[b]
            batch_residuals = residuals[b] if residuals.ndim > 1 else residuals

            new_batch = strategy.resample(batch_points, batch_residuals, domain, tag, epoch, batch_key)
            new_batches.append(new_batch)

        return jnp.stack(new_batches, axis=0)
