"""Base class for resampling strategies."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from ...utils.logger import Logger, PrintFallback, get_logger


def _get_candidate_pool(
    candidates: Optional[np.ndarray],
    domain,
    tag: str,
) -> Optional[jnp.ndarray]:
    """Return candidate points for resampling as a JAX array, or None."""
    if candidates is not None:
        return jnp.array(candidates)
    if hasattr(domain, "draw_candidates"):
        pool, _ = domain.draw_candidates(tag)
        if pool is not None and len(pool) > 0:
            return jnp.array(pool)
    return None


def _require_pool(pool, tag: str, strategy: str):
    """Fail loudly when a strategy has nothing to resample from. **Currently unused.**

    With no pool, CR3, R3 and RandomResampling return the caller's own points: the caller asked for
    resampling, gets a silent no-op, and the run looks healthy while the collocation points never
    move. That is an odd fit for a codebase whose first rule is never fail silently -- but it is a
    DELIBERATE contract, asserted by
    ``test_resampling.py::test_random_resampling_without_candidates_returns_input``, so switching it
    is a behaviour change for callers to decide on rather than one to make in passing. This is the
    error it would raise. RAD is unaffected either way: it perturbs its high-residual points instead.
    """
    if pool is None:
        raise ValueError(
            f"{strategy}.resample(tag={tag!r}): no candidate pool to draw from, so there is nothing "
            f"to resample and the points would be returned unchanged. Either pass candidates=<(M, D) "
            f"array>, or use a domain whose tag has a pool (domain.draw_candidates({tag!r}))."
        )
    return pool


def _retain_and_refill(
    points: jnp.ndarray,
    score: jnp.ndarray,
    pool: jnp.ndarray,
    rng_key: jnp.ndarray,
    *,
    threshold: jnp.ndarray,
    n_min: int,
    n_max: int,
) -> jnp.ndarray:
    """Keep the highest-scoring points, refill every other slot from ``pool``.

    This is the traceable form of "keep where score >= threshold, clamped to [n_min, n_max]".
    The obvious spelling, ``points[score >= threshold]``, has an output shape that depends on the
    VALUES of ``score``, so it cannot be jitted and cannot be differentiated -- and the follow-up
    ``int(jnp.sum(mask))`` forces a tracer to a Python int, which is the actual error.

    Ranking makes the shape static instead: slot *i* holds either the *i*-th highest-scoring current
    point or a fresh candidate, chosen by :func:`jnp.where`. Both branches are gathers, so gradients
    flow to ``points`` AND to ``pool``. ``n_min``/``n_max`` are counts derived from ``points.shape``,
    which is static under trace, so the clamp costs nothing.

    The selection itself (which slot keeps and which refills) is discrete and carries no gradient --
    ``argsort`` and the comparison are step functions. What is differentiable is the *position* of
    every returned point with respect to the points and the candidate pool it was gathered from.
    """
    n = points.shape[0]
    order = jnp.argsort(score)[::-1]  # descending: best score first
    ranked = points[order]
    rank = jnp.arange(n)
    above = score[order] >= threshold
    keep = (rank < n_min) | (above & (rank < n_max))
    fresh = pool[jax.random.choice(rng_key, pool.shape[0], shape=(n,), replace=True)]
    return jnp.where(keep[:, None], ranked, fresh)


class ResamplingStrategy(ABC):
    """Base class for collocation point resampling strategies.

    Resampling strategies adaptively update the training points during
    optimization to focus computation on regions with high errors or
    interesting dynamics.
    """

    # Whether ``resample()`` actually reads its ``residuals`` argument.
    # Subclasses that ignore residuals (e.g. uniform random resampling)
    # override to False so the training loop can skip an expensive
    # residual forward pass on resample epochs.
    needs_residuals: bool = True

    def __init__(
        self,
        resample_every: int = 100,
        resample_fraction: float = 1.0,
        start_epoch: int = 0,
    ):
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
        self.log: Logger | PrintFallback = get_logger()

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
    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """Compute new sample points.

        Args:
            points: Current points (N, D)
            residuals: Residual values at current points (N,) or (B, N)
            domain: domain object for sampling new candidates
            tag: domain tag being resampled
            epoch: Current training epoch
            rng_key: JAX random key
            candidates: Pre-drawn candidate points (N_pool, D) from
                ``domain.draw_candidates(tag)``.  When provided, strategies
                draw new points from this array; when None the strategy calls
                ``domain.draw_candidates`` itself.

                **Under ``jax.jit``, pass this explicitly.** Leaving it None still traces --
                ``draw_candidates`` runs on the host at trace time -- but the pool it returns is
                then a compile-time constant, so every later call reselects from that one frozen
                cloud. Measured: 25 600 draws from a jitted strategy with candidates=None yielded
                2559 distinct points against a pool of 2560. Selection still varies with the key,
                which is what makes it easy to miss.

        Returns:
            New points (N, D)
        """
        pass

    def update_epoch(self, epoch: int):
        """Update internal epoch tracking."""
        self._last_resample_epoch = epoch


"""CR3: Causal Retain-Resample with time gating for time-dependent problems."""


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

    def next_gamma(self, residuals: jnp.ndarray, gate_values: jnp.ndarray):
        """The gamma this strategy would advance to, as a pure function of its inputs.

        Kept separate from :meth:`resample` because gamma is state carried BETWEEN resamples: doing
        the update inside a traced call would either capture a tracer in ``self`` or be dropped.
        Returns a JAX scalar; the caller decides when to make it concrete.
        """
        Lg = jnp.mean(residuals**2 * gate_values)
        step = self.eta_g * jnp.minimum(jnp.exp(-self.epsilon * Lg), self.delta_max)
        return jnp.clip(self.gamma + step, -1.0, 2.0)

    def _update_gamma(self, residuals: jnp.ndarray, gate_values: jnp.ndarray):
        """Advance gamma in place. Eager only -- see :meth:`next_gamma`."""
        if len(residuals) == 0:
            return
        self.gamma = float(self.next_gamma(residuals, gate_values))
        self.gamma_history.append(self.gamma)

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """Resample using causal time gating.

        Args:
            points: Current points (N, D)
            residuals: Residual magnitudes (N,) or (B, N)
            domain: domain object
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key
            candidates: Pre-drawn candidate points from draw_candidates()

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

        pool = _get_candidate_pool(candidates, domain, tag)
        if pool is None:  # same no-op contract as RandomResampling; see _require_pool's note
            return points

        result = _retain_and_refill(
            points,
            F,
            pool,
            rng_key,
            threshold=jnp.mean(F),
            n_min=max(1, int(self.min_keep_frac * n_points)),
            n_max=min(n_points - 1, int(self.max_keep_frac * n_points)),
        )

        # gamma is per-call adaptive state, so advancing it here would either capture a tracer or
        # be silently skipped under jit. `next_gamma` is the pure form; the training loop advances
        # it between resamples, where the values are concrete.
        if not isinstance(jnp.asarray(F), jax.core.Tracer):
            self.gamma = float(self.next_gamma(residuals, gate_values))
            self.gamma_history.append(self.gamma)

        return result


"""HA: Hybrid Adaptive resampling strategy.

Alternates between random and adaptive phases to provide regularization
while focusing on high-error regions.
"""


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
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """Hybrid adaptive resampling.

        Args:
            points: Current points (N, D)
            residuals: Residual magnitudes (N,) or (B, N)
            domain: domain object
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key
            candidates: Pre-drawn candidate points from draw_candidates()

        Returns:
            New points (N, D)
        """
        phase = self._decide_phase()
        self._apply_count += 1

        n_points = points.shape[0]
        pool = _get_candidate_pool(candidates, domain, tag)

        if phase == "random":
            # Fully random refresh
            if pool is not None:
                indices = jax.random.choice(rng_key, pool.shape[0], shape=(n_points,), replace=True)
                return pool[indices]
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
            if n_new > 0 and pool is not None:
                fill_key, _ = jax.random.split(rng_key)
                new_indices = jax.random.choice(fill_key, pool.shape[0], shape=(n_new,), replace=True)
                new_points = pool[new_indices]
                result = jnp.concatenate([retained_points, new_points], axis=0)
                assert result.shape[0] == n_points, f"Expected {n_points}, got {result.shape[0]}"
                return result
            elif n_new == 0:
                return retained_points
            else:
                # Fallback: pad with retained points if no candidates
                pad_key, _ = jax.random.split(rng_key)
                pad_indices = jax.random.choice(pad_key, n_retain, shape=(n_new,), replace=True)
                result = jnp.concatenate([retained_points, retained_points[pad_indices]], axis=0)
                return result


"""PINNFluence: Influence function-based resampling (simplified version)."""


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
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """Resample using simplified influence-based scoring.

        Args:
            points: Current points (N, D)
            residuals: Residual magnitudes (N,) or (B, N)
            domain: domain object
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key
            candidates: Pre-drawn candidate points from draw_candidates()

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

        # Simplified scoring: residual magnitude + small penalty for uniformity
        scores = residuals + 0.1 * jnp.std(residuals)

        # Keep high-scoring points
        n_keep = n_points - int(n_points * self.resample_fraction)
        keep_indices = jnp.argsort(scores)[-n_keep:]
        kept_points = points[keep_indices]

        # Sample new points from candidates using influence-based weights
        n_new = n_points - len(kept_points)

        pool = _get_candidate_pool(candidates, domain, tag)
        if n_new > 0 and pool is not None:
            n_candidates = len(pool)

            # Evaluate a subset of candidates
            n_eval = min(n_candidates, int(n_points * self.candidate_factor))

            key1, key2 = jax.random.split(rng_key)

            if n_eval < n_candidates:
                eval_indices = jax.random.choice(key1, n_candidates, shape=(n_eval,), replace=False)
                eval_candidates = pool[eval_indices]
            else:
                eval_candidates = pool

            # Score candidates based on distance to high-residual current points (vectorized).
            n_top = min(50, n_points)
            high_res_points = points[jnp.argsort(residuals)[-n_top:]]  # (n_top, D)

            # Pairwise squared distances (n_eval, n_top); take min over anchors -> (n_eval,).
            diffs = eval_candidates[:, None, :] - high_res_points[None, :, :]
            sq_dists = jnp.sum(diffs * diffs, axis=-1)
            candidate_scores = 1.0 / (jnp.min(sq_dists, axis=-1) + 1e-4)

            # Compute sampling weights
            weights = jnp.power(candidate_scores + 1e-12, self.alpha) + self.c
            weights = jnp.clip(weights, 0, None)
            total = jnp.sum(weights)

            # Numerically safe normalization; fall back to uniform when weights collapse.
            safe_probs = jnp.where(
                total > 0,
                weights / jnp.maximum(total, 1e-20),
                jnp.ones_like(weights) / weights.shape[0],
            )
            safe_probs = safe_probs / jnp.sum(safe_probs)

            new_indices = jax.random.choice(
                key2,
                eval_candidates.shape[0],
                shape=(n_new,),
                replace=True,
                p=safe_probs,
            )
            new_points = eval_candidates[new_indices]

            result = jnp.concatenate([kept_points, new_points], axis=0)
        else:
            result = kept_points

        return result[:n_points]


"""R3: Residual-based Refinement and Resampling strategy."""


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
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """Keep high-residual points, resample low-residual points.

        Args:
            points: Current points (N, D)
            residuals: Residual magnitudes (N,) or (B, N)
            domain: domain object
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key
            candidates: Pre-drawn candidate points from draw_candidates()

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
            threshold = jnp.asarray(self.threshold_mode, dtype=residuals.dtype)
        else:
            threshold = jnp.mean(residuals)

        pool = _get_candidate_pool(candidates, domain, tag)
        if pool is None:  # same no-op contract as RandomResampling; see _require_pool's note
            return points

        return _retain_and_refill(
            points,
            residuals,
            pool,
            rng_key,
            threshold=threshold,
            n_min=int(self.min_keep_frac * n_points),
            n_max=int(self.max_keep_frac * n_points),
        )


"""RAD: Residual-based Adaptive Distribution resampling."""


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
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
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
            candidates: Pre-drawn candidate points from draw_candidates()

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
        pool = _get_candidate_pool(candidates, domain, tag)
        if pool is None:
            # Fallback: random perturbation of high-residual points
            high_res_indices = sorted_indices[-n_resample:]
            new_points = points[high_res_indices]
            key1, _ = jax.random.split(rng_key)
            noise = jax.random.normal(key1, new_points.shape) * 0.01
            new_points = new_points + noise
            return jnp.concatenate([points_kept, new_points], axis=0)

        # Sample k candidates per new slot, pick the one nearest to a high-residual anchor.
        key1, _ = jax.random.split(rng_key)
        candidate_indices = jax.random.choice(key1, pool.shape[0], shape=(n_resample * self.k,), replace=True)
        candidate_points = pool[candidate_indices].reshape(n_resample, self.k, -1)  # (n_resample, k, D)

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


"""Random resampling strategy - simple baseline."""


class RandomResampling(ResamplingStrategy):
    """Randomly resample a fraction of collocation points.

    Useful as a baseline to prevent overfitting to specific points.
    """

    needs_residuals = False

    def resample(
        self,
        points: jnp.ndarray,
        residuals: jnp.ndarray,
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """Randomly replace a fraction of points with new samples from domain.

        Args:
            points: Current points (N, D)
            residuals: Unused for random resampling
            domain: domain object for sampling
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key
            candidates: Pre-drawn candidate points from draw_candidates()

        Returns:
            New points with some randomly replaced
        """
        n_points = points.shape[0]
        n_resample = int(n_points * self.resample_fraction)

        if n_resample == 0:
            return points

        # Get all available candidate points from domain
        pool = _get_candidate_pool(candidates, domain, tag)
        if pool is None:
            return points

        # Randomly select which points to replace
        key1, key2 = jax.random.split(rng_key)
        replace_indices = jax.random.choice(key1, n_points, shape=(n_resample,), replace=False)

        # Randomly sample new points from candidates
        new_points_indices = jax.random.choice(key2, pool.shape[0], shape=(n_resample,), replace=True)
        new_points = pool[new_points_indices]

        # Replace selected points
        points = points.at[replace_indices].set(new_points)

        return points


"""RARD: Residual-based Adaptive Refinement with Distribution resampling."""


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
        domain: Any,
        tag: str,
        epoch: int,
        rng_key: jnp.ndarray,
        candidates: Optional[np.ndarray] = None,
    ) -> jnp.ndarray:
        """Resample using residual-weighted importance sampling.

        Args:
            points: Current points (N, D)
            residuals: Residual magnitudes (N,) or (B, N)
            domain: domain object
            tag: domain tag
            epoch: Current epoch
            rng_key: JAX random key
            candidates: Pre-drawn candidate points from draw_candidates()

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

        # Sample new points from candidates, weighted by residual^power at the
        # nearest current point (importance sampling over the mesh pool).
        pool = _get_candidate_pool(candidates, domain, tag)
        if pool is not None:
            # For each candidate, find nearest current point and inherit its residual-power weight.
            # Pairwise squared distances (|C|, N) — O(|C|*N*D); acceptable at resample cadence.
            diffs = pool[:, None, :] - points[None, :, :]
            sq_dists = jnp.sum(diffs * diffs, axis=-1)
            nearest = jnp.argmin(sq_dists, axis=-1)  # (|C|,)
            cand_weights = jnp.power(residuals[nearest] + 1e-10, self.power)
            cand_weights = cand_weights / jnp.sum(cand_weights)

            new_indices = jax.random.choice(
                rng_key,
                pool.shape[0],
                shape=(n_resample,),
                replace=True,
                p=cand_weights,
            )
            new_points = pool[new_indices]
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


class sampler:
    """Factory class for creating resampling strategies."""

    @staticmethod
    def random(
        resample_every: int = 100,
        resample_fraction: float = 0.1,
        start_epoch: int = 1000,
    ):
        """Random resampling - simple baseline to prevent overfitting.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to resample
            start_epoch: Start resampling after this many epochs
        """
        return RandomResampling(resample_every, resample_fraction, start_epoch)

    @staticmethod
    def rad(
        resample_every: int = 100,
        resample_fraction: float = 0.1,
        start_epoch: int = 1000,
        k: int = 10,
    ):
        """Residual-based Adaptive Distribution (RAD) resampling.

        Focuses on regions with high PDE residuals (high errors).

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to resample
            start_epoch: Start resampling after this many epochs
            k: Number of top residual points to cluster around
        """
        return RAD(resample_every, resample_fraction, start_epoch, k)

    @staticmethod
    def rard(
        resample_every: int = 100,
        resample_fraction: float = 0.1,
        start_epoch: int = 1000,
        power: float = 2.0,
    ):
        """Residual-based Adaptive Refinement with Distribution (RARD).

        Uses importance sampling based on residual^power distribution.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to resample
            start_epoch: Start resampling after this many epochs
            power: Power for residual-based importance weighting
        """
        return RARD(resample_every, resample_fraction, start_epoch, power)

    @staticmethod
    def ha(
        resample_every: int = 100,
        resample_fraction: float = 0.5,
        start_epoch: int = 1000,
        alternate: bool = True,
        random_first: bool = True,
    ):
        """Hybrid Adaptive (HA) resampling strategy.

        Alternates between random and adaptive phases for regularization
        while focusing on high-error regions.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to retain in adaptive phase
            start_epoch: Start resampling after this many epochs
            alternate: Whether to alternate between random and adaptive phases
            random_first: Start with random phase if True
        """
        return HA(resample_every, resample_fraction, start_epoch, alternate, random_first)

    @staticmethod
    def r3(
        resample_every: int = 100,
        resample_fraction: float = 0.7,
        start_epoch: int = 1000,
        threshold_mode: str = "mean",
        min_keep_frac: float = 0.3,
        max_keep_frac: float = 0.9,
    ):
        """Residual-based Refinement and Resampling (R3).

        Keeps points with high residuals and replaces low-residual points
        with new samples from the candidate pool.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Not used for R3 (uses threshold instead)
            start_epoch: Start resampling after this many epochs
            threshold_mode: Threshold for keeping points ("mean", "median", or float)
            min_keep_frac: Minimum fraction of points to keep
            max_keep_frac: Maximum fraction of points to keep
        """
        return R3(resample_every, resample_fraction, start_epoch, threshold_mode, min_keep_frac, max_keep_frac)

    @staticmethod
    def cr3(
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
        """Causal Retain-Resample (CR3) with time gating.

        Applies causal time gate for time-dependent PDEs where causality matters
        (e.g., wave equations, diffusion).

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
        return CR3(
            resample_every,
            resample_fraction,
            start_epoch,
            t_index,
            alpha,
            gamma0,
            eta_g,
            epsilon,
            delta_max,
            min_keep_frac,
            max_keep_frac,
        )

    @staticmethod
    def pinnfluence(
        resample_every: int = 500,
        resample_fraction: float = 0.2,
        start_epoch: int = 2000,
        alpha: float = 1.0,
        c: float = 1.0,
        candidate_factor: float = 3.0,
    ):
        """PINNFluence: Influence function-based adaptive sampling (simplified).

        Uses gradient-based scoring to identify points with high potential impact
        on reducing the loss.

        Args:
            resample_every: Resample every N epochs (use larger values, ~500+)
            resample_fraction: Fraction of points to replace
            start_epoch: Start resampling after this many epochs
            alpha: Score exponent for sampling (default 1.0)
            c: Additive smoothing constant (default 1.0)
            candidate_factor: Pool size multiplier (default 3.0)
        """
        return PINNFluence(resample_every, resample_fraction, start_epoch, alpha, c, candidate_factor)
