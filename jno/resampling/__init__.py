"""Resampling strategies for adaptive collocation point selection."""

from .base import ResamplingStrategy
from .random import RandomResampling
from .rad import RAD
from .rard import RARD
from .ha import HA
from .r3 import R3
from .cr3 import CR3
from .pinnfluence import PINNFluence


class sampler:
    """Factory class for creating resampling strategies."""

    @staticmethod
    def random(resample_every: int = 100, resample_fraction: float = 0.1, start_epoch: int = 1000):
        """Random resampling - simple baseline to prevent overfitting.

        Args:
            resample_every: Resample every N epochs
            resample_fraction: Fraction of points to resample
            start_epoch: Start resampling after this many epochs
        """
        return RandomResampling(resample_every, resample_fraction, start_epoch)

    @staticmethod
    def rad(resample_every: int = 100, resample_fraction: float = 0.1, start_epoch: int = 1000, k: int = 10):
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
    def rard(resample_every: int = 100, resample_fraction: float = 0.1, start_epoch: int = 1000, power: float = 2.0):
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
    def ha(resample_every: int = 100, resample_fraction: float = 0.5, start_epoch: int = 1000, alternate: bool = True, random_first: bool = True):
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
        return CR3(resample_every, resample_fraction, start_epoch, t_index, alpha, gamma0, eta_g, epsilon, delta_max, min_keep_frac, max_keep_frac)

    @staticmethod
    def pinnfluence(resample_every: int = 500, resample_fraction: float = 0.2, start_epoch: int = 2000, alpha: float = 1.0, c: float = 1.0, candidate_factor: float = 3.0):
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


__all__ = ["sampler"]
