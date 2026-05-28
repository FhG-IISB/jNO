"""User-facing stochastic noise API.

Usage::

    import jno

    # Scalar noise added to observed data — shape (N, 1)
    residual = net(x) - (u_obs + jno.noise.gaussian(std=0.01))

    # 2-D vector noise for a 2-D velocity field — shape (N, 2)
    uv_noisy = net(x, y) + jno.noise.gaussian(std=0.01, ndim=2)

    # Perturb a 1-D input coordinate
    u = net(x + jno.noise.gaussian(std=1e-3))

    # Uniform noise on boundary
    bc = net(xb) - (u_bc + jno.noise.uniform(low=-0.005, high=0.005))

Noise nodes are symbolic :class:`~jno.trace.Noise` placeholders.  During
training a fresh realisation is drawn every step from the solver's PRNG key.
The key is derived from the global seed set via :func:`jno.setup` (or the seed
in ``.jno.toml``), so noise is **fully reproducible** across runs when the same
seed is used — no manual key management is required.

When :meth:`~jno.core.core.eval` is called without an explicit key, noise nodes
return zeros so post-training evaluation is deterministic.

Output shape
------------
Every noise node produces an array of shape ``(N, ndim)`` where ``N`` is the
number of active spatial points (inferred at evaluation time from the domain
context) and ``ndim`` is the ``ndim`` argument (default ``1``).  Use ``ndim``
to match the trailing dimension of the expression you are adding noise to::

    # scalar field  (N, 1)
    u + jno.noise.gaussian(std=0.01)

    # 2-D velocity  (N, 2)
    uv + jno.noise.gaussian(std=0.01, ndim=2)

    # 3-D position  (N, 3)
    xyz + jno.noise.gaussian(std=0.01, ndim=3)
"""

from .trace import Noise


class _NoiseNamespace:
    """Namespace exposed as ``jno.noise``."""

    def gaussian(self, std: float = 1.0, ndim: int = 1) -> Noise:
        """Zero-mean Gaussian noise ~ N(0, std²).

        Parameters
        ----------
        std : float
            Standard deviation of the noise.
        ndim : int
            Number of output dimensions (last axis size). Use ``ndim=2`` for
            2-D vector noise, ``ndim=3`` for 3-D, etc.  Default ``1``.
        """
        return Noise("gaussian", std=float(std), ndim=int(ndim))

    def uniform(self, low: float = -1.0, high: float = 1.0, ndim: int = 1) -> Noise:
        """Uniform noise ~ U(low, high).

        Parameters
        ----------
        low : float
            Lower bound of the uniform distribution.
        high : float
            Upper bound of the uniform distribution.
        ndim : int
            Number of output dimensions. Default ``1``.
        """
        return Noise("uniform", low=float(low), high=float(high), ndim=int(ndim))

    def laplace(self, std: float = 1.0, ndim: int = 1) -> Noise:
        """Zero-mean Laplace noise with the given standard deviation.

        Parameters
        ----------
        std : float
            Standard deviation (scale = std / sqrt(2)).
        ndim : int
            Number of output dimensions. Default ``1``.
        """
        return Noise("laplace", std=float(std), ndim=int(ndim))


noise = _NoiseNamespace()
