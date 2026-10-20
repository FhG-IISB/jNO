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

For a shape ``ndim`` cannot express — matching a field-valued expression rather than a trailing
component count, e.g. the prior of a flow-matching path — pass ``like=`` an existing traced
expression instead::

    x0 = jno.noise.gaussian(std=1.0, like=x1)      # x0 takes x1's shape exactly

``like`` is resolved from the reference at evaluation time, so it tracks ``batchsize``,
``solve(points=...)`` and time windows automatically. Mutually exclusive with ``ndim``.
"""

from typing import Any

import jax
import jax.numpy as jnp

from .trace import FunctionCall, Noise, Placeholder, _next_op_id


def _param(name, dist, value):
    """A distribution parameter: a number, or a **schedule** -- a callable of the march time.

    A schedule is what an SDE sampler's ``sigma_t`` is, so ``jno.noise.*`` accepts one; it is only
    meaningful inside ``Model.flow(..., noise=...)`` and raises anywhere else, at the point of use.
    """
    if callable(value):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        raise TypeError(
            f"jno.noise.{dist}: {name}= expects a number or a schedule (a callable of the march "
            f"time t, for use with Model.flow(noise=...)), got {type(value).__name__}."
        ) from None


def _resolve_shape_args(name, ndim, like):
    """Validate the ``ndim`` / ``like`` pair and return the kwargs for the Noise node.

    Also reports whether a shape was **explicitly asked for**, which the caller records on the node.
    ``ndim`` defaults to 1, so the resulting params cannot distinguish "the user wants a trailing
    axis of 1" from "nothing was said" -- and ``flow(noise=...)`` has to refuse the former (its
    shape comes from the state) while accepting the latter.
    """
    if like is not None:
        if ndim is not None:
            raise ValueError(
                f"jno.noise.{name}: pass either ndim= or like=, not both. `like` takes its shape from "
                f"the referenced expression, which is what ndim would otherwise be describing."
            )
        if not isinstance(like, Placeholder):
            raise TypeError(
                f"jno.noise.{name}: like= expects a traced jNO expression (a Placeholder), got "
                f"{type(like).__name__}. For a fixed shape use ndim=, or wrap the array with "
                f"jno.np.function(lambda: arr, [], 'ref')."
            )
        return {}, like, True
    return {"ndim": 1 if ndim is None else int(ndim)}, None, ndim is not None


class _NoiseNamespace:
    """Namespace exposed as ``jno.noise``."""

    def gaussian(self, std: float = 1.0, ndim: int = None, *, like=None) -> Noise:
        """Zero-mean Gaussian noise ~ N(0, std²).

        Parameters
        ----------
        std : float or callable
            Standard deviation of the noise: a number, or a schedule ``lambda t: ...`` (a callable of the march time, only inside ``Model.flow(..., noise=...)``).
        ndim : int
            Number of output dimensions (last axis size). Use ``ndim=2`` for
            2-D vector noise, ``ndim=3`` for 3-D, etc.  Default ``1``. Mutually exclusive with ``like``.
        like : Placeholder, optional
            Draw with the shape of another traced expression instead of ``(N, ndim)`` — e.g. a data
            field, so the prior of a flow-matching path lives in the data's space by construction::

                x0 = jno.noise.gaussian(std=1.0, like=x1)

            Resolved at evaluation time from the reference's own shape, so it follows ``batchsize``,
            ``solve(points=...)`` and time windows automatically; a shape written by hand would go
            stale under those and fail by broadcasting rather than by raising. Mutually exclusive
            with ``ndim``.
        """
        shape_kw, like, shape_explicit = _resolve_shape_args("gaussian", ndim, like)
        return Noise("gaussian", like=like, shape_explicit=shape_explicit, std=_param("std", "gaussian", std), **shape_kw)

    def uniform(self, low: float = -1.0, high: float = 1.0, ndim: int = None, *, like=None) -> Noise:
        """Uniform noise ~ U(low, high).

        Parameters
        ----------
        low : float or callable
            Lower bound of the uniform distribution: a number, or a schedule ``lambda t: ...`` (a callable of the march time, only inside ``Model.flow(..., noise=...)``).
        high : float or callable
            Upper bound of the uniform distribution, likewise.
        ndim : int
            Number of output dimensions. Default ``1``. Mutually exclusive with ``like``.
        like : Placeholder, optional
            Draw with the shape of another traced expression instead of ``(N, ndim)`` — e.g. a data
            field, so the prior of a flow-matching path lives in the data's space by construction::

                x0 = jno.noise.uniform(-1.0, 1.0, like=x1)

            Resolved at evaluation time from the reference's own shape, so it follows ``batchsize``,
            ``solve(points=...)`` and time windows automatically; a shape written by hand would go
            stale under those and fail by broadcasting rather than by raising. Mutually exclusive
            with ``ndim``.
        """
        shape_kw, like, shape_explicit = _resolve_shape_args("uniform", ndim, like)
        return Noise(
            "uniform",
            like=like,
            shape_explicit=shape_explicit,
            low=_param("low", "uniform", low),
            high=_param("high", "uniform", high),
            **shape_kw,
        )

    def grf(
        self,
        *coords: Any,
        length_scale: float = 0.1,
        variance: float = 1.0,
        kernel: str = "matern",
        nu: float = 1.5,
        modes: int = 256,
        ndim: int = 1,
    ) -> FunctionCall:
        r"""A Gaussian random field over ``coords``, redrawn every training step.

        Unlike the pointwise distributions above, a GRF is **spatially correlated**: nearby points
        get nearby values, with the correlation set by ``length_scale``. That makes it an input
        *function*, which is what turns operator learning from a parametric fit into an operator --
        a fresh in-distribution input every step, with no dataset and no data API::

            f = jno.noise.grf(x, y, length_scale=0.1)
            u = net(f).scalar.bind(x=x, y=y)
            crux = jno.core([(u.laplacian(x, y) + f).mse])

        Algorithm -- **spectral representation** (a.k.a. the randomization method / random Fourier
        features):

        .. math::
            f(x) = \sqrt{\sigma^2/M} \sum_{j=1}^{M}
                   \left[ a_j \cos(\omega_j \cdot x) + b_j \sin(\omega_j \cdot x) \right],
            \qquad a_j, b_j \sim \mathcal{N}(0, 1),

        with the frequencies :math:`\omega_j` drawn from the kernel's spectral density. Conditional
        on :math:`\{\omega_j\}` this is exactly a zero-mean Gaussian field, and its covariance
        converges to the kernel at :math:`O(M^{-1/2})`.

        For Matern-:math:`\nu` the spectral density is a multivariate Student-t with :math:`2\nu`
        degrees of freedom and scale :math:`I/\ell^2`, sampled as its Gaussian scale mixture
        :math:`\omega = z / (\ell\sqrt{g})` with :math:`z \sim \mathcal{N}(0, I)` and
        :math:`g \sim \mathrm{Gamma}(\nu, 1/\nu)`. The squared-exponential kernel is the
        :math:`\nu \to \infty` limit, where :math:`g \to 1`.

        References:
            Shinozuka & Deodatis, *Simulation of Stochastic Processes by Spectral Representation*,
            Appl. Mech. Rev. **44**(4) (1991), sections 2-3 -- the spectral representation.
            Rahimi & Recht, *Random Features for Large-Scale Kernel Machines*, NIPS 20 (2007),
            section 2 -- the Monte-Carlo frequency draw.
            Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006), section 4.2
            eq. 4.15 -- the Matern spectral density.

        Args:
            *coords: The coordinate Variables the field varies over, e.g. ``x, y``.
            length_scale: Correlation length. Much larger than the domain gives a near-constant
                field; much smaller than the point spacing gives near-white noise.
            variance: Marginal variance :math:`\sigma^2` of the field.
            kernel: ``"matern"`` or ``"rbf"`` (squared exponential).
            nu: Matern smoothness. ``0.5`` is Ornstein-Uhlenbeck (rough), ``1.5`` and ``2.5`` are
                the common differentiable choices. Ignored for ``"rbf"``.
            modes: Number of spectral modes :math:`M`. Costs ``O(B x N x M)`` inside the batch vmap
                -- at ``M=256`` and ``N=16384`` that is a real memory draw, so raise it knowingly.
            ndim: Number of independent field components (the trailing axis).

        Limitations:
            An **approximate** GP sample: exact only in the limit :math:`M \to \infty`, with
            covariance error :math:`O(M^{-1/2})`. Exact circulant embedding (Dietrich & Newsam,
            *SIAM J. Sci. Comput.* **18**(4), 1997) is not implemented because it needs a regular
            grid, whereas the evaluator sees a flat point cloud. Components share one frequency
            draw, so they share its finite-``M`` error while remaining independent fields.
            Like every noise node the realisation is fixed across the timesteps of one window.
        """
        if not coords:
            raise ValueError("jno.noise.grf: pass the coordinate Variables the field varies over, e.g. grf(x, y).")
        if length_scale <= 0:
            raise ValueError(f"jno.noise.grf: length_scale must be > 0, got {length_scale}.")
        if variance < 0:
            raise ValueError(f"jno.noise.grf: variance must be >= 0, got {variance}.")
        if int(modes) < 1:
            raise ValueError(f"jno.noise.grf: modes must be >= 1, got {modes}.")
        if kernel not in ("matern", "rbf"):
            raise ValueError(f"jno.noise.grf: unknown kernel {kernel!r}. Choose from: 'matern', 'rbf'.")
        if kernel == "matern" and nu <= 0:
            raise ValueError(f"jno.noise.grf: nu must be > 0 for the matern kernel, got {nu}.")

        m, nd = int(modes), int(ndim)
        ls, var, knl, smooth = float(length_scale), float(variance), kernel, float(nu)
        # A per-node id folded into the step key, so two grf nodes in one expression are independent
        # realisations rather than the same field twice -- what `Noise` gets from its `_noise_id`.
        node_id = _next_op_id()

        def _grf(*xs, key):
            pts = jnp.concatenate([jnp.reshape(x, (x.shape[0], -1)) for x in xs], axis=-1)
            n, d = pts.shape
            if key is None:  # crux.eval() without a key -> deterministic zeros, as Noise does
                return jnp.zeros((n, nd), dtype=pts.dtype)
            k_w, k_g, k_ab = jax.random.split(jax.random.fold_in(key, node_id), 3)
            w = jax.random.normal(k_w, (d, m), dtype=pts.dtype) / ls
            if knl == "matern":
                # Gaussian scale mixture: g ~ Gamma(nu, 1/nu) has mean 1, so rbf is the nu -> inf limit.
                g = jax.random.gamma(k_g, smooth, shape=(m,), dtype=pts.dtype) / smooth
                w = w / jnp.sqrt(jnp.maximum(g, jnp.finfo(pts.dtype).tiny))
            phase = pts @ w  # (n, m)
            ab = jax.random.normal(k_ab, (2, m, nd), dtype=pts.dtype)
            f = jnp.einsum("nm,mk->nk", jnp.cos(phase), ab[0]) + jnp.einsum("nm,mk->nk", jnp.sin(phase), ab[1])
            return jnp.sqrt(var / m) * f

        return FunctionCall(_grf, list(coords), name="grf")

    def laplace(self, std: float = 1.0, ndim: int = None, *, like=None) -> Noise:
        """Zero-mean Laplace noise with the given standard deviation.

        Parameters
        ----------
        std : float or callable
            Standard deviation (scale = std / sqrt(2)): a number, or a schedule ``lambda t: ...`` (a callable of the march time, only inside ``Model.flow(..., noise=...)``).
        ndim : int
            Number of output dimensions. Default ``1``. Mutually exclusive with ``like``.
        like : Placeholder, optional
            Draw with the shape of another traced expression instead of ``(N, ndim)``. See
            :meth:`gaussian` for the full explanation and an example.
        """
        shape_kw, like, shape_explicit = _resolve_shape_args("laplace", ndim, like)
        return Noise("laplace", like=like, shape_explicit=shape_explicit, std=_param("std", "laplace", std), **shape_kw)


noise = _NoiseNamespace()
