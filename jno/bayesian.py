"""Adapters that wire blackjax MCMC / VI kernels into jno's training loop.

Most of this module is internal — users compose `blackjax` kernels
directly and attach them per parameter via :meth:`jno.trace.Model.bayesian`
or :meth:`jno.trace.Model.vi`.  A small public surface lives at
``jno.bayesian.*``: convergence diagnostics (:func:`rhat`, :func:`ess`)
and the *logdensity-aware initializer* protocol — currently
:class:`PathfinderInitializer` (:func:`pathfinder`),
:class:`LaplaceInitializer` (:func:`laplace`), and
:class:`SVGDInitializer` (:func:`svgd`).  The training loop in
:mod:`jno.core` dispatches each model's per-step update either through
:mod:`optax` or through the blackjax kernel configured here.

Supported kernel families (duck-typed via the first argument name of the
factory's ``.differentiable`` callable):

* **Full-data kernels** — first argument ``logdensity_fn``.  Covers
  :func:`blackjax.nuts` (Hoffman & Gelman 2014), :func:`blackjax.hmc`,
  :func:`blackjax.mala`, and any other kernel that takes a log-density on
  the full dataset.  Step signature: ``kernel.step(rng_key, state)`` →
  ``(new_state, info)``.

* **SG-MCMC kernels** — first argument ``grad_estimator``.  Covers
  :func:`blackjax.sgld` (Welling & Teh 2011) and :func:`blackjax.sghmc`.
  Step signature: ``kernel.step(rng_key, state, minibatch, step_size)`` →
  ``new_state``.

References:
    Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler.
        Journal of Machine Learning Research, 15(1), 1593-1623.
    Welling, M., & Teh, Y. W. (2011). Bayesian Learning via Stochastic
        Gradient Langevin Dynamics. In ICML 2011, pp. 681-688.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

_FULL_FIRST_ARG = "logdensity_fn"
_GRAD_FIRST_ARG = "grad_estimator"
_VI_SECOND_ARG = "optimizer"  # blackjax.meanfield_vi has (logdensity_fn, optimizer, ...)

# Per-kernel-family diagnostic fields pulled from the blackjax info NamedTuple.
# Detected by factory __name__ in :func:`_detect_diagnostic_fields`.  Empty
# tuple means "no per-step info to surface" (SG-MCMC, VI — those report
# convergence via ``total_loss`` / ELBO).  The dtype tag drives both the
# diagnostic-buffer allocation and the post-solve aggregation in core.py.
_INFO_SCHEMA: dict[str, tuple[tuple[str, str], ...]] = {
    # NUTS / HMC: divergences are the primary signal that the integrator
    # blew up; acceptance_rate < 0.6 typically means step_size is too
    # big; energy lets users diagnose energy-conservation issues.
    "nuts": (("is_divergent", "bool"), ("acceptance_rate", "float"), ("energy", "float")),
    "hmc": (("is_divergent", "bool"), ("acceptance_rate", "float"), ("energy", "float")),
    # MALA has Metropolis-Hastings acceptance but no Hamiltonian integrator
    # (so no is_divergent / energy concept).
    "mala": (("acceptance_rate", "float"),),
}


def default_gaussian_prior(position, *, sigma: float = 10.0):
    """Wide isotropic Gaussian log-prior over a pytree position.

    ``log p(θ) = -‖θ‖² / (2σ²)``, summed over every inexact-array leaf.

    This is the internal fallback when ``.bayesian(prior=None)``.  Users
    composing custom priors should prefer the named factories in
    :data:`priors` (e.g. ``jno.bayesian.priors.gaussian(sigma=5.0)``)
    — they share the same logdensity-over-pytree contract but document
    their arguments and validate inputs at construction time.
    """
    leaves = jax.tree_util.tree_leaves(position)
    sq = jnp.array(0.0)
    for leaf in leaves:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            sq = sq + jnp.sum(jnp.asarray(leaf) ** 2)
    return -0.5 * sq / (sigma * sigma)


# ---------------------------------------------------------------------------
# Prior namespace — jno.bayesian.priors.{gaussian, laplace, student_t,
# layerwise_gaussian}.  Each factory returns a callable
# ``pytree -> scalar log p(theta)`` matching the contract Model.bayesian /
# Model.vi expects on ``prior=``.
#
# Masked-prior tree scoping note: when configured via ``.mask(M).bayesian()``
# / ``.mask(M).vi()`` the prior factory's callable receives the *masked
# subset* of the position (the kernel state) — not the full model pytree.
# Custom priors written by users should be aware that ``p`` is whatever
# subset the kernel sees: full pytree for global ``.bayesian()``, masked
# subset for ``.mask(M).bayesian()``.  Built-in factories below operate
# leaf-by-leaf via ``jax.tree_util.tree_leaves`` so they handle both cases
# identically.
# ---------------------------------------------------------------------------


def _floating_leaves(position):
    """Yield only the floating-point inexact-array leaves of ``position``.

    Integer bookkeeping leaves (e.g. masks, indices) are skipped so they
    don't accidentally contribute to ``‖θ‖²``.
    """
    for leaf in jax.tree_util.tree_leaves(position):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            yield leaf


def _leaf_fan_in(leaf) -> int:
    """Fan-in of a weight tensor, following the standard ``(out, in, ...)``
    layout used by foundax / equinox / flax / jax.nn.initializers.

    For ``ndim >= 2``: ``prod(shape[1:])`` — covers Linear ``(out, in)`` →
    ``in`` and Conv ``(out, in, kH, kW)`` → ``in * kH * kW``.  For
    ``ndim < 2`` (biases / scalars) the concept doesn't apply; the
    caller falls back to the default sigma.
    """
    if leaf.ndim < 2:
        return 1
    fi = 1
    for d in leaf.shape[1:]:
        fi *= int(d)
    return fi


def _gaussian_prior_fn(sigma: float, fan_in_aware: bool):
    """Build the closure for :func:`priors.gaussian`."""
    if sigma <= 0.0:
        raise ValueError(f"gaussian prior: sigma must be > 0, got {sigma!r}.")
    base_inv_s2 = 1.0 / (sigma * sigma)

    def _prior(position):
        sq = jnp.array(0.0)
        for leaf in _floating_leaves(position):
            arr = jnp.asarray(leaf)
            if fan_in_aware and arr.ndim >= 2:
                inv_s2 = float(_leaf_fan_in(arr)) * base_inv_s2
            else:
                inv_s2 = base_inv_s2
            sq = sq + jnp.sum(arr * arr) * inv_s2
        return -0.5 * sq

    return _prior


def _laplace_prior_fn(scale: float):
    """Build the closure for :func:`priors.laplace`."""
    if scale <= 0.0:
        raise ValueError(f"laplace prior: scale must be > 0, got {scale!r}.")
    inv_scale = 1.0 / scale

    def _prior(position):
        s = jnp.array(0.0)
        for leaf in _floating_leaves(position):
            s = s + jnp.sum(jnp.abs(jnp.asarray(leaf)))
        return -s * inv_scale

    return _prior


def _student_t_prior_fn(df: float, scale: float):
    """Build the closure for :func:`priors.student_t`."""
    if df <= 2.0:
        raise ValueError(f"student_t prior: df must be > 2 for finite variance, got {df!r}.")
    if scale <= 0.0:
        raise ValueError(f"student_t prior: scale must be > 0, got {scale!r}.")
    inv_scale2_df = 1.0 / (scale * scale * df)
    half_df_plus_one = 0.5 * (df + 1.0)

    def _prior(position):
        s = jnp.array(0.0)
        for leaf in _floating_leaves(position):
            arr = jnp.asarray(leaf)
            s = s + jnp.sum(jnp.log1p(arr * arr * inv_scale2_df))
        return -half_df_plus_one * s

    return _prior


def _layerwise_gaussian_prior_fn(base_sigma: float, default_sigma: float, fan_in_aware: bool):
    """Build the closure for :func:`priors.layerwise_gaussian`."""
    if base_sigma <= 0.0 or default_sigma <= 0.0:
        raise ValueError(
            f"layerwise_gaussian prior: base_sigma and default_sigma must be > 0, "
            f"got base_sigma={base_sigma!r}, default_sigma={default_sigma!r}."
        )
    base_inv_s2 = 1.0 / (base_sigma * base_sigma)
    default_inv_s2 = 1.0 / (default_sigma * default_sigma)

    def _prior(position):
        sq = jnp.array(0.0)
        for leaf in _floating_leaves(position):
            arr = jnp.asarray(leaf)
            if arr.ndim >= 2:
                if fan_in_aware:
                    inv_s2 = float(_leaf_fan_in(arr)) * base_inv_s2
                else:
                    inv_s2 = base_inv_s2
            else:
                # Bias / scalar — use the looser default sigma.
                inv_s2 = default_inv_s2
            sq = sq + jnp.sum(arr * arr) * inv_s2
        return -0.5 * sq

    return _prior


class _PriorsNamespace:
    """Namespace object exposed as ``jno.bayesian.priors``.

    Each factory returns a callable ``pytree -> scalar log p(theta)``
    suitable for the ``prior=`` argument on :meth:`Model.bayesian` and
    :meth:`Model.vi`.  All four operate leaf-by-leaf over the floating
    inexact-array leaves of the pytree, so they compose transparently
    with ``.mask(M).bayesian()`` (the kernel sees only the masked
    subset; the prior closure scans whatever subset it's handed).

    References
    ----------
    Sun, Y., Song, Z., Hewitt, A., & Kingma, D. P. (2019).
        *Functional Variational Bayesian Neural Networks.* ICLR 2019
        (layer-wise priors matching He / Xavier scales).

    Wenzel, F., Roth, K., Veeling, B., Świątkowski, J., Tran, L.,
        Mandt, S., … Nowozin, S. (2020).  *How Good is the Bayes
        Posterior in Deep Neural Networks Really?*  ICML 2020.
        (Cold-posterior effect with N(0, 1/fan_in) priors.)
    """

    @staticmethod
    def gaussian(sigma: float = 10.0, *, fan_in_aware: bool = False):
        """Isotropic Gaussian log-prior — ``log p = -‖θ‖² / (2σ²)``.

        Parameters
        ----------
        sigma : float, default ``10.0``
            Standard deviation of the prior.  ``sigma=10`` is wide
            enough to be "effectively flat" at typical parameter
            scales; pass smaller values (``1.0``, ``0.1``) for stronger
            shrinkage toward zero.
        fan_in_aware : bool, default ``False``
            When ``True``, scale ``σ`` per-leaf by ``1/sqrt(fan_in)``
            — matches He / Xavier initialisation conventions.  Only
            applies to weight tensors (``ndim >= 2``); biases and
            scalars use the base ``sigma`` unchanged.

        Returns
        -------
        callable
            ``pytree -> scalar log p`` for use as ``prior=`` on
            :meth:`Model.bayesian` / :meth:`Model.vi`.
        """
        return _gaussian_prior_fn(float(sigma), bool(fan_in_aware))

    @staticmethod
    def laplace(scale: float = 1.0):
        """Laplace (L1) log-prior — ``log p = -‖θ‖₁ / scale``.

        Sparse-friendly: the unbounded gradient at zero encourages
        many components to shrink to zero exactly (under MAP) or
        cluster near zero (under MCMC).  Useful for inverse problems
        with expected-sparse coefficients.
        """
        return _laplace_prior_fn(float(scale))

    @staticmethod
    def student_t(df: float = 4.0, scale: float = 1.0):
        """Student-t log-prior — heavy-tailed alternative to Gaussian.

        ``log p = -(df+1)/2 * Σ log(1 + (θ/scale)²/df)`` (additive
        constants dropped).  Lower ``df`` → heavier tails; ``df → ∞``
        recovers the Gaussian.

        Common BNN choices: ``df=3`` or ``df=4`` — heavy enough to
        allow large outliers, ``df > 2`` keeps the variance finite.
        Often used as a practical substitute for the horseshoe prior
        on individual weights when the full hierarchical horseshoe is
        too expensive (the horseshoe needs auxiliary scale variables
        that don't fit jno's pure ``logdensity_fn`` interface).

        Parameters
        ----------
        df : float, default ``4.0``
            Degrees of freedom.  Must be ``> 2`` for finite variance.
        scale : float, default ``1.0``
            Scale parameter (analogue of σ for the Gaussian).
        """
        return _student_t_prior_fn(float(df), float(scale))

    @staticmethod
    def layerwise_gaussian(
        *,
        base_sigma: float = 1.0,
        default_sigma: float = 1.0,
        fan_in_aware: bool = True,
    ):
        """Per-leaf Gaussian with fan-in-aware scaling — the standard
        BNN-PINN prior.

        For each inexact-array leaf:

        * **ndim >= 2** (weight tensors): σ = ``base_sigma / sqrt(fan_in)``
          when ``fan_in_aware=True``, else σ = ``base_sigma``.
        * **ndim < 2** (biases / scalars): σ = ``default_sigma``.

        Default ``base_sigma=1.0`` + ``fan_in_aware=True`` reproduces
        the He-style ``N(0, 1/fan_in)`` prior used in Wenzel et al.
        2020 and Sun et al. 2019.

        Parameters
        ----------
        base_sigma : float, default ``1.0``
            Base sigma for weight tensors; scaled per-layer when
            ``fan_in_aware=True``.
        default_sigma : float, default ``1.0``
            Sigma for biases and scalars (no fan-in concept).
        fan_in_aware : bool, default ``True``
            Apply ``1/sqrt(fan_in)`` scaling on weight tensors.
        """
        return _layerwise_gaussian_prior_fn(float(base_sigma), float(default_sigma), bool(fan_in_aware))


priors = _PriorsNamespace()


def _detect_kind(factory) -> str:
    """Return ``'full'``, ``'grad_estimator'``, or ``'vi'`` for a blackjax
    kernel / VI factory.

    Inspects ``factory.differentiable`` (the wrapped callable on blackjax's
    ``GenerateSamplingAPI``) or the factory itself, and looks at the first
    parameter name and (for VI) the second.  The dispatch is:

    * first arg ``logdensity_fn``, second arg ``optimizer`` → ``"vi"``
      (e.g. :func:`blackjax.meanfield_vi`).
    * first arg ``logdensity_fn``, no special second arg → ``"full"``
      (NUTS / HMC / MALA).
    * first arg ``grad_estimator`` → ``"grad_estimator"`` (SGLD / SGHMC).
    """
    target = getattr(factory, "differentiable", factory)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not inspect signature of blackjax kernel factory {factory!r}: {exc}") from exc

    params = list(sig.parameters)
    if not params:
        raise ValueError(
            f"blackjax kernel factory {factory!r} has no positional arguments — "
            "expected logdensity_fn or grad_estimator as the first parameter."
        )

    first = params[0]
    second = params[1] if len(params) >= 2 else None
    if first == _FULL_FIRST_ARG and second == _VI_SECOND_ARG:
        return "vi"
    if first == _FULL_FIRST_ARG:
        return "full"
    if first == _GRAD_FIRST_ARG:
        return "grad_estimator"
    raise ValueError(
        f"Unrecognised blackjax kernel factory {factory!r}: first argument "
        f"is {first!r}, expected one of "
        f"{(_FULL_FIRST_ARG, _GRAD_FIRST_ARG)} (MCMC) or "
        f"({_FULL_FIRST_ARG}, {_VI_SECOND_ARG}) (VI)."
    )


def _detect_diagnostic_fields(factory, kind: str) -> tuple[tuple[str, str], ...]:
    """Return the per-step diagnostic fields jno should pull off the kernel
    info NamedTuple, as ``(name, dtype_tag)`` pairs.

    Detected by ``factory.__name__`` — falls back to ``()`` for SG-MCMC
    (``grad_estimator``) and VI (no per-step diagnostics surfaced; their
    convergence is read off the ELBO / total_loss curve).  The empty
    tuple is *not* a silent discard — it means "this kernel has no
    blackjax info object to drop in the first place", and that case
    is logged at handle creation so users know what's tracked.
    """
    if kind in ("grad_estimator", "vi"):
        return ()
    # blackjax's MCMC entry points are NamedTuples (``GenerateSamplingAPI``)
    # without a ``__name__`` — inspect the underlying ``differentiable`` /
    # ``build_kernel`` callable's qualified module path, which uniquely
    # identifies the algorithm (``blackjax.mcmc.nuts`` /
    # ``blackjax.mcmc.hmc`` / ``blackjax.mcmc.mala`` / …).  Fall back to
    # ``__name__`` for user-supplied factories.
    target = getattr(factory, "build_kernel", None) or getattr(factory, "differentiable", None) or factory
    name_hint = (
        getattr(target, "__module__", "")
        + " "
        + getattr(target, "__qualname__", "")
        + " "
        + getattr(factory, "__name__", "")
    ).lower()
    for key, fields in _INFO_SCHEMA.items():
        if key in name_hint:
            return fields
    # Unrecognised full-data kernel.  Best-effort: capture any field the
    # info object happens to expose by trying common names; jno logs the
    # detected set at handle creation so users see what's tracked.
    return (("acceptance_rate", "float"),)


def _extract_info(info, fields: tuple[tuple[str, str], ...]) -> dict[str, jnp.ndarray]:
    """Pull each named field off the blackjax info NamedTuple into a flat
    dict of jnp arrays.

    Missing fields fall back to NaN (float) or False (bool) sentinels —
    flagged at post-solve aggregation so the user sees that the kernel
    didn't supply that piece of information, not a silent zero.
    """
    out: dict[str, jnp.ndarray] = {}
    for name, dtype_tag in fields:
        v = getattr(info, name, None)
        if v is None:
            if dtype_tag == "bool":
                out[name] = jnp.asarray(False, dtype=jnp.bool_)
            else:
                out[name] = jnp.asarray(jnp.nan, dtype=jnp.float32)
        else:
            if dtype_tag == "bool":
                out[name] = jnp.asarray(v, dtype=jnp.bool_)
            else:
                out[name] = jnp.asarray(v, dtype=jnp.float32)
    return out


def _empty_info_like(fields: tuple[tuple[str, str], ...]) -> dict[str, jnp.ndarray]:
    """Construct a (scalar) all-NaN/False info dict matching ``fields``.

    Used to fill the slot for kinds without a kernel info object
    (SG-MCMC / VI) so per-handle dict shapes stay homogeneous across
    JIT boundaries.  In practice these handles have ``fields == ()`` so
    the returned dict is empty and contributes nothing.
    """
    out: dict[str, jnp.ndarray] = {}
    for name, dtype_tag in fields:
        if dtype_tag == "bool":
            out[name] = jnp.asarray(False, dtype=jnp.bool_)
        else:
            out[name] = jnp.asarray(jnp.nan, dtype=jnp.float32)
    return out


class _KernelHandle:
    """Per-model handle stashed alongside the optax chain in jno's solve loop.

    Holds the factory, dispatch kind, prior, default step size (for SG-MCMC),
    and the remaining kwargs (passed to the factory at step time).  The
    actual blackjax kernel is rebuilt every step from a logdensity_fn /
    grad_estimator closure over the live ``loss_fn`` + ``context``.
    """

    __slots__ = (
        "factory",
        "kind",
        "prior_fn",
        "step_size",
        "extra_kwargs",
        "warmup",
        "keep",
        "thin",
        "adapt",
        "num_chains",
        "init_jitter",
        # VI-only fields (None / 0 for MCMC kinds).
        "vi_optimizer",
        "vi_num_samples",
        "vi_posterior_draws",
        # VI init overrides (None for non-VI handles).  ``init_log_std``
        # sets ``state.rho`` everywhere (-3.0 → σ ≈ 0.05).
        # ``init_mu_at_position`` controls whether ``state.mu`` starts
        # at the user-supplied position (True, jno default) or at
        # zeros (False, blackjax default).
        "vi_init_log_std",
        "vi_init_mu_at_position",
        # Phase 11: ``.mask(M).bayesian()`` / ``.mask(M).vi()``.  ``None``
        # means "operate on the full position" (the default for global
        # configurators).  When set, the kernel sees only the masked
        # subset of the position; the unmasked portion is held constant
        # at the model's current ``trainable[lid]`` snapshot and
        # reassembled inside the closure before each logdensity eval.
        "param_mask",
        # Per-step diagnostic fields pulled off the blackjax info
        # NamedTuple — populated from :func:`_detect_diagnostic_fields`
        # at handle creation.  Empty tuple for SG-MCMC / VI.
        "diagnostic_fields",
        # Likelihood scale: multiplier on the negative-log-likelihood
        # term in the logdensity closure.  Default 1.0; users tuning a
        # Gaussian-noise problem typically pass ``N_obs`` (so that
        # ``residual.mse`` — which returns a *mean* — matches the
        # canonical *sum* over data points).  Without this VI is
        # silently driven by the prior alone on multi-thousand-point
        # PINN losses.
        "likelihood_scale",
    )

    def __init__(
        self,
        factory: Callable,
        kind: str,
        prior_fn: Callable,
        step_size: float | None,
        extra_kwargs: dict,
        warmup: int,
        keep: int,
        thin: int,
        adapt: bool = True,
        num_chains: int = 1,
        init_jitter: float = 0.0,
        vi_optimizer=None,
        vi_num_samples: int = 8,
        vi_posterior_draws: int = 500,
        param_mask=None,
        likelihood_scale: float = 1.0,
        vi_init_log_std: float = -3.0,
        vi_init_mu_at_position: bool = True,
    ):
        self.factory = factory
        self.kind = kind
        self.prior_fn = prior_fn
        self.step_size = step_size
        self.extra_kwargs = extra_kwargs
        self.warmup = warmup
        self.keep = keep
        self.thin = thin
        self.adapt = adapt
        self.num_chains = int(num_chains)
        self.init_jitter = float(init_jitter)
        self.vi_optimizer = vi_optimizer
        self.vi_num_samples = int(vi_num_samples)
        self.vi_posterior_draws = int(vi_posterior_draws)
        self.param_mask = param_mask
        self.diagnostic_fields = _detect_diagnostic_fields(factory, kind)
        self.likelihood_scale = float(likelihood_scale)
        self.vi_init_log_std = float(vi_init_log_std)
        self.vi_init_mu_at_position = bool(vi_init_mu_at_position)


def build_vi_handle(cfg: dict) -> _KernelHandle:
    """Construct a VI handle from a ``_vi_cfg`` dict.

    ``cfg`` is the value stored on ``Model._vi_cfg`` by
    :meth:`jno.trace.Model.vi`.  Keys: ``factory`` (e.g.
    :func:`blackjax.meanfield_vi`), ``optimizer`` (an optax
    ``GradientTransformation``), ``prior``, ``num_samples``,
    ``posterior_draws``, plus user-supplied factory kwargs.

    References
    ----------
    Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013).
        *Stochastic Variational Inference.* JMLR 14(1), 1303-1347.

    Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., & Blei, D. M.
        (2017). *Automatic Differentiation Variational Inference.*
        JMLR 18(1), 430-474.
    """
    factory = cfg["factory"]
    kind = _detect_kind(factory)
    if kind != "vi":
        raise ValueError(f"build_vi_handle expects a VI factory (e.g. blackjax.meanfield_vi); got a {kind!r} factory.")
    prior_fn = cfg.get("prior") or default_gaussian_prior
    optimizer = cfg.get("optimizer")
    if optimizer is None:
        raise ValueError("jno .vi(..., optimizer=...) is required (an optax GradientTransformation).")
    num_samples = int(cfg.get("num_samples", 8))
    posterior_draws = int(cfg.get("posterior_draws", 500))
    if num_samples < 1 or posterior_draws < 1:
        raise ValueError("num_samples>=1, posterior_draws>=1 are required.")
    likelihood_scale = float(cfg.get("likelihood_scale", 1.0))
    if likelihood_scale <= 0.0:
        raise ValueError(f"likelihood_scale must be positive, got {likelihood_scale!r}.")
    init_log_std = float(cfg.get("init_log_std", -3.0))
    init_mu_at_position = bool(cfg.get("init_mu_at_position", True))
    return _KernelHandle(
        factory=factory,
        kind="vi",
        prior_fn=prior_fn,
        step_size=None,
        extra_kwargs=dict(cfg.get("factory_kwargs", {})),
        warmup=0,
        keep=0,
        thin=1,
        adapt=False,
        num_chains=1,
        init_jitter=0.0,
        vi_optimizer=optimizer,
        vi_num_samples=num_samples,
        vi_posterior_draws=posterior_draws,
        likelihood_scale=likelihood_scale,
        vi_init_log_std=init_log_std,
        vi_init_mu_at_position=init_mu_at_position,
    )


def build_kernel_handle(cfg: dict) -> _KernelHandle:
    """Construct an internal handle from a ``_bayesian_cfg`` dict.

    ``cfg`` is the value stored on ``Model._bayesian_cfg`` by
    :meth:`jno.trace.Model.bayesian`.  Keys: ``factory``, ``prior``,
    ``warmup``, ``keep``, ``thin``, plus user-supplied kernel kwargs
    (e.g. ``step_size``, ``inverse_mass_matrix``).
    """
    factory = cfg["factory"]
    kind = _detect_kind(factory)
    if kind == "vi":
        raise ValueError(
            "build_kernel_handle: got a VI factory; use Model.vi(...) (not "
            ".bayesian(...)) to configure variational inference."
        )
    prior_fn = cfg.get("prior") or default_gaussian_prior

    user_kwargs = dict(cfg.get("kernel_kwargs", {}))
    step_size = user_kwargs.pop("step_size", None)

    warmup = int(cfg.get("warmup", 500))
    keep = int(cfg.get("keep", 1000))
    thin = int(cfg.get("thin", 1))
    adapt = bool(cfg.get("adapt", True))
    if step_size is None:
        # When adapt=True on an HMC-family kernel,
        # ``blackjax.window_adaptation`` overrides ``step_size`` anyway
        # — make it optional in that case so the common path is just
        # ``model.bayesian(blackjax.nuts, warmup=500, keep=1000)``.
        # Detect HMC family by checking the factory signature for
        # ``inverse_mass_matrix`` (same gate ``adapt_is_applicable`` uses).
        _target = getattr(factory, "differentiable", factory)
        try:
            _sig = inspect.signature(_target)
            _is_hmc_family = "inverse_mass_matrix" in _sig.parameters
        except (TypeError, ValueError):
            _is_hmc_family = False
        if adapt and warmup > 0 and _is_hmc_family:
            # Sentinel default — window adaptation's first integrator
            # iteration uses this as its ``initial_step_size`` and
            # then refines.  1.0 is the value blackjax uses internally
            # when the user lets it choose; surfacing it here keeps
            # behaviour identical.
            step_size = 1.0
        else:
            raise ValueError(
                f"jno .bayesian({getattr(factory, '__name__', factory)!r}, ...) requires a "
                f"step_size= keyword argument when adapt=False or the kernel is not HMC-family "
                f"(adapt={adapt}, hmc_family={_is_hmc_family}, warmup={warmup})."
            )
    num_chains = int(cfg.get("num_chains", 1))
    init_jitter = float(cfg.get("init_jitter", 0.0))
    likelihood_scale = float(cfg.get("likelihood_scale", 1.0))
    if warmup < 0 or keep < 0 or thin < 1:
        raise ValueError("warmup>=0, keep>=0, thin>=1 are required.")
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1, got {num_chains}.")
    if likelihood_scale <= 0.0:
        raise ValueError(f"likelihood_scale must be positive, got {likelihood_scale!r}.")

    if kind == "full":
        extra = {**user_kwargs, "step_size": float(step_size)}
        return _KernelHandle(
            factory,
            kind,
            prior_fn,
            None,
            extra,
            warmup,
            keep,
            thin,
            adapt,
            num_chains,
            init_jitter,
            likelihood_scale=likelihood_scale,
        )
    return _KernelHandle(
        factory,
        kind,
        prior_fn,
        float(step_size),
        user_kwargs,
        warmup,
        keep,
        thin,
        adapt,
        num_chains,
        init_jitter,
        likelihood_scale=likelihood_scale,
    )


def _flat_inexact_size(position) -> int:
    """Total number of inexact (floating/complex) array entries in ``position``."""
    n = 0
    for leaf in jax.tree_util.tree_leaves(position):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            n += int(leaf.size)
    return n


def _maybe_inject_inverse_mass_matrix(handle: _KernelHandle, position) -> None:
    """If the kernel accepts ``inverse_mass_matrix`` and the user didn't pass
    one, default to identity of the inferred shape ``(D,)``.  A scalar
    ``inverse_mass_matrix=1.5`` is broadcast to the same shape.

    Mutates ``handle.extra_kwargs`` in place.  Safe to call on non-full
    kernels (MALA / SGLD / SGHMC) — they don't expose the kwarg and the
    function exits early.
    """
    target = getattr(handle.factory, "differentiable", handle.factory)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return
    if "inverse_mass_matrix" not in sig.parameters:
        return

    d = _flat_inexact_size(position)
    if d == 0:
        return  # nothing trainable; let blackjax raise downstream

    current = handle.extra_kwargs.get("inverse_mass_matrix", None)
    if current is None:
        handle.extra_kwargs["inverse_mass_matrix"] = jnp.ones(d)
        return

    arr = jnp.asarray(current)
    if arr.ndim == 0:
        handle.extra_kwargs["inverse_mass_matrix"] = jnp.full((d,), float(arr))


def _kernel_accepts_kwarg(handle: _KernelHandle, kwarg_name: str) -> bool:
    """``True`` iff the kernel factory accepts ``kwarg_name``.  Signature-based
    gate used to decide whether a logdensity-aware initializer's returned
    kwarg (e.g. ``inverse_mass_matrix``) can be merged into
    ``handle.extra_kwargs``.
    """
    target = getattr(handle.factory, "differentiable", handle.factory)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    return kwarg_name in sig.parameters


def merge_initializer_kwargs(handle: _KernelHandle, kwargs_update: dict) -> tuple[dict, list[str]]:
    """Merge an initializer-returned kwargs dict into ``handle.extra_kwargs``,
    silently dropping any key the kernel doesn't accept (e.g. an IMM update
    against a MALA kernel which has no ``inverse_mass_matrix`` parameter).

    Returns ``(new_extra_kwargs, dropped_keys)`` so the caller can log.
    """
    merged = dict(handle.extra_kwargs)
    dropped: list[str] = []
    for key, val in kwargs_update.items():
        if _kernel_accepts_kwarg(handle, key):
            merged[key] = val
        else:
            dropped.append(key)
    return merged, dropped


# ---------------------------------------------------------------------------
# Logdensity-aware initializer protocol
# ---------------------------------------------------------------------------
#
# ``Model.initialize(...)`` already accepts a path, a pytree, or a stateless
# ``(shape, dtype, key) -> array`` callable.  The protocol below is a fourth
# shape: an object with ``requires_logdensity = True`` whose ``__call__``
# runs *inside* solve() with access to the loss-derived log-density.
#
# Pathfinder is the first concrete implementation; future Laplace / SVGD /
# MAP-via-Adam initializers slot in as additional subclasses with no
# changes to ``Model.initialize`` or the solve()-side dispatch.


class _BayesianInitializer:
    """Base for ``.initialize()``-routable, log-density-aware warm-start
    strategies.

    Detected by :meth:`jno.trace.Model.initialize` via the class-level
    ``requires_logdensity = True`` marker.  Subclasses implement
    :meth:`__call__` with the contract below; jno handles the
    mask-wrap, multichain dispatch, and kernel-state reinitialisation
    once at the solve() site.

    The contract — one method, one return shape:

    ``__call__(rng_key, logdensity_fn, position, num_chains)
    -> (new_position, extra_kwargs_update)``

    * ``rng_key`` — master PRNG key for this initializer.
    * ``logdensity_fn`` — closes over the loss + prior.  Already
      mask-wrapped if the model was configured via
      ``.mask(M).bayesian(...)`` — subclasses see only the masked
      subset.
    * ``position`` — current pytree of (masked) parameter values to
      use as the optimisation start.
    * ``num_chains`` — K.  For K=1 return a leaf-shape position; for
      K>1 return a (K, *leaf)-leading pytree (one warm position per
      chain).

    Returns ``(new_position, extra_kwargs_update)``:

    * ``new_position`` — replaces ``trainable[lid]``; mask
      reassembly happens jno-side.
    * ``extra_kwargs_update`` — dict merged into the kernel handle's
      ``extra_kwargs``.  Typically
      ``{"inverse_mass_matrix": ...}``; empty dict if the initializer
      doesn't produce kernel-tunable output.  Keys the kernel
      doesn't accept are silently dropped (see
      :func:`merge_initializer_kwargs`).
    """

    requires_logdensity: ClassVar[bool] = True

    def __call__(
        self,
        rng_key,
        logdensity_fn: Callable,
        position,
        num_chains: int,
    ) -> tuple:
        raise NotImplementedError


def _diagonal_variance(samples) -> jnp.ndarray:
    """Per-dimension variance over a leading sample-axis pytree, flattened
    to ``(D,)`` matching jno's diagonal IMM convention.
    """
    leaves = jax.tree_util.tree_leaves(samples)
    flat = []
    for leaf in leaves:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
            flat.append(jnp.var(leaf, axis=0).reshape(-1))
    if not flat:
        return jnp.zeros((0,))
    return jnp.concatenate(flat)


@dataclass(frozen=True)
class PathfinderInitializer(_BayesianInitializer):
    """Warm-start a chain via :func:`blackjax.pathfinder`.

    Runs L-BFGS on the log-density; the inverse-Hessian factors along
    the optimisation path are turned into a normal approximation to the
    posterior.  From that fitted ``q`` we draw (a) the warm starting
    position (the MAP-ish ``state.position`` for ``K=1``; ``K`` distinct
    samples for ``K>1`` — proper over-dispersion at zero extra cost)
    and (b) per-dimension variance estimates that become the kernel's
    diagonal ``inverse_mass_matrix``.

    Reference
    ---------
    Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022).
    *Pathfinder: Parallel quasi-Newton variational inference.*
    Journal of Machine Learning Research, 23(306), 1-49.
    https://arxiv.org/abs/2108.03782
    """

    maxiter: int = 30
    num_samples: int = 200  # ELBO sample budget passed to pathfinder.approximate
    maxcor: int = 10  # L-BFGS history size
    imm_estimator_samples: int = 500  # # draws from fitted q used to estimate diag IMM
    lbfgs_kwargs: dict = field(default_factory=dict)  # ftol / gtol / maxls / ...

    def __call__(self, rng_key, logdensity_fn, position, num_chains):
        # Lazy import — keeps the module top tidy and avoids a JAX
        # initialisation cost at import time on unrelated code paths.
        import blackjax

        k_fit, k_imm, k_init = jax.random.split(rng_key, 3)
        pf_state, _info = blackjax.pathfinder.approximate(
            rng_key=k_fit,
            logdensity_fn=logdensity_fn,
            initial_position=position,
            num_samples=self.num_samples,
            maxiter=self.maxiter,
            maxcor=self.maxcor,
            **self.lbfgs_kwargs,
        )
        # Diagonal IMM from the empirical per-dim variance of M samples
        # drawn from the fitted Gaussian approximation to the posterior.
        # blackjax returns (samples, log_densities) — we want just the samples.
        imm_samples, _ = blackjax.pathfinder.sample(k_imm, pf_state, num_samples=self.imm_estimator_samples)
        imm_diag = _diagonal_variance(imm_samples)
        # Warm position: K=1 → MAP-ish (state.position); K>1 → K
        # i.i.d. samples from the fitted q (proper over-dispersion,
        # strictly better than additive jitter on a fixed init).
        if int(num_chains) == 1:
            warm_pos = pf_state.position
        else:
            warm_samples, _ = blackjax.pathfinder.sample(k_init, pf_state, num_samples=int(num_chains))
            warm_pos = warm_samples
        return warm_pos, {"inverse_mass_matrix": imm_diag}


def pathfinder(**kwargs) -> PathfinderInitializer:
    """Build a :class:`PathfinderInitializer` for use with
    :meth:`jno.trace.Model.initialize`.

    Usage::

        net.initialize(jno.bayesian.pathfinder(maxiter=30, num_samples=200))
        net.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)

    All kwargs are forwarded to the underlying
    ``blackjax.pathfinder.approximate`` call (``maxiter``, ``num_samples``,
    ``maxcor``, plus any ``lbfgs_kwargs``-eligible LBFGS knobs).
    """
    return PathfinderInitializer(**kwargs)


@dataclass(frozen=True)
class LaplaceInitializer(_BayesianInitializer):
    """Laplace approximation to the posterior — warm-start a chain from
    the MAP plus its local Gaussian approximation.

    Three-step algorithm:

    1. Find the maximum a-posteriori (MAP) point by optimising
       ``-log p(theta | data)`` with an optax optimiser (Adam by
       default).  The optimisation runs as a JIT-compiled
       ``jax.lax.scan`` over ``map_steps`` iterations.
    2. Compute the Hessian ``H = -∇²log p`` at the MAP.  Two strategies:

       * ``hessian_strategy="full"`` — full ``(D, D)`` Hessian via
         :func:`jax.hessian`.  Numerically clean but memory cost grows
         as ``D²``.  Right choice for small models / scalar PDE
         coefficients (D up to ~1000).
       * ``hessian_strategy="diagonal"`` (default) — only the diagonal
         of the Hessian is computed via D Hessian-vector probes.
         Memory cost is ``O(D)`` instead of ``O(D²)`` — required for
         BNN-scale problems.  Compute cost is comparable to "full" but
         peak memory is much smaller because no D×D matrix is ever
         materialised.

    3. Approximate the posterior as ``N(MAP, H⁻¹)``.  For ``num_chains=1``
       the warm position is the MAP; for ``num_chains>1`` we draw K
       i.i.d. samples from this Gaussian (proper over-dispersion).  The
       diagonal of ``H⁻¹`` is returned as the kernel's
       ``inverse_mass_matrix``.

    A small ``ridge`` (default ``1e-6``) is added to the diagonal of H
    before any inversion / Cholesky to guard against rank-deficient
    Hessians at non-converged MAP estimates.

    Trade-offs vs :class:`PathfinderInitializer`
    --------------------------------------------
    * **Pathfinder** uses L-BFGS — explores the optimisation path and
      produces a normal approximation from the path's inverse-Hessian
      factors.  Often robust on multi-modal or curved posteriors;
      cheaper than computing a Hessian for very large models.
    * **Laplace** uses gradient descent on ``-log p`` and the *exact*
      Hessian at the MAP.  More accurate locally if the posterior is
      well-approximated by a Gaussian; falls back to ridge-regularised
      sampling if H is ill-conditioned.

    References
    ----------
    MacKay, D. J. C. (1992).  *A Practical Bayesian Framework for
    Backpropagation Networks.*  §6 (Laplace approximation around the
    posterior mode).  Neural Computation, 4(3), 448-472.
    https://doi.org/10.1162/neco.1992.4.3.448

    Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M.,
    & Hennig, P. (2021).  *Laplace Redux — Effortless Bayesian Deep
    Learning.*  §2 (Laplace approximations for neural networks).
    Advances in Neural Information Processing Systems (NeurIPS).
    https://arxiv.org/abs/2106.14806

    Magnani, E., Krämer, N., Pförtner, M., & Hennig, P. (2024).
    *Linearization Turns Neural Operators into Function-Valued
    Gaussian Processes.*  §3 (linearised-Laplace for neural
    operators).  https://arxiv.org/abs/2406.05072
    """

    map_steps: int = 500
    map_optimizer: Any = None  # optax.GradientTransformation; defaults to optax.adam(1e-2)
    hessian_strategy: str = "diagonal"  # "diagonal" | "full"
    ridge: float = 1e-6

    def __call__(self, rng_key, logdensity_fn, position, num_chains):
        import optax  # lazy

        flat_pos, unflatten = jax.flatten_util.ravel_pytree(position)
        D = int(flat_pos.size)

        def _neg_flat_ld(v):
            return -logdensity_fn(unflatten(v))

        # ── Step 1: MAP via optax Adam (or user-supplied optimizer) ──
        opt = self.map_optimizer if self.map_optimizer is not None else optax.adam(1e-2)
        opt_state = opt.init(flat_pos)

        def _map_step(carry, _):
            v, state = carry
            g = jax.grad(_neg_flat_ld)(v)
            updates, new_state = opt.update(g, state, v)
            new_v = optax.apply_updates(v, updates)
            return (new_v, new_state), None

        (map_flat, _), _ = jax.lax.scan(_map_step, (flat_pos, opt_state), None, length=int(self.map_steps))

        if self.hessian_strategy == "full":
            # ── Step 2a: full (D, D) Hessian → Cholesky ──
            H = jax.hessian(_neg_flat_ld)(map_flat) + float(self.ridge) * jnp.eye(D)
            L = jnp.linalg.cholesky(H)
            # IMM = diag(H⁻¹).  Solve H X = I then take diag.
            H_inv = jnp.linalg.solve(L.T, jnp.linalg.solve(L, jnp.eye(D)))
            imm_diag = jnp.diag(H_inv)
        elif self.hessian_strategy == "diagonal":
            # ── Step 2b: diagonal of Hessian via D HVPs ──
            # diag(H)[i] = e_i · (H @ e_i), computed via jax.jvp on the gradient.
            _grad_fn = jax.grad(_neg_flat_ld)

            def _hvp(v):
                # H @ v via forward-mode over grad — O(D) flops per call.
                return jax.jvp(_grad_fn, (map_flat,), (v,))[1]

            eye = jnp.eye(D)
            diag_H = jax.vmap(lambda e: jnp.dot(e, _hvp(e)))(eye) + float(self.ridge)
            imm_diag = 1.0 / diag_H
            L = None  # signal "diagonal path" to the sampling branch
        else:
            raise ValueError(f"hessian_strategy must be 'diagonal' or 'full'; got {self.hessian_strategy!r}")

        # ── Step 3: warm positions ──
        K = int(num_chains)
        if K == 1:
            warm_pos = unflatten(map_flat)
        else:
            keys = jax.random.split(rng_key, K)
            if L is not None:
                # Full-Hessian path: sample from N(MAP, H⁻¹) via Cholesky.
                # If L is Cholesky of H, then  MAP + solve(L.T, z)  ~  N(MAP, H⁻¹).
                z = jax.vmap(lambda k: jax.random.normal(k, (D,)))(keys)
                samples_centered = jax.vmap(lambda zi: jnp.linalg.solve(L.T, zi))(z)
            else:
                # Diagonal path: independent per-dim N(0, imm_diag[i]).
                std = jnp.sqrt(imm_diag)
                z = jax.vmap(lambda k: jax.random.normal(k, (D,)))(keys)
                samples_centered = std[None, :] * z
            warm_samples = map_flat[None, :] + samples_centered
            warm_pos = jax.vmap(unflatten)(warm_samples)

        return warm_pos, {"inverse_mass_matrix": imm_diag}


def laplace(**kwargs) -> LaplaceInitializer:
    """Build a :class:`LaplaceInitializer` for use with
    :meth:`jno.trace.Model.initialize`.

    Usage::

        net.initialize(jno.bayesian.laplace(map_steps=500,
                                            hessian_strategy="diagonal"))
        net.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)

    Kwargs (see :class:`LaplaceInitializer` for full descriptions):

    * ``map_steps`` (default 500) — number of optimiser iterations to
      find the MAP.
    * ``map_optimizer`` (default ``optax.adam(1e-2)``) — any optax
      ``GradientTransformation``.
    * ``hessian_strategy`` (default ``"diagonal"``) — ``"diagonal"``
      computes diag(H) via D HVPs (O(D) memory), ``"full"`` materialises
      the (D, D) Hessian.
    * ``ridge`` (default ``1e-6``) — diagonal stabiliser added to H
      before inversion.

    Reference
    ---------
    MacKay 1992 §6; Daxberger et al. 2021 §2.  See
    :class:`LaplaceInitializer` for full citations.
    """
    return LaplaceInitializer(**kwargs)


@dataclass(frozen=True)
class SVGDInitializer(_BayesianInitializer):
    """Stein Variational Gradient Descent warm-start.

    Runs SVGD (Liu & Wang 2016) — a deterministic, particle-based
    variational inference algorithm — and uses the final particle
    cloud as the warm-start.  Each particle is dragged toward the
    posterior by a kernelised functional gradient that combines a
    repulsive RBF term (keeps particles diverse) with an attractive
    log-density gradient term (pulls each particle toward higher
    posterior density).  At convergence, the particles approximate
    the posterior distribution.

    Three-step algorithm:

    1. Initialise ``num_particles`` particles by perturbing the
       user-supplied position with Gaussian noise of std
       ``init_jitter``.  When ``init_jitter`` is left at its default
       (``None``), jno picks ``max(0.1 * std(position), 1e-3)`` so
       particles start one-tenth of a parameter-scale apart — a
       sensible "small but visible" spread that adapts to whatever
       initialisation the model uses.  Explicit positive floats are
       respected as absolute std values.  Default
       ``num_particles = max(num_chains, 32)`` so we always have at
       least 32 particles for the variance estimate, even when the
       caller only asked for 1 chain.
    2. Run ``num_iters`` SVGD steps using :func:`blackjax.svgd` with
       the supplied optax optimiser (Adam by default) and the
       default RBF kernel (overridable via ``kernel``).  The whole
       run is wrapped in :func:`jax.lax.scan` for fast XLA-side
       iteration.
    3. Return:

       * For ``num_chains == 1`` — the particle-cloud **mean** as
         the warm starting position (most stable summary).
       * For ``num_chains > 1`` — the first ``num_chains`` particles
         as K distinct warm starting positions.  The particle
         dynamics already provide proper over-dispersion; no extra
         jitter step is needed.

       The per-dimension **variance of the particle cloud** is
       returned as the diagonal ``inverse_mass_matrix`` (a cheap
       empirical-Bayes estimate of the posterior covariance).

    Trade-offs vs Pathfinder / Laplace
    ----------------------------------
    * **Pathfinder** approximates the posterior via L-BFGS factors —
      cheap and unimodal.
    * **Laplace** uses the exact Hessian at the MAP — accurate
      locally but ignores posterior modes beyond the MAP basin.
    * **SVGD** captures multi-modal structure when present: the
      repulsive RBF kernel pushes particles apart so multiple modes
      can be discovered with enough particles.  Cost grows with
      ``num_particles²`` per step (pairwise kernel interactions).

    Reference
    ---------
    Liu, Q., & Wang, D. (2016).  *Stein Variational Gradient Descent:
    A General Purpose Bayesian Inference Algorithm.*  §3 (the SVGD
    update rule).  Advances in Neural Information Processing Systems
    (NeurIPS), 29, 2378-2386.
    https://arxiv.org/abs/1608.04471
    """

    num_iters: int = 500
    num_particles: int | None = None  # default: max(num_chains, 32)
    optimizer: Any = None  # optax.GradientTransformation; defaults to optax.adam(1e-1)
    kernel: Callable | None = None  # blackjax kernel; defaults to blackjax's RBF
    # Std of Gaussian perturbation around the input position.  ``None``
    # → ``max(0.1 * std(position), 1e-3)`` — scale-aware default that
    # avoids the historical ``1.0`` which was 100× larger than Xavier
    # weights at scale 0.01.  Pass an explicit positive float to
    # override (must be > 0).
    init_jitter: float | None = None

    def __call__(self, rng_key, logdensity_fn, position, num_chains):
        import blackjax  # lazy
        import optax  # lazy

        flat_pos, unflatten = jax.flatten_util.ravel_pytree(position)
        D = int(flat_pos.size)
        K = int(num_chains)

        # Choose particle count: enough for stable variance + at
        # least K for chain inits.
        N = int(self.num_particles) if self.num_particles is not None else max(K, 32)
        if N < K:
            raise ValueError(f"SVGDInitializer: num_particles ({N}) must be >= num_chains ({K}).")

        # Flat gradient of log p
        def _flat_ld(v):
            return logdensity_fn(unflatten(v))

        _flat_ld_grad = jax.grad(_flat_ld)

        # Resolve init_jitter — ``None`` (default) picks a position-aware
        # scale: ten percent of the parameter std, floored at 1e-3 so
        # constant-init parameters still get a nonzero spread.  Explicit
        # positive floats are validated up-front and respected verbatim.
        if self.init_jitter is None:
            jitter = jnp.maximum(0.1 * jnp.std(flat_pos), 1e-3)
        else:
            jitter_f = float(self.init_jitter)
            if jitter_f <= 0.0:
                raise ValueError(f"SVGDInitializer: init_jitter must be > 0, got {jitter_f!r}.")
            jitter = jitter_f

        # Initial particles: Gaussian noise around the user's flat init.
        keys = jax.random.split(rng_key, N)
        init_particles = jax.vmap(lambda k: flat_pos + jitter * jax.random.normal(k, (D,)))(keys)

        opt = self.optimizer if self.optimizer is not None else optax.adam(1e-1)
        svgd_kwargs = {}
        if self.kernel is not None:
            svgd_kwargs["kernel"] = self.kernel
        svgd = blackjax.svgd(_flat_ld_grad, opt, **svgd_kwargs)
        state = svgd.init(init_particles)

        # Run num_iters SVGD steps via lax.scan (one XLA dispatch).
        def _step(state_in, _):
            return svgd.step(state_in), None

        state, _ = jax.lax.scan(_step, state, None, length=int(self.num_iters))

        particles = state.particles  # shape (N, D)

        # Diagonal IMM from particle variance (+ small ridge).
        imm_diag = jnp.var(particles, axis=0) + 1e-6

        # Warm positions.
        if K == 1:
            warm_flat = jnp.mean(particles, axis=0)
            warm_pos = unflatten(warm_flat)
        else:
            # First K particles → K chain inits.  Particles are
            # exchangeable so any K of them work; first K is
            # deterministic.
            warm_pos = jax.vmap(unflatten)(particles[:K])

        return warm_pos, {"inverse_mass_matrix": imm_diag}


def svgd(**kwargs) -> SVGDInitializer:
    """Build an :class:`SVGDInitializer` for use with
    :meth:`jno.trace.Model.initialize`.

    Usage::

        net.initialize(jno.bayesian.svgd(num_iters=500, num_particles=32))
        net.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)

    Kwargs (see :class:`SVGDInitializer` for full descriptions):

    * ``num_iters`` (default 500) — number of SVGD update steps.
    * ``num_particles`` (default ``max(num_chains, 32)``) — particle
      count; larger captures more modes but costs ``O(N²)`` per step.
    * ``optimizer`` (default ``optax.adam(1e-1)``) — any optax
      ``GradientTransformation``.
    * ``kernel`` (default blackjax's RBF) — any positive
      semi-definite kernel; signature
      ``(particles, kernel_parameters) -> (kxx, dxkxx)``.
    * ``init_jitter`` (default ``None``) — std of Gaussian noise
      perturbing the input position to seed the initial particle
      cloud.  ``None`` picks ``max(0.1 * std(position), 1e-3)`` so
      particles start one-tenth of a parameter-scale apart;
      explicit positive floats are respected as absolute std.

    Reference
    ---------
    Liu & Wang 2016 §3.  See :class:`SVGDInitializer` for full
    citation.
    """
    return SVGDInitializer(**kwargs)


def init_state_at_warm_positions(handle: _KernelHandle, warm_position_full):
    """Build the kernel state when a logdensity-aware initializer has
    already supplied warm starting position(s).

    Unlike :func:`init_state`, this does **not** replicate-with-jitter
    for ``K>1`` — the caller has provided ``K`` distinct positions (or 1
    for ``K=1``) and we use them verbatim.  ``warm_position_full`` is the
    *full* pytree (with the unmasked complement already reassembled for
    masked Bayesian groups); we narrow internally before
    ``kernel.init``.

    VI handles are not supported here — they have their own init path
    inside :func:`init_state` keyed on ``state.mu = position``.
    """
    if handle.kind == "vi":
        raise ValueError(
            "init_state_at_warm_positions does not apply to VI handles; "
            "use init_state directly with the warm position as input."
        )

    K = int(handle.num_chains)

    # Narrow to the masked subset if applicable.  For K>1 the leading
    # axis is the chain dim; ``eqx.filter`` is leaf-by-leaf so applies
    # uniformly whether or not a K axis is present.
    if handle.param_mask is not None:
        init_position = eqx.filter(warm_position_full, handle.param_mask)
    else:
        init_position = warm_position_full

    # Inject IMM if applicable (using a representative single-chain slice
    # for shape inference when K>1).
    rep = init_position if K == 1 else jax.tree_util.tree_map(lambda x: x[0], init_position)
    _maybe_inject_inverse_mass_matrix(handle, rep)

    factory = handle.factory
    if handle.kind == "full":
        # Placeholder logdensity — kernel.init only needs the position.
        kernel = factory(handle.prior_fn, **handle.extra_kwargs)
    else:

        def _dummy_grad(p, _mb):
            return jax.tree_util.tree_map(jnp.zeros_like, p)

        kernel = factory(_dummy_grad, **handle.extra_kwargs)

    if K == 1:
        return kernel.init(init_position)
    return jax.vmap(kernel.init)(init_position)


def adapt_is_applicable(handle: _KernelHandle) -> bool:
    """``True`` iff ``run_window_adaptation`` would do real work for this
    handle (HMC-family kernel, ``adapt=True``, ``warmup>0``).  Used to gate
    the cost of building an adapt-side ``loss_fn`` closure in
    :mod:`jno.core` when no Bayesian model actually needs it.
    """
    if not handle.adapt or handle.warmup <= 0 or handle.kind != "full":
        return False
    target = getattr(handle.factory, "differentiable", handle.factory)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return False
    return "inverse_mass_matrix" in sig.parameters


def run_window_adaptation(handle: _KernelHandle, position, logdensity_fn, rng_key):
    """Run ``blackjax.window_adaptation`` for ``handle.warmup`` steps and
    return the adapted ``(state, extra_kwargs)``.

    Returns ``None`` when adaptation does not apply: non-HMC-family kernels
    (whose factories don't accept ``inverse_mass_matrix``), ``handle.adapt``
    is False, or ``handle.warmup <= 0``.

    The caller is responsible for assigning the returned state into
    ``opt_states[k]``, replacing ``handle.extra_kwargs``, and setting
    ``handle.warmup = 0`` so the main loop does not double-skip samples.
    """
    if not handle.adapt or handle.warmup <= 0 or handle.kind != "full":
        return None
    target = getattr(handle.factory, "differentiable", handle.factory)
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        return None
    if "inverse_mass_matrix" not in sig.parameters:
        return None  # Not HMC family — window_adaptation does not support it.

    # Lazy import — blackjax is a hard dep, but keeps the module top tidy.
    import blackjax

    # Pull step_size out for window_adaptation's `initial_step_size`; drop
    # any user-supplied `inverse_mass_matrix` since adaptation computes one.
    kwargs = dict(handle.extra_kwargs)
    initial_step_size = float(kwargs.pop("step_size", 1.0))
    kwargs.pop("inverse_mass_matrix", None)

    adapt = blackjax.window_adaptation(
        handle.factory,
        logdensity_fn,
        initial_step_size=initial_step_size,
        **kwargs,  # forwards e.g. max_num_doublings to the kernel algorithm
    )
    result, _info = adapt.run(rng_key, position, num_steps=int(handle.warmup))

    adapted_kwargs = {
        **kwargs,
        "step_size": float(result.parameters["step_size"]),
        "inverse_mass_matrix": result.parameters["inverse_mass_matrix"],
    }
    return result.state, adapted_kwargs


def _replicate_with_jitter(position, num_chains: int, jitter: float, rng_key):
    """Broadcast ``position`` to a leading K-axis with optional per-chain jitter.

    With ``jitter == 0``, every chain starts from the exact same position
    (broadcast).  With ``jitter > 0``, per-chain Gaussian noise
    ``N(0, jitter²)`` is added to each inexact-array leaf (integer
    bookkeeping leaves are broadcast unchanged).
    """
    leaves, treedef = jax.tree_util.tree_flatten(position)
    K = int(num_chains)
    if jitter <= 0.0:
        rep_leaves = [jnp.broadcast_to(leaf[None, ...], (K, *leaf.shape)) for leaf in leaves]
        return jax.tree_util.tree_unflatten(treedef, rep_leaves)
    # Per-chain noise per inexact leaf with distinct PRNG splits.
    inexact_idxs = [
        i for i, leaf in enumerate(leaves) if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    keys = jax.random.split(rng_key, max(len(inexact_idxs), 1))
    out = []
    ki = 0
    for i, leaf in enumerate(leaves):
        base = jnp.broadcast_to(leaf[None, ...], (K, *leaf.shape))
        if i in inexact_idxs:
            noise = jitter * jax.random.normal(keys[ki], shape=(K, *leaf.shape), dtype=leaf.dtype)
            base = base + noise
            ki += 1
        out.append(base)
    return jax.tree_util.tree_unflatten(treedef, out)


def init_state(handle: _KernelHandle, position, rng_key=None):
    """Initialise the kernel state for ``position``, with a leading K-axis.

    For full-data kernels we build a one-off kernel with a trivial
    logdensity (the prior alone) just to call ``.init`` — the real
    logdensity is rebuilt per step.  For SG-MCMC, ``init`` only needs the
    position.

    The returned state has a leading axis of length ``handle.num_chains``
    (= 1 by default). ``rng_key`` is used to draw per-chain
    ``init_jitter`` noise (no-op if ``init_jitter == 0``); pass any fixed
    key if you don't need jitter.
    """
    # Phase 11 masked path — narrow the position to the masked subset
    # *before* any init.  The unmasked portion is held constant at the
    # current model snapshot (held by ``trainable[lid]`` in the
    # caller); we don't capture it here.  ``init_position`` is the
    # ``eqx.partition``-narrowed pytree blackjax sees from now on.
    if handle.param_mask is not None:
        init_position = eqx.filter(position, handle.param_mask)
    else:
        init_position = position

    # VI handles take a separate path — no K-axis, no inverse-mass-matrix
    # injection; just the meanfield_vi-style ``MFVIState(mu, rho, opt_state)``.
    if handle.kind == "vi":

        def _dummy_logdensity(p):
            return handle.prior_fn(p)

        vi_algo = handle.factory(
            _dummy_logdensity,
            handle.vi_optimizer,
            num_samples=handle.vi_num_samples,
            **handle.extra_kwargs,
        )
        state = vi_algo.init(init_position)
        # Two overrides on blackjax's defaults, exposed as user kwargs:
        # 1. ``state.mu`` defaults to zeros regardless of position.  For
        #    non-trivial models (e.g. an MLP with Xavier init) starting
        #    at mu=zeros makes the ELBO landscape flat and convergence
        #    painfully slow.  ``vi_init_mu_at_position=True`` (jno
        #    default) sets mu to the user-supplied initial weights —
        #    matches numpyro's autoguide; pass False to keep blackjax's
        #    zero start.
        # 2. ``state.rho`` defaults to large values (≈ exp(rho) ≈ 1,
        #    σ ≈ 1.0 per weight) — extremely noisy MC ELBO gradients
        #    on multi-layer MLPs.  ``vi_init_log_std=-3.0`` (jno default
        #    → σ ≈ 0.05) keeps the initial q tight; the optimiser then
        #    *grows* rho where the posterior is genuinely wide.  Pass
        #    e.g. ``vi_init_log_std=0.0`` (σ ≈ 1.0) to restore the
        #    blackjax default, or any other float for a custom width.
        if handle.vi_init_mu_at_position:
            new_mu = init_position
        else:
            new_mu = state.mu
        new_rho = jax.tree_util.tree_map(lambda x: jnp.full_like(x, handle.vi_init_log_std), state.rho)
        return state._replace(mu=new_mu, rho=new_rho)

    _maybe_inject_inverse_mass_matrix(handle, init_position)
    factory = handle.factory
    if handle.kind == "full":
        kernel = factory(handle.prior_fn, **handle.extra_kwargs)
    else:

        def _dummy_grad(p, _mb):
            return jax.tree_util.tree_map(jnp.zeros_like, p)

        kernel = factory(_dummy_grad, **handle.extra_kwargs)

    if handle.num_chains == 1:
        # Backward-compat: no K axis on the returned state.  The K axis
        # is reattached at buffer-flush time so user-facing
        # ``posterior_samples`` is always ``(K, N, *param)``.
        return kernel.init(init_position)

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    position_K = _replicate_with_jitter(init_position, handle.num_chains, handle.init_jitter, rng_key)
    return jax.vmap(kernel.init)(position_K)


def step(
    handle: _KernelHandle,
    rng_key,
    state,
    position,
    logdensity_factory: Callable,
    grad_estimator_factory: Callable,
    minibatch_ctx,
):
    """One kernel step, vmapped over ``handle.num_chains``.

    ``state`` carries a leading K-axis (set by :func:`init_state`); we
    split ``rng_key`` into K per-chain keys and vmap the per-chain
    transition.  Returns ``(new_state, new_position, info)`` — info is a
    dict of per-step diagnostic arrays (one entry per
    ``handle.diagnostic_fields`` entry, K-leading for K>1, scalar for
    K=1).  Empty dict for SG-MCMC / VI handles that don't surface a
    blackjax info object.

    The caller supplies *factories* rather than concrete callables so
    that each chain's logdensity can see *that* chain's positions for
    every other Bayesian model in the same solve (correct
    Metropolis-within-Gibbs semantics when multiple models are
    Bayesian).  The signatures are::

        logdensity_factory(p, k_idx)               -> scalar log p
        grad_estimator_factory(p, minibatch, k_idx) -> grad pytree

    For ``num_chains == 1`` both reduce to a no-op slice of the lone
    chain — no semantic change versus pre-multi-chain code.
    """
    factory = handle.factory
    K = handle.num_chains

    # ── Phase 11: ``.mask(M).bayesian()`` / ``.mask(M).vi()`` ──
    # When the handle has a ``param_mask``, the MCMC / VI state only
    # spans the masked subset of the position.  ``position`` (the
    # caller-supplied full snapshot) provides the unmasked complement;
    # we wrap the user-supplied factories so each call reassembles
    # the full pytree before logdensity / loss evaluation.  After the
    # kernel step, the new masked position is recombined with the
    # unmasked snapshot and returned so the caller can store the full
    # pytree in ``trainable[lid]`` without knowing about the mask.
    _mask = handle.param_mask
    if _mask is not None:
        _unmasked = eqx.filter(position, _mask, inverse=True)
        _orig_ld = logdensity_factory
        _orig_ge = grad_estimator_factory

        def _wrap_full(p_masked):
            return eqx.combine(p_masked, _unmasked)

        def logdensity_factory(p_masked, *args):  # noqa: F811 — rebind
            return _orig_ld(_wrap_full(p_masked), *args)

        def grad_estimator_factory(p_masked, mb, *args):  # noqa: F811
            grad_full = _orig_ge(_wrap_full(p_masked), mb, *args)
            return eqx.filter(grad_full, _mask)

        def _reassemble_single(p_masked):
            return _wrap_full(p_masked)
    else:
        _reassemble_single = None

    _fields = handle.diagnostic_fields

    if handle.kind == "vi":
        # Variational inference path — no K axis, no MCMC kernel.
        # Build the VI algorithm with the live logdensity_fn (closure
        # over the current ``trainable`` / ``context``) and run one
        # ELBO optimisation step.  ``core.py`` passes the K=1
        # ``(p) -> log p`` closure here; we ignore the SG-MCMC factory
        # since VI doesn't use minibatch gradient estimators.
        vi_algo = factory(
            logdensity_factory,
            handle.vi_optimizer,
            num_samples=handle.vi_num_samples,
            **handle.extra_kwargs,
        )
        new_state, _info = vi_algo.step(rng_key, state)
        # Return the variational *mean* as the position so the outer
        # ``trainable[lid]`` carries a representative point estimate.
        new_pos = new_state.mu
        if _reassemble_single is not None:
            new_pos = _reassemble_single(new_pos)
        # ``_fields`` is () for VI handles — info dict is empty.
        return new_state, new_pos, _empty_info_like(_fields)

    if K == 1:
        # Backward-compatible single-chain path: state has no K axis,
        # kernel built once outside, no jax.vmap.  Bit-identical to the
        # pre-multi-chain JIT trace / gradient / PRNG behaviour.
        # ``core.py`` passes ``(p) -> log p`` style closures here (not
        # factories), matching the OLD code form.
        if handle.kind == "full":
            kernel = factory(logdensity_factory, **handle.extra_kwargs)
            new_state, _info = kernel.step(rng_key, state)
            new_pos = new_state.position
            if _reassemble_single is not None:
                new_pos = _reassemble_single(new_pos)
            return new_state, new_pos, _extract_info(_info, _fields)
        kernel = factory(grad_estimator_factory, **handle.extra_kwargs)
        new_state = kernel.step(rng_key, state, minibatch_ctx, handle.step_size)
        new_pos = new_state
        if _reassemble_single is not None:
            new_pos = _reassemble_single(new_pos)
        # SG-MCMC has no info NamedTuple — ``_fields`` is () so the dict
        # is empty.
        return new_state, new_pos, _empty_info_like(_fields)

    # Multi-chain path: per-chain PRNG keys + vmap.  ``core.py`` passes
    # *factories* taking ``(p, k_idx)`` / ``(p, mb, k_idx)`` here.
    keys = jax.random.split(rng_key, K)
    chain_idx = jnp.arange(K)

    if handle.kind == "full":

        def _one(state_k, key_k, k_idx):
            ld = lambda p, _k=k_idx: logdensity_factory(p, _k)
            kernel_k = factory(ld, **handle.extra_kwargs)
            new_s, _info = kernel_k.step(key_k, state_k)
            new_p = new_s.position
            if _reassemble_single is not None:
                new_p = _reassemble_single(new_p)
            return new_s, new_p, _extract_info(_info, _fields)

        return jax.vmap(_one)(state, keys, chain_idx)

    step_size = handle.step_size

    def _one(state_k, key_k, k_idx):
        ge = lambda p, mb, _k=k_idx: grad_estimator_factory(p, mb, _k)
        kernel_k = factory(ge, **handle.extra_kwargs)
        new_s = kernel_k.step(key_k, state_k, minibatch_ctx, step_size)
        new_p = new_s
        if _reassemble_single is not None:
            new_p = _reassemble_single(new_p)
        return new_s, new_p, _empty_info_like(_fields)

    return jax.vmap(_one)(state, keys, chain_idx)


def vi_sample(handle: _KernelHandle, state, rng_key, num_samples: int):
    """Draw ``num_samples`` i.i.d. samples from a fitted VI distribution.

    Builds the VI algorithm with a dummy logdensity (unused by
    ``.sample``) and calls its sample method.  Returns a pytree mirroring
    the position structure with a leading axis of length ``num_samples``.

    Only valid for ``handle.kind == "vi"`` handles.
    """
    if handle.kind != "vi":
        raise ValueError(f"vi_sample expects a VI handle; got kind={handle.kind!r}.")

    def _dummy_logdensity(p):
        return handle.prior_fn(p)

    vi_algo = handle.factory(
        _dummy_logdensity,
        handle.vi_optimizer,
        num_samples=handle.vi_num_samples,
        **handle.extra_kwargs,
    )
    return vi_algo.sample(rng_key, state, num_samples)


def chain_params_for_eval(models, posterior_samples_by_lid):
    """Build ``(chain_part, static_part)`` for vmap over a posterior chain.

    ``chain_part`` contains the layer IDs that have posterior samples (each
    value is a stacked pytree with leading axis = number of samples).
    ``static_part`` contains the point-estimate models for non-Bayesian
    layers.  vmap over ``chain_part`` only — non-Bayesian models broadcast.
    """
    chain_part: dict[Any, Any] = {}
    static_part: dict[Any, Any] = {}
    for lid, model in models.items():
        if lid in posterior_samples_by_lid:
            chain_part[lid] = posterior_samples_by_lid[lid]
        else:
            static_part[lid] = model
    return chain_part, static_part


# ---------------------------------------------------------------------------
# Convergence diagnostics — R-hat and ESS
# ---------------------------------------------------------------------------


def rhat(chain, *, strategy: str = "auto") -> jnp.ndarray:
    """Rank-normalised, folded R-hat (Gelman & Rubin 1992, improved by
    Vehtari et al. 2021).

    ``chain`` is shaped ``(K, N, *param)`` — K chains, N draws each.
    Returns an array of shape ``*param`` with one R-hat per parameter
    component.  R-hat ≈ 1.0 indicates the chains are exploring the same
    distribution; values above ~1.05 suggest non-convergence.

    Parameters
    ----------
    chain : array
        ``(K, N, *param)`` posterior chain.
    strategy : {"auto", "multichain", "split"}, default "auto"
        How to estimate R-hat:

        * ``"auto"`` (default): split-R-hat when K == 1, multichain
          R-hat when K >= 2 — preserves the historical behaviour.
        * ``"multichain"``: require K >= 2; raise ``ValueError`` if not.
          Use when you've explicitly run multiple chains and want a
          loud failure if the layout doesn't carry them.
        * ``"split"``: split every chain in half and treat the halves
          as 2K independent chains.  Useful as a stationarity check
          on top of cross-chain mixing for K >= 2.

    Notes
    -----
    For K=1 the default ``"auto"`` strategy falls back to single-chain
    split-R-hat (still informative for stationarity but not as strong
    as a true multi-chain diagnostic).  Pass ``strategy="multichain"``
    to forbid this silent fallback.

    References
    ----------
    Gelman, A., & Rubin, D. B. (1992). *Inference from iterative
        simulation using multiple sequences.* Statistical Science 7(4),
        457-511.

    Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C.
        (2021). *Rank-Normalization, Folding, and Localization: An
        Improved R̂ for Assessing Convergence of MCMC.* Bayesian
        Analysis 16(2), 667-718.
    """
    if strategy not in ("auto", "multichain", "split"):
        raise ValueError(f"rhat: strategy must be one of 'auto', 'multichain', 'split'; got {strategy!r}.")
    chain = jnp.asarray(chain)
    if chain.ndim < 2:
        raise ValueError(f"rhat expects (K, N, *param) array; got shape {chain.shape}.")
    K, N = chain.shape[0], chain.shape[1]
    if strategy == "multichain" and K < 2:
        raise ValueError(
            f"rhat(strategy='multichain') requires K>=2; got K={K}. "
            f"Use the default strategy='auto' or strategy='split' for single-chain input."
        )
    if strategy == "split" or (strategy == "auto" and K == 1):
        # Split: treat each chain's two halves as independent chains
        # (Gelman et al. 2014, BDA3 §11.4).  For K=1 this is the
        # historical fallback; for K>=2 it gives a stricter
        # 2K-chain diagnostic.
        half = N // 2
        if half < 2:
            return jnp.full(chain.shape[2:], jnp.nan, dtype=chain.dtype)
        chain = jnp.concatenate([chain[:, :half], chain[:, half : 2 * half]], axis=0)
        K, N = 2 * K, half

    # Operate per-parameter component by flattening trailing dims.
    flat = chain.reshape(K, N, -1)
    # Within-chain variance (mean of per-chain variances), between-chain
    # variance (variance of per-chain means scaled by N).
    chain_means = jnp.mean(flat, axis=1)
    chain_vars = jnp.var(flat, axis=1, ddof=1)
    W = jnp.mean(chain_vars, axis=0)
    B = N * jnp.var(chain_means, axis=0, ddof=1)
    var_hat = ((N - 1) / N) * W + B / N
    out = jnp.sqrt(var_hat / jnp.where(W > 0, W, 1.0))
    return out.reshape(chain.shape[2:])


def ess(chain) -> jnp.ndarray:
    """Effective sample size via FFT-based autocorrelation, averaged
    across chains.

    ``chain`` is shaped ``(K, N, *param)``.  Returns an array of shape
    ``*param`` with one ESS per parameter component.  Values are bounded
    above by ``K * N`` (total raw samples).  ESS larger than ~100 per
    parameter is typically considered sufficient.

    Implementation: per-chain autocovariance via real FFT, averaged
    across chains, summed with the initial-monotone-sequence truncation
    rule of Geyer (1992).

    References
    ----------
    Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A.,
        & Rubin, D. B. (2014). *Bayesian Data Analysis*, 3rd ed.,
        Chapman & Hall/CRC, §11.5.
    """
    chain = jnp.asarray(chain)
    if chain.ndim < 2:
        raise ValueError(f"ess expects (K, N, *param) array; got shape {chain.shape}.")
    K, N = chain.shape[0], chain.shape[1]
    flat = chain.reshape(K, N, -1)  # (K, N, D)
    # Center each chain.
    centred = flat - jnp.mean(flat, axis=1, keepdims=True)
    # FFT-based autocovariance: zero-pad to length 2N for circular -> linear.
    n_fft = int(2 ** jnp.ceil(jnp.log2(2 * N)))
    f = jnp.fft.rfft(centred, n=n_fft, axis=1)
    acov = jnp.fft.irfft(f * jnp.conj(f), n=n_fft, axis=1)[:, :N, :].real
    # Normalise by sample-count taper (unbiased autocovariance).
    taper = N - jnp.arange(N)
    acov = acov / taper[None, :, None]
    # Average across chains, then normalise to autocorrelation.
    rho = jnp.mean(acov, axis=0)  # (N, D)
    rho_0 = rho[0:1]
    rho = rho / jnp.where(rho_0 > 0, rho_0, 1.0)
    # Geyer's initial-positive-sequence: truncate at the FIRST
    # non-positive pair (rho[2k] + rho[2k+1] ≤ 0) and discard
    # everything from that pair onward — including any later positive
    # pairs that would otherwise inflate the sum.  Strict rule from
    # Geyer (1992); the previous approximation summed every positive
    # pair anywhere in the sequence (over-counting on noisy chains).
    pair_sums = rho[0:-1:2] + rho[1::2]  # (N//2, D)
    # ``cummin`` over axis 0 gives the running minimum: as soon as one
    # pair becomes non-positive the cummin drops there and stays
    # ≤ 0 forever, so ``cummin > 0`` is True iff every preceding pair
    # (and this one) was strictly positive — the strict
    # initial-positive sequence.
    strict_keep = jnp.minimum.accumulate(pair_sums, axis=0) > 0
    # tau = 1 + 2 * sum_{k >= 1} rho[k] (autocorrelation time), clipped
    # to [1, N] for numerical safety on flat chains.
    tau = 1.0 + 2.0 * jnp.sum(jnp.where(strict_keep, pair_sums, 0.0), axis=0)
    tau = jnp.clip(tau, 1.0, float(N))
    out = (K * N) / tau
    return out.reshape(chain.shape[2:])
