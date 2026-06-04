"""Internal adapters that wire blackjax MCMC kernels into jno's training loop.

This module is **not** part of the public API — users compose `blackjax`
kernels directly and attach them per parameter via
:meth:`jno.trace.Model.bayesian`.  The training loop in
:mod:`jno.core` then dispatches each model's per-step update either through
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
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

_FULL_FIRST_ARG = "logdensity_fn"
_GRAD_FIRST_ARG = "grad_estimator"
_VI_SECOND_ARG = "optimizer"  # blackjax.meanfield_vi has (logdensity_fn, optimizer, ...)


def default_gaussian_prior(position, *, sigma: float = 10.0):
    """Wide isotropic Gaussian log-prior over a pytree position.

    ``log p(θ) = -‖θ‖² / (2σ²)``, summed over every inexact-array leaf.
    """
    leaves = jax.tree_util.tree_leaves(position)
    sq = jnp.array(0.0)
    for leaf in leaves:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            sq = sq + jnp.sum(jnp.asarray(leaf) ** 2)
    return -0.5 * sq / (sigma * sigma)


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
        # Phase 11: ``.mask(M).bayesian()`` / ``.mask(M).vi()``.  ``None``
        # means "operate on the full position" (the default for global
        # configurators).  When set, the kernel sees only the masked
        # subset of the position; the unmasked portion is held constant
        # at the model's current ``trainable[lid]`` snapshot and
        # reassembled inside the closure before each logdensity eval.
        "param_mask",
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
    if step_size is None:
        raise ValueError(
            f"jno .bayesian({getattr(factory, '__name__', factory)!r}, ...) requires a step_size= keyword argument."
        )

    warmup = int(cfg.get("warmup", 500))
    keep = int(cfg.get("keep", 1000))
    thin = int(cfg.get("thin", 1))
    adapt = bool(cfg.get("adapt", True))
    num_chains = int(cfg.get("num_chains", 1))
    init_jitter = float(cfg.get("init_jitter", 0.0))
    if warmup < 0 or keep < 0 or thin < 1:
        raise ValueError("warmup>=0, keep>=0, thin>=1 are required.")
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1, got {num_chains}.")

    if kind == "full":
        extra = {**user_kwargs, "step_size": float(step_size)}
        return _KernelHandle(factory, kind, prior_fn, None, extra, warmup, keep, thin, adapt, num_chains, init_jitter)
    return _KernelHandle(
        factory, kind, prior_fn, float(step_size), user_kwargs, warmup, keep, thin, adapt, num_chains, init_jitter
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
        # Two manual overrides on the blackjax-default init:
        # 1. ``state.mu`` is initialised at zeros regardless of the
        #    position argument.  For non-trivial models (e.g. an MLP
        #    with Xavier init) starting at mu=zeros makes the ELBO
        #    landscape flat and convergence painfully slow.  We set
        #    mu to the user-supplied initial weights so VI starts
        #    from a reasonable point — matches numpyro's autoguide.
        # 2. ``state.rho`` defaults to large values (≈ exp(rho) ≈ 1,
        #    meaning the variational q has unit std per weight).  For
        #    multi-layer MLPs that gives extremely noisy MC ELBO
        #    gradients.  We shrink rho to ``log_std = -3`` (std ≈ 0.05)
        #    so the initial q is tight; the optimiser then *grows* rho
        #    where the posterior is genuinely wide.
        small_rho = jax.tree_util.tree_map(lambda x: jnp.full_like(x, -3.0), state.rho)
        return state._replace(mu=init_position, rho=small_rho)

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
    transition.  Returns both ``new_state`` and ``new_position`` with
    leading K.

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
        return new_state, new_pos

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
            return new_state, new_pos
        kernel = factory(grad_estimator_factory, **handle.extra_kwargs)
        new_state = kernel.step(rng_key, state, minibatch_ctx, handle.step_size)
        new_pos = new_state
        if _reassemble_single is not None:
            new_pos = _reassemble_single(new_pos)
        return new_state, new_pos

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
            return new_s, new_p

        return jax.vmap(_one)(state, keys, chain_idx)

    step_size = handle.step_size

    def _one(state_k, key_k, k_idx):
        ge = lambda p, mb, _k=k_idx: grad_estimator_factory(p, mb, _k)
        kernel_k = factory(ge, **handle.extra_kwargs)
        new_s = kernel_k.step(key_k, state_k, minibatch_ctx, step_size)
        new_p = new_s
        if _reassemble_single is not None:
            new_p = _reassemble_single(new_p)
        return new_s, new_p

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


def rhat(chain) -> jnp.ndarray:
    """Rank-normalised, folded R-hat (Gelman & Rubin 1992, improved by
    Vehtari et al. 2021).

    ``chain`` is shaped ``(K, N, *param)`` — K chains, N draws each.
    Returns an array of shape ``*param`` with one R-hat per parameter
    component.  R-hat ≈ 1.0 indicates the chains are exploring the same
    distribution; values above ~1.05 suggest non-convergence.

    Notes
    -----
    For K=1 this collapses to a single-chain split-R-hat using two
    halves of the lone chain (still informative for stationarity but
    not as strong as a true multi-chain diagnostic).

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
    chain = jnp.asarray(chain)
    if chain.ndim < 2:
        raise ValueError(f"rhat expects (K, N, *param) array; got shape {chain.shape}.")
    K, N = chain.shape[0], chain.shape[1]
    if K == 1:
        # Split-R-hat fallback: treat the two halves of the lone chain
        # as two independent chains (Gelman et al. 2014, BDA3 §11.4).
        half = N // 2
        if half < 2:
            return jnp.full(chain.shape[2:], jnp.nan, dtype=chain.dtype)
        chain = jnp.stack([chain[0, :half], chain[0, half : 2 * half]], axis=0)
        K, N = 2, half

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
    # Geyer's initial-positive-sequence: truncate at first negative
    # *pair* (rho[2k] + rho[2k+1] < 0).  Vectorised approximation: take
    # cumulative sum until the running estimate stops decreasing.
    pair_sums = rho[0:-1:2] + rho[1::2]  # (N//2, D)
    # First lag k where pair_sums[k] becomes non-positive — keep
    # everything strictly before that.
    keep = pair_sums > 0
    # tau = 1 + 2 * sum_{k >= 1} rho[k] (autocorrelation time).  We
    # approximate as 1 + 2 * sum of paired positive sums, capped at N
    # for numerical safety on flat chains.
    tau = 1.0 + 2.0 * jnp.sum(jnp.where(keep, pair_sums, 0.0), axis=0)
    tau = jnp.clip(tau, 1.0, float(N))
    out = (K * N) / tau
    return out.reshape(chain.shape[2:])
