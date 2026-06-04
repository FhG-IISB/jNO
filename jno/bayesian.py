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

import jax
import jax.numpy as jnp

_FULL_FIRST_ARG = "logdensity_fn"
_GRAD_FIRST_ARG = "grad_estimator"


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
    """Return ``'full'`` or ``'grad_estimator'`` for a blackjax kernel factory.

    Inspects ``factory.differentiable`` (the wrapped callable on blackjax's
    ``GenerateSamplingAPI``) and looks at the first parameter name.  Falls
    back to ``factory`` itself for plain callables.
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
    if first == _FULL_FIRST_ARG:
        return "full"
    if first == _GRAD_FIRST_ARG:
        return "grad_estimator"
    raise ValueError(
        f"Unrecognised blackjax kernel factory {factory!r}: first argument "
        f"is {first!r}, expected one of "
        f"{(_FULL_FIRST_ARG, _GRAD_FIRST_ARG)}."
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


def build_kernel_handle(cfg: dict) -> _KernelHandle:
    """Construct an internal handle from a ``_bayesian_cfg`` dict.

    ``cfg`` is the value stored on ``Model._bayesian_cfg`` by
    :meth:`jno.trace.Model.bayesian`.  Keys: ``factory``, ``prior``,
    ``warmup``, ``keep``, ``thin``, plus user-supplied kernel kwargs
    (e.g. ``step_size``, ``inverse_mass_matrix``).
    """
    factory = cfg["factory"]
    kind = _detect_kind(factory)
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
    _maybe_inject_inverse_mass_matrix(handle, position)
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
        return kernel.init(position)

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    position_K = _replicate_with_jitter(position, handle.num_chains, handle.init_jitter, rng_key)
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

    if K == 1:
        # Backward-compatible single-chain path: state has no K axis,
        # kernel built once outside, no jax.vmap.  Bit-identical to the
        # pre-multi-chain JIT trace / gradient / PRNG behaviour.
        # ``core.py`` passes ``(p) -> log p`` style closures here (not
        # factories), matching the OLD code form.
        if handle.kind == "full":
            kernel = factory(logdensity_factory, **handle.extra_kwargs)
            new_state, _info = kernel.step(rng_key, state)
            return new_state, new_state.position
        kernel = factory(grad_estimator_factory, **handle.extra_kwargs)
        new_state = kernel.step(rng_key, state, minibatch_ctx, handle.step_size)
        return new_state, new_state

    # Multi-chain path: per-chain PRNG keys + vmap.  ``core.py`` passes
    # *factories* taking ``(p, k_idx)`` / ``(p, mb, k_idx)`` here.
    keys = jax.random.split(rng_key, K)
    chain_idx = jnp.arange(K)

    if handle.kind == "full":

        def _one(state_k, key_k, k_idx):
            ld = lambda p, _k=k_idx: logdensity_factory(p, _k)
            kernel_k = factory(ld, **handle.extra_kwargs)
            new_s, _info = kernel_k.step(key_k, state_k)
            return new_s, new_s.position

        return jax.vmap(_one)(state, keys, chain_idx)

    step_size = handle.step_size

    def _one(state_k, key_k, k_idx):
        ge = lambda p, mb, _k=k_idx: grad_estimator_factory(p, mb, _k)
        kernel_k = factory(ge, **handle.extra_kwargs)
        new_s = kernel_k.step(key_k, state_k, minibatch_ctx, step_size)
        return new_s, new_s

    return jax.vmap(_one)(state, keys, chain_idx)


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
