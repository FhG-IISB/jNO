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
    ):
        self.factory = factory
        self.kind = kind
        self.prior_fn = prior_fn
        self.step_size = step_size
        self.extra_kwargs = extra_kwargs
        self.warmup = warmup
        self.keep = keep
        self.thin = thin


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
    if warmup < 0 or keep < 0 or thin < 1:
        raise ValueError("warmup>=0, keep>=0, thin>=1 are required.")

    if kind == "full":
        extra = {**user_kwargs, "step_size": float(step_size)}
        return _KernelHandle(factory, kind, prior_fn, None, extra, warmup, keep, thin)
    return _KernelHandle(factory, kind, prior_fn, float(step_size), user_kwargs, warmup, keep, thin)


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


def init_state(handle: _KernelHandle, position):
    """Initialise the kernel state for ``position``.

    For full-data kernels we build a one-off kernel with a trivial
    logdensity (the prior alone) just to call ``.init`` — the real
    logdensity is rebuilt per step.  For SG-MCMC, ``init`` only needs the
    position.

    Before constructing the kernel we may inject a default
    ``inverse_mass_matrix`` of identity shape inferred from ``position`` —
    see :func:`_maybe_inject_inverse_mass_matrix`.
    """
    _maybe_inject_inverse_mass_matrix(handle, position)
    factory = handle.factory
    if handle.kind == "full":
        kernel = factory(handle.prior_fn, **handle.extra_kwargs)
    else:

        def _dummy_grad(p, _mb):
            return jax.tree_util.tree_map(jnp.zeros_like, p)

        kernel = factory(_dummy_grad, **handle.extra_kwargs)
    return kernel.init(position)


def step(
    handle: _KernelHandle,
    rng_key,
    state,
    position,
    logdensity_fn: Callable,
    grad_estimator: Callable,
    minibatch_ctx,
):
    """One kernel step.

    Caller supplies both a ``logdensity_fn`` (used by full-data kernels)
    and a ``grad_estimator`` (used by SG-MCMC).  We pick the right one,
    build the blackjax kernel inside the JIT trace, and dispatch.

    Returns ``(new_state, new_position)``.  For SG-MCMC the state *is*
    the position; for full-data it is an ``HMCState`` whose ``.position``
    field carries the new sample.
    """
    factory = handle.factory
    if handle.kind == "full":
        kernel = factory(logdensity_fn, **handle.extra_kwargs)
        new_state, _info = kernel.step(rng_key, state)
        return new_state, new_state.position

    kernel = factory(grad_estimator, **handle.extra_kwargs)
    new_state = kernel.step(rng_key, state, minibatch_ctx, handle.step_size)
    return new_state, new_state


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
