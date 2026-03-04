"""LoRA wrapper for Linear layers and Flax models.

**Equinox models (jNO Linear layers):**
Wraps a ``Linear`` module so that the base weights are frozen and only
the low-rank adapters ``lora_A`` and ``lora_B`` are trainable.

During forward:  ``y = base(x) + (x @ A.T) @ B.T * (alpha / rank)``
After merging :  ``y = merged_linear(x)``  (no runtime overhead)

**Flax models (FlaxModelWrapper):**
Wraps a ``FlaxModelWrapper`` with LoRA adapters injected alongside every
2-D ``kernel`` array in the Flax parameter dict.  The base weights are
frozen; only the low-rank ``lora_a`` / ``lora_b`` matrices are trained.

During forward:  ``kernel_eff = kernel + (alpha / rank) * lora_b @ lora_a``
After merging :  Returns a new ``FlaxModelWrapper`` (no runtime overhead)
"""

from __future__ import annotations

from typing import Any, Optional, Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear


# =====================================================================
# Equinox LoRA (for jNO Linear layers)
# =====================================================================


class LoRALinear(eqx.Module):
    """Linear layer with frozen base weights and trainable LoRA adapters.

    Attributes:
        base:   Original ``Linear`` module (frozen during training).
        lora_A: Down-projection, shape ``(rank, in_features)``.
        lora_B: Up-projection,   shape ``(out_features, rank)``.
        rank:   LoRA rank (static).
        alpha:  Scaling factor (static).
    """

    base: Linear
    lora_A: jax.Array
    lora_B: jax.Array
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, base: Linear, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        self.rank = min(rank, min(base.in_features, base.out_features))
        self.alpha = alpha

        k1, _ = jax.random.split(key)
        std = 1.0 / jnp.sqrt(base.in_features)
        self.lora_A = jax.random.normal(k1, (self.rank, base.in_features)) * std
        self.lora_B = jnp.zeros((base.out_features, self.rank))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T * (self.alpha / self.rank)
        return y + delta


# =====================================================================
# Flax LoRA (for FlaxModelWrapper — Poseidon, ScOT, etc.)
# =====================================================================


class FlaxLoRAWrapper(eqx.Module):
    """Wraps a ``FlaxModelWrapper`` with LoRA adapters for fine-tuning.

    The base model weights are frozen; only the low-rank ``lora_a`` /
    ``lora_b`` adapter matrices stored in ``lora_params`` are trained.
    During the forward pass, adapters are merged into the kernel weights
    on-the-fly so the original ``apply_fn`` sees corrected weights.

    ``lora_params`` mirrors the nested dict structure of
    ``base.params`` but only contains entries at ``kernel`` positions,
    each holding ``{'lora_a': array, 'lora_b': array}``.
    """

    base: Any  # FlaxModelWrapper — frozen during LoRA training
    lora_params: dict  # nested dict with lora_a / lora_b at kernel positions
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, base, lora_params: dict, rank: int, alpha: float):
        self.base = base
        self.lora_params = lora_params
        self.rank = rank
        self.alpha = alpha

    # ── forward pass ─────────────────────────────────────────

    def __call__(self, *args, **kwargs):
        scale = self.alpha / self.rank
        merged_params = _merge_lora_into_flax_params(
            self.base.params,
            self.lora_params,
            scale,
        )

        # Replicate FlaxModelWrapper forward logic with merged params
        merged_kwargs = {**self.base.default_kwargs, **kwargs}
        merged_kwargs.pop("key", None)

        param_dtype = self._param_dtype()
        if param_dtype is not None:
            args = tuple(a.astype(param_dtype) if hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating) and a.dtype != param_dtype else a for a in args)

        result = self.base.apply_fn(merged_params, *args, **merged_kwargs)
        if self.base.post_fn is not None:
            result = self.base.post_fn(result)

        if param_dtype is not None and param_dtype != jnp.float32:
            result = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
                result,
            )
        return result

    def _param_dtype(self):
        return self.base._param_dtype()


# ── Flax LoRA helpers ────────────────────────────────────────────────


def _build_flax_lora_params(params, rank: int, key: jax.Array):
    """Build a nested LoRA-param dict mirroring the Flax ``params`` structure.

    For every ``kernel`` leaf with ``ndim == 2`` a sibling dict
    ``{'lora_a': (rank, in), 'lora_b': (out, rank)}`` is created.

    Returns:
        (lora_dict, n_layers, n_lora_params)
    """
    n_layers = 0
    n_params = 0

    def _recurse(d):
        nonlocal key, n_layers, n_params
        if not isinstance(d, dict):
            return None
        result = {}
        for k in sorted(d.keys()):
            v = d[k]
            if k == "kernel" and hasattr(v, "ndim") and v.ndim == 2:
                in_dim, out_dim = v.shape
                eff_rank = min(rank, min(in_dim, out_dim))
                key, k1 = jax.random.split(key)
                std = 1.0 / jnp.sqrt(float(in_dim))
                a = jax.random.normal(k1, (eff_rank, in_dim)) * std
                b = jnp.zeros((out_dim, eff_rank))
                result[k] = {"lora_a": a, "lora_b": b}
                n_layers += 1
                n_params += a.size + b.size
            else:
                sub = _recurse(v)
                if sub is not None:
                    result[k] = sub
                    key, _ = jax.random.split(key)
        return result if result else None

    lora_dict = _recurse(params)
    return lora_dict or {}, n_layers, n_params


def _merge_lora_into_flax_params(params, lora_params, scale: float):
    """Merge LoRA deltas into a Flax params dict (used in forward pass).

    For every ``kernel`` that has matching ``lora_a`` / ``lora_b``::

        kernel_eff = kernel + scale * lora_b @ lora_a
    """
    if not isinstance(params, dict) or not lora_params:
        return params
    result = {}
    for k, v in params.items():
        if k in lora_params:
            lp = lora_params[k]
            if k == "kernel" and isinstance(lp, dict) and "lora_a" in lp:
                result[k] = v + scale * (lp["lora_b"] @ lp["lora_a"]).T
            else:
                result[k] = _merge_lora_into_flax_params(v, lp, scale)
        else:
            result[k] = v
    return result


# =====================================================================
# Unified helpers  (apply / merge / trainable-filter)
# =====================================================================


def apply_lora(model: eqx.Module, rank: int, alpha: float, *, key: jax.Array) -> eqx.Module:
    """Apply LoRA to *model*.

    * If *model* is a ``FlaxModelWrapper`` → wraps it in a
      ``FlaxLoRAWrapper`` with LoRA adapters for every 2-D kernel.
    * Otherwise → replaces every jNO ``Linear`` layer with ``LoRALinear``.

    .. note::
        ``FlaxNNXWrapper`` (flax.nnx models) are not yet supported by LoRA.
        Calling ``.lora()`` on an NNX-wrapped model will have no effect and
        log a warning.
    """
    from .common import FlaxModelWrapper, FlaxNNXWrapper

    if isinstance(model, FlaxNNXWrapper):
        import logging

        logging.getLogger(__name__).warning("LoRA is not yet supported for flax.nnx models (FlaxNNXWrapper). " "The .lora() call has no effect on this model.")
        return model

    if isinstance(model, FlaxModelWrapper):
        lora_params, n_layers, n_lora_params = _build_flax_lora_params(
            model.params,
            rank,
            key,
        )
        return FlaxLoRAWrapper(model, lora_params, rank, alpha)

    # Fallback: Equinox models with jNO Linear layers
    def _replace(leaf):
        nonlocal key
        if isinstance(leaf, Linear):
            key, subkey = jax.random.split(key)
            return LoRALinear(leaf, rank, alpha, key=subkey)
        return leaf

    is_linear = lambda x: isinstance(x, Linear)
    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=is_linear)
    return jax.tree_util.tree_unflatten(treedef, [_replace(l) for l in leaves])


def merge_lora(model: eqx.Module) -> eqx.Module:
    """Collapse LoRA adapters back into base weights.

    * ``FlaxLoRAWrapper`` → returns a plain ``FlaxModelWrapper``.
    * Equinox model with ``LoRALinear`` → collapses to plain ``Linear``.
    """
    from .common import FlaxModelWrapper

    if isinstance(model, FlaxLoRAWrapper):
        scale = model.alpha / model.rank
        merged_params = _merge_lora_into_flax_params(
            model.base.params,
            model.lora_params,
            scale,
        )
        return FlaxModelWrapper(
            model.base.apply_fn,
            merged_params,
            post_fn=model.base.post_fn,
            **model.base.default_kwargs,
        )

    # Fallback: LoRALinear → Linear
    def _merge(leaf):
        if not isinstance(leaf, LoRALinear):
            return leaf
        s = leaf.alpha / leaf.rank
        w = leaf.base.weight + s * (leaf.lora_B @ leaf.lora_A)
        new = Linear.__new__(Linear)
        object.__setattr__(new, "weight", w)
        object.__setattr__(new, "bias", leaf.base.bias)
        object.__setattr__(new, "in_features", leaf.base.in_features)
        object.__setattr__(new, "out_features", leaf.base.out_features)
        return new

    is_lora = lambda x: isinstance(x, LoRALinear)
    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=is_lora)
    return jax.tree_util.tree_unflatten(treedef, [_merge(l) for l in leaves])


def lora_trainable_filter(model: eqx.Module) -> object:
    """Return a filter-spec pytree: ``True`` for trainable LoRA arrays,
    ``False`` for frozen base arrays.

    Works for both ``FlaxLoRAWrapper`` and Equinox models with
    ``LoRALinear`` layers.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(model)
    specs = []
    for path, leaf in flat:
        if not eqx.is_array(leaf):
            specs.append(leaf)
            continue

        if isinstance(model, FlaxLoRAWrapper):
            # Trainable iff inside ``lora_params``
            in_lora = any(isinstance(k, jax.tree_util.GetAttrKey) and k.name == "lora_params" for k in path)
            specs.append(in_lora)
        else:
            # Original LoRALinear logic: frozen iff inside ``base``
            frozen = any(isinstance(k, jax.tree_util.GetAttrKey) and k.name == "base" for k in path)
            specs.append(not frozen)

    return jax.tree_util.tree_unflatten(treedef, specs)
