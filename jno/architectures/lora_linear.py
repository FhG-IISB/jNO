"""LoRA wrapper for Equinox models.

Wraps ``Linear`` (or ``eqx.nn.Linear``) modules so that the base weights
are frozen and only the low-rank adapters ``lora_A`` and ``lora_B`` are
trainable.

During forward:  ``y = base(x) + (x @ A.T) @ B.T * (alpha / rank)``
After merging :  ``y = merged_linear(x)``  (no runtime overhead)

Per-target LoRA
~~~~~~~~~~~~~~~
``apply_lora`` accepts a list of ``LoRASpec`` dicts to apply different
ranks/alphas to different parts of the model, matched by regex on the
pytree path::

    specs = [
        {"target": "encoder",  "rank": 4,  "alpha": 1.0},
        {"target": "decoder",  "rank": 16, "alpha": 4.0},
    ]
    model = apply_lora(model, specs=specs, key=key)
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import re as _re

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
# LoRA spec type
# =====================================================================

# A single LoRA spec: {"target": regex_str, "rank": int, "alpha": float}
# ``target`` is matched via ``re.search`` against the slash-joined pytree
# path of each ``Linear`` leaf.
LoRASpec = dict  # keys: target (str), rank (int), alpha (float)


# =====================================================================
# Path utilities
# =====================================================================


def _path_str(path_keys) -> str:
    """Convert a JAX pytree path to a slash-joined string for regex matching."""
    parts = []
    for k in path_keys:
        if hasattr(k, "key"):
            parts.append(str(k.key))
        elif hasattr(k, "idx"):
            parts.append(str(k.idx))
        elif hasattr(k, "name"):
            parts.append(k.name)
        else:
            parts.append(str(k))
    return "/".join(parts)


# =====================================================================
# Unified helpers  (apply / merge / trainable-filter)
# =====================================================================


def apply_lora(
    model: eqx.Module,
    rank: int = 0,
    alpha: float = 1.0,
    *,
    key: jax.Array,
    target: str = None,
    specs: Sequence[LoRASpec] | None = None,
) -> eqx.Module:
    """Apply LoRA to every ``Linear`` layer in *model*.

    Two calling conventions:

    1. **Uniform** (backward-compatible): ``apply_lora(model, rank, alpha, key=key)``
       Applies the same rank/alpha to all ``Linear`` layers (optionally
       filtered by ``target`` regex).

    2. **Per-target**: ``apply_lora(model, key=key, specs=[...])``
       Each spec ``{"target": regex, "rank": int, "alpha": float}``
       is matched against the pytree path. A layer matched by multiple
       specs uses the **first** matching spec.

    Returns:
        A new model with ``LoRALinear`` replacements.
    """
    # Normalise to a list of (compiled_regex | None, rank, alpha).
    if specs is not None:
        spec_list = []
        for s in specs:
            pat = _re.compile(s["target"]) if s.get("target") else None
            spec_list.append((pat, int(s["rank"]), float(s["alpha"])))
    else:
        # Uniform LoRA — single spec.
        if rank <= 0:
            return model  # no-op
        pat = _re.compile(target) if target else None
        spec_list = [(pat, int(rank), float(alpha))]

    is_linear = lambda x: isinstance(x, Linear)
    flat_with_path, treedef = jax.tree_util.tree_flatten_with_path(model, is_leaf=is_linear)

    new_leaves = []
    for path_keys, leaf in flat_with_path:
        if isinstance(leaf, Linear):
            pstr = _path_str(path_keys)
            matched_spec = None
            for pat, r, a in spec_list:
                if pat is None or pat.search(pstr):
                    matched_spec = (r, a)
                    break
            if matched_spec is not None:
                r, a = matched_spec
                key, subkey = jax.random.split(key)
                new_leaves.append(LoRALinear(leaf, r, a, key=subkey))
            else:
                new_leaves.append(leaf)
        else:
            new_leaves.append(leaf)

    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def merge_lora(model: eqx.Module) -> eqx.Module:
    """Collapse LoRA adapters back into base weights (``LoRALinear`` → ``Linear``)."""

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


def lora_trainable_filter(
    model: eqx.Module,
    *,
    base_param_mask: Optional[object] = None,
    freeze_base: bool = True,
) -> object:
    """Return a filter-spec pytree for LoRA training.

    Args:
        model: Model containing ``LoRALinear`` layers.
        base_param_mask: Optional bool pytree where ``True`` marks base
            leaves that should stay **frozen** even when ``freeze_base=False``.
        freeze_base: If ``True`` (default), only LoRA arrays are trainable
            and all base arrays are frozen. If ``False``, base arrays are
            also trainable (useful for partial fine-tuning + LoRA).
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(model)
    specs = []
    for path, leaf in flat:
        if not eqx.is_array(leaf):
            specs.append(leaf)
            continue

        # LoRALinear: base.* is frozen, lora_A / lora_B are trainable.
        in_base = any(isinstance(k, jax.tree_util.GetAttrKey) and k.name == "base" for k in path)
        is_adapter = any(isinstance(k, jax.tree_util.GetAttrKey) and k.name in ("lora_A", "lora_B") for k in path)

        if is_adapter:
            specs.append(True)
        elif in_base:
            specs.append(not freeze_base)
        else:
            # Non-LoRA arrays in the model — follow freeze_base policy.
            specs.append(True if not freeze_base else eqx.is_inexact_array(leaf))

    return jax.tree_util.tree_unflatten(treedef, specs)
