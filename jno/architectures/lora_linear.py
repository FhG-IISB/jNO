"""LoRA wrapper for Linear layers.

Wraps a ``Linear`` module so that the base weights are frozen and only
the low-rank adapters ``lora_A`` and ``lora_B`` are trainable.

During forward:  ``y = base(x) + (x @ A.T) @ B.T * (alpha / rank)``
After merging :  ``y = merged_linear(x)``  (no runtime overhead)
"""

import jax
import jax.numpy as jnp
import equinox as eqx

from .linear import Linear


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


# ── helpers ──────────────────────────────────────────────────────────


def apply_lora(model: eqx.Module, rank: int, alpha: float, *, key: jax.Array) -> eqx.Module:
    """Replace every ``Linear`` inside *model* with a ``LoRALinear``."""

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
    """Collapse every ``LoRALinear`` back into a plain ``Linear``.

    Merged weight: ``W + (alpha / rank) * B @ A``.
    """

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
    """Return a filter-spec pytree: ``True`` for trainable arrays, ``False``
    for frozen arrays (everything inside ``LoRALinear.base``).

    Non-array leaves keep their original value so the spec can be used
    directly with ``eqx.partition(model, spec)``.
    """
    flat, treedef = jax.tree_util.tree_flatten_with_path(model)
    specs = []
    for path, leaf in flat:
        if not eqx.is_array(leaf):
            specs.append(leaf)
            continue
        # Frozen if any ancestor key is "base" (inside a LoRALinear).
        frozen = any(isinstance(k, jax.tree_util.GetAttrKey) and k.name == "base" for k in path)
        specs.append(not frozen)
    return jax.tree_util.tree_unflatten(treedef, specs)
