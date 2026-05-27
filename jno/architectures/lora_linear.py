"""LoRA wrapper for Equinox models.

Wraps linear (or any custom) layers with low-rank adapters so that base
weights are frozen and only the adapter arrays are trained.

During forward (LoRALinear):  ``y = base(x) + (x @ A.T) @ B.T * (alpha / rank)``
After merging               :  ``y = merged_linear(x)``  (no runtime overhead)

Custom adapters
~~~~~~~~~~~~~~~
Subclass ``LoRAWrapper`` to support any layer type::

    class LoRAConv(LoRAWrapper):
        adapter_fields = ("delta_w",)

        @classmethod
        def applies_to(cls, leaf):
            return isinstance(leaf, eqx.nn.Conv2d) and not isinstance(leaf, LoRAWrapper)

        def __init__(self, base, rank, alpha, *, key): ...
        def __call__(self, x): ...
        def merge(self): ...

    net.lora(rank=4, wrapper=LoRAConv)
    net.lora(specs=[
        {"target": "linear", "rank": 4, "alpha": 1.0},
        {"target": "conv",   "rank": 8, "alpha": 2.0, "wrapper": LoRAConv},
    ])
"""

from __future__ import annotations

import re as _re
from typing import Any, ClassVar, Optional, Protocol, Sequence, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

# =====================================================================
# Linear-like protocol
# =====================================================================


@runtime_checkable
class LinearLike(Protocol):
    """Structural type for any linear module LoRA can wrap."""

    weight: jax.Array
    in_features: int
    out_features: int

    def __call__(self, x: jax.Array) -> jax.Array: ...


# =====================================================================
# LoRAWrapper — base class for all LoRA adapter modules
# =====================================================================


class LoRAWrapper(eqx.Module):
    """Base class for LoRA-style parameter-efficient adapters.

    Subclass this to support LoRA on any layer type.  Three things required:

    1. Set ``adapter_fields`` to the names of the trainable adapter attributes.
    2. Implement ``applies_to(leaf)`` — return ``True`` for leaves this wrapper
       can handle.
    3. Implement ``__init__(base, rank, alpha, *, key)`` and ``__call__``.
    4. Implement ``merge()`` — collapse adapters back into the base layer.
    """

    adapter_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        raise NotImplementedError(f"{cls.__name__}.applies_to() is not implemented")

    def merge(self) -> eqx.Module:
        raise NotImplementedError(f"{type(self).__name__}.merge() is not implemented")


# =====================================================================
# LoRALinear — built-in adapter for LinearLike layers
# =====================================================================


class LoRALinear(LoRAWrapper):
    """Linear layer with frozen base weights and trainable LoRA adapters.

    Attributes:
        base:   Original linear module (frozen during training). Accepts any
                linear-like ``eqx.Module`` with ``weight``, ``in_features``,
                and ``out_features`` attributes (e.g. jNO, foundax, or eqx.nn.Linear).
        lora_A: Down-projection, shape ``(rank, in_features)``.
        lora_B: Up-projection,   shape ``(out_features, rank)``.
        rank:   LoRA rank (static).
        alpha:  Scaling factor (static).
    """

    adapter_fields = ("lora_A", "lora_B")

    base: LinearLike
    lora_A: jax.Array
    lora_B: jax.Array
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return isinstance(leaf, eqx.Module) and not isinstance(leaf, LoRAWrapper) and isinstance(leaf, LinearLike)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
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

    def merge(self) -> eqx.Module:
        w = self.base.weight + (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        return eqx.tree_at(lambda m: m.weight, self.base, w)


# =====================================================================
# LoRASpec type
# =====================================================================

# Public spec dict keys: target (str regex), rank (int), alpha (float),
# and optionally wrapper (type[LoRAWrapper] | Sequence[type[LoRAWrapper]]).
LoRASpec = dict


# =====================================================================
# Internal helpers
# =====================================================================


class _Spec:
    """Compiled internal representation of one LoRA spec."""

    __slots__ = ("pat", "rank", "alpha", "wrappers")

    def __init__(
        self,
        pat: Optional[_re.Pattern],
        rank: int,
        alpha: float,
        wrappers: tuple[type[LoRAWrapper], ...],
    ):
        self.pat = pat
        self.rank = rank
        self.alpha = alpha
        self.wrappers = wrappers


def _normalize_wrappers(
    wrapper: type[LoRAWrapper] | Sequence[type[LoRAWrapper]] | None,
) -> tuple[type[LoRAWrapper], ...]:
    """Coerce the user-facing ``wrapper`` argument to a tuple of wrapper classes."""
    if wrapper is None:
        return (LoRALinear,)
    if isinstance(wrapper, type):
        return (wrapper,)
    return tuple(wrapper)


def _path_str(path_keys: Any) -> str:
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
# Public API
# =====================================================================


def apply_lora(
    model: eqx.Module,
    rank: int = 0,
    alpha: float = 1.0,
    *,
    key: jax.Array,
    target: Optional[str] = None,
    specs: Sequence[LoRASpec] | None = None,
    wrappers: type[LoRAWrapper] | Sequence[type[LoRAWrapper]] | None = None,
) -> eqx.Module:
    """Apply LoRA adapters to matching layers in *model*.

    Two calling conventions:

    1. **Uniform**: ``apply_lora(model, rank, alpha, key=key)``
       Wraps all matching layers with the same rank/alpha.  Pass ``target``
       to filter by pytree-path regex, and ``wrappers`` to override the
       adapter class (default: ``LoRALinear``).

    2. **Per-target**: ``apply_lora(model, key=key, specs=[...])``
       Each spec ``{"target": regex, "rank": int, "alpha": float}`` may
       also carry an optional ``"wrapper"`` key to use a different adapter
       class for that group.  The first matching spec wins.

    Returns:
        A new model with ``LoRAWrapper`` replacements at the matched leaves.
    """
    default_wrappers = _normalize_wrappers(wrappers)

    # Build the compiled spec list.
    if specs is not None:
        spec_list = [
            _Spec(
                pat=_re.compile(s["target"]) if s.get("target") else None,
                rank=int(s["rank"]),
                alpha=float(s["alpha"]),
                wrappers=_normalize_wrappers(s.get("wrapper")) if "wrapper" in s else default_wrappers,
            )
            for s in specs
        ]
    else:
        if rank <= 0:
            return model  # no-op
        spec_list = [
            _Spec(
                pat=_re.compile(target) if target else None,
                rank=int(rank),
                alpha=float(alpha),
                wrappers=default_wrappers,
            )
        ]

    # Union of all wrapper classes across all specs — used for is_leaf.
    all_wrappers: set[type[LoRAWrapper]] = {w for s in spec_list for w in s.wrappers}
    is_leaf = lambda x: any(w.applies_to(x) for w in all_wrappers)

    flat_with_path, treedef = jax.tree_util.tree_flatten_with_path(model, is_leaf=is_leaf)

    new_leaves: list[Any] = []
    for path_keys, leaf in flat_with_path:
        if not any(w.applies_to(leaf) for w in all_wrappers):
            new_leaves.append(leaf)
            continue

        pstr = _path_str(path_keys)
        new_leaf = leaf
        for spec in spec_list:
            if spec.pat is None or spec.pat.search(pstr):
                wrapper_cls = next((w for w in spec.wrappers if w.applies_to(leaf)), None)
                if wrapper_cls is not None:
                    key, subkey = jax.random.split(key)
                    new_leaf = wrapper_cls(leaf, spec.rank, spec.alpha, key=subkey)
                break
        new_leaves.append(new_leaf)

    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def merge_lora(model: eqx.Module) -> eqx.Module:
    """Collapse all LoRA adapters back into their base layers."""
    is_lora = lambda x: isinstance(x, LoRAWrapper)
    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=is_lora)
    return jax.tree_util.tree_unflatten(
        treedef,
        [leaf.merge() if isinstance(leaf, LoRAWrapper) else leaf for leaf in leaves],
    )


def lora_trainable_filter(model: eqx.Module) -> object:
    """Return a filter-spec pytree that marks only LoRA adapter arrays as trainable.

    Works with any ``LoRAWrapper`` subclass via its ``adapter_fields`` declaration.
    """
    adapter_ids: set[int] = set()
    for node in jax.tree_util.tree_leaves(model, is_leaf=lambda x: isinstance(x, LoRAWrapper)):
        if isinstance(node, LoRAWrapper):
            for fname in node.adapter_fields:
                arr = getattr(node, fname, None)
                if eqx.is_array(arr):
                    adapter_ids.add(id(arr))

    flat, treedef = jax.tree_util.tree_flatten(model)
    specs = [eqx.is_array(leaf) and id(leaf) in adapter_ids for leaf in flat]
    return jax.tree_util.tree_unflatten(treedef, specs)
