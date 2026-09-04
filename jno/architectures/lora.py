"""LoRA wrapper for Equinox models.

Wraps linear (or any custom) layers with low-rank adapters so that base
weights are frozen and only the adapter arrays are trained.

During forward (LoRALinear):  ``y = base(x) + (x @ A.T) @ B.T * (alpha / rank)``
After merging               :  ``y = merged_linear(x)``  (no runtime overhead)

LoRA Zoo — Linear
~~~~~~~~~~~~~~~~~
Several ``LoRAWrapper`` subclasses are provided out of the box for linear layers:

- ``LoRALinear``   — vanilla LoRA (trainable A and B)
- ``rsLoRALinear`` — rank-stabilized: scales by ``alpha/sqrt(rank)``
- ``LoRAFALinear`` — frozen-A: only B is trained (half the adapter parameters)
- ``DoRALinear``   — weight-decomposed: trains magnitude and direction separately
- ``PiSSALinear``  — SVD-initialised: adapters start from principal weight components
- ``LoRAXSLinear`` — extra-small: A and B fixed from SVD, only a tiny R matrix is trained
- ``VeRALinear``   — frozen random A,B from a seed (XLA constants); only b,d vectors trained
- ``MiLoRALinear`` — like PiSSA but adapts the minor (noise-subspace) singular components
- ``IA3Linear``    — scales output activations via a learned vector; no low-rank matrices
- ``LoKrLinear``   — Kronecker product adapter: delta_W = kron(kron_A, kron_B)
- ``OFTLinear``    — block-diagonal orthogonal fine-tuning via Cayley map

LoRA Zoo — Conv
~~~~~~~~~~~~~~~
Matching conv variants for ``eqx.nn.Conv1d/2d/3d`` (ConvTranspose excluded):

- ``LoRAConv``   — vanilla LoRA on flattened conv weight
- ``rsLoRAConv`` — rank-stabilized conv LoRA
- ``LoRAFAConv`` — frozen-A conv LoRA
- ``DoRAConv``   — weight-decomposed conv LoRA
- ``PiSSAConv``  — SVD-initialised conv LoRA (principal components)
- ``LoRAXSConv`` — extra-small conv LoRA with trainable r×r core
- ``VeRAConv``   — seed-based frozen A,B; only b,d vectors trained
- ``MiLoRAConv`` — SVD minor-component conv LoRA
- ``IA3Conv``    — per-output-channel scale vector
- ``LoKrConv``   — Kronecker product conv adapter
- ``OFTConv``    — block-diagonal orthogonal fine-tuning for conv

Custom adapters
~~~~~~~~~~~~~~~
Subclass ``LoRAWrapper`` to support any layer type::

    class MyConvAdapter(LoRAWrapper):
        adapter_fields = ("delta_w",)

        @classmethod
        def applies_to(cls, leaf):
            return isinstance(leaf, eqx.nn.Conv2d) and not isinstance(leaf, LoRAWrapper)

        def __init__(self, base, rank, alpha, *, key): ...
        def __call__(self, x): ...
        def merge(self): ...

    net.lora(rank=4, wrapper=MyConvAdapter)
    net.lora(specs=[
        {"target": "linear", "rank": 4, "alpha": 1.0},
        {"target": "conv",   "rank": 8, "alpha": 2.0, "wrapper": MyConvAdapter},
    ])
"""

from __future__ import annotations

import math as _math
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


@runtime_checkable
class ConvLike(Protocol):
    """Structural type for any conv module LoRA can wrap (excludes ConvTranspose)."""

    weight: jax.Array
    in_channels: int
    out_channels: int

    def __call__(self, x: jax.Array) -> jax.Array: ...


try:
    _CONV_TRANSPOSE_TYPES: tuple[type, ...] = (
        eqx.nn.ConvTranspose1d,
        eqx.nn.ConvTranspose2d,
        eqx.nn.ConvTranspose3d,
    )
except AttributeError:
    _CONV_TRANSPOSE_TYPES = ()


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

        # A ~ N(0, 1/in_features), B = 0 — output is zero at init, A variance matches
        # the pre-existing weight scale. Justified in https://proceedings.neurips.cc/paper_files/paper/2024/file/d4387c37b3b06e55f86eccdb8cd1f829-Paper-Conference.pdf
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
    """Coerce the user-facing ``wrapper`` argument to a tuple of wrapper classes.

    When ``wrapper`` is ``None``, returns ``(LoRALinear, LoRAConv)`` — both linear
    and conv layers are wrapped by default.  ``LoRAConv`` is resolved at call time
    (Python late binding), so the forward reference is safe.
    """
    if wrapper is None:
        return (LoRALinear, LoRAConv)  # LoRAConv defined later in this module
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
    param_mask: Any = None,
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

    Args:
        param_mask: Optional boolean pytree mirroring the model structure.
            When provided, only modules whose corresponding sub-tree contains
            at least one ``True`` leaf are wrapped with LoRA adapters.
            Produced by ``mask(M).lora()``; pass ``None`` to wrap all
            matching layers (default behaviour).

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
                wrappers=s["wrappers"]
                if "wrappers" in s
                else _normalize_wrappers(s.get("wrapper"))
                if "wrapper" in s
                else default_wrappers,
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

    # Build set of path strings with a True mask leaf — used to restrict wrapping
    # to user-selected modules when mask(M).lora() is called.
    if param_mask is not None:
        flat_mask_with_path = jax.tree_util.tree_flatten_with_path(param_mask)[0]
        mask_true_paths: set[str] | None = {
            _path_str(mp) for mp, val in flat_mask_with_path if val is True or (isinstance(val, bool) and val)
        }
    else:
        mask_true_paths = None

    flat_with_path, treedef = jax.tree_util.tree_flatten_with_path(model, is_leaf=is_leaf)

    new_leaves: list[Any] = []
    for path_keys, leaf in flat_with_path:
        if not any(w.applies_to(leaf) for w in all_wrappers):
            new_leaves.append(leaf)
            continue

        pstr = _path_str(path_keys)

        # Honour param_mask: skip modules not covered by any True leaf.
        # Use `pstr + "/"` prefix to avoid "layer1" matching "layer10/weight".
        if mask_true_paths is not None:
            module_selected = any(tp == pstr or tp.startswith(pstr + "/") for tp in mask_true_paths)
            if not module_selected:
                new_leaves.append(leaf)
                continue

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


def partial_lora_trainable_filter(model: eqx.Module) -> object:
    """Return a filter-spec for partial LoRA (mask(M).lora() without freeze()).

    Three-bucket logic:
    - LoRA adapter arrays (b, d, lora_A, lora_B, …)  → ``True``  (trained)
    - Base arrays inside LoRA wrappers                 → ``False`` (frozen)
    - Arrays outside any LoRA wrapper                  → ``True``  (trained)

    Use this when the user says ``mask(M).lora()`` without a prior ``freeze()``:
    selected layers get LoRA with frozen bases; everything else stays trainable.
    """
    adapter_ids: set[int] = set()
    wrapped_base_ids: set[int] = set()

    is_lora = lambda x: isinstance(x, LoRAWrapper)
    for node in jax.tree_util.tree_leaves(model, is_leaf=is_lora):
        if isinstance(node, LoRAWrapper):
            for fname in node.adapter_fields:
                arr = getattr(node, fname, None)
                if eqx.is_array(arr):
                    adapter_ids.add(id(arr))
            for base_arr in jax.tree_util.tree_leaves(node.base):
                if eqx.is_array(base_arr):
                    wrapped_base_ids.add(id(base_arr))

    flat, treedef = jax.tree_util.tree_flatten(model)
    specs = []
    for leaf in flat:
        if not eqx.is_inexact_array(leaf):
            specs.append(False)
        elif id(leaf) in adapter_ids:
            specs.append(True)
        elif id(leaf) in wrapped_base_ids:
            specs.append(False)
        else:
            specs.append(True)  # unwrapped param — stays trainable
    return jax.tree_util.tree_unflatten(treedef, specs)


# =====================================================================
# LoRA Zoo — drop-in variants, all compatible with apply_lora / .lora()
# =====================================================================


def _linear_applies_to(leaf: Any) -> bool:
    """Shared predicate: any unwrapped LinearLike eqx.Module."""
    return isinstance(leaf, eqx.Module) and not isinstance(leaf, LoRAWrapper) and isinstance(leaf, LinearLike)


def _conv_applies_to(leaf: Any) -> bool:
    """Shared predicate: any unwrapped ConvLike eqx.Module (ConvTranspose excluded)."""
    return (
        isinstance(leaf, eqx.Module)
        and not isinstance(leaf, LoRAWrapper)
        and isinstance(leaf, ConvLike)
        and not isinstance(leaf, _CONV_TRANSPOSE_TYPES)
    )


class rsLoRALinear(LoRALinear):
    """Rank-Stabilized LoRA (rsLoRA).

    Replaces the ``alpha/rank`` scalar with ``alpha/sqrt(rank)``.  This keeps
    the effective gradient scale constant as rank grows, so larger ranks are
    actually useful rather than numerically unstable.

    Drop-in replacement for ``LoRALinear`` — identical API and parameter count.

    Reference: https://arxiv.org/abs/2312.03732
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T * (self.alpha / jnp.sqrt(self.rank))
        return y + delta

    def merge(self) -> eqx.Module:
        w = self.base.weight + (self.alpha / jnp.sqrt(self.rank)) * (self.lora_B @ self.lora_A)
        return eqx.tree_at(lambda m: m.weight, self.base, w)


class LoRAFALinear(LoRALinear):
    """Frozen-A LoRA (LoRAFA).

    Identical to ``LoRALinear`` but only ``lora_B`` is trained; ``lora_A``
    is frozen after random initialisation.  Halves the number of trainable
    adapter parameters at a small accuracy cost.

    Reference: https://arxiv.org/abs/2308.03303
    """

    adapter_fields = ("lora_B",)  # lora_A stays in pytree but is frozen


class DoRALinear(LoRAWrapper):
    """Weight-Decomposed Low-Rank Adaptation (DoRA).

    Decomposes the base weight into a magnitude vector and a unit-norm
    direction matrix.  Low-rank matrices update the direction; the per-output
    magnitude is a small additional trainable vector.

        W' = m * (W_0 + B @ A) / ||W_0 + B @ A||_row

    Assumes the base layer computes ``x @ weight.T (+ bias)``.

    Reference: https://arxiv.org/abs/2402.09353
    """

    adapter_fields = ("magnitude", "lora_A", "lora_B")

    base: LinearLike
    magnitude: jax.Array  # (out_features,) — learned per-output scale
    lora_A: jax.Array  # (rank, in_features)
    lora_B: jax.Array  # (out_features, rank)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        self.rank = min(rank, min(base.in_features, base.out_features))
        self.alpha = alpha
        # magnitude initialised to the row norms of the pretrained weight
        self.magnitude = jnp.linalg.norm(base.weight, axis=1)
        k1, _ = jax.random.split(key)
        std = 1.0 / jnp.sqrt(base.in_features)
        self.lora_A = jax.random.normal(k1, (self.rank, base.in_features)) * std
        self.lora_B = jnp.zeros((base.out_features, self.rank))

    def __call__(self, x: jax.Array) -> jax.Array:
        W = self.base.weight + (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        row_norms = jnp.linalg.norm(W, axis=1, keepdims=True)
        out = x @ (self.magnitude[:, None] * W / row_norms).T
        bias = getattr(self.base, "bias", None)
        return out + bias if bias is not None else out

    def merge(self) -> eqx.Module:
        W = self.base.weight + (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        row_norms = jnp.linalg.norm(W, axis=1, keepdims=True)
        W_dora = self.magnitude[:, None] * W / row_norms
        return eqx.tree_at(lambda m: m.weight, self.base, W_dora)


class PiSSALinear(LoRAWrapper):
    """Principal Singular Values and Singular Vectors Adaptation (PiSSA).

    Initialises the adapter from the top-r singular components of the
    pretrained weight.  The frozen base stores only the *residual*
    (the remaining singular components), so the model output is identical
    to the original at initialisation.  Adapters therefore start from the
    most important weight directions rather than random noise, leading to
    faster convergence when fine-tuning pretrained models.

    Reference: https://arxiv.org/abs/2404.02948
    """

    adapter_fields = ("lora_A", "lora_B")

    base: LinearLike  # holds the residual weight W - B@A at init
    lora_A: jax.Array  # (rank, in_features) — initialised from right singular vectors
    lora_B: jax.Array  # (out_features, rank) — initialised from left singular vectors
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        r = min(rank, min(base.in_features, base.out_features))
        self.rank = r
        self.alpha = alpha

        U, S, Vt = jnp.linalg.svd(base.weight, full_matrices=False)
        U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]

        # Scale so that (alpha/rank) * lora_B @ lora_A == U_r diag(S_r) Vt_r at init,
        # keeping model output identical to the original base.
        scale = jnp.sqrt(S_r * r / alpha)
        self.lora_B = U_r * scale[None, :]
        self.lora_A = Vt_r * scale[:, None]

        # Residual = W - (alpha/r)*B@A so that base + adapter == W at init.
        W_residual = base.weight - (alpha / r) * (self.lora_B @ self.lora_A)
        self.base = eqx.tree_at(lambda m: m.weight, base, W_residual)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T * (self.alpha / self.rank)
        return y + delta

    def merge(self) -> eqx.Module:
        w = self.base.weight + (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        return eqx.tree_at(lambda m: m.weight, self.base, w)


class LoRAXSLinear(LoRAWrapper):
    """LoRA-XS: extra-small LoRA with a trainable r×r core matrix.

    ``lora_A`` (r × in) and ``lora_B`` (out × r) are initialised from the
    top singular vectors of the pretrained weight and then **frozen**.  Only
    the small ``R`` matrix (r × r, initialised to zero) is trained:

        delta_W = B @ R @ A * (alpha / rank)

    This can give better parameter efficiency than vanilla LoRA for the same
    rank because ``R`` has r² parameters instead of r*(in+out).

    Reference: https://arxiv.org/abs/2405.17604
    """

    adapter_fields = ("R",)

    base: LinearLike
    lora_A: jax.Array  # (rank, in_features) — frozen, from SVD
    lora_B: jax.Array  # (out_features, rank) — frozen, from SVD
    R: jax.Array  # (rank, rank) — the only trainable parameter
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        r = min(rank, min(base.in_features, base.out_features))
        self.rank = r
        self.alpha = alpha
        self.base = base

        U, S, Vt = jnp.linalg.svd(base.weight, full_matrices=False)
        # A and B span the principal directions; R=0 ensures zero delta at init.
        self.lora_A = Vt[:r, :]
        self.lora_B = U[:, :r]
        self.R = jnp.zeros((r, r))

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        delta = (x @ self.lora_A.T @ self.R.T) @ self.lora_B.T * (self.alpha / self.rank)
        return y + delta

    def merge(self) -> eqx.Module:
        W = self.base.weight + (self.alpha / self.rank) * (self.lora_B @ self.R @ self.lora_A)
        return eqx.tree_at(lambda m: m.weight, self.base, W)


class VeRALinear(LoRAWrapper):
    """Vector-based Random Matrix Adaptation (VeRA).

    Replaces trainable A and B matrices with frozen random matrices that are
    generated on-the-fly from a stored integer seed and never materialised as
    Python/JAX arrays — JAX/XLA constant-folds their generation into the compiled
    kernel.  Only two small per-layer scaling vectors are trained:

    - ``b`` (out_features,) — scales each output row of B
    - ``d`` (rank,)         — scales each row of A

    Trainable params per layer: ``out + rank``  (vs ``r·(in + out)`` for LoRALinear).
    Checkpoints store only the seed and scaling vectors, not A or B.

    Layers with the same ``seed`` and the same shape share A, B at the XLA level
    via constant-expression deduplication (CSE), achieving true device-level sharing.

    Reference: https://arxiv.org/abs/2310.11454
    """

    adapter_fields = ("b", "d")

    base: LinearLike
    b: jax.Array  # (out_features,) trainable — scales B rows
    d: jax.Array  # (rank,)         trainable — scales A rows
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.in_features = base.in_features
        self.out_features = base.out_features
        # Derive a reproducible int seed; A,B are regenerated from it on every
        # forward pass so they are never stored as Python/JAX arrays.
        self.seed = int(jax.random.randint(key, shape=(), minval=0, maxval=2**30))
        self.b = jnp.zeros(base.out_features)
        self.d = jnp.ones(rank)

    def _frozen_AB(self) -> tuple[jax.Array, jax.Array]:
        A = jax.random.normal(jax.random.PRNGKey(self.seed), (self.rank, self.in_features)) / jnp.sqrt(self.in_features)
        B = jax.random.normal(jax.random.PRNGKey(self.seed + 1), (self.out_features, self.rank))
        return A, B

    def __call__(self, x: jax.Array) -> jax.Array:
        A, B = self._frozen_AB()
        # x @ A.T supports both 1-D (in,) and batched (batch, in) inputs.
        delta = (x @ A.T * self.d) @ B.T * self.b * (self.alpha / self.rank)
        return self.base(x) + delta

    def merge(self) -> eqx.Module:
        A, B = self._frozen_AB()
        # delta_W = diag(b) @ B @ diag(d) @ A * (alpha / rank)
        delta_W = (self.b[:, None] * B) @ (self.d[:, None] * A) * (self.alpha / self.rank)
        return eqx.tree_at(lambda m: m.weight, self.base, self.base.weight + delta_W)


class MiLoRALinear(LoRAWrapper):
    """Minor Singular Values and Singular Vectors Adaptation (MiLoRA).

    Like ``PiSSALinear`` but initialises the adapters from the *least* significant
    singular components of the pretrained weight (the "noise" subspace).  The frozen
    base retains the principal (high-energy) components, so fine-tuning only updates
    directions the base model treats as noise — preserving pretrained knowledge.

    Reference: https://arxiv.org/abs/2405.09913
    """

    adapter_fields = ("lora_A", "lora_B")

    base: LinearLike
    lora_A: jax.Array  # (rank, in_features) — minor right singular vectors
    lora_B: jax.Array  # (out_features, rank) — minor left singular vectors
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        r = min(rank, min(base.in_features, base.out_features))
        self.rank = r
        self.alpha = alpha

        U, S, Vt = jnp.linalg.svd(base.weight, full_matrices=False)
        # Take the MINOR (last r) singular components — the noise subspace.
        U_r, S_r, Vt_r = U[:, -r:], S[-r:], Vt[-r:, :]

        scale = jnp.sqrt(S_r * r / alpha)
        self.lora_B = U_r * scale[None, :]
        self.lora_A = Vt_r * scale[:, None]

        # Residual keeps principal components; adapter restores minor ones at init.
        W_residual = base.weight - (alpha / r) * (self.lora_B @ self.lora_A)
        self.base = eqx.tree_at(lambda m: m.weight, base, W_residual)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        delta = (x @ self.lora_A.T) @ self.lora_B.T * (self.alpha / self.rank)
        return y + delta

    def merge(self) -> eqx.Module:
        w = self.base.weight + (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        return eqx.tree_at(lambda m: m.weight, self.base, w)


class IA3Linear(LoRAWrapper):
    """Infused Adapter by Inhibiting and Amplifying Inner Activations (IA³).

    Applies a single per-output learned scale vector to the linear output
    (before adding bias).  No low-rank matrices — trainable params per layer
    is exactly ``out_features``.

    At initialisation ``scale = ones(out_features)`` so the output is
    unchanged.  ``rank`` is stored for API compatibility but unused.

        y = (x @ weight.T) * scale + bias

    Reference: https://arxiv.org/abs/2205.05638
    """

    adapter_fields = ("scale",)

    base: LinearLike
    scale: jax.Array  # (out_features,) — learned per-output multiplier
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = jnp.ones(base.out_features)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x @ self.base.weight.T
        y = y * self.scale
        bias = getattr(self.base, "bias", None)
        return y + bias if bias is not None else y

    def merge(self) -> eqx.Module:
        w = self.base.weight * self.scale[:, None]
        return eqx.tree_at(lambda m: m.weight, self.base, w)


class LoKrLinear(LoRAWrapper):
    """Kronecker-product Adapter (LoKr).

    Parameterises the weight delta as a Kronecker product of two small matrices::

        delta_W = kron(kron_A, kron_B)[:out_features, :in_features] * (alpha / rank)

    ``kron_A`` has shape ``(rank, rank)``; ``kron_B`` has shape
    ``(⌈out/rank⌉, ⌈in/rank⌉)``.  The Kronecker product always covers the
    target weight shape and is cropped.  ``kron_B`` is initialised to zero so
    the output is unchanged at init.

    Reference: https://arxiv.org/abs/2212.10650
    """

    adapter_fields = ("kron_A", "kron_B")

    base: LinearLike
    kron_A: jax.Array  # (rank, rank)
    kron_B: jax.Array  # (ceil(out/rank), ceil(in/rank))
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        r = min(rank, min(base.in_features, base.out_features))
        self.rank = r
        self.alpha = alpha
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.base = base

        m2 = _math.ceil(base.out_features / r)
        n2 = _math.ceil(base.in_features / r)
        self.kron_A = jax.random.normal(key, (r, r)) / jnp.sqrt(r)
        self.kron_B = jnp.zeros((m2, n2))

    def _delta_W(self) -> jax.Array:
        return jnp.kron(self.kron_A, self.kron_B)[: self.out_features, : self.in_features]

    def __call__(self, x: jax.Array) -> jax.Array:
        y = self.base(x)
        delta = x @ self._delta_W().T * (self.alpha / self.rank)
        return y + delta

    def merge(self) -> eqx.Module:
        W = self.base.weight + (self.alpha / self.rank) * self._delta_W()
        return eqx.tree_at(lambda m: m.weight, self.base, W)


class OFTLinear(LoRAWrapper):
    """Orthogonal Fine-Tuning (OFT).

    Fine-tunes the weight by multiplying with a block-diagonal orthogonal matrix
    ``R``.  Each (rank × rank) block is parameterised via the Cayley map of a
    learnable skew-symmetric matrix ``Q_i``::

        R_i = (I - Q_i)(I + Q_i)^{-1}   (orthogonal for any skew-symmetric Q_i)

    The full ``R`` is block-diagonal, assembled from ``n_blocks = ⌈out/rank⌉``
    blocks and cropped to ``(out_features, out_features)``.  At init ``Q = 0``
    so ``R = I`` and the model output is unchanged.

    ``alpha`` is stored for API compatibility; orthogonality is its own
    regularisation and no explicit scaling is applied.

    Reference: https://arxiv.org/abs/2306.07280
    """

    adapter_fields = ("Q_blocks",)

    base: LinearLike
    Q_blocks: jax.Array  # (n_blocks, rank, rank)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _linear_applies_to(leaf)

    def __init__(self, base: LinearLike, rank: int, alpha: float, *, key: jax.Array):
        r = min(rank, base.out_features)
        self.rank = r
        self.alpha = alpha
        self.out_features = base.out_features
        self.n_blocks = _math.ceil(base.out_features / r)
        self.base = base
        self.Q_blocks = jnp.zeros((self.n_blocks, r, r))

    def _R(self) -> jax.Array:
        eye = jnp.eye(self.rank)

        def cayley(Q: jax.Array) -> jax.Array:
            Qs = (Q - Q.T) / 2  # enforce skew-symmetry
            return (eye - Qs) @ jnp.linalg.inv(eye + Qs)

        blocks = jax.vmap(cayley)(self.Q_blocks)  # (n_blocks, rank, rank)
        # n_blocks is static so the list comprehension unrolls at trace time.
        R_padded = jax.scipy.linalg.block_diag(*[blocks[i] for i in range(self.n_blocks)])
        return R_padded[: self.out_features, : self.out_features]

    def __call__(self, x: jax.Array) -> jax.Array:
        W_new = self._R() @ self.base.weight
        y = x @ W_new.T
        bias = getattr(self.base, "bias", None)
        return y + bias if bias is not None else y

    def merge(self) -> eqx.Module:
        W_new = self._R() @ self.base.weight
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


# =====================================================================
# LoRA Zoo — Conv variants (LoRAConv through OFTConv)
#
# All conv adapters:
#   - target ``eqx.nn.Conv1d/2d/3d`` (ConvTranspose excluded)
#   - flatten weight to (out_channels, flat_in) for linear-algebra ops
#   - reconstruct via ``eqx.tree_at(lambda m: m.weight, base, W_new)(x)``
#     so XLA fuses the weight substitution into a single conv kernel call
#   - store ``weight_shape`` as a static field for shape-safe reshaping
# =====================================================================


class LoRAConv(LoRAWrapper):
    """Standard LoRA for convolutional layers.

    Flattens the weight tensor to ``(out_channels, flat_in)`` and applies
    a low-rank update identical to ``LoRALinear``:

        W' = W + (alpha/rank) * lora_B @ lora_A    (in flattened space)

    The updated weight is reshaped back and passed to the base conv via
    ``eqx.tree_at``, which XLA fuses into a single conv kernel call.

    ``flat_in = in_channels // groups * prod(kernel_size)``
    """

    adapter_fields = ("lora_A", "lora_B")

    base: ConvLike
    lora_A: jax.Array  # (rank, flat_in)
    lora_B: jax.Array  # (out_channels, rank)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        flat_in = _math.prod(base.weight.shape[1:])
        self.rank = min(rank, min(out_ch, flat_in))
        self.alpha = alpha
        k1, _ = jax.random.split(key)
        std = 1.0 / jnp.sqrt(flat_in)
        self.lora_A = jax.random.normal(k1, (self.rank, flat_in)) * std
        self.lora_B = jnp.zeros((out_ch, self.rank))

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = self.lora_B @ self.lora_A * (self.alpha / self.rank)
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = self.lora_B @ self.lora_A * (self.alpha / self.rank)
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class rsLoRAConv(LoRAConv):
    """Rank-Stabilized LoRA for conv layers.  Scales by ``alpha/sqrt(rank)``.

    Reference: https://arxiv.org/abs/2312.03732
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = self.lora_B @ self.lora_A * (self.alpha / jnp.sqrt(self.rank))
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = self.lora_B @ self.lora_A * (self.alpha / jnp.sqrt(self.rank))
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class LoRAFAConv(LoRAConv):
    """Frozen-A LoRA for conv layers.  Only ``lora_B`` is trained.

    Reference: https://arxiv.org/abs/2308.03303
    """

    adapter_fields = ("lora_B",)  # lora_A stays in pytree but is frozen


class DoRAConv(LoRAWrapper):
    """Weight-Decomposed Low-Rank Adaptation for conv layers.

    Operates on the flattened weight ``(out_channels, flat_in)`` exactly
    as ``DoRALinear`` does on a 2-D linear weight.

    Reference: https://arxiv.org/abs/2402.09353
    """

    adapter_fields = ("magnitude", "lora_A", "lora_B")

    base: ConvLike
    magnitude: jax.Array  # (out_channels,) — learned per-output-channel scale
    lora_A: jax.Array  # (rank, flat_in)
    lora_B: jax.Array  # (out_channels, rank)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        flat_in = _math.prod(base.weight.shape[1:])
        self.rank = min(rank, min(out_ch, flat_in))
        self.alpha = alpha
        self.base = base
        W_flat = base.weight.reshape(out_ch, flat_in)
        self.magnitude = jnp.linalg.norm(W_flat, axis=1)
        k1, _ = jax.random.split(key)
        std = 1.0 / jnp.sqrt(flat_in)
        self.lora_A = jax.random.normal(k1, (self.rank, flat_in)) * std
        self.lora_B = jnp.zeros((out_ch, self.rank))

    def _dora_weight_flat(self) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_flat = W_flat + (self.alpha / self.rank) * (self.lora_B @ self.lora_A)
        row_norms = jnp.linalg.norm(W_flat, axis=1, keepdims=True)
        return self.magnitude[:, None] * W_flat / row_norms

    def __call__(self, x: jax.Array) -> jax.Array:
        W_new = self._dora_weight_flat().reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        W_new = self._dora_weight_flat().reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class PiSSAConv(LoRAWrapper):
    """Principal Singular Values and Singular Vectors Adaptation for conv layers.

    SVD is computed on the flattened weight ``(out_channels, flat_in)``.
    Adapters start from the top-r singular components; base holds the residual.

    Reference: https://arxiv.org/abs/2404.02948
    """

    adapter_fields = ("lora_A", "lora_B")

    base: ConvLike
    lora_A: jax.Array  # (rank, flat_in)
    lora_B: jax.Array  # (out_channels, rank)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        flat_in = _math.prod(base.weight.shape[1:])
        r = min(rank, min(out_ch, flat_in))
        self.rank = r
        self.alpha = alpha

        W_flat = base.weight.reshape(out_ch, flat_in)
        U, S, Vt = jnp.linalg.svd(W_flat, full_matrices=False)
        U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]

        scale = jnp.sqrt(S_r * r / alpha)
        self.lora_B = U_r * scale[None, :]
        self.lora_A = Vt_r * scale[:, None]

        W_residual = (W_flat - (alpha / r) * (self.lora_B @ self.lora_A)).reshape(self.weight_shape)
        self.base = eqx.tree_at(lambda m: m.weight, base, W_residual)

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (W_flat + self.lora_B @ self.lora_A * (self.alpha / self.rank)).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (W_flat + self.lora_B @ self.lora_A * (self.alpha / self.rank)).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class LoRAXSConv(LoRAWrapper):
    """LoRA-XS extra-small adapter for conv layers.

    ``lora_A`` and ``lora_B`` are frozen (from SVD of flattened weight).
    Only the small ``R`` (rank × rank) matrix is trained.

    Reference: https://arxiv.org/abs/2405.17604
    """

    adapter_fields = ("R",)

    base: ConvLike
    lora_A: jax.Array  # (rank, flat_in) — frozen, from SVD
    lora_B: jax.Array  # (out_channels, rank) — frozen, from SVD
    R: jax.Array  # (rank, rank) — the only trainable parameter
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        flat_in = _math.prod(base.weight.shape[1:])
        r = min(rank, min(out_ch, flat_in))
        self.rank = r
        self.alpha = alpha
        self.base = base

        W_flat = base.weight.reshape(out_ch, flat_in)
        U, _, Vt = jnp.linalg.svd(W_flat, full_matrices=False)
        self.lora_A = Vt[:r, :]
        self.lora_B = U[:, :r]
        self.R = jnp.zeros((r, r))

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = self.lora_B @ self.R @ self.lora_A * (self.alpha / self.rank)
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = self.lora_B @ self.R @ self.lora_A * (self.alpha / self.rank)
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class VeRAConv(LoRAWrapper):
    """Vector-based Random Matrix Adaptation for conv layers.

    A and B are generated on-the-fly from a stored seed (XLA constants).
    Only per-channel ``b`` (out_channels,) and ``d`` (rank,) are trained.

    Reference: https://arxiv.org/abs/2310.11454
    """

    adapter_fields = ("b", "d")

    base: ConvLike
    b: jax.Array  # (out_channels,) — trainable output-channel scale
    d: jax.Array  # (rank,)         — trainable row scale for A
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        self.rank = rank
        self.alpha = alpha
        self.seed = int(jax.random.randint(key, shape=(), minval=0, maxval=2**30))
        self.b = jnp.zeros(out_ch)
        self.d = jnp.ones(rank)

    def _frozen_AB(self) -> tuple[jax.Array, jax.Array]:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        A = jax.random.normal(jax.random.PRNGKey(self.seed), (self.rank, flat_in)) / jnp.sqrt(flat_in)
        B = jax.random.normal(jax.random.PRNGKey(self.seed + 1), (out_ch, self.rank))
        return A, B

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        A, B = self._frozen_AB()
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = (self.b[:, None] * B) @ (self.d[:, None] * A) * (self.alpha / self.rank)
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        A, B = self._frozen_AB()
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        delta = (self.b[:, None] * B) @ (self.d[:, None] * A) * (self.alpha / self.rank)
        W_new = (W_flat + delta).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class MiLoRAConv(LoRAWrapper):
    """Minor Singular Values and Singular Vectors Adaptation for conv layers.

    Like ``PiSSAConv`` but adapts the minor (least-significant) singular
    components; base retains the principal directions.

    Reference: https://arxiv.org/abs/2405.09913
    """

    adapter_fields = ("lora_A", "lora_B")

    base: ConvLike
    lora_A: jax.Array  # (rank, flat_in) — minor right singular vectors
    lora_B: jax.Array  # (out_channels, rank) — minor left singular vectors
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        flat_in = _math.prod(base.weight.shape[1:])
        r = min(rank, min(out_ch, flat_in))
        self.rank = r
        self.alpha = alpha

        W_flat = base.weight.reshape(out_ch, flat_in)
        U, S, Vt = jnp.linalg.svd(W_flat, full_matrices=False)
        U_r, S_r, Vt_r = U[:, -r:], S[-r:], Vt[-r:, :]

        scale = jnp.sqrt(S_r * r / alpha)
        self.lora_B = U_r * scale[None, :]
        self.lora_A = Vt_r * scale[:, None]

        W_residual = (W_flat - (alpha / r) * (self.lora_B @ self.lora_A)).reshape(self.weight_shape)
        self.base = eqx.tree_at(lambda m: m.weight, base, W_residual)

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (W_flat + self.lora_B @ self.lora_A * (self.alpha / self.rank)).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (W_flat + self.lora_B @ self.lora_A * (self.alpha / self.rank)).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class IA3Conv(LoRAWrapper):
    """IA³ output-scaling adapter for conv layers.

    Applies a learned per-output-channel scale to the conv weight before
    the convolution.  No low-rank matrices — trainable params = ``out_channels``.

    Reference: https://arxiv.org/abs/2205.05638
    """

    adapter_fields = ("scale",)

    base: ConvLike
    scale: jax.Array  # (out_channels,) — learned per-channel multiplier
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.base = base
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        self.rank = rank
        self.alpha = alpha
        self.scale = jnp.ones(out_ch)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Broadcast scale over all non-output axes: (out_ch, 1, 1, ...)
        scale_shape = (self.weight_shape[0],) + (1,) * (len(self.weight_shape) - 1)
        W_new = self.base.weight * self.scale.reshape(scale_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        scale_shape = (self.weight_shape[0],) + (1,) * (len(self.weight_shape) - 1)
        W_new = self.base.weight * self.scale.reshape(scale_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class LoKrConv(LoRAWrapper):
    """Kronecker-product adapter for conv layers.

    The weight delta is a Kronecker product of two small matrices, cropped to
    ``(out_channels, flat_in)`` and reshaped to the original weight shape.

    Reference: https://arxiv.org/abs/2212.10650
    """

    adapter_fields = ("kron_A", "kron_B")

    base: ConvLike
    kron_A: jax.Array  # (rank, rank)
    kron_B: jax.Array  # (ceil(out_ch/rank), ceil(flat_in/rank))
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        flat_in = _math.prod(base.weight.shape[1:])
        r = min(rank, min(out_ch, flat_in))
        self.rank = r
        self.alpha = alpha
        self.base = base

        m2 = _math.ceil(out_ch / r)
        n2 = _math.ceil(flat_in / r)
        self.kron_A = jax.random.normal(key, (r, r)) / jnp.sqrt(r)
        self.kron_B = jnp.zeros((m2, n2))

    def _delta_W_flat(self) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        return jnp.kron(self.kron_A, self.kron_B)[:out_ch, :flat_in]

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (W_flat + self._delta_W_flat() * (self.alpha / self.rank)).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (W_flat + self._delta_W_flat() * (self.alpha / self.rank)).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)


class OFTConv(LoRAWrapper):
    """Orthogonal Fine-Tuning for conv layers.

    The block-diagonal Cayley map is applied to the output-channel axis of
    the flattened weight ``(out_channels, flat_in)``, identical to
    ``OFTLinear`` on a 2-D weight.

    Reference: https://arxiv.org/abs/2306.07280
    """

    adapter_fields = ("Q_blocks",)

    base: ConvLike
    Q_blocks: jax.Array  # (n_blocks, rank, rank)
    rank: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)
    weight_shape: tuple = eqx.field(static=True)

    @classmethod
    def applies_to(cls, leaf: Any) -> bool:
        return _conv_applies_to(leaf)

    def __init__(self, base: ConvLike, rank: int, alpha: float, *, key: jax.Array):
        self.weight_shape = tuple(base.weight.shape)
        out_ch = base.weight.shape[0]
        r = min(rank, out_ch)
        self.rank = r
        self.alpha = alpha
        self.n_blocks = _math.ceil(out_ch / r)
        self.base = base
        self.Q_blocks = jnp.zeros((self.n_blocks, r, r))

    def _R(self) -> jax.Array:
        out_ch = self.weight_shape[0]
        eye = jnp.eye(self.rank)

        def cayley(Q: jax.Array) -> jax.Array:
            Qs = (Q - Q.T) / 2
            return (eye - Qs) @ jnp.linalg.inv(eye + Qs)

        blocks = jax.vmap(cayley)(self.Q_blocks)
        R_padded = jax.scipy.linalg.block_diag(*[blocks[i] for i in range(self.n_blocks)])
        return R_padded[:out_ch, :out_ch]

    def __call__(self, x: jax.Array) -> jax.Array:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (self._R() @ W_flat).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)(x)

    def merge(self) -> eqx.Module:
        out_ch = self.weight_shape[0]
        flat_in = _math.prod(self.weight_shape[1:])
        W_flat = self.base.weight.reshape(out_ch, flat_in)
        W_new = (self._R() @ W_flat).reshape(self.weight_shape)
        return eqx.tree_at(lambda m: m.weight, self.base, W_new)
