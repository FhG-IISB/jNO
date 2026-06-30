"""Magnitude–Direction (MD) Decoupling — a generic optimizer wrapper.

Hägele, Hernández-Cano, Kosson & Jaggi, *Improving Neural Network Training by Decoupling
the Magnitude and Direction of Weight Vectors* (2026), arXiv:2606.25971 — Algorithm 2
(general per-row/per-column gains).

Each 2-D weight matrix ``W`` of shape ``(d_out, d_in)`` is reparameterized as

    W = diag(γ_row) · Ŵ · diag(γ_col)

where the direction ``Ŵ`` is held on a fixed-norm (Frobenius) sphere — its norm pinned to the
value ``c = ‖W_init‖_F`` it had at initialization — and the positive gains
``γ = softplus(γ̃)`` are *learnable magnitudes* stepped at their own rate. The split lives
entirely inside the optimizer: the model only ever sees the single fused tensor ``W``. The method
is **agnostic to the base optimizer** — any optax ``GradientTransformation`` steps the direction.

Per step, for each 2-D leaf (``G = ∂L/∂W``):

    1. γ_row = softplus(γ̃_row), γ_col = softplus(γ̃_col)
    2. Ŵ      = W / γ_row[:,None] / γ_col[None,:]            (recover the on-sphere direction)
    3. g_γrow = (Ŵ⊙G) @ γ_col,  g_γcol = γ_row @ (Ŵ⊙G)       (gain gradients)
    4. g_γ̃   = g_γ ⊙ sigmoid(γ̃)                             (through the softplus map)
    5. Ĝ_Ŵ   = γ_row[:,None] · G · γ_col[None,:]             (direction gradient ∂L/∂Ŵ)
    6. Ŵ      ← Ŵ + base.update(Ĝ_Ŵ)                         (any base optimizer; its LR is η_W)
    7. Ŵ      ← Ŵ / ‖Ŵ‖_F · c                                (project back onto the sphere)
    8. γ̃      ← γ̃ + adam(gain_lr).update(g_γ̃)               (step the raw gains; LR η_γ)
    9. W_new  = softplus(γ̃_row)[:,None] · Ŵ · softplus(γ̃_col)[None,:]   (reassemble)

and the optimizer returns ``updates = W_new − W``. Non-2-D leaves (biases, 1-D norm gains,
scalars) pass straight through the base optimizer unmodified.

Because ``W_new − W`` is **nonlinear** in the weights, it must not be re-scaled by a learning
rate. Prefer the :func:`md` sentinel (``net.optimizer(jno.optimizers.md(...))``), which makes
``jno.core`` neutralize the outer LR-scale automatically. The bare :func:`md_decouple`
``GradientTransformation`` is exported for plain optax loops / power users — there you must keep
the surrounding learning rate at ``1.0``.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

# softplus⁻¹(1) — the raw gain whose softplus is exactly 1, so γ=1 (and Ŵ=W) at initialization
# and step 0 is the identity (no jump onto the sphere). Computed in pure-Python double precision so
# its accuracy never depends on JAX's x64 flag at import time (a float32 value here would leave
# softplus(γ̃)≈1±1e-7, perturbing every init).
_SOFTPLUS_INV_ONE = math.log(math.expm1(1.0))


class MDDecoupleState(NamedTuple):
    """State for :func:`md_decouple`. Gains and sphere norms live here, never in the params."""

    base_state: Any  # base optimizer state over the full pytree (direction Ŵ on 2-D leaves)
    gain_state: Any  # optax.adam(gain_lr) state over the raw-gain pytrees (raw_row, raw_col)
    raw_row: Any  # pytree mirroring params: raw row gain on 2-D leaves, () placeholder elsewhere
    raw_col: Any  # pytree mirroring params: raw col gain on 2-D leaves, () placeholder elsewhere
    sphere_c: Any  # pytree mirroring params: ‖W_init‖_F on 2-D leaves, () placeholder elsewhere
    count: jax.Array


def _parse_gain_axis(gain_axis: Any) -> tuple[bool, bool]:
    """Return (use_row, use_col) for a ``gain_axis`` spec. ``"scalar"`` is not yet supported."""
    if isinstance(gain_axis, str):
        if gain_axis == "scalar":
            raise NotImplementedError(
                "gain_axis='scalar' (a single tied scalar gain, Algorithm 1) is not yet "
                "implemented. Use ('row', 'col') (default), ('row',), or ('col',)."
            )
        gain_axis = (gain_axis,)
    axes = tuple(gain_axis)
    unknown = set(axes) - {"row", "col"}
    if unknown:
        raise ValueError(f"gain_axis entries must be 'row' and/or 'col'; got {sorted(unknown)}.")
    if not axes:
        raise ValueError("gain_axis must enable at least one of 'row' / 'col'.")
    return "row" in axes, "col" in axes


def md_decouple(
    base_optimizer: optax.GradientTransformation,
    *,
    gain_lr: float = 1e-3,
    gain_axis: Any = ("row", "col"),
    gain_map: str = "softplus",
) -> optax.GradientTransformationExtraArgs:
    """Wrap ``base_optimizer`` with Magnitude–Direction Decoupling (Hägele et al. 2026, Alg. 2).

    Args:
        base_optimizer: any optax ``GradientTransformation`` — it steps the on-sphere direction
            ``Ŵ`` and carries the direction learning rate η_W (e.g. ``optax.adam(1e-3)``).
        gain_lr: learning rate η_γ of the separate ``optax.adam`` that steps the magnitude gains.
        gain_axis: which gains are learned — ``("row", "col")`` (default, the paper's combined
            per-row-and-column gain), ``("row",)``, or ``("col",)``. A disabled axis is held at
            γ=1. ``"scalar"`` is not yet implemented.
        gain_map: the positive map φ applied to the raw gains. Only ``"softplus"`` in v1.

    Returns:
        An optax ``GradientTransformationExtraArgs``. The returned update is ``W_new − W`` on 2-D
        weight leaves (nonlinear — keep the surrounding LR at 1.0) and the plain base step on all
        other leaves. Prefer the :func:`md` sentinel so the host neutralizes the outer LR-scale.

    References:
        Hägele, Hernández-Cano, Kosson & Jaggi, *Improving Neural Network Training by Decoupling
        the Magnitude and Direction of Weight Vectors*, 2026, arXiv:2606.25971, Algorithm 2.
    """
    if gain_map != "softplus":
        raise NotImplementedError(f"gain_map={gain_map!r} not implemented; only 'softplus' in v1.")
    use_row, use_col = _parse_gain_axis(gain_axis)

    base = optax.with_extra_args_support(base_optimizer)
    gain_opt = optax.adam(gain_lr)

    phi = jax.nn.softplus  # γ = softplus(γ̃)
    dphi = jax.nn.sigmoid  # d/dγ̃ softplus(γ̃) = sigmoid(γ̃)

    def _is_md(x):
        return isinstance(x, jax.Array) and x.ndim == 2

    def _gamma(raw, n, active):
        """Positive gain vector of length ``n``: ``softplus(raw)`` if learned, else all-ones."""
        if active:
            return phi(raw)
        return jnp.ones((n,), dtype=raw.dtype)

    def _direction(W, raw_row, raw_col):
        if not _is_md(W):
            return W
        gr = _gamma(raw_row, W.shape[0], use_row)
        gc = _gamma(raw_col, W.shape[1], use_col)
        return W / gr[:, None] / gc[None, :]

    def _dir_grad(W, G, raw_row, raw_col):
        if not _is_md(W):
            return G
        gr = _gamma(raw_row, W.shape[0], use_row)
        gc = _gamma(raw_col, W.shape[1], use_col)
        return gr[:, None] * G * gc[None, :]

    def _graw_row(W, G, raw_row, raw_col):
        if not _is_md(W) or not use_row:
            return jnp.zeros_like(raw_row)
        what = _direction(W, raw_row, raw_col)
        gc = _gamma(raw_col, W.shape[1], use_col)
        g_gamma = (what * G) @ gc  # reduce over columns -> (d_out,)
        return g_gamma * dphi(raw_row)

    def _graw_col(W, G, raw_row, raw_col):
        if not _is_md(W) or not use_col:
            return jnp.zeros_like(raw_col)
        what = _direction(W, raw_row, raw_col)
        gr = _gamma(raw_row, W.shape[0], use_row)
        g_gamma = gr @ (what * G)  # reduce over rows -> (d_in,)
        return g_gamma * dphi(raw_col)

    def _reassemble(W, what, base_update, raw_row, raw_col, c):
        if not _is_md(W):
            return base_update  # pass-through: biases / 1-D / scalars
        what_new = what + base_update
        what_new = what_new / (jnp.linalg.norm(what_new) + 1e-30) * c  # project onto the sphere
        gr = _gamma(raw_row, W.shape[0], use_row)
        gc = _gamma(raw_col, W.shape[1], use_col)
        w_new = gr[:, None] * what_new * gc[None, :]
        return w_new - W

    def init_fn(params):
        def _rr(W):
            return jnp.full((W.shape[0],), _SOFTPLUS_INV_ONE, dtype=W.dtype) if _is_md(W) else jnp.zeros((), W.dtype)

        def _rc(W):
            return jnp.full((W.shape[1],), _SOFTPLUS_INV_ONE, dtype=W.dtype) if _is_md(W) else jnp.zeros((), W.dtype)

        def _c(W):
            return jnp.linalg.norm(W) if _is_md(W) else jnp.zeros((), W.dtype)

        raw_row = jax.tree.map(_rr, params)
        raw_col = jax.tree.map(_rc, params)
        sphere_c = jax.tree.map(_c, params)
        what0 = jax.tree.map(_direction, params, raw_row, raw_col)  # = params at init (γ=1)
        return MDDecoupleState(
            base_state=base.init(what0),
            gain_state=gain_opt.init((raw_row, raw_col)),
            raw_row=raw_row,
            raw_col=raw_col,
            sphere_c=sphere_c,
            count=jnp.zeros([], jnp.int32),
        )

    def update_fn(updates, state, params=None, **extra_args):
        if params is None:
            raise ValueError("md_decouple requires params (the fused weights) in update().")
        grads = updates
        rr, rc = state.raw_row, state.raw_col

        what = jax.tree.map(_direction, params, rr, rc)
        mod_grads = jax.tree.map(_dir_grad, params, grads, rr, rc)
        g_raw_row = jax.tree.map(_graw_row, params, grads, rr, rc)
        g_raw_col = jax.tree.map(_graw_col, params, grads, rr, rc)

        # Direction sub-step: any base optimizer steps Ŵ (extra args forwarded for completeness).
        base_updates, new_base_state = base.update(mod_grads, state.base_state, what, **extra_args)

        # Magnitude sub-step: a separate Adam steps the raw gains at η_γ.
        gain_updates, new_gain_state = gain_opt.update((g_raw_row, g_raw_col), state.gain_state, (rr, rc))
        rr_new, rc_new = optax.apply_updates((rr, rc), gain_updates)

        new_updates = jax.tree.map(_reassemble, params, what, base_updates, rr_new, rc_new, state.sphere_c)
        new_state = MDDecoupleState(
            base_state=new_base_state,
            gain_state=new_gain_state,
            raw_row=rr_new,
            raw_col=rc_new,
            sphere_c=state.sphere_c,
            count=state.count + 1,
        )
        return new_updates, new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


# ---------------------------------------------------------------------------------------------
# Sentinel — the recommended user-facing entry point.
# ---------------------------------------------------------------------------------------------
#
# Not yet implemented (v1 — md_decouple / md cover Algorithm 2 with row+col Frobenius gains):
#   * gain_axis="scalar" — a single tied scalar gain per matrix (Algorithm 1).
#   * Conv / >2-D weight kernels — they pass through the base optimizer unchanged (no decoupling).
#   * Embedding / LM-head per-row unit-L2-norm special casing.
#   * The Muon RMS-matching update-scale factor sqrt(max(d_out/d_in, d_in/d_out)).
#   * Gain reparameterizations other than softplus (gain_map is a hook, only "softplus" wired up).
#   * Line-search base optimizers (ssbroyden / ssbfgs / engd): their value_fn closes over the full
#     fused W, not the direction Ŵ, so a line search on the sub-problem is semantically wrong —
#     use a pure-gradient base (adam / sgd / soap / muon).
#   * Driving the direction LR η_W from the model's .scale() schedule — in v1 η_W is carried by the
#     base optimizer you pass, and the sentinel forces the model's outer LR-scale to 1.0.
class MDOptimizer:
    """Sentinel stored on a model via ``net.optimizer(jno.optimizers.md(...))``.

    Not an optax ``GradientTransformation`` — ``jno.core.solve()`` detects this object, builds the
    real :func:`md_decouple` transform from it via :meth:`build`, and forces the model's outer
    learning-rate scale to ``1.0`` so the nonlinear ``W_new − W`` update is applied intact. The
    direction learning rate η_W is carried by the ``base`` optimizer you pass.

    See :func:`md` for the public constructor and the list of not-yet-implemented features above
    this class.
    """

    def __init__(
        self,
        base: optax.GradientTransformation,
        *,
        gain_lr: float = 1e-3,
        gain_axis: Any = ("row", "col"),
        gain_map: str = "softplus",
    ):
        if not isinstance(base, optax.GradientTransformation):
            raise TypeError(
                "md(base, ...) needs an optax GradientTransformation as the direction optimizer "
                f"(e.g. optax.adam(1e-3)); got {type(base).__name__}. Note: line-search optimizers "
                "(jno.optimizers.ssbroyden/ssbfgs/engd) are not supported as the MD base."
            )
        _parse_gain_axis(gain_axis)  # validate eagerly
        self._base = base
        self._gain_lr = gain_lr
        self._gain_axis = gain_axis
        self._gain_map = gain_map

    def build(self) -> optax.GradientTransformationExtraArgs:
        """Construct the real :func:`md_decouple` transform this sentinel stands for."""
        return md_decouple(
            self._base,
            gain_lr=self._gain_lr,
            gain_axis=self._gain_axis,
            gain_map=self._gain_map,
        )

    def __repr__(self):
        return f"MDOptimizer(gain_lr={self._gain_lr}, gain_axis={self._gain_axis!r})"


def md(
    base: optax.GradientTransformation,
    *,
    gain_lr: float = 1e-3,
    gain_axis: Any = ("row", "col"),
    gain_map: str = "softplus",
) -> MDOptimizer:
    """Return a Magnitude–Direction Decoupling sentinel for ``net.optimizer()``.

    Magnitude–Direction (MD) Decoupling (Hägele, Hernández-Cano, Kosson & Jaggi, 2026,
    arXiv:2606.25971, Algorithm 2) factorizes each weight matrix ``W = diag(γ_row) Ŵ diag(γ_col)``
    into a fixed-Frobenius-norm direction ``Ŵ`` on a hypersphere and learnable per-row/per-column
    magnitude gains, updated at separate rates. It removes the need for weight decay and warmup and
    transfers the optimal learning rate across model width. The split lives entirely inside the
    optimizer — the model still sees one fused weight tensor.

    Unlike the bare :func:`md_decouple` transform, this sentinel form lets ``jno.core`` force the
    model's outer learning-rate scale to ``1.0`` automatically, so the nonlinear ``W_new − W``
    update is never corrupted — you do **not** need a separate ``net.scale(1.0)`` call. The
    direction learning rate η_W is the LR of the ``base`` optimizer you pass.

    Args:
        base: the direction optimizer — any pure-gradient optax ``GradientTransformation`` carrying
            η_W (e.g. ``optax.adam(1e-3)``). Line-search bases are not supported (see the module).
        gain_lr: learning rate η_γ of the separate Adam that steps the magnitude gains.
        gain_axis: ``("row", "col")`` (default), ``("row",)``, or ``("col",)``. ``"scalar"`` is not
            yet implemented.
        gain_map: positive map for the gains; only ``"softplus"`` in v1.

    Example::

        import optax, jno
        net.optimizer(jno.optimizers.md(optax.adam(1e-3), gain_lr=1e-3))
        crux = jno.core(losses)
        crux.solve(2000)

    References:
        Hägele, Hernández-Cano, Kosson & Jaggi, *Improving Neural Network Training by Decoupling
        the Magnitude and Direction of Weight Vectors*, 2026, arXiv:2606.25971, Algorithm 2.
    """
    return MDOptimizer(base, gain_lr=gain_lr, gain_axis=gain_axis, gain_map=gain_map)
