# Vendored from SOAP_JAX (https://github.com/haydn-jones/SOAP_JAX) -- MIT License, (c) 2024 Haydn Jones.
# Only the chex / jaxtyping type-hint imports were stripped (to avoid extra deps); the algorithm is unchanged.
#
# SOAP: Improving and Stabilizing Shampoo using Adam. Vyas, Morwani, Zhao, Shapira, Brandfonbrener,
# Janson & Kakade (2024), arXiv:2409.11321.
"""SOAP optimizer -- Shampoo with Adam in the preconditioner's eigenbasis (optax-compatible)."""

from itertools import chain
from typing import Any, Callable, Iterable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from jax import Array
from optax import GradientTransformation, Updates

Numeric = Any  # was chex.Numeric (type hint only)
PreconditionerMatrix = Union[Array, None]


@jtu.register_pytree_node_class
class Preconditioner:
    """Per-parameter preconditioner matrices (None for dimensions exceeding max_precond_dim)."""

    __slots__ = ("matrices",)

    def __init__(self, matrices: Iterable[PreconditionerMatrix]):
        self.matrices = tuple(matrices)

    def tree_flatten(self) -> tuple[tuple[PreconditionerMatrix, ...], None]:
        return (self.matrices, None)

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple[PreconditionerMatrix, ...]) -> "Preconditioner":
        return cls(children)

    def map(self, fn: Callable[[PreconditionerMatrix], PreconditionerMatrix]) -> "Preconditioner":
        return Preconditioner(fn(m) for m in self.matrices)


class SOAPState(NamedTuple):
    count: Array
    exp_avg: Updates
    exp_avg_sq: Updates
    GG: Updates  # Pytree of Preconditioner
    Q: Updates  # Pytree of Preconditioners


class PostDecayState(NamedTuple):
    count: Array


def soap(
    learning_rate: optax.ScalarOrSchedule = 3e-3,
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    correct_bias: bool = True,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype: Optional[Any] = None,
    qr_dtype: Any = jnp.float32,
) -> optax.GradientTransformationExtraArgs:
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321). Based on the original implementation at https://github.com/nikhilvyas/SOAP.

    Args:
        learning_rate (optax.ScalarOrSchedule): The learning rate to use.
        b1 (float, optional): Adam's beta1 parameter. Defaults to 0.95.
        b2 (float, optional): Adam's beta2 parameter. Defaults to 0.95.
        shampoo_beta (float, optional): If >= 0, use this beta for the preconditioner (`L` and `R` in paper, `GG` below)
            moving average instead of b2. Defaults to -1.
        eps (float, optional): Adam's epsilon for numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay coefficient. Defaults to 0.0.
        correct_bias (bool, optional): Whether to use bias correction for the Adam moments. Defaults to True.
        precondition_frequency (int, optional): How often to update the preconditioner. Defaults to 10.
        max_precond_dim (int, optional): Maximum dimension of the preconditioner.
            Set to 10000 to exclude most common vocab sizes while including layers. Defaults to 10000.
        precondition_1d (bool, optional): Whether to precondition 1D gradients. If False, 1D params use Adam-style
            updates. Defaults to False.
        precision (jax.lax.PrecisionLike, optional): Precision to use. Defaults to jax.lax.Precision.HIGHEST.
        mu_dtype (Any, optional): dtype for the first and second moment estimates (exp_avg and exp_avg_sq).
            If None, uses the same dtype as the parameters. Useful for mixed-precision training. Defaults to None.
        qr_dtype (Any, optional): dtype used for eigen/QR computations and preconditioner storage.
            Defaults to float32 to avoid float64 upcasts in mixed precision.

    Returns:
        optax.GradientTransformationExtraArgs: The SOAP optimizer.
    """
    return optax.chain(
        scale_by_soap(
            b1=b1,
            b2=b2,
            shampoo_beta=shampoo_beta,
            eps=eps,
            correct_bias=correct_bias,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            precondition_1d=precondition_1d,
            precision=precision,
            mu_dtype=mu_dtype,
            qr_dtype=qr_dtype,
        ),
        optax.scale_by_learning_rate(learning_rate),
        add_decayed_weights_post(weight_decay, learning_rate),
    )


def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    correct_bias: bool = True,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype: Optional[Any] = None,
    qr_dtype: Any = jnp.float32,
) -> GradientTransformation:
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321). Based on the original implementation at https://github.com/nikhilvyas/SOAP.

    Args:
        b1 (float, optional): Adam's beta1 parameter. Defaults to 0.95.
        b2 (float, optional): Adam's beta2 parameter. Defaults to 0.95.
        shampoo_beta (float, optional): If >= 0, use this beta for the preconditioner (`L` and `R` in paper, `GG` below)
            moving average instead of b2. Defaults to -1.
        eps (float, optional): Adam's epsilon for numerical stability. Defaults to 1e-8.
        correct_bias (bool, optional): Whether to use bias correction for the Adam moments. Defaults to True.
        precondition_frequency (int, optional): How often to update the preconditioner. Defaults to 10.
        max_precond_dim (int, optional): Maximum dimension of the preconditioner.
            Set to 10000 to exclude most common vocab sizes while including layers. Defaults to 10000.
        precondition_1d (bool, optional): Whether to precondition 1D gradients. If False, 1D params use Adam-style
            updates. Defaults to False.
        precision (jax.lax.PrecisionLike, optional): Precision to use. Defaults to jax.lax.Precision.HIGHEST.
        mu_dtype (Any, optional): dtype for the first and second moment estimates (exp_avg and exp_avg_sq).
            If None, uses the same dtype as the parameters. Useful for mixed-precision training. Defaults to None.
        qr_dtype (Any, optional): dtype used for eigen/QR computations and preconditioner storage.
            Defaults to float32 to avoid float64 upcasts in mixed precision.

    Returns:
        GradientTransformation: The SOAP gradient transformation.
    """
    if not (0 <= b1 < 1):
        raise ValueError("b1 must be in [0, 1)")
    if not (0 <= b2 < 1):
        raise ValueError("b2 must be in [0, 1)")
    if shampoo_beta >= 1 or (shampoo_beta < 0 and shampoo_beta != -1):
        raise ValueError("shampoo_beta must be in [0, 1) or -1")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if precondition_frequency <= 0:
        raise ValueError("precondition_frequency must be a positive integer")
    if max_precond_dim <= 0:
        raise ValueError("max_precond_dim must be a positive integer")

    shampoo_beta = shampoo_beta if shampoo_beta >= 0 else b2

    def init_fn(params: Updates) -> SOAPState:
        exp_avg = otu.tree_zeros_like(params, dtype=mu_dtype)
        exp_avg_sq = otu.tree_zeros_like(params, dtype=mu_dtype)
        GG = jtu.tree_map(
            lambda p: init_conditioner(p, max_precond_dim, precondition_1d, qr_dtype),
            params,
        )
        Q = jtu.tree_map(
            lambda p: init_conditioner(p, max_precond_dim, precondition_1d, qr_dtype),
            params,
        )
        return SOAPState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=GG,
            Q=Q,
        )

    def init_step(
        updates: Updates,
        state: SOAPState,
    ) -> tuple[Updates, SOAPState]:
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta, precision),
            updates,
            state.GG,
            is_leaf=_is_preconditioner,
        )

        new_Q = jtu.tree_map(
            lambda gg: gg.map(lambda m: get_orthogonal_matrix(m, qr_dtype)),
            new_GG,
            is_leaf=_is_preconditioner,
        )

        # Replace updates with zeros
        new_updates = otu.tree_zeros_like(updates)

        return new_updates, state._replace(GG=new_GG, Q=new_Q)

    def update_step(
        updates: Updates,
        state: SOAPState,
    ) -> tuple[Updates, SOAPState]:
        # Project gradients
        grad_projected = jtu.tree_map(
            lambda grad, q: project(grad, q, precision),
            updates,
            state.Q,
            is_leaf=_is_preconditioner,
        )

        # Update moments
        exp_avg = otu.tree_update_moment(grad_projected, state.exp_avg, b1, 1)
        exp_avg_sq = otu.tree_update_moment_per_elem_norm(grad_projected, state.exp_avg_sq, b2, 2)
        if mu_dtype is not None:
            exp_avg = otu.tree_cast(exp_avg, mu_dtype)
            exp_avg_sq = otu.tree_cast(exp_avg_sq, mu_dtype)

        # Project back
        norm_updates = jtu.tree_map(
            lambda e_avg, e_avg_sq, q: project_back(e_avg / (jnp.sqrt(e_avg_sq) + eps), q, precision),
            exp_avg,
            exp_avg_sq,
            state.Q,
            is_leaf=_is_preconditioner,
        )

        if correct_bias:
            # Bias correction: use (count - 1) because count=1 is the init step with no moment updates
            effective_step = state.count - 1
            bc1 = 1 - b1**effective_step
            bc2 = 1 - b2**effective_step
            corr = jnp.sqrt(bc2) / bc1

            # Bias correction on the updates
            norm_updates = jtu.tree_map(
                lambda p: p * corr,
                norm_updates,
            )

        # Update the preconditioner
        new_GG = jtu.tree_map(
            lambda grad, gg: update_preconditioner(grad, gg, shampoo_beta, precision),
            updates,
            state.GG,
            is_leaf=_is_preconditioner,
        )

        # Update the orthogonal matrix / exp_avg_sq
        def refresh_preconditioner() -> tuple[Updates, Updates, Updates]:
            new_Q_and_exp_avg_sq = jtu.tree_map(
                lambda e, gg, q: get_orthogonal_matrix_QR(gg, q, e, precision, qr_dtype),
                exp_avg_sq,
                new_GG,
                state.Q,
                is_leaf=_is_preconditioner,
            )
            new_Q = jtu.tree_map(
                lambda _, x: x[0],
                updates,
                new_Q_and_exp_avg_sq,
            )
            new_exp_avg_sq = jtu.tree_map(
                lambda _, x: x[1],
                updates,
                new_Q_and_exp_avg_sq,
            )
            new_exp_avg = jtu.tree_map(
                lambda e, old_q, new_q: project(project_back(e, old_q, precision), new_q, precision),
                exp_avg,
                state.Q,
                new_Q,
                is_leaf=_is_preconditioner,
            )
            return new_Q, new_exp_avg_sq, new_exp_avg

        def keep_preconditioner() -> tuple[Updates, Updates, Updates]:
            return state.Q, exp_avg_sq, exp_avg

        new_Q, exp_avg_sq, exp_avg = jax.lax.cond(
            (state.count - 1) % precondition_frequency == 0,
            refresh_preconditioner,
            keep_preconditioner,
        )

        new_state = SOAPState(
            count=state.count,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=new_GG,
            Q=new_Q,
        )

        return norm_updates, new_state

    def update_fn(updates: Updates, state: SOAPState, params: Optional[Updates] = None) -> tuple[Updates, SOAPState]:
        del params
        count_inc = jnp.asarray(optax.safe_int32_increment(state.count))
        state = state._replace(count=count_inc)

        updates, new_state = jax.lax.cond(
            count_inc == 1,
            lambda: init_step(updates, state),
            lambda: update_step(updates, state),
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def add_decayed_weights_post(
    weight_decay: float,
    learning_rate: optax.ScalarOrSchedule,
) -> GradientTransformation:
    if weight_decay == 0.0:
        return optax.identity()

    def init_fn(params: Updates) -> PostDecayState:
        del params
        return PostDecayState(count=jnp.zeros([], jnp.int32))

    def update_fn(
        updates: Updates, state: PostDecayState, params: Optional[Updates] = None
    ) -> tuple[Updates, PostDecayState]:
        if params is None:
            raise ValueError("add_decayed_weights_post requires parameters.")

        count_inc = jnp.asarray(optax.safe_int32_increment(state.count))
        lr = _resolve_learning_rate(learning_rate, count_inc)
        decay = lr * weight_decay
        decay = jnp.where(count_inc == 1, jnp.zeros_like(decay), decay)
        updates = jtu.tree_map(lambda u, p: u - decay * p, updates, params)
        return updates, state._replace(count=count_inc)

    return optax.GradientTransformation(init_fn, update_fn)  # ty:ignore[invalid-argument-type]


def update_preconditioner(
    grad: Array,
    GG: Preconditioner,
    beta: float,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Preconditioner:
    if grad.ndim == 1:
        if GG.matrices[0] is None:
            return GG
        return Preconditioner(
            [lerp(GG.matrices[0], jnp.matmul(grad[:, None], grad[None, :], precision=precision), 1 - beta)]
        )

    new_GG = []
    for idx, gg in enumerate(GG.matrices):
        if gg is None:
            new_GG.append(None)
            continue

        outer_product = jnp.tensordot(
            grad,
            grad,
            axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
            precision=precision,
        )
        new_GG.append(lerp(gg, outer_product, 1 - beta))

    return Preconditioner(new_GG)


def project(
    grad: Array,
    Q: Preconditioner,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q.matrices:
        if mat is not None:
            grad = jnp.tensordot(
                grad,
                mat,
                axes=((0,), (0,)),
                precision=precision,
            )
        else:
            permute_order = list(range(1, len(grad.shape))) + [0]
            grad = jnp.transpose(grad, permute_order)

    return grad


def project_back(
    grad: Array,
    Q: Preconditioner,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q.matrices:
        if mat is not None:
            grad = jnp.tensordot(
                grad,
                mat,
                axes=((0,), (1,)),
                precision=precision,
            )
        else:
            grad = jnp.moveaxis(grad, 0, -1)

    return grad


def get_orthogonal_matrix(gg: Union[Array, None], qr_dtype: Any) -> Union[Array, None]:
    if gg is None:
        return None

    gg_mat = gg.astype(qr_dtype) if gg.dtype != qr_dtype else gg
    jitter = jnp.asarray(1e-30, dtype=qr_dtype)
    _, eigh = jnp.linalg.eigh(gg_mat + jitter * jnp.eye(gg_mat.shape[0], dtype=qr_dtype))
    q = jnp.flip(eigh, axis=1)
    if q.dtype != gg.dtype:
        q = q.astype(gg.dtype)
    return q


def get_orthogonal_matrix_QR(
    GG: Preconditioner,
    Q: Preconditioner,
    exp_avg_sq: Array,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    qr_dtype: Any = jnp.float32,
) -> tuple[Preconditioner, Array]:
    final_Q = []
    for ind, (m, o) in enumerate(zip(GG.matrices, Q.matrices)):
        if m is None or o is None:
            final_Q.append(None)
            continue

        m_mat = m.astype(qr_dtype) if m.dtype != qr_dtype else m
        o_mat = o.astype(qr_dtype) if o.dtype != qr_dtype else o
        est_eig = jnp.diag(
            jnp.matmul(
                jnp.matmul(o_mat.T, m_mat, precision=precision),
                o_mat,
                precision=precision,
            )
        )
        sort_idx = jnp.argsort(est_eig, descending=True)
        exp_avg_sq = jnp.take(exp_avg_sq, sort_idx, axis=ind)
        o_mat = o_mat[:, sort_idx]
        power_iter = jnp.matmul(m_mat, o_mat, precision=precision)
        Q_new, _ = jnp.linalg.qr(power_iter)

        if Q_new.dtype != m.dtype:
            Q_new = Q_new.astype(m.dtype)
        final_Q.append(Q_new)

    return Preconditioner(final_Q), exp_avg_sq


def lerp(
    start: Array,
    end: Array,
    weight: Numeric,
) -> Array:
    return start + weight * (end - start)


def init_conditioner(
    p: Array,
    max_precond_dim: int,
    precondition_1d: bool,
    dtype: Any,
) -> Preconditioner:
    if p.ndim == 1:
        if not precondition_1d or p.shape[0] > max_precond_dim:
            return Preconditioner([None])
        return Preconditioner([jnp.zeros((p.shape[0], p.shape[0]), dtype=dtype)])

    return Preconditioner([jnp.zeros((s, s), dtype=dtype) if s <= max_precond_dim else None for s in p.shape])


def _is_preconditioner(value: object) -> bool:
    return isinstance(value, Preconditioner)


def _resolve_learning_rate(learning_rate: optax.ScalarOrSchedule, count: Array) -> Array:
    if callable(learning_rate):
        return learning_rate(count)  # ty:ignore[invalid-return-type]

    return jnp.asarray(learning_rate)
