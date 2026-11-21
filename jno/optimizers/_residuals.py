"""The stacked, weighted residual vector of a PINN and its parameter Jacobian, for the
Gauss-Newton family of optimizers (``lm``, ``dngd``, ``nystrom_ngd``, ``dsgnar``).

Based on:
  Müller, J., & Zeinhofer, M. (2023).
  Achieving High Accuracy with PINNs via Energy Natural Gradient Descent.
  ICML 2023, arXiv:2302.13163, Sec. 3 -- the energy Gram matrix
  :math:`G = \\sum_k (w_k/N_k)\\, J_k^\\top J_k` that :func:`jno.optimizers.engd` builds.
  Jnini, A., & Vella, F. (2025).
  Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks.
  arXiv:2505.21404, Sec. 2.2 -- the residual classes stacked as :math:`r_k/\\sqrt{N_k}`.

Each residual class :math:`k` (a ``(NetworkGradient, weight)`` pair of ``gram_terms``, whose
``.target`` is the raw per-point residual) contributes the rows

.. math::

    \\hat r_k = \\sqrt{w_k / N_k}\\; r_k(\\theta) \\in \\mathbb{R}^{N_k},

so that :math:`\\hat J^\\top \\hat J` is exactly ENGD's Gram matrix and
:math:`\\|\\hat r\\|^2 = \\sum_k w_k\\,\\operatorname{mean}(r_k^2)` is the weighted sum of the
``.mse`` losses. The functions here only ever differentiate this one vector, so the residual, its
Jacobian and its Jacobian-vector products always agree with each other.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp


def check_gram_terms(gram_terms, who: str) -> int:
    """Validate ``(NetworkGradient, weight)`` pairs and return the one model's layer id."""
    from jno.trace import NetworkGradient

    if not gram_terms:
        raise ValueError(f"{who}: gram_terms must not be empty.")
    for i, term in enumerate(gram_terms):
        expr, w = term
        if not isinstance(expr, NetworkGradient):
            raise TypeError(
                f"{who}: gram_terms[{i}][0] must be a NetworkGradient placeholder (e.g. "
                f"residual.grad(model)), got {type(expr).__name__!r}. Build one via ``expr.grad(model)``."
            )
        if not float(w) > 0.0:
            raise ValueError(f"{who}: gram_terms[{i}] has weight {w!r}; weights must be > 0.")
    lids = [expr.model_node.layer_id for expr, _ in gram_terms]
    if len(set(lids)) != 1:
        raise ValueError(
            f"{who}: all gram_terms must reference the same model (found layer_ids {lids}). Attach one optimizer per model."
        )
    return lids[0]


def make_residual_fn(gram_terms, all_ops, frozen, static, lid: int, unravel, min_consecutive: int = 1):
    """Build ``residual_fn(flat_p, trainable, context, rng) -> r̂`` (shape ``(m,)``).

    ``flat_p`` is the flattened parameter vector of model ``lid`` (the one being optimised); every
    other model keeps its value in ``trainable``. Full batch: every context point enters ``r̂``.
    """
    import equinox as eqx
    import paramax as _paramax

    from jno.trace_compiler import TraceCompiler

    compiled = [(TraceCompiler.compile_multi_expression([expr.target], all_ops), float(w)) for expr, w in gram_terms]

    def residual_fn(flat_p, trainable, context, rng):
        tr = {**trainable, lid: unravel(flat_p)}
        full = _paramax.unwrap(eqx.combine(tr, frozen, static))
        parts = []
        for comp_fn, w in compiled:
            r = comp_fn(full, context, batchsize=None, key=rng, min_consecutive=min_consecutive)[0]
            r = jnp.reshape(r, (-1,))
            parts.append(jnp.sqrt(w / r.shape[0]) * r)
        return jnp.concatenate(parts)

    return residual_fn


class Linearized(NamedTuple):
    """The residual at a point and its Jacobian as two matrix-free maps."""

    r: jax.Array  # (m,)
    jvp: Callable  # v (P,) -> J v (m,)
    vjp: Callable  # u (m,) -> J^T u (P,)


def linearize(r_of: Callable, flat_p: jax.Array) -> Linearized:
    """Linearize ``r_of`` at ``flat_p``: one forward pass, then ``J v`` and ``Jᵀ u`` at the cost of a JVP each."""
    r, jvp = jax.linearize(r_of, flat_p)
    vjp_t = jax.linear_transpose(jvp, flat_p)
    return Linearized(r=r, jvp=jvp, vjp=lambda u: vjp_t(u)[0])


def jacobian(r_of: Callable, flat_p: jax.Array, m: int) -> jax.Array:
    """Dense ``J = ∂r̂/∂θ`` (``m x P``), forward mode when there are fewer parameters than rows."""
    P = flat_p.shape[0]
    return jax.jacfwd(r_of)(flat_p) if P <= m else jax.jacrev(r_of)(flat_p)


def second_directional(r_of: Callable, flat_p: jax.Array, v: jax.Array) -> jax.Array:
    """:math:`\\frac{d^2}{dt^2} r(\\theta + t v)|_{t=0}`, by two nested JVPs (geodesic acceleration)."""
    return jax.jvp(lambda q: jax.jvp(r_of, (q,), (v,))[1], (flat_p,), (v,))[1]
