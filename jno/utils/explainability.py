"""Builder functions for JIT-compilable explainability analysis."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def make_residual_stats_fn(
    compiled_constraints_fn,
    n_constraints: int,
    batchsize,
    frozen,
    static,
    min_consecutive: int = 1,
):
    """Build a function that summarises per-constraint residual distributions.

    The compiled constraints function returns one ``(B, T, ...)`` residual
    array per constraint *before* the training loss applies ``jnp.mean``.
    This builder wraps that call and reduces each array to four scalar
    statistics — exposing *where* in the domain the PDE is poorly satisfied
    (Sec. 3, [2207.10289]).

    Returns a JIT-friendly callable::

        f(trainable, context, rng) -> (means, stds, maxes, p99, raw)

    where
      means  : float32 (n_constraints,)  — per-constraint residual mean
      stds   : float32 (n_constraints,)  — per-constraint residual std
      maxes  : float32 (n_constraints,)  — per-constraint residual max
      p99    : float32 (n_constraints,)  — 99th-percentile residual
      raw    : tuple of 1-D arrays, one per constraint, flattened — for
               host-side histogram logging

    Args:
        compiled_constraints_fn: Combined compiled JAX function for all constraints.
        n_constraints: Number of constraint terms (kept for symmetry with the
            other builders; not used directly here but documents expected output).
        batchsize: Mini-batch size (``None`` for full-batch).
        frozen: Frozen parameter pytree (from ``eqx.partition``).
        static: Static (non-array) pytree.
        min_consecutive: Forwarded to ``compiled_constraints_fn``.
    """
    del n_constraints  # documented for symmetry only; output length matches the call

    def stats(trainable, context, rng):
        full_models = eqx.combine(trainable, frozen, static)
        residuals = compiled_constraints_fn(
            full_models,
            context,
            batchsize=batchsize,
            key=rng,
            min_consecutive=min_consecutive,
        )
        means = jnp.stack([jnp.mean(r) for r in residuals])
        stds = jnp.stack([jnp.std(r) for r in residuals])
        maxes = jnp.stack([jnp.max(r) for r in residuals])
        p99 = jnp.stack([jnp.percentile(r, 99.0) for r in residuals])
        raw = tuple(r.ravel() for r in residuals)
        return means, stds, maxes, p99, raw

    return stats


def make_per_loss_grad_fn(
    compiled_constraints_fn,
    n_constraints: int,
    batchsize,
    frozen,
    static,
    param_mask=None,
    min_consecutive: int = 1,
):
    """Build a function that computes per-loss gradient statistics.

    Returns a JIT-friendly callable::

        f(trainable, context, rng) -> (norms, cos_matrix, total_alignment)

    where
      norms           : float32 (n_constraints,)  — gradient norm per loss term
      cos_matrix      : float32 (n_constraints, n_constraints)  — pairwise cosine sim
      total_alignment : float32 scalar  — ||Σgᵢ|| / Σ||gᵢ||  (Eq. 3.1, [2502.00604])

    Args:
        compiled_constraints_fn: Combined compiled JAX function for all constraints.
        n_constraints: Number of constraint terms.
        batchsize: Mini-batch size (None for full-batch).
        frozen: Frozen parameter pytree (from eqx.partition).
        static: Static (non-array) pytree.
        param_mask: Optional pytree of booleans matching *trainable* structure.
            When set, only the selected subset of parameters is differentiated,
            which dramatically reduces cost for large models.
        min_consecutive: Forwarded to compiled_constraints_fn.
    """

    def _eval_losses(selected, held_fixed, context, rng):
        if held_fixed is not None:
            trainable_full = eqx.combine(selected, held_fixed)
        else:
            trainable_full = selected
        full_models = eqx.combine(trainable_full, frozen, static)
        residuals = compiled_constraints_fn(
            full_models,
            context,
            batchsize=batchsize,
            key=rng,
            min_consecutive=min_consecutive,
        )
        return jnp.stack([jnp.mean(r) for r in residuals])  # (N,)

    def grad_alignment(trainable, context, rng):
        _, step_rng = jax.random.split(rng)

        if param_mask is not None:
            selected, held_fixed = eqx.partition(trainable, param_mask)
        else:
            selected, held_fixed = trainable, None

        # jacrev → pytree matching `selected`; array leaves gain shape (N, *leaf_shape).
        # Equinox sentinels at frozen/non-selected positions are empty pytrees,
        # so jax.tree_util.tree_leaves skips them — only gradient arrays appear.
        per_grads = jax.jacrev(lambda s: _eval_losses(s, held_fixed, context, step_rng))(selected)

        leaves = jax.tree_util.tree_leaves(per_grads)  # list of (N, *leaf_shape) arrays

        # Build G: (N, P_mask) by slicing row i from each leaf.
        # Python loop over range(N) is unrolled at trace time.
        G_rows = []
        for i in range(n_constraints):
            g_i = jnp.concatenate([leaf[i].ravel() for leaf in leaves])
            G_rows.append(g_i)
        G = jnp.stack(G_rows)  # (N, P_mask)

        norms = jnp.linalg.norm(G, axis=1)  # (N,)
        G_hat = G / (norms[:, None] + 1e-12)
        cos_matrix = G_hat @ G_hat.T  # (N, N)

        g_sum = jnp.sum(G, axis=0)
        total_alignment = jnp.linalg.norm(g_sum) / (jnp.sum(norms) + 1e-12)

        return norms, cos_matrix, total_alignment

    return grad_alignment


def make_landscape_fn(
    compiled_constraints_fn,
    batchsize,
    frozen,
    static,
    n_grid: int = 15,
    alpha_range: float = 1.0,
    param_mask=None,
    min_consecutive: int = 1,
):
    """Build a function that evaluates the 2-D loss landscape.

    Returns a JIT-friendly callable::

        f(trainable, context, rng) -> float32 (n_grid, n_grid)

    Two random directions in parameter space are sampled and normalized to
    the scale of the selected parameters (global filter normalization).  The
    total loss is evaluated on an (n_grid × n_grid) perturbation grid around
    the current parameter values.

    Args:
        compiled_constraints_fn: Combined compiled JAX function for all constraints.
        batchsize: Mini-batch size (None for full-batch).
        frozen: Frozen parameter pytree.
        static: Static (non-array) pytree.
        n_grid: Number of grid points per axis.  Total evaluations = n_grid².
        alpha_range: Perturbation range in units of parameter norm.
        param_mask: Optional pytree of booleans.  When set, only selected
            parameters are perturbed; the rest are held at current values.
        min_consecutive: Forwarded to compiled_constraints_fn.
    """

    def landscape(trainable, context, rng):
        rng, k1, k2 = jax.random.split(rng, 3)

        if param_mask is not None:
            selected, held_fixed = eqx.partition(trainable, param_mask)
        else:
            selected, held_fixed = trainable, None

        flat_params, unravel = jax.flatten_util.ravel_pytree(selected)
        pnorm = jnp.linalg.norm(flat_params) + 1e-12

        delta = jax.random.normal(k1, flat_params.shape)
        eta = jax.random.normal(k2, flat_params.shape)
        delta = delta / (jnp.linalg.norm(delta) + 1e-12) * pnorm
        eta = eta / (jnp.linalg.norm(eta) + 1e-12) * pnorm

        alphas = jnp.linspace(-alpha_range, alpha_range, n_grid)
        betas = jnp.linspace(-alpha_range, alpha_range, n_grid)

        def eval_at(alpha, beta):
            sel = unravel(flat_params + alpha * delta + beta * eta)
            if held_fixed is not None:
                trainable_full = eqx.combine(sel, held_fixed)
            else:
                trainable_full = sel
            full_models = eqx.combine(trainable_full, frozen, static)
            residuals = compiled_constraints_fn(
                full_models,
                context,
                batchsize=batchsize,
                key=rng,
                min_consecutive=min_consecutive,
            )
            losses = jnp.stack([jnp.mean(r) for r in residuals])
            return jnp.mean(losses)

        # lax.map over outer axis → peak memory = n_grid × model (not n_grid² × model)
        return jax.lax.map(
            lambda alpha: jax.vmap(lambda beta: eval_at(alpha, beta))(betas),
            alphas,
        )

    return landscape
