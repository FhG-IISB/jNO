"""Builder functions for JIT-compilable explainability analysis."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


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


def make_expression_eval_fn(
    expr,
    all_ops,
    batchsize,
    frozen,
    static,
    min_consecutive: int = 1,
):
    """Build a function that evaluates any placeholder expression during training.

    Compiles a user-supplied jno placeholder expression — for example
    ``u.d(x)`` for an input gradient (saliency), ``Jacobian(u, [x, y])``
    for a multi-variable spatial Jacobian, or any composite expression —
    using the same :class:`~jno.trace_compiler.TraceCompiler` pathway that
    the solver uses for constraints and trackers (Sec. 3, [1312.6034]).

    Returns a JIT-friendly callable::

        f(trainable, context, rng) -> values

    where ``values`` is the array (or pytree) returned by evaluating
    ``expr`` against the current parameters and the collocation points
    bound to ``expr``'s :class:`~jno.trace.Variable` references.

    Args:
        expr: Any :class:`~jno.trace.Placeholder` expression.
        all_ops: List of :class:`~jno.trace.OperationDef` (the solver's
            ``self.all_ops``).  Forwarded to ``compile_multi_expression``.
        batchsize: Mini-batch size (``None`` for full-batch).
        frozen: Frozen parameter pytree (from ``eqx.partition``).
        static: Static (non-array) pytree.
        min_consecutive: Forwarded to the compiled expression.
    """
    from jno.trace_compiler import TraceCompiler

    compiled_fn = TraceCompiler.compile_multi_expression([expr], all_ops)

    def eval_expr(trainable, context, rng):
        full_models = eqx.combine(trainable, frozen, static)
        results = compiled_fn(
            full_models,
            context,
            batchsize=batchsize,
            key=rng,
            min_consecutive=min_consecutive,
        )
        return results[0]

    return eval_expr


def make_ntk_spectrum_fn(
    grad_expr,
    all_ops,
    batchsize,
    frozen,
    static,
    n_points: int = 256,
    top_k: int = 10,
    min_consecutive: int = 1,
):
    """Build a function that returns the empirical NTK eigenvalue spectrum.

    Compiles a :class:`~jno.trace.NetworkGradient` placeholder (e.g.
    ``u.grad(net)`` or, for a parameter subset, ``u.grad(net.mask(mask))``)
    to produce a per-point parameter Jacobian
    :math:`J \\in \\mathbb{R}^{N \\times P}` and forms the empirical Neural
    Tangent Kernel :math:`K = J J^\\top \\in \\mathbb{R}^{N \\times N}`.

    The returned closure subsamples ``n_points`` rows (without replacement)
    from the full collocation grid, builds :math:`K`, and reports its
    eigenvalue spectrum.  This is the canonical diagnostic for the PINN
    spectral-bias problem (Sec. 3-4, [2007.14527]) — a wide eigenvalue
    spread indicates that the network is learning some directions vastly
    faster than others.

    Returns a JIT-friendly callable::

        f(trainable, context, rng) ->
            (eigvals_topk, lambda_min, lambda_max, condition, all_eigvals)

    where eigenvalues are returned in descending order, ``condition`` is
    :math:`\\lambda_{\\max} / \\lambda_{\\min}`, and ``all_eigvals`` has
    length ``n_points``.

    Args:
        grad_expr: A :class:`~jno.trace.NetworkGradient` placeholder, e.g.
            ``u.grad(net)``.  Use ``net.mask(...)`` to restrict to a
            parameter subset — masking lives in the placeholder itself
            rather than in a separate callback argument.
        all_ops: Solver's ``self.all_ops``.
        batchsize: Mini-batch size (``None`` for full-batch).
        frozen: Frozen parameter pytree (from ``eqx.partition``).
        static: Static (non-array) pytree.
        n_points: Number of collocation points to subsample for the kernel.
            Cost is ``O(n_points² × P)``; default ``256``.
        top_k: Number of largest eigenvalues to report.  Default ``10``.
        min_consecutive: Forwarded to the compiled expression.

    Note:
        The point subsample uses a fixed seed (``jax.random.PRNGKey(0)``)
        so the *same* rows are selected at every call — this makes the
        recorded eigenvalue spectrum directly comparable across epochs.
    """
    from jno.trace_compiler import TraceCompiler

    compiled_fn = TraceCompiler.compile_multi_expression([grad_expr], all_ops)
    sample_key_const = jax.random.PRNGKey(0)  # closed over below

    def spectrum(trainable, context, rng):
        full_models = eqx.combine(trainable, frozen, static)
        results = compiled_fn(
            full_models,
            context,
            batchsize=batchsize,
            key=rng,
            min_consecutive=min_consecutive,
        )
        # NetworkGradient compiles to shape (B, N, P) for scalar output and
        # (B, N, D, P) for D-dim output.  We require scalar output: silently
        # mixing the D output components into the row axis would produce a
        # confidently-wrong kernel.
        J_full = results[0]
        if J_full.ndim != 3:
            raise ValueError(
                "ntk_spectrum expects a scalar-output NetworkGradient (shape "
                f"(B, N, P)); got J with shape {J_full.shape}.  For a "
                "vector-valued network, project to a scalar first — e.g. "
                "u[..., 0].grad(net)."
            )
        J = J_full.reshape(-1, J_full.shape[-1])  # (N_total, P)

        N_total = J.shape[0]
        n_take = min(n_points, N_total)
        idx = jax.random.choice(sample_key_const, N_total, shape=(n_take,), replace=False)
        J_sub = J[idx]  # (n_take, P)

        K = J_sub @ J_sub.T  # (n_take, n_take)
        eigvals_asc = jnp.linalg.eigvalsh(K)
        eigvals_desc = eigvals_asc[::-1]
        top = eigvals_desc[:top_k]
        lambda_max = eigvals_desc[0]
        lambda_min = eigvals_desc[-1]
        condition = lambda_max / (jnp.abs(lambda_min) + 1e-12)
        return top, lambda_min, lambda_max, condition, eigvals_desc

    return spectrum


def make_hessian_spectrum_fn(
    compiled_constraints_fn,
    batchsize,
    frozen,
    static,
    param_mask=None,
    min_consecutive: int = 1,
    k: int = 10,
    n_iter: int = 30,
    constraint_indices=None,
):
    """Build a function that returns the top-k Hessian eigenvalues via Lanczos.

    Constructs the Hessian-vector product (HVP) of the total training
    loss with respect to the (optionally masked) trainable parameters via
    ``jvp(grad(L), …, …)``, runs ``n_iter`` Lanczos iterations with full
    reorthogonalisation, and returns the top-k eigenvalues of the
    Lanczos tridiagonal — which approximate the top-k eigenvalues of the
    full Hessian (Sec. 3.1-3.2, [1912.07145]).  The largest of these is
    the *sharpness* of [Keskar et al., Sec. 2.2, 1609.04836]: a high
    value predicts that the optimiser is sitting in a sharp minimum,
    typically associated with worse generalisation.

    Returns a Python driver function (the inner HVP is JIT-compiled;
    the Lanczos loop runs in Python and the tridiagonal is eigen-
    decomposed host-side via ``scipy.linalg.eigh_tridiagonal``)::

        f(trainable, context, rng) -> (eigvals_topk, lambda_max, all_eigvals)

    where eigenvalues are returned in descending order.

    Args:
        compiled_constraints_fn: Combined compiled JAX function for all constraints.
        batchsize: Mini-batch size (``None`` for full-batch).
        frozen: Frozen parameter pytree.
        static: Static (non-array) pytree.
        param_mask: Optional pytree of booleans matching *trainable*; when
            set, the Hessian is computed only on the selected subset.
            Essential for large models.
        min_consecutive: Forwarded to ``compiled_constraints_fn``.
        k: Number of top eigenvalues to report.  Default ``10``.
        n_iter: Number of Lanczos iterations.  Default ``30``.
        constraint_indices: Optional tuple of integer indices into the
            solver's compiled constraints.  When set, the Hessian is taken
            over the mean of *only* those constraint losses rather than
            the full training loss — useful for diagnosing which
            constraint drives the sharpness.  Default ``None`` (full loss).
    """
    _selected_idx = jnp.array(constraint_indices) if constraint_indices is not None else None

    def _total_loss(selected, held_fixed, context, rng):
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
        losses = jnp.stack([jnp.mean(r) for r in residuals])
        if _selected_idx is not None:
            losses = losses[_selected_idx]
        return jnp.mean(losses)

    @jax.jit
    def _hvp_pytree(selected, held_fixed, context, rng, v_pytree):
        grad_fn = jax.grad(lambda s: _total_loss(s, held_fixed, context, rng))
        _, hv = jax.jvp(grad_fn, (selected,), (v_pytree,))
        return hv

    # Fixed seed for the Lanczos start vector — see PR-4 commit for rationale.
    start_key_const = jax.random.PRNGKey(0)

    def spectrum(trainable, context, rng):
        if param_mask is not None:
            selected, held_fixed = eqx.partition(trainable, param_mask)
        else:
            selected, held_fixed = trainable, None

        flat_v0, unravel = jax.flatten_util.ravel_pytree(selected)
        n = flat_v0.size

        def matvec(v_flat):
            v_pytree = unravel(v_flat)
            hv_pytree = _hvp_pytree(selected, held_fixed, context, rng, v_pytree)
            hv_flat, _ = jax.flatten_util.ravel_pytree(hv_pytree)
            return hv_flat

        # Lanczos with full reorthogonalisation — Python loop, JIT'd matvec.
        v = jax.random.normal(start_key_const, (n,))
        v = v / (jnp.linalg.norm(v) + 1e-30)
        Q = [v]
        alphas: list = []
        betas: list = []
        for j in range(n_iter):
            w = matvec(Q[-1])
            alpha = jnp.dot(Q[-1], w)
            alphas.append(alpha)
            w = w - alpha * Q[-1]
            if j > 0:
                w = w - betas[-1] * Q[-2]
            for q in Q[:-1]:
                w = w - jnp.dot(q, w) * q
            beta = jnp.linalg.norm(w)
            Q.append(w / (beta + 1e-30))
            betas.append(beta)

        from scipy.linalg import eigh_tridiagonal

        alphas_h = np.array([float(a) for a in alphas], dtype=np.float64)
        betas_h = np.array([float(b) for b in betas], dtype=np.float64)
        eigvals = eigh_tridiagonal(alphas_h, betas_h[:-1], eigvals_only=True)
        eigvals_desc = np.sort(eigvals)[::-1].astype(np.float32)
        top = eigvals_desc[:k]
        lambda_max = float(eigvals_desc[0])
        return top, lambda_max, eigvals_desc

    return spectrum


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
