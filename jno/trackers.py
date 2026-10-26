"""Live training-loop trackers for adaptive components and diagnostics.

The objects produced by this module are the same callback classes exposed
through :class:`jno.callbacks` for explainability, but surfaced under a
namespace that matches their conceptual role: *trackers* whose latest value
is read by other in-loop components (e.g. NTK-balanced loss weighting,
gradient-norm-balanced loss weighting).

Each tracker is registered via the solver's ``callbacks=[...]`` list and
exposes two attributes on top of the existing ``cb.result`` history:

* :attr:`tracker.value <jno.utils.adaptive.callbacks._LiveValue.value>` —
  the most recent computed value as a numpy ``dict`` (``None`` until the
  first interval fires).
* :attr:`tracker.latest_epoch <jno.utils.adaptive.callbacks._LiveValue.latest_epoch>`
  — epoch index when :attr:`value` was last updated.

Note:
    :func:`jno.tracker` (singular) is unrelated: it wraps a single
    :class:`~jno.trace.Placeholder` expression and stores it in
    ``logs["track_stats"]`` (scalar → 2-D float array; non-scalar →
    ``list[list[np.ndarray]]``). The explainability trackers here cannot use
    that path because they depend on the compiled-constraints function,
    partition pytrees, masks, and host-side reductions (Lanczos, eigh).

    See :mod:`jno.utils.adaptive.weights` for adaptive weight schemes
    (``NTKBalanced``, ``GradientNormBalanced``) that consume these trackers.
"""

from __future__ import annotations

from typing import Any, Optional

from .utils.adaptive.callbacks import (
    CosSimilarityCallback,
    GradientAlignmentCallback,
    GradientNormsCallback,
    HessianSpectrumCallback,
    InputSensitivityCallback,
    LossLandscapeCallback,
    NTKSpectrumCallback,
    ResidualStatsCallback,
)


# docstring-from: jno.utils.adaptive.callbacks.GradientNormsCallback
def gradient_norms(interval: int = 100, mask=None) -> GradientNormsCallback:
    """Track the gradient norm of each loss term during training.

    At every ``interval`` outer steps, computes ``‖∇L_i‖₂`` for each
    constraint *i* by differentiating through the constraint function.

    Args:
        interval: Compute every *n* outer training steps.  Default ``100``.
        mask: Optional pytree of booleans matching ``crux.models`` structure.
            When set, gradients are computed only for the selected parameter
            subset, reducing cost for large models.

    Example::

        cb = jno.callbacks.gradient_norms(interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["norms"])   # (n_samples, n_constraints)
    """
    return GradientNormsCallback(interval=interval, mask=mask)


# docstring-from: jno.utils.adaptive.callbacks.CosSimilarityCallback
def cos_similarity(interval: int = 100, mask=None) -> CosSimilarityCallback:
    """Track pairwise gradient cosine similarity between loss terms.

    Computes the full ``(N × N)`` cosine similarity matrix (upper triangle
    carries the pairwise values; diagonal is always 1).

    Args:
        interval: Compute every *n* outer training steps.  Default ``100``.
        mask: Optional pytree of booleans — see :class:`GradientNormsCallback`.

    Example::

        cb = jno.callbacks.cos_similarity(interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["cos_sim_matrix"])   # (n_samples, N, N)
    """
    return CosSimilarityCallback(interval=interval, mask=mask)


# docstring-from: jno.utils.adaptive.callbacks.GradientAlignmentCallback
def gradient_alignment(interval: int = 100, mask=None) -> GradientAlignmentCallback:
    """Track the total gradient alignment scalar during training.

    Computes ``2‖(1/N) Σ ĝᵢ‖² − 1`` with ``ĝᵢ = gᵢ / ‖gᵢ‖`` (Eq. 3.1,
    [2502.00604]), a value in ``[-1, 1]`` that measures how well all loss
    gradients point in the same direction.  Near ``1`` means perfect
    alignment; ``0`` means orthogonal; near ``-1`` means anti-aligned
    (gradients actively cancel).  For ``N = 2`` this reduces to the
    ordinary cosine similarity (Proposition 1 of the paper).

    Args:
        interval: Compute every *n* outer training steps.  Default ``100``.
        mask: Optional pytree of booleans — see :class:`GradientNormsCallback`.

    Example::

        cb = jno.callbacks.gradient_alignment(interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["alignment"])   # (n_samples,)
    """
    return GradientAlignmentCallback(interval=interval, mask=mask)


# docstring-from: jno.utils.adaptive.callbacks.ResidualStatsCallback
def residual_stats(interval: int = 100, constraints=None) -> ResidualStatsCallback:
    """Track per-constraint residual distributions during training.

    At every ``interval`` outer steps, evaluates each constraint's
    *un-reduced* residual array (shape ``(B, T, ...)``) and records four
    summary statistics per constraint — mean, std, max, and 99th percentile
    — plus a histogram of the raw residual magnitudes (Sec. 3, [2207.10289]).

    A constraint whose ``max`` or ``p99`` stays orders of magnitude above
    the others indicates a region of the domain where the PDE is poorly
    satisfied; this complements :class:`GradientNormsCallback` which only
    reflects a constraint's *aggregated* contribution to the gradient.

    Args:
        interval: Compute every *n* outer training steps. Default ``100``.
        constraints: Optional list of constraint placeholders to scope the
            callback to a subset of the solver's constraints.  Pass the
            *same* Python objects that were given to ``jno.core([...])`` —
            assign your constraints to variables rather than re-accessing
            ``.mse`` (which returns a fresh placeholder each access).
            When ``None`` (default) all constraints are tracked.

    Example::

        pde_loss = pde.mse                       # assign once
        bc_loss  = bc.mse
        solver = jno.core([pde_loss, bc_loss])
        cb = jno.callbacks.residual_stats(interval=100, constraints=[pde_loss])
        crux.solve(5000, callbacks=[cb])
        print(cb.result["maxes"])   # (n_samples, 1)  — just pde_loss
    """
    return ResidualStatsCallback(interval=interval, constraints=constraints)


# docstring-from: jno.utils.adaptive.callbacks.NTKSpectrumCallback
def ntk_spectrum(
    grad_expr,
    n_points: int = 256,
    top_k: int = 10,
    interval: int = 500,
) -> NTKSpectrumCallback:
    r"""Track the empirical Neural Tangent Kernel eigenvalue spectrum.

    For a :class:`~jno.trace.NetworkGradient` placeholder ``u.grad(net)``
    (per-point parameter Jacobian :math:`J \in \mathbb{R}^{N \times P}`)
    constructs the empirical NTK :math:`K = J J^\top` at every
    ``interval`` outer steps, subsampled to ``n_points`` collocation
    points, and reports the top-k eigenvalues, condition number, and full
    spectrum.

    The eigenvalue spread is the canonical diagnostic for PINN spectral
    bias (Sec. 3-4, [2007.14527]): widely separated eigenvalues mean some
    directions in parameter space train orders of magnitude faster than
    others, producing the characteristic PINN failure mode where
    high-frequency features lag low-frequency ones.

    Args:
        grad_expr: A :class:`~jno.trace.NetworkGradient` placeholder
            (the result of ``expr.grad(model)``).  Use ``model.mask(...)``
            to restrict to a parameter subset, e.g.
            ``u.grad(net.mask(output_only_mask))`` — masking lives in the
            placeholder rather than as a separate argument.
        n_points: Number of collocation points to subsample for the
            kernel.  Cost is ``O(n_points² × P)``; default ``256``.
        top_k: Number of largest eigenvalues to report.  Default ``10``.
        interval: Compute every *n* outer training steps.  Default ``500``
            (expensive — keep large for real runs).

    Example::

        cb = jno.callbacks.ntk_spectrum(u.grad(u_net), n_points=128, top_k=10)
        crux.solve(10_000, callbacks=[cb])
        print(cb.result["lambda_max"])         # (n_samples,)
        print(cb.result["condition_number"])   # (n_samples,)
    """
    return NTKSpectrumCallback(
        grad_expr=grad_expr,
        n_points=n_points,
        top_k=top_k,
        interval=interval,
    )


# docstring-from: jno.utils.adaptive.callbacks.HessianSpectrumCallback
def hessian_spectrum(
    k: int = 10,
    n_iter: int = 30,
    interval: int = 500,
    mask=None,
    constraints=None,
) -> HessianSpectrumCallback:
    r"""Track the top-k Hessian eigenvalues of the total training loss.

    Constructs :math:`\nabla^2_\theta L` implicitly via Hessian-vector
    products (``jvp(grad(L), …, …)``) and runs ``n_iter`` Lanczos
    iterations with full reorthogonalisation.  The resulting tridiagonal
    is eigendecomposed host-side via ``scipy.linalg.eigh_tridiagonal``
    (Sec. 3.1-3.2, [1912.07145]) to obtain the top-k eigenvalues.

    The largest eigenvalue is the *sharpness* of the loss surface at the
    current iterate (Sec. 2.2, [1609.04836]); high values predict a sharp
    minimum, typically associated with worse generalisation.

    Args:
        k: Number of largest eigenvalues to report.  Default ``10``.
        n_iter: Number of Lanczos iterations.  Default ``30``.
        interval: Compute every *n* outer training steps.  Default ``500``.
        mask: Optional pytree of booleans matching the *trainable*
            structure.  Essential for large models.
        constraints: Optional list of constraint placeholders to scope the
            Hessian to the mean of those constraints' losses (instead of
            the total training loss).  Pass the same Python objects given
            to ``jno.core([...])`` — assign your constraints to variables
            rather than re-accessing ``.mse``.
    """
    return HessianSpectrumCallback(
        k=k,
        n_iter=n_iter,
        interval=interval,
        mask=mask,
        constraints=constraints,
    )


# docstring-from: jno.utils.adaptive.callbacks.InputSensitivityCallback
def input_sensitivity(expr, interval: int = 100) -> InputSensitivityCallback:
    """Evaluate any placeholder expression at training collocation points.

    Most useful for input-saliency: pass ``u.d(x)`` (single coordinate) or
    ``jno.Jacobian(u, [x, y])`` (multi-coordinate) and the callback will
    record the per-point sensitivity of the network output to its inputs
    every ``interval`` outer steps.  Conceptually equivalent to the
    class-saliency map of Sec. 3 of [1312.6034] — high magnitude points
    correspond to spatial regions where the network response is most
    responsive to a perturbation of the input coordinate.

    The expression is compiled via
    :meth:`~jno.trace_compiler.TraceCompiler.compile_multi_expression`
    (the same machinery used by the solver's constraints and trackers),
    so any composite placeholder expression works — not just first
    derivatives.

    Args:
        expr: A :class:`~jno.trace.Placeholder` expression to evaluate.
            Common choices: ``u.d(x)``, ``jno.Jacobian(u, [x, y])``,
            ``jno.np.linalg.norm(u.d(x))``.
        interval: Compute every *n* outer training steps. Default ``100``.

    Example::

        cb = jno.callbacks.input_sensitivity(u.d(x), interval=100)
        crux.solve(5000, callbacks=[cb])
        print(cb.result["values"].shape)   # (n_samples, *expr_shape)
    """
    return InputSensitivityCallback(expr=expr, interval=interval)


# docstring-from: jno.utils.adaptive.callbacks.LossLandscapeCallback
def loss_landscape(
    interval: int = 500,
    mask=None,
    n_grid: int = 15,
    alpha_range: float = 1.0,
) -> LossLandscapeCallback:
    """Evaluate the 2-D loss landscape periodically during training.

    At every ``interval`` outer steps, samples two random filter-normalized
    directions in parameter space and evaluates the total loss on an
    ``(n_grid × n_grid)`` perturbation grid centred on the current
    parameters.  The directions change each call (stochastic sampling).

    This is expensive: each call requires ``n_grid²`` forward passes.
    Choose a large ``interval`` (e.g. 500–1000) for typical runs.

    Args:
        interval: Compute every *n* outer training steps.  Default ``500``.
        mask: Optional pytree of booleans matching ``crux.models`` structure.
            When set, only the selected parameters are perturbed; the rest
            are held fixed.  Strongly recommended for large models.
        n_grid: Number of grid points per axis.  Default ``15``.
        alpha_range: Perturbation scale in units of ``‖θ_selected‖``.
            Default ``1.0``.

    Example::

        cb = jno.callbacks.loss_landscape(interval=500, n_grid=15)
        crux.solve(5000, callbacks=[cb])
        landscapes = cb.result["landscapes"]   # (n_samples, 15, 15)
    """
    return LossLandscapeCallback(
        interval=interval,
        mask=mask,
        n_grid=n_grid,
        alpha_range=alpha_range,
    )


# Re-exports for explicit class access.
__all__ = [
    "gradient_norms",
    "cos_similarity",
    "gradient_alignment",
    "residual_stats",
    "ntk_spectrum",
    "hessian_spectrum",
    "input_sensitivity",
    "loss_landscape",
    # Class re-exports
    "CosSimilarityCallback",
    "GradientAlignmentCallback",
    "GradientNormsCallback",
    "HessianSpectrumCallback",
    "InputSensitivityCallback",
    "LossLandscapeCallback",
    "NTKSpectrumCallback",
    "ResidualStatsCallback",
]


def _tracker_value(tracker: Any) -> Optional[dict]:
    """Defensive accessor used by adaptive components (returns ``None`` if
    the object does not expose a ``.value`` attribute yet)."""
    return getattr(tracker, "value", None)
