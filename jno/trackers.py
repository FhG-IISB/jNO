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

Note
----
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


def gradient_norms(interval: int = 100, mask=None) -> GradientNormsCallback:
    """Live tracker for per-loss gradient L2 norms.

    See :class:`~jno.utils.adaptive.callbacks.GradientNormsCallback` for the
    full reference. After the first interval, ``tracker.value["norms"]``
    is a ``(N,)`` numpy array of per-constraint gradient norms.
    """
    return GradientNormsCallback(interval=interval, mask=mask)


def cos_similarity(interval: int = 100, mask=None) -> CosSimilarityCallback:
    """Live tracker for pairwise gradient cosine similarity.

    After the first interval, ``tracker.value["cos_sim_matrix"]`` is the
    ``(N, N)`` matrix.
    """
    return CosSimilarityCallback(interval=interval, mask=mask)


def gradient_alignment(interval: int = 100, mask=None) -> GradientAlignmentCallback:
    """Live tracker for the total gradient alignment scalar.

    After the first interval, ``tracker.value["alignment"]`` is a Python
    float in ``[-1, 1]``.
    """
    return GradientAlignmentCallback(interval=interval, mask=mask)


def residual_stats(interval: int = 100, constraints=None) -> ResidualStatsCallback:
    """Live tracker for per-constraint residual mean/std/max/p99.

    After the first interval, ``tracker.value`` contains ``means``, ``stds``,
    ``maxes``, ``p99`` (each ``(K,)``) and ``indices`` (``(K,)`` int).
    """
    return ResidualStatsCallback(interval=interval, constraints=constraints)


def ntk_spectrum(
    grad_expr,
    n_points: int = 256,
    top_k: int = 10,
    interval: int = 500,
) -> NTKSpectrumCallback:
    """Live tracker for the empirical NTK eigenvalue spectrum.

    After the first interval, ``tracker.value`` contains ``eigvals_topk``,
    ``lambda_min``, ``lambda_max``, ``condition_number``, ``all_eigvals``,
    and ``trace`` (sum of eigenvalues — convenient for NTK loss balancing).
    """
    return NTKSpectrumCallback(
        grad_expr=grad_expr,
        n_points=n_points,
        top_k=top_k,
        interval=interval,
    )


def hessian_spectrum(
    k: int = 10,
    n_iter: int = 30,
    interval: int = 500,
    mask=None,
    constraints=None,
) -> HessianSpectrumCallback:
    """Live tracker for the top-k Hessian eigenvalues (sharpness).

    After the first interval, ``tracker.value`` contains ``eigvals`` and
    ``sharpness``.
    """
    return HessianSpectrumCallback(
        k=k,
        n_iter=n_iter,
        interval=interval,
        mask=mask,
        constraints=constraints,
    )


def input_sensitivity(expr, interval: int = 100) -> InputSensitivityCallback:
    """Live tracker for an arbitrary placeholder expression.

    After the first interval, ``tracker.value["values"]`` is the evaluated
    array (e.g. ``u.d(x)`` per collocation point).
    """
    return InputSensitivityCallback(expr=expr, interval=interval)


def loss_landscape(
    interval: int = 500,
    mask=None,
    n_grid: int = 15,
    alpha_range: float = 1.0,
) -> LossLandscapeCallback:
    """Live tracker for the 2-D filter-normalized loss landscape.

    After the first interval, ``tracker.value["landscape"]`` is the
    ``(n_grid, n_grid)`` grid.
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
