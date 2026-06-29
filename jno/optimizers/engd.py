"""ENGD optimizer sentinel — passes ENGD config to ``jno.core.solve()``."""

from __future__ import annotations


class ENGDOptimizer:
    """Sentinel stored on a model via ``net.optimizer(jno.optimizers.engd(...))``.

    Not an optax ``GradientTransformation`` — ``jno.core.solve()`` detects
    this object early, auto-detects ``gram_terms`` from all loss expressions
    involving the model, injects ``optax.sgd(1.0)`` as the actual parameter
    update transform, and prepends an
    :class:`~jno.utils.adaptive.callbacks.ENGDCallback`.

    See :func:`engd` for the public constructor.
    """

    def __init__(
        self,
        gram_terms=None,
        gram_interval: int = 1,
        rcond=None,
        line_search: bool = True,
    ):
        self._gram_terms = gram_terms
        self._gram_interval = gram_interval
        self._rcond = rcond
        self._line_search = line_search

    def __repr__(self):
        parts = [f"gram_interval={self._gram_interval}"]
        if self._gram_terms is not None:
            parts.append("gram_terms=<explicit>")
        if self._line_search:
            parts.append("line_search=True")
        return f"ENGDOptimizer({', '.join(parts)})"


def engd(
    gram_terms=None,
    *,
    gram_interval: int = 1,
    rcond=None,
    line_search: bool = True,
) -> ENGDOptimizer:
    """Return an ENGD optimizer sentinel to pass to ``net.optimizer()``.

    Energy Natural Gradient Descent (E-NGD, Müller & Zeinhofer, ICML 2023,
    arXiv:2302.13163, Sec. 3) preconditions parameter gradients with the
    inverse energy Gram matrix G⁻¹, converting gradient descent into a
    Newton-like step in the PDE function-space norm.  In practice E-NGD
    reaches several orders of magnitude lower relative L² error than Adam or
    L-BFGS in far fewer iterations.

    Unlike ``jno.callbacks.engd()``, this form:

    * does not require a separate ``net.optimizer(optax.sgd(1.0))`` call
      (``optax.sgd(1.0)`` is injected automatically);
    * auto-detects ``gram_terms`` from **all** constraints at ``solve()`` time
      when ``gram_terms=None`` (default).

    Args:
        gram_terms: ``None`` (default) — build gram_terms automatically at
            ``solve()`` time from every loss expression involving this model,
            each weighted 1.0.  Pass an explicit list of
            ``(NetworkGradient_expr, weight)`` pairs — the same format as
            ``jno.callbacks.engd(gram_terms=...)`` — for custom weighting or
            a subset of terms.
        gram_interval: Recompute G every *n* outer steps (default 1).
        rcond: Condition-number cutoff for ``jnp.linalg.lstsq``.
            ``None`` → machine epsilon (best for float64).
        line_search: If ``True`` (default), perform a 31-point grid line
            search α ∈ {0.5⁰, …, 0.5³⁰} each step.  Strongly recommended;
            the Gram is ill-conditioned near initialisation and the line search
            is essential for convergence.

    Example::

        import jax, jno

        jax.config.update("jax_enable_x64", True)   # float64 for full accuracy

        net, losses, _, eval_error = jno.baseline.Poisson2D().build(seed=0)
        net.optimizer(jno.optimizers.engd())          # replaces sgd(1.0) + callback
        crux = jno.core(losses)
        crux.solve(500)
        print(eval_error(crux))

    Note:
        **float64** is required for full accuracy.  Enable it with
        ``jax.config.update("jax_enable_x64", True)`` before training.

    References:
        Müller & Zeinhofer, *Achieving High Accuracy with PINNs via Energy
        Natural Gradient Descent*, ICML 2023, arXiv:2302.13163, Sec. 3.
    """
    return ENGDOptimizer(
        gram_terms=gram_terms,
        gram_interval=gram_interval,
        rcond=rcond,
        line_search=line_search,
    )
