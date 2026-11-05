"""ENGD optimizer sentinel — passes ENGD config to ``jno.core.solve()``."""

from __future__ import annotations


# docstring-from: jno.optimizers.engd.engd
class ENGDOptimizer:
    r"""Return an ENGD optimizer sentinel to pass to ``net.optimizer()``.

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
        momentum: SPRING momentum :math:`\mu \in [0, 1)` (Goldshlager, Abrahamsen & Lin
            2024, arXiv:2401.10190; Jnini et al. 2026, arXiv:2604.05230, Sec. 2.3, eq. 26):
            :math:`\phi_k = \mu\phi_{k-1} + (G + \lambda I)^{-1}(\nabla L - \mu G \phi_{k-1})`,
            then :math:`\theta \leftarrow \theta - \alpha_k \phi_k`. ``0`` (default) is plain
            ENGD; ``0.5``-``0.99`` were tested. With ``damping=0`` the momentum acts only in the
            null space of the Jacobian (more parameters than residual points), so pair it with
            ``damping > 0``. With ``line_search=True`` the carried :math:`\phi_{k-1}` is the step
            taken, :math:`\alpha_{k-1}\phi_{k-1}`, not eq. 26's unscaled direction (which
            stalled at ``momentum=0.9``). :math:`\phi` is reset at every ``solve()``.
        damping: Levenberg-Marquardt :math:`\lambda \ge 0` added to :math:`G` and solved by
            Cholesky instead of ``lstsq`` (then ``rcond`` is unused). ``0`` (default) keeps the
            pseudo-inverse. A large :math:`\lambda` turns the step into gradient descent with
            rate :math:`1/\lambda` and floors the reachable accuracy; start near ``1e-8``-``1e-6``
            in float64.

    Example::

        import jax, jno

        jax.config.update("jax_enable_x64", True)   # float64 for full accuracy

        net, losses, _, eval_error = jno.baseline.Poisson2D().build(seed=0)
        net.optimizer(jno.optimizers.engd())          # replaces sgd(1.0) + callback
        # SPRING: Gauss-Newton with momentum, damped (arXiv:2401.10190)
        # net.optimizer(jno.optimizers.engd(momentum=0.9, damping=1e-8))
        crux = jno.core(losses)
        crux.solve(500)
        print(eval_error(crux))

    Note:
        **float64** is required for full accuracy.  Enable it with
        ``jax.config.update("jax_enable_x64", True)`` before training.

    References:
        Müller & Zeinhofer, *Achieving High Accuracy with PINNs via Energy
        Natural Gradient Descent*, ICML 2023, arXiv:2302.13163, Sec. 3.
        Goldshlager, Abrahamsen & Lin, *A Kaczmarz-inspired approach to
        accelerate the optimization of neural network wavefunctions*, 2024,
        arXiv:2401.10190 (SPRING).
        Jnini et al., *Curvature-Aware Optimization for High-Accuracy
        Physics-Informed Neural Networks*, 2026, arXiv:2604.05230, Sec. 2.3.
    """

    def __init__(
        self,
        gram_terms=None,
        gram_interval: int = 1,
        rcond=None,
        line_search: bool = True,
        momentum: float = 0.0,
        damping: float = 0.0,
    ):
        from jno.utils.adaptive.callbacks import _check_spring_args

        _check_spring_args(momentum, damping, "jno.optimizers.engd")
        self._gram_terms = gram_terms
        self._gram_interval = gram_interval
        self._rcond = rcond
        self._line_search = line_search
        self._momentum = float(momentum)
        self._damping = float(damping)

    def __repr__(self):
        parts = [f"gram_interval={self._gram_interval}"]
        if self._gram_terms is not None:
            parts.append("gram_terms=<explicit>")
        if self._line_search:
            parts.append("line_search=True")
        if self._momentum:
            parts.append(f"momentum={self._momentum}")
        if self._damping:
            parts.append(f"damping={self._damping}")
        return f"ENGDOptimizer({', '.join(parts)})"


def engd(
    gram_terms=None,
    *,
    gram_interval: int = 1,
    rcond=None,
    line_search: bool = True,
    momentum: float = 0.0,
    damping: float = 0.0,
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
        momentum: SPRING momentum :math:`\\mu \\in [0, 1)` (Goldshlager, Abrahamsen & Lin
            2024, arXiv:2401.10190; Jnini et al. 2026, arXiv:2604.05230, Sec. 2.3, eq. 26):
            :math:`\\phi_k = \\mu\\phi_{k-1} + (G + \\lambda I)^{-1}(\\nabla L - \\mu G \\phi_{k-1})`,
            then :math:`\\theta \\leftarrow \\theta - \\alpha_k \\phi_k`. ``0`` (default) is plain
            ENGD; ``0.5``-``0.99`` were tested. With ``damping=0`` the momentum acts only in the
            null space of the Jacobian (more parameters than residual points), so pair it with
            ``damping > 0``. With ``line_search=True`` the carried :math:`\\phi_{k-1}` is the step
            taken, :math:`\\alpha_{k-1}\\phi_{k-1}`, not eq. 26's unscaled direction (which
            stalled at ``momentum=0.9``). :math:`\\phi` is reset at every ``solve()``.
        damping: Levenberg-Marquardt :math:`\\lambda \\ge 0` added to :math:`G` and solved by
            Cholesky instead of ``lstsq`` (then ``rcond`` is unused). ``0`` (default) keeps the
            pseudo-inverse. A large :math:`\\lambda` turns the step into gradient descent with
            rate :math:`1/\\lambda` and floors the reachable accuracy; start near ``1e-8``-``1e-6``
            in float64.

    Example::

        import jax, jno

        jax.config.update("jax_enable_x64", True)   # float64 for full accuracy

        net, losses, _, eval_error = jno.baseline.Poisson2D().build(seed=0)
        net.optimizer(jno.optimizers.engd())          # replaces sgd(1.0) + callback
        # SPRING: Gauss-Newton with momentum, damped (arXiv:2401.10190)
        # net.optimizer(jno.optimizers.engd(momentum=0.9, damping=1e-8))
        crux = jno.core(losses)
        crux.solve(500)
        print(eval_error(crux))

    Note:
        **float64** is required for full accuracy.  Enable it with
        ``jax.config.update("jax_enable_x64", True)`` before training.

    References:
        Müller & Zeinhofer, *Achieving High Accuracy with PINNs via Energy
        Natural Gradient Descent*, ICML 2023, arXiv:2302.13163, Sec. 3.
        Goldshlager, Abrahamsen & Lin, *A Kaczmarz-inspired approach to
        accelerate the optimization of neural network wavefunctions*, 2024,
        arXiv:2401.10190 (SPRING).
        Jnini et al., *Curvature-Aware Optimization for High-Accuracy
        Physics-Informed Neural Networks*, 2026, arXiv:2604.05230, Sec. 2.3.
    """
    return ENGDOptimizer(
        gram_terms=gram_terms,
        gram_interval=gram_interval,
        rcond=rcond,
        line_search=line_search,
        momentum=momentum,
        damping=damping,
    )
