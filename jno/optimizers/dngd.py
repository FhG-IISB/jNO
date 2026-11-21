"""Dual Natural Gradient Descent (D-NGD): the damped Gauss-Newton / natural-gradient step of a PINN,
solved in the space of residual rows instead of the space of parameters.

Based on:
  Jnini, A., & Vella, F. (2025).
  Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks.
  arXiv:2505.21404.

With the stacked residual :math:`\\hat r \\in \\mathbb{R}^m` (rows scaled by :math:`1/\\sqrt{N_k}`, Sec.
2.2), its Jacobian :math:`J \\in \\mathbb{R}^{m \\times P}`, the loss :math:`L = \\tfrac12\\|\\hat r\\|^2`
and the residual Gramian :math:`K = J J^\\top`, the damped Gauss-Newton step of Eqs. (6)-(7) is

.. math::

    v = -J^\\top (K + \\lambda I_m)^{-1} \\hat r ,

an ``m x m`` solve, so its cost does not grow with the number of parameters. Eqs. (6)-(7) write the
same vector as :math:`-\\tfrac1\\lambda (J^\\top y + \\nabla L)` with :math:`(K + \\lambda I)y = -J\\nabla L`;
the push-through form above avoids the cancellation in that difference at small :math:`\\lambda`.
The geodesic-acceleration correction (Prop. 3.2, Eq. (10)) reuses the factorisation:

.. math::

    f_{vv} = \\tfrac{d^2}{dt^2} \\hat r(\\theta + t v)\\big|_{t=0}, \\qquad
    a = -J^\\top (K + \\lambda I)^{-1} f_{vv},

and is kept only when :math:`2\\|a\\|/\\|v\\| \\le 0.5` (Algorithm 4), giving the step :math:`v + a/2`.
A line search over :math:`\\eta \\in (0, 1]` then sets :math:`\\theta \\leftarrow \\theta + \\eta\\,\\mathrm{step}`
(Algorithm 1).

Choices the paper leaves open, made here: the 31 line-search points are ENGD's grid
:math:`\\eta \\in \\{0.5^0, \\ldots, 0.5^{30}\\}`; the acceleration is skipped when
:math:`\\|v\\| \\le \\epsilon_{\\text{mach}}(\\|\\theta\\| + 1)`; the damping is floored at
:math:`10\\,\\epsilon_{\\text{mach}}\\operatorname{tr} K` so that the Cholesky factorisation exists when
:math:`m > P` (then :math:`K` is singular and :func:`lm` or :func:`engd` are the better choice).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from ._residual_callback import grid_line_search, half_sq_norm
from ._residuals import jacobian, second_directional
from ._sentinel import ResidualOptimizer

_AUTO_DAMPING_CAP = 1e-5  # lambda_k = min(loss, 1e-5), the setting of several benchmarks (Tables 2-8)


def _dngd_damping(damping, loss, trace_k, dtype):
    lam = jnp.minimum(loss, _AUTO_DAMPING_CAP) if damping == "auto" else jnp.asarray(float(damping), dtype)
    return jnp.maximum(lam, 10.0 * jnp.finfo(dtype).eps * trace_k)


def _dngd_step(r_of, flat_p, r, J, solve, *, geodesic: bool, ratio_max: float):
    """The primary step ``v`` and, if accepted, ``v + a/2``; ``solve(b) = (K + λI)^{-1} b``."""
    v = -(J.T @ solve(r))
    info = {}
    if geodesic:
        fvv = second_directional(r_of, flat_p, v)
        a = -(J.T @ solve(fvv))
        nv = jnp.linalg.norm(v)
        use = (nv > jnp.finfo(r.dtype).eps * (jnp.linalg.norm(flat_p) + 1.0)) & (2.0 * jnp.linalg.norm(a) <= ratio_max * nv)
        v = jnp.where(use, v + 0.5 * a, v)
        info["geodesic"] = use
    return v, info


def _dngd_direction(r_of, flat_p, state, *, damping="auto", geodesic=True, ratio_max=0.5, line_search=True):
    """One dense D-NGD step; returns ``(d, state, info)`` with new parameters ``θ - d``."""
    r = r_of(flat_p)
    J = jacobian(r_of, flat_p, r.shape[0])
    K = J @ J.T
    loss = half_sq_norm(r)
    lam = _dngd_damping(damping, loss, jnp.trace(K), r.dtype)
    c = cho_factor(K + lam * jnp.eye(K.shape[0], dtype=K.dtype))

    def solve(b):
        return cho_solve(c, b)

    step, info = _dngd_step(r_of, flat_p, r, J, solve, geodesic=geodesic, ratio_max=ratio_max)
    if line_search:
        eta, _ = grid_line_search(lambda e: half_sq_norm(r_of(flat_p + e * step)), dtype=r.dtype)
    else:
        eta = jnp.asarray(1.0, r.dtype)
    return -eta * step, state, {**info, "damping": lam, "eta": eta, "loss": loss}


# docstring-from: jno.optimizers.dngd.dngd
class DNGDOptimizer(ResidualOptimizer):
    r"""Dual Natural Gradient Descent: the Gauss-Newton step solved over residual rows, not parameters.

    Pass to ``net.optimizer(...)``. Each epoch forms the residual Gramian :math:`K = J J^\top`
    (``m x m`` for ``m`` residual rows), factorises :math:`K + \lambda I` once, and takes the step
    :math:`-J^\top (K + \lambda I)^{-1} \hat r`, optionally corrected by geodesic acceleration, along
    a line search. The memory and the solve depend on ``m``, not on the number of parameters
    ``P``, so this is the Gauss-Newton method for networks with more parameters than residual
    rows; with fewer, :func:`lm` or :func:`engd` solve the smaller ``P x P`` system. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs.
        damping: ``"auto"`` (default) uses :math:`\lambda = \min(L, 10^{-5})` with
            :math:`L = \tfrac12\|\hat r\|^2`, the paper's setting for most benchmarks; a float fixes it.
        geodesic_acceleration: Add the second-order correction :math:`a/2` when
            :math:`2\|a\|/\|v\| \le 0.5` (default ``True``; one extra second directional derivative).
        line_search: Pick the step length on the grid :math:`\{0.5^k\}_{k=0}^{30}` (default ``True``).

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.dngd())
        jno.core([pde.mse, bc.mse]).solve(500)

    References:
        Jnini & Vella, *Dual Natural Gradient Descent for Scalable Training of Physics-Informed
        Neural Networks*, 2025, arXiv:2505.21404, Eqs. (6), (7), (10), Algorithms 1 and 4.
    """

    _who = "jno.optimizers.dngd"

    def __init__(self, gram_terms=None, *, damping="auto", geodesic_acceleration=True, line_search=True):
        if damping != "auto" and not float(damping) > 0.0:
            raise ValueError(f"jno.optimizers.dngd: damping must be 'auto' or > 0, got {damping!r}.")
        super().__init__(gram_terms)
        self._damping = damping
        self._geodesic = bool(geodesic_acceleration)
        self._line_search = bool(line_search)

    def build_callback(self, gram_terms):
        """Return the callback that computes this optimizer's step inside ``solve()``.

        Args:
            gram_terms: ``(NetworkGradient, weight)`` pairs, one per residual class (auto-detected by
                ``solve()`` unless given to the factory).
        """
        from ._residual_callback import ResidualStepCallback

        spec = self

        class DNGDCallback(ResidualStepCallback):
            who = "jno.optimizers.dngd"

            def direction(self, r_of, flat_p, state):
                return _dngd_direction(
                    r_of,
                    flat_p,
                    state,
                    damping=spec._damping,
                    geodesic=spec._geodesic,
                    line_search=spec._line_search,
                )

        return DNGDCallback(gram_terms)

    def __repr__(self):
        return (
            f"DNGDOptimizer(damping={self._damping!r}, geodesic_acceleration={self._geodesic}, "
            f"line_search={self._line_search})"
        )


def dngd(gram_terms=None, *, damping="auto", geodesic_acceleration=True, line_search=True) -> DNGDOptimizer:
    """Dual Natural Gradient Descent: the Gauss-Newton step solved over residual rows, not parameters.

    Pass to ``net.optimizer(...)``. Each epoch forms the residual Gramian :math:`K = J J^\\top`
    (``m x m`` for ``m`` residual rows), factorises :math:`K + \\lambda I` once, and takes the step
    :math:`-J^\\top (K + \\lambda I)^{-1} \\hat r`, optionally corrected by geodesic acceleration, along
    a line search. The memory and the solve depend on ``m``, not on the number of parameters
    ``P``, so this is the Gauss-Newton method for networks with more parameters than residual
    rows; with fewer, :func:`lm` or :func:`engd` solve the smaller ``P x P`` system. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs.
        damping: ``"auto"`` (default) uses :math:`\\lambda = \\min(L, 10^{-5})` with
            :math:`L = \\tfrac12\\|\\hat r\\|^2`, the paper's setting for most benchmarks; a float fixes it.
        geodesic_acceleration: Add the second-order correction :math:`a/2` when
            :math:`2\\|a\\|/\\|v\\| \\le 0.5` (default ``True``; one extra second directional derivative).
        line_search: Pick the step length on the grid :math:`\\{0.5^k\\}_{k=0}^{30}` (default ``True``).

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.dngd())
        jno.core([pde.mse, bc.mse]).solve(500)

    References:
        Jnini & Vella, *Dual Natural Gradient Descent for Scalable Training of Physics-Informed
        Neural Networks*, 2025, arXiv:2505.21404, Eqs. (6), (7), (10), Algorithms 1 and 4.
    """
    return DNGDOptimizer(gram_terms, damping=damping, geodesic_acceleration=geodesic_acceleration, line_search=line_search)
