"""Levenberg-Marquardt for PINN residuals: Gauss-Newton with a damping that adapts to how well the
linear model predicted the last step.

Based on:
  Levenberg, K. (1944).
  A method for the solution of certain non-linear problems in least squares.
  Quarterly of Applied Mathematics, 2(2), 164-168.
  Marquardt, D. W. (1963).
  An algorithm for least-squares estimation of nonlinear parameters.
  Journal of the Society for Industrial and Applied Mathematics, 11(2), 431-441.
  Madsen, K., Nielsen, H. B., & Tingleff, O. (2004).
  Methods for Non-Linear Least Squares Problems (2nd ed.), Sec. 3.2. IMM, Technical University
  of Denmark -- the damping update and its initial value used here.
In PINNs:
  Taylor, J. A., Wang, W., Bala, B., & Bednarz, T. (2022).
  Optimizing the optimizer for data driven deep neural networks and physics informed neural
  networks. arXiv:2205.07430.
  Bonfanti, A., Bruno, G., & Cipriani, C. (2024).
  The Challenges of the Nonlinear Regime for Physics-Informed Neural Networks. arXiv:2402.03864.

With the weighted residual :math:`\\hat r` and its Jacobian :math:`J` (see ``_residuals``), each step
minimises :math:`\\tfrac12\\|\\hat r - J d\\|^2 + \\tfrac{\\lambda}{2}\\|d\\|^2` and moves to
:math:`\\theta - d`:

.. math::

    d = (J^\\top J + \\lambda I)^{-1} J^\\top \\hat r = J^\\top (J J^\\top + \\lambda I)^{-1} \\hat r,

the first form when :math:`P \\le m`, the second (the same vector) when there are more parameters
than residual rows. One symmetric eigendecomposition of the smaller Gram matrix per step makes every
trial :math:`\\lambda` a diagonal solve. The gain ratio :math:`\\rho` (actual over predicted decrease
of :math:`\\tfrac12\\|\\hat r\\|^2`) decides: :math:`\\rho > 0` accepts and sets
:math:`\\lambda \\leftarrow \\lambda\\max(1/3, 1-(2\\rho-1)^3)`, otherwise :math:`\\lambda \\leftarrow
\\nu\\lambda`, :math:`\\nu \\leftarrow 2\\nu` and the step is retried, up to ``max_tries`` times.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._residuals import jacobian
from ._sentinel import ResidualOptimizer


class _LMState(NamedTuple):
    damping: jax.Array  # current lambda; < 0 means "not initialised yet"
    nu: jax.Array  # growth factor after a rejected step


def _lm_init(dtype, damping=None) -> _LMState:
    lam = -1.0 if damping is None else float(damping)
    return _LMState(jnp.asarray(lam, dtype), jnp.asarray(2.0, dtype))


def _lm_direction(r_of, flat_p, state: _LMState, *, tau: float = 1e-3, max_tries: int = 10):
    """One Levenberg-Marquardt step; returns ``(d, new_state, info)`` with new parameters ``θ - d``."""
    r = r_of(flat_p)
    m, P = r.shape[0], flat_p.shape[0]
    J = jacobian(r_of, flat_p, m)
    dtype = r.dtype
    f0 = 0.5 * jnp.vdot(r, r)

    if P <= m:
        s, V = jnp.linalg.eigh(J.T @ J)
        b = V.T @ (J.T @ r)

        def step(lam):
            return V @ (b / (s + lam))
    else:
        s, U = jnp.linalg.eigh(J @ J.T)
        b = U.T @ r

        def step(lam):
            return J.T @ (U @ (b / (s + lam)))

    s = jnp.maximum(s, 0.0)
    s_max = jnp.max(s)
    lam_floor = 10.0 * jnp.finfo(dtype).eps * s_max
    # Madsen et al. (2004), Sec. 3.2: lambda_0 = tau * max_i (J^T J)_ii
    lam0 = jnp.where(state.damping < 0, tau * jnp.max(jnp.sum(J * J, axis=0)), state.damping)
    lam0 = jnp.maximum(lam0, lam_floor)

    eps = jnp.finfo(dtype).eps

    g = J.T @ r

    def trial(lam):
        d = step(lam)
        # Predicted decrease of the damped model, f0 - 1/2||r - J d||^2 = 1/2 d^T (lam d + J^T r),
        # in the form without cancellation (Madsen et al. 2004, eq. 3.9b in their sign convention).
        pred = 0.5 * jnp.vdot(d, lam * d + g)
        r1 = r_of(flat_p - d)
        f1 = 0.5 * jnp.vdot(r1, r1)
        rho = (f0 - f1) / pred
        # Converged: the predicted decrease is within the rounding of f0 - f1 (a sum of m squares,
        # error ~ sqrt(m) eps f0), so the measured ratio is noise. The step is then taken as is (it
        # cannot hurt beyond rounding) and lambda kept; rejecting it would grow lambda at every
        # epoch spent at the minimum.
        # The second test is Madsen et al.'s step-size stop, ||d|| <= e (||theta|| + e): it catches a
        # residual that has reached its own absolute rounding floor (a consistent system), where the
        # relative test above does not apply.
        noise = 100.0 * jnp.sqrt(m) * eps * f0
        e2 = 1e3 * eps
        tiny_step = jnp.linalg.norm(d) <= e2 * (jnp.linalg.norm(flat_p) + e2)
        stalled = jnp.isfinite(f1) & (((pred <= noise) & (f1 <= f0 + noise)) | tiny_step)
        ok = (jnp.isfinite(f1) & jnp.isfinite(rho) & (pred > 0) & (rho > 0)) | stalled
        return d, rho, ok, stalled

    def cond(c):
        i, _, _, _, _, ok, _ = c
        return (i < max_tries) & ~ok

    def body(c):
        i, lam, nu, _, _, _, _ = c
        lam_try = jnp.where(i == 0, lam, lam * nu)
        nu = jnp.where(i == 0, nu, 2.0 * nu)
        d, rho, ok, stalled = trial(lam_try)
        return i + 1, lam_try, nu, d, rho, ok, stalled

    false = jnp.asarray(False)
    init = (0, lam0, state.nu, jnp.zeros_like(flat_p), jnp.asarray(0.0, dtype), false, false)
    tries, lam, nu, d, rho, ok, stalled = jax.lax.while_loop(cond, body, init)
    lam_next = jnp.where(ok, lam * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), lam * nu)
    lam_next = jnp.where(stalled, lam, lam_next)
    nu_next = jnp.where(ok, 2.0, jnp.minimum(2.0 * nu, 2.0**30))
    # Repeated rejections double nu each time; past 1e16 x the largest curvature the step is zero to
    # working precision, so cap lambda there instead of letting it overflow.
    lam_cap = 1e16 * jnp.maximum(s_max, jnp.finfo(dtype).tiny)
    new_state = _LMState(jnp.clip(lam_next, lam_floor, lam_cap), jnp.asarray(nu_next, dtype))
    info = {"damping": lam, "rho": rho, "accepted": ok, "stalled": stalled, "tries": tries, "loss": f0}
    return jnp.where(ok, d, jnp.zeros_like(d)), new_state, info


# docstring-from: jno.optimizers.lm.lm
class LMOptimizer(ResidualOptimizer):
    r"""Levenberg-Marquardt: damped Gauss-Newton whose damping adapts to the gain ratio.

    Pass to ``net.optimizer(...)``; ``jno.core.solve()`` builds the residual Jacobian from every loss
    term involving the model (or from ``gram_terms``) and applies one LM step per epoch. A rejected
    trial step raises the damping and is retried inside the same epoch, so the loss never increases.
    The step is dense: it forms the full Jacobian and decomposes the smaller of
    :math:`J^\top J` (``P x P``) and :math:`J J^\top` (``m x m``), so it suits up to roughly
    :math:`10^4` parameters or residual rows. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise an explicit list of ``(NetworkGradient, weight)`` pairs;
            the weights define the objective :math:`\sum_k w_k\,\mathrm{mean}(r_k^2)`.
        damping: Initial :math:`\lambda > 0`. ``None`` (default) uses
            :math:`\tau \max_i (J^\top J)_{ii}` at the first step (Madsen et al. 2004).
        tau: The factor of that automatic initial damping (default ``1e-3``).
        max_tries: Trial steps per epoch before giving up on the epoch (the parameters then stay put
            and the damping keeps growing at the next epoch).

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.lm())
        jno.core([pde.mse, bc.mse]).solve(200)

    References:
        Levenberg (1944), Q. Appl. Math. 2, 164; Marquardt (1963), SIAM J. Appl. Math. 11, 431;
        Madsen, Nielsen & Tingleff (2004), *Methods for Non-Linear Least Squares Problems*, Sec. 3.2.
        For PINNs: Taylor et al. (2022), arXiv:2205.07430; Bonfanti et al. (2024), arXiv:2402.03864.
    """

    _who = "jno.optimizers.lm"

    def __init__(self, gram_terms=None, *, damping=None, tau: float = 1e-3, max_tries: int = 10):
        if damping is not None and not float(damping) > 0.0:
            raise ValueError(f"jno.optimizers.lm: damping must be > 0 (or None for automatic), got {damping!r}.")
        if not float(tau) > 0.0:
            raise ValueError(f"jno.optimizers.lm: tau must be > 0, got {tau!r}.")
        if int(max_tries) < 1:
            raise ValueError(f"jno.optimizers.lm: max_tries must be >= 1, got {max_tries!r}.")
        super().__init__(gram_terms)
        self._damping, self._tau, self._max_tries = damping, float(tau), int(max_tries)

    def build_callback(self, gram_terms):
        """Return the callback that computes this optimizer's step inside ``solve()``.

        Args:
            gram_terms: ``(NetworkGradient, weight)`` pairs, one per residual class (auto-detected by
                ``solve()`` unless given to the factory).
        """
        from ._residual_callback import ResidualStepCallback

        spec = self

        class LMCallback(ResidualStepCallback):
            who = "jno.optimizers.lm"

            def init_state(self, flat_p, m):
                return _lm_init(flat_p.dtype, spec._damping)

            def direction(self, r_of, flat_p, state):
                return _lm_direction(r_of, flat_p, state, tau=spec._tau, max_tries=spec._max_tries)

        return LMCallback(gram_terms)

    def __repr__(self):
        return f"LMOptimizer(damping={self._damping}, tau={self._tau}, max_tries={self._max_tries})"


def lm(gram_terms=None, *, damping=None, tau: float = 1e-3, max_tries: int = 10) -> LMOptimizer:
    """Levenberg-Marquardt: damped Gauss-Newton whose damping adapts to the gain ratio.

    Pass to ``net.optimizer(...)``; ``jno.core.solve()`` builds the residual Jacobian from every loss
    term involving the model (or from ``gram_terms``) and applies one LM step per epoch. A rejected
    trial step raises the damping and is retried inside the same epoch, so the loss never increases.
    The step is dense: it forms the full Jacobian and decomposes the smaller of
    :math:`J^\\top J` (``P x P``) and :math:`J J^\\top` (``m x m``), so it suits up to roughly
    :math:`10^4` parameters or residual rows. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise an explicit list of ``(NetworkGradient, weight)`` pairs;
            the weights define the objective :math:`\\sum_k w_k\\,\\mathrm{mean}(r_k^2)`.
        damping: Initial :math:`\\lambda > 0`. ``None`` (default) uses
            :math:`\\tau \\max_i (J^\\top J)_{ii}` at the first step (Madsen et al. 2004).
        tau: The factor of that automatic initial damping (default ``1e-3``).
        max_tries: Trial steps per epoch before giving up on the epoch (the parameters then stay put
            and the damping keeps growing at the next epoch).

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.lm())
        jno.core([pde.mse, bc.mse]).solve(200)

    References:
        Levenberg (1944), Q. Appl. Math. 2, 164; Marquardt (1963), SIAM J. Appl. Math. 11, 431;
        Madsen, Nielsen & Tingleff (2004), *Methods for Non-Linear Least Squares Problems*, Sec. 3.2.
        For PINNs: Taylor et al. (2022), arXiv:2205.07430; Bonfanti et al. (2024), arXiv:2402.03864.
    """
    return LMOptimizer(gram_terms, damping=damping, tau=tau, max_tries=max_tries)
