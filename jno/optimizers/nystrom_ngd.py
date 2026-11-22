"""Matrix-free natural gradient descent for PINNs with a randomized Nyström preconditioner: the
Gauss-Newton system is never formed, only applied, and solved by preconditioned CG.

Based on:
  Bioli, I., Marcati, C., & Sangalli, G. (2026).
  Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra.
  Computer Methods in Applied Mechanics and Engineering, doi:10.1016/j.cma.2026.119456.
  Frangella, Z., Tropp, J. A., & Udell, M. (2023).
  Randomized Nyström Preconditioning. SIAM J. Matrix Anal. Appl., 44(2), 718-752.

With the weighted residual :math:`\\hat r`, :math:`L = \\tfrac12\\|\\hat r\\|^2`, :math:`g = J^\\top \\hat r`
and the Gramian :math:`G = J^\\top J` applied as :math:`G v = J^\\top (J v)` (one JVP and one VJP,
Sec. 3.2), each epoch solves Eq. (7),

.. math::

    (G + \\mu I)\\, d = g, \\qquad \\mu = \\max(\\gamma\\,\\epsilon_{\\text{mach}}\\hat\\lambda_1,\\; 10^{-4} L^2),

with :math:`\\gamma = 10` and :math:`\\hat\\lambda_1` the largest Nyström eigenvalue (Sec. 4.3 and the
general setup of Sec. 5), by conjugate gradients preconditioned with Eq. (15) from a fresh rank-ℓ
Gaussian Nyström sketch of :math:`G` (Algorithm 6), then moves to :math:`\\theta - t d` with Armijo
backtracking (Algorithm 1).

The rank adapts between epochs as in Algorithm 4: while the smallest sketched eigenvalue
:math:`\\hat\\lambda_\\ell` exceeds :math:`10\\mu` the next sketch is :math:`\\delta_\\ell = 20` larger (up to
:math:`\\ell_{\\max} = 500`), otherwise it shrinks to the first index with :math:`\\hat\\lambda_i < 10\\mu`,
rounded up to a multiple of :math:`\\delta_\\ell` (the paper's "small offset"; the rounding also bounds
the step to at most :math:`\\ell_{\\max}/\\delta_\\ell` compiled sizes). The preconditioner must cover every
eigenvalue above the damping: with a fixed rank of 100 out of 129 parameters, a Poisson problem
stalled at rel-L2 5e-3, the eigenvalues between the sketch and :math:`\\mu` left to 20 CG iterations.

Choices the paper leaves open, made here: the first sketch has ``rank`` columns; CG stops at relative residual ``cg_tol`` (default
``1e-10``) or ``cg_maxiter`` iterations (default 20, the paper's limit) from a zero start; Armijo
uses :math:`c_1 = 10^{-4}`, halving, at most 30 halvings. The paper's diagonal-sampling sketch needs
:math:`\\operatorname{diag}(G)`, ``P`` extra JVPs, so only the Gaussian sketch is offered.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ._randnla import nystrom_approx, nystrom_precond, pcg
from ._residual_callback import armijo, half_sq_norm
from ._residuals import linearize
from ._sentinel import ResidualOptimizer


class _NysState(NamedTuple):
    key: jax.Array


def _nystrom_ngd_direction(
    r_of, flat_p, state: _NysState, *, rank: int, gamma=10.0, cg_tol=1e-10, cg_maxiter=20, damping=None
):
    """One Nyström-preconditioned NGD step; returns ``(d, state, info)`` with new parameters ``θ - d``."""
    lin = linearize(r_of, flat_p)
    r = lin.r
    dtype = r.dtype
    P = flat_p.shape[0]
    loss = half_sq_norm(r)
    g = lin.vjp(r)

    def gram(v):
        return lin.vjp(lin.jvp(v))

    key, sub = jax.random.split(state.key)
    ny = nystrom_approx(gram, P, min(rank, P), sub, dtype)
    lam1 = ny.lam[0]
    if damping is None:
        mu = jnp.maximum(gamma * jnp.finfo(dtype).eps * lam1, 1e-4 * loss**2)
    else:
        mu = jnp.asarray(float(damping), dtype)
    mu = jnp.maximum(mu, jnp.finfo(dtype).tiny)
    precond = nystrom_precond(ny, mu, 1.0 / (ny.lam[-1] + mu))
    sol = pcg(lambda v: gram(v) + mu * v, g, precond, tol=cg_tol, maxiter=cg_maxiter)
    d = sol.x
    slope = -jnp.vdot(g, d)  # derivative of L(θ - t d) at t = 0
    t, f_new = armijo(lambda t: half_sq_norm(r_of(flat_p - t * d)), loss, slope)
    info = {
        "damping": mu,
        "t": t,
        "loss": loss,
        "cg_iters": sol.iters,
        "cg_rel_residual": sol.rel_residual,
        "lambda_max": lam1,
        "lambda_min_sketch": ny.lam[-1],
        "n_above_10mu": jnp.sum(ny.lam >= 10.0 * mu),
        "rank": jnp.asarray(ny.lam.shape[0]),
    }
    return t * d, _NysState(key), info


def _next_rank(rank: int, info, step: int, rank_max: int) -> int:
    """Algorithm 4's rank rule: grow by ``step`` while the sketch's smallest eigenvalue is above 10 mu,
    else shrink to the eigenvalues above 10 mu, rounded up to a multiple of ``step``."""
    if float(info["lambda_min_sketch"]) > 10.0 * float(info["damping"]):
        new = rank + step
    else:
        new = -(-(int(info["n_above_10mu"]) + 1) // step) * step
    return int(min(max(new, min(step, rank_max)), rank_max))


# docstring-from: jno.optimizers.nystrom_ngd.nystrom_ngd
class NystromNGDOptimizer(ResidualOptimizer):
    r"""Matrix-free natural gradient with a randomized Nyström preconditioner and Armijo steps.

    Pass to ``net.optimizer(...)``. The Gramian :math:`J^\top J` is only ever applied (a JVP and a
    VJP of the residual), never formed, so memory stays at a few vectors of length ``P`` plus the
    ``P x rank`` sketch: this is the natural-gradient method for networks too large for
    :func:`engd` or :func:`lm`. Each epoch draws a new sketch, solves the damped system by at most
    ``cg_maxiter`` preconditioned CG iterations, and backtracks along the result. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs.
        rank: Size ℓ of the first Nyström sketch (default 100; each column costs one Gramian product
            per epoch).
        adaptive_rank: Adapt ℓ between epochs as in the paper's Algorithm 4 (default ``True``);
            each new size compiles the step once.
        rank_step: The paper's :math:`\delta_\ell` (default 20).
        rank_max: The paper's :math:`\ell_{\max}` (default 500; capped at the parameter count).
        damping: ``None`` (default) uses :math:`\mu = \max(10\,\epsilon\hat\lambda_1, 10^{-4}L^2)`,
            which vanishes as the loss does (a PINN whose residual can reach zero); for a loss with a
            positive minimum (noisy data) it stays at :math:`10^{-4}L_\min^2` and slows the final
            digits, so fix a float there.
        cg_tol: Relative-residual tolerance of CG (default ``1e-10``).
        cg_maxiter: CG iterations per epoch (default 20, the paper's setting).
        seed: Seed of the sketches.

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.nystrom_ngd(rank=200))
        jno.core([pde.mse, bc.mse]).solve(500)

    References:
        Bioli, Marcati & Sangalli, *Accelerating Natural Gradient Descent for PINNs with Randomized
        Numerical Linear Algebra*, CMAME (2026), doi:10.1016/j.cma.2026.119456, Eqs. (7), (15),
        Algorithms 1, 3, 4 and 6. Frangella, Tropp & Udell, *Randomized Nyström Preconditioning*,
        SIAM J. Matrix Anal. Appl. 44(2), 2023.
    """

    _who = "jno.optimizers.nystrom_ngd"

    def __init__(
        self,
        gram_terms=None,
        *,
        rank=100,
        adaptive_rank=True,
        rank_step=20,
        rank_max=500,
        damping=None,
        cg_tol=1e-10,
        cg_maxiter=20,
        seed=0,
    ):
        if int(rank) < 1 or int(rank_step) < 1 or int(rank_max) < int(rank):
            raise ValueError(
                f"jno.optimizers.nystrom_ngd: need 1 <= rank <= rank_max and rank_step >= 1, got "
                f"rank={rank!r}, rank_step={rank_step!r}, rank_max={rank_max!r}."
            )
        if damping is not None and not float(damping) > 0.0:
            raise ValueError(f"jno.optimizers.nystrom_ngd: damping must be > 0 or None, got {damping!r}.")
        if int(cg_maxiter) < 1:
            raise ValueError(f"jno.optimizers.nystrom_ngd: cg_maxiter must be >= 1, got {cg_maxiter!r}.")
        super().__init__(gram_terms)
        self._rank, self._damping = int(rank), damping
        self._adaptive, self._rank_step, self._rank_max = bool(adaptive_rank), int(rank_step), int(rank_max)
        self._cg_tol, self._cg_maxiter, self._seed = float(cg_tol), int(cg_maxiter), int(seed)

    def build_callback(self, gram_terms):
        """Return the callback that computes this optimizer's step inside ``solve()``.

        Args:
            gram_terms: ``(NetworkGradient, weight)`` pairs, one per residual class (auto-detected by
                ``solve()`` unless given to the factory).
        """
        from ._residual_callback import ResidualStepCallback

        spec = self

        class NystromNGDCallback(ResidualStepCallback):
            who = "jno.optimizers.nystrom_ngd"

            def init_state(self, flat_p, m):
                return _NysState(jax.random.PRNGKey(spec._seed))

            def initial_static(self):
                return {"rank": min(spec._rank, self.P)}

            def next_static(self, static, info):
                if not spec._adaptive:
                    return static
                return {"rank": _next_rank(static["rank"], info, spec._rank_step, min(spec._rank_max, self.P))}

            def direction(self, r_of, flat_p, state, rank):
                return _nystrom_ngd_direction(
                    r_of,
                    flat_p,
                    state,
                    rank=rank,
                    cg_tol=spec._cg_tol,
                    cg_maxiter=spec._cg_maxiter,
                    damping=spec._damping,
                )

        return NystromNGDCallback(gram_terms)

    def __repr__(self):
        return (
            f"NystromNGDOptimizer(rank={self._rank}, damping={self._damping}, cg_tol={self._cg_tol}, "
            f"cg_maxiter={self._cg_maxiter})"
        )


def nystrom_ngd(
    gram_terms=None,
    *,
    rank: int = 100,
    adaptive_rank: bool = True,
    rank_step: int = 20,
    rank_max: int = 500,
    damping=None,
    cg_tol: float = 1e-10,
    cg_maxiter: int = 20,
    seed: int = 0,
) -> NystromNGDOptimizer:
    """Matrix-free natural gradient with a randomized Nyström preconditioner and Armijo steps.

    Pass to ``net.optimizer(...)``. The Gramian :math:`J^\\top J` is only ever applied (a JVP and a
    VJP of the residual), never formed, so memory stays at a few vectors of length ``P`` plus the
    ``P x rank`` sketch: this is the natural-gradient method for networks too large for
    :func:`engd` or :func:`lm`. Each epoch draws a new sketch, solves the damped system by at most
    ``cg_maxiter`` preconditioned CG iterations, and backtracks along the result. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs.
        rank: Size ℓ of the first Nyström sketch (default 100; each column costs one Gramian product
            per epoch).
        adaptive_rank: Adapt ℓ between epochs as in the paper's Algorithm 4 (default ``True``);
            each new size compiles the step once.
        rank_step: The paper's :math:`\\delta_\\ell` (default 20).
        rank_max: The paper's :math:`\\ell_{\\max}` (default 500; capped at the parameter count).
        damping: ``None`` (default) uses :math:`\\mu = \\max(10\\,\\epsilon\\hat\\lambda_1, 10^{-4}L^2)`,
            which vanishes as the loss does (a PINN whose residual can reach zero); for a loss with a
            positive minimum (noisy data) it stays at :math:`10^{-4}L_\\min^2` and slows the final
            digits, so fix a float there.
        cg_tol: Relative-residual tolerance of CG (default ``1e-10``).
        cg_maxiter: CG iterations per epoch (default 20, the paper's setting).
        seed: Seed of the sketches.

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.nystrom_ngd(rank=200))
        jno.core([pde.mse, bc.mse]).solve(500)

    References:
        Bioli, Marcati & Sangalli, *Accelerating Natural Gradient Descent for PINNs with Randomized
        Numerical Linear Algebra*, CMAME (2026), doi:10.1016/j.cma.2026.119456, Eqs. (7), (15),
        Algorithms 1, 3, 4 and 6. Frangella, Tropp & Udell, *Randomized Nyström Preconditioning*,
        SIAM J. Matrix Anal. Appl. 44(2), 2023.
    """
    return NystromNGDOptimizer(
        gram_terms,
        rank=rank,
        adaptive_rank=adaptive_rank,
        rank_step=rank_step,
        rank_max=rank_max,
        damping=damping,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        seed=seed,
    )
