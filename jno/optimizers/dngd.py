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

``solver="nystrom_pcg"`` never forms :math:`K`: it applies :math:`K p = J(J^\\top p)` (one VJP and
one JVP, Sec. 3.3), builds the rank-ℓ landmark Nyström preconditioner of Eq. (11),
:math:`U(\\hat\\Lambda + \\lambda I)^{-1}U^\\top + \\lambda^{-1}(I - UU^\\top)`, once per epoch, and
solves both systems by preconditioned CG (Algorithm 6; tolerance ``1e-10`` and at most 500
iterations, the settings of Appendix Tables 6-7). The landmarks are drawn uniformly; the damping
floor uses the largest Nyström eigenvalue in place of :math:`\\operatorname{tr} K`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from ._randnla import nystrom_approx, nystrom_precond, pcg
from ._residual_callback import grid_line_search, half_sq_norm
from ._residuals import jacobian, linearize, second_directional
from ._sentinel import ResidualOptimizer

_AUTO_DAMPING_CAP = 1e-5  # lambda_k = min(loss, 1e-5), the setting of several benchmarks (Tables 2-8)


def _dngd_damping(damping, loss, scale, dtype):
    lam = jnp.minimum(loss, _AUTO_DAMPING_CAP) if damping == "auto" else jnp.asarray(float(damping), dtype)
    return jnp.maximum(lam, 10.0 * jnp.finfo(dtype).eps * scale)


def _dngd_step(r_of, flat_p, r, JT, solve, *, geodesic: bool, ratio_max: float):
    """The primary step ``v`` and, if accepted, ``v + a/2``; ``solve(b) ≈ (K + λI)^{-1} b``, ``JT(u) = Jᵀu``."""
    v = -JT(solve(r))
    info = {}
    if geodesic:
        fvv = second_directional(r_of, flat_p, v)
        a = -JT(solve(fvv))
        nv = jnp.linalg.norm(v)
        use = (nv > jnp.finfo(r.dtype).eps * (jnp.linalg.norm(flat_p) + 1.0)) & (2.0 * jnp.linalg.norm(a) <= ratio_max * nv)
        v = jnp.where(use, v + 0.5 * a, v)
        info["geodesic"] = use
    return v, info


def _dngd_finish(r_of, flat_p, r, step, line_search):
    if line_search:
        eta, _ = grid_line_search(lambda e: half_sq_norm(r_of(flat_p + e * step)), dtype=r.dtype)
    else:
        eta = jnp.asarray(1.0, r.dtype)
    return -eta * step, eta


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

    step, info = _dngd_step(r_of, flat_p, r, lambda u: J.T @ u, solve, geodesic=geodesic, ratio_max=ratio_max)
    d, eta = _dngd_finish(r_of, flat_p, r, step, line_search)
    return d, state, {**info, "damping": lam, "eta": eta, "loss": loss}


def _dngd_direction_pcg(
    r_of,
    flat_p,
    key,
    *,
    rank: int,
    damping="auto",
    geodesic=True,
    ratio_max=0.5,
    line_search=True,
    cg_tol=1e-10,
    cg_maxiter=500,
):
    """One matrix-free D-NGD step (Nyström-preconditioned CG on ``K + λI``); ``state`` is a PRNG key."""
    lin = linearize(r_of, flat_p)
    r = lin.r
    m = r.shape[0]
    loss = half_sq_norm(r)

    def kmv(p):
        return lin.jvp(lin.vjp(p))

    key, sub = jax.random.split(key)
    ny = nystrom_approx(kmv, m, min(rank, m), sub, r.dtype, sketch="sampling")
    lam = _dngd_damping(damping, loss, ny.lam[0], r.dtype)
    precond = nystrom_precond(ny, lam, 1.0 / lam)
    iters = []

    def solve(b):
        sol = pcg(lambda p: kmv(p) + lam * p, b, precond, tol=cg_tol, maxiter=cg_maxiter)
        iters.append(sol.iters)
        return sol.x

    step, info = _dngd_step(r_of, flat_p, r, lin.vjp, solve, geodesic=geodesic, ratio_max=ratio_max)
    d, eta = _dngd_finish(r_of, flat_p, r, step, line_search)
    return d, key, {**info, "damping": lam, "eta": eta, "loss": loss, "cg_iters": iters[0]}


_SOLVERS = ("dense", "nystrom_pcg")


# docstring-from: jno.optimizers.dngd.dngd
class DNGDOptimizer(ResidualOptimizer):
    r"""Dual Natural Gradient Descent: the Gauss-Newton step solved over residual rows, not parameters.

    Pass to ``net.optimizer(...)``. Each epoch solves with the residual Gramian
    :math:`K = J J^\top` (``m x m`` for ``m`` residual rows) and takes the step
    :math:`-J^\top (K + \lambda I)^{-1} \hat r`, optionally corrected by geodesic acceleration, along
    a line search. The solve depends on ``m``, not on the number of parameters ``P``, so this is
    the Gauss-Newton method for networks with more parameters than residual rows; with fewer,
    :func:`lm` or :func:`engd` solve the smaller ``P x P`` system. ``solver="dense"`` forms and
    factorises :math:`K` (memory ``m x P`` for :math:`J`); ``"nystrom_pcg"`` only applies it and
    solves by Nyström-preconditioned CG, for ``m`` in the tens of thousands. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs.
        damping: ``"auto"`` (default) uses :math:`\lambda = \min(L, 10^{-5})` with
            :math:`L = \tfrac12\|\hat r\|^2`, the paper's setting for most benchmarks; a float fixes it.
        geodesic_acceleration: Add the second-order correction :math:`a/2` when
            :math:`2\|a\|/\|v\| \le 0.5` (default ``True``; one extra second directional derivative).
        line_search: Pick the step length on the grid :math:`\{0.5^k\}_{k=0}^{30}` (default ``True``).
        solver: ``"dense"`` (default) or ``"nystrom_pcg"`` (matrix-free).
        rank: Nyström landmarks for ``"nystrom_pcg"`` (default 500, capped at ``m``).
        cg_tol: CG relative-residual tolerance for ``"nystrom_pcg"`` (default ``1e-10``).
        cg_maxiter: CG iterations per solve for ``"nystrom_pcg"`` (default 500).
        seed: Seed of the landmark draws.

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.dngd())
        jno.core([pde.mse, bc.mse]).solve(500)

    References:
        Jnini & Vella, *Dual Natural Gradient Descent for Scalable Training of Physics-Informed
        Neural Networks*, 2025, arXiv:2505.21404, Eqs. (6), (7), (10), (11), Algorithms 1, 4 and 6.
    """

    _who = "jno.optimizers.dngd"

    def __init__(
        self,
        gram_terms=None,
        *,
        damping="auto",
        geodesic_acceleration=True,
        line_search=True,
        solver="dense",
        rank=500,
        cg_tol=1e-10,
        cg_maxiter=500,
        seed=0,
    ):
        if damping != "auto" and not float(damping) > 0.0:
            raise ValueError(f"jno.optimizers.dngd: damping must be 'auto' or > 0, got {damping!r}.")
        if solver not in _SOLVERS:
            raise ValueError(f"jno.optimizers.dngd: solver must be one of {_SOLVERS}, got {solver!r}.")
        if int(rank) < 1 or int(cg_maxiter) < 1:
            raise ValueError("jno.optimizers.dngd: rank and cg_maxiter must be >= 1.")
        super().__init__(gram_terms)
        self._damping = damping
        self._geodesic = bool(geodesic_acceleration)
        self._line_search = bool(line_search)
        self._solver, self._rank = solver, int(rank)
        self._cg_tol, self._cg_maxiter, self._seed = float(cg_tol), int(cg_maxiter), int(seed)

    def build_callback(self, gram_terms):
        """Return the callback that computes this optimizer's step inside ``solve()``.

        Args:
            gram_terms: ``(NetworkGradient, weight)`` pairs, one per residual class (auto-detected by
                ``solve()`` unless given to the factory).
        """
        from ._residual_callback import ResidualStepCallback

        spec = self
        common = dict(damping=self._damping, geodesic=self._geodesic, line_search=self._line_search)

        class DNGDCallback(ResidualStepCallback):
            who = "jno.optimizers.dngd"

            def init_state(self, flat_p, m):
                return jax.random.PRNGKey(spec._seed)

            def direction(self, r_of, flat_p, state):
                if spec._solver == "dense":
                    return _dngd_direction(r_of, flat_p, state, **common)
                return _dngd_direction_pcg(
                    r_of,
                    flat_p,
                    state,
                    rank=spec._rank,
                    cg_tol=spec._cg_tol,
                    cg_maxiter=spec._cg_maxiter,
                    **common,
                )

        return DNGDCallback(gram_terms)

    def __repr__(self):
        return (
            f"DNGDOptimizer(damping={self._damping!r}, geodesic_acceleration={self._geodesic}, "
            f"line_search={self._line_search}, solver={self._solver!r})"
        )


def dngd(
    gram_terms=None,
    *,
    damping="auto",
    geodesic_acceleration: bool = True,
    line_search: bool = True,
    solver: str = "dense",
    rank: int = 500,
    cg_tol: float = 1e-10,
    cg_maxiter: int = 500,
    seed: int = 0,
) -> DNGDOptimizer:
    """Dual Natural Gradient Descent: the Gauss-Newton step solved over residual rows, not parameters.

    Pass to ``net.optimizer(...)``. Each epoch solves with the residual Gramian
    :math:`K = J J^\\top` (``m x m`` for ``m`` residual rows) and takes the step
    :math:`-J^\\top (K + \\lambda I)^{-1} \\hat r`, optionally corrected by geodesic acceleration, along
    a line search. The solve depends on ``m``, not on the number of parameters ``P``, so this is
    the Gauss-Newton method for networks with more parameters than residual rows; with fewer,
    :func:`lm` or :func:`engd` solve the smaller ``P x P`` system. ``solver="dense"`` forms and
    factorises :math:`K` (memory ``m x P`` for :math:`J`); ``"nystrom_pcg"`` only applies it and
    solves by Nyström-preconditioned CG, for ``m`` in the tens of thousands. Use float64.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs.
        damping: ``"auto"`` (default) uses :math:`\\lambda = \\min(L, 10^{-5})` with
            :math:`L = \\tfrac12\\|\\hat r\\|^2`, the paper's setting for most benchmarks; a float fixes it.
        geodesic_acceleration: Add the second-order correction :math:`a/2` when
            :math:`2\\|a\\|/\\|v\\| \\le 0.5` (default ``True``; one extra second directional derivative).
        line_search: Pick the step length on the grid :math:`\\{0.5^k\\}_{k=0}^{30}` (default ``True``).
        solver: ``"dense"`` (default) or ``"nystrom_pcg"`` (matrix-free).
        rank: Nyström landmarks for ``"nystrom_pcg"`` (default 500, capped at ``m``).
        cg_tol: CG relative-residual tolerance for ``"nystrom_pcg"`` (default ``1e-10``).
        cg_maxiter: CG iterations per solve for ``"nystrom_pcg"`` (default 500).
        seed: Seed of the landmark draws.

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.dngd())
        jno.core([pde.mse, bc.mse]).solve(500)

    References:
        Jnini & Vella, *Dual Natural Gradient Descent for Scalable Training of Physics-Informed
        Neural Networks*, 2025, arXiv:2505.21404, Eqs. (6), (7), (10), (11), Algorithms 1, 4 and 6.
    """
    return DNGDOptimizer(
        gram_terms,
        damping=damping,
        geodesic_acceleration=geodesic_acceleration,
        line_search=line_search,
        solver=solver,
        rank=rank,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        seed=seed,
    )
