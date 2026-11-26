"""NysNewton-CG (NNCG): a damped Newton step on the PINN loss, solved by conjugate gradients with a
randomized Nyström preconditioner of the Hessian, followed by Armijo backtracking.

Based on:
  Rathore, P., Lei, W., Frangella, Z., Lu, L., & Udell, M. (2024).
  Challenges in Training PINNs: A Loss Landscape Perspective.
  ICML 2024, arXiv:2402.01868, Algorithms 4-7 and Appendix E.2.
Implemented from the paper's text; no code was used.

Each step solves (Algorithm 4)

.. math::

    (H + \\mu I)\\, d = \\nabla L, \\qquad \\theta \\leftarrow \\theta - t\\, d,

with Hessian-vector products :math:`Hv` by forward-over-reverse differentiation of the loss, by PCG
(Algorithm 6) warm-started from the previous :math:`d`, preconditioned by
:math:`P^{-1} = (\\hat\\lambda_s + \\mu) U(\\hat\\Lambda + \\mu I)^{-1}U^\\top + (I - UU^\\top)` from a rank-``s``
Gaussian Nyström sketch of :math:`H` rebuilt every ``F`` steps (Algorithm 5, with its extra shift
when the sketched core is not positive definite). The step length :math:`t` backtracks from 1 by
:math:`\\beta = 0.5` until :math:`L(\\theta - t d) \\le L(\\theta) - \\alpha\\, t\\, \\nabla L^\\top d` with
:math:`\\alpha = 0.1` (Algorithm 7). Defaults are the paper's: ``rank=60``, ``precond_every=20``,
``cg_tol=1e-16``, ``cg_maxiter=1000``; :math:`\\mu` was tuned there over
:math:`\\{10^{-5}, \\ldots, 10^{-1}\\}`, with :math:`10^{-2}` and :math:`10^{-1}` best, so ``damping=1e-2``.

The paper runs NNCG only after Adam and L-BFGS (41 000 iterations together), as a fine-tuning stage::

    net.optimizer(optax.adam(1e-3)); crux.solve(10_000)
    net.optimizer(jno.optimizers.nncg()); crux.solve(2_000)       # or crux.set_optimizer(...)

Safeguards the paper does not state, added here: :math:`H + \\mu I` need not be positive definite away
from a minimum, so CG stops at the first direction of non-positive curvature (keeping the iterate
reached), and when the result is not a descent direction (:math:`\\nabla L^\\top d \\le 0`) the step
falls back to :math:`d = \\nabla L`; at most 30 backtracking halvings, after which no step is taken.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax

from ._randnla import Nystrom, nystrom_approx, nystrom_precond, pcg
from ._residual_callback import armijo


class NNCGState(NamedTuple):
    """State of :func:`nncg`: the step counter, the sketch key, the current preconditioner and the last step."""

    count: jax.Array
    key: jax.Array
    U: jax.Array
    lam: jax.Array
    d_prev: jax.Array


def nncg(
    damping: float = 1e-2,
    *,
    rank: int = 60,
    precond_every: int = 20,
    cg_tol: float = 1e-16,
    cg_maxiter: int = 1000,
    armijo_alpha: float = 0.1,
    armijo_beta: float = 0.5,
    seed: int = 0,
) -> optax.GradientTransformationExtraArgs:
    """NysNewton-CG: damped Newton by Nyström-preconditioned CG, with Armijo steps.

    An optax transform that needs the loss itself: its ``update`` takes ``value_fn`` (and uses the
    gradient passed as ``updates``), which ``jno.core`` supplies to every optimizer. Each step costs
    up to ``cg_maxiter`` Hessian-vector products (each about two gradients), plus ``rank`` more
    every ``precond_every`` steps, so it is a fine-tuning stage after Adam / L-BFGS, as in the paper.
    Memory: ``rank + 3`` vectors of the parameter size. Use float64.

    Args:
        damping: :math:`\\mu` added to the Hessian (default ``1e-2``; the paper tuned over
            ``1e-5``-``1e-1``).
        rank: Nyström sketch size ``s`` (default 60).
        precond_every: Rebuild the preconditioner every ``F`` steps (default 20).
        cg_tol: Relative-residual tolerance of PCG (default ``1e-16``).
        cg_maxiter: PCG iterations per step (default 1000).
        armijo_alpha: Sufficient-decrease constant (default 0.1).
        armijo_beta: Backtracking factor (default 0.5).
        seed: Seed of the sketches.

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(optax.adam(1e-3)); crux = jno.core(losses); crux.solve(5000)
        net.optimizer(jno.optimizers.nncg()); crux.solve(1000)

    References:
        Rathore, Lei, Frangella, Lu & Udell, *Challenges in Training PINNs: A Loss Landscape
        Perspective*, ICML 2024, arXiv:2402.01868, Algorithms 4-7, Appendix E.2.
    """
    if not float(damping) > 0.0:
        raise ValueError(f"jno.optimizers.nncg: damping must be > 0, got {damping!r}.")
    if int(rank) < 1 or int(precond_every) < 1 or int(cg_maxiter) < 1:
        raise ValueError("jno.optimizers.nncg: rank, precond_every and cg_maxiter must be >= 1.")
    mu = float(damping)

    def init_fn(params):
        flat, _ = jax.flatten_util.ravel_pytree(params)
        P = flat.shape[0]
        s = min(int(rank), P)
        return NNCGState(
            count=jnp.zeros([], jnp.int32),
            key=jax.random.PRNGKey(seed),
            U=jnp.zeros((P, s), flat.dtype),
            lam=jnp.zeros((s,), flat.dtype),
            d_prev=jnp.zeros_like(flat),
        )

    def update_fn(updates, state, params=None, *, value_fn=None, value=None, **extra):
        if params is None or value_fn is None:
            raise ValueError(
                "jno.optimizers.nncg needs params= and value_fn= in update() (jno.core passes both; in a "
                "plain optax loop call opt.update(grads, state, params, value_fn=loss_fn))."
            )
        flat_p, unravel = jax.flatten_util.ravel_pytree(params)
        g, _ = jax.flatten_util.ravel_pytree(updates)
        dtype = flat_p.dtype
        P = flat_p.shape[0]
        s = state.lam.shape[0]

        def f(q):
            return value_fn(unravel(q))

        grad_f = jax.grad(f)

        def hvp(v):
            return jax.jvp(grad_f, (flat_p,), (v,))[1]

        key, sub = jax.random.split(state.key)

        def rebuild(_):
            ny = nystrom_approx(hvp, P, s, sub, dtype)
            return ny.U, ny.lam

        U, lam = jax.lax.cond(state.count % precond_every == 0, rebuild, lambda _: (state.U, state.lam), None)
        ny = Nystrom(U, lam)
        # Algorithm 6's (lam_s + mu) U (Lam + mu)^-1 U^T + (I - U U^T), divided by (lam_s + mu): CG is
        # invariant to that positive factor.
        precond = nystrom_precond(ny, mu, 1.0 / (lam[-1] + mu))
        sol = pcg(lambda v: hvp(v) + mu * v, g, precond, x0=state.d_prev, tol=cg_tol, maxiter=cg_maxiter)
        d = sol.x
        descent = jnp.vdot(g, d) > 0
        d = jnp.where(descent & jnp.all(jnp.isfinite(d)), d, g)
        f0 = f(flat_p) if value is None else jnp.asarray(value, dtype)
        t, _ = armijo(lambda t: f(flat_p - t * d), f0, -jnp.vdot(g, d), c1=armijo_alpha, beta=armijo_beta)
        new_state = NNCGState(state.count + 1, key, U, lam, d)
        return unravel(-t * d), new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
