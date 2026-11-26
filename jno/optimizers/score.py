"""SCORE: a self-concordance-inspired quasi-Newton method for PINNs -- self-scaled Broyden with a
candidate step set by the quasi-Newton decrement and a secant pair shifted by that decrement.

Based on:
  Si, C.-H., An, K., Ma, S., & Yan, M. (2026).
  From Non-Convex Self-Concordant Regularization to Scalable Quasi-Newton Training of PINNs.
  arXiv:2608.04206, Algorithm 1, Eqs. (20)-(24).
Built on the self-scaled Broyden update of :func:`jno.optimizers.ssbroyden`:
  Urbán, J. F., Stefanou, P., & Pons, J. A. (2025).
  Unveiling the optimization process of physics informed neural networks: How accurate and
  competitive can PINNs be? Journal of Computational Physics, 523, 113656.
Implemented from the paper's text; no code was used.

With the inverse metric :math:`H_k \\succ 0` and :math:`g_k = \\nabla L(\\theta_k)`, each step is

.. math::

    d_k = -H_k g_k, \\qquad \\lambda_k = \\sqrt{\\max(g_k^\\top H_k g_k, 0)}, \\qquad
    \\alpha_k^{sc} = \\frac{1}{1 + \\lambda_k},

the decrement and the candidate step of Eqs. (22)-(23). The candidate is kept when it satisfies the
strong Wolfe conditions; otherwise a strong-Wolfe line search along :math:`d_k` sets
:math:`\\alpha_k` (optax's zoom search). With :math:`s_k = \\alpha_k d_k`, :math:`y_k = g_{k+1} - g_k` and
the shift :math:`\\mu_k = \\operatorname{clip}(\\eta\\lambda_k, [\\mu_{\\min}, \\mu_{\\max}])` of Eq. (24),
the inverse metric is updated by self-scaled Broyden with the shifted pair
:math:`(s_k, y_k + \\mu_k s_k)` (Eq. (20)), whose curvature :math:`s_k^\\top y_k + \\mu_k\\|s_k\\|^2` the
shift keeps positive where the loss is not convex. The loss and the Wolfe tests use the unshifted
objective.

Values the paper does not give, chosen here: :math:`\\eta = 10^{-2}`, :math:`\\mu_{\\min} = 0`,
:math:`\\mu_{\\max} = 1`, the Wolfe constants :math:`c_1 = 10^{-4}`, :math:`c_2 = 0.9` (optax's), at most 20
line-search steps, :math:`H_0` the identity scaled by :math:`\\min(1, 1/\\|g_0\\|)` (as ``ssbroyden``), the
self-scaling :math:`\\tau_k, \\phi_k` of ``ssbroyden``'s rule, and its safeguard (the update is skipped
when it is not finite or the shifted curvature is not positive). Like ``ssbroyden`` it stores the
dense ``P x P`` inverse metric.

The shift factor is not tuned: on a 129-parameter Poisson PINN (500 steps from a small initialisation)
``eta=0`` reached rel-L2 2.0e-4, the default ``eta=1e-2`` 8.8e-4 and ``ssbroyden`` 4.5e-4.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.flatten_util
import jax.numpy as jnp
import optax

from .ssbroyden import _update_hessian_inv


class SCOREState(NamedTuple):
    """State of :func:`score`: the inverse metric and what the next secant pair needs."""

    count: jax.Array
    hessian_inv: jax.Array
    flat_params: jax.Array  # θ_{k-1}
    flat_grad: jax.Array  # g_{k-1}
    alpha: jax.Array  # α_{k-1}
    decrement: jax.Array  # λ_{k-1}
    ls_state: Any


def score(
    *,
    eta: float = 1e-2,
    mu_min: float = 0.0,
    mu_max: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_linesearch_steps: int = 20,
) -> optax.GradientTransformationExtraArgs:
    """SCORE: self-scaled Broyden with a decrement-based step and a decrement-shifted secant pair.

    An optax transform that needs ``params``, ``value`` and ``value_fn`` in ``update`` (``jno.core``
    passes them to every optimizer). Each step tries the self-concordant step
    :math:`1/(1 + \\lambda_k)` along the quasi-Newton direction and keeps it if it meets the strong
    Wolfe conditions, else line-searches; one extra gradient per step for the test. It stores a
    dense ``P x P`` matrix, so it suits up to a few thousand parameters. Use float64; the paper
    refines after Adam, as for ``ssbroyden``.

    Args:
        eta: Shift factor :math:`\\eta` in :math:`\\mu_k = \\operatorname{clip}(\\eta\\lambda_k)` (default
            ``1e-2``; not given by the paper).
        mu_min: Lower clip of the shift (default ``0``).
        mu_max: Upper clip of the shift (default ``1``).
        c1: Sufficient-decrease constant of the Wolfe tests (default ``1e-4``).
        c2: Curvature constant of the Wolfe tests (default ``0.9``).
        max_linesearch_steps: Steps of the fallback zoom line search (default 20).

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(optax.adam(1e-3)); crux = jno.core(losses); crux.solve(5000)
        net.optimizer(jno.optimizers.score()); crux.solve(2000)

    References:
        Si, An, Ma & Yan, *From Non-Convex Self-Concordant Regularization to Scalable Quasi-Newton
        Training of PINNs*, 2026, arXiv:2608.04206, Algorithm 1, Eqs. (20)-(24). Urbán, Stefanou &
        Pons, J. Comput. Phys. 523 (2025) 113656 (self-scaled Broyden).
    """
    if not (0.0 <= float(mu_min) <= float(mu_max)):
        raise ValueError(f"jno.optimizers.score: need 0 <= mu_min <= mu_max, got {mu_min!r}, {mu_max!r}.")
    if not (0.0 < float(c1) < float(c2) < 1.0):
        raise ValueError(f"jno.optimizers.score: need 0 < c1 < c2 < 1, got c1={c1!r}, c2={c2!r}.")
    ls = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps, slope_rtol=c1, curv_rtol=c2, initial_guess_strategy="one"
    )

    def init_fn(params):
        flat, _ = jax.flatten_util.ravel_pytree(params)
        n, dtype = flat.shape[0], flat.dtype
        zero = jnp.zeros([], dtype)
        return SCOREState(
            count=jnp.zeros([], jnp.int32),
            hessian_inv=jnp.eye(n, dtype=dtype),
            flat_params=jnp.zeros_like(flat),
            flat_grad=jnp.zeros_like(flat),
            alpha=zero,
            decrement=zero,
            ls_state=ls.init(params),
        )

    def update_fn(updates, state, params=None, *, value=None, value_fn=None, **extra):
        if params is None or value_fn is None:
            raise ValueError(
                "jno.optimizers.score needs params= and value_fn= in update() (jno.core passes both; in a "
                "plain optax loop call opt.update(grads, state, params, value=loss, value_fn=loss_fn))."
            )
        g, unravel = jax.flatten_util.ravel_pytree(updates)
        theta = jax.flatten_util.ravel_pytree(params)[0]
        dtype = theta.dtype

        def f(q):
            return value_fn(unravel(q))

        f0 = f(theta) if value is None else jnp.asarray(value, dtype)

        # 1. the inverse metric from the previous step's shifted secant pair (Eqs. (20), (24))
        s = theta - state.flat_params
        y = g - state.flat_grad
        mu = jnp.clip(eta * state.decrement, mu_min, mu_max)
        H = jnp.where(
            state.count > 0,
            _update_hessian_inv(state.hessian_inv, s, y + mu * s, state.alpha, state.flat_grad, "ssbroyden"),
            jnp.minimum(1.0, 1.0 / jnp.maximum(jnp.linalg.norm(g), 1e-30)) * state.hessian_inv,
        )

        # 2. direction, decrement and the self-concordant candidate (Eqs. (22)-(23))
        d = -(H @ g)
        gd = jnp.vdot(g, d)
        lam = jnp.sqrt(jnp.maximum(-gd, 0.0))
        a_sc = 1.0 / (1.0 + lam)
        f_sc, g_sc = jax.value_and_grad(f)(theta + a_sc * d)
        wolfe = (
            jnp.isfinite(f_sc) & (f_sc <= f0 + c1 * a_sc * gd) & (jnp.abs(jnp.vdot(g_sc, d)) <= c2 * jnp.abs(gd)) & (gd < 0)
        )

        # 3. otherwise a strong-Wolfe line search along d
        def fallback(ls_state):
            step, ls_state = ls.update(unravel(d), ls_state, params, value=f0, grad=updates, value_fn=value_fn)
            return jax.flatten_util.ravel_pytree(step)[0], ls_state

        step, ls_state = jax.lax.cond(wolfe, lambda st: (a_sc * d, st), fallback, state.ls_state)
        alpha = jnp.where(wolfe, a_sc, jnp.vdot(step, d) / jnp.maximum(jnp.vdot(d, d), jnp.finfo(dtype).tiny))
        new_state = SCOREState(
            count=state.count + 1,
            hessian_inv=H,
            flat_params=theta,
            flat_grad=g,
            alpha=alpha,
            decrement=lam,
            ls_state=ls_state,
        )
        return unravel(step), new_state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
