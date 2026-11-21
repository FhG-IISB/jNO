# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Self-Scaled Broyden (SSBroyden/SSBFGS) optimizer.

Ported from the PyTorch implementation provided by
SciMBA (https://www.scimba.org/).

Based on:
  Urbán, J. F., Stefanou, P., & Pons, J. A. (2025).
  Unveiling the optimization process of physics informed neural networks:
  How accurate and competitive can PINNs be?.
  Journal of Computational Physics, 523, 113656.

The limited-memory form (``pairs=m``) is an addition of jNO. It shares its aim with
  Liu, D. C., & Nocedal, J. (1989).
  On the limited memory BFGS method for large scale optimization.
  Mathematical Programming, 45, 503-528,
but not the construction: L-BFGS rebuilds the metric from the last ``m`` pairs at every step, while
here the metric of the dense solver is kept on a subspace of at most ``m`` dimensions.

With ``Q`` an ``m x P`` matrix of orthonormal rows, the inverse metric is

.. math::

    H = c\,(I - Q^\top Q) + Q^\top H_Q\, Q,

an ``m x m`` matrix :math:`H_Q` on the span of the rows and the scalar :math:`c` on its complement
(at the start ``Q`` is empty and :math:`H = c I`). A new pair :math:`(s, y)` first extends ``Q`` by
the parts of :math:`s` and :math:`y` outside the span (Gram-Schmidt, two passes; in practice one new
row per pair, as :math:`s` lies in the span of ``Q`` and the last gradient). Then the update is the
dense formula applied to :math:`H_Q` with the coordinates :math:`Q s, Q y`, and
:math:`c \leftarrow c / \tau`: the arithmetic of the dense solver in fewer dimensions, and no
``P x P`` matrix is formed. Until ``Q`` is full the steps are those of the dense solver.

When fewer than two free rows are left, the metric is projected: of the eigen-directions of
:math:`H_Q`, the quarter of lowest curvature (the largest eigenvalues of :math:`H_Q`) is dropped,
``Q`` becomes the remaining eigenvectors and :math:`H_Q` diagonal, and :math:`c` is set to the
geometric mean of the dropped eigenvalues, so the dropped directions keep a metric of their own
magnitude. The projection keeps :math:`H` positive definite, and the next update is consistent with
it, since its step is taken with the projected metric. Which directions to keep and what to give the
dropped ones was measured on a PINN with 1185 parameters and 100 rows (2026-11-21): this rule ended
at a 12 times lower loss than leaving :math:`c` as the self-scaling made it, which by then lay above
every eigenvalue of :math:`H_Q` and made the line search backtrack; keeping the low-curvature
directions instead did not converge.

The memory is ``m P`` numbers and a step costs six products with ``Q``.

Two forms that were tried first and failed on PINNs (2026-11-21), for the record: keeping
:math:`H` in the basis of the stored vectors themselves (the compact form of Byrd, Nocedal &
Schnabel, Math. Program. 63 (1994) 129-156) loses positive definiteness after a few hundred steps,
that basis being ill-conditioned once the steps shrink; and rebuilding :math:`H_Q` from the last
``m`` pairs when old ones are dropped needs :math:`s^\top H^{-1} s` in a metric the steps were not
taken with, and lost positive definiteness at ``m = 300``.

A step that raises the loss by more than 10 % is rejected and the metric restarted: the zoom line
search returns a step even when it fails, and without this check limited-memory runs were lost to
one such step.
"""

from typing import Any, NamedTuple, Optional, Union

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
from optax._src import base, combine, numerics, transform
from optax._src import linesearch as _linesearch
from optax.tree_utils import tree_get as _tree_get
from optax.tree_utils import tree_set as _tree_set


class ScaleBySSQuasiNewtonState(NamedTuple):
    """State for the Self-Scaled Broyden/BFGS solver.

    Attributes:
      count: iteration counter.
      flat_params: flattened parameters from the previous step.
      flat_updates: flattened gradients from the previous step.
      hessian_inv: dense inverse Hessian approximation, shape ``[n, n]``.
      flat_prec_grad: flattened preconditioned gradient ``H_k g_k`` from the
        previous step, used to recover the effective step size.
    """

    count: jax.typing.ArrayLike
    flat_params: jax.typing.ArrayLike
    flat_updates: jax.typing.ArrayLike
    hessian_inv: jax.typing.ArrayLike
    flat_prec_grad: jax.typing.ArrayLike


def _ss_coefficients(
    yk_dot_Hkyk: jax.Array,
    yk_dot_sk: jax.Array,
    alpha_sk_dot_gk: jax.Array,
    n: int,
    method: str,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""The scalars of one self-scaled update: :math:`\tau_k`, :math:`\phi_k` and :math:`\theta_k`.

    ``alpha_sk_dot_gk`` is :math:`\alpha_k s_k^\top g_k = -s_k^\top B_k s_k`, the curvature of the
    metric along the step. :math:`\phi_k` weights the Broyden term of the inverse form
    (:math:`H`, 1 for BFGS) and :math:`\theta_k` that of the direct form (:math:`B`, 0 for BFGS); they
    are the same member of the Broyden class.
    """
    safe_yk_dot_sk = jnp.where(jnp.abs(yk_dot_sk) > 0, yk_dot_sk, jnp.ones_like(yk_dot_sk))

    # SS-BFGS tau
    denominator = alpha_sk_dot_gk
    safe_denom = jnp.where(jnp.abs(denominator) > 0, denominator, 1.0)
    tau_k = jnp.where(
        jnp.abs(denominator) > 0,
        jnp.minimum(1.0, -yk_dot_sk / safe_denom),
        1.0,
    )
    phi_k = jnp.ones_like(tau_k)
    theta_k = jnp.zeros_like(tau_k)

    if method == "ssbroyden":
        b_k = -denominator / safe_yk_dot_sk
        h_k = yk_dot_Hkyk / safe_yk_dot_sk
        a_k = h_k * b_k - 1.0
        # c_k = sqrt(a_k / (a_k + 1));
        # guard against negative or zero denominator
        safe_a_k_ratio = jnp.where(
            (a_k > 0) & (a_k + 1.0 > 0),
            a_k / (a_k + 1.0),
            jnp.zeros_like(a_k),
        )
        c_k = jnp.sqrt(jnp.maximum(safe_a_k_ratio, 0.0))
        rhom_k = jnp.minimum(1.0, h_k * (1.0 - c_k))

        safe_a_k = jnp.where(jnp.abs(a_k) > 0, a_k, jnp.ones_like(a_k))
        thetam_k = (rhom_k - 1.0) / safe_a_k
        safe_rhom_k = jnp.where(jnp.abs(rhom_k) > 0, rhom_k, jnp.ones_like(rhom_k))
        thetap_k = 1.0 / safe_rhom_k

        safe_b_k = jnp.where(jnp.abs(b_k) > 0, b_k, jnp.ones_like(b_k))
        theta_k = jnp.maximum(
            thetam_k,
            jnp.minimum(thetap_k, (1.0 - b_k) / safe_b_k),
        )

        sigma_k = 1.0 + a_k * theta_k
        exp = -1.0 / jnp.maximum(n - 1, 1)
        sigma_k_pow = jnp.where(n > 1, sigma_k**exp, 1.0)

        safe_theta_k = jnp.where(
            jnp.abs(theta_k) > 0,
            theta_k,
            jnp.ones_like(theta_k),
        )
        tau_k = jnp.where(
            theta_k > 0,
            tau_k * jnp.minimum(sigma_k_pow, 1.0 / safe_theta_k),
            jnp.minimum(tau_k * sigma_k_pow, sigma_k),
        )
        phi_k = (1.0 - theta_k) / (1.0 + a_k * theta_k)

    return tau_k, phi_k, theta_k


def _ss_update(
    hessian_inv: jax.Array,
    s_k: jax.Array,
    y_k: jax.Array,
    alpha_sk_dot_gk: jax.Array,
    n: int,
    method: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """One self-scaled update of ``hessian_inv``; also returns ``tau_k``, ``theta_k`` and whether it was applied.

    ``n`` is the number of parameters (``hessian_inv`` may be the metric on a subspace). The update
    is skipped (``hessian_inv`` returned, ``tau_k = 1``) when it is not finite or the curvature
    :math:`y^\\top s` is not positive.
    """
    Hk_yk = hessian_inv @ y_k
    yk_dot_Hkyk = y_k @ Hk_yk
    yk_dot_sk = y_k @ s_k

    # Safeguard denominators
    safe_yk_dot_Hkyk = jnp.where(jnp.abs(yk_dot_Hkyk) > 0, yk_dot_Hkyk, jnp.ones_like(yk_dot_Hkyk))
    safe_yk_dot_sk = jnp.where(jnp.abs(yk_dot_sk) > 0, yk_dot_sk, jnp.ones_like(yk_dot_sk))

    v_k = jnp.sqrt(jnp.maximum(yk_dot_Hkyk, 0.0)) * (s_k / safe_yk_dot_sk - Hk_yk / safe_yk_dot_Hkyk)

    tau_k, phi_k, theta_k = _ss_coefficients(yk_dot_Hkyk, yk_dot_sk, alpha_sk_dot_gk, n, method)

    safe_tau_k = jnp.where(jnp.abs(tau_k) > 0, tau_k, jnp.ones_like(tau_k))

    temp1 = jnp.outer(Hk_yk, Hk_yk) / safe_yk_dot_Hkyk
    temp2 = phi_k * jnp.outer(v_k, v_k)
    temp3 = jnp.outer(s_k, s_k) / safe_yk_dot_sk
    H_new = (1.0 / safe_tau_k) * (hessian_inv - temp1 + temp2) + temp3

    # Guard against NaN or non-positive curvature
    valid = ~jnp.any(jnp.isnan(H_new)) & ~jnp.any(jnp.isinf(H_new)) & (yk_dot_sk > 0)
    H_new = jnp.where(valid, H_new, hessian_inv)
    return H_new, jnp.where(valid, safe_tau_k, jnp.ones_like(safe_tau_k)), theta_k, valid


def _update_hessian_inv(
    hessian_inv: jax.Array,
    s_k: jax.Array,
    y_k: jax.Array,
    alpha_k: jax.Array,
    flat_prev_grad: jax.Array,
    method: str,
) -> jax.Array:
    """Update the inverse Hessian approximation using SS-BFGS or SS-Broyden."""
    sk_dot_gk = s_k @ flat_prev_grad
    return _ss_update(hessian_inv, s_k, y_k, alpha_k * sk_dot_gk, s_k.shape[0], method)[0]


# ---------------------------------------------------------------------------------------------
# Limited memory: the dense metric on a subspace of at most ``pairs`` dimensions
# ---------------------------------------------------------------------------------------------

_DROP_FRACTION = 0.25  # share of the directions dropped when the memory is full
_MAX_LOSS_INCREASE = 0.1  # a step raising the loss by more than this (relative) is rejected
_SPAN_RTOL = 1e3  # times machine epsilon: a vector this close to the span adds no basis row
_MIN_PAIRS = 4


class ScaleByLimitedSSQuasiNewtonState(NamedTuple):
    """State of the limited-memory Self-Scaled Broyden/BFGS solver.

    Attributes:
      count: iteration counter.
      flat_params: flattened parameters from the previous step.
      flat_updates: flattened gradients from the previous step.
      flat_prec_grad: flattened ``H_k g_k`` from the previous step (recovers the step size).
      basis: ``Q``, ``[m, n]``; its first ``rank`` rows are orthonormal, the others are not read.
      scale: ``c``, the inverse metric on the complement of the span of ``Q``.
      hessian_inv: ``H_Q``, ``[m, m]``, the inverse metric on the span; zero outside the first
        ``rank`` rows and columns.
      rank: basis rows in use.
    """

    count: jax.typing.ArrayLike
    flat_params: jax.typing.ArrayLike
    flat_updates: jax.typing.ArrayLike
    flat_prec_grad: jax.typing.ArrayLike
    basis: jax.typing.ArrayLike
    scale: jax.typing.ArrayLike
    hessian_inv: jax.typing.ArrayLike
    rank: jax.typing.ArrayLike


def _limited_project(basis, hessian_inv, scale, rank, keep: int):
    r"""Project the metric onto the ``keep`` eigen-directions of ``H_Q`` of highest curvature.

    Those are the smallest eigenvalues of the inverse metric :math:`H_Q`. Returns the new
    ``(Q, H_Q, c)``: the kept eigenvectors (in parameter space), the diagonal matrix of their
    eigenvalues, and :math:`c` set to the geometric mean of the eigenvalues dropped, so the dropped
    directions keep a metric of their own magnitude. An eigenvalue that is not positive (round-off)
    is dropped first and does not enter that mean.
    """
    m = hessian_inv.shape[0]
    dtype = hessian_inv.dtype
    active = jnp.arange(m) < rank
    # an unused row is decoupled: it gets an eigenvalue of its own and is told apart by its vector
    H = 0.5 * (hessian_inv + hessian_inv.T) + scale * jnp.diag((~active).astype(dtype))
    lam, vecs = jnp.linalg.eigh(H)
    good = (lam > 0) & active[jnp.argmax(jnp.abs(vecs), axis=0)]
    log_lam = jnp.log(jnp.where(good, lam, 1.0))
    order = jnp.argsort(jnp.where(good, log_lam, jnp.inf))  # highest curvature first
    kept, dropped = order[:keep], order[keep:]
    weight = good[dropped].astype(dtype)
    n_dropped = weight.sum()
    new_scale = jnp.where(
        n_dropped > 0, jnp.exp((weight * log_lam[dropped]).sum() / jnp.maximum(n_dropped, 1.0)), scale
    ).astype(dtype)
    lam_kept = jnp.where(good[kept], lam[kept], new_scale)
    rotation = jnp.zeros((m, m), dtype).at[:, :keep].set(vecs[:, kept] * good[kept].astype(dtype))
    new_basis = rotation.T @ (basis * active[:, None].astype(dtype))
    new_hessian_inv = jnp.zeros((m, m), dtype).at[jnp.arange(keep), jnp.arange(keep)].set(lam_kept)
    return new_basis, new_hessian_inv, new_scale


def _scale_by_limited_ss_quasi_newton(method: str, scale_init_precond: bool, pairs: int) -> base.GradientTransformation:
    """The limited-memory branch of :func:`scale_by_ss_quasi_newton`."""
    m = int(pairs)
    keep = m - max(2, int(m * _DROP_FRACTION))  # a pair adds at most two rows

    def init_fn(params: base.Params) -> ScaleByLimitedSSQuasiNewtonState:
        flat_params, _ = jax.flatten_util.ravel_pytree(params)
        n = flat_params.shape[0]
        dtype = flat_params.dtype
        return ScaleByLimitedSSQuasiNewtonState(
            count=jnp.zeros([], jnp.int32),
            flat_params=jnp.zeros(n, dtype=dtype),
            flat_updates=jnp.zeros(n, dtype=dtype),
            flat_prec_grad=jnp.zeros(n, dtype=dtype),
            basis=jnp.zeros((m, n), dtype=dtype),
            scale=jnp.ones([], dtype=dtype),
            hessian_inv=jnp.zeros((m, m), dtype=dtype),
            rank=jnp.zeros([], jnp.int32),
        )

    def update_fn(
        updates: base.Updates,
        state: ScaleByLimitedSSQuasiNewtonState,
        params: base.Params,
    ) -> tuple[base.Updates, ScaleByLimitedSSQuasiNewtonState]:
        flat_updates_new, unravel_fn = jax.flatten_util.ravel_pytree(updates)
        flat_params_new = jax.flatten_util.ravel_pytree(params)[0]
        n = flat_params_new.shape[0]
        dtype = flat_params_new.dtype
        slots = jnp.arange(m)
        tol = _SPAN_RTOL * jnp.finfo(dtype).eps

        # --- 1. The pair of the previous step, as in the dense solver ---
        s_k = flat_params_new - state.flat_params
        y_k = flat_updates_new - state.flat_updates
        prec_sq = state.flat_prec_grad @ state.flat_prec_grad
        alpha_k = jnp.where(prec_sq > 0, -(s_k @ state.flat_prec_grad) / jnp.where(prec_sq > 0, prec_sq, 1.0), 0.0)
        alpha_sk_dot_gk = alpha_k * (s_k @ state.flat_updates)

        def store(st):
            basis, scale, H, rank = st.basis, st.scale, st.hessian_inv, st.rank

            # --- 1a. Extend the basis by the parts of s and y outside it (Gram-Schmidt, two passes) ---
            active = (slots < rank)[:, None].astype(dtype)
            pair = jnp.stack([s_k, y_k], axis=1)  # [n, 2]
            coords = (basis @ pair) * active
            resid = pair - (coords.T @ basis).T
            more = (basis @ resid) * active
            coords, resid = coords + more, resid - (more.T @ basis).T
            s_perp, y_perp = resid[:, 0], resid[:, 1]

            norm_s = jnp.linalg.norm(s_perp)
            add_s = norm_s > tol * jnp.linalg.norm(s_k)
            q_s = jnp.where(add_s, s_perp / jnp.where(add_s, norm_s, 1.0), 0.0)
            y_on_qs = q_s @ y_perp
            y_perp = y_perp - y_on_qs * q_s
            norm_y = jnp.linalg.norm(y_perp)
            add_y = norm_y > tol * jnp.linalg.norm(y_k)
            q_y = jnp.where(add_y, y_perp / jnp.where(add_y, norm_y, 1.0), 0.0)

            i_s = rank  # rank <= m - 2 here: the projection below keeps two rows free
            i_y = rank + add_s.astype(rank.dtype)
            s_hat = coords[:, 0].at[i_s].add(jnp.where(add_s, norm_s, 0.0))
            y_hat = coords[:, 1].at[i_s].add(y_on_qs).at[i_y].add(jnp.where(add_y, norm_y, 0.0))
            # on a new direction the metric is still the one of the complement
            H_ext = H.at[i_s, i_s].add(add_s.astype(dtype) * scale).at[i_y, i_y].add(add_y.astype(dtype) * scale)

            # --- 1b. The dense update, on the span ---
            H_new, tau_k, _, valid = _ss_update(H_ext, s_hat, y_hat, alpha_sk_dot_gk, n, method)
            H = jnp.where(valid, H_new, H)
            scale = scale / tau_k
            basis = basis.at[i_s].set(jnp.where(valid & add_s, q_s, basis[i_s]))
            basis = basis.at[i_y].set(jnp.where(valid & add_y, q_y, basis[i_y]))
            rank = rank + valid.astype(rank.dtype) * (add_s.astype(rank.dtype) + add_y.astype(rank.dtype))

            # --- 1c. Memory full: project onto the directions where the most has been learned ---
            basis, H, scale, rank = jax.lax.cond(
                rank + 2 > m,
                lambda: (*_limited_project(basis, H, scale, rank, keep), jnp.asarray(keep, rank.dtype)),
                lambda: (basis, H, scale, rank),
            )
            return basis, scale, H, rank

        def skip(st):
            return st.basis, st.scale, st.hessian_inv, st.rank

        use = (state.count > 0) & (s_k @ y_k > 0)
        basis, scale, H, rank = jax.lax.cond(use, store, skip, state)

        # --- 2. Scale the initial preconditioner at the first step ---
        if scale_init_precond:
            update_norm = jnp.linalg.norm(flat_updates_new)
            capped_inv_norm = jnp.minimum(1.0, 1.0 / jnp.maximum(update_norm, 1e-30))
            scale = jnp.where(state.count == 0, capped_inv_norm * scale, scale)

        # --- 3. Precondition the current gradient: H g = c g + Q^T (H_Q - c I) Q g ---
        g_hat = (basis @ flat_updates_new) * (slots < rank).astype(dtype)
        prec_grad_flat = scale * flat_updates_new + (H @ g_hat - scale * g_hat) @ basis

        new_state = ScaleByLimitedSSQuasiNewtonState(
            count=numerics.safe_increment(state.count),
            flat_params=flat_params_new,
            flat_updates=flat_updates_new,
            flat_prec_grad=prec_grad_flat,
            basis=basis,
            scale=scale,
            hessian_inv=H,
            rank=rank,
        )
        return unravel_fn(prec_grad_flat), new_state

    return base.GradientTransformation(init_fn, update_fn)


class GuardedSSQuasiNewtonState(NamedTuple):
    """State of a limited-memory solver with its step check.

    Attributes:
      inner: state of the chain (metric, sign, line search).
      rejected: steps rejected so far.
    """

    inner: Any
    rejected: jax.typing.ArrayLike


def _reject_loss_increase(
    inner: base.GradientTransformationExtraArgs, max_increase: float = _MAX_LOSS_INCREASE
) -> base.GradientTransformationExtraArgs:
    r"""Reject a step of ``inner`` that raises the loss by more than ``max_increase`` (relative).

    ``inner`` is a chain whose first state is a :class:`ScaleByLimitedSSQuasiNewtonState`. A rejected
    step leaves the parameters unchanged and restarts the metric from the identity scaled by
    :math:`\min(1, 1/\|g\|)`, so the next step is a line search along the gradient. It costs one more
    loss evaluation per step.
    """

    def init_fn(params):
        return GuardedSSQuasiNewtonState(inner.init(params), jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None, *, value, value_fn, **extra_args):
        new_updates, inner_state = inner.update(updates, state.inner, params, value=value, value_fn=value_fn, **extra_args)
        new_value = value_fn(jax.tree.map(lambda p, u: p + u, params, new_updates))
        ok = jnp.isfinite(new_value) & (new_value <= (1.0 + max_increase) * value)
        new_updates = jax.tree.map(lambda u: jnp.where(ok, u, jnp.zeros_like(u)), new_updates)

        metric = inner_state[0]
        grad_norm = jnp.linalg.norm(jax.flatten_util.ravel_pytree(updates)[0])
        restart = jnp.minimum(1.0, 1.0 / jnp.maximum(grad_norm, 1e-30)).astype(metric.scale.dtype)
        metric = metric._replace(  # basis rows beyond ``rank`` are never read
            scale=jnp.where(ok, metric.scale, restart),
            hessian_inv=jnp.where(ok, metric.hessian_inv, 0.0),
            rank=jnp.where(ok, metric.rank, 0),
        )
        inner_state = (metric, *inner_state[1:])
        # a line search caches the value and gradient at the point it stepped to
        # (optax.value_and_grad_from_state reads them); after a rejection that is not where we are
        cached_value = _tree_get(inner_state, "value")
        if cached_value is not None and "grad" in extra_args:
            inner_state = _tree_set(
                inner_state,
                value=jnp.where(ok, cached_value, value),
                grad=jax.tree.map(
                    lambda new, old: jnp.where(ok, new, old), _tree_get(inner_state, "grad"), extra_args["grad"]
                ),
            )
        rejected = state.rejected + (~ok).astype(state.rejected.dtype)
        return new_updates, GuardedSSQuasiNewtonState(inner_state, rejected)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def scale_by_ss_quasi_newton(
    method: str = "ssbfgs",
    scale_init_precond: bool = True,
    pairs: Optional[int] = None,
) -> base.GradientTransformation:
    r"""Scale updates by the Self-Scaled Broyden/BFGS inverse Hessian.

    This maintains a dense approximation of the inverse Hessian
    :math:`H_k` and returns the preconditioned gradient
    :math:`H_k \nabla f(w_k)`. Unlike L-BFGS, the full
    :math:`n \times n` matrix is stored, so this is suitable only for
    small to medium scale problems.

    The inverse Hessian is updated using a self-scaled formula:

    .. math::

      H_{k+1} = \frac{1}{\tau_k}(H_k
      - \frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}
      + \phi_k v_k v_k^\top) + \frac{s_k s_k^\top}{y_k^\top s_k}

    where the self-scaling factors :math:`\tau_k` and :math:`\phi_k` depend on
    the chosen ``method`` (``'ssbfgs'`` or ``'ssbroyden'``).

    This function is typically chained with a line search such as
    :func:`optax.scale_by_zoom_linesearch`.

    Args:
      method: ``'ssbfgs'`` or ``'ssbroyden'``.
        Controls the self-scaling formula.
      scale_init_precond: if ``True``, scale the initial identity
        preconditioner
        by a capped reciprocal of the gradient norm at the first step.
      pairs: ``None`` (default) keeps the dense :math:`n \times n` matrix, which
        holds every pair. An integer ``m`` keeps the metric on a subspace of
        at most ``m`` dimensions in ``m n`` numbers, without the dense matrix
        (see the module docstring). This transform alone does not check the
        step; :func:`ssbroyden` and :func:`ssbfgs` add that check.

    Returns:
      A :class:`optax.GradientTransformation` object.

    References:
      Urbán et al, `Unveiling the optimization process of physics
      informed neural networks: How accurate and competitive can
      PINNs be?
      <https://doi.org/10.1016/j.jcp.2024.113656>`_, 2025

    .. warning::
      This optimizer stores a dense :math:`n \\times n` matrix
      where :math:`n` is the total number of parameters. It is
      memory intensive and best suited for small to medium scale
      problems.
    """
    if method not in ("ssbfgs", "ssbroyden"):
        raise ValueError(f"method must be 'ssbfgs' or 'ssbroyden', got '{method}'")
    if pairs is not None:
        if isinstance(pairs, bool) or not isinstance(pairs, (int, np.integer)) or pairs < _MIN_PAIRS:
            raise ValueError(f"pairs must be None (all pairs, dense) or an integer >= {_MIN_PAIRS}, got {pairs!r}")
        return _scale_by_limited_ss_quasi_newton(method, scale_init_precond, int(pairs))

    def init_fn(params: base.Params) -> ScaleBySSQuasiNewtonState:
        flat_params, _ = jax.flatten_util.ravel_pytree(params)
        n = flat_params.shape[0]
        dtype = flat_params.dtype
        return ScaleBySSQuasiNewtonState(
            count=jnp.zeros([], jnp.int32),
            flat_params=jnp.zeros(n, dtype=dtype),
            flat_updates=jnp.zeros(n, dtype=dtype),
            hessian_inv=jnp.eye(n, dtype=dtype),
            flat_prec_grad=jnp.zeros(n, dtype=dtype),
        )

    def update_fn(
        updates: base.Updates,
        state: ScaleBySSQuasiNewtonState,
        params: base.Params,
    ) -> tuple[base.Updates, ScaleBySSQuasiNewtonState]:
        flat_updates_new, unravel_fn = jax.flatten_util.ravel_pytree(updates)
        flat_params_new = jax.flatten_util.ravel_pytree(params)[0]

        # --- 1. Update inverse Hessian using info from previous step ---
        s_k = flat_params_new - state.flat_params
        y_k = flat_updates_new - state.flat_updates

        # Zero out at first step (no previous data)
        s_k = jnp.where(state.count > 0, s_k, jnp.zeros_like(s_k))
        y_k = jnp.where(state.count > 0, y_k, jnp.zeros_like(y_k))

        # Recover the effective step size from the previous
        # iteration.  If the chain is
        # scale_by_ss -> scale(-1) -> linesearch, then
        # s_k = -alpha * prec_grad, so
        # alpha = -(s_k . prec_grad) / ||prec_grad||^2
        prec_sq = state.flat_prec_grad @ state.flat_prec_grad
        alpha_k = jnp.where(
            prec_sq > 0,
            -(s_k @ state.flat_prec_grad) / prec_sq,
            0.0,
        )

        hessian_inv = jnp.where(
            state.count > 0,
            _update_hessian_inv(
                state.hessian_inv,
                s_k,
                y_k,
                alpha_k,
                state.flat_updates,
                method,
            ),
            state.hessian_inv,
        )

        # --- 2. Scale the initial preconditioner at the first step ---
        if scale_init_precond:
            update_norm = jnp.linalg.norm(flat_updates_new)
            capped_inv_norm = jnp.minimum(1.0, 1.0 / jnp.maximum(update_norm, 1e-30))
            hessian_inv = jnp.where(state.count == 0, capped_inv_norm * hessian_inv, hessian_inv)

        # --- 3. Precondition the current gradient ---
        prec_grad_flat = hessian_inv @ flat_updates_new
        prec_grad = unravel_fn(prec_grad_flat)

        new_state = ScaleBySSQuasiNewtonState(
            count=numerics.safe_increment(state.count),
            flat_params=flat_params_new,
            flat_updates=flat_updates_new,
            hessian_inv=hessian_inv,
            flat_prec_grad=prec_grad_flat,
        )
        return prec_grad, new_state

    return base.GradientTransformation(init_fn, update_fn)


def ssbroyden(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    scale_init_precond: bool = True,
    linesearch: Optional[
        Union[base.GradientTransformationExtraArgs, base.GradientTransformation]
    ] = _linesearch.scale_by_zoom_linesearch(max_linesearch_steps=20, initial_guess_strategy="one"),
    pairs: Optional[int] = None,
) -> base.GradientTransformationExtraArgs:
    r"""Self-Scaled Broyden optimizer.

    SSBroyden is a quasi-Newton method that maintains a dense approximation of
    the inverse Hessian. Unlike L-BFGS which uses a limited-memory
    approximation, this method stores the full :math:`n \times n` matrix and
    updates it using the self-scaled Broyden formula, yielding improved scaling
    of the search direction at the cost of higher memory usage.

    The inverse Hessian :math:`H_k` is updated each iteration as:

    .. math::

      H_{k+1} = \frac{1}{\tau_k}\!\left(H_k
      - \frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}
      + \phi_k\, v_k v_k^\top\right)
      + \frac{s_k s_k^\top}{y_k^\top s_k}

    where the self-scaling factors :math:`\tau_k, \phi_k` use the Broyden
    formula.

    Args:
      learning_rate: optional global scaling factor, either fixed or evolving
        along iterations with a scheduler. By default the learning rate is
        handled by the line search.
      scale_init_precond: whether to scale the initial identity
        preconditioner by a capped reciprocal of the gradient norm.
      linesearch: an instance of
        :class:`optax.GradientTransformationExtraArgs` such as
        :func:`optax.scale_by_zoom_linesearch` that computes a
        step size satisfying sufficient decrease and curvature
        conditions. Pass ``None`` to disable the line search.
      pairs: size of the memory. ``None`` (default) keeps every secant pair in
        the dense :math:`n \times n` matrix. An integer ``m`` keeps the metric
        on at most ``m`` directions (a pair adds one, at most two) in ``m n``
        numbers and never forms that matrix, for networks too large for it.
        Until the memory is full the steps are those of the dense solver; then
        the quarter of the directions of lowest curvature is dropped. More
        memory, lower error. With ``pairs`` set, a step that
        raises the loss by more than 10 % is rejected and the metric
        restarted, at the cost of one more loss evaluation per step, and
        ``update`` needs ``value`` and ``value_fn`` also when ``linesearch``
        is ``None``.

    Returns:
      A :class:`optax.GradientTransformationExtraArgs` object.

    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)
      >>> solver = optax.contrib.ssbroyden()
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> value_and_grad = optax.value_and_grad_from_state(f)
      >>> for _ in range(5):
      ...   value, grad = value_and_grad(params, state=opt_state)
      ...   updates, opt_state = solver.update(
      ...      grad, opt_state, params,
      ...      value=value, grad=grad, value_fn=f,
      ...   )
      ...   params = optax.apply_updates(params, updates)

    References:
      Urbán et al, `Unveiling the optimization process of
      physics informed neural networks: How accurate and
      competitive can PINNs be?
      <https://doi.org/10.1016/j.jcp.2024.113656>`_, 2025

    .. warning::
      This optimizer stores a dense :math:`n \\times n` matrix
      where :math:`n` is the total number of parameters. It is
      memory intensive and best suited for small to medium
      scale problems.

    .. warning::
      This optimizer works best with a line search (default is a zoom line
      search). See the example above for best use in a non-stochastic setting
      where gradients computed by the line search can be recycled using
      :func:`optax.value_and_grad_from_state`.

    .. seealso:: :func:`optax.contrib.ssbfgs`
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = transform.scale_by_learning_rate(learning_rate)
    if linesearch is None:
        linesearch = base.identity()
    chain = combine.chain(
        scale_by_ss_quasi_newton(method="ssbroyden", scale_init_precond=scale_init_precond, pairs=pairs),
        base_scaling,
        linesearch,
    )
    return chain if pairs is None else _reject_loss_increase(chain)


def ssbfgs(
    learning_rate: Optional[base.ScalarOrSchedule] = None,
    scale_init_precond: bool = True,
    linesearch: Optional[
        Union[base.GradientTransformationExtraArgs, base.GradientTransformation]
    ] = _linesearch.scale_by_zoom_linesearch(max_linesearch_steps=20, initial_guess_strategy="one"),
    pairs: Optional[int] = None,
) -> base.GradientTransformationExtraArgs:
    r"""Self-Scaled BFGS optimizer.

    SSBFGS is a quasi-Newton method that maintains a dense approximation of
    the inverse Hessian. Unlike L-BFGS which uses a limited-memory
    approximation, this method stores the full :math:`n \times n` matrix and
    updates it using the self-scaled BFGS formula, yielding improved scaling
    of the search direction at the cost of higher memory usage.

    The inverse Hessian :math:`H_k` is updated each iteration as:

    .. math::

      H_{k+1} = \frac{1}{\tau_k}\!\left(H_k
      - \frac{H_k y_k y_k^\top H_k}{y_k^\top H_k y_k}
      + v_k v_k^\top\right)
      + \frac{s_k s_k^\top}{y_k^\top s_k}

    where the self-scaling factor :math:`\tau_k` uses the BFGS formula
    (:math:`\phi_k = 1`).

    Args:
      learning_rate: optional global scaling factor, either fixed or evolving
        along iterations with a scheduler. By default the learning rate is
        handled by the line search.
      scale_init_precond: whether to scale the initial identity
        preconditioner by a capped reciprocal of the gradient norm.
      linesearch: an instance of
        :class:`optax.GradientTransformationExtraArgs` such as
        :func:`optax.scale_by_zoom_linesearch` that computes a
        step size satisfying sufficient decrease and curvature
        conditions. Pass ``None`` to disable the line search.
      pairs: size of the memory. ``None`` (default) keeps every secant pair in
        the dense :math:`n \times n` matrix. An integer ``m`` keeps the metric
        on at most ``m`` directions (a pair adds one, at most two) in ``m n``
        numbers and never forms that matrix, for networks too large for it.
        Until the memory is full the steps are those of the dense solver; then
        the quarter of the directions of lowest curvature is dropped. More
        memory, lower error. With ``pairs`` set, a step that
        raises the loss by more than 10 % is rejected and the metric
        restarted, at the cost of one more loss evaluation per step, and
        ``update`` needs ``value`` and ``value_fn`` also when ``linesearch``
        is ``None``.

    Returns:
      A :class:`optax.GradientTransformationExtraArgs` object.

    Examples:
      >>> import optax
      >>> import jax
      >>> import jax.numpy as jnp
      >>> def f(x): return jnp.sum(x ** 2)
      >>> solver = optax.contrib.ssbfgs()
      >>> params = jnp.array([1., 2., 3.])
      >>> print('Objective function: ', f(params))
      Objective function:  14.0
      >>> opt_state = solver.init(params)
      >>> value_and_grad = optax.value_and_grad_from_state(f)
      >>> for _ in range(5):
      ...   value, grad = value_and_grad(params, state=opt_state)
      ...   updates, opt_state = solver.update(
      ...      grad, opt_state, params,
      ...      value=value, grad=grad, value_fn=f,
      ...   )
      ...   params = optax.apply_updates(params, updates)

    References:
      Urbán et al, `Unveiling the optimization process of
      physics informed neural networks: How accurate and
      competitive can PINNs be?
      <https://doi.org/10.1016/j.jcp.2024.113656>`_, 2025

    .. warning::
      This optimizer stores a dense :math:`n \\times n` matrix
      where :math:`n` is the total number of parameters. It is
      memory intensive and best suited for small to medium
      scale problems.

    .. warning::
      This optimizer works best with a line search (default is a zoom line
      search). See the example above for best use in a non-stochastic setting
      where gradients computed by the line search can be recycled using
      :func:`optax.value_and_grad_from_state`.

    .. seealso:: :func:`optax.contrib.ssbroyden`
    """
    if learning_rate is None:
        base_scaling = transform.scale(-1.0)
    else:
        base_scaling = transform.scale_by_learning_rate(learning_rate)
    if linesearch is None:
        linesearch = base.identity()
    chain = combine.chain(
        scale_by_ss_quasi_newton(method="ssbfgs", scale_init_precond=scale_init_precond, pairs=pairs),
        base_scaling,
        linesearch,
    )
    return chain if pairs is None else _reject_loss_increase(chain)
