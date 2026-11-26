"""SS-ESOAP: SOAP with a scalar secant-energy correction in its Kronecker geometry, an eigenbasis that
is refreshed only when it has drifted, and a downscaled variance state after each refresh.

Based on:
  Wang, G.-Y., Toftrup, M., Loeschcke, S., Wang, Y., & Anandkumar, A. (2026).
  SS-ESOAP: Self-Scaled Adaptive Preconditioning for Physics-Informed Learning.
  arXiv:2608.29448, Algorithm 2, Eqs. (1), (3)-(6), Appendix C.4.
Built on SOAP (:func:`jno.optimizers.soap`):
  Vyas, N., Morwani, D., Zhao, R., Shapira, I., Brandfonbrener, D., Janson, L., & Kakade, S. (2024).
  SOAP: Improving and Stabilizing Shampoo using Adam. arXiv:2409.11321.
Implemented from the paper's text, with SOAP's preconditioner helpers from ``soap.py``.

For a matrix parameter :math:`W` with gradient :math:`G_t`, the Kronecker factors are
:math:`L_t = \\beta_2 L_{t-1} + (1-\\beta_2) G_tG_t^\\top`, :math:`R_t = \\beta_2 R_{t-1} + (1-\\beta_2) G_t^\\top G_t`
(Eq. (1), from :math:`\\epsilon I`), and Adam runs on :math:`\\tilde G_t = Q_L^\\top G_t Q_R` in their
eigenbasis. Three changes to SOAP (Algorithm 2):

* **Adaptive basis.** Every step the off-diagonal mass
  :math:`\\rho(A, Q) = \\|Q^\\top A Q - \\operatorname{diag}\\|_F / (\\|Q^\\top A Q\\|_F + \\epsilon)` of both factors
  is measured (Eq. (4)); when :math:`\\rho_t = \\max(\\rho_L, \\rho_R) > 0.2` the eigenbases are recomputed
  and the first moment is rotated into them.
* **Variance transition.** The second moment is not rotated but scaled,
  :math:`V \\leftarrow \\gamma V` with :math:`\\gamma = 0.25, 0.5, 0.75` for :math:`\\rho_t > 0.8, > 0.5`, otherwise.
* **Secant-energy correction.** With :math:`S_t = W_{t-1} - W_{t-2}`, :math:`Y_t = G_t - G_{t-1}`,
  :math:`c_t = \\langle Y_t, S_t\\rangle` and :math:`a_t = \\sum_{ij} \\tilde S_{ij}^2 / (D_{L,i} D_{R,j})` in the
  eigenbasis (Eqs. (3), (5)-(6)), :math:`\\tau_t = \\min(1, \\max(\\tau_{\\min}, c_t/a_t))` when both are
  positive, else 1, and the rotated Adam update is multiplied by :math:`\\tau_t^{-1/2}`.

Defaults are the paper's (Table 6): :math:`\\beta_1 = 0.9`, :math:`\\beta_2 = 0.95`, :math:`\\epsilon = 10^{-8}`,
trigger 0.2, checked every step. Values the paper does not give, chosen here:
:math:`\\tau_{\\min} = 10^{-2}` (the update is never scaled up by more than 10), warm-up 0 (the basis
may refresh from the first step), and no weight decay. The paper describes matrices; here every
dimension of a parameter up to ``max_precond_dim`` gets a factor, as in SOAP, a dimension above it
counts as :math:`D = 1`, and a vector parameter (no factor unless ``precondition_1d``) gets plain Adam
with :math:`\\tau = 1`. No line search and no loss value are needed.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from .soap import (
    Preconditioner,
    _is_preconditioner,
    add_decayed_weights_post,
    get_orthogonal_matrix,
    init_conditioner,
    project,
    project_back,
    update_preconditioner,
)

_HI = jax.lax.Precision.HIGHEST


class SSESOAPState(NamedTuple):
    """State of :func:`scale_by_ss_esoap`: SOAP's moments and factors, plus the previous parameters
    and gradient for the secant pair."""

    count: jax.Array
    exp_avg: Any  # rotated first moment
    exp_avg_sq: Any  # second moment in the rotated basis
    GG: Any  # Kronecker factors (Preconditioner per leaf)
    Q: Any  # their eigenbases
    prev_params: Any
    prev_grad: Any


def _rotated_diag(A, Q):
    B = Q.T @ jnp.matmul(A, Q, precision=_HI)
    return B, jnp.diag(B)


def _off_diagonal_mass(A, Q, eps):
    B, d = _rotated_diag(A, Q)
    return jnp.linalg.norm(B - jnp.diag(d)) / (jnp.linalg.norm(B) + eps)


def _leaf_step(g, w, w_prev, g_prev, m, v, gg, q, t, *, b1, b2, eps, tau_min, trigger, check_every, warmup):
    """One SS-ESOAP step for one parameter; returns ``(update, m, v, gg, q)``."""
    gg = update_preconditioner(g, gg, b2, _HI)
    mats = [(i, a, qq) for i, (a, qq) in enumerate(zip(gg.matrices, q.matrices)) if a is not None]

    if mats:
        rho = jnp.max(jnp.stack([_off_diagonal_mass(a, qq, eps) for _, a, qq in mats]))
        do = (t > warmup) & (t % check_every == 0) & (rho > trigger)

        def refresh(args):
            m, v, q = args
            new_q = Preconditioner(get_orthogonal_matrix(a, a.dtype) if a is not None else None for a in gg.matrices)
            m = project(project_back(m, q, _HI), new_q, _HI)
            gamma = jnp.where(rho > 0.8, 0.25, jnp.where(rho > 0.5, 0.5, 0.75))
            return m, gamma * v, new_q

        m, v, q = jax.lax.cond(do, refresh, lambda args: args, (m, v, q))

    gt = project(g, q, _HI)
    m = b1 * m + (1.0 - b1) * gt
    v = b2 * v + (1.0 - b2) * gt * gt

    if mats:
        # secant energy in the Kronecker geometry (Eqs. (3), (5)-(6)); dimensions without a factor have D = 1
        s, y = w - w_prev, g - g_prev
        c = jnp.vdot(y, s)
        st2 = project(s, q, _HI) ** 2
        for i, a, qq in mats:
            _, d = _rotated_diag(a, qq)
            shape = [1] * st2.ndim
            shape[i] = d.shape[0]
            st2 = st2 / jnp.maximum(d, eps).reshape(shape)
        a_t = jnp.sum(st2)
        tau = jnp.where((t > 1) & (c > 0) & (a_t > 0), jnp.minimum(1.0, jnp.maximum(tau_min, c / a_t)), 1.0)
    else:
        tau = 1.0

    m_hat = m / (1.0 - b1**t)
    v_hat = v / (1.0 - b2**t)
    u = project_back(tau ** (-0.5) * m_hat / (jnp.sqrt(v_hat) + eps), q, _HI)
    return u, m, v, gg, q


def scale_by_ss_esoap(
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    *,
    tau_min: float = 1e-2,
    trigger: float = 0.2,
    check_every: int = 1,
    warmup: int = 0,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
) -> optax.GradientTransformation:
    """The SS-ESOAP direction without the learning rate (see the module docstring).

    Args:
        b1: First-moment decay (default 0.9).
        b2: Second-moment and Kronecker-factor decay (default 0.95).
        eps: Adam's epsilon, the factors' initial :math:`\\epsilon I` and the eigenvalue floor.
        tau_min: Floor of the secant-energy scale :math:`\\tau` (default ``1e-2``; not given by the paper).
        trigger: Off-diagonal mass above which the eigenbasis is recomputed (default 0.2).
        check_every: Steps between basis checks (default 1).
        warmup: Steps before the first basis check (default 0; not given by the paper).
        max_precond_dim: Dimensions larger than this get no Kronecker factor (as in SOAP).
        precondition_1d: Give vector parameters a factor too (default ``False``, as in SOAP).

    Returns:
        optax.GradientTransformation: the transform.
    """
    if not (0 <= b1 < 1 and 0 <= b2 < 1):
        raise ValueError(f"jno.optimizers.ss_esoap: b1 and b2 must lie in [0, 1), got {b1!r}, {b2!r}.")
    if not (0 < float(tau_min) <= 1):
        raise ValueError(f"jno.optimizers.ss_esoap: tau_min must lie in (0, 1], got {tau_min!r}.")
    if int(check_every) < 1:
        raise ValueError(f"jno.optimizers.ss_esoap: check_every must be >= 1, got {check_every!r}.")
    kw = dict(b1=b1, b2=b2, eps=eps, tau_min=tau_min, trigger=trigger, check_every=check_every, warmup=warmup)

    def init_fn(params):
        def factors(p, fill):
            pc = init_conditioner(p, max_precond_dim, precondition_1d, p.dtype)
            return pc.map(lambda a: None if a is None else fill(a.shape[0], p.dtype))

        zeros = jtu.tree_map(jnp.zeros_like, params)
        return SSESOAPState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=zeros,
            exp_avg_sq=zeros,
            GG=jtu.tree_map(lambda p: factors(p, lambda n, dt: eps * jnp.eye(n, dtype=dt)), params),
            Q=jtu.tree_map(lambda p: factors(p, lambda n, dt: jnp.eye(n, dtype=dt)), params),
            prev_params=zeros,
            prev_grad=zeros,
        )

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("jno.optimizers.ss_esoap needs params= in update() for its secant pair.")
        t = state.count + 1
        g_leaves, treedef = jtu.tree_flatten(updates)
        cols = [
            jtu.tree_leaves(tree, is_leaf=_is_preconditioner)
            for tree in (params, state.prev_params, state.prev_grad, state.exp_avg, state.exp_avg_sq, state.GG, state.Q)
        ]
        out = [_leaf_step(g, *rest, t.astype(g.dtype), **kw) for g, *rest in zip(g_leaves, *cols)]
        u, m, v, gg, q = (treedef.unflatten([o[k] for o in out]) for k in range(5))
        new_state = SSESOAPState(t, m, v, gg, q, params, updates)
        return u, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def ss_esoap(
    learning_rate: optax.ScalarOrSchedule = 1e-3,
    b1: float = 0.9,
    b2: float = 0.95,
    eps: float = 1e-8,
    *,
    weight_decay: float = 0.0,
    tau_min: float = 1e-2,
    trigger: float = 0.2,
    check_every: int = 1,
    warmup: int = 0,
    max_precond_dim: int = 10000,
    precondition_1d: bool = False,
) -> optax.GradientTransformation:
    """SS-ESOAP: SOAP with a secant-energy step scale, adaptive eigenbasis refreshes and a downscaled
    variance state, for stiff physics-informed training.

    An optax transform; the learning rate is baked in (``net.optimizer(jno.optimizers.ss_esoap(1e-3))``).
    It needs ``params`` in ``update`` for the secant pair (``jno.core`` passes them). Memory: SOAP's two
    factors per matrix parameter plus two more parameter-sized trees.

    Args:
        learning_rate: Step size :math:`\\eta` (default ``1e-3``; the paper tunes it per problem over
            ``1e-4``-``1e-2``).
        b1: First-moment decay (default 0.9).
        b2: Second-moment and Kronecker-factor decay (default 0.95).
        eps: Adam's epsilon (default ``1e-8``).
        weight_decay: Decoupled weight decay :math:`\\lambda` (default 0).
        tau_min: Floor of the secant-energy scale (default ``1e-2``).
        trigger: Off-diagonal mass that triggers a basis refresh (default 0.2).
        check_every: Steps between basis checks (default 1).
        warmup: Steps before the first basis check (default 0).
        max_precond_dim: Dimensions larger than this get no Kronecker factor.
        precondition_1d: Give vector parameters a factor too (default ``False``).

    Example::

        net.optimizer(jno.optimizers.ss_esoap(1e-3))
        jno.core([pde.mse, bc.mse]).solve(20_000)

    References:
        Wang, Toftrup, Loeschcke, Wang & Anandkumar, *SS-ESOAP: Self-Scaled Adaptive Preconditioning for
        Physics-Informed Learning*, 2026, arXiv:2608.29448, Algorithm 2. Vyas et al., *SOAP*, 2024,
        arXiv:2409.11321.
    """
    return optax.chain(
        scale_by_ss_esoap(
            b1,
            b2,
            eps,
            tau_min=tau_min,
            trigger=trigger,
            check_every=check_every,
            warmup=warmup,
            max_precond_dim=max_precond_dim,
            precondition_1d=precondition_1d,
        ),
        optax.scale_by_learning_rate(learning_rate),
        add_decayed_weights_post(weight_decay, learning_rate),
    )
