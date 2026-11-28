"""Mass-matrix handling shared by the schemes that reduce a symmetric-definite pencil ``· = λ M ·`` (or
``M u̇ = ...``) to a plain symmetric form — the generalized eigensolver and the exponential integrator.

Two treatments:

* **lumped** — the row-sum diagonal ``d_i = Σ_j M_ij = ∫ φ_i`` (partition of unity): matrix-free, and it
  gives a discrete maximum principle. ``M^{-1/2}`` is then ``1/√d`` — trivial.
* **consistent** — the full ``M`` via a dense Cholesky ``M = L Lᵀ``: no lumping error, at the cost of a
  factorization (done **once**; a matrix-free M-inner-product Lanczos is the large-``n`` alternative).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _safe_norm(sq):
    """``sqrt(max(sq, 0))`` whose derivative is 0, not ``inf * 0 = NaN``, where ``sq`` is 0."""
    pos = sq > 0.0
    return jnp.where(pos, jnp.sqrt(jnp.where(pos, sq, 1.0)), 0.0)


def _dense(M):
    return jnp.asarray(M.todense() if hasattr(M, "todense") else M)


def lumped_diagonal(M):
    """Row-sum (lumped) mass diagonal ``d`` — one matvec, matrix-free."""
    return M @ jnp.ones(M.shape[0], _dense(M).dtype if not hasattr(M, "data") else jnp.asarray(M.data).dtype)


def cholesky_spd(M):
    """Lower Cholesky factor ``L`` of an SPD (sub)matrix ``M = L Lᵀ`` — the symmetric reduction applies
    ``L⁻¹``/``L⁻ᵀ`` (triangular solves). Symmetrised for roundoff. Dense (moderate ``n``)."""
    Md = _dense(M)
    return jnp.linalg.cholesky(0.5 * (Md + Md.T))


_MIN_STAGE = 16


def _stage_ends(order):
    """Krylov sizes at which the error estimate is read: 16, 32, 64, ... up to ``order`` (geometric, so a
    problem that needs ``m`` vectors pays for at most about ``2 m`` and the small ``eigh`` calls stay small)."""
    ends, b = [], _MIN_STAGE
    while b < order:
        ends.append(b)
        b *= 2
    ends.append(int(order))
    return ends


_WARNED = set()


def _report_cap_reached(bad, est, tol, order):
    """Host side of the traced "cap reached" warning. Run under ``vmap`` (a ``cond`` becomes a ``select``
    and the callback fires either way), so the flag is re-checked here. One warning per cap size (the first estimate is
    reported), so a march of many steps does not repeat it."""
    import warnings

    if not bool(bad):
        return
    key = int(order)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"jno.solve.exponential: the Krylov (Lanczos) space reached its cap of {int(order)} vectors with the "
        f"estimated relative error {float(est):.1e} still above the tolerance {float(tol):.0e}, so this step is "
        "less accurate than asked. Take a smaller dt (the Krylov size needed grows like dt*lambda_max, "
        "lambda_max = largest eigenvalue of M^-1 A, and a stiff plate or beam needs many) or raise order= "
        "(the cap).",
        RuntimeWarning,
        stacklevel=2,
    )


def m_inner_funm(L_mv, m_inner, e0, v, fun, order, *, err_fun=None, tol=None, m_mv=None, return_info=False):
    """``f(L)·v`` where ``L`` is self-adjoint in the **M-inner-product** ``m_inner(a,b) = aᵀ M b`` — via a
    generalized (M-orthonormal) **Lanczos** iteration. Fully **matrix-free** (``L_mv`` applies ``L=M⁻¹A``,
    ``m_inner`` uses only an M-matvec) and **differentiable** (``eigh`` of the small tridiagonal is
    differentiable) — the scalable, autodiff-friendly consistent-mass path, with no dense factorization and
    no host-side interior extraction. ``e0`` is a fixed M-unit fallback for a null start.

    The three-term recurrence ``β_{j+1} q_{j+1} = L q_j − α_j q_j − β_j q_{j-1}`` uses the M-inner-product
    throughout (``α_j = ⟨q_j, L q_j⟩_M``), giving a symmetric tridiagonal ``T``; then
    ``f(L) v = ‖v‖_M · Q f(T) e_1``.

    **Order.** ``order`` is the Krylov size, a fixed one when ``err_fun`` is ``None``. With ``err_fun`` it is
    the **cap** of an adaptive size: after 16, 32, 64, ... (and ``order``) steps the a-posteriori error
    estimate (Saad 1992, "Analysis of some Krylov subspace approximations to the matrix exponential operator",
    SIAM J. Numer. Anal. 29(1), 209-228, Sec. 5.2 / Thm 5.1: the generalized residual
    ``β_{m+1} ‖v‖_M |e_mᵀ φ₁(-τ T_m) e_1| τ`` for ``exp(-τ L) v``; for ``φ_k`` the same with ``φ_{k+1}``)
    is read, and the iteration stops at the first size where it is ``≤ tol ‖v‖_M``, or at the cap. ``err_fun(λ)``
    must return ``τ·φ_{k+1}(-τ λ)`` for ``fun = φ_k(-τ ·)``. At the cap with the estimate still above
    ``tol`` a ``RuntimeWarning`` names the cap, the estimate and the remedy (from inside a trace too).
    The loop is pure JAX: fixed-size buffers, a ``lax.scan`` per stage skipped by a ``lax.cond`` once
    converged. So it jits, vmaps and differentiates in reverse mode, with no host round trip per step (the
    estimate only decides how many stages run; it carries no gradient). Under ``vmap`` the ``cond`` becomes a ``select`` and every stage runs.
    **Reverse-mode memory** stays ``O(order·n)``: each stage is ``jax.checkpoint``-ed (the backward pass re-runs
    it, so only its inputs are saved) and keeps its basis vectors in its own block, closed over by the later
    stages, so no per-step copy of the whole basis is stored. Measured (3 chained Lanczos calls, 2-D Laplacian,
    CPU, peak RSS): n = 5e4, cap 200: 4.3 GB; the per-step-carry design needed about 10x that.
    ``m_mv`` (``b ↦ M b``) speeds the reorthogonalization up (one M-matvec per step); without it ``m_inner``
    is applied once per stored vector. ``return_info`` returns ``(f(L) v, size_used, relative_estimate)``."""
    order = int(order)
    adaptive = err_fun is not None
    ends = _stage_ends(order) if adaptive else [order]
    dt_ = e0.dtype
    beta0 = _safe_norm(m_inner(v, v))
    q0 = jnp.where(beta0 > 1e-300, v / jnp.where(beta0 > 1e-300, beta0, 1.0), e0)
    zero = jnp.zeros((), dt_)

    def project(Qb, w):  # ⟨q_k, w⟩_M for every stored k
        if m_mv is not None:
            return Qb @ m_mv(w)
        return jax.vmap(lambda qk: m_inner(qk, w))(Qb)

    # BREAKDOWN: once the Krylov space is invariant (it always is after as many steps as free unknowns, and
    # a symmetric mesh can get there sooner) the next beta is round-off. Dividing by it builds a garbage
    # vector, and keeping it repeats an eigenvalue of T, where the derivative of `eigh` divides by the gap:
    # the gradient came back NaN (measured: lumped `jno.solve.exponential` with a trainable coefficient).
    # From breakdown on, a step contributes a zero vector, a zero coupling and a diagonal entry above the
    # spectrum, distinct per step: a decoupled block that leaves f(T) e1 unchanged and every eigenvalue simple.
    sizes = [b - a for a, b in zip([0] + ends[:-1], ends)]
    state = dict(
        blocks=tuple(jnp.zeros((L, q0.shape[0]), dt_) for L in sizes),  # q_a .. q_{b-1} of each stage
        q=q0,
        q_prev=jnp.zeros_like(q0),
        beta=zero,
        done=jnp.zeros((), bool),
        scale=zero,
        alphas=jnp.zeros(order, dt_),
        betas=jnp.zeros(order, dt_),
        result=jnp.zeros_like(q0),
        finished=jnp.zeros((), bool),
        est=zero,
        used=jnp.zeros((), jnp.int32),
        bad=jnp.zeros((), bool),
    )

    def make_stage(si, a, b):
        last = b == order

        def run(s):
            prev = s["blocks"][:si]  # earlier stages' vectors: constants of this stage's scan (saved once)

            def step(c, j):
                own, q, q_prev, beta, done, scale, alphas, betas = c
                own = own.at[j - a].set(q)  # q_j joins the basis
                w = L_mv(q)
                alpha = m_inner(q, w)
                w = w - alpha * q - beta * q_prev
                for _ in range(2):  # FULL M-reorthogonalization (twice, classical) -- without it ghost
                    for Qk in (*prev, own):  # eigenvalues corrupt the gradient; rows not yet filled are 0
                        w = w - project(Qk, w) @ Qk
                beta_next = _safe_norm(m_inner(w, w))
                scale = jnp.maximum(scale, jax.lax.stop_gradient(jnp.abs(alpha) + beta + beta_next))
                alpha = jnp.where(done, jax.lax.stop_gradient(scale) * (2.0 + j.astype(dt_)), alpha)
                done = done | (beta_next <= 1e-10 * scale)
                beta_next = jnp.where(done, 0.0, beta_next)
                q_new = jnp.where(done, 0.0, w / jnp.where(done, 1.0, beta_next))
                return (own, q_new, q, beta_next, done, scale, alphas.at[j].set(alpha), betas.at[j].set(beta_next)), None

            c0 = (s["blocks"][si], s["q"], s["q_prev"], s["beta"], s["done"], s["scale"], s["alphas"], s["betas"])
            (own, q, q_prev, beta, done, scale, alphas, betas), _ = jax.lax.scan(step, c0, jnp.arange(a, b))
            blocks = (*prev, own, *s["blocks"][si + 1 :])
            T = jnp.diag(alphas[:b]) + jnp.diag(betas[: b - 1], 1) + jnp.diag(betas[: b - 1], -1)
            T = 0.5 * (T + T.T)

            def combine(y):  # ‖v‖_M · Q y
                out, off = 0.0, 0
                for Qk in (*prev, own):
                    out = out + Qk.T @ y[off : off + Qk.shape[0]]
                    off += Qk.shape[0]
                return beta0 * out

            def final(Tm):  # differentiable f(T) e₁
                evals, evecs = jnp.linalg.eigh(Tm)
                return combine(evecs @ (fun(evals) * evecs[0, :]))

            if adaptive:
                evals, evecs = jnp.linalg.eigh(jax.lax.stop_gradient(T))  # the estimate carries no gradient
                est = jnp.abs(jax.lax.stop_gradient(betas[b - 1]) * jnp.sum(evecs[b - 1, :] * err_fun(evals) * evecs[0, :]))
                # β_{m+1} |e_mᵀ τφ_{k+1}(-τT) e_1|: the error relative to ‖v‖_M
                conv = done | (est <= tol) | last
                bad = last & ~done & (est > tol)
            else:
                est, conv, bad = zero, jnp.asarray(last), jnp.zeros((), bool)
            cand = jax.lax.cond(conv, final, lambda Tm: jnp.zeros_like(q0), T)
            return dict(
                blocks=blocks, q=q, q_prev=q_prev, beta=beta, done=done, scale=scale, alphas=alphas, betas=betas,
                result=jnp.where(conv, cand, s["result"]), finished=conv, est=est,
                used=jnp.asarray(b, jnp.int32), bad=bad,
            )  # fmt: skip

        body = jax.checkpoint(run)  # backward re-runs the stage: only its inputs are saved
        return lambda s: jax.lax.cond(s["finished"], lambda x: x, body, s)

    a = 0
    for si, b in enumerate(ends):
        state = make_stage(si, a, b)(state)
        a = b
    out = state["result"]
    if adaptive:
        rel = state["est"]
        bad = state["bad"]
        jax.lax.cond(
            bad,
            lambda: jax.debug.callback(_report_cap_reached, bad, rel, tol, order),
            lambda: None,
        )
    if return_info:
        return out, state["used"], state["est"]
    return out
