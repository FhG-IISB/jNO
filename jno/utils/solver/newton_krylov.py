"""Jacobian-free Newton-Krylov root find -- the optimistix-free nonlinear default.

Solves ``residual_fn(u) = 0`` with Newton's method whose per-step linear solve
``J delta = -R`` is done matrix-free: ``J @ v`` comes from a JVP (``jax.linearize``)
so the Jacobian is never formed, and the solve is BiCGStab (handles non-symmetric
``J``). Implicit differentiation is provided by ``jax.lax.custom_root`` (so gradients
reach parameters closed over by ``residual_fn`` without unrolling Newton), and the
inner BiCGStab is wrapped in ``jax.lax.custom_linear_solve`` so the reverse pass --
which solves ``J^T`` -- gets an analytic transpose rule instead of trying to
differentiate the solver's ``while_loop``.

No external solver dependency (no optimistix / lineax).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["newton_krylov", "bicgstab"]

_EPS = 1e-300


def bicgstab(matvec, b, *, tol=1e-10, maxit=2000):
    """Matrix-free BiCGStab; returns ``x`` solving ``matvec(x) = b`` (general matrices)."""
    bnorm = jnp.linalg.norm(b)

    def cond(s):
        _, r, *_, k = s
        return (jnp.linalg.norm(r) > tol * bnorm) & (k < maxit)

    def body(s):
        x, r, rhat, rho, alpha, omega, v, p, k = s
        rho_new = rhat @ r
        beta = (rho_new / (rho + _EPS)) * (alpha / (omega + _EPS))
        p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho_new / (rhat @ v + _EPS)
        sv = r - alpha * v
        t = matvec(sv)
        omega = (t @ sv) / (t @ t + _EPS)
        x = x + alpha * p + omega * sv
        r = sv - omega * t
        return x, r, rhat, rho_new, alpha, omega, v, p, k + 1

    z = jnp.zeros_like(b)
    one = jnp.array(1.0, b.dtype)
    x0 = jnp.zeros_like(b)
    state = (x0, b - matvec(x0), b, one, one, one, z, z, 0)
    x, *_ = jax.lax.while_loop(cond, body, state)
    return x


def _linsolve(matvec, b, *, tol, maxit):
    """Transposable black-box linear solve: BiCGStab forward, BiCGStab on ``J^T`` for the
    reverse rule (via ``custom_linear_solve``) -- so it never differentiates the loop."""
    solve = lambda mv, rhs: bicgstab(mv, rhs, tol=tol, maxit=maxit)
    return jax.lax.custom_linear_solve(matvec, b, solve, transpose_solve=solve)


def newton_krylov(residual_fn, u0, *, rtol=1e-8, atol=1e-8, max_steps=100,
                  inner_tol=1e-10, inner_maxit=2000):
    """Root-find ``residual_fn(u) = 0`` from guess ``u0``; differentiable w.r.t. any value
    ``residual_fn`` closes over. Drop-in for the ``(residual_fn, u0) -> u`` solver contract."""
    # feax residuals can hand back a plain numpy array for concrete inputs; coerce so the
    # jax.lax.custom_root primitive only ever sees JAX values.
    f0 = lambda u: jnp.asarray(residual_fn(u)).reshape(-1)
    u0 = jnp.asarray(u0).reshape(-1)

    def solve(f, x0):
        r0n = jnp.linalg.norm(f(x0))

        def cond(state):
            _, r, k = state
            return (jnp.linalg.norm(r) > atol + rtol * r0n) & (k < max_steps)

        def body(state):
            u, _r, k = state
            ru, jvp = jax.linearize(f, u)           # ru = f(u); jvp(v) = J @ v, reused across inner iters
            delta = _linsolve(jvp, -ru, tol=inner_tol, maxit=inner_maxit)
            u = u + delta
            return u, f(u), k + 1

        u, _r, _k = jax.lax.while_loop(cond, body, (x0, f(x0), 0))
        return u

    tangent_solve = lambda g, y: _linsolve(g, y, tol=inner_tol, maxit=inner_maxit)
    return jax.lax.custom_root(f0, u0, solve, tangent_solve)
