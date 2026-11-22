"""Pseudo-transient continuation: a steady solve that marches in pseudo-time until it is steady.

Based on:
    Kelley, C. T. & Keyes, D. E. (1998). Convergence analysis of pseudo-transient continuation. SIAM Journal on
    Numerical Analysis 35, 508-523.
    Mulder, W. A. & van Leer, B. (1985). Experiments with implicit upwind methods for the Euler equations.
    Journal of Computational Physics 59, 232-246 (switched evolution relaxation, SER).

Each step solves ``(J + D/Δτ) δ = -F(u)`` and takes ``u + δ``: a backward-Euler step of ``D u̇ = -F(u)``. Far from
the root a small ``Δτ`` keeps the steps on the transient path (where Newton from a poor start jumps off, a shock
or a flame front it has not formed yet), and ``Δτ`` grows as the residual falls (``Δτ ← Δτ ‖F_prev‖ / ‖F‖``), so the
end is Newton's quadratic convergence. ``D = |diag J|`` makes ``Δτ`` dimensionless (``Δτ = 1`` doubles the
diagonal) without asking the problem for a mass; a row with no diagonal (a saddle point's constraint) stays
algebraic, as a pressure row should.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class PseudoTransient:
    """``jno.solve.ptc(...)``: a nonlinear driver with the Newton driver's calling convention."""

    direct = True  # it needs the assembled tangent
    name = "ptc"

    def __init__(self, dtau0=1.0, dtau_max=1e12, max_steps=200, rtol=1e-10, atol=1e-12):
        if not dtau0 > 0 or not dtau_max >= dtau0:
            raise ValueError(f"jno.solve.ptc: need 0 < dtau0 <= dtau_max (got {dtau0}, {dtau_max}).")
        self.dtau0, self.dtau_max = float(dtau0), float(dtau_max)
        self.max_steps, self.rtol, self.atol = int(max_steps), float(rtol), float(atol)
        self.cache_key = ("ptc", self.dtau0, self.dtau_max, self.max_steps, self.rtol, self.atol)

    def __repr__(self):
        return (
            f"jno.solve.ptc(dtau0={self.dtau0:g}, dtau_max={self.dtau_max:g}, max_steps={self.max_steps}, "
            f"rtol={self.rtol:g}, atol={self.atol:g})"
        )

    def __hash__(self):
        return hash(self.cache_key)

    def __eq__(self, other):
        return isinstance(other, PseudoTransient) and other.cache_key == self.cache_key

    def __call__(self, residual_fn, u0, *, linear_solve=None, jacobian=None, project=None, **_):
        import jax.experimental.sparse as jsparse

        if jacobian is None:
            jacobian = jax.jacfwd(residual_fn)
        if linear_solve is None:
            from ... import solve as _solve
            from .solver_api import compose_linear_solve_fn

            linear_solve = compose_linear_solve_fn(_solve.lu(), None, None)

        def damped(J, dtau):
            if isinstance(J, jsparse.BCOO):
                n = J.shape[0]
                D = jax.ops.segment_sum(jnp.abs(J.data), J.indices[:, 0], num_segments=n)
                idx = jnp.concatenate([J.indices, jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)])
                return jsparse.BCOO((jnp.concatenate([J.data, D / dtau]), idx), shape=J.shape)
            J = jnp.asarray(J)
            return J + jnp.diag(jnp.abs(J).sum(axis=1)) / dtau

        u0 = jnp.asarray(u0)
        n0 = jnp.linalg.norm(residual_fn(u0))
        tol = self.atol + self.rtol * n0
        tiny = jnp.finfo(u0.dtype).tiny

        def cond(c):
            _, dtau, _, k, n_now = c
            return (n_now > tol) & (k < self.max_steps) & (dtau > 1e-12 * self.dtau0)

        def body(c):
            u, dtau, n_prev, k, n_now = c
            # SER, at most doubling per step
            grow = jnp.minimum(n_prev / jnp.maximum(n_now, tiny), 2.0)
            dtau = jnp.where(k > 0, jnp.minimum(dtau * grow, self.dtau_max), dtau)
            du = linear_solve(damped(jacobian(u), dtau), -residual_fn(u))
            cand = u + du
            if project is not None:
                cand = project(cand)
            n_cand = jnp.linalg.norm(residual_fn(cand))
            # a step that leaves the problem (a negative density, an overflow) or blows the residual up is
            # refused, and the pseudo-time step cut: the march takes the smaller step from the same state
            ok = jnp.isfinite(n_cand) & (n_cand <= 100.0 * n_now)
            u = jnp.where(ok, cand, u)
            n_new = jnp.where(ok, n_cand, n_now)
            dtau = jnp.where(ok, dtau, 0.1 * dtau)
            return u, dtau, jnp.where(ok, n_now, n_prev), k + 1, n_new

        init = (u0, jnp.asarray(self.dtau0, u0.dtype), n0, jnp.asarray(0), n0)
        u, _, _, k, n = jax.lax.while_loop(cond, body, init)
        self.last = (k, n, tol)  # concrete only when called eagerly
        return u
