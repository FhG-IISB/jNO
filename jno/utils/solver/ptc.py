"""Pseudo-transient continuation: a steady solve that marches in pseudo-time until it is steady.

Based on:
    Kelley, C. T. & Keyes, D. E. (1998). Convergence analysis of pseudo-transient continuation. SIAM Journal on
    Numerical Analysis 35, 508-523.
    Mulder, W. A. & van Leer, B. (1985). Experiments with implicit upwind methods for the Euler equations.
    Journal of Computational Physics 59, 232-246 (switched evolution relaxation, SER).

Each step solves ``(J + D/Δτ) δ = -F(u)`` and takes ``u + δ``: a backward-Euler step of ``D u̇ = -F(u)``. Far from
the root a small ``Δτ`` keeps the steps on the transient path (where Newton from a poor start jumps off, a shock
or a flame front it has not formed yet), and ``Δτ`` grows as the residual falls (``Δτ ← Δτ ‖F_prev‖ / ‖F‖``), so the
end is Newton's quadratic convergence. ``D``, the row sums of ``|J|``, makes ``Δτ`` dimensionless (a local CFL
number) without asking the problem for a mass; every row is damped, a saddle point's constraint too.

Quantities that must stay positive (a density, a pressure, a concentration) are declared to the driver as a function
``positive(u)`` of the state, which the problem builds from ``ptc(positive=[...])``: a step is cut to the largest
``α`` in ``1, 1/2, 1/4, ...`` that leaves each of them positive and at least ``keep`` of its value, and ``Δτ``
shrinks with ``α``. Without it a step through a state where the residual is still finite (a negative density) is
taken, and the march leaves the problem for good.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class PseudoTransient:
    """``jno.solve.ptc(...)``: a nonlinear driver with the Newton driver's calling convention."""

    direct = True  # it needs the assembled tangent
    name = "ptc"

    _HALVINGS = 16  # the step fractions tried for positivity: 1, 1/2, ..., 2^-15

    def __init__(self, dtau0=1.0, dtau_max=1e12, max_steps=200, rtol=1e-10, atol=1e-12, positive=None, keep=0.2):
        if not dtau0 > 0 or not dtau_max >= dtau0:
            raise ValueError(f"jno.solve.ptc: need 0 < dtau0 <= dtau_max (got {dtau0}, {dtau_max}).")
        if not 0.0 < keep < 1.0:
            raise ValueError(
                f"jno.solve.ptc: keep= is the share of a positive quantity a step must leave, in (0, 1) (got {keep})."
            )
        self.dtau0, self.dtau_max = float(dtau0), float(dtau_max)
        self.max_steps, self.rtol, self.atol = int(max_steps), float(rtol), float(atol)
        self.positive = None if positive is None else tuple(positive if isinstance(positive, (list, tuple)) else [positive])
        self.keep = float(keep)
        self.cache_key = (
            "ptc", self.dtau0, self.dtau_max, self.max_steps, self.rtol, self.atol, self.keep,
            None if self.positive is None else tuple(id(e) for e in self.positive),
        )  # fmt: skip

    def __repr__(self):
        pos = "" if self.positive is None else f", positive=[{len(self.positive)} expressions], keep={self.keep:g}"
        return (
            f"jno.solve.ptc(dtau0={self.dtau0:g}, dtau_max={self.dtau_max:g}, max_steps={self.max_steps}, "
            f"rtol={self.rtol:g}, atol={self.atol:g}{pos})"
        )

    def __hash__(self):
        return hash(self.cache_key)

    def __eq__(self, other):
        return isinstance(other, PseudoTransient) and other.cache_key == self.cache_key

    def __call__(self, residual_fn, u0, *, linear_solve=None, jacobian=None, project=None, positive=None, **_):
        import jax.experimental.sparse as jsparse

        if self.positive is not None and positive is None:
            raise ValueError(
                "jno.solve.ptc(positive=[...]): this problem does not evaluate the positive expressions; declare them on "
                "a jno.fvm problem, or pass the driver a function positive(u)."
            )
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

        fractions = 0.5 ** jnp.arange(self._HALVINGS, dtype=u0.dtype)

        def step_fraction(u, du):
            """The largest of 1, 1/2, ... that leaves every positive quantity positive and >= keep of its value
            (0 if none does)."""
            if positive is None:
                return jnp.asarray(1.0, u.dtype)
            g0 = positive(u)

            def admissible(a):
                g1 = positive(u + a * du)
                return jnp.all((g1 > 0.0) & (g1 >= self.keep * g0))

            ok = jax.vmap(admissible)(fractions)
            return jnp.where(jnp.any(ok), fractions[jnp.argmax(ok)], 0.0)

        def body(c):
            u, dtau, n_prev, k, n_now = c
            # SER, at most doubling per step
            grow = jnp.minimum(n_prev / jnp.maximum(n_now, tiny), 2.0)
            dtau = jnp.where(k > 0, jnp.minimum(dtau * grow, self.dtau_max), dtau)
            du = linear_solve(damped(jacobian(u), dtau), -residual_fn(u))
            alpha = step_fraction(u, du)
            cand = u + alpha * du
            if project is not None:
                cand = project(cand)
            n_cand = jnp.linalg.norm(residual_fn(cand))
            # a step that leaves the problem (a negative density, an overflow), that no fraction keeps positive, or
            # that blows the residual up is refused, and the pseudo-time step cut: the march takes the smaller step
            # from the same state. A step cut short for positivity shortens the pseudo-time step with it.
            ok = (alpha > 0.0) & jnp.isfinite(n_cand) & (n_cand <= 100.0 * n_now)
            u = jnp.where(ok, cand, u)
            n_new = jnp.where(ok, n_cand, n_now)
            dtau = jnp.where(ok, dtau * alpha, 0.1 * dtau)
            return u, dtau, jnp.where(ok, n_now, n_prev), k + 1, n_new

        init = (u0, jnp.asarray(self.dtau0, u0.dtype), n0, jnp.asarray(0), n0)
        u, _, _, k, n = jax.lax.while_loop(cond, body, init)
        self.last = (k, n, tol)  # concrete only when called eagerly
        return u
