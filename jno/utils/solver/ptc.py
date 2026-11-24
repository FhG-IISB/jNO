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
number) without asking the problem for a mass; every row is damped, a saddle point's constraint too. A problem with
several unknowns per cell hands the driver ``row_groups`` (the cell of each row), and each cell's rows then share the
largest of their row sums: one local time step per cell, as a march has, so the density, momentum and energy of a gas
advance together. With a row sum per conserved variable they advanced at different pseudo-time rates, and the
pressure, a difference of them, was eaten away (the Mach 3 wedge of git-bug d7f66c4). ``ramp`` lets ``Δτ`` grow by at
least that factor after a full step, where switched evolution relaxation alone creeps while a shock forms and the
residual hardly falls.

Quantities that must stay positive (a density, a pressure, a concentration) are declared to the driver as a function
``positive(u)`` of the state, which the problem builds from ``ptc(positive=[...])``: a step is cut to the largest
``α`` in ``1, 1/2, 1/4, ...`` that leaves each of them positive and at least ``keep`` of its value, and ``Δτ``
shrinks with ``α``. Without it a step through a state where the residual is still finite (a negative density) is
taken, and the march leaves the problem for good.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class PseudoTransient:
    """``jno.solve.ptc(...)``: a nonlinear driver with the Newton driver's calling convention.

    Args:
        dtau0: the first pseudo-time step.
        dtau_max: the largest pseudo-time step.
        max_steps: the most pseudo-time steps.
        rtol, atol: the residual tolerances at which it stops.
        positive: expressions of the unknowns that every step keeps positive (fraction to the boundary).
        keep: the share of each positive quantity a step must leave, in (0, 1).
        ramp: the least growth of the pseudo-time step after a full step, >= 1.
    """

    direct = True  # it needs the assembled tangent
    name = "ptc"

    _HALVINGS = 16  # the step fractions tried for positivity: 1, 1/2, ..., 2^-15

    def __init__(self, dtau0=1.0, dtau_max=1e12, max_steps=200, rtol=1e-10, atol=1e-12, positive=None, keep=0.2, ramp=1.2):
        if not dtau0 > 0 or not dtau_max >= dtau0:
            raise ValueError(f"jno.solve.ptc: need 0 < dtau0 <= dtau_max (got {dtau0}, {dtau_max}).")
        if not 0.0 < keep < 1.0:
            raise ValueError(
                f"jno.solve.ptc: keep= is the share of a positive quantity a step must leave, in (0, 1) (got {keep})."
            )
        if not ramp >= 1.0:
            raise ValueError(f"jno.solve.ptc: ramp= is the least growth of the pseudo-time step, >= 1 (got {ramp}).")
        self.dtau0, self.dtau_max, self.ramp = float(dtau0), float(dtau_max), float(ramp)
        self.max_steps, self.rtol, self.atol = int(max_steps), float(rtol), float(atol)
        self.positive = None if positive is None else tuple(positive if isinstance(positive, (list, tuple)) else [positive])
        self.keep = float(keep)
        self.cache_key = (
            "ptc", self.dtau0, self.dtau_max, self.max_steps, self.rtol, self.atol, self.keep, self.ramp,
            None if self.positive is None else tuple(id(e) for e in self.positive),
        )  # fmt: skip

    def __repr__(self):
        pos = "" if self.positive is None else f", positive=[{len(self.positive)} expressions], keep={self.keep:g}"
        return (
            f"jno.solve.ptc(dtau0={self.dtau0:g}, dtau_max={self.dtau_max:g}, max_steps={self.max_steps}, "
            f"rtol={self.rtol:g}, atol={self.atol:g}, ramp={self.ramp:g}{pos})"
        )

    def __hash__(self):
        return hash(self.cache_key)

    def __eq__(self, other):
        return isinstance(other, PseudoTransient) and other.cache_key == self.cache_key

    def __call__(
        self, residual_fn, u0, *, linear_solve=None, jacobian=None, project=None, positive=None, row_groups=None, **_
    ):
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

        # The march must carry no tangents: the gradient comes from custom_root at the root (the implicit function
        # theorem, one solve with J^T), never from the loop, which reverse mode cannot differentiate. So the residual,
        # the tangent and the positive quantities run on their constants held out of the gradient, as newton_direct.
        f0 = lambda u: jnp.asarray(residual_fn(u)).reshape(-1)  # noqa: E731
        u0 = jnp.asarray(u0).reshape(-1)
        f_c, f_consts = jax.closure_convert(f0, u0)
        J_c, J_consts = jax.closure_convert(jacobian, u0)
        f_consts, J_consts = jax.lax.stop_gradient((f_consts, J_consts))
        f_fwd = lambda u: f_c(u, *f_consts)  # noqa: E731
        J_fwd = lambda u: J_c(u, *J_consts)  # noqa: E731
        g_fwd = None
        if positive is not None:
            g_c, g_consts = jax.closure_convert(positive, u0)
            g_consts = jax.lax.stop_gradient(g_consts)
            g_fwd = lambda u: g_c(u, *g_consts)  # noqa: E731

        groups = None if row_groups is None else np.asarray(row_groups, dtype=np.int32)
        n_groups = None if groups is None else int(groups.max()) + 1

        def per_group(D):
            """Each row's ``D`` replaced by the largest of its group's (one factor per cell)."""
            if groups is None:
                return D
            return jax.ops.segment_max(D, groups, num_segments=n_groups)[groups]

        def damped(J, dtau):
            if isinstance(J, jsparse.BCOO):
                n = J.shape[0]
                D = per_group(jax.ops.segment_sum(jnp.abs(J.data), J.indices[:, 0], num_segments=n))
                idx = jnp.concatenate([J.indices, jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)])
                return jsparse.BCOO((jnp.concatenate([J.data, D / dtau]), idx), shape=J.shape)
            J = jnp.asarray(J)
            return J + jnp.diag(per_group(jnp.abs(J).sum(axis=1))) / dtau

        n0 = jnp.linalg.norm(f_fwd(u0))
        tol = self.atol + self.rtol * n0
        tiny = jnp.finfo(u0.dtype).tiny

        def cond(c):
            _, dtau, _, k, n_now, _ = c
            return (n_now > tol) & (k < self.max_steps) & (dtau > 1e-12 * self.dtau0)

        fractions = 0.5 ** jnp.arange(self._HALVINGS, dtype=u0.dtype)

        def step_fraction(u, du):
            """The largest of 1, 1/2, ... that leaves every positive quantity positive and >= keep of its value
            (0 if none does)."""
            if g_fwd is None:
                return jnp.asarray(1.0, u.dtype)
            g0 = g_fwd(u)

            def admissible(a):
                g1 = g_fwd(u + a * du)
                return jnp.all((g1 > 0.0) & (g1 >= self.keep * g0))

            ok = jax.vmap(admissible)(fractions)
            return jnp.where(jnp.any(ok), fractions[jnp.argmax(ok)], 0.0)

        def body(c):
            u, dtau, n_prev, k, n_now, full = c
            # SER, at most doubling per step; at least `ramp` after a full step
            grow = jnp.minimum(n_prev / jnp.maximum(n_now, tiny), 2.0)
            grow = jnp.maximum(grow, jnp.where(full, self.ramp, 1.0))
            dtau = jnp.where(k > 0, jnp.minimum(dtau * grow, self.dtau_max), dtau)
            du = linear_solve(damped(J_fwd(u), dtau), -f_fwd(u))
            alpha = step_fraction(u, du)
            cand = u + alpha * du
            if project is not None:
                cand = project(cand)
            n_cand = jnp.linalg.norm(f_fwd(cand))
            # a step that leaves the problem (a negative density, an overflow), that no fraction keeps positive, or
            # that blows the residual up is refused, and the pseudo-time step cut: the march takes the smaller step
            # from the same state. A step cut short for positivity shortens the pseudo-time step with it.
            ok = (alpha > 0.0) & jnp.isfinite(n_cand) & (n_cand <= 100.0 * n_now)
            u = jnp.where(ok, cand, u)
            n_new = jnp.where(ok, n_cand, n_now)
            dtau = jnp.where(ok, dtau * alpha, 0.1 * dtau)
            return u, dtau, jnp.where(ok, n_now, n_prev), k + 1, n_new, ok & (alpha == 1.0)

        init = (u0, jnp.asarray(self.dtau0, u0.dtype), n0, jnp.asarray(0), n0, jnp.asarray(False))
        root, _, _, k, n, _ = jax.lax.while_loop(cond, body, init)
        if not isinstance(k, jax.core.Tracer):  # concrete only when called eagerly; a tracer must not leak out
            self.last = (k, n, tol)

        from .newton_krylov import _gated_direct, _solve_at

        def tangent(g, y):  # solve J x = y (J^T on the reverse pass) at the root: the implicit-function gradient
            J = jacobian(root)
            who = "the ptc tangent"
            fwd = lambda _mv, rhs: _solve_at(root, _gated_direct, J, rhs, linear_solve, who, "forward")  # noqa: E731
            tsp = lambda _mv, rhs: _solve_at(root, _gated_direct, J.T, rhs, linear_solve, who, "transpose")  # noqa: E731
            return jax.lax.custom_linear_solve(g, y, fwd, transpose_solve=tsp)

        return jax.lax.custom_root(f0, root, lambda _f, _x0: root, tangent)
