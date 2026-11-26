"""IMEX Runge-Kutta: the transport explicit, the stiff local terms implicit.

Based on:
    Ascher, U. M., Ruuth, S. J. & Spiteri, R. J. (1997). Implicit-explicit Runge-Kutta methods for time-dependent
    partial differential equations. Applied Numerical Mathematics 25, 151-167 (the ARS(2,2,2) and ARS(4,4,3)
    pairs).
    Pareschi, L. & Russo, G. (2005). Implicit-explicit Runge-Kutta schemes and applications to hyperbolic
    systems with relaxation. Journal of Scientific Computing 25, 129-155.

A semidiscrete ``M u̇ + R(u) = 0`` whose residual splits as ``R = E + L``: ``E`` couples neighbours (fluxes) and is
stepped explicitly, at its own CFL limit; ``L`` acts on each cell alone (a reaction, a drag, a relaxation, an
algebraic closure such as the temperature of an energy) and is stepped implicitly. Because ``L`` is cell-local,
its Jacobian is block diagonal, one ``nf x nf`` block per cell, read with ``nf`` Jacobian-vector products, and
each implicit stage is a batched dense Newton per cell: a chemistry ten thousand times faster than the flow costs
no smaller step. Rows with no mass (algebraic) are local equations solved in the same Newton at every stage.

The problem supplies the split through its block's metadata: ``local_residual(u, t)``, ``local_layout`` (the
blocked layout ``(n_fields, n_cells)``), ``mass_diag(t)`` and, for ``cfl=``, ``cfl_dt_flux(u, cfl)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

_G = 1.0 - 1.0 / np.sqrt(2.0)
_DELTA = 1.0 - 1.0 / (2.0 * _G)

# (A_explicit, A_implicit), both stiffly accurate: the step's result is the last stage
TABLEAUS = {
    "ars222": (
        np.array([[0.0, 0.0, 0.0], [_G, 0.0, 0.0], [_DELTA, 1.0 - _DELTA, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, _G, 0.0], [0.0, 1.0 - _G, _G]]),
        2,
    ),
    "ars443": (
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0, 0.0],
                [11.0 / 18.0, 1.0 / 18.0, 0.0, 0.0, 0.0],
                [5.0 / 6.0, -5.0 / 6.0, 0.5, 0.0, 0.0],
                [0.25, 1.75, 0.75, -1.75, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0, 0.0],
                [0.0, 1.0 / 6.0, 0.5, 0.0, 0.0],
                [0.0, -0.5, 0.5, 0.5, 0.0],
                [0.0, 1.5, -1.5, 0.5, 0.5],
            ]
        ),
        3,
    ),
}


class IMEXScheme:
    """``jno.solve.imex(method, cfl=, newton_tol=, newton_steps=)``."""

    def __init__(self, method="ars222", cfl=None, newton_tol=1e-12, newton_steps=30):
        if method not in TABLEAUS:
            raise ValueError(f"jno.solve.imex(method={method!r}): use one of {sorted(TABLEAUS)}.")
        self.method, self.cfl = method, cfl
        self.newton_tol, self.newton_steps = float(newton_tol), int(newton_steps)
        self.A_ex, self.A_im, self._order = TABLEAUS[method]
        self.c = self.A_im.sum(axis=1)
        self.name = f"imex-{method}"

    def __repr__(self):
        cfl = "" if self.cfl is None else f", cfl={self.cfl}"
        return f"jno.solve.imex({self.method!r}{cfl})"

    def step_order(self):
        """The order of accuracy of the scheme in the step (2 for ARS(2,2,2), 3 for ARS(4,4,3))."""
        return self._order

    def _parts(self, block, args):
        meta = block.metadata or {}
        local = meta.get("local_residual")
        if local is None:
            raise NotImplementedError(
                f"{self!r} needs a problem that splits its residual into fluxes and cell-local terms; jno.fvm does."
            )
        nf, n = meta["local_layout"]
        mass_diag = meta["mass_diag"]

        def explicit(u, t):
            return block.residual(u, t, args) - local(u, t)

        return local, explicit, mass_diag, nf, n

    def stepper(self, block, args):
        """``step(u, t, dt) -> u_next``: one IMEX step of ``block`` with its runtime arguments ``args``. The fluxes
        are taken explicitly, the cell-local terms implicitly, by a Newton on each cell's ``nf x nf`` block."""
        local, explicit, mass_diag, nf, n = self._parts(block, args)
        A_ex, A_im, c = self.A_ex, self.A_im, self.c
        s = A_ex.shape[0]
        tol, steps = self.newton_tol, self.newton_steps
        eye = jnp.eye(nf)

        def blocks_of(U, t):
            """The ``(n, nf, nf)`` diagonal blocks of ∂L/∂U: column k from one jvp along field k of every cell."""
            cols = []
            for k in range(nf):
                tang = jnp.zeros((nf, n), U.dtype).at[k].set(1.0).reshape(-1)
                cols.append(jax.jvp(lambda w: local(w, t), (U,), (tang,))[1].reshape(nf, n).T)  # (n, nf)
            return jnp.stack(cols, axis=2)  # (n, row field, column field)

        def solve_stage(rhs, g, t, guess, m):
            """``M U + g L(U) = rhs`` on the differential rows, ``L(U) = 0`` on the algebraic ones, per cell."""
            alg = (m == 0.0).reshape(nf, n).T  # (n, nf)
            mm = m.reshape(nf, n).T

            def G(U):
                Lu = local(U, t).reshape(nf, n).T
                Uc = U.reshape(nf, n).T
                r = rhs.reshape(nf, n).T
                return jnp.where(alg, Lu, mm * Uc - r + g * Lu)

            def body(state):
                U, k, _ = state
                Gu = G(U)
                J = blocks_of(U, t)
                Jg = jnp.where(alg[:, :, None], J, mm[:, :, None] * eye[None] + g * J)
                dU = jnp.linalg.solve(Jg, -Gu[..., None])[..., 0]  # (n, nf)
                U = U + dU.T.reshape(-1)
                return U, k + 1, jnp.max(jnp.abs(dU))

            def cond(state):
                _, k, step = state
                return (k < steps) & (step > tol * (1.0 + jnp.max(jnp.abs(state[0]))))

            U, _, _ = jax.lax.while_loop(cond, body, (guess, 0, jnp.asarray(jnp.inf, guess.dtype)))
            return U

        def step(u, t, h):
            m = mass_diag(t)
            Ms = []
            Es, Ls, Us = [], [], []
            for i in range(s):
                ti = t + c[i] * h
                rhs = m * u
                for j in range(i):
                    rhs = rhs - h * (A_ex[i, j] * Es[j] + A_im[i, j] * Ls[j])
                if A_im[i, i] == 0.0:
                    U = u if i == 0 else rhs / jnp.where(m == 0.0, 1.0, m)
                else:
                    U = solve_stage(rhs, h * A_im[i, i], ti, Us[-1] if Us else u, m)
                Us.append(U)
                Es.append(explicit(U, ti))
                Ls.append(local(U, ti))
                Ms.append(m)
            return Us[-1]  # stiffly accurate: the last stage is the step

        return step

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        """March ``block`` with its runtime arguments ``args`` over its time grid and return the states at
        ``save_ts``. With ``cfl=`` every step is split into substeps from the stable step of its start state.
        ``linear_solve`` and ``nonlinear_solve`` are accepted for the time-scheme interface and unused: each
        cell's implicit stage is solved by its own Newton."""
        from functools import partial

        from .backend_blocks import _block_time_grid, _resample_trajectory
        from .timeschemes import _cfl_substeps

        s0 = jnp.asarray(block.state0).reshape(-1)
        grid = jnp.asarray(_block_time_grid(block), s0.dtype)
        dt = float(block.dt)
        cfl_dt = (block.metadata or {}).get("cfl_dt_flux")
        if self.cfl is not None and cfl_dt is None:
            raise ValueError(f"{self!r}: cfl= needs the problem's stable transport step; this block has none.")
        traced = isinstance(s0, jax.core.Tracer) or any(
            isinstance(a, jax.core.Tracer) for a in jax.tree_util.tree_leaves(args)
        )
        cache = {} if traced else block.__dict__.setdefault("_imex_compiled", {})
        ident = (self.method, self.newton_tol, self.newton_steps, int(grid.shape[0]), dt)
        step = cache.get(("step", ident))
        if step is None:
            step = cache[("step", ident)] = self.stepper(block, args)

        def run(u, t_start, sub):
            h = dt / sub
            return jax.lax.fori_loop(0, sub, lambda i, w: step(w, t_start + i * h, h), u)

        if self.cfl is not None and not traced:
            states, u = [s0], s0
            for k in range(1, grid.shape[0]):
                sub = max(1, int(np.ceil(dt / float(cfl_dt(u, self.cfl)) - 1e-12)))
                if ("run", ident, sub) not in cache:
                    cache[("run", ident, sub)] = jax.jit(partial(run, sub=sub))
                u = cache[("run", ident, sub)](u, grid[k - 1])
                states.append(u)
            return _resample_trajectory(jnp.stack(states), grid, save_ts, s0.dtype, self._order)
        # traced (a coupling's window, a gradient): one number of substeps from the initial state, itself traced
        # when that state is
        sub = 1 if self.cfl is None else _cfl_substeps(dt, cfl_dt(jax.lax.stop_gradient(s0), self.cfl))
        march = cache.get(("march", ident, sub)) if isinstance(sub, int) else None
        if march is None:

            def advance(u, t_next):
                u = run(u, t_next - dt, sub)
                return u, u

            def march(s0, grid):
                ys = jax.lax.scan(jax.checkpoint(advance), s0, grid[1:])[1]
                return jnp.concatenate([s0[None, :], ys], axis=0)

            if not traced and isinstance(sub, int):
                march = cache[("march", ident, sub)] = jax.jit(march)
        return _resample_trajectory(march(s0, grid), grid, save_ts, s0.dtype, self._order)
