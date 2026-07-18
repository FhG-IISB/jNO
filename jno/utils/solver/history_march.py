"""Load-step **history march** for path-dependent FEM — the plasticity load-path contract.

March a ``domain(tau=(start, end, n))`` pseudo-time (load) grid, carrying per-quadrature-point internal
state on ``args["__history__"]`` from one step to the next, and advancing each state by its
``state.evolves(formula)`` update after every equilibrium solve. The pseudo-time coordinate τ is threaded
as the residual's temporal coordinate, so a load written as a function of τ *in the weak form* ramps
through the path.

Nothing is passed to ``fem.solve()``: the presence of ``.i(k)`` step-history together with a ``tau=`` grid
triggers this march (exactly as a ``u.t`` term triggers the transient stepper). The whole march is a
single ``lax.scan`` whose carry is ``(u, buffers)``, so it is reverse-mode differentiable end to end — the
per-step ``newton_krylov`` root-find is implicit-diff (``custom_root``), and the state readout + buffer
roll are pure JAX. See the FEM contract in the jNO skill.

Scope (stated up front): real, steady, single-field native-Lagrange forms — the equilibrium residual is
solved to equilibrium at each τ with the previous state frozen (a fully implicit return map when the
constitutive stress embeds it), then the state advances. Whole-domain state only (the readout runs on
every cell; sub-region-restricted plasticity is not wired yet).
"""

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
from jax import lax


def run_history_march(fem, solve_fn=None, **kwargs):
    """March ``fem`` over its domain's pseudo-time grid and return the ``(n_steps, n_dofs)`` trajectory.

    ``solve_fn`` (if given) is a nonlinear solver ``(residual_fn, u0) -> u`` — e.g. the one composed from
    ``fem.solve(nonlinear=jno.solve.newton(...))``; ``None`` uses the matrix-free ``newton_krylov`` default.
    """
    op = fem._op
    domain = fem.domain
    specs: Dict[Any, Any] = op.history_specs
    readout = op.state_readout

    n_dofs = int(op.size)
    tau_pts = np.asarray(getattr(domain, "_time_points", [0.0]), dtype=float)
    dtype = jnp.zeros(()).dtype  # x64-aware default float
    tau_grid = jnp.asarray(tau_pts, dtype=dtype)

    # Virgin (zeroed) per-quadrature-point buffers, one per buffered state:
    # (n_cell, n_quad, depth, *value_shape). Depth = the most-negative `.i(k)` the form uses.
    buffers0 = {k: jnp.zeros(s["shape"], dtype=dtype) for k, s in specs.items()}
    u0 = jnp.zeros(n_dofs, dtype=dtype)

    def _newton(res, u_prev):
        if solve_fn is not None:
            return jnp.asarray(solve_fn(res, u_prev)).reshape(-1)
        from .newton_krylov import newton_krylov

        return newton_krylov(res, u_prev)

    def _roll(buf, nv):
        """Push the just-computed state ``nv`` (n_cell, n_quad, *shape) into slot 0 — the next step's
        ``.i(-1)`` — and drop the oldest slot, keeping the buffer exactly ``depth`` deep. Depth 1 simply
        replaces; depth ≥ 2 (e.g. a BDF2 ``u.i(-2)``) shifts the tail back one."""
        return jnp.concatenate([nv[:, :, None, ...], buf[:, :, :-1, ...]], axis=2)

    def _march(param_args):
        """Run the whole load path as one ``lax.scan`` given the resolved runtime-parameter values
        (empty for a non-parametric forward march). Reverse-mode differentiable w.r.t. those values —
        the per-step ``newton_krylov`` is ``custom_root`` and the readout/roll are pure JAX."""

        def step(carry, tau_k):
            u_prev, buffers = carry
            args = {"__history__": buffers, **param_args}
            # Equilibrium at this load level, previous state frozen on the buffers. τ enters the load
            # through the residual's temporal coordinate; jacfwd sees the buffers as constants → the
            # consistent tangent (return map with the previous state held).
            u = _newton(lambda u: op.residual(u, args, tau_k), u_prev)
            # Advance every buffered state: internal states via their `.evolves` formula, a primary-unknown
            # history via the bare field interpolated to the quad points (both are `readout_formulas`).
            new_states = readout(u, tau_k, args)
            new_buffers = {k: _roll(buffers[k], new_states[k]) for k in buffers}
            return (u, new_buffers), u

        _final, traj = lax.scan(step, (u0, buffers0), tau_grid)
        return traj  # (n_steps, n_dofs)

    # A runtime-parametric form (an inverse problem: a material/load parameter created with
    # ``jno.np.parameter``) returns a differentiable trace node — crux resolves the parameters to values
    # and runs the march, ``∂trajectory/∂θ`` flowing through the scan (exactly like the steady nonlinear
    # ``op.solve()`` node). A non-parametric forward march returns the eager ``(n_steps, n_dofs)`` array.
    exprs = getattr(op, "runtime_parameter_exprs", {}) or {}
    if not exprs:
        return _march({})
    from ...trace import FunctionCall

    names = list(exprs)
    params = [exprs[n] for n in names]
    return FunctionCall(lambda *values: _march(dict(zip(names, values))), params, name="fem_history_march")
