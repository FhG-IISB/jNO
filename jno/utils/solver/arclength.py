"""Arc-length continuation for the load path — `fem.solve(tau=jno.solve.arclength(...))`.

Load control cannot pass a limit point. Past the peak there is no equilibrium at a higher load, so no
amount of step cutting finds one — the adaptive stepper says exactly that when it hits the floor
(:func:`_pilot_schedule`'s "UNSTABLE branch" error). Arc-length control replaces "advance the load by
Δλ" with "advance **along the equilibrium path** by Δs", which lets the path turn around: λ becomes an
unknown and the system gains one scalar constraint.

The trick that makes this small in jNO: **τ is not a DOF.** It reaches the residual only as the third
argument of ``op.residual(u, args, t)``, packed into the element kernel as a plain JAX scalar
(``fem_native._runtime_vals``). So arc-length does not add a degree of freedom to the operator — it
promotes an existing *argument* to an unknown, and ``∂R/∂λ`` is whatever ``jax.linearize`` makes of it.
``op.size`` never changes; the border lives in a residual wrapper, exactly as ``field.bounds(lo, hi)``
wraps the residual in a min-map without touching the operator (``FEM._bounded_solve_fn``).

The step, from the last converged ``(u_p, λ_p)`` with the internal state frozen on the buffers, solves
for the increment ``v = [Δu; Δλ]``::

    F(v) = [ R(u_p + Δu, args, λ_p + Δλ)                ]      N rows
           [ Δu·Δu + psi^2 Δλ^2 − Δs^2                  ]      1 row

the *total-increment* (spherical/cylindrical) constraint of Crisfield, **"A fast incremental/iterative
solution procedure that handles snap-through"**, Computers & Structures 13 (1981) 55-62, and
*Non-linear Finite Element Analysis of Solids and Structures* Vol. 1 (Wiley 1991) §9.3.2 — the
constraint is imposed on the increment from the last converged point rather than on a linearized normal
plane, so the iterates cannot drift off the surface.

Two honest deviations from the textbook, both stated because they change what the knobs mean:

* Crisfield weights the load term by the **reference load vector**, ``psi^2 Δλ^2 qᵀq``. jNO has no such
  vector by construction — the load is an arbitrary formula in τ (``peak*tau**8`` is legal, and a
  τ-dependent Dirichlet is not a load vector at all) — so ``psi`` here is a plain user weight with units
  of displacement per unit load factor. It defaults to ``0.0``, the **cylindrical** constraint, which
  Crisfield §9.3.2 reports works well in practice. Nothing is silently substituted for ``qᵀq``.
* The predictor is the **secant** (previous accepted increment, rescaled), not a tangent solve. It costs
  no extra linear solve, and it carries the traversal direction with it, so no ``sign(det K)`` test is
  needed to avoid doubling back at the fold.

Scope, up front:

* **Matrix-free drivers only.** The bordered system is presented to an unchanged ``newton_krylov``,
  whose ``jax.linearize`` builds the whole bordered JVP — border row, border column and all — for free.
  A driver that wants an assembled tangent (``jno.solve.newton(direct=True)``) is refused by name: the
  bordered ``(N+1, N+1)`` tangent is not built here.
* **λ is observability, not an output.** The trajectory is differentiable as always; the load factors
  reached are recorded on ``fem.tau_schedule`` as a concrete array for plotting the force-displacement
  curve, and do not carry a gradient.
* The declared ``domain(tau=(lo, hi, n))`` is **reinterpreted**: ``lo`` is the first (load-controlled)
  solve, ``n`` is the number of output rows, and ``hi`` sets both the traversal **direction** and the
  default ``ds``. It is not a target — arc-length takes ``n`` steps and stops wherever the path has
  reached, which is the point, since where the path goes is the answer. Note the consequence: ``n``
  buys resolution, not reach. To follow the path *further*, widen ``hi``; adding steps at a fixed
  ``hi`` re-resolves the same total arc more finely.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


@dataclass(frozen=True)
class ArcLengthSpec:
    """``jno.solve.arclength(...)`` — see :func:`jno.solve.arclength`."""

    psi: float = 0.0
    ds: Optional[float] = None

    name = "arclength"


def _bordered_residual(residual_of, u_p, lam_p, psi2, ds):
    """``F(v)`` for the increment ``v = [Δu; Δλ]`` — the physics rows plus Crisfield's constraint."""
    n = u_p.shape[0]

    def F(v):
        du, dlam = v[:n], v[n]
        r = residual_of(u_p + du, lam_p + dlam)
        c = jnp.dot(du, du) + psi2 * dlam * dlam - ds * ds
        return jnp.concatenate([jnp.asarray(r).reshape(-1), jnp.reshape(c, (1,))])

    return F


def march_arclength(fem, spec, *, solve_fn, op, readout, surf_readout, buffers0, sbuffers0, u0, tau_pts, dtype):
    """Run the load path under arc-length control; return the ``(n_steps, n_dofs)`` trajectory.

    Mirrors :func:`jno.utils.solver.history_march.run_history_march`'s fixed-grid leg step for step —
    same buffer roll, same convergence bookkeeping — but the scan carries ``λ`` and the previous
    increment, and each step solves the bordered system above instead of a residual at a prescribed τ.
    """
    from .history_march import _check_march_converged

    n_dofs = int(u0.shape[0])
    n_steps = int(np.asarray(tau_pts).shape[0])
    if n_steps < 3:
        raise ValueError(
            f"fem.solve(tau=jno.solve.arclength(...)): the declared load path has {n_steps} step(s). "
            "Arc-length needs at least 3: one load-controlled solve to start, one to set the initial "
            "direction, and at least one arc-length step. Widen `domain(tau=(start, end, n))`."
        )
    lam0 = float(np.asarray(tau_pts)[0])
    lam_hi = float(np.asarray(tau_pts)[-1])
    if abs(lam_hi - lam0) <= 1e-14 * max(1.0, abs(lam0)):
        # The span is not only the ds calibration -- the first load-controlled step is what gives the
        # march its initial DIRECTION along the path. With zero width that step moves nowhere, the
        # secant predictor has no direction to continue, and an explicit `ds=` cannot supply one.
        raise ValueError(
            "fem.solve(tau=jno.solve.arclength(...)): the declared load path has zero width "
            f"(start == end == {lam0:g}). Arc-length takes one load-controlled step across the declared "
            "span to establish which way along the equilibrium path to travel, so a degenerate span "
            "leaves the direction undefined — an explicit `ds=` cannot supply it either. Declare "
            "domain(tau=(start, end, n)) with end != start; `end` sets the direction and the default "
            "arc length, not a target to reach."
        )
    dlam_ref = (lam_hi - lam0) / (n_steps - 1)
    psi2 = float(spec.psi) ** 2

    def _solve_at(res_fn, start, jac=None):
        if solve_fn is not None:
            if jac is not None and getattr(solve_fn, "wants_jacobian", False):
                return jnp.asarray(solve_fn(res_fn, start, jacobian=jac)).reshape(-1)
            return jnp.asarray(solve_fn(res_fn, start)).reshape(-1)
        from .newton_krylov import newton_krylov

        return newton_krylov(res_fn, start)

    def _advance(u, lam, buffers, sbuffers):
        """Roll every buffered state forward using the equilibrium just found at ``lam``."""
        args = {"__history__": buffers, "__surface_history__": sbuffers, "__loadpath__": {}}
        from .history_march import _roll_buffer

        new_states = readout(u, lam, args)
        nb = {k: _roll_buffer(buffers[k], new_states[k]) for k in buffers}
        ns = sbuffers
        if surf_readout is not None and sbuffers:
            new_surf = surf_readout(u, lam, args)
            ns = {k: _roll_buffer(sbuffers[k], new_surf[k]) for k in sbuffers}
        return nb, ns

    def _args_of(buffers, sbuffers):
        return {"__history__": buffers, "__surface_history__": sbuffers, "__loadpath__": {}}

    def _residual_of(buffers, sbuffers):
        args = _args_of(buffers, sbuffers)
        prep = getattr(solve_fn, "prepare_residual", None)

        def res(uu, ll):
            r = lambda x: op.residual(x, args, ll)  # noqa: E731
            if prep is None:
                return r(uu)
            fn, _start = prep(r, uu)
            return fn(uu)

        return res

    # ---- step 0: an ordinary LOAD-CONTROLLED solve. There is no previous point to measure an arc
    # length from, and at lam0 (usually 0) the problem is the unloaded one. --------------------------
    args0 = _args_of(buffers0, sbuffers0)
    _jac0 = (lambda u: op.jacobian(u, args0, lam0)) if getattr(op, "jacobian", None) is not None else None
    u_0 = _solve_at(lambda u: op.residual(u, args0, lam0), u0, _jac0)
    b0, s0 = _advance(u_0, lam0, buffers0, sbuffers0)

    # ---- step 1: one more LOAD-CONTROLLED step. Its increment sets the initial direction AND
    # calibrates ds, so on a linear problem arc-length reproduces the declared uniform grid exactly.
    lam_1 = lam0 + dlam_ref
    args1 = _args_of(b0, s0)
    _jac1 = (lambda u: op.jacobian(u, args1, lam_1)) if getattr(op, "jacobian", None) is not None else None
    u_1 = _solve_at(lambda u: op.residual(u, args1, lam_1), u_0, _jac1)
    b1, s1 = _advance(u_1, lam_1, b0, s0)
    du_1, dlam_1 = u_1 - u_0, jnp.asarray(dlam_ref, dtype=dtype)
    ds = (
        jnp.sqrt(jnp.dot(du_1, du_1) + psi2 * dlam_1 * dlam_1)
        if spec.ds is None
        else jnp.asarray(float(spec.ds), dtype=dtype)
    )

    # ---- steps 2..n-1: bordered arc-length steps, one fixed-length scan. -------------------------
    def step(carry, _x):
        u_p, lam_p, du_p, dlam_p, buffers, sbuffers = carry
        res = _residual_of(buffers, sbuffers)
        # Secant predictor: continue the last accepted increment, rescaled to this step's arc length.
        # It carries the traversal DIRECTION, which is what keeps the march from doubling back at the
        # fold — no sign(det K) test, and nothing that needs a factorization.
        scale = ds / jnp.sqrt(jnp.dot(du_p, du_p) + psi2 * dlam_p * dlam_p + 1e-300)
        v0 = jnp.concatenate([du_p * scale, jnp.reshape(dlam_p * scale, (1,))])
        F = _bordered_residual(res, u_p, lam_p, psi2, ds)
        v = _solve_at(F, v0)
        du, dlam = v[:n_dofs], v[n_dofs]
        u, lam = u_p + du, lam_p + dlam
        nb, ns = _advance(u, lam, buffers, sbuffers)
        # Carry the residual norms out for the same post-hoc convergence check the fixed-grid march
        # does — the per-step driver's own check is a no-op under the scan (everything is a tracer).
        r_end = jnp.linalg.norm(jnp.asarray(F(v)))
        r_start = jnp.linalg.norm(jnp.asarray(F(v0)))
        return (u, lam, du, dlam, nb, ns), (u, lam, r_end, r_start)

    carry0 = (u_1, jnp.asarray(lam_1, dtype=dtype), du_1, dlam_1, b1, s1)
    _final, (traj, lams, r_end, r_start) = lax.scan(step, carry0, None, length=n_steps - 2)
    _check_march_converged(r_end, r_start, np.asarray(tau_pts)[2:], solve_fn)

    lam_all = jnp.concatenate([jnp.asarray([lam0, lam_1], dtype=dtype), lams])
    fem._tau_schedule = np.asarray(jax.lax.stop_gradient(lam_all))
    fem._tau_schedule_kind = "arclength"
    return jnp.concatenate([u_0[None, :], u_1[None, :], traj], axis=0)
