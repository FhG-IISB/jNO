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

Scope (stated up front): real, steady native-Lagrange forms, **single-field or coupled** — the residual is
solved to equilibrium at each τ with the previous state frozen (a fully implicit return map when the
constitutive stress embeds it), then the state advances. Nothing here is per-field: ``n_dofs`` is the whole
block vector and the buffers are indexed by cell, so a state written by one field and read by another (a
phase-field history coupling damage to displacement) marches identically. Whole-domain state only (the
readout runs on every cell; sub-region-restricted plasticity is not wired yet).
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def run_history_march(fem, solve_fn=None, path=None, **kwargs):
    """March ``fem`` over its domain's pseudo-time grid and return the ``(n_steps, n_dofs)`` trajectory.

    ``solve_fn`` (if given) is a nonlinear solver ``(residual_fn, u0) -> u`` — e.g. the one composed from
    ``fem.solve(nonlinear=jno.solve.newton(...))``; ``None`` uses the matrix-free ``newton_krylov`` default.

    ``path`` (``fem.solve(tau=jno.solve.adaptive(limit=...))``) sizes the steps adaptively instead of
    taking the domain's uniform grid — see :func:`_pilot_schedule`. The output is resampled back
    onto the domain's grid either way, so the returned shape does not depend on the step sizes taken.
    """
    op = fem._op
    domain = fem.domain
    specs: Dict[Any, Any] = op.history_specs
    readout = op.state_readout
    surf_specs: Dict[Any, Any] = getattr(op, "surface_history_specs", {}) or {}
    surf_readout = getattr(op, "surface_state_readout", None)

    n_dofs = int(op.size)
    tau_pts = np.asarray(getattr(domain, "_time_points", [0.0]), dtype=float)
    dtype = jnp.zeros(()).dtype  # x64-aware default float
    tau_grid = jnp.asarray(tau_pts, dtype=dtype)

    # Virgin (zeroed) buffers per buffered state. VOLUME states live on cell quad points
    # (n_cell, n_quad, depth, *shape); SURFACE states (e.g. a friction slip) on boundary FACE quad points
    # (n_bfaces, n_quad_surf, depth, *shape). Depth = the most-negative `.i(k)` the form uses.
    buffers0 = {k: jnp.zeros(s["shape"], dtype=dtype) for k, s in specs.items()}
    sbuffers0 = {k: jnp.zeros(s["shape"], dtype=dtype) for k, s in surf_specs.items()}
    u0 = jnp.zeros(n_dofs, dtype=dtype)

    # Read-only per-load-step fields (``freeze_path``): one nodal frame per load step, scanned alongside
    # ``tau_grid`` and delivered to the residual/readout on ``args["__loadpath__"]`` (like ``__history__``,
    # but never rolled/advanced). Empty in the common case; validated to match the load-step count.
    path_specs: Dict[Any, Any] = getattr(op, "path_specs", {}) or {}
    path_frames: Dict[Any, Any] = {}
    for _fid, _spec in path_specs.items():
        _fr = jnp.asarray(_spec["frames"], dtype=dtype)
        if int(_fr.shape[0]) != int(tau_grid.shape[0]):
            raise ValueError(
                f"jno.fem: load-path field {_spec.get('name', '?')!r} carries {int(_fr.shape[0])} frames "
                f"but the tau= grid has {int(tau_grid.shape[0])} load steps — `freeze_path(frames)` needs "
                "one nodal field per load step (frames.shape[0] must equal n in domain(tau=(start, end, n)))."
            )
        path_frames[_fid] = _fr

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

    def _step_once(u_prev, buffers, sbuffers, tau_k, path_k, param_args):
        """One accepted load step: equilibrium at ``tau_k``, then advance every buffered state.

        The single definition of what a step *is* — the scan body below and the eager pilot both call
        it, so an adaptive schedule cannot drift from the fixed one by taking a subtly different step."""
        args = {"__history__": buffers, "__surface_history__": sbuffers, "__loadpath__": path_k, **param_args}
        # Equilibrium at this load level, previous state frozen on the buffers. τ enters the load
        # through the residual's temporal coordinate (and the load-path field slices its frames); jacfwd
        # sees the buffers (volume AND surface) and the path slice as constants → the consistent tangent.
        u = _newton(lambda u: op.residual(u, args, tau_k), u_prev)
        # Advance every buffered state: volume states via their `.evolves` formula / a primary-unknown
        # history; surface states (a friction slip) via the surface readout on the region's faces.
        new_states = readout(u, tau_k, args)
        new_buffers = {k: _roll(buffers[k], new_states[k]) for k in buffers}
        new_sbuffers = sbuffers
        if surf_readout is not None and sbuffers:
            new_surf = surf_readout(u, tau_k, args)
            new_sbuffers = {k: _roll(sbuffers[k], new_surf[k]) for k in sbuffers}
        return u, new_buffers, new_sbuffers

    def _march(param_args, grid=None):
        """Run the whole load path as one ``lax.scan`` given the resolved runtime-parameter values
        (empty for a non-parametric forward march). Reverse-mode differentiable w.r.t. those values —
        the per-step ``newton_krylov`` is ``custom_root`` and the readout/roll are pure JAX.

        ``grid`` overrides the domain's τ points — this is the *replay* leg of the adaptive path, run
        over a schedule the pilot already froze, so it is an ordinary fixed-length scan with nothing to
        reject and no dynamic shapes."""
        _grid = tau_grid if grid is None else grid
        _frames = path_frames if grid is None else {}

        def step(carry, xs):
            tau_k, path_k = xs  # this step's τ and its per-load-step field slices ({fid: (n_nodes,)})
            u_prev, buffers, sbuffers = carry
            u, new_buffers, new_sbuffers = _step_once(u_prev, buffers, sbuffers, tau_k, path_k, param_args)
            return (u, new_buffers, new_sbuffers), u

        _final, traj = lax.scan(step, (u0, buffers0, sbuffers0), (_grid, _frames))
        return traj  # (n_steps, n_dofs)

    # ---- ADAPTIVE load stepping (`fem.solve(tau=jno.solve.adaptive(limit=...))`) --------------------
    # Pilot -> freeze -> replay. The pilot marches EAGERLY with rejection to discover a step schedule
    # that keeps the per-step solution change under `limit`; the schedule is frozen; the replay is the
    # ordinary fixed-length scan above over that schedule, so it stays reverse-mode differentiable.
    #
    # Rejection is why the pilot has to be separate. The transient `adaptive_march` accepts EVERY
    # attempt, and its source records why: "adding rejection turns a finite gradient (AD -21.92 vs FD
    # -21.99) into NaN" -- a discarded state makes the per-step adjoint run at zero cotangent. Splitting
    # discovery from replay sidesteps that entirely: the replay has nothing to reject.
    #
    # The schedule is a PIECEWISE-CONSTANT function of the parameters (perturb one infinitesimally and
    # the same steps are accepted), so the gradient over a frozen schedule is the true derivative almost
    # everywhere -- the same contract `adapt=` already makes for a frozen mesh sequence.
    if path is not None:
        if _is_explicit_schedule(path):
            # `tau=<array>`: replay a schedule the caller already has — from an earlier pilot
            # (`fem.tau_schedule`), or simply a non-uniform grid they chose. Same replay leg, no pilot.
            _explicit = _as_schedule(path, tau_pts)
            if path_frames:
                raise NotImplementedError(
                    "jno.fem: an explicit `tau=<schedule>` does not compose with a per-load-step field "
                    "(`freeze_path(frames)`) — those frames are indexed by the declared step count."
                )
            fem._tau_schedule = _explicit
            _frozen = jnp.asarray(_explicit, dtype=dtype)

            def _run_explicit(param_args):
                traj = _march(param_args, _frozen)
                return jax.vmap(lambda col: jnp.interp(tau_grid, _frozen, col), in_axes=1, out_axes=1)(traj)

            _driver = _run_explicit
            exprs = getattr(op, "runtime_parameter_exprs", {}) or {}
            if not exprs:
                return _driver({})
            from ...trace import FunctionCall

            _names = list(exprs)
            return FunctionCall(
                lambda *values: _driver(dict(zip(_names, values))),
                [exprs[n] for n in _names],
                name="fem_history_march",
            )
        _validate_path_spec(path, fem, path_frames)
        limits = _resolve_limits(path.limit, fem, n_dofs)

        def _run_adaptive(param_args, replay=True):
            schedule, states = _pilot_schedule(
                path, limits, tau_pts, u0, buffers0, sbuffers0, _step_once, param_args, dtype, op
            )
            fem._tau_schedule = np.asarray(schedule)  # observability: what the pilot actually chose
            src = jnp.asarray(schedule, dtype=dtype)
            # The pilot ALREADY solved every accepted step. Replaying them is only needed to make the
            # result differentiable (a fixed-length scan the adjoint can run through); a plain forward
            # solve can take the states it already has. Measured on the phase-field SENT problem: the
            # replay was ~15% of the adaptive run, and it was recomputing an answer we were discarding.
            traj = _march(param_args, src) if replay else jnp.stack(states)
            # Resample onto the domain's declared grid, exactly as the transient stepper resamples onto
            # `save_ts`: the step size is then decoupled from the sample times, and `fem.solve()` returns
            # the same shape whether or not `tau=` was passed.
            return jax.vmap(lambda col: jnp.interp(tau_grid, src, col), in_axes=1, out_axes=1)(traj)

        _driver, _is_adaptive = _run_adaptive, True
    else:
        _driver, _is_adaptive = _march, False

    # A runtime-parametric form (an inverse problem: a material/load parameter created with
    # ``jno.np.parameter``) returns a differentiable trace node — crux resolves the parameters to values
    # and runs the march, ``∂trajectory/∂θ`` flowing through the scan (exactly like the steady nonlinear
    # ``op.solve()`` node). A non-parametric forward march returns the eager ``(n_steps, n_dofs)`` array.
    exprs = getattr(op, "runtime_parameter_exprs", {}) or {}
    if not exprs:
        # Non-parametric: the answer is an array, nothing will differentiate it, so the adaptive path
        # keeps the states its pilot already solved rather than marching them a second time.
        return _driver({}, replay=False) if _is_adaptive else _driver({})
    from ...trace import FunctionCall

    names = list(exprs)
    params = [exprs[n] for n in names]
    if path is not None:
        # The pilot must see CONCRETE values to accept or reject a step, and a trace node's arguments are
        # tracers under `jax.grad`. Piloting at the parameters' *stored* values looks like the obvious
        # answer and is a trap: a fresh `jno.np.parameter` stores 0.0, so the pilot would march with the
        # parameter switched off and freeze a schedule for a load path that never happened -- silently,
        # and the result would look entirely plausible. Refuse, and name the two-step workflow that does
        # work, which is also the `adapt=<trajectory>` precedent.
        raise NotImplementedError(
            "jno.fem: `tau=jno.solve.adaptive(...)` cannot discover a schedule for a form carrying a "
            f"runtime parameter ({sorted(names)}). The pilot needs concrete values to accept or reject a "
            "step, and a differentiable solve hands it tracers; piloting at the stored values instead "
            "would silently adapt to whatever they happen to be (0.0 for a fresh jno.np.parameter). Run "
            "the study forward first and replay the schedule it found:\n"
            "    fem.solve(tau=jno.solve.adaptive(limit=...))   # forward, at the values you want\n"
            "    fem.solve(tau=fem.tau_schedule)                # differentiable replay of that schedule\n"
            "`tau=<array>` also accepts any non-uniform grid you choose."
        )
    return FunctionCall(lambda *values: _march(dict(zip(names, values))), params, name="fem_history_march")


def _is_explicit_schedule(path):
    """Is ``tau=`` a recorded schedule to replay rather than a spec to discover one with?"""
    return path is not None and not hasattr(path, "limit")


def _as_schedule(path, tau_pts):
    """Validate an explicit ``tau=<array>`` schedule against the declared path."""
    sched = np.asarray(path, dtype=float).reshape(-1)
    if sched.size < 2 or not np.all(np.diff(sched) > 0):
        raise ValueError("fem.solve(tau=<schedule>): the schedule must be a strictly increasing 1-D array of τ values.")
    lo, hi = float(tau_pts[0]), float(tau_pts[-1])
    if abs(sched[0] - lo) > 1e-9 * max(1.0, abs(lo)) or abs(sched[-1] - hi) > 1e-9 * max(1.0, abs(hi)):
        raise ValueError(
            f"fem.solve(tau=<schedule>): the schedule spans [{sched[0]:g}, {sched[-1]:g}] but the domain's "
            f"load path is [{lo:g}, {hi:g}] — they must agree at both ends."
        )
    return sched


def _validate_path_spec(path, fem, path_frames):
    """Refuse a ``tau=`` spec that cannot mean what it says, before any work is done."""
    if getattr(path, "limit", None) is None:
        raise ValueError(
            "fem.solve(tau=jno.solve.adaptive(...)) needs `limit=` — how much the solution may change in "
            "one load step. The `rtol`/`atol` step-doubling criterion the transient uses estimates a "
            "LOCAL TRUNCATION ERROR, and a rate-independent load path has none: each step is an "
            "equilibrium, not an approximation to a trajectory. Pass e.g. limit=0.05, or "
            "limit=[(damage_field, 0.05)]."
        )
    if getattr(path, "base", None) is not None:
        raise NotImplementedError(
            "fem.solve(tau=...) sizes a load path, which has no time-integration scheme to attach to — "
            "`theta(...).adaptive(...)` is for `time=`. Use a bare jno.solve.adaptive(limit=...)."
        )
    if path_frames:
        raise NotImplementedError(
            "jno.fem: adaptive load stepping does not compose with a per-load-step field "
            "(`freeze_path(frames)`) — those frames are indexed by the DECLARED step count, and an "
            "adaptive schedule has a different one. Use the fixed `domain(tau=...)` grid for a load path "
            "that carries prescribed per-step data."
        )


def _resolve_limits(limit, fem, n_dofs):
    """``[(dof_index_array_or_None, tol)]`` — the per-step change bound, resolved to DOF space.

    A bare number bounds every DOF; ``[(field, tol), ...]`` bounds each field's own block, which is the
    usual case (a damage variable lives in ``[0,1]`` while the displacement beside it does not)."""
    if isinstance(limit, (int, float, np.floating, np.integer)):
        return [(None, float(limit))]
    pairs = list(limit.items()) if isinstance(limit, dict) else list(limit)
    blocks = getattr(fem, "blocks", None)
    out = []
    for field, tol in pairs:
        if blocks is None:
            raise ValueError("fem.solve(tau=...): a per-field `limit` needs a block-structured problem.")
        sl = blocks[fem.block_index(field)]
        out.append((np.arange(int(sl.start), int(sl.stop)), float(tol)))
    if not out:
        raise ValueError("fem.solve(tau=...): `limit` resolved to no fields at all.")
    return out


#: Residual norm (relative to the state's magnitude) below which a piloted step counts as converged.
#: Loose on purpose: this decides ACCEPT vs CUT, not the answer -- the accepted step's own solve already
#: ran to the driver's tolerance, and the replay re-solves it anyway.
_RESID_TOL = 1e-6


def _pilot_schedule(path, limits, tau_pts, u0, buffers0, sbuffers0, step_once, param_args, dtype, op=None):
    """March EAGERLY with rejection; return ``(schedule, states)`` — the accepted τ values and the
    solution at each of them.

    A step is rejected when the per-step solve fails to converge or when any limited field moved more
    than its tolerance. Rejection halves the step; a comfortable step grows it.

    The step is **compiled once** and reused across attempts. Left uncompiled, every attempt paid its
    own trace and compile: measured on the phase-field SENT problem, an eager step cost ~5x the same
    step inside the replay scan, which made the pilot most of the adaptive run.

    Compiling it costs the eager convergence check, though — ``newton_krylov`` skips its guard under a
    trace, by design, because the test cannot concretise there. So the rejection signal is *returned*
    rather than raised: the compiled step hands back the residual norm and the tolerance it was judged
    against, and this loop compares them concretely. That is the more explicit arrangement anyway; the
    exception it replaces was being caught by type.
    """
    lo, hi = float(tau_pts[0]), float(tau_pts[-1])
    span = hi - lo
    if span <= 0:
        return np.asarray(tau_pts, dtype=float), [u0]

    import jax as _jax

    def _attempt(u_prev, bufs, sbufs, tau_k, pargs):
        """One step plus the numbers the acceptance test needs — all inside one compiled program."""
        u, nb, nsb = step_once(u_prev, bufs, sbufs, tau_k, {}, pargs)
        rn = jnp.linalg.norm(
            jnp.asarray(
                op.residual(u, {"__history__": bufs, "__surface_history__": sbufs, "__loadpath__": {}, **pargs}, tau_k)
            )
        )
        return u, nb, nsb, rn

    _attempt_c = _jax.jit(_attempt) if op is not None else _attempt
    dtau = float(path.dt0) if path.dt0 is not None else span / max(1, len(tau_pts) - 1)
    dtau_min = span * 1e-8
    shrink, grow = float(path.shrink), float(path.grow)
    max_steps = int(path.max_steps)

    def _changed_too_much(u_new, u_old):
        """The largest per-limit overshoot ratio (<= 1 means every field is inside its bound)."""
        worst = 0.0
        for idx, tol in limits:
            d = np.abs(np.asarray(u_new) - np.asarray(u_old))
            d = float(d.max()) if idx is None else float(d[idx].max())
            worst = max(worst, d / max(tol, 1e-300))
        return worst

    # The first point of the declared grid is a solved step in the fixed march too (the virgin state at
    # the starting load), so the schedule starts there and no acceptance test applies to it.
    u, bufs, sbufs = u0, buffers0, sbuffers0
    u, bufs, sbufs, _r0 = _attempt_c(u, bufs, sbufs, jnp.asarray(lo, dtype), param_args)
    schedule, states = [lo], [u]
    tau = lo
    attempts = 0
    while tau < hi - 1e-12 * max(1.0, abs(hi)):
        if attempts >= max_steps:
            raise RuntimeError(
                f"jno.fem: adaptive load stepping used its budget of max_steps={max_steps} attempts and "
                f"reached only τ={tau:.6g} of {hi:.6g}. Raise max_steps, or loosen `limit`."
            )
        attempts += 1
        dt = min(dtau, hi - tau)
        trial = tau + dt
        try:
            u_try, bufs_try, sbufs_try, rnorm = _attempt_c(u, bufs, sbufs, jnp.asarray(trial, dtype), param_args)
            ratio = _changed_too_much(u_try, u)
            # The compiled step cannot raise on non-convergence (the guard needs a concrete value and
            # there is none under a trace), so it hands the residual norm back and the test happens here.
            converged = bool(np.all(np.isfinite(np.asarray(u_try)))) and float(rnorm) < _RESID_TOL * max(
                1.0, float(np.abs(np.asarray(u_try)).max())
            )
        except RuntimeError:  # a driver that still raises eagerly (a non-compiled solve_fn) -- same signal
            ratio, converged = np.inf, False
        if converged and ratio <= 1.0:
            u, bufs, sbufs = u_try, bufs_try, sbufs_try
            tau = trial
            schedule.append(tau)
            states.append(u)
            if ratio < 0.5:  # comfortably inside the bound -- reach for a bigger step
                dtau = min(dtau * grow, span)
            continue
        dtau = dt * shrink
        if dtau < dtau_min:
            raise RuntimeError(
                f"jno.fem: adaptive load stepping cut the step to {dtau:.3g} (below the floor {dtau_min:.3g}) "
                f"at τ={tau:.6g} and still could not keep the solution change within `limit` "
                f"(overshoot x{ratio:.3g}). That is the signature of an UNSTABLE branch, not of a step "
                "that is merely too big: under load control a snap-back has no nearby equilibrium, so no "
                "amount of load-step refinement finds one. Loosen `limit` to accept the jump, or drive "
                "the path by a displacement/arc-length measure instead of the load."
            )
    return np.asarray(schedule, dtype=float), states
