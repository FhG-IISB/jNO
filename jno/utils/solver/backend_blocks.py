from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------
# Solver-agnostic semidiscrete block returned by weak.assemble(...)
# ---------------------------------------------------------------------


def _verdict(G, u_prev, wn, report):
    """The step's own residual norms, for a march that wants to judge its steps afterwards.

    ``G`` is the function the driver actually root-finds, so both norms are for the SAME equation --
    one at the incoming iterate and one at the solved state. That is what lets the caller apply the
    driver's own ``atol + rtol*||r(u_prev)||`` test outside the trace, where it can concretise.

    Costs two residual evaluations per step, and only when asked. Same arrangement, for the same
    reason, as the load-path march in ``history_march.py``.
    """
    if not report:
        return wn
    import jax.numpy as jnp

    return (
        wn,
        jnp.linalg.norm(jnp.asarray(G(wn)).reshape(-1)),
        jnp.linalg.norm(jnp.asarray(G(u_prev)).reshape(-1)),
    )


def _algebraic_rows(M, n, dtype):
    """``True`` on the rows of the mass ``M`` that are entirely zero: equations with no time derivative
    (a Dirichlet row, an FDM flux row, a pressure row), i.e. the algebraic part of the DAE."""
    import jax.numpy as jnp

    if hasattr(M, "todense"):  # BCOO: duplicates are summed, so a row is zero iff its |data| sums to zero
        mass = jnp.zeros((n,), dtype).at[M.indices[:, 0]].add(jnp.abs(M.data).astype(dtype))
    else:
        mass = jnp.sum(jnp.abs(jnp.asarray(M, dtype)), axis=1)
    return mass == 0


def _row_scaled(A, w):
    """``diag(w) · A`` for a BCOO or dense ``A``."""
    import jax.numpy as jnp

    if hasattr(A, "todense"):
        import jax.experimental.sparse as jsp

        return jsp.BCOO((A.data * w[A.indices[:, 0]], A.indices), shape=A.shape)
    return w[:, None] * jnp.asarray(A)


def _theta_row_weights(M, theta, n, dtype):
    """Per-row θ of a θ-step on the DAE ``M u̇ + R(u) = 0``, or ``None`` when θ = 1 (nothing to change).

    A zero-mass row is a constraint ``R_i(u) = 0``, not an ODE. The θ-average turns it into
    ``θ R_i(u⁺) + (1−θ) R_i(u) = 0``: at θ = 0 the row does not involve ``u⁺`` at all (a singular step,
    measured as a NaN once the leftover boundary residual fell below the tolerance), and at θ = ½ it
    only holds on average, so an initial state that violates it flips sign every step and never decays
    (measured: a Dirichlet boundary started at 1 stays at |u| = 1 under Crank–Nicolson, in FEM and FDM).
    Those rows are imposed at the new time instead (weight 1), which is the θ-method applied to the
    ODE on the constraint manifold (Hairer & Wanner, *Solving ODEs II*, §VI.1, the state-space form);
    it keeps the method's order on the differential rows."""
    if theta >= 1.0:
        return None
    import jax.numpy as jnp

    return jnp.where(_algebraic_rows(M, n, dtype), jnp.asarray(1.0, dtype), jnp.asarray(theta, dtype))


def _default_step_solve(step_op, rhs, x0, diag, *, krylov=None):
    """jNO's default solve of one implicit step ``step_op(w) = rhs`` -- Jacobi-preconditioned BiCGStab with a
    verified GMRES rescue (GMRES outright when the assembly asks for it via ``metadata["krylov"]``).

    Shared by :meth:`SemidiscreteTimeBlock.step` and the Rosenbrock stages, whose stage operator is the same
    kind of ``M + scale*J`` step operator. ``diag`` is that operator's diagonal (zero entries left unscaled).
    """
    import jax
    import jax.numpy as jnp

    from .krylov import gmres as _scaled_gmres  # scale-invariant: JAX's Arnoldi zeroes a tiny ||b||

    n = rhs.shape[0]
    inv = 1.0 / jnp.where(jnp.abs(diag) > 1e-30, diag, 1.0)
    # ``metadata["krylov"]`` lets an assembly pick the Krylov method its operator needs. BiCGStab is
    # the default and is right for the symmetric real blocks; the complex real-equivalent block
    # ``[[A_r,-A_i],[A_i,A_r]]`` is genuinely non-symmetric and asks for GMRES, which does not break
    # down there. Restart is capped at 40 (as the dedicated complex marcher used) to bound memory.
    if krylov == "gmres":
        # The tolerance must be REACHABLE in the working precision. jNO defaults to float32 (x64 is
        # opt-in), whose eps is 1.2e-7, so the 1e-10 relative target asked for here could never be
        # met -- the termination test never fired, and GMRES, which has no other way out, paid its
        # full ``10*n`` restarts every step however easy the system. Measured on a 377-dof parametric
        # transient: **5485.6 ms/step -> 20.0 ms/step (249x)**, for the same answer (final |u|
        # 0.141276836 vs 0.141276851).
        #
        # Scaled to the dtype rather than capped by ``maxiter``: an easy system then exits as soon as
        # it converges and a hard one keeps working, where a fixed cap would silently under-solve the
        # hard one. The 100x factor is not slack -- at 10*eps (1.2e-6) GMRES still never terminated
        # (5494.0 ms/step measured). In float64 the 1e-10 floor keeps the previous behaviour exactly.
        ktol = max(1e-10, 100.0 * float(jnp.finfo(rhs.dtype).eps))
        wn, _ = _scaled_gmres(step_op, rhs, x0=x0, tol=ktol, atol=0.0, restart=min(n, 40), M=lambda x: inv * x)
        return wn
    # BiCGStab asks for the same unreachable 1e-10 and is deliberately LEFT ALONE. It never grinds:
    # its breakdown test fires once the residual stalls at the float32 noise floor, so the effect is
    # "solve as tightly as this precision allows" -- 1.0 ms/step here, i.e. the defect is masked at no
    # measurable cost. Giving it the reachable tolerance measured 0.6 ms/step but moved every real
    # transient's answer by ~3e-6 relative (0.141276836 -> 0.141277224) and its gradient by ~1e-5,
    # trading accuracy for 0.4 ms/step. Not worth it. If JAX's breakdown handling ever changes, this
    # becomes the GMRES bug and wants the same `ktol`.
    wn, _ = jax.scipy.sparse.linalg.bicgstab(step_op, rhs, x0=x0, tol=1e-10, atol=0.0, maxiter=20_000, M=lambda x: inv * x)
    # BiCGStab's breakdown/stall exit is only benign on the SYMMETRIC blocks the default was
    # chosen for. Measured on a coupled first-order block with a velocity-identity coupling
    # (genuinely non-symmetric, cond(M+dtA)=54): with a degenerate warm start it returns NaN
    # outright (exact mid-iteration convergence makes ``omega = 0/0``, and NaN passes jax's
    # ``omega != 0`` breakdown test), and with a healthy warm start it EXITS SILENTLY at ~1e-2
    # relative residual — each step slightly wrong, compounding to 1e62 over 60 steps. The steady
    # default would have raised (its eager residual check); a traced scan cannot raise, so VERIFY
    # the step and re-solve with GMRES when the residual is not small — GMRES has no breakdown
    # division and measured 1e-16 per step on the same block. Cost: one extra matvec + one scalar
    # reduce per step; the comparison is False for a NaN residual too, so both failure modes take
    # the rescue. The healthy stall floor (measured 8e-11 in f64, ~1e-5 in f32) sits well under
    # the dtype-scaled threshold, so a symmetric march never pays the GMRES.
    eps = float(jnp.finfo(rhs.dtype).eps)
    r_rel = jnp.linalg.norm(step_op(wn) - rhs) / jnp.maximum(jnp.linalg.norm(rhs), eps)
    ktol = max(1e-10, 100.0 * eps)
    return jax.lax.cond(
        r_rel < max(1e-9, 1e4 * eps),
        lambda: wn,
        lambda: _scaled_gmres(step_op, rhs, x0=x0, tol=ktol, atol=0.0, restart=min(n, 40), M=lambda x: inv * x)[0],
    )


@dataclass
class SemidiscreteTimeBlock:
    """
    Solver-agnostic semidiscrete transient block.

    A `SemidiscreteTimeBlock` is returned by:

        weak_expr.assemble(target="fem_time")

    It represents the spatially discretized transient weak-form problem, but it
    does not perform time integration by itself. Read its flat pieces (`M`, `A` /
    `residual`, `state0`, `dt`) and step it with your own integrator, or hand it to
    `fem.solve()`'s default backward-Euler.

    Supported payloads
    ------------------
    Linear semidiscrete payload:

        M u_dot + A(t, args) u = c + f(t, args)

    stored as:

        M
        A                       # optional constant operator matrix
        operator_fn             # optional runtime operator callback
        affine_bias
        forcing_vector_fn

    At least one of ``A`` or ``operator_fn`` must be populated.  The runtime
    callback has signature ``operator_fn(t, args) -> matrix`` and takes
    precedence over the constant matrix when both are present.

    Nonlinear semidiscrete payload:

        M(t) u_dot + R(u, t) = 0

    stored as:

        mass(t, args)
        residual(u, t, args)
        jacobian(u, t, args)

    Important fields
    ----------------
    backend:
        Backend identifier. Usually `"fem_time"`.
    mode:
        Time-integration mode hint, usually `"implicit"` or `"explicit"`.
    time_order:
        Temporal order of the semidiscrete problem. Currently first-order
        FEM-time blocks are supported by the adapters.
    spatial_kind:
        Spatial discretization origin, usually `"weak_form"`.
    ir:
        Lowered weak-form IR used to construct this block.
    mass_expr:
        Symbolic mass-like weak-form expression, if available.
    residual_expr:
        Symbolic residual-like weak-form expression, if available.
    boundary_exprs:
        Boundary weak-form terms grouped by boundary region id.
    rhs:
        Optional RHS callable. Usually unused for semidiscrete weak-form blocks.
    jacobian:
        Nonlinear residual Jacobian callable `jacobian(u, t, args)`.
    mass:
        Mass operator callable `mass(t, args)`.
    residual:
        Nonlinear residual callable `residual(u, t, args)`.
    nonlinear_runtime:
        Runtime diagnostics for nonlinear semidiscrete-time assembly.
    state0:
        Initial state vector.
    initial_conditions:
        Raw initial-condition object, if supplied.
    t0, t1:
        Start and end time.
    dt:
        Time-step size or time-step hint.
    eval_context:
        FEM evaluation context copied from the domain.
    metadata:
        Classification, lowering, and diagnostic metadata.
    M, A:
        Linear semidiscrete mass and optional constant operator matrices.
    operator_fn:
        Optional runtime linear-operator callback ``operator_fn(t, args)``.
        When populated, adapters evaluate it at runtime instead of using the
        constant ``A`` matrix. This keeps inverse parameters differentiably
        connected to the physical solve.
    affine_bias:
        Constant affine vector `c` in `M u_dot + A u = c + f(t)`.
    forcing_vector_fn:
        Optional forcing callback `f(t, args)`.
    forcing_mode:
        Text label describing how forcing is represented, for example
        `"none"`, `"weak_auto"`, `"user_callback"`, or `"embedded_residual"`.
    """

    backend: str = "transient"
    mode: str = "implicit"
    time_order: int = 1
    spatial_kind: str = "weak_form"

    ir: Any = None

    mass_expr: Any = None
    residual_expr: Any = None
    boundary_exprs: Dict[str, Any] = field(default_factory=dict)

    # nonlinear/general semidiscrete payload
    rhs: Optional[Callable] = None
    jacobian: Optional[Callable] = None
    mass: Optional[Callable] = None
    residual: Optional[Callable] = None
    nonlinear_runtime: Dict[str, Any] = field(default_factory=dict)
    # STATE-DEPENDENT (nonlinear) MASS. When a transient mass term's coefficient depends on the unknown
    # (``c(u)·u_t·v``) the mass cannot be a fixed matrix. It is carried instead as the backward-Euler mass
    # *residual* ``mass_residual(u, t, args) = ∫ c(u)·(u − u_prev)·v`` (its ``1/dt`` applied by the step)
    # and its exact Jacobian ``mass_residual_jac(u, t, args)``; the previous step's nodal values ``u_prev``
    # are delivered by :meth:`step` on ``args["__loadpath__"]`` per the ``prev_state_slices`` metadata.
    # Backward Euler only (θ=1) — a state-dependent mass makes a θ≠1 half-step ill-defined.
    mass_residual: Optional[Callable] = None
    mass_residual_jac: Optional[Callable] = None

    state0: Any = None
    # Optional runtime callback ``state0_fn(args) -> s0`` for a PARAMETRIC initial state (a net-valued
    # initial condition ``u(initial) - net(x)``, recovered from a trajectory). ``state0`` stays as the
    # static placeholder (the net at its stored weights); when ``state0_fn`` is set the integrator re-forms
    # the initial state from ``args`` so ``∂traj/∂weights`` flows through the IC as well as the operator.
    state0_fn: Optional[Callable] = None
    initial_conditions: Any = None

    t0: float = 0.0
    t1: float = 1.0
    dt: Optional[float] = None

    eval_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # linear semidiscrete payload
    M: Any = None
    A: Any = None
    # Optional runtime callback:
    #     operator_fn(t, args) -> A(t, args)
    #
    # For the heat inverse example:
    #     A(t, args) = A0 + args["nu"] * K
    operator_fn: Optional[Callable] = None
    # Optional runtime callback for a PARAMETRIC MASS ``mass_fn(t, args) -> M(t, args)`` on the *linear*
    # path (e.g. an unknown density ``rho(x)*u_t`` recovered from a trajectory). ``M`` stays as the static
    # placeholder (``.M`` / ``represents_linear``); when ``mass_fn`` is set the step re-assembles the mass
    # from ``args`` each step, so ``∂/∂args`` flows through the mass as well as the operator.
    mass_fn: Optional[Callable] = None
    # Diagnostic payload generated during symbolic lowering.
    runtime_parameter_exprs: Dict[str, Any] = field(default_factory=dict)
    operator_basis: Dict[str, Any] = field(default_factory=dict)
    affine_bias: Any = None
    forcing_vector_fn: Optional[Callable] = None
    # Optional affine source/load basis callbacks:
    #     forcing_basis[name](t) -> vector
    forcing_basis: Dict[str, Callable] = field(default_factory=dict)
    # Periodic prolongation matrix P (n_full x n_red); None when absent.
    prolongation: Any = None

    # optional hints
    forcing_mode: str = "none"

    def is_linear(self) -> bool:
        """
        Return True if this block contains a linear semidiscrete payload.

        A block is considered linear when ``M`` and either ``A`` or
        ``operator_fn`` are populated. The represented system is:

            M u_dot + A(t, args) u = affine_bias + forcing_vector_fn(t, args)
        """
        return self.M is not None and (self.A is not None or self.operator_fn is not None)

    def prolong(self, reduced):
        """Map reduced periodic DOFs back to the full nodal layout (single- or multi-field)."""
        if self.prolongation is None:
            return reduced
        if isinstance(self.prolongation, dict):  # multifield periodic carries the per-field reduction
            from .fem_utils import prolong_periodic

            return prolong_periodic(self.prolongation, reduced)
        from .fem_utils import prolong as _prolong

        return _prolong(self.prolongation, reduced)

    def is_nonlinear(self) -> bool:
        """
        Return True if this block contains a nonlinear semidiscrete payload.

        A block is considered nonlinear when both `mass` and `residual`
        callables are populated. The represented system is:

            mass(t) u_dot + residual(u, t) = 0

        A **state-dependent mass** block (``mass_residual`` set) is also nonlinear even if the fixed
        ``mass`` matrix path is unused — the mass action lives in the residual there.
        """
        return (self.mass is not None and self.residual is not None) or (
            self.mass_residual is not None and self.residual is not None
        )

    def step(self, u, t, dt, args=None, theta=None, *, linear_solve=None, nonlinear_solve=None, report=False):
        """Advance the semidiscrete state by one implicit step: ``u(t) -> u(t + dt)``.

        The composable one-step primitive behind :func:`_default_transient_integrate` (which is just
        a ``lax.scan`` over this method) and the building block for operator-splitting / IMEX schemes.
        Functional (returns the next state) and reverse-mode differentiable.

        * **linear** block -> one theta-step
          ``(M + theta dt A) u_next = (M - (1-theta) dt A) u + dt c + dt f`` via the matrix-free
          BiCGStab + Jacobi solver (operators applied only as matvecs, so a BCOO ``A`` stays sparse);
        * **nonlinear** block -> one backward-Euler Newton solve
          ``M(t+dt)(u_next - u)/dt + R(u_next, t+dt, args) = 0`` (matrix-free Newton-Krylov).

        ``theta`` defaults to ``metadata["theta"]`` (1 backward Euler / 1/2 trapezoidal). Operates in
        the block's (periodic-reduced) DOF space; use :meth:`prolong` for the full nodal field.

        The defaults above are overridable — this is where ``fem.solve``'s solver slots plug in
        (see ``jno.utils.solver.solver_api.compose_transient_step_solvers``):

        * ``linear_solve(matvec, rhs, x0, diag_fn) -> x`` replaces the theta-step linear solve. A
          solver that sets ``wants_scale = True`` additionally receives ``scale=`` (the coefficient
          of A in ``M + scale*A``), which jNO's composed step solver uses to pick the right
          pre-built operator when a scheme steps at something other than the block's theta*dt.
          (``matvec`` applies ``M + theta dt A``; ``diag_fn()`` is its exact diagonal; ``x0`` the
          previous state as warm start);
        * ``nonlinear_solve(G, u0) -> u`` replaces the per-step Newton solve.
        * ``report=True`` additionally returns ``(u, ||G(u)||, ||G(u_prev)||)`` on a NONLINEAR
          step, so a marcher can judge the step outside the trace -- see :func:`_verdict`. A
          linear step is a linear solve with its own guard and ignores the flag.
        """
        import jax.numpy as jnp

        args = args or {}
        u = jnp.asarray(u).reshape(-1)
        dtype = u.dtype
        t_next = t + dt

        # A `jno.derived(..., every="step")` field is evaluated ONCE per step, from the state this step
        # starts at, and put on the load-path channel for the whole step. The assembler's `_derived_args`
        # leaves a key a driver already supplied alone, so this decides the cadence on its own: the rule
        # no longer runs inside the Newton loop. The price is an operator splitting -- the step converges,
        # to a problem whose coupling lags by dt -- which is why `every="residual"` is the default.
        _step_derived = {
            _fid: _s for _fid, _s in ((self.metadata or {}).get("derived_specs") or {}).items() if _s["every"] == "step"
        }
        if _step_derived:
            _lp_d = dict(args.get("__loadpath__", {}) or {})
            for _fid, _s in _step_derived.items():
                _xs = [u[a:b] if v == 1 else u[a:b].reshape(-1, v) for (a, b, v) in _s["in_slices"]]
                _lp_d[_fid] = _s["fn"](*_xs, args) if _s["params"] else _s["fn"](*_xs)
            args = {**args, "__loadpath__": _lp_d}

        # keep a BCOO operator as-is (matrix-free matvec) but coerce a dense one to a JAX array
        def _operand(x):
            return x if hasattr(x, "todense") else jnp.asarray(x, dtype)

        if self.is_nonlinear():
            from .newton_krylov import newton_default, newton_krylov

            # θ-method: M(y⁺−y)/dt + θ R(y⁺) + (1−θ) R(y) = 0. θ=1 (default) is backward Euler — the
            # existing first-order behaviour; a second-order (u_tt) block sets θ=½ (trapezoidal /
            # Newmark average-acceleration) so an undamped nonlinear wave is not spuriously damped.
            thn = theta if theta is not None else (float(self.metadata.get("theta", 1.0)) if self.metadata else 1.0)

            # STATE-DEPENDENT MASS: the backward-Euler mass action lives in ``mass_residual(y⁺)`` (its
            # coefficient c(y⁺) cannot be a fixed matrix), so the step residual is
            #     G(y⁺) = mass_residual(y⁺; u_prev=y)/dt + R(y⁺)
            # Newton on G is exact (both matrix-free — jax linearizes through c(y⁺) and (y⁺−y) — and
            # sparse-direct, which adds mass_residual_jac(y⁺)/dt to R's Jacobian). Every step is a θ=1 step:
            # backward Euler takes it from u^n over dt, and BDF2 (`jno.solve.bdf2`) from the shifted state
            # u* = (4u^n - u^{n-1})/3 over 2dt/3, which lands on BDF2's own non-conservative mass action.
            if self.mass_residual is not None:
                if abs(thn - 1.0) > 1e-12:
                    raise ValueError(
                        "jno.fem: a state-dependent (nonlinear) transient mass `c(u)·u_t` is marched by backward "
                        f"Euler or BDF2, not by a theta-step with theta={thn:g}. A θ≠1 half-step needs the mass at "
                        "the half-state, which is ill-defined for a coefficient that depends on the unknown. "
                        "Drop `jno.solve.theta(...)` (use the default), or use `jno.solve.bdf2()` for second order."
                    )
                # Deliver the previous state y as each prev-field's nodal slice on the load-path channel.
                # A vector field's DOFs are node-major interleaved (node·vec + comp), so reshape its slice to
                # (n_nodes, vec); a scalar field stays 1-D. The assembler's load-path gather handles either.
                _lp = dict((args or {}).get("__loadpath__", {}) or {})
                _uprev = jnp.asarray(u, dtype).reshape(-1)
                for _fid, _s0, _s1, _vec in self.metadata.get("prev_state_slices", []):
                    _slice = _uprev[_s0:_s1]
                    _lp[_fid] = _slice if _vec == 1 else _slice.reshape(-1, _vec)
                _ap = {**(args or {}), "__loadpath__": _lp}

                def G(wn):
                    m = jnp.asarray(self.mass_residual(wn, t_next, _ap), dtype).reshape(-1) / dt
                    return m + jnp.asarray(self.residual(wn, t_next, args), dtype).reshape(-1)

                if nonlinear_solve is not None:
                    if getattr(nonlinear_solve, "wants_jacobian", False) and self.mass_residual_jac is not None:
                        from .solver_api import _add_step_operator

                        def jac_step(wn):
                            # J = J_spatial(wn) + (1/dt)·J_mass(wn); both assembled BCOO (exact ∂M/∂u).
                            return _add_step_operator(
                                self.jacobian(wn, t_next, args), self.mass_residual_jac(wn, t_next, _ap), 1.0 / dt
                            )

                        return _verdict(G, u, nonlinear_solve(G, u, jacobian=jac_step), report)
                    return _verdict(G, u, nonlinear_solve(G, u), report)
                # default Newton: on the assembled step tangent when both Jacobians exist
                if self.jacobian is not None and self.mass_residual_jac is not None:
                    from .solver_api import _add_step_operator

                    def jac_default(wn):
                        return _add_step_operator(
                            self.jacobian(wn, t_next, args), self.mass_residual_jac(wn, t_next, _ap), 1.0 / dt
                        )

                    return _verdict(G, u, newton_default(G, u, jacobian=jac_default), report)
                return _verdict(G, u, newton_krylov(G, u), report)

            M_t = _operand(self.mass(t_next, args))
            # θ per row: the zero-mass (constraint) rows are imposed at t+dt; see `_theta_row_weights`.
            w = _theta_row_weights(M_t, thn, u.size, dtype)
            r_now = (1.0 - w) * jnp.asarray(self.residual(u, t, args), dtype).reshape(-1) if w is not None else None

            def G(wn):
                r_next = jnp.asarray(self.residual(wn, t_next, args), dtype).reshape(-1)
                g = (M_t @ (wn - u)) / dt + (r_next if w is None else w * r_next)
                return g if r_now is None else g + r_now

            if nonlinear_solve is not None:
                # A sparse-direct Newton (``jno.solve.newton(direct=True)``) factorizes the ASSEMBLED
                # step tangent each iteration rather than a matrix-free Krylov inner solve — it flags
                # ``wants_jacobian`` so we build the backward-Euler step Jacobian here and thread it in.
                # The step residual is ``G = M(t+dt)(wn-u)/dt + R(wn, t+dt)`` so its Jacobian is
                # ``M(t+dt)/dt + jacobian(wn, t+dt)`` (reusing the assembled ``self.jacobian``). Every
                # other nonlinear driver stays matrix-free (jacobian left None).
                if getattr(nonlinear_solve, "wants_jacobian", False) and self.jacobian is not None:
                    from .solver_api import _add_step_operator

                    def jac_step(wn):  # ∂G/∂wn = M/dt + diag(w)·J_R; it used to drop the θ, a wrong tangent for θ < 1
                        J = self.jacobian(wn, t_next, args)
                        return _add_step_operator(J if w is None else _row_scaled(J, w), M_t, 1.0 / dt)

                    return _verdict(G, u, nonlinear_solve(G, u, jacobian=jac_step), report)
                return _verdict(G, u, nonlinear_solve(G, u), report)
            # default Newton: on the assembled step tangent M/dt + J when the assembler provides J
            if self.jacobian is not None:
                from .solver_api import _add_step_operator

                def jac_default(wn):  # the same ∂G/∂wn = M/dt + diag(w)·J_R as `jac_step`
                    J = self.jacobian(wn, t_next, args)
                    return _add_step_operator(J if w is None else _row_scaled(J, w), M_t, 1.0 / dt)

                return _verdict(G, u, newton_default(G, u, jacobian=jac_default), report)
            return _verdict(G, u, newton_krylov(G, u), report)

        from .linear import matrix_diagonal, sparse_matvec

        th = theta if theta is not None else (float(self.metadata.get("theta", 1.0)) if self.metadata else 1.0)
        # A parametric mass (``mass_fn``) is re-assembled from ``args`` each step (unknown-density inverse);
        # otherwise the static ``self.M``.
        M = _operand(self.mass_fn(t_next, args) if self.mass_fn is not None else self.M)
        n = M.shape[0]
        c = jnp.zeros((n,), dtype) if self.affine_bias is None else jnp.asarray(self.affine_bias, dtype).reshape(-1)
        A = _operand(self.operator_fn(t_next, args) if self.operator_fn is not None else self.A)

        def _forcing(tt):
            if self.forcing_vector_fn is None:
                return jnp.zeros((n,), dtype)
            return jnp.asarray(self.forcing_vector_fn(tt, args), dtype).reshape(-1)

        # (M + theta dt A) u_next = (M - (1-theta) dt A) u + dt c + dt(theta f_next + (1-theta) f_now)
        f_next = _forcing(t_next)
        f_avg = th * f_next + (1.0 - th) * _forcing(t)
        # Index work split once per step, outside the Krylov loop that applies `step_op` (BCOO's own `@`
        # redoes it on every call; see `sparse_matvec`).
        mv_M, mv_A = sparse_matvec(M), sparse_matvec(A)
        rhs = mv_M(u) - (1.0 - th) * dt * mv_A(u) + dt * c + dt * f_avg
        step_op = lambda wn: mv_M(wn) + th * dt * mv_A(wn)  # noqa: E731  the theta-method step operator
        a_scale = th * dt  # the coefficient of A in the step operator, per row where it differs
        w = _theta_row_weights(M, th, n, dtype)
        if w is not None:
            # Zero-mass (constraint) rows are imposed at t+dt: ``A_i u⁺ = c_i + f_i(t+dt)``, see
            # `_theta_row_weights`. Scaled by θ·dt for θ > 0 so the step operator stays M + θ·dt·A (the
            # matrix a composed solver pre-builds); at θ = 0 that row of M + 0·A is empty, so it takes dt.
            kappa = th if th > 0.0 else 1.0
            alg = w == 1.0  # θ < 1 here, so weight 1 marks exactly the constraint rows
            rhs = jnp.where(alg, kappa * dt * (c + f_next), rhs)
            if th == 0.0:
                step_op = lambda wn: mv_M(wn) + dt * (w * mv_A(wn))  # noqa: E731
                a_scale = dt * w
        if linear_solve is not None:
            # slot-composed per-step solve; the exact step diagonal keeps jacobi-type specs exact
            # ``scale`` is the coefficient of A in the step operator (M + scale*A). A scheme may take a
            # step that is not the block's own theta*dt -- BDF2 uses 2dt/3, and an adaptive march
            # re-sizes dt every step -- and the composed solver needs it to build the RIGHT operator
            # rather than the block's default one.
            _diag = lambda: matrix_diagonal(M) + a_scale * matrix_diagonal(A)  # noqa: E731
            # `scale` is OPT-IN. The documented contract for a caller-supplied `linear_solve` is
            # `(matvec, rhs, x0, diag_fn)`; only jNO's own composed step solver advertises that it can
            # also take the step scale, so only it is handed one.
            if getattr(linear_solve, "wants_scale", False):
                return linear_solve(step_op, rhs, u, _diag, scale=th * dt)
            return linear_solve(step_op, rhs, u, _diag)
        # diagonal (Jacobi) preconditioner 1/diag(M + theta dt A); zero diagonals left unscaled
        d = matrix_diagonal(M) + a_scale * matrix_diagonal(A)
        return _default_step_solve(step_op, rhs, u, d, krylov=(self.metadata or {}).get("krylov"))

    def solve(self, solve_fn=None, *, save_ts=None):
        """Differentiable transient forward solve -> the trajectory ``u(save_ts)`` as a
        trace node (mirrors :meth:`FemLinearSystem.solve` for the steady case).

        When evaluated (e.g. inside ``crux.solve``) any runtime parameters are resolved to
        their current values and ``solve_fn(self, args, save_ts)`` integrates the block;
        gradients flow back to the parameters through the integrator, so a *time-dependent*
        inverse problem is just::

            alpha = jno.np.parameter((1,), name="alpha")
            fem = jno.fem([ui.t * vi + alpha * (ui.x*vi.x + ui.y*vi.y),  # transient + parametric
                           u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])
            crux = jno.core([(fem.solve() - u_obs).mse], domain=obs)
            crux.solve(n)                       # recovers alpha from the u(t) trajectory

        ``solve_fn`` is **your** integrator: any ``(block, args, save_ts) -> ys`` callable
        returning a ``(len(save_ts), n_dofs)`` trajectory; jNO writes none and imposes no
        library. The default :func:`_default_transient_integrate` is a backward-Euler
        ``lax.scan`` over the block's own assembled ``dt``. To bring your own integrator,
        build it from the block's flat pieces -- ``block.M``, ``block.A`` (or
        ``block.operator_fn(t, args)``) and ``block.state0`` -- and form ``u_dot = M^-1(c - A u)``.
        Note a Dirichlet problem zeroes M's Dirichlet rows (a DAE), so the implicit
        ``(M + dt A)`` default is preferred there; an explicit field must hold those rows.

        Enable x64 (``jax_enable_x64``); the assembly is float64.
        """
        from ...trace import FunctionCall  # lazy: avoid an import cycle with jno.trace

        if solve_fn is None:
            solve_fn = _default_transient_integrate
        if save_ts is None:
            save_ts = _block_time_grid(self)

        names = list(self.runtime_parameter_exprs)
        params = [self.runtime_parameter_exprs[n] for n in names]

        def _solve(*values):
            import time as _time

            import jax

            from .history_march import LAST_MARCH_STATS

            LAST_MARCH_STATS.clear()
            _t_eval = _time.perf_counter()
            ys = solve_fn(self, dict(zip(names, values)), save_ts)
            if not isinstance(ys, jax.core.Tracer):
                # An EAGER evaluation (`.fn()`): record what it did for `fem.stats["march"]`. Under
                # jno.core / jit this is a tracer and nothing is recorded -- a step inside the
                # compiled scan is not a host-visible event.
                #
                # NEVER block to get a time: `.fn()` returns asynchronously and must stay that way.
                # A nonlinear march has ALREADY synchronised (its per-step convergence check reads
                # the residuals on the host), so its elapsed time is real and free; a linear march
                # has not, and timing it would mean forcing the very sync this must not add.
                rec = dict(LAST_MARCH_STATS)
                synced = rec.get("residual") is not None
                self._n_evaluations = getattr(self, "_n_evaluations", 0) + 1
                self._last_evaluation = {
                    **rec,
                    "wall_s": (_time.perf_counter() - _t_eval) if synced else None,
                    "evaluation": self._n_evaluations,
                    "at": _t_eval,
                }
                if not synced:
                    self._last_evaluation["note"] = (
                        "not timed: .fn() returns asynchronously, and timing it would force a device "
                        "sync — wrap it in jax.block_until_ready yourself to time it"
                    )
            if self.prolongation is not None:
                # Periodic tie: the block integrates in the reduced main-DOF space. Prolong each saved
                # step ``u = P·u_red`` back to the full nodal layout, so the returned trajectory lives on the
                # mesh nodes (matching ``fem.points`` / ``fem.offsets``) -- exactly as the steady operators
                # (``FemLinearSystem.solve``) and the complex-transient path already do. Without this a
                # periodic transient hands back reduced DOFs a caller then mis-slices with full offsets.
                import jax

                ys = jax.vmap(self.prolong)(ys)
            if (self.metadata or {}).get("complex"):
                # A complex transient integrates as the real-equivalent 2n block over y=[u_r; u_i].
                # Recombine ONCE, here, after any periodic prolongation -- P is real and linear, so
                # prolong-then-split is identical to split-then-prolong, and doing it last means the
                # marcher, the time schemes and the solver slots all stay real-only and complex-unaware.
                h = ys.shape[-1] // 2
                ys = ys[..., :h] + 1j * ys[..., h:]
            return ys

        return FunctionCall(_solve, params, name="fem_transient_solve")


def _block_time_grid(block):
    """The block's own integration grid ``t0 .. t1`` at its assembled step ``dt`` -- the
    default ``save_ts`` (the domain's ``time=(t0, t1, n_time)`` grid)."""
    import jax.numpy as jnp

    t0, t1, dt = float(block.t0), float(block.t1), float(block.dt)
    n_steps = max(1, round((t1 - t0) / dt))
    return jnp.linspace(t0, t1, n_steps + 1)


def _refreshing_transient_integrate(block, args, save_ts, *, cadence, compose, theta=None):
    """March in chunks of ``cadence`` steps, re-composing the per-step solvers between them.

    A preconditioner whose setup cannot run under a trace (an AMG hierarchy, an ILU) is frozen once,
    before the scan. That is correct and fast while the operator stays put -- measured 4.8x faster and
    7.4x lighter than re-factorising per linearisation on a melt pool -- and it stalls when the
    operator does not stay put: between solid and molten that same problem moves its Carman-Kozeny
    drag by 1e13, its PSPG ``tau`` by 1e9 and its stiffness by 1e6, and a setup frozen on the cold
    state stagnates at ~1e-4 however many Krylov iterations it is given.

    ``jno.precond.cached(spec, refresh=k)`` already spells the remedy -- "rebuilds every k-th
    materialization -- the cadence policy for a ... transient march whose operator values drift step by
    step". A rebuild must happen OUTSIDE the trace, so honouring it means running ``ceil(n/k)`` scans
    instead of one, each starting from the carried state and re-frozen against it.

    The compiled step is shared across chunks (identical shapes), so the extra cost is one host-side
    setup per chunk. Reverse mode still works -- each chunk's scan is checkpointed exactly as before.
    """
    import dataclasses

    import jax.numpy as jnp
    import numpy as _np

    ts = _np.asarray(save_ts, dtype=float).reshape(-1)
    t0, t1, dt = float(block.t0), float(block.t1), float(block.dt)
    n_steps = max(1, round((t1 - t0) / dt))
    tol = 1e-9 * max(abs(t1 - t0), 1.0)
    state, out, lo, done = block.state0, [], t0, 0
    while done < n_steps:
        k = int(min(cadence, n_steps - done))
        hi = t0 + (done + k) * dt
        lo_ok = ts >= lo - tol if not out else ts > lo + tol
        sel = ts[lo_ok & (ts <= hi + tol)]
        sub = _np.unique(_np.concatenate([sel, _np.asarray([hi])]))
        # state0_fn would re-form the INITIAL state from args and undo the carry, so it goes with it.
        chunk = dataclasses.replace(block, t0=lo, t1=hi, state0=state, state0_fn=None)
        lin_s, nonlin_s = compose(chunk, state)
        ys = _default_transient_integrate(
            chunk, args, jnp.asarray(sub), linear_solve=lin_s, nonlinear_solve=nonlin_s, theta=theta
        )
        state = ys[-1]
        keep = _np.isin(sub, sel)
        if keep.any():
            out.append(ys[_np.flatnonzero(keep)])
        done += k
        lo = hi
    return jnp.concatenate(out, axis=0)


def _default_transient_integrate(block, args, save_ts, *, linear_solve=None, nonlinear_solve=None, theta=None):
    """Default transient integrator: backward Euler at the block's *own* assembled step ``dt``,
    advanced with ``jax.lax.scan`` (reverse-mode differentiable) and sampled at ``save_ts`` by
    linear interpolation, so the integration step is always the assembled ``dt`` and never an
    accident of how the output is sampled.

    * **Linear** block -- the implicit scheme the transient assembly is built for::

          (M + dt A(t_next, args)) u_next = M u + dt (c + f(t_next, args))

    * **Nonlinear** block (residual route, ``M(t) u_dot = -R(u, t, args)``) -- backward Euler
      solves, per step, ``G(u_next) = M (u_next - u)/dt + R(u_next, t_next, args) = 0`` with the
      matrix-free Newton-Krylov solver (``jno/utils/solver/newton_krylov.py``, no optimistix);
      implicit-diff via ``jax.lax.custom_root`` keeps the gradient flowing to ``args`` without
      unrolling Newton -- the same solver the steady nonlinear ``.solve`` now uses.

    This is a *default*: pass any ``solve_fn(block, args, save_ts) -> ys`` to
    :meth:`SemidiscreteTimeBlock.solve` (a hand-rolled stepper built from the block's
    ``M`` / ``A`` / ``state0``) to use a different integrator.
    """
    import jax
    import jax.numpy as jnp

    sharded = _sharded_transient(block, args, save_ts, linear_solve, nonlinear_solve, theta)
    if sharded is not None:
        return sharded

    from .matvec_format import prime

    prime(block.M, getattr(block, "A", None))  # CSR or COO, measured on the real (concrete) operators
    _s0f = getattr(block, "state0_fn", None)  # parametric initial state (net-valued IC): re-form from args
    s0 = jnp.asarray(_s0f(args) if _s0f is not None else block.state0).reshape(-1)
    dtype = s0.dtype
    grid_ts = jnp.asarray(_block_time_grid(block), dtype)
    dt = float(block.dt)

    # One scan step = one implicit advance of the block. `block.step` is the single definition of
    # that step (theta-method for a linear block, backward-Euler Newton for a nonlinear one); read
    # theta from the block so a linear step uses the assembled scheme. Operators are only applied as
    # matvecs inside block.step, so a BCOO operator stays sparse.
    if theta is None:  # jno.solve.theta(...) overrides the assembly's default (1 backward-Euler / ½ trapezoidal)
        theta = float(block.metadata.get("theta", 1.0)) if getattr(block, "metadata", None) else 1.0

    # A NONLINEAR step is a Newton solve, and the driver's own convergence guard cannot fire inside
    # this scan (it needs a concrete residual). So the step hands its residual norms back and they are
    # judged below, outside the trace -- the arrangement the load-path march already uses. A linear
    # step is a linear solve with its own guard and is not judged here.
    _judge = bool(block.is_nonlinear())

    def march(s0, grid_ts, args):
        blk = hoist_time_invariant(block, args, grid_ts[0])  # static loads/operators: once, not per step

        def step(w, t_next):
            out = blk.step(
                w,
                t_next - dt,
                dt,
                args=args,
                theta=theta,
                linear_solve=linear_solve,
                nonlinear_solve=nonlinear_solve,
                report=_judge,
            )
            if not _judge:
                return out, out
            wn, r_end, r_start = out
            return wn, (wn, r_end, r_start)

        return jax.lax.scan(jax.checkpoint(step), s0, grid_ts[1:])[1]

    # ``jax.checkpoint`` on the scan body: reverse-mode otherwise saves every step's *internal*
    # residuals (the rhs, the θ-combination, the Krylov solve's saved primals — measured ~32 vectors
    # per step) for the whole march; the trajectory itself is the scan output and is kept either way.
    # Rematerializing the step in the backward pass trades one forward recompute for that stash.
    # Measured at 8,355 DOFs × 399 steps (RTX 3070, x64): peak memory **967.7 → 112.0 MB (8.6×)** for
    # a gradient cost of **2975.5 → 4756.0 ms (+60%)**, gradient identical to 10 digits. The memory
    # side wins the default: a differentiable march OOMs long before it is time-walled on the cards
    # this library targets (an un-checkpointed 6000-step × 18k-DOF case failed to allocate 5.72 GiB
    # on an 8 GB card — see the sampling note below). A pure forward solve pays nothing — checkpoint
    # is the identity outside differentiation.
    from .sharding import element_devices

    # A nonlinear march may split its element loops over devices (`_element_split_devices`): the split is
    # read while the march is TRACED, so it is active around the (cached) trace and part of the cache key --
    # a program traced for one device must not be reused for a split run, or the reverse.
    split = _element_split_devices(block, args)
    with element_devices(split):
        ys = _cached_march(
            block, (linear_solve, nonlinear_solve, theta, dt, tuple(d.id for d in split)), march, s0, grid_ts, args
        )
    if _judge:
        from .history_march import _TRANSIENT_ADVICE, _check_march_converged

        ys, _r_end, _r_start = ys
        _check_march_converged(
            _r_end,
            _r_start,
            grid_ts[1:],
            nonlinear_solve,
            states=ys,
            what="transient march",
            coord="t",
            advice=_TRANSIENT_ADVICE,
        )

    traj = jnp.concatenate([s0[None, :], ys], axis=0)  # (n_grid, n_dofs) at grid_ts
    return _resample_trajectory(traj, grid_ts, save_ts, dtype)


_PER_STEP_CALLABLES = ("forcing_vector_fn", "operator_fn", "mass_fn", "mass")


def hoist_time_invariant(block, args, t0):
    """``block`` with every per-step ``f(t, args)`` that does not depend on ``t`` evaluated ONCE.

    The step re-evaluates ``forcing_vector_fn``, ``operator_fn`` and ``mass_fn`` (linear route) or
    ``mass`` (nonlinear route) at every step -- the forcing twice, at ``t`` and ``t + dt``. The assembled
    forcing is ``-R(0, t)``: the WHOLE spatial residual re-assembled at zero state, element loop and
    Jacobian inverses included, whether or not the problem has a source. Measured on a 69k-DOF P1 heat
    march with no source: removing it took the march 511 -> 235 ms. In a parametric/inverse march
    ``operator_fn(t, args)`` re-assembles the operator every step for the same reason.

    Independence is decided EXACTLY, not guessed: each callable is traced at ``(t, args)`` and
    dead-code-eliminated; ``t`` is independent only when no surviving equation reads it (assembly threads
    ``t`` through the quadrature points even when no term uses it, so "t appears" would never hoist).
    A hoisted value is computed from the march's own ``args``, so gradients with respect to them flow
    through it unchanged. Anything that does depend on ``t`` (a time-dependent source or Dirichlet
    value) -- or whose independence cannot be proven -- keeps its per-step evaluation.

    Call it inside the march, before the time loop. Returns ``block`` itself when nothing hoists.
    """
    import dataclasses

    changes = {}
    for name in _PER_STEP_CALLABLES:
        fn = getattr(block, name, None)
        if not callable(fn) or not _independent_of_t(fn, t0, args):
            continue
        value = fn(t0, args)
        changes[name] = lambda t, a=None, _v=value: _v
    if not changes or not dataclasses.is_dataclass(block):
        return block
    return dataclasses.replace(block, **changes)


def _independent_of_t(fn, t0, args) -> bool:
    import jax

    try:
        closed = jax.make_jaxpr(lambda t: fn(t, args))(t0)
        from jax._src.interpreters import partial_eval as pe

        _, used = pe.dce_jaxpr(closed.jaxpr, [True] * len(closed.jaxpr.outvars))
        return not used[0]
    except Exception:  # noqa: BLE001 -- cannot prove independence: keep the per-step evaluation
        return False


_MARCH_CACHE_SIZE = 4  # per block: the configurations re-evaluated in turn (e.g. two solver slots)


def _cached_march(block, config, march, *inputs):
    """Run ``march(*inputs)``, reusing its trace and compiled program across EAGER evaluations.

    Called eagerly (``fem.solve().fn()``), ``jax.lax.scan`` traces its body into a fresh jaxpr on every
    call, so JAX's dispatch cache never hits and each evaluation re-traces ``block.step``, re-lowers the
    march and re-fetches the executable. Measured on a 19k-DOF, 19-step P1 heat march (RTX 3070): ~160 ms
    tracing + ~120 ms lowering + ~60 ms cache fetch against ~115 ms of GPU work -- the same cost for a
    brand-new node and for re-evaluating the same one.

    Here the march is traced ONCE per (block, configuration, input shapes) with ``make_jaxpr`` and run
    through a ``jax.jit`` of ``eval_jaxpr`` that takes the jaxpr's constants as ARGUMENTS. Jitting the
    closure directly would bake the operators and mesh arrays into the executable as constants; as
    arguments they stay the block's own device buffers.

    Only the eager path is cached. Under an outer trace (``jno.core``, ``jax.grad``, ``vmap``) an input
    is a tracer and the march runs inline exactly as before -- the enclosing ``jit`` owns caching there.
    The cache lives on the block and is keyed on the identity of the block's public fields, so
    reassigning any of them (jNO does, e.g. for contact and coupled residuals) re-traces rather than
    reusing stale constants. At most ``_MARCH_CACHE_SIZE`` configurations are kept per block.
    """
    import collections

    import jax
    import jax.numpy as jnp

    leaves, treedef = jax.tree_util.tree_flatten(inputs)
    if any(isinstance(x, jax.core.Tracer) for x in leaves):
        return march(*inputs)
    sig = (
        _value_identity(config),
        treedef,
        tuple((jnp.shape(x), jnp.result_type(x)) for x in leaves),
        _block_fingerprint(block),
        _trace_time_settings(),
    )
    cache = block.__dict__.setdefault("_march_cache", collections.OrderedDict())
    hit = cache.get(sig)
    if hit is None:
        closed, out_shape = jax.make_jaxpr(march, return_shape=True)(*inputs)
        if any(isinstance(c, jax.core.Tracer) for c in closed.consts):  # the block itself holds tracers
            return march(*inputs)
        jaxpr = closed.jaxpr
        from .placement import to_solve_device

        # The constants (operators, mesh arrays) move to the solving device ONCE, here, instead of being
        # copied from the host on every run; configurations cached on the same block share the copies.
        hit = (
            to_solve_device(list(closed.consts), block.__dict__.setdefault("_device_consts", {}), numpy=True),
            jax.jit(lambda consts, flat: jax.core.eval_jaxpr(jaxpr, consts, *flat)),
            jax.tree_util.tree_structure(out_shape),
        )
        cache[sig] = hit
        while len(cache) > _MARCH_CACHE_SIZE:
            cache.popitem(last=False)
            # ...and the device copies only the evicted configuration used go with it.
            live = {id(c) for h in cache.values() for c in h[0]}
            memo = block.__dict__["_device_consts"]
            for k in [k for k, (_, moved) in memo.items() if id(moved) not in live]:
                del memo[k]
    else:
        cache.move_to_end(sig)
    consts, run, out_tree = hit
    return jax.tree_util.tree_unflatten(out_tree, run(consts, leaves))


def _value_identity(config):
    """``config`` with each composed solver replaced by its VALUE identity (``cache_key``) where it has one.

    ``fem.solve`` composes fresh per-step solvers on every call, so keying the march cache on the objects
    themselves missed every time: a warm nonlinear march with ``time=`` (or any solver slot) re-traced and
    re-compiled on every call -- measured ~1.1 s per call on a 3k-DOF 2-D problem whose march takes 30 ms.
    The value identity describes the solver completely (the compiled-solve caches already key on it).
    """
    if not isinstance(config, tuple):
        return config
    out = []
    for c in config:
        k = getattr(c, "cache_key", None)
        out.append(("cache_key", k) if k is not None else c)
    return tuple(out)


def _trace_time_settings():
    """The ``jno.setup`` settings a march bakes in when it is TRACED: the sparse-operator storage
    (``matvec_format``) and how many systems a vmapped device LU stacks (``lu_stack``). Their setters
    clear JAX's caches, which does not reach this block-level cache -- so they are part of its key, or a
    changed setting would silently keep the old one for any block that had already marched."""
    from . import matvec_format, sparse_batching

    return (matvec_format._FORMAT, sparse_batching._LU_STACK)


def _block_fingerprint(block):
    """Identity of every public field (dict fields one level deep). Private attributes are excluded:
    the evaluation bookkeeping (``_n_evaluations``, ``_last_evaluation``) changes on every call."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(block)} if dataclasses.is_dataclass(block) else set()
    out = []
    for k, v in sorted(vars(block).items()):
        if k.startswith("_") and k not in fields:
            continue
        if isinstance(v, dict):
            out.append((k, tuple(sorted((str(kk), id(vv)) for kk, vv in v.items()))))
        else:
            out.append((k, id(v)))
    return tuple(out)


def _element_split_devices(block, args):
    """The devices a NONLINEAR march splits its cells over, or ``[]`` to stay on one.

    A Newton step has no assembled operator to partition, so each residual (and each ``J.v``) is
    evaluated per device on a share of the elements, with one all-reduce -- the steady nonlinear route,
    applied inside the scan (:func:`jno.utils.solver.sharding.sharded_element_add`). Taken for a march the
    FEM dispatch marked plain (``split_cells``: no ``adapt=``, no moving geometry), evaluated eagerly:
    under a trace the constraint would have to agree with the device commitments of the caller's ``jit``,
    which cannot be known here (``docs/fem/inverse.md``)."""
    import jax

    from .sharding import resolve_devices

    md = getattr(block, "metadata", None) or {}
    if not (block.is_nonlinear() and md.get("split_cells")):
        return []
    if any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(args)):
        return []
    return resolve_devices(md.get("shard"))


class _TripletOperator:
    """An assembled operator as its ``(data, row, col)`` triplets, applied by an explicit segment-sum.

    Stands in for the BCOO inside a sharded march: with the triplets partitioned on their nonzero axis
    the segment-sum is partial per device and XLA combines it with one all-reduce -- exactly the
    matvec :func:`jno.utils.solver.sharding.sharded_solve` uses. ``todense`` marks it as an operator
    (not an array) for :meth:`SemidiscreteTimeBlock.step`, and ``matrix_diagonal`` reads it through
    ``indices``/``data``/``shape`` as it reads a BCOO."""

    todense = None

    def __init__(self, data, indices, shape):
        self.data, self.indices, self.shape = data, indices, tuple(shape)

    def __matmul__(self, v):
        import jax

        return jax.ops.segment_sum(self.data * v[self.indices[:, 1]], self.indices[:, 0], num_segments=self.shape[0])


def _sharded_transient(block, args, save_ts, linear_solve, nonlinear_solve, theta):
    """Run a LINEAR march across every visible device, or return ``None`` to stay on one.

    The assembled ``M`` and ``A`` are partitioned on their nonzero axis and passed into the compiled
    scan as jit ARGUMENTS -- closed over, they would be baked in as constants and replicated to every
    device, with the right answer and no memory saving. The state vector stays replicated, so each
    step's Krylov solve is unchanged: every matvec is partial per device plus one all-reduce, every
    vector operation identical on all devices. The answer moves only by reduction order.

    Taken automatically (``fem.solve(shard=...)`` opts out or pins devices, as for a steady solve) for
    a linear, non-parametric march on assembled operators with the default step solve, evaluated
    eagerly. A nonlinear march, a parametric operator, solver slots or a traced evaluation keep the
    single-device path."""
    import jax

    from .sharding import SHARD_AXIS, operator_mesh, pad_triplets, resolve_devices, shard_triplets

    shard = (getattr(block, "metadata", None) or {}).get("shard")
    if shard is False or shard == 1:
        return None
    eligible = (
        not block.is_nonlinear()
        and getattr(block, "mass_fn", None) is None
        and getattr(block, "operator_fn", None) is None
        and getattr(block, "state0_fn", None) is None
        and linear_solve is None
        and nonlinear_solve is None
        and hasattr(block.M, "indices")
        and hasattr(block.A, "indices")
        and not any(isinstance(v, jax.core.Tracer) for v in jax.tree_util.tree_leaves(args))
    )
    if not eligible:
        return None
    devices = resolve_devices(shard)
    if not devices:
        return None
    import copy

    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = operator_mesh(devices)
    nd = len(devices)
    Md, Mi, _ = pad_triplets(block.M.data, block.M.indices, nd)
    Ad, Ai, _ = pad_triplets(block.A.data, block.A.indices, nd)
    Md, Mi = shard_triplets(Md, Mi, mesh)
    Ad, Ai = shard_triplets(Ad, Ai, mesh)
    split, repl = NamedSharding(mesh, P(SHARD_AXIS)), NamedSharding(mesh, P())
    shape_m, shape_a = block.M.shape, block.A.shape

    def march(md, mi, ad, ai):
        local = copy.copy(block)
        local.M, local.A = _TripletOperator(md, mi, shape_m), _TripletOperator(ad, ai, shape_a)
        local.metadata = {**(block.metadata or {}), "shard": False}  # the recursion runs the plain scan
        return _default_transient_integrate(local, args, save_ts, theta=theta)

    return jax.jit(march, in_shardings=(split, split, split, split), out_shardings=repl)(Md, Mi, Ad, Ai)


def _resample_trajectory(traj, grid_ts, save_ts, dtype):
    """Sample a march's own-grid trajectory at ``save_ts``. Shared by every scheme's ``integrate``,
    so they cannot drift apart on the fast path or the clamping convention."""
    import jax.numpy as jnp
    import numpy as np

    save_ts = jnp.asarray(save_ts, dtype)

    # The DEFAULT save_ts is the block's own grid (``solve`` fills it from ``_block_time_grid``), so
    # the sampling below is an identity -- but it used to be paid for anyway, and expensively: a
    # per-DOF-column vmap of ``jnp.interp`` over the whole trajectory. At 6000 steps x 18k DOFs the
    # trajectory is 878 MB, and the resampled copy plus the vmap's workspace on top of it is what
    # made that case fail to allocate 5.72 GiB on an 8 GB card. Return it directly instead.
    try:
        same_grid = save_ts.shape == grid_ts.shape and bool(np.allclose(np.asarray(save_ts), np.asarray(grid_ts)))
    except Exception:  # traced save_ts: cannot compare values, take the general path
        same_grid = False
    if same_grid:
        return traj

    # Otherwise interpolate ONCE for all DOFs rather than once per column: locate each save time in
    # the grid, then blend the two bracketing states. Matches ``jnp.interp``'s clamping outside the
    # grid (weight clipped to [0, 1] holds the endpoint state).
    hi = jnp.clip(jnp.searchsorted(grid_ts, save_ts, side="right"), 1, grid_ts.size - 1)
    lo = hi - 1
    span = grid_ts[hi] - grid_ts[lo]
    w = jnp.clip(jnp.where(span > 0, (save_ts - grid_ts[lo]) / jnp.where(span > 0, span, 1.0), 0.0), 0.0, 1.0)
    return traj[lo] * (1.0 - w[:, None]) + traj[hi] * w[:, None]
