from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------
# Solver-agnostic semidiscrete block returned by weak.assemble(...)
# ---------------------------------------------------------------------


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

    state0: Any = None
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
        """
        return self.mass is not None and self.residual is not None

    def step(self, u, t, dt, args=None, theta=None, *, linear_solve=None, nonlinear_solve=None):
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

        * ``linear_solve(matvec, rhs, x0, diag_fn) -> x`` replaces the theta-step linear solve
          (``matvec`` applies ``M + theta dt A``; ``diag_fn()`` is its exact diagonal; ``x0`` the
          previous state as warm start);
        * ``nonlinear_solve(G, u0) -> u`` replaces the per-step Newton solve.
        """
        import jax
        import jax.numpy as jnp

        args = args or {}
        u = jnp.asarray(u).reshape(-1)
        dtype = u.dtype
        t_next = t + dt

        # keep a BCOO operator as-is (matrix-free matvec) but coerce a dense one to a JAX array
        def _operand(x):
            return x if hasattr(x, "todense") else jnp.asarray(x, dtype)

        if self.is_nonlinear():
            from .newton_krylov import newton_krylov

            # θ-method: M(y⁺−y)/dt + θ R(y⁺) + (1−θ) R(y) = 0. θ=1 (default) is backward Euler — the
            # existing first-order behaviour; a second-order (u_tt) block sets θ=½ (trapezoidal /
            # Newmark average-acceleration) so an undamped nonlinear wave is not spuriously damped.
            thn = theta if theta is not None else (float(self.metadata.get("theta", 1.0)) if self.metadata else 1.0)
            M_t = _operand(self.mass(t_next, args))
            r_now = (1.0 - thn) * jnp.asarray(self.residual(u, t, args), dtype).reshape(-1) if thn < 1.0 else None

            def G(wn):
                g = (M_t @ (wn - u)) / dt + thn * jnp.asarray(self.residual(wn, t_next, args), dtype).reshape(-1)
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

                    def jac_step(wn):
                        return _add_step_operator(self.jacobian(wn, t_next, args), M_t, 1.0 / dt)

                    return nonlinear_solve(G, u, jacobian=jac_step)
                return nonlinear_solve(G, u)
            return newton_krylov(G, u)

        from .linear import matrix_diagonal

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
        f_avg = th * _forcing(t_next) + (1.0 - th) * _forcing(t)
        rhs = M @ u - (1.0 - th) * dt * (A @ u) + dt * c + dt * f_avg
        step_op = lambda wn: M @ wn + th * dt * (A @ wn)  # noqa: E731  the theta-method step operator
        if linear_solve is not None:
            # slot-composed per-step solve; the exact step diagonal keeps jacobi-type specs exact
            return linear_solve(step_op, rhs, u, lambda: matrix_diagonal(M) + th * dt * matrix_diagonal(A))
        # diagonal (Jacobi) preconditioner 1/diag(M + theta dt A); zero diagonals left unscaled
        d = matrix_diagonal(M) + th * dt * matrix_diagonal(A)
        inv = 1.0 / jnp.where(jnp.abs(d) > 1e-30, d, 1.0)
        wn, _ = jax.scipy.sparse.linalg.bicgstab(
            step_op, rhs, x0=u, tol=1e-10, atol=0.0, maxiter=20_000, M=lambda x: inv * x
        )
        return wn

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
            return solve_fn(self, dict(zip(names, values)), save_ts)

        return FunctionCall(_solve, params, name="fem_transient_solve")


def _block_time_grid(block):
    """The block's own integration grid ``t0 .. t1`` at its assembled step ``dt`` -- the
    default ``save_ts`` (the domain's ``time=(t0, t1, n_time)`` grid)."""
    import jax.numpy as jnp

    t0, t1, dt = float(block.t0), float(block.t1), float(block.dt)
    n_steps = max(1, round((t1 - t0) / dt))
    return jnp.linspace(t0, t1, n_steps + 1)


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

    s0 = jnp.asarray(block.state0).reshape(-1)
    dtype = s0.dtype
    grid_ts = jnp.asarray(_block_time_grid(block), dtype)
    dt = float(block.dt)

    # One scan step = one implicit advance of the block. `block.step` is the single definition of
    # that step (theta-method for a linear block, backward-Euler Newton for a nonlinear one); read
    # theta from the block so a linear step uses the assembled scheme. Operators are only applied as
    # matvecs inside block.step, so a BCOO operator stays sparse.
    if theta is None:  # jno.solve.theta(...) overrides the assembly's default (1 backward-Euler / ½ trapezoidal)
        theta = float(block.metadata.get("theta", 1.0)) if getattr(block, "metadata", None) else 1.0

    def step(w, t_next):
        wn = block.step(
            w, t_next - dt, dt, args=args, theta=theta, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve
        )
        return wn, wn

    _, ys = jax.lax.scan(step, s0, grid_ts[1:])

    traj = jnp.concatenate([s0[None, :], ys], axis=0)  # (n_grid, n_dofs) at grid_ts
    save_ts = jnp.asarray(save_ts, dtype)
    # sample at save_ts (identity when save_ts == the grid); decouples output from the step
    return jax.vmap(lambda col: jnp.interp(save_ts, grid_ts, col), in_axes=1, out_axes=1)(traj)
