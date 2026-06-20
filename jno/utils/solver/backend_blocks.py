from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# ---------------------------------------------------------------------
# Solver-facing output blocks
# ---------------------------------------------------------------------


@dataclass
class DiffraxBlock:
    """
    Solver-facing block for Diffrax-based time integration.

    A `DiffraxBlock` is returned by strong-form time assembly routes.

    It stores the complete information needed to call Diffrax externally:
    the initial state, time interval, step-size hint, right-hand side function,
    optional mass operator, and metadata describing how the block was produced.

    Typical equation represented
    ----------------------------
    First-order explicit system:

        y_dot = rhs(t, y, args)

    For converted FEAX-time blocks, the RHS is usually created from either:

        M u_dot + A u = c + f(t)

    or:

        M(t) u_dot + R(u, t) = 0

    Important fields
    ----------------
    backend:
        Backend identifier. Usually `"diffrax"`.
    form:
        Description of the lowered system form, for example
        `"explicit_first_order"` or `"explicit_first_order_nonlinear"`.
    time_order:
        Original temporal order of the symbolic problem.
    original_expr:
        Original symbolic expression or weak-form IR used to build the block.
    lowered_rhs:
        Optional lowered symbolic RHS expression, when available.
    rewritten_system:
        Optional metadata for rewritten systems, such as second-order to
        first-order reductions.
    state0:
        Initial solver state.
    initial_conditions:
        Raw user-provided initial-condition object, if supplied.
    t0, t1:
        Start and end time.
    dt0:
        Initial time-step hint for Diffrax.
    rhs:
        Callable with signature `rhs(t, y, args)`.
    term:
        Diffrax term object, usually `diffrax.ODETerm(rhs)`.
    args:
        Optional static/runtime arguments passed to the RHS.
    mass:
        Optional mass operator callable.
    state_meta:
        Metadata about the state layout.
    metadata:
        Diagnostic and lowering metadata.
    """

    backend: str = "diffrax"
    form: str = "explicit_first_order"
    time_order: int = 1

    original_expr: Any = None
    lowered_rhs: Any = None
    rewritten_system: Any = None

    state0: Any = None
    initial_conditions: Any = None

    t0: float = 0.0
    t1: float = 1.0
    dt0: Optional[float] = None

    rhs: Optional[Callable] = None
    term: Any = None
    args: Any = None
    mass: Optional[Callable] = None

    state_meta: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Periodic prolongation matrix P (n_full x n_red); None when absent.
    prolongation: Any = None

    def prolong(self, reduced):
        """Map reduced periodic DOFs back to the full nodal layout."""
        if self.prolongation is None:
            return reduced
        from .feax_utils import prolong as _prolong

        return _prolong(self.prolongation, reduced)


# ---------------------------------------------------------------------
# Solver-agnostic semidiscrete block returned by weak.assemble(...)
# ---------------------------------------------------------------------


@dataclass
class FeaxTimeBlock:
    """
    Solver-agnostic semidiscrete transient block.

    A `FeaxTimeBlock` is returned by:

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
        Optional RHS callable. Usually unused for FEAX-time weak-form blocks.
    jacobian:
        Nonlinear residual Jacobian callable `jacobian(u, t, args)`.
    mass:
        Mass operator callable `mass(t, args)`.
    residual:
        Nonlinear residual callable `residual(u, t, args)`.
    nonlinear_runtime:
        Runtime diagnostics for nonlinear FEAX-time assembly.
    state0:
        Initial state vector.
    initial_conditions:
        Raw initial-condition object, if supplied.
    t0, t1:
        Start and end time.
    dt:
        Time-step size or time-step hint.
    feax_context:
        FEAX/FEM context copied from the domain.
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
    feax_mesh:
        FEAX mesh used by FEAX pipeline conversion.
    forcing_mode:
        Text label describing how forcing is represented, for example
        `"none"`, `"weak_auto"`, `"user_callback"`, or `"embedded_residual"`.
    """

    backend: str = "feax_time"
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

    feax_context: Dict[str, Any] = field(default_factory=dict)
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

    # optional mesh / hints
    feax_mesh: Any = None
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
        """Map reduced periodic DOFs back to the full nodal layout."""
        if self.prolongation is None:
            return reduced
        from .feax_utils import prolong as _prolong

        return _prolong(self.prolongation, reduced)

    def is_nonlinear(self) -> bool:
        """
        Return True if this block contains a nonlinear semidiscrete payload.

        A block is considered nonlinear when both `mass` and `residual`
        callables are populated. The represented system is:

            mass(t) u_dot + residual(u, t) = 0
        """
        return self.mass is not None and self.residual is not None

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
        ``lax.scan`` over the block's own assembled ``dt``. To bring your own (e.g. diffrax),
        build it from the block's flat pieces -- ``block.M``, ``block.A`` (or
        ``block.operator_fn(t, args)``) and ``block.state0`` -- and form ``u_dot = M^-1(c - A u)``.
        Note a Dirichlet problem zeroes M's Dirichlet rows (a DAE), so the implicit
        ``(M + dt A)`` default is preferred there; an explicit field must hold those rows.

        Enable x64 (``jax_enable_x64``); the feax assembly is float64.
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


def _default_transient_integrate(block, args, save_ts):
    """Default transient integrator: backward Euler at the block's *own* assembled step ``dt``,
    advanced with ``jax.lax.scan`` (reverse-mode differentiable) and sampled at ``save_ts`` by
    linear interpolation, so the integration step is always the assembled ``dt`` and never an
    accident of how the output is sampled.

    * **Linear** block -- the implicit scheme the transient assembly is built for::

          (M + dt A(t_next, args)) u_next = M u + dt (c + f(t_next, args))

    * **Nonlinear** block (residual route, ``M(t) u_dot = -R(u, t, args)``) -- backward Euler
      solves, per step, ``G(u_next) = M (u_next - u)/dt + R(u_next, t_next, args) = 0`` with an
      ``optimistix`` Newton ``root_find`` (implicit-diff, so the gradient reaches ``args``
      without unrolling Newton -- the same library the steady nonlinear ``.solve`` uses).

    This is a *default*: pass any ``solve_fn(block, args, save_ts) -> ys`` to
    :meth:`FeaxTimeBlock.solve` (a hand-rolled stepper, or diffrax built from the block's
    ``M`` / ``A`` / ``state0``) to use a different integrator.
    """
    import jax
    import jax.numpy as jnp

    s0 = jnp.asarray(block.state0).reshape(-1)
    dtype = s0.dtype
    grid_ts = jnp.asarray(_block_time_grid(block), dtype)
    dt = float(block.dt)

    if block.is_nonlinear():
        import optimistix as optx

        mass_fn, residual_fn = block.mass, block.residual

        def step(w, t_next):
            M_t = jnp.asarray(mass_fn(t_next, args), dtype)

            def _resid(wn, _):  # backward-Euler residual G(wn) = 0
                return (M_t @ (wn - w)) / dt + jnp.asarray(residual_fn(wn, t_next, args), dtype).reshape(-1)

            sol = optx.root_find(_resid, optx.Newton(rtol=1e-8, atol=1e-8), w, max_steps=64, throw=False)
            return sol.value, sol.value

        _, ys = jax.lax.scan(step, s0, grid_ts[1:])
    else:
        M = jnp.asarray(block.M, dtype)
        n = M.shape[0]
        c = jnp.zeros((n,), dtype) if block.affine_bias is None else jnp.asarray(block.affine_bias, dtype).reshape(-1)

        def step(w, t_next):
            A = jnp.asarray(block.operator_fn(t_next, args) if block.operator_fn is not None else block.A, dtype)
            rhs = M @ w + dt * c
            if block.forcing_vector_fn is not None:
                rhs = rhs + dt * jnp.asarray(block.forcing_vector_fn(t_next, args), dtype).reshape(-1)
            w_next = jnp.linalg.solve(M + dt * A, rhs)
            return w_next, w_next

        _, ys = jax.lax.scan(step, s0, grid_ts[1:])

    traj = jnp.concatenate([s0[None, :], ys], axis=0)  # (n_grid, n_dofs) at grid_ts
    save_ts = jnp.asarray(save_ts, dtype)
    # sample at save_ts (identity when save_ts == the grid); decouples output from the step
    return jax.vmap(lambda col: jnp.interp(save_ts, grid_ts, col), in_axes=1, out_axes=1)(traj)
