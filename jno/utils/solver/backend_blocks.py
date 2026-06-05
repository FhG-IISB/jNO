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

    A `DiffraxBlock` is returned by strong-form time assembly routes, or by
    converting a semidiscrete `FeaxTimeBlock` through `FeaxTimeBlock.as_diffrax()`.

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


@dataclass
class FeaxPipelineBlock:
    """
    Solver-facing block wrapping a FEAX `TimePipeline`.

    A `FeaxPipelineBlock` is produced by converting a semidiscrete
    `FeaxTimeBlock` through `FeaxTimeBlock.as_feax_pipeline(...)`.

    It stores the FEAX time pipeline together with the FEAX mesh, initial state,
    time interval, time-step size, and conversion metadata.

    This object does not itself perform time integration. It is a small container
    around the FEAX pipeline and provides `make_time_config(...)` to construct a
    FEAX `TimeConfig` using the stored time settings.

    Important fields
    ----------------
    backend:
        Backend identifier. Usually `"fem_time"`.
    scheme:
        Time-integration scheme used by the generated pipeline, for example
        `"backward_euler"` or `"forward_euler"`.
    pipeline:
        FEAX `TimePipeline` instance.
    mesh:
        FEAX mesh used by the pipeline.
    state0:
        Initial state vector.
    initial_conditions:
        Raw initial-condition object, if supplied.
    t0, t1:
        Start and end time.
    dt:
        Time-step size.
    metadata:
        Conversion and diagnostic metadata.
    """

    backend: str = "feax_time"
    scheme: str = "backward_euler"

    pipeline: Any = None
    mesh: Any = None

    state0: Any = None
    initial_conditions: Any = None

    t0: float = 0.0
    t1: float = 1.0
    dt: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def make_time_config(
        self,
        *,
        dt: Optional[float] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        print_every: int = 1,
        save_every: int = 10,
        **kwargs,
    ):
        """
        Create a FEAX `TimeConfig` from this pipeline block.

        Parameters
        ----------
        dt:
            Optional override for the time-step size. If omitted, `self.dt`
            is used.
        t_start:
            Optional override for the start time. If omitted, `self.t0` is used.
        t_end:
            Optional override for the end time. If omitted, `self.t1` is used.
        print_every:
            FEAX monitor print interval.
        save_every:
            FEAX output/save interval.
        **kwargs:
            Additional keyword arguments forwarded to `feax.solvers.time_solver.TimeConfig`.

        Returns
        -------
        TimeConfig
            FEAX time-solver configuration object.

        Raises
        ------
        ValueError
            If no time-step size is available.
        """
        from feax.solvers.time_solver import TimeConfig

        dt_use = self.dt if dt is None else float(dt)
        if dt_use is None:
            raise ValueError("No dt available on FeaxPipelineBlock. Pass dt=... explicitly.")

        return TimeConfig(
            dt=dt_use,
            t_start=self.t0 if t_start is None else float(t_start),
            t_end=self.t1 if t_end is None else float(t_end),
            print_every=print_every,
            save_every=save_every,
            **kwargs,
        )


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
    does not perform time integration by itself. It can be converted either to
    a `DiffraxBlock` through `as_diffrax(...)` or to a FEAX time pipeline through
    `as_feax_pipeline(...)`.

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

    def as_diffrax(self, *, forcing_vector_fn=None, operator_fn=None, args=None):
        """
        Convert this semidiscrete FEAX-time block into a `DiffraxBlock`.

        Parameters
        ----------
        forcing_vector_fn:
            Optional override for the forcing callback in the linear case.
            If omitted, `self.forcing_vector_fn` is used.
        operator_fn:
            Optional override for the runtime linear operator callback. The
            callback signature is ``operator_fn(t, args) -> matrix``. If
            omitted, ``self.operator_fn`` is used, falling back to ``self.A``.
        args:
            Optional runtime arguments passed to the generated Diffrax RHS.

        Returns
        -------
        DiffraxBlock
            A Diffrax-compatible block containing `rhs`, `term`, `state0`,
            time interval information, and conversion metadata.

        Notes
        -----
        Linear conversion uses:

            u_dot = solve(M, affine_bias + f(t, args) - A(t, args) u)

        Nonlinear conversion uses:

            u_dot = solve(M(t), -R(u, t))
        """
        from .time_adapters import make_diffrax_block

        return make_diffrax_block(
            self,
            forcing_vector_fn=forcing_vector_fn,
            operator_fn=operator_fn,
            args=args,
        )

    def as_feax_pipeline(
        self,
        *,
        scheme=None,
        forcing_vector_fn=None,
        operator_fn=None,
        args=None,
        monitor_index=None,
        newton_tol=1e-8,
        newton_maxiter=20,
        snapshot_times=None,
        newton_damping=1.0,
        compile_step=True,
    ):
        """
        Convert this semidiscrete FEAX-time block into a FEAX pipeline block.

        Parameters
        ----------
        scheme:
            Optional time-integration scheme. Supported adapter schemes are
            `"backward_euler"` and `"forward_euler"`. If omitted, the scheme is
            selected from `self.mode`.
        forcing_vector_fn:
            Optional override for the forcing callback in the linear case.
        operator_fn:
            Optional override for the runtime linear operator callback. The
            callback signature is ``operator_fn(t, args) -> matrix``.
        args:
            Optional runtime arguments passed to generated step functions.
        monitor_index:
            Optional state index to report as `u_monitor` in FEAX monitors.
        newton_tol:
            Newton convergence tolerance for nonlinear backward Euler.
        newton_maxiter:
            Maximum Newton iterations for nonlinear backward Euler.
        snapshot_times:
            Optional list of times at which snapshots should be stored by the
            generated FEAX pipeline.
        newton_damping:
            Damping factor applied to Newton updates.
        compile_step:
            If True, JIT-compile the generated per-step function.

        Returns
        -------
        FeaxPipelineBlock
            A block containing a FEAX `TimePipeline`, FEAX mesh, initial state,
            time interval information, and conversion metadata.

        Notes
        -----
        Linear backward Euler solves:

            (M + dt A(t_next, args)) u_next = M u + dt(c + f(t_next, args))

        Nonlinear backward Euler solves Newton iterations for:

            M(t_next) (u_next - u) / dt + R(u_next, t_next) = 0
        """
        from .time_adapters import make_feax_pipeline

        return make_feax_pipeline(
            self,
            scheme=scheme,
            forcing_vector_fn=forcing_vector_fn,
            operator_fn=operator_fn,
            args=args,
            monitor_index=monitor_index,
            newton_tol=newton_tol,
            newton_maxiter=newton_maxiter,
            snapshot_times=snapshot_times,
            newton_damping=newton_damping,
            compile_step=compile_step,
        )
