from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------
# Solver-facing output blocks
# ---------------------------------------------------------------------
@dataclass
class DiffraxBlock:
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


@dataclass
class FeaxPipelineBlock:
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
    Common semidiscrete transient block.

    This is the solver-agnostic output of weak.assemble(target="feax_time", ...).

    Linear payload:
        M u_dot + A u = c + f(t)

    Nonlinear payload:
        M(t) u_dot + R(u,t) = 0
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
    affine_bias: Any = None
    forcing_vector_fn: Optional[Callable] = None

    # optional mesh / hints
    feax_mesh: Any = None
    forcing_mode: str = "none"

    def is_linear(self) -> bool:
        return self.M is not None and self.A is not None

    def is_nonlinear(self) -> bool:
        return self.mass is not None and self.residual is not None

    # -----------------------------------------------------------------
    # Thin compatibility wrappers only
    # -----------------------------------------------------------------
    def as_diffrax(self, *, forcing_vector_fn: Optional[Callable] = None, args: Any = None) -> "DiffraxBlock":
        return make_diffrax_block(self, forcing_vector_fn=forcing_vector_fn, args=args)

    def as_feax_pipeline(
        self,
        *,
        scheme: Optional[str] = None,
        forcing_vector_fn: Optional[Callable] = None,
        args: Any = None,
        monitor_index: Optional[int] = None,
        newton_tol: float = 1e-8,
        newton_maxiter: int = 20,
        snapshot_times: Optional[list[float]] = None,
        newton_damping: float = 1.0,
        compile_step: bool = True,
    ) -> "FeaxPipelineBlock":
        return make_feax_pipeline(
            self,
            scheme=scheme,
            forcing_vector_fn=forcing_vector_fn,
            args=args,
            monitor_index=monitor_index,
            newton_tol=newton_tol,
            newton_maxiter=newton_maxiter,
            snapshot_times=snapshot_times,
            newton_damping=newton_damping,
            compile_step=compile_step,
        )


# ---------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------
def _require_first_order(block: FeaxTimeBlock):
    if block.time_order != 1:
        raise NotImplementedError(
            "Only first-order-in-time semidiscrete blocks are currently supported."
        )


def _select_scheme(block: FeaxTimeBlock, scheme: Optional[str]) -> str:
    if scheme is None:
        scheme = "backward_euler" if str(block.mode).lower() == "implicit" else "forward_euler"

    scheme = str(scheme).lower()
    if scheme not in {"backward_euler", "forward_euler"}:
        raise ValueError(
            f"Unsupported FEAX adapter scheme '{scheme}'. "
            "Supported: 'backward_euler', 'forward_euler'."
        )
    return scheme


# ---------------------------------------------------------------------
# Standalone Diffrax adapter
# ---------------------------------------------------------------------
def make_diffrax_block(
    block: FeaxTimeBlock,
    *,
    forcing_vector_fn: Optional[Callable] = None,
    args: Any = None,
) -> DiffraxBlock:
    _require_first_order(block)

    import diffrax

    # --------------------------------------------------
    # Linear path
    # --------------------------------------------------
    if block.is_linear():
        M = jnp.asarray(block.M)
        A = jnp.asarray(block.A)

        if block.affine_bias is None:
            c = jnp.zeros((M.shape[0],), dtype=M.dtype)
        else:
            c = jnp.asarray(block.affine_bias, dtype=M.dtype).reshape(-1)

        f_fn = forcing_vector_fn if forcing_vector_fn is not None else block.forcing_vector_fn

        def rhs(t, y, solver_args):
            ff = (
                jnp.zeros_like(c)
                if f_fn is None
                else jnp.asarray(
                    f_fn(t, args if solver_args is None else solver_args),
                    dtype=M.dtype,
                ).reshape(-1)
            )
            return jnp.linalg.solve(M, c + ff - A @ y)

        return DiffraxBlock(
            backend="diffrax",
            form="explicit_first_order",
            time_order=1,
            original_expr=block.ir,
            lowered_rhs=None,
            rewritten_system=None,
            state0=block.state0,
            initial_conditions=block.initial_conditions,
            t0=block.t0,
            t1=block.t1,
            dt0=block.dt,
            rhs=rhs,
            term=diffrax.ODETerm(rhs),
            args=args,
            mass=lambda t, a=None: M,
            state_meta={},
            metadata={
                **block.metadata,
                "converted_from": "feax_time",
                "conversion": "u_dot = solve(M, c + f(t) - A u)",
                "jax_runtime": True,
            },
        )

    # --------------------------------------------------
    # Nonlinear path
    # --------------------------------------------------
    if block.is_nonlinear():
        mass_fn = block.mass
        residual_fn = block.residual

        def rhs(t, y, solver_args):
            M_t = jnp.asarray(mass_fn(t, args if solver_args is None else solver_args))
            R_t = jnp.asarray(
                residual_fn(y, t, args if solver_args is None else solver_args)
            ).reshape(-1)
            return jnp.linalg.solve(M_t, -R_t)

        return DiffraxBlock(
            backend="diffrax",
            form="explicit_first_order_nonlinear",
            time_order=1,
            original_expr=block.ir,
            lowered_rhs=None,
            rewritten_system=None,
            state0=block.state0,
            initial_conditions=block.initial_conditions,
            t0=block.t0,
            t1=block.t1,
            dt0=block.dt,
            rhs=rhs,
            term=diffrax.ODETerm(rhs),
            args=args,
            mass=mass_fn,
            state_meta={},
            metadata={
                **block.metadata,
                "converted_from": "feax_time",
                "conversion": "u_dot = solve(M(t), -R(u,t))",
                "jax_runtime": True,
            },
        )

    raise ValueError(
        "make_diffrax_block(...) requires either a linear payload (M,A,...) "
        "or a nonlinear payload (mass,residual,...)."
    )


# ---------------------------------------------------------------------
# Standalone FEAX pipeline adapter
# ---------------------------------------------------------------------
def make_feax_pipeline(
    block: FeaxTimeBlock,
    *,
    scheme: Optional[str] = None,
    forcing_vector_fn: Optional[Callable] = None,
    args: Any = None,
    monitor_index: Optional[int] = None,
    newton_tol: float = 1e-8,
    newton_maxiter: int = 20,
    snapshot_times: Optional[list[float]] = None,
    newton_damping: float = 1.0,
    compile_step: bool = True,
) -> FeaxPipelineBlock:
    _require_first_order(block)

    if block.feax_mesh is None:
        raise ValueError(
            "make_feax_pipeline(...) requires feax_mesh on the semidiscrete block."
        )

    from feax.solvers.time_solver import TimePipeline

    scheme_use = _select_scheme(block, scheme)
    snapshot_times_use = [] if snapshot_times is None else [float(tt) for tt in snapshot_times]

    # --------------------------------------------------
    # Linear path
    # --------------------------------------------------
    if block.is_linear():
        M = jnp.asarray(block.M)
        A = jnp.asarray(block.A)
        y0 = jnp.asarray(block.state0).reshape(-1)

        if block.affine_bias is None:
            c = jnp.zeros((M.shape[0],), dtype=M.dtype)
        else:
            c = jnp.asarray(block.affine_bias, dtype=M.dtype).reshape(-1)

        f_fn = forcing_vector_fn if forcing_vector_fn is not None else block.forcing_vector_fn

        class _LinearSemidiscretePipeline(TimePipeline):
            def build(self, mesh):
                self.mesh = mesh
                self.saved_snapshots = {}
                self._saved_flags = {float(tt): False for tt in snapshot_times_use}

                def _step_impl(state, t, dt):
                    if scheme_use == "backward_euler":
                        t_eval = t + dt
                        ff = (
                            jnp.zeros_like(c)
                            if f_fn is None
                            else jnp.asarray(f_fn(t_eval, args), dtype=M.dtype).reshape(-1)
                        )
                        lhs = M + dt * A
                        rhs_vec = M @ state + dt * (c + ff)
                        return jnp.linalg.solve(lhs, rhs_vec)

                    t_eval = t
                    ff = (
                        jnp.zeros_like(c)
                        if f_fn is None
                        else jnp.asarray(f_fn(t_eval, args), dtype=M.dtype).reshape(-1)
                    )
                    return state + dt * jnp.linalg.solve(M, c + ff - A @ state)

                self._step_impl = jax.jit(_step_impl) if compile_step else _step_impl

            def initial_state(self):
                return y0

            def _maybe_store_snapshot(self, state, t):
                for ts in snapshot_times_use:
                    if (not self._saved_flags[ts]) and abs(float(t) - ts) < 5e-7:
                        self.saved_snapshots[ts] = np.asarray(state).copy()
                        self._saved_flags[ts] = True

            def step(self, state, t, dt):
                self._maybe_store_snapshot(state, t)
                u_new = self._step_impl(state, t, dt)
                self._maybe_store_snapshot(u_new, t + dt)
                return u_new

            def monitor(self, state, step, t):
                out = {"state_norm": float(jnp.linalg.norm(state))}
                if monitor_index is not None:
                    out["u_monitor"] = float(state[int(monitor_index)])
                return out

        return FeaxPipelineBlock(
            backend="feax_time",
            scheme=scheme_use,
            pipeline=_LinearSemidiscretePipeline(),
            mesh=block.feax_mesh,
            state0=block.state0,
            initial_conditions=block.initial_conditions,
            t0=block.t0,
            t1=block.t1,
            dt=block.dt,
            metadata={
                **block.metadata,
                "converted_from": "feax_time",
                "conversion": f"semidiscrete_{scheme_use}",
                "forcing_mode": block.forcing_mode,
                "snapshot_support": bool(len(snapshot_times_use) > 0),
                "compile_step": bool(compile_step),
            },
        )

    # --------------------------------------------------
    # Nonlinear path
    # --------------------------------------------------
    if block.is_nonlinear():
        mass_fn = block.mass
        residual_fn = block.residual
        jacobian_fn = block.jacobian
        y0 = jnp.asarray(block.state0).reshape(-1)

        runtime_info = dict(getattr(block, "nonlinear_runtime", {}) or {})
        mass_is_constant = bool(runtime_info.get("mass_is_constant", False))

        if scheme_use == "forward_euler":
            class _NonlinearForwardEulerPipeline(TimePipeline):
                def build(self, mesh):
                    self.mesh = mesh
                    self.saved_snapshots = {}
                    self._saved_flags = {float(tt): False for tt in snapshot_times_use}
                    self._M_const = jnp.asarray(mass_fn(block.t0, args)) if mass_is_constant else None

                    def _mass_eval(t_eval):
                        if self._M_const is not None:
                            return self._M_const
                        return jnp.asarray(mass_fn(t_eval, args))

                    def _step_impl(state, t, dt):
                        M_t = _mass_eval(t)
                        R_t = jnp.asarray(residual_fn(state, t, args)).reshape(-1)
                        return state - dt * jnp.linalg.solve(M_t, R_t)

                    self._step_impl = jax.jit(_step_impl) if compile_step else _step_impl

                def initial_state(self):
                    return y0

                def _maybe_store_snapshot(self, state, t):
                    for ts in snapshot_times_use:
                        if (not self._saved_flags[ts]) and abs(float(t) - ts) < 5e-7:
                            self.saved_snapshots[ts] = np.asarray(state).copy()
                            self._saved_flags[ts] = True

                def step(self, state, t, dt):
                    self._maybe_store_snapshot(state, t)
                    u_new = self._step_impl(state, t, dt)
                    self._maybe_store_snapshot(u_new, t + dt)
                    return u_new

                def monitor(self, state, step, t):
                    out = {"state_norm": float(jnp.linalg.norm(state))}
                    if monitor_index is not None:
                        out["u_monitor"] = float(state[int(monitor_index)])
                    return out

            return FeaxPipelineBlock(
                backend="feax_time",
                scheme=scheme_use,
                pipeline=_NonlinearForwardEulerPipeline(),
                mesh=block.feax_mesh,
                state0=block.state0,
                initial_conditions=block.initial_conditions,
                t0=block.t0,
                t1=block.t1,
                dt=block.dt,
                metadata={
                    **block.metadata,
                    "converted_from": "feax_time",
                    "conversion": "nonlinear_forward_euler",
                    "mass_is_constant": mass_is_constant,
                    "snapshot_support": bool(len(snapshot_times_use) > 0),
                    "compile_step": bool(compile_step),
                },
            )

        if jacobian_fn is None:
            raise ValueError(
                "Nonlinear backward Euler requires jacobian(u,t). "
                "Re-assemble with a residual+jacobian-capable nonlinear route."
            )

        class _NonlinearBackwardEulerPipeline(TimePipeline):
            def build(self, mesh):
                self.mesh = mesh
                self.saved_snapshots = {}
                self._saved_flags = {float(tt): False for tt in snapshot_times_use}

                self._mass_is_constant = mass_is_constant
                self._M_const = jnp.asarray(mass_fn(block.t0, args)) if self._mass_is_constant else None

                tol = jnp.asarray(float(newton_tol), dtype=y0.dtype)
                damping = jnp.asarray(float(newton_damping), dtype=y0.dtype)

                def _mass_eval(t_next):
                    if self._M_const is not None:
                        return self._M_const
                    return jnp.asarray(mass_fn(t_next, args))

                def _step_impl(state, t, dt):
                    t_next = t + dt
                    M_next = _mass_eval(t_next)

                    def cond_fun(carry):
                        u, k, du_norm = carry
                        return jnp.logical_and(k < int(newton_maxiter), du_norm > tol)

                    def body_fun(carry):
                        u, k, _ = carry
                        R = jnp.asarray(residual_fn(u, t_next, args)).reshape(-1)
                        J = jnp.asarray(jacobian_fn(u, t_next, args))
                        G = M_next @ ((u - state) / dt) + R
                        dG = M_next / dt + J
                        du = jnp.linalg.solve(dG, -G)
                        u_new = u + damping * du
                        du_norm = jnp.linalg.norm(du)
                        return (u_new, k + 1, du_norm)

                    init = (
                        state,
                        jnp.asarray(0, dtype=jnp.int32),
                        jnp.asarray(jnp.inf, dtype=y0.dtype),
                    )
                    u_final, k_final, du_norm_final = jax.lax.while_loop(cond_fun, body_fun, init)
                    converged = du_norm_final <= tol
                    return u_final, k_final, du_norm_final, converged

                self._step_impl = jax.jit(_step_impl) if compile_step else _step_impl
                self._last_newton_iters = 0
                self._last_newton_du_norm = np.inf
                self._last_newton_converged = False

            def initial_state(self):
                return y0

            def _maybe_store_snapshot(self, state, t):
                for ts in snapshot_times_use:
                    if (not self._saved_flags[ts]) and abs(float(t) - ts) < 5e-7:
                        self.saved_snapshots[ts] = np.asarray(state).copy()
                        self._saved_flags[ts] = True

            def step(self, state, t, dt):
                self._maybe_store_snapshot(state, t)
                u_new, k_final, du_norm_final, converged = self._step_impl(state, t, dt)
                self._last_newton_iters = int(k_final)
                self._last_newton_du_norm = float(du_norm_final)
                self._last_newton_converged = bool(converged)
                self._maybe_store_snapshot(u_new, t + dt)
                return u_new

            def monitor(self, state, step, t):
                out = {
                    "state_norm": float(jnp.linalg.norm(state)),
                    "newton_iters": int(self._last_newton_iters),
                    "newton_du_norm": float(self._last_newton_du_norm),
                    "newton_converged": int(self._last_newton_converged),
                }
                if monitor_index is not None:
                    out["u_monitor"] = float(state[int(monitor_index)])
                return out

        return FeaxPipelineBlock(
            backend="feax_time",
            scheme=scheme_use,
            pipeline=_NonlinearBackwardEulerPipeline(),
            mesh=block.feax_mesh,
            state0=block.state0,
            initial_conditions=block.initial_conditions,
            t0=block.t0,
            t1=block.t1,
            dt=block.dt,
            metadata={
                **block.metadata,
                "converted_from": "feax_time",
                "conversion": "nonlinear_backward_euler_compiled",
                "newton_tol": float(newton_tol),
                "newton_maxiter": int(newton_maxiter),
                "newton_damping": float(newton_damping),
                "mass_is_constant": mass_is_constant,
                "snapshot_support": bool(len(snapshot_times_use) > 0),
                "compile_step": bool(compile_step),
            },
        )

    raise ValueError(
        "make_feax_pipeline(...) requires either a linear payload (M,A,...) "
        "or a nonlinear payload (mass,residual,...)."
    )