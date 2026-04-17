from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp


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


@dataclass
class FeaxTimeBlock:
    backend: str = "feax_time"
    mode: str = "implicit"
    time_order: int = 1
    spatial_kind: str = "weak_form"

    ir: Any = None

    mass_expr: Any = None
    residual_expr: Any = None
    boundary_exprs: Dict[str, Any] = field(default_factory=dict)

    rhs: Optional[Callable] = None
    jacobian: Optional[Callable] = None
    mass: Optional[Callable] = None
    residual: Optional[Callable] = None

    state0: Any = None
    initial_conditions: Any = None

    t0: float = 0.0
    t1: float = 1.0
    dt: Optional[float] = None

    feax_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- linear semidiscrete payload ---
    M: Any = None
    A: Any = None
    affine_bias: Any = None
    forcing_vector_fn: Optional[Callable] = None

    # --- adapter payload ---
    feax_mesh: Any = None
    forcing_mode: str = "none"

    def _resolve_forcing(self, forcing_vector_fn: Optional[Callable], args: Any):
        return forcing_vector_fn if forcing_vector_fn is not None else self.forcing_vector_fn

    def as_diffrax(self, *, forcing_vector_fn: Optional[Callable] = None, args: Any = None) -> "DiffraxBlock":
        """
        Convert a linear first-order semidiscrete FEM system

            M u_dot + A u = c + f(t)

        into a Diffrax ODETerm:

            u_dot = solve(M, c + f(t) - A u)
        """
        if self.time_order != 1:
            raise NotImplementedError(
                "FeaxTimeBlock.as_diffrax() currently supports only first-order-in-time systems."
            )

        if self.M is None or self.A is None:
            raise ValueError(
                "FeaxTimeBlock.as_diffrax() requires preassembled linear semidiscrete operators "
                "M and A. Re-assemble with target='feax_time' and linear=True."
            )

        import diffrax

        M = jnp.asarray(self.M)
        A = jnp.asarray(self.A)

        if self.affine_bias is None:
            c = jnp.zeros((M.shape[0],), dtype=M.dtype)
        else:
            c = jnp.asarray(self.affine_bias, dtype=M.dtype).reshape(-1)

        f_fn = self._resolve_forcing(forcing_vector_fn, args)

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
            original_expr=self.ir,
            lowered_rhs=None,
            rewritten_system=None,
            state0=self.state0,
            initial_conditions=self.initial_conditions,
            t0=self.t0,
            t1=self.t1,
            dt0=self.dt,
            rhs=rhs,
            term=diffrax.ODETerm(rhs),
            args=args,
            mass=lambda t, a: M,
            state_meta={},
            metadata={
                **self.metadata,
                "converted_from": "feax_time",
                "conversion": "u_dot = solve(M, c + f(t) - A u)",
                "jax_runtime": True,
            },
        )

    def as_feax_pipeline(
        self,
        *,
        scheme: Optional[str] = None,
        forcing_vector_fn: Optional[Callable] = None,
        args: Any = None,
        monitor_index: Optional[int] = None,
    ) -> "FeaxPipelineBlock":
        """
        Build a FEAX TimePipeline adapter from the already-assembled semidiscrete block.

        Supported schemes
        -----------------
        - "backward_euler"
        - "forward_euler"

        Default scheme:
        - if self.mode == "implicit"  -> backward_euler
        - else                        -> forward_euler
        """
        if self.time_order != 1:
            raise NotImplementedError(
                "FeaxTimeBlock.as_feax_pipeline() currently supports only first-order-in-time systems."
            )

        if self.M is None or self.A is None:
            raise ValueError(
                "FeaxTimeBlock.as_feax_pipeline() requires preassembled linear semidiscrete operators "
                "M and A. Re-assemble with target='feax_time' and linear=True."
            )

        if self.feax_mesh is None:
            raise ValueError(
                "FeaxTimeBlock.as_feax_pipeline() requires feax_mesh on the block. "
                "Populate it during feax_time assembly."
            )

        from feax.solvers.time_solver import TimePipeline

        M = jnp.asarray(self.M)
        A = jnp.asarray(self.A)
        y0 = jnp.asarray(self.state0).reshape(-1)

        if self.affine_bias is None:
            c = jnp.zeros((M.shape[0],), dtype=M.dtype)
        else:
            c = jnp.asarray(self.affine_bias, dtype=M.dtype).reshape(-1)

        f_fn = self._resolve_forcing(forcing_vector_fn, args)

        scheme_use = scheme
        if scheme_use is None:
            scheme_use = "backward_euler" if str(self.mode).lower() == "implicit" else "forward_euler"
        scheme_use = str(scheme_use).lower()

        if scheme_use not in {"backward_euler", "forward_euler"}:
            raise ValueError(
                f"Unsupported FEAX adapter scheme '{scheme_use}'. "
                "Supported: 'backward_euler', 'forward_euler'."
            )

        class _SemidiscretePipeline(TimePipeline):
            def build(self, mesh):
                self.mesh = mesh

            def initial_state(self):
                return y0

            def step(self, state, t, dt):
                if scheme_use == "backward_euler":
                    # M (u^{n+1} - u^n)/dt + A u^{n+1} = c + f(t_{n+1})
                    t_eval = t + dt
                    ff = (
                        jnp.zeros_like(c)
                        if f_fn is None
                        else jnp.asarray(f_fn(t_eval, args), dtype=M.dtype).reshape(-1)
                    )
                    lhs = M + dt * A
                    rhs = M @ state + dt * (c + ff)
                    return jnp.linalg.solve(lhs, rhs)

                # forward Euler
                # u^{n+1} = u^n + dt * M^{-1}(c + f(t_n) - A u^n)
                t_eval = t
                ff = (
                    jnp.zeros_like(c)
                    if f_fn is None
                    else jnp.asarray(f_fn(t_eval, args), dtype=M.dtype).reshape(-1)
                )
                return state + dt * jnp.linalg.solve(M, c + ff - A @ state)

            def monitor(self, state, step, t):
                out = {"state_norm": float(jnp.linalg.norm(state))}
                if monitor_index is not None:
                    out["u_monitor"] = float(state[int(monitor_index)])
                return out

        return FeaxPipelineBlock(
            backend="feax_time",
            scheme=scheme_use,
            pipeline=_SemidiscretePipeline(),
            mesh=self.feax_mesh,
            state0=self.state0,
            initial_conditions=self.initial_conditions,
            t0=self.t0,
            t1=self.t1,
            dt=self.dt,
            metadata={
                **self.metadata,
                "converted_from": "feax_time",
                "conversion": f"semidiscrete_{scheme_use}",
                "forcing_mode": self.forcing_mode,
            },
        )