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
    def as_diffrax(self, *, forcing_vector_fn=None, args=None):
        from .time_adapters import make_diffrax_block

        return make_diffrax_block(
            self,
            forcing_vector_fn=forcing_vector_fn,
            args=args,
        )


    def as_feax_pipeline(
        self,
        *,
        scheme=None,
        forcing_vector_fn=None,
        args=None,
        monitor_index=None,
        newton_tol=1e-8,
        newton_maxiter=20,
        snapshot_times=None,
        newton_damping=1.0,
        compile_step=True,
    ):
        from .time_adapters import make_feax_pipeline

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
