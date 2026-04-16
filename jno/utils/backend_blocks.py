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

    # --- new linear semidiscrete JAX payload ---
    M: Any = None
    A: Any = None
    affine_bias: Any = None
    forcing_vector_fn: Optional[Callable] = None

    def as_diffrax(self, *, forcing_vector_fn: Optional[Callable] = None, args: Any = None) -> "DiffraxBlock":
        """
        Convert a linear first-order semidiscrete FEM system

            M u_dot + A u = c + f(t)

        into a Diffrax ODETerm:

            u_dot = solve(M, c + f(t) - A u)

        Requirements
        ------------
        - self.M and self.A must be dense JAX arrays
        - forcing_vector_fn(t, args) must be JAX-compatible if provided
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

        f_fn = forcing_vector_fn if forcing_vector_fn is not None else self.forcing_vector_fn

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