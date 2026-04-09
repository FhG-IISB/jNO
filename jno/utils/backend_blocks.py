from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class DiffraxBlock:
    """Backend-neutral IVP block for Diffrax-style time solvers.

    Notes
    -----
    This object is intentionally solver-facing but still generic enough to be
    adapted to other time integrators later.  Phase 1 focuses on carrying the
    canonical pieces needed for a first-order IVP solve, even when the original
    symbolic problem was higher-order in time and requires an internal rewrite.
    """

    backend: str = "diffrax"
    form: str = "explicit_first_order"
    time_order: int = 1
    original_expr: Any = None
    lowered_rhs: Any = None
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
    """Backend-neutral transient FEM block for FEAX-style time solvers.

    Phase 1 returns the weak-form IR and enough metadata for external FEAX or
    custom JAX-native solvers to choose an implicit or explicit time loop.
    """

    backend: str = "feax"
    mode: str = "implicit"
    time_order: int = 1
    spatial_kind: str = "weak_form"
    ir: Any = None
    residual_expr: Any = None
    boundary_exprs: Any = None
    mass_expr: Any = None
    rhs: Optional[Callable] = None
    jacobian: Optional[Callable] = None
    state0: Any = None
    initial_conditions: Any = None
    t0: float = 0.0
    t1: float = 1.0
    dt: Optional[float] = None
    feax_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
