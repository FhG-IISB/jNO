"""``jno.fem`` — assemble a traced weak form into feax FEM matrices/operators.

Author the physics as ordinary jNO residual expressions and hand the flat list
to :func:`fem`. Each residual is classified by **role** (does it contain the
test function?) and **region** (carried by the bound coordinates), then
assembled through feax. The returned :class:`FEM` exposes the assembled
artefacts — ``A``/``b`` for a linear problem, ``residual``/``jacobian`` for a
nonlinear one — plus the feax ``problem``, ``mesh`` and ``dofs``.

Classification rule
-------------------
* a residual that contains the **test function** is a weak term — integrated
  over its region (volume, or a boundary region for surface/Neumann/Robin
  terms);
* a residual with the **trial only** (no test function) on a **boundary**
  region is an essential (Dirichlet) condition ``u - g`` — its DOFs are pinned;
* a trial-only residual on a volume region is an error (a forgotten test
  function).

This module does **not** solve. Users drive their own solve (``jnp.linalg.solve``,
``scipy``); a ``.solve()`` layer over lineax/optimistix/diffrax is future work.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import jax.numpy as jnp

from .trace import FemLinearSystem, TestFunction, TrialFunction, Variable

__all__ = ["fem", "FEM"]


# ---------------------------------------------------------------------------
# expression helpers
# ---------------------------------------------------------------------------
def _is_view(obj: Any) -> bool:
    """A typed semantic view wraps its Placeholder in ``_expr``."""
    return hasattr(obj, "_expr")


def _bare(obj: Any):
    """Underlying Placeholder behind an optional typed view."""
    return obj.expr if _is_view(obj) else obj


def _walk(node: Any):
    """Yield every node in a Placeholder tree (deduplicated by id)."""
    from .utils.solver.solver_helper import iter_placeholder_children

    seen: set[int] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        for child in iter_placeholder_children(n):
            stack.append(child)


def _spatial_coord_vars(constraint: Any) -> List[Variable]:
    """Spatial coordinate Variables relevant to a constraint.

    Includes both the tree coordinates and the bound-view ``_coord_vars`` (which
    survive arithmetic), so a plain-value term like ``u(xl, yl) - 0`` still
    exposes its region.
    """
    out: dict[int, Variable] = {}
    cv = getattr(constraint, "_coord_vars", None)
    if cv:
        for v in cv.values():
            if isinstance(v, Variable) and getattr(v, "axis", None) != "temporal":
                out[id(v)] = v
    for n in _walk(_bare(constraint)):
        if isinstance(n, Variable) and getattr(n, "axis", None) != "temporal":
            out[id(n)] = n
    return list(out.values())


def _contains(constraint: Any, cls) -> bool:
    from .utils.solver.solver_helper import contains_node_type

    return contains_node_type(_bare(constraint), cls)


def _discover_domain(constraints: List[Any]):
    for c in constraints:
        for v in _spatial_coord_vars(c):
            d = getattr(v, "_domain", None)
            if d is not None:
                return d
    raise ValueError(
        "jno.fem: could not discover a domain from the constraints. "
        "Author the weak form with coordinates from domain.variable(...)."
    )


def _infer_vec(constraints: List[Any]) -> int:
    """Infer the vector size from the trial's ``value_shape`` across the constraints.

    Scalar → 1, ``value_shape=(2,)`` → 2, etc. (reuses ``_infer_trial_metadata``).
    """
    from .utils.solver.feax_utils import _infer_trial_metadata

    for c in constraints:
        meta = _infer_trial_metadata(_bare(c))
        if meta.get("has_trial"):
            return max(1, int(meta.get("vec", 1)))
    return 1


def _region_and_support(constraint: Any, domain: Any) -> Tuple[str, str]:
    """Return ``(support, region_id)`` for a constraint.

    ``support`` is ``"volume"`` or ``"boundary"``; ``region_id`` is ``"volume"``
    for volume terms and the boundary tag otherwise. Region comes from the tags
    of the constraint's coordinate Variables; a constraint must resolve to a
    single region.
    """
    tags = {v.tag for v in _spatial_coord_vars(constraint) if isinstance(v.tag, str) and not v.tag.startswith("__")}
    # The t=t0 slice is its own support; an IC residual lives here.
    if "initial" in tags:
        return "initial", "initial"
    boundary_tags = {t for t in tags if t in getattr(domain, "_boundary_regions", {})}
    interiorish = tags - boundary_tags

    if len(boundary_tags) > 1 or (boundary_tags and interiorish):
        raise ValueError(
            f"jno.fem: a residual spans multiple regions {sorted(tags)}; each residual must live on a single region."
        )
    if boundary_tags:
        return "boundary", next(iter(boundary_tags))
    return "volume", "volume"


def _retag_coords_for_quadrature(constraint: Any, support: str, region_id: str) -> None:
    """Point a weak term's coordinate Variables at the FEM quadrature pool.

    The feax kernels bind a coordinate Variable to the live quadrature points
    only when its tag is ``"fem_gauss"`` (volume) or ``"gauss_<tag>"`` (surface);
    Jacobians use the coordinate's ``dim`` (axis), so derivatives are unaffected.
    """
    target = "fem_gauss" if support == "volume" else f"gauss_{region_id}"
    for v in _spatial_coord_vars(constraint):
        if isinstance(v.tag, str) and v.tag != "fem_gauss" and not v.tag.startswith("gauss_"):
            v.tag = target


def _constant_of(node: Any) -> Optional[float]:
    """Best-effort scalar extraction from a constant/Literal node."""
    for attr in ("value", "val", "data", "constant"):
        if hasattr(node, attr):
            try:
                return float(getattr(node, attr))
            except (TypeError, ValueError):
                return None
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def _essential_value_node(bare: Any) -> Any:
    """Return the value side ``g`` of an essential residual ``u(region) - g``.

    Returns a Placeholder expression (or ``0.0`` for a bare ``u``); raises if the
    residual is not the affine ``u - g`` form. Used for both Dirichlet values and
    initial conditions.
    """
    op = getattr(bare, "op", None)
    if op == "-" and hasattr(bare, "left") and hasattr(bare, "right"):
        left, right = bare.left, bare.right
        left_has_trial = _contains(left, TrialFunction)
        right_has_trial = _contains(right, TrialFunction)
        node = right if left_has_trial and not right_has_trial else (left if right_has_trial else None)
        if node is not None:
            return node
    if isinstance(bare, TrialFunction):
        return 0.0
    raise ValueError(
        "jno.fem: could not read an essential condition from the residual. "
        "Write it as `u(region) - value`, e.g. `u(xl, yl) - 0.0` or `u(xl, yl) - jno.fn(g, [xl, yl])`."
    )


def _eval_value_node_at(value_node: Any, points: Any) -> Any:
    """Evaluate a coordinate value expression at ``points`` (1-D result).

    Reuses the existing :class:`~jno.trace_evaluator.TraceEvaluator` (the engine
    behind ``.eval``) — the coordinates are concrete (the mesh is built), so this
    is a single forward pass. No bespoke expression walker is introduced.
    """
    from .trace_evaluator import TraceEvaluator

    tags = {v.tag for v in _walk(value_node) if isinstance(v, Variable)}
    pts = jnp.atleast_2d(jnp.asarray(points))
    return jnp.reshape(TraceEvaluator({}).evaluate(value_node, context={t: pts for t in tags}), (-1,))


def _coord_value_fn(value_node: Any) -> Callable:
    """feax ``value(point)`` callable for a coordinate Dirichlet value — the same
    per-point hook ``domain.dirichlet("left", lambda p: p[1])`` uses."""

    def value_fn(p):
        p_arr = jnp.asarray(p)
        out = _eval_value_node_at(value_node, jnp.atleast_2d(p_arr))
        return out[0] if p_arr.ndim == 1 else out

    return value_fn


def _dirichlet_value(bare: Any) -> Any:
    """Constant or ``value(point)`` callable for an essential Dirichlet ``u - g``."""
    node = _essential_value_node(bare)
    const = _constant_of(node)
    return const if const is not None else _coord_value_fn(node)


def _initial_state(bare: Any, domain: Any) -> Any:
    """Initial nodal state vector from an IC residual ``u(initial) - u0``.

    ``u0`` is evaluated at the mesh nodes (concrete, since the mesh is built) via
    the existing evaluator — a single forward pass.
    """
    node = _essential_value_node(bare)
    n_nodes = int(jnp.asarray(domain.mesh.points).shape[0])
    const = _constant_of(node)
    if const is not None:
        return jnp.full((n_nodes,), float(const))
    pts = jnp.asarray(domain.mesh.points)[:, : domain.dimension]
    vals = _eval_value_node_at(node, pts)
    return jnp.broadcast_to(vals, (n_nodes,)) if vals.shape[0] == 1 else vals


# ---------------------------------------------------------------------------
# the FEM container
# ---------------------------------------------------------------------------
class FEM:
    """Assembled FEM artefacts produced by :func:`fem` (no solve).

    Attributes
    ----------
    domain, mesh, problem:
        the owning jNO domain, its meshio mesh, and the feax problem.
    dofs:
        total number of degrees of freedom.
    classification:
        human-readable summary of how each residual was bucketed.
    """

    def __init__(self, domain: Any, op: Any, classification: List[str], *, mode: str):
        # mode: "linear" | "nonlinear" | "transient"
        self.domain = domain
        self._op = op
        self._mode = mode
        self.classification = classification
        self.mesh = getattr(domain, "mesh", None)
        self.problem = getattr(domain, "_feax_problem", None)

        self._A = self._b = None
        if mode == "linear":
            # _assemble_fem_system_from_ir returns either a raw (A, b) tuple
            # (non-parametric) or a FemLinearSystem (runtime-parametric).
            if isinstance(op, FemLinearSystem):
                self._A, self._b = op.A, op.b
            else:
                self._A, self._b = op[0], op[1]

    @property
    def is_transient(self) -> bool:
        return self._mode == "transient"

    @property
    def is_linear(self) -> bool:
        if self._mode == "transient":
            return bool(self._op.is_linear())
        return self._mode == "linear"

    @property
    def operator(self) -> Any:
        """The raw assembled block — ``(A, b)`` / ``FemLinearSystem`` /
        ``FemResidualOperator`` (steady) or ``FeaxTimeBlock`` (transient)."""
        return self._op

    # -- steady linear --
    @property
    def A(self):
        if self._mode != "linear":
            raise AttributeError(f"FEM is {self._mode}; .A is only for a steady linear problem (see .operator / .M).")
        return self._A

    @property
    def b(self):
        if self._mode != "linear":
            raise AttributeError(f"FEM is {self._mode}; .b is only for a steady linear problem (see .operator / .M).")
        return self._b

    # -- steady nonlinear --
    @property
    def residual(self):
        if self._mode != "nonlinear":
            raise AttributeError(f"FEM is {self._mode}; .residual is only for a steady nonlinear problem (see .operator).")
        return self._op.residual

    @property
    def jacobian(self):
        if self._mode != "nonlinear":
            raise AttributeError(f"FEM is {self._mode}; .jacobian is only for a steady nonlinear problem (see .operator).")
        return self._op.jacobian

    # -- transient (semidiscrete: M u_dot + ... ; integration window from the domain) --
    @property
    def M(self):
        """Mass matrix of the semidiscrete transient system."""
        if self._mode != "transient":
            raise AttributeError("FEM is steady; .M (mass matrix) is only for a transient problem.")
        return self._op.M

    @property
    def state0(self):
        """Initial nodal state vector (from the `u(initial) - u0` residual, else zeros)."""
        if self._mode != "transient":
            raise AttributeError("FEM is steady; .state0 is only for a transient problem.")
        return self._op.state0

    @property
    def t0(self):
        return self._op.t0 if self._mode == "transient" else None

    @property
    def t1(self):
        return self._op.t1 if self._mode == "transient" else None

    @property
    def dt(self):
        return self._op.dt if self._mode == "transient" else None

    @property
    def dofs(self) -> Optional[int]:
        if self._mode == "linear" and self._b is not None:
            return int(jnp.asarray(self._b).reshape(-1).shape[0])
        if self._mode == "nonlinear":
            size = getattr(self._op, "size", None)
            if size is not None:
                return int(size)
        if self._mode == "transient":
            if getattr(self._op, "state0", None) is not None:
                return int(jnp.asarray(self._op.state0).reshape(-1).shape[0])
            if getattr(self._op, "M", None) is not None:
                return int(jnp.asarray(self._op.M).shape[0])
        prob = self.problem
        return int(prob.num_total_dofs_all_vars) if prob is not None else None

    def __repr__(self) -> str:
        kind = (
            self._mode if self._mode != "transient" else ("transient-linear" if self.is_linear else "transient-nonlinear")
        )
        return f"FEM({kind}, dofs={self.dofs}, terms={self.classification})"


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def fem(constraints: Any, *, quad_degree: int = 2, element_type: str = "TRI3", vec: Optional[int] = None) -> FEM:
    """Assemble a flat list of traced residuals into an :class:`FEM`.

    Parameters
    ----------
    constraints:
        A residual expression or list of them. Weak terms (containing the test
        function) and essential conditions (``u(region) - g``) are auto-classified.
    quad_degree, element_type:
        FEM discretisation options (forwarded to the quadrature setup).
    vec:
        Vector size of the unknown. ``None`` (default) infers it from the trial's
        ``value_shape`` (scalar → 1, ``(2,)`` → 2, …); pass an int to override.
    """
    from .utils.solver.fem_route import _assemble_fem_residual_from_ir, _assemble_fem_system_from_ir, neumann
    from .utils.solver.weak_form import (
        LoweredChannelTerm,
        LoweredWeakForm,
        _apply_sign,
        _contains_temporal_derivative,
        _infer_solver_target,
        _split_additive_terms,
    )

    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]
    if len(constraints) == 0:
        raise ValueError("jno.fem: no constraints provided.")

    domain = _discover_domain(constraints)
    if vec is None:
        vec = _infer_vec(constraints)

    volume_terms: List[Any] = []
    boundary_terms: dict[str, List[Any]] = {}
    dirichlet_values: dict[str, Any] = {}
    ic_residuals: List[Any] = []
    classification: List[str] = []

    for c in constraints:
        has_test = _contains(c, TestFunction)
        has_trial = _contains(c, TrialFunction)
        support, region = _region_and_support(c, domain)

        if support == "initial":
            # initial condition: a trial-only residual `u(initial) - u0` on the t0 slice
            if has_test:
                raise ValueError("jno.fem: an initial-condition residual on 'initial' must not contain the test function.")
            ic_residuals.append(_bare(c))
            classification.append("initial")
        elif has_test:
            _retag_coords_for_quadrature(c, support, region)
            bare = _bare(c)
            if support == "volume":
                volume_terms.append(bare)
                classification.append("volume")
            else:
                boundary_terms.setdefault(region, []).append(bare)
                classification.append(f"surface@{region}")
        elif has_trial:
            if support != "boundary":
                raise ValueError(
                    "jno.fem: a residual with the trial but no test function must live on a boundary "
                    "region (Dirichlet) or the 'initial' region (IC). Got a volume region — did you forget the test function?"
                )
            dirichlet_values[region] = _dirichlet_value(_bare(c))
            classification.append(f"dirichlet@{region}")
        else:
            raise ValueError("jno.fem: a residual contains neither the trial nor the test function.")

    # ---- quadrature + BC setup (reuse init_fem) ----
    bcs = [domain.dirichlet(tag, value) for tag, value in dirichlet_values.items()]
    if boundary_terms:
        bcs.append(neumann(list(boundary_terms.keys())))
    domain.init_fem(element_type=element_type, quad_degree=quad_degree, bcs=bcs, vec=vec, fem_solver=True)

    # ---- build IR with explicit regions, then assemble through feax ----
    # Split each weak constraint into additive sub-terms (one LoweredChannelTerm
    # each), matching lower_weak_form's granularity — required so the transient
    # route can separate the mass term (u_t * phi) from the spatial operator.
    terms: List[LoweredChannelTerm] = []

    def _emit_terms(bare: Any, support: str, region_id: str) -> None:
        for sign, sub in _split_additive_terms(domain, bare):
            terms.append(
                LoweredChannelTerm(
                    sign=sign,
                    support=support,
                    region_id=region_id,
                    channel="raw",
                    coeff=_apply_sign(domain, sign, sub),
                    variable_id=0,
                    value_shape=(),
                    original_expr=sub,
                )
            )

    for bare in volume_terms:
        _emit_terms(bare, "volume", "volume")
    for tag, exprs in boundary_terms.items():
        for bare in exprs:
            _emit_terms(bare, "boundary", tag)

    if not terms:
        raise ValueError("jno.fem: no weak-form (test-function) terms found to assemble.")

    ir = LoweredWeakForm(domain=domain, terms=terms)

    # ---- transient (a weak term carries a temporal derivative) vs steady ----
    weak_bares = volume_terms + [e for exprs in boundary_terms.values() for e in exprs]
    is_transient = any(_contains_temporal_derivative(b) for b in weak_bares)

    if is_transient:
        from .utils.solver.time_route import _assemble_feax_time_from_ir

        if len(ic_residuals) > 1:
            raise NotImplementedError("jno.fem: multiple initial conditions (multi-field) are not supported yet.")
        n_nodes = int(jnp.asarray(domain.mesh.points).shape[0])
        state0 = _initial_state(ic_residuals[0], domain) if ic_residuals else jnp.zeros((n_nodes,))
        # The integration window (t0, t1, dt) is read from jno.domain(time=...).
        block = _assemble_feax_time_from_ir(domain, ir, state0=state0)
        return FEM(domain=domain, op=block, classification=classification, mode="transient")

    if ic_residuals:
        raise ValueError("jno.fem: an initial condition was given but the weak form has no time derivative.")

    probe = ir.volume_expr if ir.volume_expr is not None else next(iter(ir.boundary_exprs.values()))
    target = _infer_solver_target(domain, probe)
    if target == "fem_system":
        op = _assemble_fem_system_from_ir(domain, ir)
        return FEM(domain=domain, op=op, classification=classification, mode="linear")

    op = _assemble_fem_residual_from_ir(domain, ir)
    return FEM(domain=domain, op=op, classification=classification, mode="nonlinear")
