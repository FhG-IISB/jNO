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

# Component names feax uses for per-component (roller/symmetry) Dirichlet specs.
_COMPONENT_NAMES = {0: "x", 1: "y", 2: "z"}


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

    def _add_coords(cv):
        # `_coord_vars` ({name: Variable}) is carried by bound views and, after a
        # `jno.np` reduction, by the resulting FunctionCall — so a bound boundary
        # term keeps its region even through `inner(...)` / arithmetic wrapping.
        if cv:
            for v in cv.values():
                if isinstance(v, Variable) and getattr(v, "axis", None) != "temporal":
                    out[id(v)] = v

    _add_coords(getattr(constraint, "_coord_vars", None))
    for n in _walk(_bare(constraint)):
        _add_coords(getattr(n, "_coord_vars", None))
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


# (dimension, polynomial order) -> simplex element type; 1D uses the native LINE2 path.
_ELEMENT_FOR = {(2, 1): "TRI3", (2, 2): "TRI6", (3, 1): "TET4", (3, 2): "TET10"}
_P2_ELEMENTS = {"TRI6", "TET10", "QUAD8"}


def _infer_order(constraints: List[Any]) -> int:
    """Max element polynomial order across the trial fields (P1=1, P2=2)."""
    orders = [int(getattr(n, "order", 1)) for c in constraints for n in _walk(_bare(c)) if isinstance(n, TrialFunction)]
    return max(orders) if orders else 1


def _element_for(dimension: int, order: int) -> str:
    """Simplex element type for a ``(dimension, order)`` pair (2D/3D; orders 1, 2)."""
    et = _ELEMENT_FOR.get((int(dimension), int(order)))
    if et is None:
        raise ValueError(
            f"jno.fem: no built-in element for dimension {dimension}, order {order} "
            "(supported: 2D/3D at order 1 or 2; pass element_type=... to override)."
        )
    return et


def _order_of_element(element_type: str) -> int:
    return 2 if element_type in _P2_ELEMENTS else 1


def _field_keys(constraints: List[Any]) -> List[Any]:
    """Ordered distinct trial ``field_key``s across constraints (first appearance).

    Each ``fem_symbols()`` call is one coupled field (its trial + test share a key).
    More than one key ⇒ a coupled / mixed multi-field problem."""
    keys: List[Any] = []
    seen: set = set()
    for c in constraints:
        for n in _walk(_bare(c)):
            if isinstance(n, TrialFunction):
                k = getattr(n, "field_key", n.op_id)
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
    return keys


def _field_key_of(constraint: Any) -> Any:
    """The trial ``field_key`` of an essential (Dirichlet) constraint."""
    for n in _walk(_bare(constraint)):
        if isinstance(n, TrialFunction):
            return getattr(n, "field_key", n.op_id)
    return None


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


def _component_index_of(node: Any) -> Optional[int]:
    """If ``node`` is a single component of the trial (``u[..., i]``), return ``i``.

    Reads the index recorded on the ``getitem`` node (``Placeholder.__getitem__``),
    so a per-component (roller) BC ``u(region)[i] - g`` is recoverable. Returns
    ``None`` for the bare trial (all components).
    """
    if getattr(node, "_name", None) == "getitem" and hasattr(node, "getitem_key"):
        args = getattr(node, "args", None) or []
        if len(args) == 1 and _contains(args[0], TrialFunction):
            ints = [k for k in node.getitem_key if isinstance(k, int)]
            if len(ints) == 1:
                return ints[0]
    return None


def _essential_spec(bare: Any) -> Tuple[Optional[int], Any]:
    """``(component_index_or_None, value_node)`` for an essential residual.

    Handles ``u(region) - g`` (component ``None`` = all components) and
    ``u(region)[i] - g`` (component ``i``). The unknown side must be the bare
    trial or a single component of it — a nonlinear/scaled form (e.g. ``u**2 - 1``)
    raises rather than being silently read as ``u - 1``.
    """
    op = getattr(bare, "op", None)
    if op == "-" and hasattr(bare, "left") and hasattr(bare, "right"):
        left, right = bare.left, bare.right
        left_has_trial = _contains(left, TrialFunction)
        right_has_trial = _contains(right, TrialFunction)
        if left_has_trial and not right_has_trial:
            trial_side, value_node = left, right
        elif right_has_trial and not left_has_trial:
            trial_side, value_node = right, left
        else:
            trial_side = value_node = None
        if value_node is not None:
            comp = _component_index_of(trial_side)
            if not (isinstance(trial_side, TrialFunction) or comp is not None):
                raise ValueError(
                    "jno.fem: an essential boundary/initial condition must be affine in the unknown — "
                    f"write `u(region) - value` or `u(region)[i] - value`, not a nonlinear/scaled trial "
                    f"expression. Got: {trial_side!r}."
                )
            return comp, value_node
    if isinstance(bare, TrialFunction):
        return None, 0.0
    raise ValueError(
        "jno.fem: could not read an essential condition from the residual. "
        "Write it as `u(region) - value` (e.g. `u(xl, yl) - 0.0`) or `u(xl, yl)[i] - value`."
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


def _value_from_node(node: Any) -> Any:
    """Constant or ``value(point)`` callable from an essential value node."""
    const = _constant_of(node)
    return const if const is not None else _coord_value_fn(node)


def _dirichlet_spec(bare: Any) -> Tuple[Optional[int], Any]:
    """``(component_index_or_None, value)`` for an essential Dirichlet residual.

    ``component`` is ``None`` for an all-component clamp (``u(region) - g``) or an
    integer for a per-component / roller BC (``u(region)[i] - g``).
    """
    comp, value_node = _essential_spec(bare)
    return comp, _value_from_node(value_node)


def _initial_state(bare: Any, domain: Any) -> Any:
    """Initial nodal state vector from an IC residual ``u(initial) - u0``.

    ``u0`` is evaluated at the mesh nodes (concrete, since the mesh is built) via
    the existing evaluator — a single forward pass.
    """
    _comp, node = _essential_spec(bare)
    n_nodes = int(jnp.asarray(domain.mesh.points).shape[0])
    const = _constant_of(node)
    if const is not None:
        return jnp.full((n_nodes,), float(const))
    pts = jnp.asarray(domain.mesh.points)[:, : domain.dimension]
    vals = _eval_value_node_at(node, pts)
    return jnp.broadcast_to(vals, (n_nodes,)) if vals.shape[0] == 1 else vals


def _multifield_initial_state(domain: Any, prob: Any, fields: List[Any], field_index: dict, ic_residuals: List[Any]) -> Any:
    """Block initial-state vector from per-field IC residuals ``u(initial) - u0``.

    Each field's IC is evaluated at **that field's** assembly-mesh nodes (a P2 field
    carries edge nodes, so ``domain.mesh`` is wrong for it) and written into its block
    ``offset[i]:offset[i+1]``. Fields without an IC default to zero. Mirrors the
    per-field Dirichlet bucketing used for the steady coupled path.
    """
    offsets = list(prob.offset) + [int(prob.num_total_dofs_all_vars)]
    state0 = jnp.zeros((int(prob.num_total_dofs_all_vars),))
    for ic in ic_residuals:
        fidx = field_index.get(_field_key_of(ic))
        if fidx is None:
            continue
        _comp, node = _essential_spec(_bare(ic))
        pts = jnp.asarray(prob.mesh[fidx].points)[:, : domain.dimension]
        n_i = int(pts.shape[0])
        const = _constant_of(node)
        if const is not None:
            vals = jnp.full((n_i,), float(const))
        else:
            v = jnp.asarray(_eval_value_node_at(node, pts))
            vals = jnp.broadcast_to(v, (n_i,)) if v.shape[0] == 1 else v
        block = jnp.asarray(vals).reshape(-1)
        if block.shape[0] != offsets[fidx + 1] - offsets[fidx]:
            raise NotImplementedError(
                "jno.fem: vector-field initial conditions in coupled transient are not supported yet "
                f"(field {fidx}: got {block.shape[0]} values for {offsets[fidx + 1] - offsets[fidx]} dofs)."
            )
        state0 = state0.at[offsets[fidx] : offsets[fidx + 1]].set(block)
    return state0


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

    @property
    def points(self):
        """Node coordinates the DOFs live on.

        For a higher-order (P2) element these are the assembly-mesh nodes
        (vertices + edge midpoints), which differ from the linear ``mesh`` the
        domain keeps — use these to interpret the solution vector."""
        prob = self.problem
        meshes = getattr(prob, "mesh", None) if prob is not None else None
        if meshes:
            return jnp.asarray(meshes[0].points)
        if self.mesh is not None:
            return jnp.asarray(self.mesh.points)[:, : self.domain.dimension]
        return None

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
def fem(constraints: Any, *, quad_degree: int = 2, element_type: Optional[str] = None, vec: Optional[int] = None) -> FEM:
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
    multifield = len(_field_keys(constraints)) > 1
    if vec is None and not multifield:
        vec = _infer_vec(constraints)  # single-field only (coupled fields carry per-field vec)

    volume_terms: List[Any] = []
    boundary_terms: dict[str, List[Any]] = {}
    dirichlet_values: dict[str, Any] = {}
    dirichlet_raw: List[Any] = []  # (field_key, region, comp, value) for the multi-field path
    ic_residuals: List[Any] = []
    classification: List[str] = []

    # Classify every constraint against its ORIGINAL coordinate tags before touching
    # any of them. `_retag_coords_for_quadrature` mutates the shared coordinate
    # Variables in place, so classifying up front prevents one constraint's retag from
    # leaking into another's region detection — e.g. two surface terms on the same
    # region (a Neumann load + a Robin term) that share the bound boundary coordinates.
    classified = [
        (c, _contains(c, TestFunction), _contains(c, TrialFunction), *_region_and_support(c, domain)) for c in constraints
    ]

    for c, has_test, has_trial, support, region in classified:
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
            comp, value = _dirichlet_spec(_bare(c))
            dirichlet_raw.append((_field_key_of(c), region, comp, value))
            if comp is None:  # all components: u(region) - g
                if isinstance(dirichlet_values.get(region), dict):
                    raise ValueError(f"jno.fem: region {region!r} mixes all-component and per-component Dirichlet.")
                dirichlet_values[region] = value
                classification.append(f"dirichlet@{region}")
            else:  # one component (roller/symmetry): u(region)[i] - g
                if comp not in _COMPONENT_NAMES:
                    raise ValueError(
                        f"jno.fem: Dirichlet component index {comp} out of range (vector components are 0..2)."
                    )
                current = dirichlet_values.get(region)
                if current is not None and not isinstance(current, dict):
                    raise ValueError(f"jno.fem: region {region!r} mixes all-component and per-component Dirichlet.")
                current = dict(current or {})
                current[_COMPONENT_NAMES[comp]] = value
                dirichlet_values[region] = current
                classification.append(f"dirichlet@{region}[{_COMPONENT_NAMES[comp]}]")
        else:
            raise ValueError("jno.fem: a residual contains neither the trial nor the test function.")

    # ---- 1D (segment): feax has no LINE2 element, so assemble natively ----
    # The native 1D assembler reuses the same integrand evaluator and returns the
    # same (op, mode) the FEM container expects; it needs none of init_fem's feax
    # scaffolding (coordinate vars resolve from the per-element quadrature points).
    if getattr(domain, "dimension", None) == 1:
        from .utils.solver.fem_1d import assemble_fem_1d

        op, mode = assemble_fem_1d(
            domain, volume_terms, boundary_terms, dirichlet_values, ic_residuals, vec=vec, quad_degree=quad_degree
        )
        return FEM(domain=domain, op=op, classification=classification, mode=mode)

    order = _infer_order(constraints)
    quad_degree = max(quad_degree, 2 * order)

    # ---- coupled / mixed multi-field -> block (multi-variable) assembly ----
    if multifield:
        return _assemble_multifield(
            domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, classification, quad_degree=quad_degree
        )

    # ---- single field: element defaults to the field order (P1->TRI3/TET4,
    # P2->TRI6/TET10); a higher-order field bumps the quadrature for exactness. ----
    if element_type is None:
        element_type = _element_for(domain.dimension, order)
    quad_degree = max(quad_degree, 2 * _order_of_element(element_type))

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


def _assemble_multifield(domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, classification, *, quad_degree):
    """Assemble a coupled (multi-field) steady weak form into a block ``FEM``.

    Builds the same IR as the single-field path, buckets Dirichlet per field, and
    hands off to ``_assemble_fem_system_from_ir`` — which routes through the
    multi-field branch of ``_build_feax_problem`` (multi-variable feax Problem +
    per-field kernel) and lets feax autodiff the block matrix."""
    from .utils.solver.feax_utils import _infer_fields
    from .utils.solver.fem_route import _assemble_fem_residual_from_ir, _assemble_fem_system_from_ir
    from .utils.solver.weak_form import (
        LoweredChannelTerm,
        LoweredWeakForm,
        _apply_sign,
        _contains_temporal_derivative,
        _is_obviously_nonlinear_in_unknown,
        _split_additive_terms,
    )

    weak_bares = volume_terms + [e for exprs in boundary_terms.values() for e in exprs]
    is_transient = bool(ic_residuals) or any(_contains_temporal_derivative(b) for b in weak_bares)

    domain._fem_quad_degree = quad_degree
    domain._variational_initialized = True

    # Same IR as the single-field path: one LoweredChannelTerm per additive sub-term.
    # Coupled surface (Neumann/Robin) terms are emitted on their boundary region_id;
    # _build_multifield_feax_problem groups them per tag into per-field surface kernels.
    # (Coords were retagged to gauss_<region> by the classifier's _retag_coords_for_quadrature.)
    terms: List[Any] = []

    def _emit(bare: Any, support: str, region_id: str) -> None:
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
        _emit(bare, "volume", "volume")
    for tag, exprs in boundary_terms.items():
        for bare in exprs:
            _emit(bare, "boundary", tag)
    if not terms:
        raise ValueError("jno.fem: no weak-form (test-function) terms found to assemble.")
    ir = LoweredWeakForm(domain=domain, terms=terms)

    # Field ordering is taken from ir.volume_expr — the same source _build_feax_problem
    # uses — so the Dirichlet field indices match the kernel/Problem field order.
    fields, field_index = _infer_fields(ir.volume_expr)
    by_field: dict[int, dict[str, Any]] = {}
    for field_key, region, comp, value in dirichlet_raw:
        fidx = field_index.get(field_key)
        if fidx is None:
            continue
        region_values = by_field.setdefault(fidx, {})
        if comp is None:
            region_values[region] = value
        else:
            current = region_values.get(region)
            current = dict(current) if isinstance(current, dict) else {}
            current[_COMPONENT_NAMES[comp]] = value
            region_values[region] = current
    domain._fem_dirichlet_by_field = by_field

    # Coupled transient (multi-field + time): block M + block spatial operator A.
    if is_transient:
        return _assemble_multifield_transient(domain, ir, fields, field_index, ic_residuals, classification)

    # Nonlinear coupled: feax autodiffs the block residual/Jacobian on the multi-field
    # problem (same _build_feax_problem path as linear), so the nonlinear route works
    # for coupled fields too. A nonlinear volume *or* surface term routes here (a linear
    # Robin term stays on the linear path, where its surface contribution lands in A).
    if any(_is_obviously_nonlinear_in_unknown(domain, b) for b in weak_bares):
        op = _assemble_fem_residual_from_ir(domain, ir)
        return FEM(domain=domain, op=op, classification=classification, mode="nonlinear")
    op = _assemble_fem_system_from_ir(domain, ir)
    return FEM(domain=domain, op=op, classification=classification, mode="linear")


def _assemble_multifield_transient(domain, ir, fields, field_index, ic_residuals, classification):
    """Assemble a coupled (multi-field) first-order transient weak form.

    Reuses the steady block assembly: the IR is split into a mass IR (additive terms
    carrying a temporal derivative, with ``u_t`` rewritten to ``u``) and a spatial
    operator IR. The constant block mass ``M`` is assembled from the mass IR; the
    spatial part is assembled from the operator IR — as a block matrix ``A`` when the
    weak form is linear, or as a block residual/Jacobian (feax autodiff) when it is
    nonlinear. A **shared** ``(fields, field_index)`` is threaded into every block
    assembly so the separately-built pieces line up (one field ordering, a block for
    every field). Per-field initial conditions form the block ``state0``; the user
    drives backward Euler — ``(M + dt·A) w = M w`` (linear) or a Newton solve of
    ``M (w-w_old)/dt + R(w) = 0`` (nonlinear).

    Scope: every field must carry a time derivative (algebraic/DAE fields — a zero
    mass block, e.g. transient Stokes pressure — are not handled yet)."""
    from .utils.solver.backend_blocks import FeaxTimeBlock
    from .utils.solver.feax_utils import _dense_array, _infer_fields, _lower_statefield_to_trial
    from .utils.solver.fem_route import _assemble_fem_residual_from_ir, _assemble_fem_system_from_ir
    from .utils.solver.time_route import _infer_time_window, _is_linear_first_order_ir, _split_first_order_linear_terms

    mass_ir, op_ir, src_ir = _split_first_order_linear_terms(ir)
    if len(src_ir.terms) > 0:
        raise NotImplementedError("jno.fem: standalone source/forcing terms in coupled transient are not supported yet.")
    mass_keys = {f["field_key"] for f in _infer_fields(_lower_statefield_to_trial(mass_ir.volume_expr, {}))[0]}
    if mass_keys != {f["field_key"] for f in fields}:
        raise NotImplementedError(
            "jno.fem: coupled transient requires every field to carry a time derivative (u_t * test); "
            "algebraic (DAE) fields are not supported yet."
        )

    # Shared field layout so the separately-assembled mass and operator blocks align.
    override = (fields, field_index)
    M = jnp.asarray(_dense_array(_assemble_fem_system_from_ir(domain, mass_ir, fields_override=override)[0]))
    t0, t1, dt = _infer_time_window(domain)
    common = dict(
        backend="feax_time",
        mode="implicit",
        time_order=1,
        spatial_kind="weak_form",
        ir=ir,
        t0=t0,
        t1=t1,
        dt=dt,
        feax_context=getattr(domain, "_feax_context", {}) or {},
    )

    if _is_linear_first_order_ir(ir):
        A_sys, bA = _assemble_fem_system_from_ir(domain, op_ir, fields_override=override)
        prob = domain._feax_problem  # op_ir block problem (same block layout as mass_ir via override)
        state0 = _multifield_initial_state(domain, prob, fields, field_index, ic_residuals)
        block = FeaxTimeBlock(
            M=M, A=jnp.asarray(_dense_array(A_sys)), affine_bias=jnp.asarray(bA).reshape(-1), state0=state0, **common
        )
        return FEM(domain=domain, op=block, classification=classification, mode="transient")

    # Nonlinear: M u_dot + R(u) = 0, with R/J the block spatial residual/Jacobian that
    # feax autodiffs on the multi-field problem (same path as steady nonlinear coupled).
    spatial = _assemble_fem_residual_from_ir(domain, op_ir, fields_override=override)
    prob = domain._feax_problem
    state0 = _multifield_initial_state(domain, prob, fields, field_index, ic_residuals)
    block = FeaxTimeBlock(
        mass=lambda t, args=None, _M=M: _M,
        residual=lambda u, t, args=None, _r=spatial.residual: _r(jnp.asarray(u)),
        jacobian=lambda u, t, args=None, _j=spatial.jacobian: _j(jnp.asarray(u)),
        state0=state0,
        **common,
    )
    return FEM(domain=domain, op=block, classification=classification, mode="transient")
