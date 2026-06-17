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


def _region_and_support(constraint: Any, domain: Any) -> Tuple[str, str]:
    """Return ``(support, region_id)`` for a constraint.

    ``support`` is ``"volume"`` or ``"boundary"``; ``region_id`` is ``"volume"``
    for volume terms and the boundary tag otherwise. Region comes from the tags
    of the constraint's coordinate Variables; a constraint must resolve to a
    single region.
    """
    tags = {v.tag for v in _spatial_coord_vars(constraint) if isinstance(v.tag, str) and not v.tag.startswith("__")}
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


def _coord_value_fn(value_node: Any) -> Callable:
    """Wrap a coordinate value expression as feax's ``value(point)`` callable.

    The boundary coordinates are concrete (the mesh is built), so the value is
    obtained with a single evaluation through the existing
    :class:`~jno.trace_evaluator.TraceEvaluator` at the boundary node(s) feax
    supplies — the same per-point value hook ``domain.dirichlet("left", lambda
    p: p[1])`` uses. No bespoke expression walker is introduced.
    """
    from .trace_evaluator import TraceEvaluator

    tags = {v.tag for v in _walk(value_node) if isinstance(v, Variable)}
    evaluator = TraceEvaluator({})

    def value_fn(p):
        p_arr = jnp.asarray(p)
        pts = jnp.atleast_2d(p_arr)
        out = jnp.reshape(evaluator.evaluate(value_node, context={t: pts for t in tags}), (-1,))
        return out[0] if p_arr.ndim == 1 else out

    return value_fn


def _extract_dirichlet_value(bare: Any) -> Any:
    """Extract the prescribed value from an essential residual ``u(region) - g``.

    Returns a constant for ``u - c``, or a ``value(point)`` callable (backed by
    :class:`TraceEvaluator`) for a coordinate expression ``u - g(x, y)`` such as
    ``u(xl, yl) - jno.fn(func, [xl, yl])`` or ``u(xl, yl) - yl``.
    """
    op = getattr(bare, "op", None)
    if op == "-" and hasattr(bare, "left") and hasattr(bare, "right"):
        left, right = bare.left, bare.right
        left_has_trial = _contains(left, TrialFunction)
        right_has_trial = _contains(right, TrialFunction)
        value_node = right if left_has_trial and not right_has_trial else (left if right_has_trial else None)
        if value_node is not None:
            const = _constant_of(value_node)
            return const if const is not None else _coord_value_fn(value_node)
    if isinstance(bare, TrialFunction):
        return 0.0
    raise ValueError(
        "jno.fem: could not read an essential boundary condition from the residual. "
        "Write it as `u(region) - value`, e.g. `u(xl, yl) - 0.0` or `u(xl, yl) - jno.fn(g, [xl, yl])`."
    )


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

    def __init__(self, domain: Any, op: Any, classification: List[str], *, linear: bool):
        self.domain = domain
        self._op = op
        self._linear = linear
        self.classification = classification
        self.mesh = getattr(domain, "mesh", None)
        self.problem = getattr(domain, "_feax_problem", None)

        self._A = self._b = None
        if linear:
            # _assemble_fem_system_from_ir returns either a raw (A, b) tuple
            # (non-parametric) or a FemLinearSystem (runtime-parametric).
            if isinstance(op, FemLinearSystem):
                self._A, self._b = op.A, op.b
            else:
                self._A, self._b = op[0], op[1]

    @property
    def is_linear(self) -> bool:
        return self._linear

    @property
    def operator(self) -> Any:
        """The raw assembled block — (A, b) / FemLinearSystem / FemResidualOperator."""
        return self._op

    @property
    def A(self):
        if not self._linear:
            raise AttributeError("FEM problem is nonlinear; use .residual / .jacobian, not .A.")
        return self._A

    @property
    def b(self):
        if not self._linear:
            raise AttributeError("FEM problem is nonlinear; use .residual / .jacobian, not .b.")
        return self._b

    @property
    def residual(self):
        if self._linear:
            raise AttributeError("FEM problem is linear; use .A / .b, not .residual.")
        return self._op.residual

    @property
    def jacobian(self):
        if self._linear:
            raise AttributeError("FEM problem is linear; use .A / .b, not .jacobian.")
        return self._op.jacobian

    @property
    def dofs(self) -> Optional[int]:
        if not self._linear:
            size = getattr(self._op, "size", None)
            if size is not None:
                return int(size)
        elif self._b is not None:
            return int(jnp.asarray(self._b).reshape(-1).shape[0])
        prob = self.problem
        return int(prob.num_total_dofs_all_vars) if prob is not None else None

    def __repr__(self) -> str:
        kind = "linear" if self.is_linear else "nonlinear"
        return f"FEM({kind}, dofs={self.dofs}, terms={self.classification})"


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def fem(constraints: Any, *, quad_degree: int = 2, element_type: str = "TRI3", vec: int = 1) -> FEM:
    """Assemble a flat list of traced residuals into an :class:`FEM`.

    Parameters
    ----------
    constraints:
        A residual expression or list of them. Weak terms (containing the test
        function) and essential conditions (``u(region) - g``) are auto-classified.
    quad_degree, element_type, vec:
        FEM discretisation options (forwarded to the quadrature setup).
    """
    from .utils.solver.fem_route import _assemble_fem_residual_from_ir, _assemble_fem_system_from_ir, neumann
    from .utils.solver.weak_form import LoweredChannelTerm, LoweredWeakForm, _infer_solver_target

    if not isinstance(constraints, (list, tuple)):
        constraints = [constraints]
    if len(constraints) == 0:
        raise ValueError("jno.fem: no constraints provided.")

    domain = _discover_domain(constraints)

    volume_terms: List[Any] = []
    boundary_terms: dict[str, List[Any]] = {}
    dirichlet_values: dict[str, Any] = {}
    classification: List[str] = []

    for c in constraints:
        has_test = _contains(c, TestFunction)
        has_trial = _contains(c, TrialFunction)
        support, region = _region_and_support(c, domain)

        if has_test:
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
                    "jno.fem: a residual with the trial but no test function must live on a "
                    "boundary region (Dirichlet). Got a volume region — did you forget the test function?"
                )
            dirichlet_values[region] = _extract_dirichlet_value(_bare(c))
            classification.append(f"dirichlet@{region}")
        else:
            raise ValueError("jno.fem: a residual contains neither the trial nor the test function.")

    # ---- quadrature + BC setup (reuse init_fem) ----
    bcs = [domain.dirichlet(tag, value) for tag, value in dirichlet_values.items()]
    if boundary_terms:
        bcs.append(neumann(list(boundary_terms.keys())))
    domain.init_fem(element_type=element_type, quad_degree=quad_degree, bcs=bcs, vec=vec, fem_solver=True)

    # ---- build IR with explicit regions, then assemble through feax ----
    terms: List[LoweredChannelTerm] = []
    for bare in volume_terms:
        terms.append(
            LoweredChannelTerm(
                sign=1.0,
                support="volume",
                region_id="volume",
                channel="raw",
                coeff=bare,
                variable_id=0,
                value_shape=(),
                original_expr=bare,
            )
        )
    for tag, exprs in boundary_terms.items():
        for bare in exprs:
            terms.append(
                LoweredChannelTerm(
                    sign=1.0,
                    support="boundary",
                    region_id=tag,
                    channel="raw",
                    coeff=bare,
                    variable_id=0,
                    value_shape=(),
                    original_expr=bare,
                )
            )

    if not terms:
        raise ValueError("jno.fem: no weak-form (test-function) terms found to assemble.")

    ir = LoweredWeakForm(domain=domain, terms=terms)

    probe = ir.volume_expr if ir.volume_expr is not None else next(iter(ir.boundary_exprs.values()))
    target = _infer_solver_target(domain, probe)
    if target == "fem_system":
        op = _assemble_fem_system_from_ir(domain, ir)
        return FEM(domain=domain, op=op, classification=classification, linear=True)

    op = _assemble_fem_residual_from_ir(domain, ir)
    return FEM(domain=domain, op=op, classification=classification, linear=False)
