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

import jax
import jax.numpy as jnp
import numpy as np

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


def _normalize_quad_tag(tag: Any, boundary_regions: Any) -> Any:
    """Undo a quadrature retag this module produced, so region detection is robust to a
    coordinate Variable that was already retagged by an earlier (or interleaved) ``jno.fem``
    call (``_retag_coords_for_quadrature`` mutates ``Variable.tag`` in place; reusing a stored
    coord var across two ``fem()`` calls otherwise leaves its tag as ``"gauss_<region>"`` and
    silently misclassifies the term as volume). The retag is total + reserved
    (``boundary -> "gauss_{region}"``), so stripping ``gauss_`` recovers the region exactly.

    Defensive: only strip ``gauss_<r>`` when ``<r>`` is an actual boundary region, so a
    user region with an unusual name is never silently rewritten -- we only ever undo a tag
    we ourselves produced. ``"fem_gauss"`` (volume) is left as-is and falls through to volume.
    """
    if isinstance(tag, str) and tag.startswith("gauss_"):
        region = tag[len("gauss_") :]
        if region in (boundary_regions or {}):
            return region
    return tag


def _region_and_support(constraint: Any, domain: Any) -> Tuple[str, str]:
    """Return ``(support, region_id)`` for a constraint.

    ``support`` is ``"volume"`` or ``"boundary"``; ``region_id`` is ``"volume"``
    for volume terms and the boundary tag otherwise. Region comes from the tags
    of the constraint's coordinate Variables; a constraint must resolve to a
    single region.
    """
    _bregions = getattr(domain, "_boundary_regions", {})
    tags = {
        _normalize_quad_tag(v.tag, _bregions)
        for v in _spatial_coord_vars(constraint)
        if isinstance(v.tag, str) and not v.tag.startswith("__")
    }
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


def _is_complex_form(domain: Any, ir: Any) -> bool:
    """True if any lowered term's expression contains a complex constant — the signal to route a
    (steady, linear) weak form through the real-equivalent complex solver instead of feax's real
    assembly. We walk the tree for a complex-valued literal (a user-written ``1j``) rather than
    evaluating, because the lowered coeff embeds the (non-evaluable) trial/test channel."""
    del domain
    for term in ir.terms:
        for node in _walk(term.coeff):
            for attr in ("value", "val", "data", "constant"):
                v = getattr(node, attr, None)
                if v is None:
                    continue
                try:
                    if jnp.iscomplexobj(jnp.asarray(v)):
                        return True
                except Exception:  # not array-like; ignore
                    continue
    return False


def _solve_complex_block(ops: Any, solve_fn: Optional[Callable] = None) -> Any:
    """Solve a complex linear FEM system via the real-equivalent block (feax untouched —
    everything was assembled as two *real* systems)::

        [[A_r, -A_i], [A_i, A_r]] [u_r; u_i] = [b_r; b_i],     u = u_r + i u_i.

    ``ops = (op_r, op_i)`` are the Re-coeff and Im-coeff real systems (each a raw ``(A, b)``)."""
    from .trace import FemLinearSystem

    def _ab(op):
        if isinstance(op, FemLinearSystem):
            raise NotImplementedError(
                "Complex FEM with a runtime jno.np.parameter (the complex *inverse*) is a follow-on; "
                "this path is the forward complex solve. (A real parameter recovered through a complex "
                "forward works under the same real-equivalent block, but is not wired here yet.)"
            )
        A, b = op
        A = jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)
        return A, jnp.asarray(b).reshape(-1)

    (A_r, b_r), (A_i, b_i) = _ab(ops[0]), _ab(ops[1])
    n = A_r.shape[0]
    block = jnp.block([[A_r, -A_i], [A_i, A_r]])
    rhs = jnp.concatenate([b_r, b_i])
    solve_fn = solve_fn or (lambda A, b: jnp.linalg.solve(A, b))
    sol = jnp.asarray(solve_fn(block, rhs))
    return sol[:n] + 1j * sol[n:]


def _solve_complex_transient(blocks: Any, save_ts: Any = None) -> Any:
    """Integrate a complex *transient* system via the real-equivalent block (feax untouched):
    ``(M_r + i M_i) u_dot + (A_r + i A_i) u = (c_r + i c_i)`` becomes the real ``2N`` system
    ``M_blk u_dot + A_blk u = c_blk`` with ``M_blk=[[M_r,-M_i],[M_i,M_r]]`` (likewise A, c) and
    ``u=[u_r;u_i]``. Backward Euler over the block, then recombine to ``u_r + i u_i``. ``blocks =
    (block_r, block_i)`` are the Re-coeff and Im-coeff real ``FeaxTimeBlock``s. This covers
    Schrodinger (``i u_t = H u`` -> M_r = 0) since the *block* mass stays non-singular."""

    def _dn(a):
        return jnp.asarray(a.todense()) if hasattr(a, "todense") else jnp.asarray(a)

    def _MAc(b):
        M = _dn(b.M)
        A = _dn(b.operator_fn(0.0, {}) if b.operator_fn is not None else b.A)
        c = jnp.zeros((M.shape[0],), M.dtype) if b.affine_bias is None else jnp.asarray(b.affine_bias).reshape(-1)
        return M, A, c

    br, bi = blocks
    Mr, Ar, cr = _MAc(br)
    Mi, Ai, ci = _MAc(bi)
    n = Mr.shape[0]
    M_blk = jnp.block([[Mr, -Mi], [Mi, Mr]])
    A_blk = jnp.block([[Ar, -Ai], [Ai, Ar]])
    c_blk = jnp.concatenate([cr, ci])
    w0 = jnp.concatenate([jnp.asarray(br.state0).reshape(-1), jnp.asarray(bi.state0).reshape(-1)])
    t0, t1, dt = float(br.t0), float(br.t1), float(br.dt)
    grid = jnp.linspace(t0, t1, max(1, round((t1 - t0) / dt)) + 1)
    fr, fi = br.forcing_vector_fn, bi.forcing_vector_fn

    def step(w, t_next):  # backward Euler: (M_blk + dt A_blk) w_next = M_blk w + dt (c_blk + f)
        rhs = M_blk @ w + dt * c_blk
        if fr is not None or fi is not None:
            f_r = jnp.asarray(fr(t_next, {})).reshape(-1) if fr is not None else jnp.zeros((n,), c_blk.dtype)
            f_i = jnp.asarray(fi(t_next, {})).reshape(-1) if fi is not None else jnp.zeros((n,), c_blk.dtype)
            rhs = rhs + dt * jnp.concatenate([f_r, f_i])
        w_next = jnp.linalg.solve(M_blk + dt * A_blk, rhs)
        return w_next, w_next

    _, ws = jax.lax.scan(step, w0, grid[1:])
    traj = jnp.concatenate([w0[None, :], ws], axis=0)  # (n_grid, 2N)
    save_ts = grid if save_ts is None else jnp.asarray(save_ts)
    traj = jax.vmap(lambda col: jnp.interp(save_ts, grid, col), in_axes=1, out_axes=1)(traj)
    return traj[:, :n] + 1j * traj[:, n:]  # (n_save, N) complex


def _is_temporal_value_node(node: Any) -> bool:
    """True if an essential value carries a temporal Variable (time-varying ``g(x,t)``)."""
    return any(isinstance(v, Variable) and getattr(v, "axis", None) == "temporal" for v in _walk(node))


def _eval_value_node_at_time(value_node: Any, points: Any, t: Any) -> Any:
    """Evaluate a time-dependent coordinate value ``g(x, t)`` at ``points`` and time ``t``.

    Like :func:`_eval_value_node_at`, but the **temporal** Variable carries its own tag
    (``'__time__'``, separate from the spatial coordinate tag), so its context entry is a
    ``(n, 1)`` array filled with ``t`` while each spatial tag maps to the points. (Mapping
    every tag to the points, as the steady evaluator does, makes the time variable read a
    spatial column.) Reuses the existing ``TraceEvaluator``."""
    from .trace_evaluator import TraceEvaluator

    pts = jnp.atleast_2d(jnp.asarray(points))
    ctx: dict = {}
    for v in _walk(value_node):
        if isinstance(v, Variable):
            ctx[v.tag] = jnp.full((pts.shape[0], 1), t, dtype=pts.dtype) if getattr(v, "axis", None) == "temporal" else pts
    return jnp.reshape(TraceEvaluator({}).evaluate(value_node, context=ctx), (-1,))


def _dirichlet_spec(bare: Any) -> Tuple[Optional[int], Any, Any]:
    """``(component_index_or_None, value, value_node)`` for an essential Dirichlet residual.

    ``component`` is ``None`` for an all-component clamp (``u(region) - g``) or an integer
    for a per-component / roller BC. ``value`` is a constant or ``value(point)`` callable;
    ``value_node`` is the raw expression (kept so the transient route can detect/evaluate a
    time-varying ``g(x,t)``)."""
    comp, value_node = _essential_spec(bare)
    return comp, _value_from_node(value_node), value_node


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
        comp, node = _essential_spec(_bare(ic))
        vec = int(fields[fidx]["vec"])
        pts = jnp.asarray(prob.mesh[fidx].points)[:, : domain.dimension]
        n_i = int(pts.shape[0])
        const = _constant_of(node)
        if const is not None:
            vals = jnp.full((n_i,), float(const))
        else:
            v = jnp.asarray(_eval_value_node_at(node, pts))
            vals = jnp.broadcast_to(v, (n_i,)) if v.shape[0] == 1 else v
        vals = jnp.asarray(vals).reshape(-1)
        if comp is not None:  # one component c of a vector field: place at node*vec + c
            idx = offsets[fidx] + jnp.arange(n_i) * vec + comp
        elif vec == 1:  # scalar field
            idx = offsets[fidx] + jnp.arange(n_i)
        else:  # all components of a vector field: u(initial) - g
            if vals.shape[0] == n_i:
                # a single scalar value (e.g. u(initial) - 0) applies to every component
                vals = jnp.repeat(vals, vec)  # node-major: [n0c0, n0c1, n1c0, ...]
            elif vals.shape[0] != n_i * vec:
                raise NotImplementedError(
                    "jno.fem: vector initial condition must be a scalar (u(initial) - g, broadcast to "
                    "all components) or per component (u(initial)[i] - g_i) for coupled transient."
                )
            idx = offsets[fidx] + jnp.arange(n_i * vec)
        if vals.shape[0] != idx.shape[0]:
            raise ValueError(f"jno.fem: IC value count {vals.shape[0]} != {idx.shape[0]} dofs for field {fidx}.")
        state0 = state0.at[idx].set(vals)
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
        self._periodic = None  # periodic-tie reduction (prolongation P), attached by fem()

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
        return self._mode in ("transient", "complex_transient")

    @property
    def is_linear(self) -> bool:
        if self._mode == "transient":
            return bool(self._op.is_linear())
        return self._mode == "linear"

    @property
    def is_complex(self) -> bool:
        """A complex-valued problem (steady or transient), solved via the real-equivalent block."""
        if self._mode == "complex_transient":
            return True
        return self._mode == "complex"

    @property
    def operator(self) -> Any:
        """The raw assembled block — ``(A, b)`` / ``FemLinearSystem`` /
        ``FemResidualOperator`` (steady) or ``FeaxTimeBlock`` (transient)."""
        return self._op

    def solve(self, solve_fn=None, **kwargs) -> Any:
        """Differentiable forward solve as a trace node — the inverse-problem entry.

        Delegates to :meth:`FemLinearSystem.solve` (steady linear),
        :meth:`FemResidualOperator.solve` (steady nonlinear), or
        :meth:`FeaxTimeBlock.solve` (transient). The result is a jNO field: compare it
        to data and train any ``jno.np.parameter`` in the weak form through
        ``crux.solve``::

            alpha = jno.np.parameter((1,), name="alpha")
            fem = jno.fem([alpha * (ui.x * vi.x + ui.y * vi.y) - f * vi, u(xb, yb) - 0.0])
            crux = jno.core([(fem.solve() - u_obs).mse], domain=obs_domain)
            crux.solve(n)                      # recovers alpha

        ``solve_fn`` is **your** solver (jNO writes none):

        * steady linear: ``(A, b) -> u`` (default :func:`jax.numpy.linalg.solve`; e.g.
          ``lineax``);
        * steady nonlinear: ``(residual_fn, u0) -> u`` (default an
          ``optimistix.root_find`` Newton, implicit-diff; pass ``u0=`` for the guess);
        * transient: ``(block, args, save_ts) -> ys`` returning a ``(len(save_ts),
          n_dofs)`` trajectory (default a backward-Euler ``lax.scan`` over the block's
          assembled ``dt``; ``save_ts=`` overrides the sample times, default the domain's
          time grid). diffrax (``block.as_diffrax``) and a feax pipeline
          (``block.as_feax_pipeline``) are documented overrides.

        Enable x64 — the feax assembly is float64.

        For a **complex** steady linear problem (complex coefficients in the weak form),
        ``solve()`` returns the complex solution ``u_r + i·u_i`` via the real-equivalent block
        ``[[A_r,-A_i],[A_i,A_r]]`` (feax assembled only real systems); pass ``solve_fn=(A, b) -> u``
        to choose the real block solver.
        """
        if self._mode == "complex":
            return _solve_complex_block(self._op, solve_fn)
        if self._mode == "complex_transient":
            return _solve_complex_transient(self._op, save_ts=kwargs.get("save_ts"))
        if self._mode == "linear" and not isinstance(self._op, FemLinearSystem):
            # Non-parametric steady linear: solve the assembled (A, b) directly. ``solve_fn`` is
            # your ``(A, b) -> u`` (default dense ``jnp.linalg.solve``); A is densified from feax's
            # BCOO. (The runtime-parametric case is a FemLinearSystem and falls through below.)
            A, b = self._op
            A = jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)
            b = jnp.asarray(b).reshape(-1)
            _solve = solve_fn or (lambda A_, b_: jnp.linalg.solve(A_, b_))
            if self._periodic is not None:
                # periodic tie: eliminate slave DOFs via the prolongation P, solve the reduced
                # (P^T A P) u_red = P^T b, then prolong u = P u_red back to the full nodal layout.
                from .utils.solver.feax_utils import prolong, reduce_matrix, reduce_vector

                P = self._periodic["P"]
                u_red = _solve(reduce_matrix(P, A), reduce_vector(P, b))
                return prolong(P, u_red)
            return _solve(A, b)
        if self._mode == "nonlinear" and self._periodic is not None:
            # Periodic nonlinear: solve Newton in the reduced space -- r_red(u_red) = P^T r(P u_red) = 0,
            # then prolong u = P u_red (the tie is then satisfied exactly). Wraps the user's solve_fn
            # (or the operator's default Newton) so it operates on the reduced residual.
            from .utils.solver.feax_utils import restrict_state

            P = jnp.asarray(self._periodic["P"])
            kept, pvec = self._periodic["kept_nodes"], self._periodic["vec"]
            user_fn = solve_fn

            def _reduced(residual_fn, y0):
                def _base(rf, y):
                    if user_fn is not None:
                        return user_fn(rf, y)
                    import optimistix as optx

                    return optx.root_find(
                        lambda u, _a: rf(u), optx.Newton(rtol=1e-8, atol=1e-8), y, args=None, max_steps=100
                    ).value

                ur = _base(
                    lambda ur: P.T @ jnp.asarray(residual_fn(P @ ur)).reshape(-1), restrict_state(P, y0, kept, vec=pvec)
                )
                return P @ ur

            return self._op.solve(solve_fn=_reduced, **kwargs)
        return self._op.solve(solve_fn, **kwargs)

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

    def _time_block(self):
        """The (real) FeaxTimeBlock carrying the integration window — for complex_transient it is
        the Re-coeff block of the (block_r, block_i) pair."""
        return self._op[0] if self._mode == "complex_transient" else self._op

    @property
    def t0(self):
        return self._time_block().t0 if self.is_transient else None

    @property
    def t1(self):
        return self._time_block().t1 if self.is_transient else None

    @property
    def dt(self):
        return self._time_block().dt if self.is_transient else None

    @property
    def dofs(self) -> Optional[int]:
        if self._mode == "linear" and self._b is not None:
            return int(jnp.asarray(self._b).reshape(-1).shape[0])
        if self._mode == "nonlinear":
            size = getattr(self._op, "size", None)
            if size is not None:
                return int(size)
        if self.is_transient:
            tb = self._time_block()
            if getattr(tb, "state0", None) is not None:
                return int(jnp.asarray(tb.state0).reshape(-1).shape[0])
            if getattr(tb, "M", None) is not None:
                return int(jnp.asarray(tb.M).shape[0])
        prob = self.problem
        return int(prob.num_total_dofs_all_vars) if prob is not None else None

    def __repr__(self) -> str:
        kind = (
            self._mode if self._mode != "transient" else ("transient-linear" if self.is_linear else "transient-nonlinear")
        )
        return f"FEM({kind}, dofs={self.dofs}, terms={self.classification})"


# ---------------------------------------------------------------------------
# periodic ties:  u(A) - u(B)  (same trial, two boundary regions)
# ---------------------------------------------------------------------------
def _periodic_tie_spec(constraint: Any, domain: Any) -> Optional[Tuple[str, str, Optional[int], Any]]:
    """Recognise a **periodic tie** ``u(A) - u(B)`` and return ``(master_tag, slave_tag, comp,
    field_key)``; ``None`` for any non-tie constraint.

    The two regions are carried on the constraint's ``_periodic_tie`` attribute, stamped by the trace
    layer when it builds ``u(A) - u(B)`` (the only point where each side's region survives — the
    ``BinaryOp`` discards the per-side bound views). ``self`` (the left operand) is the eliminated
    slave; ``other`` (right) is the retained master — the relation ``u(A)=u(B)`` is symmetric.
    """
    tie = getattr(constraint, "_periodic_tie", None)
    if tie is None:
        return None
    # The trace stamps `_periodic_tie` on *any* trial-trial combination with clashing coords; only a
    # plain `u(A) - u(B)` (bare trial sides, no scaling, not `+`) is a tie. Reject the rest loudly
    # rather than silently mis-reading e.g. `u(A)+u(B)` (anti-periodic) or `2*u(A)-u(B)` as periodic.
    bare = _bare(constraint)
    if getattr(bare, "op", None) != "-" or any(
        getattr(side, "op", None) in {"+", "-", "*", "/"} for side in (bare.left, bare.right)
    ):
        raise ValueError(
            "jno.fem: a periodic tie must be `u(A) - u(B)` with bare trial sides (no scaling, no `+`). "
            "Anti-periodic or scaled relations between two boundaries are not supported."
        )
    slave_tag, master_tag = tie
    breg = getattr(domain, "_boundary_regions", {})
    if not (isinstance(slave_tag, str) and isinstance(master_tag, str)):
        return None
    if slave_tag == master_tag or slave_tag not in breg or master_tag not in breg:
        raise ValueError(
            f"jno.fem: a periodic tie `u(A) - u(B)` must connect two distinct boundary regions; "
            f"got {slave_tag!r} and {master_tag!r} (known boundary tags: {sorted(breg)})."
        )
    return (master_tag, slave_tag, None, _field_key_of(constraint))


def _boundary_facets(points: Any, cells: Any, dim: int, order: int) -> Optional[np.ndarray]:
    """Boundary facets of the **assembly** mesh as global node-id rows, with higher-order nodes.

    Vertex sub-connectivity is the first ``dim+1`` columns of ``cells`` (meshio orders vertices
    first); a facet is a cell edge (2D) / triangular face (3D) of those vertices that appears in
    exactly **one** cell. For ``order == 2`` each facet's edge midpoints are attached by coordinate
    (a P2 midpoint sits at the average of its two endpoint vertices) -- convention-light, with no
    dependence on feax's higher-order node ordering. Returns ``(n_facets, k)`` with
    ``k = 2/3`` (2D P1/P2) or ``3/6`` (3D P1/P2)."""
    points = np.asarray(points, dtype=float)
    cells = np.asarray(cells, dtype=int)
    if cells.ndim != 2 or cells.shape[0] == 0:
        return None
    verts = cells[:, : dim + 1]
    combos = [(0, 1), (1, 2), (2, 0)] if dim == 2 else [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    allf = np.concatenate([verts[:, list(c)] for c in combos], axis=0)
    _uniq, idx, counts = np.unique(np.sort(allf, axis=1), axis=0, return_index=True, return_counts=True)
    bverts = allf[idx[counts == 1]]  # boundary facets in original vertex orientation
    if order < 2:
        return bverts
    span = float(np.linalg.norm(points.max(0) - points.min(0))) or 1.0
    scale = span * 1.0e-7
    keymap = {tuple(np.round(p / scale).astype(np.int64)): i for i, p in enumerate(points)}

    def _mid(a: int, b: int) -> int:
        return keymap[tuple(np.round(((points[a] + points[b]) / 2.0) / scale).astype(np.int64))]

    edges_of = (lambda f: [(f[0], f[1])]) if dim == 2 else (lambda f: [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])])
    return np.asarray([list(f) + [_mid(a, b) for a, b in edges_of(f)] for f in bverts], dtype=int)


def _face_nodes(domain: Any, points: Any, bnodes: Optional[np.ndarray], tag: str) -> Optional[np.ndarray]:
    """Global node ids on a periodic face, taken from the tag's **predicate** (the user's intent),
    evaluated over the **assembly** boundary nodes ``bnodes`` (so P2 midpoints and 3D face nodes are
    included). ``domain.tag`` partitions each boundary node into a single tag, so a corner shared by
    two faces lands in only one -- re-evaluating the predicate recovers it in every face it satisfies
    (multi-direction needs that), while a corner-excluding predicate still omits the Dirichlet
    corners. Tags without a stored predicate fall back to the (exclusive, P1-mesh) ``tag_indices``."""
    ti = getattr(domain, "tag_indices", {})
    pred = getattr(domain, "_tag_predicates", {}).get(tag)
    if pred is not None and bnodes is not None and len(bnodes):
        coords = np.asarray(points)[bnodes]
        mask = np.asarray(pred(*(coords[:, i] for i in range(coords.shape[1]))), dtype=bool).reshape(-1)
        if mask.any():
            return np.asarray(bnodes)[mask]
    return np.asarray(ti[tag], dtype=int).reshape(-1) if tag in ti else None


def _chain_facets(points: Any, ids: Any) -> Optional[np.ndarray]:
    """Flat-face P1 boundary edges (global node-id pairs) as a fallback when assembly cells are
    unavailable (e.g. the native 1D route): sort the face nodes along their tangent and pair
    consecutive ones. ``None`` for < 2 nodes (a 1D face is a single node -> node-to-node tie)."""
    ids = np.asarray(ids, dtype=int).reshape(-1)
    if ids.size < 2:
        return None
    pts = np.asarray(points)[ids]
    tdim = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
    chain = ids[np.argsort(pts[:, tdim])]
    return np.column_stack([chain[:-1], chain[1:]])


def _assembly_cells(prob: Any) -> Tuple[Optional[np.ndarray], int]:
    """``(cells, element_order)`` of a feax problem's assembly mesh, or ``(None, 1)`` (e.g. native 1D)."""
    meshes = getattr(prob, "mesh", None) if prob is not None else None
    if not meshes:
        return None, 1
    am = meshes[0]
    order = 2 if str(getattr(am, "ele_type", "")).upper() in _P2_ELEMENTS else 1
    return np.asarray(am.cells, dtype=int), order


def _build_periodic_reduction(domain: Any, ties: List[Any], points: Any, cells: Any, ele_order: int, vec: int) -> dict:
    """Build the prolongation ``P`` for the collected ties on the **assembly** mesh (``points`` +
    ``cells``; ``cells=None`` for the native 1D route falls back to flat-chain facets)."""
    from .utils.solver.feax_utils import build_periodic_prolongation

    # Multidirectional periodicity needs each face to carry its shared corners (a corner is a slave
    # in several directions). ``domain.tag`` partitions a corner into one edge, so we recover full
    # faces from the tag predicate -- which an auto-generated tag (no `domain.tag` call) lacks. Reject
    # that case loudly rather than silently under-identify the corners and mis-solve.
    if len(ties) > 1:
        preds = getattr(domain, "_tag_predicates", {})
        missing = sorted({t for (m, s, _c, _fk) in ties for t in (m, s) if t not in preds})
        if missing:
            raise NotImplementedError(
                "jno.fem: multidirectional periodicity requires each periodic face to be defined via "
                f"`domain.tag(name, predicate)` so shared corners are included in every face; tag(s) "
                f"{missing} are auto-generated (no predicate). Define them with domain.tag(...)."
            )

    points = np.asarray(points)
    dim = int(getattr(domain, "dimension", points.shape[1]) or points.shape[1])
    bfacets = _boundary_facets(points, cells, dim, ele_order) if cells is not None else None
    bnodes = np.unique(bfacets) if bfacets is not None and bfacets.size else None

    pairs = [(master, slave) for (master, slave, _comp, _fk) in ties]
    faces: dict = {}
    for master, slave, _comp, _fk in ties:
        for tag in (master, slave):
            if tag not in faces and (f := _face_nodes(domain, points, bnodes, tag)) is not None:
                faces[tag] = f

    facets: dict = {}
    if bfacets is not None and bfacets.size:
        for master, _slave, _comp, _fk in ties:
            fn = set(np.asarray(faces.get(master, np.empty(0, int))).tolist())
            keep = np.array([set(row.tolist()).issubset(fn) for row in bfacets], dtype=bool)
            if keep.any():
                facets[master] = bfacets[keep]
    else:  # native 1D / no assembly cells -> flat-chain fallback
        facets = {m: ff for (m, _s, _c, _fk) in ties if (ff := _chain_facets(points, faces.get(m, ()))) is not None}

    return build_periodic_prolongation(points, pairs, faces, vec=vec, facets=facets)


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

    # Periodic ties `u(A) - u(B)` are enforced by algebraic reduction (a prolongation P that
    # eliminates the slave-face DOFs), not by assembly. Separate them out *before* the weak/Dirichlet
    # classification (`_region_and_support` would otherwise reject a residual that spans two regions).
    periodic_ties: List[Any] = []
    core_constraints: List[Any] = []
    for c in constraints:
        spec = _periodic_tie_spec(c, domain)
        (periodic_ties.append(spec) if spec is not None else core_constraints.append(c))
    constraints = core_constraints
    if periodic_ties and not constraints:
        raise ValueError("jno.fem: only periodic ties were given — add the PDE weak form (and any other conditions).")

    multifield = len(_field_keys(constraints)) > 1
    if vec is None and not multifield:
        vec = _infer_vec(constraints)  # single-field only (coupled fields carry per-field vec)

    # The transient route reduces M/A from the feax context at *assembly* time, so its P must be built
    # and injected before the time block is assembled (see the single-field transient branch below).
    # This holder carries that P so `_finalize` reuses it rather than rebuilding.
    periodic_holder: List[Any] = []

    def _finalize(fem_obj: "FEM") -> "FEM":
        """Attach the periodic reduction (if any): linear & nonlinear via ``FEM.solve``, transient via
        the time route's existing context-driven reduction. Still scoped to scalar single-field real."""
        if not periodic_ties:
            return fem_obj
        if fem_obj._mode in ("complex", "complex_transient"):
            raise NotImplementedError(
                "jno.fem: periodic ties on complex problems are not supported yet (the real-equivalent "
                "block would need the reduction applied to both the real and imaginary sub-blocks)."
            )
        if isinstance(fem_obj._op, FemLinearSystem):
            # The reduction is applied on the non-parametric (A, b) branch of FEM.solve; the
            # runtime-parametric FemLinearSystem.solve path would silently ignore it.
            raise NotImplementedError("jno.fem: periodic ties are not yet supported on runtime-parametric linear problems.")
        if multifield or (vec or 1) != 1:
            raise NotImplementedError(
                "jno.fem: periodic ties are currently supported only for scalar single-field problems."
            )
        # Transient already had its P built + fed to the feax context *before* block assembly (the
        # route reduces M/A at assembly time); reuse it. Linear & nonlinear build here and reduce in
        # FEM.solve. (1D has no assembly cells -> flat-chain facets via points only.)
        if periodic_holder:
            fem_obj._periodic = periodic_holder[0]
        else:
            cells, ele_order = _assembly_cells(getattr(fem_obj, "problem", None))
            fem_obj._periodic = _build_periodic_reduction(domain, periodic_ties, fem_obj.points, cells, ele_order, vec or 1)
        return fem_obj

    volume_terms: List[Any] = []
    boundary_terms: dict[str, List[Any]] = {}
    dirichlet_values: dict[str, Any] = {}
    # All-component vs per-component Dirichlet is tracked per (field, region), not per region:
    # in a multi-field problem a vector field's per-component BC and a scalar field's
    # all-component BC may legitimately share a wall. (The actual coupled BCs come from the
    # field-keyed `dirichlet_raw` below; `dirichlet_values` only feeds the single-field path.)
    dirichlet_style: dict[tuple, str] = {}  # (field_key, region) -> "all" | "per_component"
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
            comp, value, value_node = _dirichlet_spec(_bare(c))
            fk = _field_key_of(c)
            dirichlet_raw.append((fk, region, comp, value, value_node))
            style_key = (fk, region)
            if comp is None:  # all components: u(region) - g
                if dirichlet_style.get(style_key) == "per_component":
                    raise ValueError(
                        f"jno.fem: the same field on region {region!r} mixes all-component "
                        f"(u({region})-g) and per-component (u({region})[i]-g) Dirichlet."
                    )
                dirichlet_style[style_key] = "all"
                dirichlet_values[region] = value
                classification.append(f"dirichlet@{region}")
            else:  # one component (roller/symmetry): u(region)[i] - g
                if comp not in _COMPONENT_NAMES:
                    raise ValueError(
                        f"jno.fem: Dirichlet component index {comp} out of range (vector components are 0..2)."
                    )
                if dirichlet_style.get(style_key) == "all":
                    raise ValueError(
                        f"jno.fem: the same field on region {region!r} mixes all-component "
                        f"(u({region})-g) and per-component (u({region})[i]-g) Dirichlet."
                    )
                dirichlet_style[style_key] = "per_component"
                current = dirichlet_values.get(region)
                current = dict(current) if isinstance(current, dict) else {}
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
        from .utils.solver.fem_1d import assemble_fem_1d, assemble_fem_1d_multifield

        if multifield:  # coupled 1D -> native block assembly (feax has no LINE2 element)
            op, mode = assemble_fem_1d_multifield(
                domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, quad_degree=quad_degree
            )
        else:
            op, mode = assemble_fem_1d(
                domain, volume_terms, boundary_terms, dirichlet_values, ic_residuals, vec=vec, quad_degree=quad_degree
            )
        return _finalize(FEM(domain=domain, op=op, classification=classification, mode=mode))

    order = _infer_order(constraints)
    quad_degree = max(quad_degree, 2 * order)

    # ---- coupled / mixed multi-field -> block (multi-variable) assembly ----
    if multifield:
        return _finalize(
            _assemble_multifield(
                domain, volume_terms, boundary_terms, dirichlet_raw, ic_residuals, classification, quad_degree=quad_degree
            )
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

    # Periodic + transient: the time route reduces M/A from the domain feax context *at assembly time*,
    # so build P on the (now-built) assembly mesh and feed it in BEFORE the time block is assembled
    # (mirrors what domain.periodic sets). FEM.solve then prolongs the trajectory via the route.
    if periodic_ties and is_transient and not multifield and (vec or 1) == 1:
        _am = domain._feax_context["mesh"]  # the assembly mesh init_fem just built
        _eo = 2 if str(getattr(_am, "ele_type", "")).upper() in _P2_ELEMENTS else 1
        periodic_holder.append(
            _build_periodic_reduction(domain, periodic_ties, np.asarray(_am.points), np.asarray(_am.cells), _eo, vec or 1)
        )
        domain._feax_context["P"] = periodic_holder[0]["P"]
        domain._feax_context["periodic"] = periodic_holder[0]
        domain.fem_context["prolongation"] = periodic_holder[0]["P"]
        domain.fem_context["periodic"] = periodic_holder[0]

    if is_transient:
        from .utils.solver.time_route import _assemble_feax_time_from_ir

        if any(_is_temporal_value_node(vnode) for *_rest, vnode in dirichlet_raw):
            raise NotImplementedError(
                "jno.fem: time-varying Dirichlet g(x,t) on a single-field transient problem "
                "is not wired yet; it is supported on the coupled (multi-field) transient path. "
                "(Constant non-homogeneous Dirichlet works on both.)"
            )
        if len(ic_residuals) > 1:
            raise NotImplementedError("jno.fem: multiple initial conditions (multi-field) are not supported yet.")
        n_nodes = int(jnp.asarray(domain.mesh.points).shape[0])
        state0 = _initial_state(ic_residuals[0], domain) if ic_residuals else jnp.zeros((n_nodes,))
        # The integration window (t0, t1, dt) is read from jno.domain(time=...).
        # ---- complex transient (e.g. Schrodinger): real-equivalent split of M, A, and the IC ----
        if _is_complex_form(domain, ir):
            from .utils.solver.parametric_helpers import _clone_term_with_coeff

            s0 = jnp.asarray(state0)
            real_ir = LoweredWeakForm(domain=domain, terms=[_clone_term_with_coeff(t, t.coeff.real) for t in ir.terms])
            imag_ir = LoweredWeakForm(domain=domain, terms=[_clone_term_with_coeff(t, t.coeff.imag) for t in ir.terms])
            block_r = _assemble_feax_time_from_ir(domain, real_ir, state0=jnp.real(s0))
            block_i = _assemble_feax_time_from_ir(domain, imag_ir, state0=jnp.imag(s0))
            return _finalize(
                FEM(domain=domain, op=(block_r, block_i), classification=classification, mode="complex_transient")
            )
        block = _assemble_feax_time_from_ir(domain, ir, state0=state0)
        return _finalize(FEM(domain=domain, op=block, classification=classification, mode="transient"))

    if ic_residuals:
        raise ValueError("jno.fem: an initial condition was given but the weak form has no time derivative.")

    # ---- complex (steady linear): real-equivalent split, feax sees only real forms ----
    if _is_complex_form(domain, ir):
        from .utils.solver.parametric_helpers import _clone_term_with_coeff

        # Re(c·T) = Re(c)·T and Im(c·T) = Im(c)·T (the FE trial/test T is real), so two ordinary
        # real assemblies give A_r/b_r and A_i/b_i; FEM.solve() then forms the real-equivalent
        # block and recombines to a complex u. No feax change, no native-complex reliance.
        real_ir = LoweredWeakForm(domain=domain, terms=[_clone_term_with_coeff(t, t.coeff.real) for t in ir.terms])
        imag_ir = LoweredWeakForm(domain=domain, terms=[_clone_term_with_coeff(t, t.coeff.imag) for t in ir.terms])
        op_r = _assemble_fem_system_from_ir(domain, real_ir)
        op_i = _assemble_fem_system_from_ir(domain, imag_ir)
        return _finalize(FEM(domain=domain, op=(op_r, op_i), classification=classification, mode="complex"))

    probe = ir.volume_expr if ir.volume_expr is not None else next(iter(ir.boundary_exprs.values()))
    target = _infer_solver_target(domain, probe)
    if target == "fem_system":
        op = _assemble_fem_system_from_ir(domain, ir)
        return _finalize(FEM(domain=domain, op=op, classification=classification, mode="linear"))

    op = _assemble_fem_residual_from_ir(domain, ir)
    return _finalize(FEM(domain=domain, op=op, classification=classification, mode="nonlinear"))


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
    dirichlet_tv: List[Any] = []  # (field_idx, region, comp, value_node) for time-varying g(x,t)
    for field_key, region, comp, value, value_node in dirichlet_raw:
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
        if _is_temporal_value_node(value_node):
            dirichlet_tv.append((fidx, region, comp, value_node))
    domain._fem_dirichlet_by_field = by_field

    # Coupled transient (multi-field + time): block M + block spatial operator A.
    if is_transient:
        return _assemble_multifield_transient(domain, ir, fields, field_index, ic_residuals, classification, dirichlet_tv)

    # Nonlinear coupled: feax autodiffs the block residual/Jacobian on the multi-field
    # problem (same _build_feax_problem path as linear), so the nonlinear route works
    # for coupled fields too. A nonlinear volume *or* surface term routes here (a linear
    # Robin term stays on the linear path, where its surface contribution lands in A).
    if any(_is_obviously_nonlinear_in_unknown(domain, b) for b in weak_bares):
        op = _assemble_fem_residual_from_ir(domain, ir)
        return FEM(domain=domain, op=op, classification=classification, mode="nonlinear")
    op = _assemble_fem_system_from_ir(domain, ir)
    return FEM(domain=domain, op=op, classification=classification, mode="linear")


def _sum_forcings(f1, f2):
    """Sum two optional forcing callables ``f(t, args) -> vector`` (None acts as 0)."""
    if f1 is None:
        return f2
    if f2 is None:
        return f1

    def summed(t, args=None):
        return jnp.asarray(f1(t, args)).reshape(-1) + jnp.asarray(f2(t, args)).reshape(-1)

    return summed


def _time_varying_dirichlet_lift(domain, prob, bc, A, dirichlet_tv, fields, field_index):
    """JIT-friendly time-varying Dirichlet load ``c_dir(t)`` (or ``None`` if no time-varying BC).

    Returns the per-time load that makes ``A w = c_dir(t)`` carry ``g(.,t)`` on the Dirichlet
    rows and the lift ``-A_fd·g(t)`` on the free rows — i.e. the steady RHS for the Dirichlet
    values at time ``t``, computed exactly as the steady path does: ``c(t) = A·u0(t) -
    res_bc(u0(t), bc_t)`` with ``u0(t)`` the ``g(t)``-lifted state (``g`` on the Dirichlet
    DOFs). ``g_vals(t)`` evaluates each time-varying spec's ``g(x,t)`` at the Dirichlet DOFs'
    coordinates (precomputed once) and masks it onto that spec's rows; constant rows keep
    ``bc.bc_vals``. All per-``t`` work is pure JAX (``where`` + ``replace_vals`` + a matvec +
    the parametric residual), so it traces under ``jax.jit`` / ``lax.scan``."""
    if not dirichlet_tv:
        return None
    import feax as fe
    import jax
    from feax.assembler import create_res_bc_parametric

    rows = np.asarray(bc.bc_rows).reshape(-1)
    if rows.shape[0] == 0:
        return None
    offsets = list(prob.offset) + [int(prob.num_total_dofs_all_vars)]
    dim = domain.dimension
    # per-row (field, component, coordinate)
    row_field = np.searchsorted(np.asarray(offsets), rows, side="right") - 1
    bc_coords = np.zeros((rows.shape[0], dim))
    row_comp = np.zeros(rows.shape[0], dtype=int)
    for i, r in enumerate(rows):
        f = int(row_field[i])
        vt = int(fields[f]["vec"])
        local = int(r) - offsets[f]
        bc_coords[i] = np.asarray(prob.mesh[f].points)[local // vt][:dim]
        row_comp[i] = local % vt
    bc_coords_j = jnp.asarray(bc_coords)
    # per-spec mask over the bc rows + the value node
    tv = []
    for fidx, region, comp, value_node in dirichlet_tv:
        loc = domain._make_tag_location_fn(region)
        in_region = np.asarray(jax.vmap(loc)(bc_coords_j)).reshape(-1)
        mask = (row_field == fidx) & in_region
        if comp is not None:
            mask = mask & (row_comp == comp)
        tv.append((jnp.asarray(mask), value_node))

    res_bc_param = create_res_bc_parametric(prob)
    iv = fe.InternalVars()
    A = jnp.asarray(A)
    rows_j = jnp.asarray(rows)
    zeros = jnp.zeros((offsets[-1],), dtype=A.dtype)
    base_vals = jnp.asarray(bc.bc_vals)

    def lift_fn(t, args=None):
        g_vals = base_vals
        for mask, node in tv:
            g_vals = jnp.where(mask, _eval_value_node_at_time(node, bc_coords_j, t), g_vals)
        u0 = zeros.at[rows_j].set(g_vals)  # g(t)-lifted state (g on the Dirichlet DOFs)
        res = jnp.asarray(res_bc_param(u0, iv, bc.replace_vals(g_vals))).reshape(-1)
        return A @ u0 - res  # steady RHS for g(t): g on Dirichlet rows, -A_fd·g on free rows

    return lift_fn


def _assemble_multifield_transient(domain, ir, fields, field_index, ic_residuals, classification, dirichlet_tv=()):
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

    An **algebraic / DAE field** (one with no time derivative — e.g. the pressure of a
    transient Stokes problem) is handled structurally: building the mass against the full
    field set gives it a **zero mass block**. The resulting ``(M + dt·A)`` is then a saddle
    system, well-posed exactly when the steady block is (inf-sup + a pressure pin via
    ``domain.point_region``). jno constructs it faithfully; the user's solve reveals an
    ill-posed setup (e.g. a forgotten ``u_t``)."""
    from .utils.solver.backend_blocks import FeaxTimeBlock
    from .utils.solver.feax_utils import _dense_array, _zero_forcing_dirichlet_rows, _zero_mass_dirichlet_rows
    from .utils.solver.fem_route import _assemble_fem_residual_from_ir, _assemble_fem_system_from_ir
    from .utils.solver.time_route import (
        _build_auto_forcing_vector_fn,
        _infer_time_window,
        _is_linear_first_order_ir,
        _split_first_order_linear_terms,
    )
    from .utils.solver.weak_form import LoweredWeakForm

    mass_ir, op_ir, src_ir = _split_first_order_linear_terms(ir)

    # Shared field layout so the separately-assembled mass and operator blocks align.
    # Threaded into the mass assembly too, so a field with no temporal term gets a zero
    # mass block (the DAE case) rather than a mis-sized M. The mass is assembled *raw*
    # (apply_dirichlet=False) and only its Dirichlet ROWS are zeroed below — the Dirichlet
    # columns are kept so the stepper's M(w_new-w_old) captures M_fd·ġ for time-varying
    # Dirichlet (it cancels when ġ=0). store_on_domain=False keeps this scratch problem
    # off the domain so the operator problem/bc remain the ones exposed on `fem`.
    override = (fields, field_index)
    M_raw = jnp.asarray(
        _dense_array(
            _assemble_fem_system_from_ir(
                domain, mass_ir, fields_override=override, apply_dirichlet=False, store_on_domain=False
            )[0]
        )
    )
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
        A = jnp.asarray(_dense_array(A_sys))
        bc, prob = domain._feax_bc, domain._feax_problem  # operator block problem + Dirichlet
        M = _zero_mass_dirichlet_rows(M_raw, bc)
        # Source/forcing terms (no trial) -> a block forcing f(t), zeroed at Dirichlet rows
        # (a source contributes only to free DOFs); constant or temporal.
        source_fn = None
        if len(src_ir.terms) > 0:
            source_fn = _zero_forcing_dirichlet_rows(
                _build_auto_forcing_vector_fn(
                    domain, src_ir, size=int(A.shape[0]), dtype=A.dtype, fields_override=override
                )[0],
                bc,
            )
        # Time-varying Dirichlet g(x,t) -> a t-dependent lift c_dir(t) (carries g(.,t) on the
        # Dirichlet rows); the constant lift `bA` is then subsumed, so affine_bias = 0.
        tv_lift_fn = _time_varying_dirichlet_lift(domain, prob, bc, A, dirichlet_tv, fields, field_index)
        if tv_lift_fn is not None:
            forcing_vector_fn = _sum_forcings(tv_lift_fn, source_fn)
            affine_bias = jnp.zeros((int(A.shape[0]),), dtype=A.dtype)
        else:
            forcing_vector_fn = source_fn
            affine_bias = jnp.asarray(bA).reshape(-1)
        state0 = _multifield_initial_state(domain, prob, fields, field_index, ic_residuals)
        block = FeaxTimeBlock(
            M=M, A=A, affine_bias=affine_bias, forcing_vector_fn=forcing_vector_fn, state0=state0, **common
        )
        return FEM(domain=domain, op=block, classification=classification, mode="transient")

    # Nonlinear: M u_dot + R(u) = 0, with R/J the block spatial residual/Jacobian that feax
    # autodiffs on the multi-field problem (same path as steady nonlinear coupled). A
    # (time-constant) source rides the residual by folding src_ir into the operator IR.
    residual_ir = LoweredWeakForm(domain=domain, terms=list(op_ir.terms) + list(src_ir.terms))
    spatial = _assemble_fem_residual_from_ir(domain, residual_ir, fields_override=override)
    prob = domain._feax_problem
    M = _zero_mass_dirichlet_rows(M_raw, domain._feax_bc)
    state0 = _multifield_initial_state(domain, prob, fields, field_index, ic_residuals)
    block = FeaxTimeBlock(
        mass=lambda t, args=None, _M=M: _M,
        residual=lambda u, t, args=None, _r=spatial.residual: _r(jnp.asarray(u)),
        jacobian=lambda u, t, args=None, _j=spatial.jacobian: _j(jnp.asarray(u)),
        state0=state0,
        **common,
    )
    return FEM(domain=domain, op=block, classification=classification, mode="transient")


# ---------------------------------------------------------------------------
# Regularization for nodal field parameters (jno.np.parameter(phi))
# ---------------------------------------------------------------------------
def _lower_volume_form(domain: Any, expr: Any):
    """Lower one scalar **volume** weak form to a ``LoweredWeakForm`` (no ``init_fem``)."""
    from .utils.solver.weak_form import (
        LoweredChannelTerm,
        LoweredWeakForm,
        _apply_sign,
        _split_additive_terms,
    )

    terms: List[Any] = []
    for sign, sub in _split_additive_terms(domain, _bare(expr)):
        terms.append(
            LoweredChannelTerm(
                sign=sign,
                support="volume",
                region_id="volume",
                channel="raw",
                coeff=_apply_sign(domain, sign, sub),
                variable_id=0,
                value_shape=(),
                original_expr=sub,
            )
        )
    return LoweredWeakForm(domain=domain, terms=terms)


def _fe_symbols_bound(domain: Any):
    """``(ui, vi, axes)`` -- P1 trial/test bound to the domain's interior coordinates."""
    u_sym, v_sym = domain.fem_symbols()
    coords = domain.variable("interior", split=True)
    axes = ("x", "y", "z")[: int(domain.dimension)]
    ui = u_sym.bind(**{ax: coords[i] for i, ax in enumerate(axes)})
    vi = v_sym.bind(**{ax: coords[i] for i, ax in enumerate(axes)})
    return ui, vi, axes


def _assemble_fe_gram(domain: Any, kind: str = "stiffness"):
    """Raw P1 Gram matrix on the domain's scalar space (no Dirichlet, no ``init_fem``,
    ``store_on_domain=False`` so it never clobbers a cached FEM problem):

    * ``'stiffness'`` -> ``L = integral grad(phi_i).grad(phi_j)``  (``k^T L k = integral |grad k|^2``)
    * ``'mass'``      -> ``M = integral phi_i phi_j``              (``k^T M k = integral k^2``)
    """
    from .utils.solver.fem_route import _assemble_fem_system_from_ir

    ui, vi, axes = _fe_symbols_bound(domain)
    if kind == "stiffness":
        form = None
        for ax in axes:
            t = getattr(ui, ax) * getattr(vi, ax)
            form = t if form is None else form + t
    elif kind == "mass":
        form = ui * vi
    else:
        raise ValueError(f"_assemble_fe_gram: unknown kind {kind!r}")
    ir = _lower_volume_form(domain, form)
    A, _b = _assemble_fem_system_from_ir(domain, ir, apply_dirichlet=False, store_on_domain=False)
    return jnp.asarray(A.todense() if hasattr(A, "todense") else A)


def _assemble_h1_stiffness(domain: Any):  # back-compat alias
    return _assemble_fe_gram(domain, "stiffness")


def _fe_element_gradient_data(domain: Any):
    """``(shape_grads, JxW, cells)`` for the domain's P1 space -- per-element gradient
    geometry for total variation ``integral |grad k|``. Built standalone (no clobber)."""
    from .utils.solver.feax_utils import _build_feax_problem

    ui, vi, axes = _fe_symbols_bound(domain)
    form = None
    for ax in axes:
        t = getattr(ui, ax) * getattr(vi, ax)
        form = t if form is None else form + t
    ir = _lower_volume_form(domain, form)
    problem, _bc = _build_feax_problem(domain, ir, apply_dirichlet=False, store_on_domain=False)
    return (
        jnp.asarray(problem.shape_grads),
        jnp.asarray(problem.JxW),
        jnp.asarray(problem.fes[0].cells),
    )


_REG_KINDS = ("h1seminorm", "tv", "l2", "nonneg", "bounded")


def _field_regularizer_term(param: Any, kind: str = "h1seminorm", **kwargs):
    """Build a regularization loss term (a :class:`FunctionCall` over ``param``) for a
    nodal field parameter. See :meth:`ModelCall.regularize` for the kinds + options."""
    from .trace import FunctionCall

    domain = getattr(param.model, "_fem_field_domain", None)
    if domain is None:
        raise ValueError("regularize(...) is only for a FEM field parameter (jno.np.parameter(<fem symbol>)).")
    k = str(kind).lower()

    if k in ("h1seminorm", "h1", "smooth"):  # integral |grad k|^2 = k^T L k
        L = _assemble_fe_gram(domain, "stiffness")

        def _h1(kv, _L=L):
            kf = jnp.asarray(kv).reshape(-1)
            return kf * (_L @ kf)

        return FunctionCall(_h1, [param], name="h1seminorm")

    if k in ("l2", "tikhonov", "ridge"):  # integral (k - ref)^2 = (k-ref)^T M (k-ref)
        M = _assemble_fe_gram(domain, "mass")
        ref = jnp.asarray(kwargs.get("ref", 0.0))

        def _l2(kv, _M=M, _ref=ref):
            kf = jnp.asarray(kv).reshape(-1)
            kf = kf - (_ref.reshape(-1) if _ref.ndim else _ref)
            return kf * (_M @ kf)

        return FunctionCall(_l2, [param], name="l2")

    if k in ("tv", "totalvariation"):  # integral |grad k|  (edge-preserving; eps-smoothed)
        sg, jxw, cells = _fe_element_gradient_data(domain)
        eps = float(kwargs.get("eps", 1.0e-8))

        def _tv(kv, _sg=sg, _jxw=jxw, _cells=cells, _eps=eps):
            kc = jnp.asarray(kv).reshape(-1)[_cells]  # (n_cells, n_local)
            gradk = jnp.einsum("cqld,cl->cqd", _sg, kc)  # (n_cells, n_quad, dim)
            mag = jnp.sqrt(jnp.sum(gradk**2, axis=-1) + _eps)  # (n_cells, n_quad)
            jxw = jnp.reshape(_jxw, mag.shape)  # JxW arrives (n_cells, 1, n_quad)
            return jnp.sum(mag * jxw, axis=1)  # (n_cells,) per-element TV

        return FunctionCall(_tv, [param], name="tv")

    if k == "nonneg":  # soft positivity barrier strength * relu(-k)
        strength = float(kwargs.get("strength", 1.0))

        def _nn(kv, _s=strength):
            return _s * jnp.maximum(0.0, -jnp.asarray(kv).reshape(-1))

        return FunctionCall(_nn, [param], name="nonneg")

    if k == "bounded":  # soft two-sided barrier outside [lo, hi]
        if "lo" not in kwargs or "hi" not in kwargs:
            raise ValueError("regularize('bounded', lo=..., hi=...) requires both lo and hi.")
        lo, hi = float(kwargs["lo"]), float(kwargs["hi"])

        def _bd(kv, _lo=lo, _hi=hi):
            kf = jnp.asarray(kv).reshape(-1)
            return jnp.maximum(0.0, kf - _hi) + jnp.maximum(0.0, _lo - kf)

        return FunctionCall(_bd, [param], name="bounded")

    raise ValueError(f"Unknown regularizer kind {kind!r}; supported: {_REG_KINDS}.")
