"""Strong-form trace utilities shared by :func:`jno.fdm` and :func:`jno.fvm`.

Both read the same constraint list (``u = domain.unknown()``, ``u.bind(...)``, ``u(xb, yb) - g``,
``ur.d(nr) - h``, ``u(xi, yi) - u0``) and differ only in how a PDE term is discretised: collocated at the
mesh nodes (fdm) or integrated over control volumes (fvm). The helpers here walk that trace: they find the
unknowns, classify a constraint by its region and its derivatives, and rewrite time derivatives for the
method-of-lines split. Nothing in this module discretises.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _rtol(dtype=None) -> float:
    """The relative tolerance to which two floating-point results that should be equal are compared (an
    affine probe, a symmetry probe, a node on a lattice): ``√ε`` of ``dtype`` (the default float if ``None``),
    1.5e-8 in float64 and 3.5e-4 in float32. It replaces fixed 1e-10 / 1e-9 / 1e-6 thresholds, which were
    below float32's resolution (every float32 operator probed "nonlinear" and "non-symmetric")."""
    return float(np.sqrt(np.finfo(jnp.result_type(float) if dtype is None else dtype).eps))


def _unwrap(node):
    return getattr(node, "_expr", node)  # view -> underlying Placeholder


def _iter(node):
    from .utils.solver.solver_helper import iter_children

    return iter_children(node) or ()


def _refuse_fem_only_unknowns(constraints, who="jno.fdm"):
    """``domain.unknown(order=2)`` (or a non-nodal space, a complex field, ``symmetric=True``) has no valued P1
    nodal field behind it -- it is an FEM trial symbol -- so there is nothing for a strong-form solve to
    hold. Say that, rather than reporting that no unknown was found."""
    from .trace import TrialFunction

    def walk(n):
        n = _unwrap(n)
        if isinstance(n, TrialFunction) and n.__dict__.get("_fem_only_unknown", False):
            raise NotImplementedError(
                f"{who}([...]): the unknown {n.name!r} (order={n.order}, space={n.space!r}, "
                f"value_shape={n.value_shape}) is FEM-only -- {who} solves for a real P1 nodal field with "
                "full storage, i.e. `domain.unknown(...)` with order=1, the default space, complex=False and "
                "symmetric=False. Solve this one with jno.fem([...])."
            )
        for c in _iter(n):
            walk(c)

    for c in constraints:
        walk(c)


def _has_time_variable(node):
    """Does a value expression use the temporal Variable (``tb`` in ``g(xb, yb, tb)``)?"""
    from .trace import Variable

    n = _unwrap(node) if not isinstance(node, (int, float)) else None
    if n is None:
        return False
    if isinstance(n, Variable):
        return getattr(n, "axis", None) == "temporal"
    return any(_has_time_variable(c) for c in _iter(n))


def _find_unknowns(constraints, who="jno.fdm"):
    """All ``domain.unknown()`` fields (nodal-field-parameter ModelCalls' Models) in the constraints, in
    **declaration** order — a **coupled** system has several, and ``.solve()`` returns one row per field in
    this order. (It used to be first-appearance order: listing the v-equation first silently swapped the
    returned ``u`` and ``v``.) At least one is required. The k-th PDE equation fills the k-th DOF block's
    interior rows, and each ``u_k(region) - g`` BC the boundary rows of its own field's block; the joint
    system is the same whichever order the equations are listed in."""
    from .trace import ModelCall

    _refuse_fem_only_unknowns(constraints, who)
    seen, order = set(), []

    def walk(n):
        n = _unwrap(n)
        if isinstance(n, ModelCall) and getattr(n.model, "_fem_field", None) == "node" and n.model.layer_id not in seen:
            seen.add(n.model.layer_id)
            order.append(n.model)
        for c in _iter(n):
            walk(c)

    for c in constraints:
        walk(c)
    if not order:
        raise ValueError(
            f"{who}([...]): expected at least one domain.unknown() field. Author the strong form with "
            "`u = domain.unknown()` (declare several for a coupled system)."
        )
    return sorted(order, key=lambda m: m.layer_id)  # layer ids come from a global, increasing counter


def _periodic_axis(points_a, points_b):
    """The grid axis a periodic tie ``u(A) - u(B)`` wraps, read from the geometry of its two faces: the
    coordinate that is constant over each face, and different between them. ``None`` if the two regions
    are not a pair of opposite axis-aligned faces. (It used to be read from the face NAMES, which silently
    depended on which naming convention the grid used.)"""
    pa, pb = np.asarray(points_a), np.asarray(points_b)
    if pa.size == 0 or pb.size == 0:
        return None
    ax_a, ax_b = int(np.argmin(np.ptp(pa, axis=0))), int(np.argmin(np.ptp(pb, axis=0)))
    tol = _rtol(pa.dtype) * max(float(np.ptp(np.concatenate([pa, pb]), axis=0).max()), np.finfo(float).tiny)
    flat = np.ptp(pa[:, ax_a]) <= tol and np.ptp(pb[:, ax_b]) <= tol  # relative to the faces' extent
    if not flat or ax_a != ax_b or abs(pa[0, ax_a] - pb[0, ax_b]) <= tol:
        return None
    return ax_a


def _contains_unknown(node, model):
    from .trace import ModelCall

    n = _unwrap(node)
    if isinstance(n, ModelCall) and n.model is model:
        return True
    return any(_contains_unknown(c, model) for c in _iter(n))


def _region_tag(constraint):
    cv = getattr(constraint, "_coord_vars", None) or {}
    tags = {v.tag for v in cv.values() if getattr(v, "axis", None) != "temporal"}  # spatial region only
    return next(iter(tags)) if len(tags) == 1 else (tags or {None})


def _has_temporal(node):
    """Does the expression contain a strong-form time derivative ``u.t`` (:class:`TemporalDerivative`)?"""
    from .trace import TemporalDerivative

    n = _unwrap(node)
    if isinstance(n, TemporalDerivative):
        return True
    return any(_has_temporal(c) for c in _iter(n))


def _has_temporal_of(node, unknown):
    """Does ``node`` contain a time derivative ``u.t`` of this ``unknown``?"""
    from .trace import TemporalDerivative

    n = _unwrap(node)
    if isinstance(n, TemporalDerivative) and _contains_unknown(n.target, unknown):
        return True
    return any(_has_temporal_of(c, unknown) for c in _iter(n))


def _has_unknown_derivative(node, unknown):
    """Does the expression contain a **derivative of the unknown** (a Jacobian/Hessian/TemporalDerivative
    whose target is the unknown)? This is what distinguishes a **PDE** residual from a value-only
    **pinning** condition ``u(region) - g`` (Dirichlet / IC / sub-region pin) — the strong-form analogue
    of the fem "does it contain the test function?" rule."""
    from .trace import Hessian, Jacobian, TemporalDerivative

    n = _unwrap(node)
    if isinstance(n, (Jacobian, Hessian, TemporalDerivative)) and _contains_unknown(getattr(n, "target", None), unknown):
        return True
    return any(_has_unknown_derivative(c, unknown) for c in _iter(n))


def _mesh_nodes_in(pts, geom):
    """Indices of the mesh nodes ``pts`` inside a geometric region ``geom`` (registered via
    ``domain.region(name, region)``) — resolves the region to a node subset for pinning/solving on a
    subdomain. A ``jno.shape`` uses the analytic, shapely-free :meth:`shape.contains` (2-D and 3-D — the
    primary path); a shapely geometry falls back to shapely (2-D mesh-conforming regions in
    ``polygon_domain``, which stay shapely by scope)."""
    from .geometry import shape

    p = np.asarray(pts)
    if isinstance(geom, shape):
        mask = np.asarray(geom.contains(p[:, : geom.dim]))
    else:
        import shapely

        # a boundary node counts as inside: buffer by a tolerance relative to the region's size, not 1e-9
        minx, miny, maxx, maxy = geom.bounds
        mask = np.asarray(shapely.contains_xy(geom.buffer(_rtol() * max(maxx - minx, maxy - miny)), p[:, 0], p[:, 1]))
    return np.nonzero(mask)[0].astype(int)


def _temporal_order(node):
    """Highest order of time derivative in ``node``: ``u.t`` is 1, ``u.tt`` (a chained
    :class:`TemporalDerivative`) is 2."""
    from .trace import TemporalDerivative

    n = _unwrap(node)
    if isinstance(n, TemporalDerivative):
        return 1 + _temporal_order(n.target)
    return max((_temporal_order(c) for c in _iter(n)), default=0)


def _set_temporal(node, val, val_tt=0.0):
    """Replace every ``u.t`` (:class:`TemporalDerivative`) in ``node`` with the constant ``val`` and every
    ``u.tt`` with ``val_tt``, leaving the rest of the expression intact. Probes recover, for a residual
    ``F = m·u.tt + c·u.t + R_spatial`` affine in the time derivatives, the **spatial** residual
    ``R_spatial = F(0, 0)`` and the coefficients ``c = F(u.t=1) − F(0, 0)`` and ``m = F(u.tt=1) − F(0, 0)``
    — so a general ``c(x)·u.t`` term (variable material, e.g. ``ρcₚ(x)·u.t``) is handled without parsing
    its structure. (``u.tt`` used to be replaced by ``val`` as
    if it were ``u.t``, which silently solved a wave equation as a heat equation.)"""
    from .trace import BinaryOp, FunctionCall, Hessian, Jacobian, Literal, Placeholder, TemporalDerivative

    if isinstance(node, TemporalDerivative):
        return Literal(float(val_tt if _temporal_order(node) >= 2 else val))
    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, _set_temporal(node.left, val, val_tt), _set_temporal(node.right, val, val_tt))
    if isinstance(node, FunctionCall):
        return node.copy_with_args([_set_temporal(a, val, val_tt) if isinstance(a, Placeholder) else a for a in node.args])
    if isinstance(node, Jacobian):
        return Jacobian(_set_temporal(node.target, val, val_tt), node.variables, node.scheme)
    if isinstance(node, Hessian):
        return Hessian(_set_temporal(node.target, val, val_tt), node.variables, node.scheme, node.trace)
    return node


def _zero_temporal(node):
    """Drop the ``u.t`` terms (``_set_temporal(node, 0)``) → the spatial residual ``R_spatial`` for the
    method-of-lines split ``M u̇ + R_spatial = 0``."""
    return _set_temporal(node, 0.0)


def _normal_jacobian(node):
    """The normal-derivative node ``ui.d(n, scheme)`` (a :class:`Jacobian` w.r.t. a normal Variable —
    tag ``n_<region>``, from ``domain.variable(region, normals=True)``) inside ``node``, or ``None``.
    Its presence marks a Neumann/Robin (flux) condition; its ``.scheme`` is the FD stencil to honour."""
    from .trace import Jacobian

    n = _unwrap(node)
    if isinstance(n, Jacobian) and any(str(getattr(v, "tag", "")).startswith("n_") for v in getattr(n, "variables", [])):
        return n
    for c in _iter(n):
        found = _normal_jacobian(c)
        if found is not None:
            return found
    return None


def _is_normal_jacobian(node):
    from .trace import Jacobian

    return isinstance(node, Jacobian) and any(
        str(getattr(v, "tag", "")).startswith("n_") for v in getattr(node, "variables", [])
    )


def _spatial_tags(c):
    """The spatial coordinate tags a constraint is bound to: those of its bound views, and of every
    coordinate Variable inside it. Vector-view arithmetic (``Ui.div()``, ``Ui.grad() @ Ui``) does not carry
    the views' bindings, so reading them alone left ``interior`` out of the evaluation context (a
    ``KeyError`` deep in a derivative). Normal tags ``n_*`` are left to the flux machinery."""
    tags = {v.tag for v in (getattr(c, "_coord_vars", None) or {}).values() if getattr(v, "axis", None) != "temporal"}
    tags |= {
        v.tag
        for v in _iter_variables(c)
        if getattr(v, "axis", "spatial") == "spatial"
        and not str(v.tag).startswith(("n_", "fdm_bgrad_", "fvm_"))
        and v.tag not in ("cell_size", "__time__")
    }
    return tags


def _context_leaf(tag, component, domain):
    """A :class:`Variable` reading column ``component`` of the per-node array ``context[tag]`` (all columns
    when ``component`` is ``None``) — how a rewritten boundary condition reads a value the solver supplies
    (a boundary gradient, a normal)."""
    from .trace import Variable

    leaf = Variable.__new__(Variable)
    leaf.tag, leaf.dim, leaf.axis, leaf.fem_meta, leaf.size, leaf._domain = (
        tag,
        [0, None] if component is None else [component, component + 1],
        "spatial",
        None,
        1,
        domain,
    )
    return leaf


def _map_children(node, fn):
    """``node`` with ``fn`` applied to its children (the node types a strong-form condition is made of)."""
    from .trace import BinaryOp, FunctionCall, Hessian, Jacobian, Placeholder

    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, fn(node.left), fn(node.right))
    if isinstance(node, FunctionCall):
        return node.copy_with_args([fn(a) if isinstance(a, Placeholder) else a for a in node.args])
    if isinstance(node, Hessian):
        return Hessian(fn(node.target), node.variables, node.scheme, node.trace)
    if isinstance(node, Jacobian):
        return Jacobian(fn(node.target), node.variables, node.scheme)
    return node


def _iter_variables(node):
    """Every :class:`Variable` in the expression."""
    from .trace import Variable

    n = _unwrap(node)
    if isinstance(n, Variable):
        return [n]
    return [v for c in _iter(n) for v in _iter_variables(c)]


def _normal_jacobians(node):
    """Every normal-derivative node ``ub.d(n)`` in the expression."""
    n = _unwrap(node)
    if _is_normal_jacobian(n):
        return [n]
    return [j for c in _iter(n) for j in _normal_jacobians(c)]
