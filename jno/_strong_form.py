"""Strong-form trace utilities shared by :func:`jno.fdm` and :func:`jno.fvm`.

Both read the same constraint list (``u = domain.unknown()``, ``u.bind(...)``, ``u(xb, yb) - g``,
``ur.d(nr) - h``, ``u(xi, yi) - u0``) and differ only in how a PDE term is discretised: collocated at the
mesh nodes (fdm) or integrated over control volumes (fvm). The helpers here walk that trace: they find the
unknowns, classify a constraint by its region and its derivatives, and rewrite time derivatives for the
method-of-lines split. Nothing in this module discretises.
"""

from __future__ import annotations

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np


def _decided(decide):
    """``decide()``, or ``None`` when it needs a concrete value that is traced (``float``/``bool`` of a tracer):
    a structural probe of a residual that closes over a coefficient ``jax.grad`` is differentiating by."""
    import jax

    try:
        return decide()
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerBoolConversionError):
        return None


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


def _refuse_inside(constraints, who):
    """``.inside`` reads a field in the cell next to a boundary face: only a finite-volume problem has one."""
    from .trace import InsideValue

    def walk(n):
        n = _unwrap(n)
        if isinstance(n, InsideValue):
            raise ValueError(
                f"{who}([...]): `.inside` reads a field in the cell next to a boundary face, which only a "
                "finite-volume problem has. Give the boundary value itself, or solve with jno.fvm([...])."
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
    if who != "jno.fvm":
        _refuse_inside(constraints, who)
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
    if not tags:
        # A raw expression with no bound view on top (`F.d(x) - jnn.stack(...)`, a steady vector equation of
        # stacked fluxes) carries no binding of its own: its region is the one its coordinates are sampled on.
        # Without it the equation had none, and an algebraic row beside it was taken for a pin.
        tags = {
            v.tag
            for v in _iter_variables(constraint)
            if getattr(v, "axis", "spatial") != "temporal" and v.tag is not None and not str(v.tag).startswith("n_")
        }
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
    from .trace import BinaryOp, FunctionCall, Hessian, InsideValue, Jacobian, Placeholder

    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, fn(node.left), fn(node.right))
    if isinstance(node, InsideValue):
        return InsideValue(fn(node.target))
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


class StrongFormProblem:
    """What :func:`jno.fdm` and :func:`jno.fvm` problems share: classifying the constraint list, the blocked
    DOF layout, and the parameter machinery that lets a solve be differentiated in ``jno.core`` (trainable
    parameters, data fields, networks, and the probes that decide structure for every parameter value).

    A subclass sets ``self.unknowns``, ``self.domain`` and ``self._N`` (DOFs per field component) and
    ``_who`` (the name in error messages), and implements the discretisation.
    """

    _who = "jno.strong_form"

    def _pde_region(self):
        """``(region_tag, geometry)`` of the named sub-region carried by the PDE's coordinate variables
        (from ``domain.region(name, poly)``), or ``(None, None)`` for a whole-domain problem."""
        src = getattr(self.domain, "_source_regions", {}) or {}
        tags = {
            v.tag
            for c in self._pde
            for v in (getattr(c, "_coord_vars", None) or {}).values()
            if getattr(v, "axis", None) != "temporal"
        }
        named = [t for t in tags if t in src]
        return (named[0], src[named[0]]) if len(named) == 1 else (None, None)

    def _referenced_models(self):
        """``{layer_id: module}`` of every model the constraints read, other than the unknowns: a known
        nodal field (a ``jno.np.parameter`` carrying data, no optimizer), a network. A data field used as a
        PDE coefficient used to fail with "No model for Model N": only trainable parameters were in the
        evaluation scope, and data fields only worked inside boundary values.

        The walk over the constraints is cached; the modules are read LIVE at every call. Caching the modules
        themselves kept a data field's old value after the documented eager swap
        ``K.model.module = eqx.tree_at(...)``: the repeat solve returned the old field's answer (measured:
        κ 1 → 2 left max u at 0.0734 instead of 0.0367), although the data fingerprint had changed."""
        models = self.__dict__.get("_models_cache")
        if models is None:
            from .trace import ModelCall

            models, seen, stack = {}, set(), [_unwrap(c) for c in self._constraints]
            while stack:
                n = stack.pop()
                if id(n) in seen:
                    continue
                seen.add(id(n))
                if isinstance(n, ModelCall) and all(n.model is not u for u in self.unknowns):
                    models[n.model.layer_id] = n.model
                stack.extend(_unwrap(c) for c in _iter(n))
            self._models_cache = models
        found = {}
        for lid, model in models.items():
            module = getattr(model, "module", None)
            if module is not None:
                found[lid] = self._nodal_module(self._with_initial_value(model, module))
        return found

    @staticmethod
    def _with_initial_value(model, module):
        """A parameter's ``.initialize(...)`` is lazy (the training loop applies it), so a stored value still at
        the zeros the parameter is born with read as 0 in an eager solve without an optimizer: ``-Δu - f`` with
        ``f.initialize(constant(1.0))`` solved with f = 0 and no error. Resolve the initializer the way
        ``jno.fem`` does; a value set by hand (not all zero) is kept."""
        value = getattr(module, "value", None)
        if (
            value is None
            or not getattr(model, "_is_parameter", False)
            or isinstance(value, jax.core.Tracer)
            or np.any(np.asarray(value))
        ):
            return module
        from .geometry.parametric import model_initial_value

        init = model_initial_value(model)
        weight = getattr(model, "_weight_tree", None)
        if init is None and weight is not None and not isinstance(weight, (str, os.PathLike)):
            init = weight
        if init is None:
            return module
        import equinox as eqx

        return eqx.tree_at(lambda m: m.value, module, jnp.reshape(jnp.asarray(init, value.dtype), jnp.shape(value)))

    def _nodal_module(self, module):
        """``module``, with a value of one entry per node (``(N,)``) shaped ``(N, 1)``: a per-node scalar field,
        laid out like every other one in the strong form. As ``(N,)`` it broadcast against an ``(N, 1)``
        derivative to ``(N, N)``. Applied to every module an evaluation reads, stored or injected, so a value
        evaluates the same whichever route it takes."""
        value = getattr(module, "value", None)
        if value is None or np.ndim(value) != 1 or np.shape(value)[0] != self._N:
            return module
        import equinox as eqx
        import jax

        with jax.ensure_compile_time_eval():  # a stored value stays concrete inside a trace: structural probes read it
            value = jnp.reshape(value, (self._N, 1))
        return eqx.tree_at(lambda m: m.value, module, value)

    def _params_scope(self, extra_params=None):
        """The module every trainable parameter resolves to in an evaluation: its current value, then the
        values a traced solve injects (:attr:`_override`, set while a crux-driven march is traced), then
        ``extra_params``. Every evaluator this solver builds reads it, so a parameter works wherever it is
        written — in the PDE, a boundary value, a flux condition, a time coefficient, an initial value."""
        scope = self._referenced_models()  # data fields and networks, at their current values
        scope.update({lid: n.model.module for lid, n in self._trainable_params().items()})
        scope.update(getattr(self, "_override", None) or {})
        scope.update(extra_params or {})
        # an injected per-node field takes the (N, 1) layout its stored value has: injected as (N,), a data field
        # in a Neumann row `∇u·n - g` broadcast the row to (N, N), lost its gradient part and made the solve
        # singular (NaN with lu)
        return {lid: self._nodal_module(m) for lid, m in scope.items()}

    def _current_params(self):
        """``{layer_id: module}`` of every trainable parameter at its current value."""
        return {lid: n.model.module for lid, n in self._trainable_params().items()}

    def _live_params(self):
        """Parameter values an evaluator applies when it is CALLED, over the ones it was built with: the
        probe values of :meth:`_perturbed_params` while a structural decision is made, else none."""
        return getattr(self, "_perturbation", None) or {}

    @contextlib.contextmanager
    def _perturbed_params(self):
        """While active, every trainable parameter reads a probe value ``(1 + a)·p + b`` near its current
        value ``p`` (``a, b`` drawn from [0.1, 0.2], fixed seed): a generic point that keeps the sign of a
        nonzero value, so a probe does not wander into a regime (a negative diffusivity) the problem never
        visits, while a zero moves off zero. A network's probe moves every weight the same way."""
        import equinox as eqx
        import jax

        rng = np.random.default_rng(11)

        def moved(x):
            p = np.asarray(x)
            a, b = rng.uniform(0.1, 0.2, p.shape), rng.uniform(0.1, 0.2, p.shape)
            return jnp.asarray((1.0 + a) * p + b, dtype=x.dtype)

        probe = {}
        for lid, n in self._trainable_params().items():
            m = n.model.module
            if not getattr(n.model, "_is_parameter", False):  # a network: every floating-point weight
                probe[lid] = jax.tree_util.tree_map(lambda x: moved(x) if eqx.is_inexact_array(x) else x, m)
                continue
            probe[lid] = eqx.tree_at(lambda mm: mm.value, m, moved(m.value))
        saved = getattr(self, "_perturbation", None)
        self._perturbation = probe
        try:
            yield
        finally:
            self._perturbation = saved

    def _for_every_parameter(self, decide):
        """``[decide(False)]``, plus ``decide(True)`` under :meth:`_perturbed_params` when the problem has
        trainable parameters. A structural decision (is the residual affine in u? its operator symmetric?
        constant in time?) must hold for every value a parameter takes in an inverse, not only the one it
        starts at. Measured: with k starting at 0, ``-Δu + k·u³ = f`` looked affine, and the solve returned
        the LINEAR solution at k = 1 (max error 5.0, true residual 1e4), with no error. ``decide(perturbed)``
        must build what it evaluates when ``perturbed`` is set, if that is jitted (a jitted function keeps
        its first trace); an un-jitted evaluator reads the probe values at call time."""
        out = [decide(False)]
        if self._trainable_params():
            with self._perturbed_params():
                out.append(decide(True))
        return out

    def _inject(self, dofs):
        """``{layer_id: module}`` with every unknown holding its slice of the blocked DOF vector — a scalar
        field its ``(N,)`` block, a vector field its ``(N, c)`` components."""
        import equinox as eqx

        out, N = {}, self._N
        self._require_real(dofs, "the state (an initial guess x0?)")
        for w, b0, c in zip(self.unknowns, self._block0, self._ncomp):
            sl = dofs[b0 * N : (b0 + c) * N]
            value = sl if c == 1 and np.ndim(w.module.value) == 1 else sl.reshape(c, N).T.reshape(w.module.value.shape)
            out[w.layer_id] = eqx.tree_at(lambda m: m.value, w.module, value.astype(w.module.value.dtype))
        return out

    def _as_blocks(self, value, ncomp):
        """An equation's value at every node, laid out component by component (``(N,)`` or ``(N, c)`` →
        ``(c·N,)``), checked against the number of components it has to drive."""
        v = jnp.asarray(value)
        v = jnp.broadcast_to(v.reshape(-1), (self._N,)) if v.size in (1, self._N) else v.reshape(self._N, -1)
        c = 1 if v.ndim == 1 else int(v.shape[1])
        if c != ncomp:
            raise ValueError(
                f"{self._who}([...]): an equation has {c} component(s) but drives an unknown with {ncomp}. A vector "
                "unknown (`domain.unknown(value_shape=(2,))`) needs a vector equation, and a scalar one a scalar."
            )
        return v if v.ndim == 1 else v.T.reshape(-1)

    def _blocks_of(self, k):
        """The DOF blocks of unknown ``k`` (one per component)."""
        return list(range(self._block0[k], self._block0[k] + self._ncomp[k]))

    def _split_components(self, k, vals, n_nodes):
        """A value for unknown ``k`` at ``n_nodes`` nodes, one ``(n_nodes,)`` array per component."""
        v = jnp.asarray(vals)
        c = self._ncomp[k]
        if c == 1:
            return [jnp.broadcast_to(v.reshape(-1), (n_nodes,)) if v.size in (1, n_nodes) else v.reshape(n_nodes)]
        v = jnp.broadcast_to(v, (n_nodes, c)) if v.size == c else v.reshape(n_nodes, -1)
        if v.shape[1] != c:
            raise ValueError(
                f"{self._who}([...]): a condition on a {c}-component unknown gives {v.shape[1]} value(s) per node. Give "
                "every component, e.g. `U(xb, yb) - jnn.stack([gx, gy], axis=-1)`."
            )
        return [v[:, j] for j in range(c)]

    def _start_time(self):
        window = getattr(self.domain, "time", None)
        return float(window[0]) if window is not None and self._transient else 0.0

    @staticmethod
    def _symmetry_probe(K):
        """``(wᵀKv, vᵀKw)`` for random ``v, w``: equal (to rounding) iff ``K`` is symmetric, almost surely."""
        rng = np.random.default_rng(2)
        v = jnp.asarray(rng.standard_normal(K.shape[0]))
        w = jnp.asarray(rng.standard_normal(K.shape[0]))
        return float(w @ (K @ v)), float(v @ (K @ w))

    def _is_symmetric(self, make_K, key=None):
        """Is the assembled operator ``make_K()`` symmetric, for any value of the parameters
        (:meth:`_for_every_parameter`)? Cached per ``key``."""
        cache = self.__dict__.setdefault("_symmetric_cache", {})
        if key is not None and key in cache:
            return cache[key]

        def symmetric(_):
            a, b = self._symmetry_probe(make_K())
            return abs(a - b) <= _rtol() * max(abs(a), abs(b), np.finfo(float).tiny)

        out = _decided(lambda: all(self._for_every_parameter(symmetric)))
        if out is None:  # a traced coefficient (jax.grad of the solve): not decidable here, so not symmetric
            return False
        if key is not None:
            cache[key] = out
        return out

    def _is_affine(self, key, make, n):
        """Is the function ``make()`` builds affine in the DOF vector? Its JVP must be the same at two
        different states. Decided once per problem, eagerly, for the parameters' current values and a probe
        value (:meth:`_for_every_parameter`)."""
        cache = self.__dict__.setdefault("_affine_cache", {})
        if key not in cache:
            import jax

            def affine(_):
                fun = make()
                rng = np.random.default_rng(1)
                v = jnp.asarray(rng.standard_normal(n))
                w = jnp.asarray(rng.standard_normal(n))
                j0 = jax.jvp(fun, (jnp.zeros(n),), (v,))[1]
                j1 = jax.jvp(fun, (w,), (v,))[1]
                return bool(jnp.linalg.norm(j1 - j0) <= _rtol(j0.dtype) * (float(jnp.linalg.norm(j0)) or 1.0))

            out = _decided(lambda: all(self._for_every_parameter(affine)))
            if out is None:
                # The residual closes over a traced value (a coefficient differentiated by jax.grad), so its JVPs
                # are traced and cannot be compared here. Not affine, then: Newton solves an affine residual in
                # one step, so the answer is the same, and nothing is cached for a later concrete call.
                return False
            cache[key] = out
        return cache[key]

    @staticmethod
    def _same(a, b):
        """Are two evaluations of the same quantity equal to floating-point accuracy (:func:`_rtol`), relative
        to their magnitude?"""
        a, b = jnp.asarray(a), jnp.asarray(b)
        # Compared inside JAX: the values can be tracers (a coefficient differentiated by jax.grad), whose
        # comparison resolves but whose magnitude does not (`float()` of a tracer raises).
        scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))), jnp.finfo(a.dtype).tiny)
        return bool(jnp.all(jnp.abs(a - b) <= _rtol(a.dtype) * scale))

    def _trainable_params(self):
        """**Trainable** models in the constraints — a ``jno.np.parameter`` or a ``jno.nn`` network with an
        attached optimizer (``.optimizer(...)``, i.e. ``model._opt_fn is not None``) that is not the unknown
        (and, for a network, not ``.freeze()``d): the inverse parameters (a source amplitude, a diffusivity,
        a network coefficient ``k(x)``, …). Their presence makes :meth:`solve` return a deferred
        ``crux``-drivable node. A parameter **without** an optimizer is *data* — a known nodal field (e.g. a
        neighbour's field in a coupled solve) — so it stays an eager solve and is gathered as a value by
        :meth:`_eval_g`; a network without one is a known coefficient, evaluated at its stored weights.
        Returns ``{layer_id: ModelCall}``.

        A network used to be left out: with its optimizer attached, the solve still ran eagerly at the
        stored weights and returned a constant array, so ``jno.core`` never trained it (no error)."""
        from .trace import ModelCall

        found = {}

        def walk(n):
            n = _unwrap(n)
            if (
                isinstance(n, ModelCall)
                and all(n.model is not u for u in self.unknowns)
                and getattr(n.model, "_opt_fn", None) is not None  # trainable ⇔ has an optimizer
                and (getattr(n.model, "_is_parameter", False) or not getattr(n.model, "_frozen", False))
            ):
                found[n.model.layer_id] = n
            for c in _iter(n):
                walk(c)

        for c in self._pde + self._dirichlet + self._neumann + self._ic:
            walk(c)
        return found

    def _record_fixed_networks(self):
        """Mark every network this solve bakes in at its stored weights (no optimizer yet, not frozen).

        Trainability is decided here, at ``.solve()``: an optimizer attached to such a network AFTERWARDS does
        not reach the array this solve returned, so training would silently leave the network alone. The mark
        makes that late ``.optimizer(...)`` raise (see ``Model.optimizer``)."""
        from .trace import ModelCall

        def walk(n):
            n = _unwrap(n)
            m = getattr(n, "model", None) if isinstance(n, ModelCall) else None
            if (
                m is not None
                and all(m is not u for u in self.unknowns)
                and not getattr(m, "_is_parameter", False)
                and not getattr(m, "_frozen", False)
                and getattr(m, "_opt_fn", None) is None
            ):
                m._jno_fixed_by_eager_solve = True
            for c in _iter(n):
                walk(c)

        for c in self._pde + self._dirichlet + self._neumann + self._ic:
            walk(c)

    @staticmethod
    def _trainable_arg(node):
        """What a trainable contributes to the deferred solve node's arguments: a parameter's own
        ``ModelCall`` (it evaluates to the parameter's current value), a network's ``ModelWeights`` (it
        evaluates to the network's live module -- a call would evaluate the network instead)."""
        from .trace import ModelWeights

        return node if getattr(node.model, "_is_parameter", False) else ModelWeights(node.model)

    @staticmethod
    def _trainable_module(node, value):
        """The module a trainable resolves to inside the solve, from the value its :meth:`_trainable_arg`
        evaluated to (possibly traced): the parameter's module holding ``value``, or the network's module
        itself."""
        if not getattr(node.model, "_is_parameter", False):
            return value
        import equinox as eqx

        module = node.model.module
        return eqx.tree_at(lambda m: m.value, module, jnp.asarray(value).astype(module.value.dtype))

    def _require_real(self, value, what):
        """``value``, unless it is complex while every unknown is real: a real solve would keep only its real
        part (JAX casts complex to real with a warning), which is a wrong answer, not a supported one."""
        if jnp.iscomplexobj(value) and not any(jnp.iscomplexobj(w.module.value) for w in self.unknowns):
            raise NotImplementedError(
                f"{self._who}: {what} is complex, but the unknown is real, and a real solve would keep only the real "
                "part. {self._who} has no complex fields yet: write the real and imaginary parts as two real unknowns "
                "(a coupled system)."
            )
        return value

    @staticmethod
    def _uses_params(node, params):
        from .trace import ModelCall

        n = _unwrap(node)
        if isinstance(n, ModelCall) and n.model.layer_id in params:
            return True
        return any(StrongFormProblem._uses_params(c, params) for c in _iter(n))

    def _is_boundary_derivative_condition(self, c):
        """A condition whose coordinates all live on boundary regions and which differentiates an unknown:
        a boundary row (``nx*ub.x + ny*ub.y - g``, ``ub.d((nx, ny)) - g``, a tangential ``ub.x - g``). It
        used to be read as a second PDE and summed into the first."""
        tags = _spatial_tags(c)
        boundary = set(getattr(self.domain, "_boundary_registry", {}) or {})
        return (
            bool(tags)
            and all(t in boundary or str(t).startswith("n_") for t in tags)
            and any(_has_unknown_derivative(c, u) for u in self.unknowns)
        )

    def _classify(self, constraints):
        """Sort the constraints into the PDE, Dirichlet, flux, initial and initial-velocity lists by their
        structure, record the periodic ties, and check the time order and the coupled-system layout. The
        same rules hold for every strong-form discretisation; only what is done with each list differs."""
        self._pde, self._dirichlet, self._neumann, self._ic, self._vel_ic = [], [], [], [], []
        self._periodic_ties = []  # the region pairs of every `u(A) - u(B)` periodic constraint
        # Classify by structure (not by which region tag), so a value-only pin works on ANY region —
        # a boundary edge OR a geometric sub-region (`domain.region(name, geom)`, used by coupled /
        # domain-decomposition solves to pin a subdomain's complement to a neighbour's field):
        #   * a periodic tie `u(A) - u(B)` (opposite faces)  → wrap the grid axis (check first);
        #   * a normal derivative `ui.d(n, ...)`           → a Neumann/Robin flux row;
        #   * `u.t` on the `initial` region                → the initial velocity (u_tt problems);
        #   * a derivative of the unknown (Laplacian, u.t) → the PDE residual;
        #   * the `initial` region, value-only            → the initial condition;
        #   * otherwise (value-only, affine in u)          → a Dirichlet pin on its region.
        kinds = []
        for c in constraints:
            tie = getattr(c, "_periodic_tie", None)
            if tie is not None:
                self._periodic_ties.append(tie)
                kinds.append(None)
            elif _normal_jacobian(c) is not None or self._is_boundary_derivative_condition(c):
                kinds.append("neumann")  # a boundary row: Neumann, Robin, oblique, nonlinear flux, …
            elif _region_tag(c) == "initial" and _has_temporal(c):
                kinds.append("vel_ic")
            elif any(_has_unknown_derivative(c, u) for u in self.unknowns):
                kinds.append("pde")
            elif _region_tag(c) == "initial":
                kinds.append("ic")
            else:
                kinds.append("dirichlet")
        # An ALGEBRAIC equation carries no derivative at all -- a constitutive law `v - u**2`, an
        # incompressibility-like closure, a fibre reading a state. Structure alone cannot tell it from a
        # value pin, since both are value-only; the REGION can. A pin is written on a boundary edge or a
        # sub-region, an equation on the same region as the differential equations it sits beside, so a
        # value-only constraint on a PDE region is one more equation (with a zero mass row: it constrains
        # no time derivative). Without this it fell through to `_dirichlet`, and a coupled system whose
        # second equation happened to be algebraic was rejected for having too few equations.
        #
        # The tags are compared, not hashed: a constraint spanning several regions carries a set of them.
        pde_regions = [_region_tag(c) for c, k in zip(constraints, kinds) if k == "pde"]
        kinds = [
            "pde" if k == "dirichlet" and any(_region_tag(c) == t for t in pde_regions) else k
            for c, k in zip(constraints, kinds)
        ]
        for c, k in zip(constraints, kinds):
            if k is not None:
                {
                    "pde": self._pde,
                    "neumann": self._neumann,
                    "vel_ic": self._vel_ic,
                    "ic": self._ic,
                    "dirichlet": self._dirichlet,
                }[k].append(c)
        if not self._pde:
            raise ValueError(f"{self._who}([...]): no PDE residual found (a term with a derivative of the unknown).")
        self._transient = bool(self._ic)
        pde_has_dt = any(_has_temporal(c) for c in self._pde)
        self._time_order = max(_temporal_order(c) for c in self._pde)
        if self._time_order > 2:
            raise NotImplementedError(
                f"{self._who}([...]): a time derivative of order {self._time_order} was found; only `u.t` "
                "(first order) and `u.tt` (second order) are supported."
            )
        if self._vel_ic and self._time_order != 2:
            raise ValueError(
                f"{self._who}([...]): an initial velocity `ui0.t - v0` was given, but the PDE has no `u.tt` "
                "term. An initial velocity only belongs to a second-order-in-time problem."
            )
        if self._vel_ic and not self._ic:
            raise ValueError(
                f"{self._who}([...]): an initial velocity `ui0.t - v0` was given without an initial displacement "
                "`u(xi, yi) - u0` — a second-order-in-time problem needs both (the velocity defaults to 0)."
            )
        if self._transient:
            if not (getattr(self.domain, "_is_time_dependent", False) and self.domain.time is not None):
                raise ValueError(
                    f"{self._who}([...]): an initial condition `u(initial) - u0` requires a time-dependent domain "
                    "— build it with `jno.domain(..., time=(t0, t1, n_steps))`."
                )
            if not pde_has_dt:
                raise ValueError(
                    f"{self._who}([...]): an initial condition was given but the PDE residual has no time derivative "
                    "`u.t` — add the `u.t` term (e.g. `ui.t - nu*(ui.d2(x) + ui.d2(y))`)."
                )
        elif pde_has_dt:
            raise ValueError(
                f"{self._who}([...]): the PDE residual has a time derivative `u.t` but no initial condition — "
                "add `u(xi, yi) - u0` (with `xi, yi = domain.variable('initial', split=True)`)."
            )
        if len(self.unknowns) > 1:  # coupled (multi-field)
            if len(self._pde) != len(self.unknowns):
                raise ValueError(
                    f"{self._who}([...]): a coupled system needs exactly one PDE equation per unknown — got "
                    f"{len(self.unknowns)} unknowns but {len(self._pde)} PDE equation(s). Author one equation per "
                    "field, in the order the unknowns are declared (equation k drives unknown k)."
                )
            for k, (eq, own) in enumerate(zip(self._pde, self.unknowns)):
                other = [j for j, w in enumerate(self.unknowns) if w is not own and _has_temporal_of(eq, w)]
                if other:
                    raise NotImplementedError(
                        f"{self._who}([...]): equation {k} of the coupled system carries the time derivative of "
                        f"unknown {other[0]}, not of its own unknown {k}. Equation k drives unknown k and may only "
                        "carry that unknown's `u.t` (a diagonal mass); reorder the equations, or solve for the "
                        "combination that appears differentiated."
                    )
