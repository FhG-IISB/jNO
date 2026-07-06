"""Finite-difference PDE solver — the strong-form sibling of :func:`jno.fem`.

Write the **strong-form residual** ``R(u)`` (``u`` is the nodal field) with the FD operators in this
module; :func:`fdm` folds in Dirichlet BCs and hands the residual straight to the ``jno.solve``
**nonlinear driver** — the *same* Newton–Krylov + implicit-``custom_root`` machinery :func:`jno.fem`
uses, so **linear and nonlinear** problems are handled uniformly (a linear residual converges in one
Newton step). Collocation at the mesh nodes means no quadrature, no test functions, no mass matrix
(so it is leaner than the weak-form assembler).

Because the residual is a plain differentiable function of the DOFs and the solve differentiates
through ``custom_root``, gradients to parameters inside it (a source, a coefficient field,
``jno.nn.wrap`` net) flow through — so ``jno.fdm`` composes into ``jno.core`` for inverse problems,
exactly like ``jno.fem.solve()``.

Two front-ends: a **fem-style constraint list** (preferred) — ``jno.fdm([residual, u(xb, yb) - g,
u(xi, yi) - u0])`` authored with ``u = domain.unknown()`` exactly as ``jno.fem([...])``, where the
initial condition is *found from the constraints* (never a config flag) and ``t_span``/step-count are
inferred from ``domain.time`` — and a low-level **function form** (:class:`FDMSystem`).

v1 scope: scalar, 2-D triangular mesh, and **any mix** of steady boundary conditions — Dirichlet
``u(region) - g``, and **any flux BC affine in the normal derivative** ``∂u/∂n`` written with that
edge's boundary tags: Neumann ``ur.d(n) - h``, Robin ``ur.d(n) + α(u - u∞)``, a coordinate-coefficient
``κ(x)·ur.d(n)``, either sign (``ur = u.bind(x=xr, y=yr)``, ``n = domain.variable(region, normals=True)``;
corner nodes fall back to the PDE residual). Plus **transient** problems by method-of-lines (``M = I``;
a unit-coefficient ``u.t`` term, e.g. ``ui.t - νΔu``) — all with linear + nonlinear residuals. Transient
flux BCs, periodic, a general ``u.t`` mass coefficient, and a structured-grid stencil backend are planned
extensions (see ``plans/fdm-solver.md``). A pure-Neumann problem (no Dirichlet node) is singular
(solution up to a constant) and is solved as-is.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import solve as _solve
from .differential_operators import DifferentialOperators as _D

__all__ = ["fdm", "laplacian", "gradient"]


def _mesh(domain):
    mc = domain.mesh_connectivity
    dim = int(getattr(domain, "dimension", 2))
    pts = jnp.asarray(np.asarray(mc["points"])[:, :dim])
    if dim != 2:
        raise NotImplementedError("jno.fdm: only 2-D triangular meshes are supported in v1.")
    return pts, jnp.asarray(mc["triangles"])


def laplacian(u, domain, method: str = "cotangent"):
    """FD Laplacian ``Δu`` of the nodal field ``u`` on the domain's mesh. ``method="cotangent"``
    (symmetric, accurate — CG-compatible) or ``"gradient_of_gradient"`` / ``"lsq_of_gradient"``."""
    pts, tris = _mesh(domain)
    return _D.compute_fd_laplacian_2d_simple(u, pts, tris, dims=(0, 1), method=method)


def gradient(u, domain, method: str = "area_weighted"):
    """FD gradient ``∇u`` of the nodal field ``u`` — shape ``(N, 2)``. ``method`` selects the stencil
    (``"area_weighted"`` default, ``"uniform"``, ``"inverse_distance"``, ``"least_squares"``)."""
    pts, tris = _mesh(domain)
    gx = _D.compute_fd_gradient_2d_simple(u, pts, tris, 0, method=method)
    gy = _D.compute_fd_gradient_2d_simple(u, pts, tris, 1, method=method)
    return jnp.stack([gx, gy], axis=1)


class FDMSystem:
    """A finite-difference discretization built from a strong-form residual + Dirichlet BCs.

    ``residual(u_dofs) -> R_at_nodes`` must be a differentiable function of the nodal DOF vector, zero
    at the PDE solution (e.g. ``lambda u: -jno.fdm.laplacian(u, d) - f``). ``dirichlet`` maps a
    boundary-region tag to a value (scalar, or a coordinate function ``g(x, y)``). For a time-dependent
    domain (``time=(t0, t1, n)``), ``.solve(u0)`` marches in time from the initial state ``u0`` — the
    IC is explicit data to the march, never a constructor config flag."""

    def __init__(self, domain, residual, dirichlet=None):
        self.domain = domain
        self.residual = residual
        self.dirichlet = dict(dirichlet or {})
        self._N = int(np.asarray(domain.mesh_connectivity["points"]).shape[0])

    @property
    def points(self):
        """DOF (mesh-node) coordinates, shape ``(N, dim)``."""
        pts, _ = _mesh(self.domain)
        return pts

    def _region_nodes(self, tag):
        reg = getattr(self.domain, "_boundary_registry", {}).get(tag)
        if reg is not None and "point_indices" in reg and len(reg["point_indices"]) > 0:
            return np.asarray(reg["point_indices"], dtype=int)
        return np.asarray(self.domain.mesh_connectivity["boundary_indices"], dtype=int)  # whole boundary

    def _dirichlet_mask_values(self):
        pts = np.asarray(self.points)
        mask = np.zeros(self._N, dtype=bool)
        vals = np.zeros(self._N)
        for tag, g in self.dirichlet.items():
            idx = self._region_nodes(tag)
            mask[idx] = True
            vals[idx] = g(*[pts[idx, k] for k in range(pts.shape[1])]) if callable(g) else float(g)
        return jnp.asarray(mask), jnp.asarray(vals)

    def solve(self, nonlinear=None, x0=None):
        """Solve the steady system, reusing the SAME ``jno.solve`` Newton–Krylov + implicit-``custom_root``
        machinery ``jno.fem`` uses (linear and nonlinear uniformly; a linear residual converges in one
        step; fully matrix-free). Dirichlet is folded into the residual (boundary rows become ``u - g``).
        Gradients to parameters in ``residual`` flow through ``custom_root``, so ``jno.fdm`` composes
        into ``jno.core`` for inverse problems.

        Transient problems march with :meth:`solve_transient`; the fem-style constraint-list authoring
        (``u(initial) - u0`` as a constraint) is the planned high-level path (see ``plans/fdm-solver.md``).
        """
        bmask, bvals = self._dirichlet_mask_values()

        def residual_with_bc(u):
            return jnp.where(bmask, u - bvals, self.residual(u))  # Dirichlet rows: u - g

        driver = nonlinear or _solve.newton()
        u0 = jnp.zeros(self._N) if x0 is None else jnp.asarray(x0)
        return driver(residual_with_bc, u0)

    def solve_transient(self, u0, t_span, nsteps, *, save_ts=None):
        """Explicit method-of-lines march (when you don't drive time through ``domain.time``). See
        :meth:`solve`; ``dt = (t1 - t0) / nsteps``."""
        t0, t1 = float(t_span[0]), float(t_span[1])
        return self._march(jnp.asarray(u0), t0, t1, (t1 - t0) / int(nsteps), save_ts)

    def _march(self, u0, t0, t1, dt, save_ts):
        """Method-of-lines transient of ``u̇ = 𝒩(u) = -residual(u)`` (the SAME residual serves steady
        and transient). Reuses jNO's solver-agnostic :class:`SemidiscreteTimeBlock` + backward-Euler
        ``lax.scan`` integrator — no new time-stepping code, and ``custom_root`` differentiable.
        Collocation gives ``M = I`` (leaner than FEM); Dirichlet nodes carry a zero mass row so they
        stay pinned at ``g``. Autonomous residuals only in v1 (no explicit ``t`` / time-varying source)."""
        import jax.experimental.sparse as jsparse

        from .utils.solver.backend_blocks import (
            SemidiscreteTimeBlock,
            _block_time_grid,
            _default_transient_integrate,
        )

        bmask, bvals = self._dirichlet_mask_values()
        diag = jnp.stack([jnp.arange(self._N), jnp.arange(self._N)], axis=1)
        M = jsparse.BCOO((jnp.where(bmask, 0.0, 1.0), diag), shape=(self._N, self._N))  # 0 mass on Dirichlet rows

        def residual(wn, t, args):  # M u̇ + residual = 0  →  interior u̇ = -R = 𝒩;  boundary wn = g
            return jnp.where(bmask, wn - bvals, self.residual(wn))

        block = SemidiscreteTimeBlock(
            mass=lambda t, args: M, residual=residual, state0=jnp.asarray(u0), t0=float(t0), t1=float(t1), dt=float(dt)
        )
        ts = _block_time_grid(block) if save_ts is None else jnp.asarray(save_ts)
        return _default_transient_integrate(block, {}, ts)


# ============================================================================
# Trace-based constraint-list front-end — authored like jno.fem, with u = domain.unknown()
# ============================================================================


def _unwrap(node):
    return getattr(node, "_expr", node)  # view -> underlying Placeholder


def _iter(node):
    from .utils.solver.solver_helper import iter_children

    return iter_children(node) or ()


def _find_unknown(constraints):
    """The single ``domain.unknown()`` field (a nodal-field-parameter ModelCall's Model) in the list."""
    from .trace import ModelCall

    models = {}

    def walk(n):
        n = _unwrap(n)
        if isinstance(n, ModelCall) and getattr(n.model, "_fem_field", None) == "node":
            models[n.model.layer_id] = n.model
        for c in _iter(n):
            walk(c)

    for c in constraints:
        walk(c)
    if len(models) != 1:
        raise ValueError(
            f"jno.fdm([...]): expected exactly one domain.unknown() field in the constraints, found "
            f"{len(models)}. Author the strong form with a single `u = domain.unknown()`."
        )
    return next(iter(models.values()))


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


def _zero_temporal(node):
    """Strong-form method-of-lines split: drop the ``u.t`` (:class:`TemporalDerivative`) terms, leaving
    the **spatial** residual ``R_spatial``. The semidiscrete form ``M u̇ + R_spatial = 0`` (collocation
    ⇒ ``M = I``) then reads ``u̇ = -R_spatial`` — e.g. ``u.t - νΔu`` ⟶ ``-νΔu`` ⟹ ``u̇ = νΔu``. This
    mirrors fem's :func:`_strip_temporal_trial_derivative` (which folds ``d/dt(u)·φ`` into the mass
    operator); here ``M = I`` carries ``u̇`` so the term is dropped outright. v1 assumes a **unit
    coefficient** on ``u.t`` (the standard ``u.t - 𝒩(u)`` form); a general mass coefficient is future work."""
    from .trace import BinaryOp, FunctionCall, Hessian, Jacobian, Literal, Placeholder, TemporalDerivative

    if isinstance(node, TemporalDerivative):
        return Literal(0.0)
    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, _zero_temporal(node.left), _zero_temporal(node.right))
    if isinstance(node, FunctionCall):
        return node.copy_with_args([_zero_temporal(a) if isinstance(a, Placeholder) else a for a in node.args])
    if isinstance(node, Jacobian):
        return Jacobian(_zero_temporal(node.target), node.variables, node.scheme)
    if isinstance(node, Hessian):
        return Hessian(_zero_temporal(node.target), node.variables, node.scheme, node.trace)
    return node


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


def _set_normal(node, val):
    """Replace the normal-derivative node ``ui.d(n, ...)`` with the constant ``val``, leaving the rest of
    the constraint intact. Evaluating the result at ``val = 0`` and ``val = 1`` recovers, for a condition
    ``F(∂u/∂n) = a·∂u/∂n + b`` affine in the flux, the intercept ``b = F(0)`` and slope ``a = F(1) - F(0)``
    — so **any** flux BC (Neumann ``∂u/∂n - h``, Robin ``∂u/∂n + α(u - u∞)``, a coordinate-coefficient
    ``κ(x)·∂u/∂n``, either sign) is handled by ``a·(∇u·n) + b`` without parsing its structure."""
    from .trace import BinaryOp, FunctionCall, Hessian, Jacobian, Literal, Placeholder

    if _is_normal_jacobian(node):
        return Literal(float(val))
    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, _set_normal(node.left, val), _set_normal(node.right, val))
    if isinstance(node, FunctionCall):
        return node.copy_with_args([_set_normal(a, val) if isinstance(a, Placeholder) else a for a in node.args])
    if isinstance(node, Jacobian):
        return Jacobian(_set_normal(node.target, val), node.variables, node.scheme)
    if isinstance(node, Hessian):
        return Hessian(_set_normal(node.target, val), node.variables, node.scheme, node.trace)
    return node


class _TraceFDM:
    """Finite-difference system authored as a fem-style constraint list with ``u = domain.unknown()``:
    ``jno.fdm([-u.d2(x) - u.d2(y) - f, u(xb, yb) - g]).solve()``. Constraints are classified by the
    region their coordinate variables carry — the ``interior`` → the strong-form PDE residual, a
    boundary tag → a Dirichlet condition ``u(region) - g``, the ``initial`` region → the initial
    condition ``u(initial) - u0`` (exactly as in :func:`jno.fem`), and a **flux condition** carrying a
    normal derivative ``ur.d(n)`` (``n = domain.variable(region, normals=True)``, the field bound to the
    edge's tags ``ur = u.bind(x=xr, y=yr)``) → a boundary row at that edge's nodes. Any condition
    **affine in** ``∂u/∂n`` is handled — Neumann ``ur.d(n) - h``, Robin ``ur.d(n) + α(u - u∞)``, a
    coordinate coefficient ``κ(x)·ur.d(n)`` — by writing the row as ``a·(∇u·n) + b`` with the two-probe
    coefficients ``a = F(1) - F(0)``, ``b = F(0)`` (:meth:`_flux_value_fn`); no structural parsing, so
    the whole edge equation is written with that edge's boundary tags. (Flux BCs are authored differently
    from :func:`jno.fem`, where a Neumann is a *natural* weak term ``h·v`` — the strong form has no test
    function, so the flux is imposed directly.) A problem is **transient** iff it carries an initial
    condition; ``t_span`` and the step
    count are then inferred from ``domain.time`` (never passed as args) and the system marches with the
    same method-of-lines stepper :func:`jno.fem` uses. The one config that stays on the object it
    describes is the FD **stencil** per operator (``u.d2(x, scheme=...)``, ``ui.d(n, scheme=...)``)."""

    def __init__(self, constraints):
        self.unknown = _find_unknown(constraints)
        self.domain = self.unknown._fem_field_domain
        self._N = int(np.asarray(self.domain.mesh_connectivity["points"]).shape[0])
        self._pts = jnp.asarray(np.asarray(self.domain.mesh_connectivity["points"])[:, : self.domain.dimension])
        self._pde, self._dirichlet, self._neumann, self._ic = [], [], [], []
        boundary_tags = set(getattr(self.domain, "_boundary_registry", {}).keys()) | {"boundary"}
        for c in constraints:
            # A normal derivative `ui.d(n, ...)` marks a Neumann/Robin flux row — check it FIRST, because
            # the field is bound to the interior coords (so `_region_tag` is ambiguous); the region comes
            # from the normal variable's tag instead (`n_right` → `right`), via `_neumann_region`.
            if _normal_jacobian(c) is not None:
                self._neumann.append(c)
                continue
            tag = _region_tag(c)
            if isinstance(tag, str) and tag == "initial":
                self._ic.append(c)
            elif isinstance(tag, str) and tag in boundary_tags:
                self._dirichlet.append(c)
            else:
                self._pde.append(c)
        if not self._pde:
            raise ValueError("jno.fdm([...]): no interior PDE residual found (only boundary/initial conditions).")
        self._transient = bool(self._ic)
        pde_has_dt = any(_has_temporal(c) for c in self._pde)
        if self._transient:
            if not (getattr(self.domain, "_is_time_dependent", False) and self.domain.time is not None):
                raise ValueError(
                    "jno.fdm([...]): an initial condition `u(initial) - u0` requires a time-dependent domain "
                    "— build it with `jno.domain(..., time=(t0, t1, n_steps))`."
                )
            if not pde_has_dt:
                raise ValueError(
                    "jno.fdm([...]): an initial condition was given but the PDE residual has no time derivative "
                    "`u.t` — add the `u.t` term (e.g. `ui.t - nu*(ui.d2(x) + ui.d2(y))`)."
                )
        elif pde_has_dt:
            raise ValueError(
                "jno.fdm([...]): the PDE residual has a time derivative `u.t` but no initial condition — "
                "add `u(xi, yi) - u0` (with `xi, yi = domain.variable('initial', split=True)`)."
            )
        if self._neumann and self._transient:
            raise ValueError(
                "jno.fdm([...]): Neumann/Robin flux conditions on a transient problem are not supported in v1 "
                "(a flux node keeps its mass row, unlike a pinned Dirichlet node). Use a steady problem, or "
                "impose the flux as a Dirichlet condition."
            )

    def _region_nodes(self, tag):
        if tag == "initial":
            return np.arange(self._N, dtype=int)  # the IC is the whole spatial field at t=t0
        reg = getattr(self.domain, "_boundary_registry", {}).get(tag)
        if reg is not None and len(reg.get("point_indices", [])) > 0:
            return np.asarray(reg["point_indices"], dtype=int)
        return np.asarray(self.domain.mesh_connectivity["boundary_indices"], dtype=int)

    def _pde_residual_fn(self, *, spatial=False, extra_params=None):
        """Differentiable residual over the nodal DOF vector, collocated at the mesh nodes. With
        ``spatial=True`` the ``u.t`` terms are dropped (:func:`_zero_temporal`) to give the
        method-of-lines spatial residual ``R_spatial`` for the semidiscrete march. ``extra_params``
        (``{layer_id: module}``) injects the current value of any **trainable** ``jno.np.parameter`` in
        the residual — how a ``crux``-driven inverse reaches the solve (see :meth:`_parametric_node`)."""
        import equinox as eqx

        from .trace_evaluator import TraceEvaluator

        expr = self._pde[0]
        for c in self._pde[1:]:
            expr = expr + c
        expr = _unwrap(expr)
        if spatial:
            expr = _zero_temporal(expr)
        spatial_tags = {  # collocate every spatial term at the mesh nodes (temporal tags carry no field)
            v.tag
            for c in self._pde
            for v in (getattr(c, "_coord_vars", None) or {}).values()
            if getattr(v, "axis", None) != "temporal"
        }
        context = {t: self._pts for t in spatial_tags}
        lid, base = self.unknown.layer_id, self.unknown.module

        def residual_fn(dofs):
            mod = eqx.tree_at(lambda m: m.value, base, jnp.asarray(dofs).astype(base.value.dtype))
            params = {lid: mod, **(extra_params or {})}
            ev = TraceEvaluator(params=params)
            return jnp.asarray(ev.evaluate(expr, context=context, var_bindings={})).reshape(-1)

        return residual_fn

    def _trainable_params(self):
        """**Trainable** ``jno.np.parameter`` fields in the constraints — a parameter with an attached
        optimizer (``.optimizer(...)``, i.e. ``model._opt_fn is not None``) that is not the unknown: the
        inverse parameters (a source amplitude, a diffusivity, …). Their presence makes :meth:`solve`
        return a deferred ``crux``-drivable node. A parameter **without** an optimizer is *data* — a
        known nodal field (e.g. a neighbour's field in a coupled solve) — so it stays an eager solve and
        is gathered as a value by :meth:`_eval_g`. Returns ``{layer_id: ModelCall}``."""
        from .trace import ModelCall

        found = {}

        def walk(n):
            n = _unwrap(n)
            if (
                isinstance(n, ModelCall)
                and getattr(n.model, "_is_parameter", False)
                and n.model is not self.unknown
                and getattr(n.model, "_opt_fn", None) is not None  # trainable ⇔ has an optimizer
            ):
                found[n.model.layer_id] = n
            for c in _iter(n):
                walk(c)

        for c in self._pde + self._dirichlet + self._neumann + self._ic:
            walk(c)
        return found

    def _eval_g(self, g_node, idx):
        """Value ``g`` at the nodes ``idx`` — a constant, a coordinate expression, or a **known nodal
        field** (a ``jno.np.parameter`` / ``domain.unknown()`` carrying data, e.g. a neighbour's current
        field in a coupled solve), in which case its per-node values are gathered at ``idx``."""
        from ._fem import _eval_value_node_at
        from .trace import ModelCall

        idx = np.asarray(idx, dtype=int)
        if isinstance(g_node, (int, float)):
            return jnp.full((idx.shape[0],), float(g_node))
        inner = _unwrap(g_node)
        if isinstance(inner, ModelCall) and getattr(inner.model, "_is_parameter", False):
            return jnp.asarray(inner.model.module.value).reshape(-1)[jnp.asarray(idx)]  # nodal data → gather
        return jnp.asarray(_eval_value_node_at(g_node, np.asarray(self._pts)[idx])).reshape(-1)

    def _condition_value(self, constraint, idx):
        """Value ``g`` of an affine condition ``u(region) - g`` (Dirichlet or IC), evaluated at the
        region's nodes ``idx`` — reused for both boundary conditions and the initial state."""
        inner = _unwrap(constraint)
        g_node = 0.0
        if getattr(inner, "op", None) == "-":  # u(region) - g  →  g is the side without the unknown
            g_node = inner.right if _contains_unknown(inner.left, self.unknown) else inner.left
        return self._eval_g(g_node, idx)

    def _dirichlet_rows(self):
        rows = []
        for c in self._dirichlet:
            idx = self._region_nodes(_region_tag(c))
            rows.append((jnp.asarray(idx), self._condition_value(c, idx)))
        return rows

    def _node_normals(self, region):
        """Unit outward normals for the **smooth-edge** nodes of ``region``, aligned to those nodes.

        Computed from the mesh **boundary segments** (``mesh_connectivity["boundary_edges"]``): each
        segment's exact perpendicular is averaged over the (two) segments meeting at a node, so an
        axis-aligned edge yields an exact ``(±1, 0)`` / ``(0, ±1)`` — much cleaner than the domain's
        smoothed per-point normal, which bleeds a tangential component near corners and would spoil the
        flux. Outward orientation is taken (sign only) from ``domain.variable(region, normals=True)``.

        A **corner** node averages two differently-oriented segment normals, so the averaged magnitude
        drops (≈0.71 at a right angle): the outward normal is undefined there, so corners are **dropped**
        from the flux row and keep their interior PDE residual (give a corner an explicit Dirichlet
        condition if it needs one). Returns ``(kept_node_indices, unit_n)``."""
        dim = self.domain.dimension
        pts = np.asarray(self._pts)
        edges = np.asarray(self.domain.mesh_connectivity["boundary_edges"], dtype=int)  # (E, 2) node pairs
        tang = pts[edges[:, 1]] - pts[edges[:, 0]]
        seg_n = np.stack([tang[:, 1], -tang[:, 0]], axis=1)  # 2-D perpendicular of each segment
        seg_n /= np.linalg.norm(seg_n, axis=1, keepdims=True) + 1e-30
        node_n = np.zeros((self._N, dim))
        cnt = np.zeros(self._N)
        for e, (i, j) in enumerate(edges):
            node_n[i] += seg_n[e]
            node_n[j] += seg_n[e]
            cnt[i] += 1
            cnt[j] += 1
        node_n /= np.maximum(cnt, 1)[:, None]  # average incident segment normals (unit on a flat edge)

        self.domain.variable(region, normals=True, split=True)  # domain's (oriented) normals for the sign
        bpts = np.asarray(self.domain.context[region]).reshape(-1, dim)
        dom_n = np.asarray(self.domain.context[f"n_{region}"]).reshape(-1, dim)
        idx = np.asarray(self._region_nodes(region), dtype=int)
        order = [int(np.argmin(np.sum((bpts - p) ** 2, axis=1))) for p in pts[idx]]
        raw = node_n[idx]
        flip = np.sum(raw * dom_n[order], axis=1) < 0  # orient outward to match the domain normal
        raw[flip] *= -1
        smooth = np.linalg.norm(raw, axis=1) > 0.9  # a corner has averaged magnitude ≈0.71 ≪ 1 → drop it
        idx, raw = idx[smooth], raw[smooth]
        n = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-30)
        return idx, jnp.asarray(n)

    def _flux_value_fn(self, constraint, val, extra_params=None):
        """Evaluate the flux constraint over ALL nodes with the normal derivative ``∂u/∂n`` pinned to the
        constant ``val`` (:func:`_set_normal`) — everything else (the field value ``u``, ``α``, ``u∞``,
        coordinate coefficients) evaluates normally against the nodal DOFs. Two such evaluations
        (``val = 0`` and ``val = 1``) give the affine decomposition of the boundary condition in the flux.
        ``extra_params`` injects trainable-parameter values, as in :meth:`_pde_residual_fn`."""
        import equinox as eqx

        from .trace_evaluator import TraceEvaluator

        expr = _set_normal(_unwrap(constraint), val)
        spatial_tags = {  # every spatial term collocates at the mesh nodes; the normal tag is gone now
            v.tag
            for v in (getattr(constraint, "_coord_vars", None) or {}).values()
            if getattr(v, "axis", None) != "temporal" and not str(getattr(v, "tag", "")).startswith("n_")
        }
        context = {t: self._pts for t in spatial_tags}
        lid, base = self.unknown.layer_id, self.unknown.module

        def value_fn(dofs):
            mod = eqx.tree_at(lambda m: m.value, base, jnp.asarray(dofs).astype(base.value.dtype))
            ev = TraceEvaluator(params={lid: mod, **(extra_params or {})})
            out = jnp.asarray(ev.evaluate(expr, context=context, var_bindings={})).reshape(-1)
            return jnp.broadcast_to(out, (self._N,)) if out.shape[0] == 1 else out  # a constant `-h` → per-node

        return value_fn

    def _flux_rows(self, extra_params=None):
        """Rows for **any** flux boundary condition affine in ``∂u/∂n`` — Neumann ``ui.d(n) - h``, Robin
        ``ui.d(n) + α(u - u∞)``, a coordinate-coefficient ``κ(x)·ui.d(n)``, either sign. Writes the whole
        edge equation with that edge's boundary tags (``xr, yr, nr = domain.variable(region, ...)``). Per
        row: node indices, unit normals, the FD stencil, and the two-probe value functions ``F(0)`` and
        ``F(1)`` (see :meth:`_flux_value_fn`) — the residual is ``(F(1) - F(0))·(∇u·n) + F(0)``. A
        condition that is **not** affine in ``∂u/∂n`` (a third probe ``F(2)`` disagrees) raises."""
        rows = []
        probe = jnp.zeros(self._N)
        for c in self._neumann:
            jac = _normal_jacobian(c)
            nvar = next(v for v in jac.variables if str(getattr(v, "tag", "")).startswith("n_"))
            region = nvar.tag[len("n_") :]  # `n_right` → `right`
            idx, nrm = self._node_normals(region)
            _, grad_method, _ = _D.parse_fd_scheme(getattr(jac, "scheme", "finite_difference"))
            v0, v1 = self._flux_value_fn(c, 0.0, extra_params), self._flux_value_fn(c, 1.0, extra_params)
            f0, f1, f2 = v0(probe), v1(probe), self._flux_value_fn(c, 2.0, extra_params)(probe)
            if not bool(jnp.allclose(f2 - f0, 2.0 * (f1 - f0), atol=1e-6)):
                raise ValueError(
                    "jno.fdm([...]): a flux boundary condition must be affine in the normal derivative "
                    "∂u/∂n — e.g. Neumann `ui.d(n) - h` or Robin `ui.d(n) + α*(u - u∞)`. A condition "
                    "nonlinear in ∂u/∂n is not supported."
                )
            rows.append((jnp.asarray(idx), nrm, grad_method or "area_weighted", v0, v1))
        return rows

    def _initial_state(self):
        """Initial nodal state ``u0`` (shape ``(N,)``) from the ``u(initial) - u0`` condition(s), the
        same way :func:`jno.fem` reads its IC — the IC is data found from the constraints, never a flag."""
        u0 = jnp.zeros(self._N)
        allnodes = np.arange(self._N, dtype=int)
        for c in self._ic:
            idx = self._region_nodes(_region_tag(c))  # "initial" → all nodes
            vals = self._condition_value(c, idx)
            u0 = u0.at[jnp.asarray(idx if len(idx) else allnodes)].set(vals)
        return u0

    def solve(self, nonlinear=None, x0=None):
        """Solve the strong-form system. **Steady** problems fold the Dirichlet rows into the residual
        (``u - g`` on the region) and hand it to the same ``jno.solve`` Newton–Krylov + ``custom_root``
        machinery ``jno.fem`` uses (linear/nonlinear uniform, differentiable for inverse problems).
        **Transient** problems (an ``u(initial) - u0`` condition is present) march by method-of-lines —
        ``t_span`` and the step count come from ``domain.time`` and the initial state from the IC — and
        return the trajectory (``(n_save, N)``); ``x0`` is rejected (the IC owns the initial state)."""
        if self._transient:
            if x0 is not None:
                raise ValueError("jno.fdm([...]): x0= is rejected for a transient problem — the IC owns the state.")
            return self._march(nonlinear=nonlinear)

        trainable = self._trainable_params()
        if trainable:
            return self._parametric_node(trainable, nonlinear=nonlinear, x0=x0)
        return self._steady_solve(nonlinear=nonlinear, x0=x0)

    def _steady_solve(self, *, nonlinear=None, x0=None, extra_params=None):
        """The eager steady solve: fold the flux and Dirichlet rows into the residual and hand it to the
        ``jno.solve`` Newton–Krylov driver. ``extra_params`` carries the current values of any trainable
        ``jno.np.parameter`` (from :meth:`_parametric_node`); it is empty for a plain forward solve."""
        residual_fn = self._pde_residual_fn(extra_params=extra_params)
        rows = self._dirichlet_rows()
        flux_rows = self._flux_rows(extra_params)

        def residual_with_bc(u):
            r = residual_fn(u)
            # Flux rows first (`a·(∇u·n) + b`, with a = F(1)-F(0), b = F(0) — Neumann/Robin/etc.), then
            # Dirichlet: a node shared by both (a corner touching a Dirichlet edge) resolves to the
            # essential Dirichlet value.
            for idx, nrm, method, v0, v1 in flux_rows:
                grad = gradient(u, self.domain, method=method)  # (N, 2), differentiable
                flux = jnp.sum(grad[idx] * nrm, axis=1)  # ∇u·n at the edge nodes
                b = v0(u)
                a = v1(u) - b
                r = r.at[idx].set(a[idx] * flux + b[idx])
            for idx, gvals in rows:
                r = r.at[idx].set(u[idx] - gvals)
            return r

        driver = nonlinear or _solve.newton()
        u0 = jnp.zeros(self._N) if x0 is None else jnp.asarray(x0)
        return driver(residual_with_bc, u0)

    def _parametric_node(self, trainable, *, nonlinear=None, x0=None):
        """When the constraints carry a trainable ``jno.np.parameter`` (an inverse parameter), return the
        solve as a **trace node** instead of an array — exactly as ``fem.solve()`` does — so it composes
        into ``jno.core``: ``jno.core([(jno.fdm([...]).solve() - u_obs).mse])`` with the parameter's
        attached optimizer recovers it. At each ``crux`` step the parameter node resolves to its current
        value, the solve re-runs (differentiably, through ``custom_root``), and the gradient flows back."""
        import equinox as eqx

        from .trace import FunctionCall

        lids = list(trainable)
        param_nodes = [trainable[lid] for lid in lids]  # the parameter ModelCalls -> FunctionCall args
        modules = {lid: trainable[lid].model.module for lid in lids}

        def _solve(*values):  # values = the parameters' current (crux-trained) values
            extra = {
                lid: eqx.tree_at(lambda m: m.value, modules[lid], jnp.asarray(v).astype(modules[lid].value.dtype))
                for lid, v in zip(lids, values)
            }
            return self._steady_solve(nonlinear=nonlinear, x0=x0, extra_params=extra)

        node = FunctionCall(_solve, param_nodes, name="fdm_solve")
        node._domain = self.domain  # so jno.core infers the domain from the graph (no explicit domain= needed)
        return node

    def _march(self, *, nonlinear=None, save_ts=None):
        """Method-of-lines march of ``u̇ = -R_spatial(u)`` reusing jNO's solver-agnostic
        :class:`SemidiscreteTimeBlock` + backward-Euler ``lax.scan`` integrator (no new time-stepping
        code, ``custom_root`` differentiable). ``M = I`` on interior nodes; Dirichlet nodes carry a zero
        mass row so they stay pinned at ``g``. ``t_span``/``dt`` are inferred from ``domain.time``."""
        import jax.experimental.sparse as jsparse

        from .utils.solver.backend_blocks import (
            SemidiscreteTimeBlock,
            _block_time_grid,
            _default_transient_integrate,
        )
        from .utils.solver.time_route import _infer_time_window

        t0, t1, dt = _infer_time_window(self.domain)
        if dt is None:
            raise ValueError("jno.fdm([...]): domain.time must specify n_steps >= 2 for a transient march.")

        rows = self._dirichlet_rows()
        bmask = np.zeros(self._N, dtype=bool)
        bvals = np.zeros(self._N)
        for idx, gv in rows:
            bmask[np.asarray(idx)] = True
            bvals[np.asarray(idx)] = np.asarray(gv)
        bmask, bvals = jnp.asarray(bmask), jnp.asarray(bvals)

        spatial_res = self._pde_residual_fn(spatial=True)
        diag = jnp.stack([jnp.arange(self._N), jnp.arange(self._N)], axis=1)
        M = jsparse.BCOO((jnp.where(bmask, 0.0, 1.0), diag), shape=(self._N, self._N))  # 0 mass on Dirichlet rows

        def residual(wn, t, args):  # M u̇ + R_spatial = 0  →  interior u̇ = -R_spatial;  boundary wn = g
            return jnp.where(bmask, wn - bvals, spatial_res(wn))

        block = SemidiscreteTimeBlock(
            mass=lambda t, args: M,
            residual=residual,
            state0=self._initial_state(),
            t0=float(t0),
            t1=float(t1),
            dt=float(dt),
        )
        ts = _block_time_grid(block) if save_ts is None else jnp.asarray(save_ts)
        return _default_transient_integrate(block, {}, ts)


def fdm(constraints_or_domain, residual=None, dirichlet=None):
    """Finite-difference PDE solver — the strong-form sibling of :func:`jno.fem`.

    **Constraint-list form** (fem-style, preferred): ``jno.fdm([residual, u(xb, yb) - g])`` authored with
    ``u = domain.unknown()`` and strong-form derivatives (``u.d2(x, scheme=...)``). Constraints are
    classified by region (interior → PDE residual, boundary → Dirichlet).

    **Function form** (low-level): ``jno.fdm(domain, residual=lambda u: ..., dirichlet={region: g})`` —
    a plain differentiable residual over the nodal DOF vector; see :class:`FDMSystem`."""
    if isinstance(constraints_or_domain, (list, tuple)):
        return _TraceFDM(list(constraints_or_domain))
    return FDMSystem(constraints_or_domain, residual, dirichlet)


fdm.laplacian = laplacian  # convenience: jno.fdm.laplacian(u, domain)
fdm.gradient = gradient
