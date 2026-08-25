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
inferred from ``domain.time``.

Scope: scalar fields — or a **coupled system** of several ``domain.unknown()`` fields (steady + Dirichlet;
one PDE equation per unknown, equation *k* driving unknown *k*, ``.solve()`` returning ``(nf, N)``) — on a
**2-D triangular or 3-D tetrahedral mesh**. The interior operators
(``jno.fdm.laplacian`` / ``jno.fdm.gradient``, and the constraint-list ``u.d2(x)+u.d2(y)+u.d2(z)``
authoring) dispatch on ``domain.dimension``; the default ``cotangent`` Laplacian is the cotangent-weight
operator in 2-D and its exact analogue, the **P1 tetrahedral finite-element** Laplace-Beltrami operator,
in 3-D (symmetric, second-order for the solve; ``gradient_of_gradient`` is the first-order local
alternative). **Any mix** of boundary conditions — Dirichlet ``u(region) - g``, and **any flux BC
affine in the normal derivative** ``∂u/∂n`` written with that region's boundary tags: Neumann
``ur.d(n) - h``, Robin ``ur.d(n) + α(u - u∞)``, a coordinate-coefficient ``κ(x)·ur.d(n)``, either sign
(``ur = u.bind(x=xr, y=yr[, z=zr])``, ``n = domain.variable(region, normals=True)``). Flux normals come
from the mesh boundary **segments in 2-D** (a corner node — undefined normal — falls back to the PDE
residual) and boundary **faces in 3-D** (each oriented outward exactly via its owning tet's apex, so a
flat face gives an exact axis normal, no corner heuristic). Plus **transient** problems by
method-of-lines with a ``u.t`` term carrying a **unit or a general ``c(x)·u.t`` mass coefficient** (e.g.
``ρcₚ(x)·ui.t - νΔu``; the coefficient is extracted by a two-probe ``c = F(u.t=1) − F(u.t=0)`` and carried
as ``M = diag(c)``, a nonlinear ``c(u)`` fails loud) — all with linear + nonlinear residuals, and with
the time scheme selectable via ``.solve(time=…)`` exactly as ``fem.solve(time=…)`` (``jno.solve.theta``
for backward Euler / Crank–Nicolson, ``jno.solve.adaptive``; backward Euler by default — the exponential
integrator needs a linear block the matrix-free residual doesn't assemble, so it fails loud). **Flux BCs
compose with transient too** — a flux node becomes an algebraic zero-mass-row constraint imposing the
same ``a·(∇u·n) + b`` at each instant (its value determined by the interior via the flux). **Periodic**
boundaries are a tie constraint ``u(left) - u(right)`` (opposite faces, exactly as ``jno.fem``): on a
**structured grid** it wraps that axis (the ``jnp.roll`` stencil gives the true periodic Laplacian, not a
one-sided edge), structured-only since a strong-form stencil must wrap. A pure-Neumann
problem (no Dirichlet node) is singular (solution up to a constant) and is solved as-is.

**Structured grid.** ``jno.Shape.rect(x0, y0, x1, y1, size=h).structured().domain()`` (2-D) or
``jno.Shape.box(x0, y0, z0, x1, y1, z1, size=h).structured().domain()`` (3-D) builds a regular
grid — a right-triangulation in 2-D, a Kuhn 6-tets-per-voxel mesh in 3-D — and records a grid descriptor
on ``mesh_connectivity["grid"]``; the interior operators (``jno.fdm.laplacian`` / ``gradient`` and the
constraint-list ``u.d2(x)`` authoring) then take the assembly-free direct finite-difference stencils (the
5-point Laplacian in 2-D, 7-point in 3-D) instead of the cotangent operator, without per-element assembly.
The canonical ``jno.fdm([-ui.d2(x) - ui.d2(y) - f, u(bnd) - g]).solve()`` works unchanged and stays
differentiable; because the reduced-Dirichlet stencil operator is nonsymmetric, a structured solve
defaults its inner Krylov to **GMRES** (robust for nonsymmetric systems, still matrix-free) instead of
BiCGStab, **preconditioned by a geometric-multigrid V-cycle** (:func:`jno.precond.gmg`) — O(N),
grid-independent convergence — with a plain-GMRES fallback when the grid is too small to coarsen. All
automatic, no authoring change. Composite/CSG and cut-cell geometry are planned.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import solve as _solve
from .differential_operators import DifferentialOperators as _D

__all__ = ["fdm", "laplacian", "gradient"]


def _fd_operator_noise(residual_fn, u0) -> float:
    """Relative **evaluation noise** of the residual's linearization at ``u0`` — the precision floor
    below which no solver can drive the residual, however many iterations it spends.

    Measured by the **additivity** of the JVP: ``jvp`` is exactly linear in its tangent for *any*
    differentiable residual, so ``jvp(v1+v2) == jvp(v1) + jvp(v2)`` in exact arithmetic. Whatever gap
    appears is pure floating-point noise in the operator's evaluation — and, crucially, the probe is
    **immune to nonlinearity of the residual itself** (verified: ``u -> u**3`` measures 1.2e-16).
    Three JVPs, negligible beside a Newton solve.

    Why this matters here: on an unstructured mesh a second derivative defaults to
    ``gradient_of_gradient`` — a *nested* least-squares/area-weighted gradient — and nesting amplifies
    roundoff. Measured on a 2-D Poisson residual (mesh 0.06, x64): **5.3e-08** relative, against
    **1.8e-16** for the ``cotangent`` Laplacian stencil on the same mesh, i.e. nine orders worse. The
    absolute floor is that figure times the residual scale, which is exactly where Newton was
    observed to stall (~7e-05 here). Asking for a tighter gate cannot succeed.
    """
    import jax

    # EAGER-ONLY, like every other measurement guard in the library: under a trace (the parametric /
    # crux inverse path, where the residual closes over tracers) the norms below cannot concretise.
    # Returning 0.0 leaves the driver's own defaults untouched rather than fabricating a floor.
    try:
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        v1 = jax.random.normal(k1, u0.shape, u0.dtype)
        v2 = jax.random.normal(k2, u0.shape, u0.dtype)
        a = jax.jvp(residual_fn, (u0,), (v1,))[1]
        b = jax.jvp(residual_fn, (u0,), (v2,))[1]
        c = jax.jvp(residual_fn, (u0,), (v1 + v2,))[1]
        den = float(jnp.linalg.norm(c))
        if not np.isfinite(den) or den == 0.0:
            return 0.0
        val = float(jnp.linalg.norm(c - (a + b)) / den)
    except Exception:  # traced, non-differentiable, or otherwise not characterisable here
        return 0.0
    return val if np.isfinite(val) else 0.0


def _fd_newton_tolerances(residual_fn, u0, *, safety: float = 1000.0) -> dict:
    """Newton tolerances for the FD residual: never tighter than the driver's own defaults, and never
    below the operator's measured precision floor (:func:`_fd_operator_noise`).

    The strong-form FD operators jNO builds on an unstructured mesh are **noisy by construction** (a
    nested gradient-of-gradient second derivative), so the stock ``atol=rtol=1e-8`` gate is
    unreachable for them — Newton then burns its whole step budget and the convergence guard reports
    a genuine non-convergence. That guard is right; the *request* was wrong. Scaling the gate to the
    measured floor makes the solve ask for what its own discretization can actually deliver.

    This does **not** paper over a bad solve: the floor is measured per problem, the gate is only ever
    *loosened*, and an operator with no noise (the ``cotangent`` stencil, a structured grid) keeps the
    full 1e-8. For a genuinely accurate Laplacian, author it as ``scheme="finite_difference:cotangent"``
    — exact to machine precision on the same mesh — rather than as nested ``u.d2(x) + u.d2(y)``.

    ``safety`` is 1000 rather than 1: the probe measures the floor of a **single** operator
    evaluation, while Newton's achievable residual is that noise amplified through the inner Krylov
    solve and the outer iteration. Measured on the failing suite, one evaluation's floor sat 3-30x
    below where Newton actually stalled, so three orders of margin covers the chain without being
    open-ended — and the gate is still *derived from a measurement of this problem*, not a constant.
    """
    noise = _fd_operator_noise(residual_fn, u0)
    if noise <= 0.0:
        return {}
    try:
        r0 = float(jnp.linalg.norm(jnp.asarray(residual_fn(u0))))
    except Exception:  # traced residual: leave the driver's defaults alone
        return {}
    floor = safety * noise * r0
    if not np.isfinite(floor) or floor <= 1e-8:
        return {}
    return {"atol": floor, "rtol": 1e-8}


def _structured_linear_solve(domain):
    """Inner linear solve for the matrix-free Newton–Krylov on a **structured grid**: GMRES rather than
    the driver's default BiCGStab. The reduced-Dirichlet 5-/7-point operator is nonsymmetric, and BiCGStab
    can break down on it (a strong-form ``u.d2(x)+u.d2(y)`` returns NaN), whereas GMRES is robust for
    nonsymmetric systems while staying matrix-free and differentiable (the driver firewalls it in
    ``custom_linear_solve``, so the reverse pass runs GMRES on ``Aᵀ``).

    The GMRES is **preconditioned by a geometric-multigrid V-cycle** (:func:`build_vcycle`) built from the
    grid — O(N), grid-independent convergence (~0.1 residual reduction per cycle) on Poisson-type
    operators — falling back to plain GMRES when the grid is too small to coarsen (a single level). The
    V-cycle is a fixed linear operator, so standard GMRES (not FGMRES) suffices. Returns ``None`` for an
    unstructured mesh, so the driver keeps its (BiCGStab) default there."""
    if getattr(domain, "mesh_connectivity", None) is None or domain.mesh_connectivity.get("grid") is None:
        return None
    from .utils.solver.geometric_mg import build_vcycle
    from .utils.solver.solver_api import LinearOperator

    grid = domain.mesh_connectivity["grid"]
    gmres = _solve.gmres()
    if any(grid.get("periodic") or ()):  # the GMG V-cycle assumes Dirichlet boundaries; skip it (plain GMRES)
        return lambda mv, rhs: gmres(LinearOperator.from_matvec(mv), rhs)
    vcycle, n_levels = build_vcycle(grid["shape"], grid["spacing"])
    precond = vcycle if n_levels >= 2 else None  # skip GMG when the grid can't be coarsened
    return lambda mv, rhs: gmres(LinearOperator.from_matvec(mv), rhs, M=precond)


def _integrate_transient(block, ts, time):
    """March the semidiscrete ``block`` over the save-times ``ts`` with the chosen **time scheme** — a
    ``jno.solve.theta`` / ``adaptive`` / ``exponential`` slot, via its ``.integrate`` — or the default
    backward-Euler ``lax.scan`` when ``time is None``. This is the FDM analogue of ``fem.solve(time=…)``:
    the scheme is the *same* slot object ``jno.fem`` uses, so θ / Crank–Nicolson, adaptive step size, and
    the exponential integrator all compose onto the strong-form method-of-lines march."""
    from .utils.solver.backend_blocks import _default_transient_integrate

    if time is None:
        return _default_transient_integrate(block, {}, ts)
    return time.integrate(block, {}, ts, linear_solve=None, nonlinear_solve=None)


def _mesh(domain):
    """``(points, cells)`` for the domain — 2-D triangles or 3-D tetrahedra, dispatched on
    ``domain.dimension``. 1-D is not exposed here (``jno.fdm`` is a 2-D/3-D collocation solver)."""
    mc = domain.mesh_connectivity
    dim = int(getattr(domain, "dimension", 2))
    pts = jnp.asarray(np.asarray(mc["points"])[:, :dim])
    if dim == 2:
        return pts, jnp.asarray(mc["triangles"])
    if dim == 3:
        return pts, jnp.asarray(mc["tetrahedra"])
    raise NotImplementedError("jno.fdm: only 2-D triangular and 3-D tetrahedral meshes are supported.")


def laplacian(u, domain, method: str = "cotangent"):
    """FD Laplacian ``Δu`` of the nodal field ``u`` on the domain's mesh. ``method="cotangent"`` (the
    default) is the symmetric, CG-compatible Laplace–Beltrami stencil — the cotangent-weight operator on
    a 2-D triangular mesh and its exact analogue, the **P1 tetrahedral finite-element** operator, on a
    3-D tet mesh (second-order for the Galerkin solve). ``"gradient_of_gradient"`` (first-order double
    difference) and ``"lsq_of_gradient"`` are the local alternatives; ``"lsq_of_gradient"`` is unstable
    for the *second* derivative on tetrahedra (nested least-squares amplifies) and is not recommended in
    3-D."""
    pts, cells = _mesh(domain)
    dim = int(getattr(domain, "dimension", 2))
    grid = domain.mesh_connectivity.get("grid")  # structured-grid fast path (2-D), else None
    if dim == 3:
        return _D.compute_fd_laplacian_3d_simple(u, pts, cells, dims=(0, 1, 2), method=method, grid=grid)
    return _D.compute_fd_laplacian_2d_simple(u, pts, cells, dims=(0, 1), method=method, grid=grid)


def gradient(u, domain, method: str = "area_weighted"):
    """FD gradient ``∇u`` of the nodal field ``u`` — shape ``(N, dim)``. ``method`` selects the stencil
    (``"area_weighted"`` default, ``"uniform"``, ``"inverse_distance"``, ``"least_squares"``); the same
    names apply on a 2-D triangular or 3-D tetrahedral mesh."""
    pts, cells = _mesh(domain)
    dim = int(getattr(domain, "dimension", 2))
    grid = domain.mesh_connectivity.get("grid")  # structured-grid fast path (2-D), else None
    if dim == 3:
        comps = [_D.compute_fd_gradient_3d_simple(u, pts, cells, d, method=method, grid=grid) for d in range(3)]
        return jnp.stack(comps, axis=1)
    gx = _D.compute_fd_gradient_2d_simple(u, pts, cells, 0, method=method, grid=grid)
    gy = _D.compute_fd_gradient_2d_simple(u, pts, cells, 1, method=method, grid=grid)
    return jnp.stack([gx, gy], axis=1)


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


def _find_unknowns(constraints):
    """All ``domain.unknown()`` fields (nodal-field-parameter ModelCalls' Models) in the constraints, in
    first-appearance order — a **coupled** system has several. At least one is required. The k-th PDE
    equation (constraint order) drives the k-th unknown's DOF block; a ``u_k(region) - g`` BC is folded
    into that block by whichever unknown it contains."""
    from .trace import ModelCall

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
            "jno.fdm([...]): expected at least one domain.unknown() field. Author the strong form with "
            "`u = domain.unknown()` (declare several for a coupled system)."
        )
    return order


def _periodic_axis(tag_a, tag_b):
    """The grid axis a periodic tie ``u(A) - u(B)`` wraps, from its two opposite-face tags:
    ``left``/``right`` → 0 (x), ``bottom``/``top`` → 1 (y), ``front``/``back`` → 2 (z). ``None`` if the
    two tags are not an opposite-face pair."""
    faces = {frozenset(("left", "right")): 0, frozenset(("bottom", "top")): 1, frozenset(("front", "back")): 2}
    return faces.get(frozenset((tag_a, tag_b)))


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
    subdomain. A ``jno.Shape`` uses the analytic, shapely-free :meth:`Shape.contains` (2-D and 3-D — the
    primary path); a shapely geometry falls back to shapely (2-D mesh-conforming regions in
    ``polygon_domain``, which stay shapely by scope)."""
    from .geometry import Shape

    p = np.asarray(pts)
    if isinstance(geom, Shape):
        mask = np.asarray(geom.contains(p[:, : geom.dim]))
    else:
        import shapely

        mask = np.asarray(shapely.contains_xy(geom.buffer(1e-9), p[:, 0], p[:, 1]))
    return np.nonzero(mask)[0].astype(int)


def _set_temporal(node, val):
    """Replace every ``u.t`` (:class:`TemporalDerivative`) in ``node`` with the constant ``val``, leaving
    the rest of the expression intact. Two probes recover, for a residual ``F(u.t, u) = c·u.t + R_spatial``
    affine in the time derivative, the **spatial** residual ``R_spatial = F(u.t=0)`` and the **mass
    coefficient** ``c = F(u.t=1) − F(u.t=0)`` — so a general ``c(x)·u.t`` term (variable material, e.g.
    ``ρcₚ(x)·u.t``) is handled without parsing its structure, exactly as ``_set_normal`` handles a flux."""
    from .trace import BinaryOp, FunctionCall, Hessian, Jacobian, Literal, Placeholder, TemporalDerivative

    if isinstance(node, TemporalDerivative):
        return Literal(float(val))
    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, _set_temporal(node.left, val), _set_temporal(node.right, val))
    if isinstance(node, FunctionCall):
        return node.copy_with_args([_set_temporal(a, val) if isinstance(a, Placeholder) else a for a in node.args])
    if isinstance(node, Jacobian):
        return Jacobian(_set_temporal(node.target, val), node.variables, node.scheme)
    if isinstance(node, Hessian):
        return Hessian(_set_temporal(node.target, val), node.variables, node.scheme, node.trace)
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
        self._constraints = list(constraints)  # kept verbatim so a coupled solve can re-author with an interface pin
        self.unknowns = _find_unknowns(constraints)  # coupled system ⇒ several, in declaration order
        self.unknown = self.unknowns[0]  # the single-field paths (transient/flux/parametric) use this
        self._nf = len(self.unknowns)
        self.domain = self.unknown._fem_field_domain
        self._N = int(np.asarray(self.domain.mesh_connectivity["points"]).shape[0])  # nodes per field
        self._Ntot = self._nf * self._N  # blocked DOF vector [field_0 (N), …, field_{nf-1} (N)]
        self._pts = jnp.asarray(np.asarray(self.domain.mesh_connectivity["points"])[:, : self.domain.dimension])
        self._pde, self._dirichlet, self._neumann, self._ic = [], [], [], []
        self._periodic_axes = []  # grid axes tied by a `u(A) - u(B)` periodic constraint (structured only)
        for c in constraints:
            # Classify by structure (not by which region tag), so a value-only pin works on ANY region —
            # a boundary edge OR a geometric sub-region (`domain.region(name, geom)`, used by coupled /
            # domain-decomposition solves to pin a subdomain's complement to a neighbour's field):
            #   * a periodic tie `u(A) - u(B)` (opposite faces)  → wrap the grid axis (check first);
            #   * a normal derivative `ui.d(n, ...)`           → a Neumann/Robin flux row;
            #   * a derivative of the unknown (Laplacian, u.t) → the PDE residual;
            #   * the `initial` region, value-only            → the initial condition;
            #   * otherwise (value-only, affine in u)          → a Dirichlet pin on its region.
            tie = getattr(c, "_periodic_tie", None)
            if tie is not None:
                ax = _periodic_axis(*tie)
                if ax is None:
                    raise ValueError(
                        f"jno.fdm([...]): a periodic tie `u(A) - u(B)` must connect two OPPOSITE faces "
                        f"(left/right, bottom/top, or front/back); got {tie}."
                    )
                self._periodic_axes.append(ax)
            elif _normal_jacobian(c) is not None:
                self._neumann.append(c)
            elif any(_has_unknown_derivative(c, u) for u in self.unknowns):
                self._pde.append(c)
            elif _region_tag(c) == "initial":
                self._ic.append(c)
            else:
                self._dirichlet.append(c)
        if not self._pde:
            raise ValueError("jno.fdm([...]): no PDE residual found (a term with a derivative of the unknown).")
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
        if self._nf > 1:  # coupled (multi-field): v1 is STEADY + Dirichlet only
            if len(self._pde) != self._nf:
                raise ValueError(
                    f"jno.fdm([...]): a coupled system needs exactly one PDE equation per unknown — got "
                    f"{self._nf} unknowns but {len(self._pde)} PDE equation(s). Author one equation per "
                    "field, in the order the unknowns are declared (equation k drives unknown k)."
                )
            if self._transient or self._neumann:
                raise NotImplementedError(
                    "jno.fdm([...]): a coupled (multi-field) system is v1-limited to a STEADY problem with "
                    "Dirichlet BCs — transient / flux BCs on coupled fields are not yet supported."
                )
        self._grid = None
        if self._periodic_axes:  # mark the grid axes the wrap stencil must handle (structured only)
            grid = self.domain.mesh_connectivity.get("grid")
            if grid is None:
                raise NotImplementedError(
                    "jno.fdm([...]): a periodic tie `u(A) - u(B)` requires a STRUCTURED grid — build the "
                    "domain with `jno.Shape.rect(...).structured().domain()`. Periodic on an unstructured mesh is "
                    "not supported (the FD stencil must wrap the grid, which a boundary tie alone cannot)."
                )
            per = list(grid.get("periodic") or (False,) * len(grid["shape"]))
            for ax in self._periodic_axes:
                per[ax] = True
            grid["periodic"] = tuple(per)  # the FD kernels read this to wrap those axes
            self._grid = grid
        # The sub-domain this problem owns, if its PDE coordinates carry a named region
        # (`domain.region(name, poly)`). Used by `jno.core([...])` to couple subdomains automatically.
        self.region, self.region_geometry = self._pde_region()

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

    def _region_nodes(self, tag):
        if tag == "initial":
            return np.arange(self._N, dtype=int)  # the IC is the whole spatial field at t=t0
        reg = getattr(self.domain, "_boundary_registry", {}).get(tag)
        if reg is not None and len(reg.get("point_indices", [])) > 0:
            return np.asarray(reg["point_indices"], dtype=int)
        ptags = getattr(self.domain, "_polygon_tags", {})  # a geometric sub-region (domain.region(...))
        if tag in ptags and ptags[tag][0] == "interior":
            return _mesh_nodes_in(np.asarray(self._pts), ptags[tag][1])
        return np.asarray(self.domain.mesh_connectivity["boundary_indices"], dtype=int)

    def _pde_residual_fn(self, *, spatial=False, extra_params=None):
        """Differentiable residual over the nodal DOF vector, collocated at the mesh nodes. With
        ``spatial=True`` the ``u.t`` terms are dropped (:func:`_zero_temporal`) to give the
        method-of-lines spatial residual ``R_spatial`` for the semidiscrete march. ``extra_params``
        (``{layer_id: module}``) injects the current value of any **trainable** ``jno.np.parameter`` in
        the residual — how a ``crux``-driven inverse reaches the solve (see :meth:`_parametric_node`)."""
        import equinox as eqx

        from .trace_evaluator import TraceEvaluator

        if self._nf == 1:  # single field: sum all PDE terms into the one equation (historic behaviour)
            expr = self._pde[0]
            for c in self._pde[1:]:
                expr = expr + c
            exprs = [_unwrap(expr)]
        else:  # coupled: one equation per field, order-paired (equation k → block k → unknown k)
            exprs = [_unwrap(self._pde[k]) for k in range(self._nf)]
        if spatial:
            exprs = [_zero_temporal(e) for e in exprs]
        spatial_tags = {  # collocate every spatial term at the mesh nodes (temporal tags carry no field)
            v.tag
            for c in self._pde
            for v in (getattr(c, "_coord_vars", None) or {}).values()
            if getattr(v, "axis", None) != "temporal"
        }
        context = {t: self._pts for t in spatial_tags}
        N, unknowns = self._N, self.unknowns

        def residual_fn(dofs):
            dofs = jnp.asarray(dofs)
            params = dict(extra_params or {})
            for k, unk in enumerate(unknowns):  # inject each field's DOF slice into its module
                slice_k = dofs[k * N : (k + 1) * N] if len(unknowns) > 1 else dofs
                params[unk.layer_id] = eqx.tree_at(lambda m: m.value, unk.module, slice_k.astype(unk.module.value.dtype))
            ev = TraceEvaluator(params=params)
            blocks = [jnp.asarray(ev.evaluate(e, context=context, var_bindings={})).reshape(-1) for e in exprs]
            return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks)

        return residual_fn

    def _mass_coefficient(self):
        """Per-node coefficient ``c`` on ``u.t`` (the diagonal mass ``M = diag(c)``), via the two-probe
        ``c = F(u.t=1) − F(u.t=0)`` (:func:`_set_temporal`) — the spatial residual cancels between the
        probes, leaving ``c``. A plain ``ui.t - 𝒩(u)`` gives ``c = 1``; a ``ρcₚ(x)·ui.t`` term gives the
        node values of ``ρcₚ(x)``. ``c`` must be constant in ``u`` (a nonlinear mass ``c(u)·u.t`` raises).
        Single-field only (the transient march is)."""
        import equinox as eqx

        from .trace_evaluator import TraceEvaluator

        expr = self._pde[0]
        for c in self._pde[1:]:
            expr = expr + c
        expr = _unwrap(expr)
        spatial_tags = {
            v.tag
            for c in self._pde
            for v in (getattr(c, "_coord_vars", None) or {}).values()
            if getattr(v, "axis", None) != "temporal"
        }
        context = {t: self._pts for t in spatial_tags}
        lid, base = self.unknown.layer_id, self.unknown.module

        def probe(u_val):
            def at(temporal):
                mod = eqx.tree_at(lambda m: m.value, base, jnp.full(self._N, u_val, base.value.dtype))
                ev = TraceEvaluator(params={lid: mod})
                return jnp.asarray(ev.evaluate(_set_temporal(expr, temporal), context=context, var_bindings={})).reshape(-1)

            return at(1.0) - at(0.0)

        c0 = probe(0.0)
        if not bool(jnp.allclose(c0, probe(1.0), atol=1e-6, rtol=1e-6)):  # u-dependence ⇒ nonlinear mass
            raise ValueError(
                "jno.fdm([...]): the `u.t` mass coefficient depends on u (a nonlinear mass `c(u)·u.t`) — v1 "
                "supports a constant or coordinate-dependent `c(x)·u.t` only."
            )
        return c0

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
                and all(n.model is not u for u in self.unknowns)
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
        if getattr(inner, "op", None) == "-":  # u(region) - g  →  g is the side without any unknown
            left_has_u = any(_contains_unknown(inner.left, u) for u in self.unknowns)
            g_node = inner.right if left_has_u else inner.left
        return self._eval_g(g_node, idx)

    def _field_index(self, constraint):
        """Which unknown's DOF block a value-only constraint (Dirichlet) pins — 0 for a single field."""
        for k, u in enumerate(self.unknowns):
            if _contains_unknown(constraint, u):
                return k
        return 0

    def _dirichlet_rows(self):
        """Per-field Dirichlet rows ``(field_index, node_indices, values)``: ``field_index`` selects the
        DOF block (0 for a single field), ``node_indices`` the region's nodes, ``values`` the pinned g."""
        rows = []
        for c in self._dirichlet:
            idx = self._region_nodes(_region_tag(c))
            rows.append((self._field_index(c), jnp.asarray(idx), self._condition_value(c, idx)))
        return rows

    def _periodic_rows(self):
        """``(secondary_idx, main_idx)`` per periodic axis: the secondary (last-index) face DOFs tied to the
        main (first-index) face DOFs, matched by the other-axis indices — the tie ``u[secondary] = u[main]``
        pinning the redundant face (node ``L ≡ 0``) that the wrap stencil already identifies."""
        if not self._periodic_axes:
            return []
        shape = self._grid["shape"]
        idx = np.arange(self._N).reshape(shape)
        rows = []
        for ax in self._periodic_axes:
            main = np.take(idx, 0, axis=ax).ravel()
            secondary = np.take(idx, shape[ax] - 1, axis=ax).ravel()
            rows.append((jnp.asarray(secondary), jnp.asarray(main)))
        return rows

    def _node_normals(self, region):
        """Unit outward normals for the flux nodes of ``region``, aligned to those nodes.

        **2-D** — computed from the mesh **boundary segments** (``mesh_connectivity["boundary_edges"]``):
        each segment's exact perpendicular is averaged over the (two) segments meeting at a node, so an
        axis-aligned edge yields an exact ``(±1, 0)`` / ``(0, ±1)`` — much cleaner than the domain's
        smoothed per-point normal, which bleeds a tangential component near corners and would spoil the
        flux. Outward orientation is taken (sign only) from ``domain.variable(region, normals=True)``. A
        **corner** node averages two differently-oriented segment normals, so the averaged magnitude
        drops (≈0.71 at a right angle): the outward normal is undefined there, so corners are **dropped**
        from the flux row and keep their interior PDE residual (give a corner an explicit Dirichlet
        condition if it needs one).

        **3-D** — see :meth:`_node_normals_3d`: the region's boundary triangles are oriented outward
        exactly via each face's owning-tet apex, so no corner heuristic is needed.

        Returns ``(kept_node_indices, unit_n)``."""
        dim = self.domain.dimension
        if dim == 3:
            return self._node_normals_3d(region)
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

    def _node_normals_3d(self, region):
        """Outward unit normals for the boundary-face nodes of ``region`` on a **tetrahedral** mesh.

        The region's boundary triangles — the mesh boundary faces (a face shared by exactly one tet)
        whose three vertices are all tagged ``region`` — are extracted from the tet connectivity
        (:meth:`MeshUtils._boundary_faces_with_apex`). Each face normal is oriented outward **exactly**
        via its owning tet's apex, then area-weighted and averaged per node
        (:meth:`MeshUtils._compute_normals_from_boundary_faces`), so a flat face gives an exact axis
        normal and a curved region an accurate one. Restricting to the region's **own** faces keeps a
        region-edge node's normal consistent (all contributing faces are coplanar for a flat face), so —
        unlike the 2-D path — no corner-dropping is needed. Returns ``(node_indices, unit_n)``."""
        from .domain.mesh_utils import MeshUtils

        pts = np.asarray(self._pts)
        tets = np.asarray(self.domain.mesh_connectivity["tetrahedra"], dtype=int)
        bfaces, bapex = MeshUtils._boundary_faces_with_apex(tets)
        region_nodes = np.asarray(self._region_nodes(region), dtype=int)
        on_region = np.isin(bfaces, region_nodes).all(axis=1)  # a face lies on the region ⟺ all 3 verts tagged
        rfaces, rapex = bfaces[on_region], bapex[on_region]
        n, idx = MeshUtils._compute_normals_from_boundary_faces(pts, rfaces, apex_points=pts[rapex])
        return np.asarray(idx, dtype=int), jnp.asarray(n)

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

    def solve(self, nonlinear=None, x0=None, profile=False, time=None):
        """Solve the strong-form system. **Steady** problems fold the Dirichlet rows into the residual
        (``u - g`` on the region) and hand it to the same ``jno.solve`` Newton–Krylov + ``custom_root``
        machinery ``jno.fem`` uses (linear/nonlinear uniform, differentiable for inverse problems).
        **Transient** problems (an ``u(initial) - u0`` condition is present) march by method-of-lines —
        ``t_span`` and the step count come from ``domain.time`` and the initial state from the IC — and
        return the trajectory (``(n_save, N)``); ``x0`` is rejected (the IC owns the initial state).

        ``time=`` selects the time scheme exactly as ``fem.solve(time=…)`` does — ``jno.solve.theta(θ)``
        (Crank–Nicolson at θ=0.5), ``jno.solve.adaptive(…)`` (step-doubling adaptive step size), or
        ``jno.solve.exponential(…)`` — defaulting to backward Euler. ``profile=True`` runs the (eager,
        non-parametric) solve inside a JAX Perfetto trace and writes it to ``./jno_traces``."""

        def _run():
            if self._transient:
                if x0 is not None:
                    raise ValueError("jno.fdm([...]): x0= is rejected for a transient problem — the IC owns the state.")
                return self._march(nonlinear=nonlinear, time=time)
            trainable = self._trainable_params()
            if trainable:
                return self._parametric_node(trainable, nonlinear=nonlinear, x0=x0)
            return self._steady_solve(nonlinear=nonlinear, x0=x0)

        if not profile:
            return _run()
        from .utils.profiling import profile_solve

        return profile_solve(_run, label=f"fdm profile · {self._N} nodes · {'transient' if self._transient else 'steady'}")

    def _steady_solve(self, *, nonlinear=None, x0=None, extra_params=None, extra_pins=None):
        """The eager steady solve: fold the flux and Dirichlet rows into the residual and hand it to the
        ``jno.solve`` Newton–Krylov driver. ``extra_params`` carries the current values of any trainable
        ``jno.np.parameter`` (from :meth:`_parametric_node`). ``extra_pins`` is an ``(idx, values)`` pair
        of nodes pinned to given values on top of the authored BCs — the interface pin a coupled /
        domain-decomposition Schwarz step applies (:meth:`solve_pinned`)."""
        N, single = self._N, self._nf == 1
        residual_fn = self._pde_residual_fn(extra_params=extra_params)
        rows = self._dirichlet_rows()
        flux_rows = self._flux_rows(extra_params) if single else []  # flux is single-field (guarded at build)
        periodic_rows = self._periodic_rows()  # (secondary, main) face DOF pairs per periodic axis

        def residual_with_bc(u):
            r = residual_fn(u)
            # Flux rows first (`a·(∇u·n) + b`, with a = F(1)-F(0), b = F(0) — Neumann/Robin/etc.), then
            # the periodic ties, then Dirichlet: a node carrying several (a 2-D corner, or a 3-D edge)
            # resolves to the essential Dirichlet value — the Dirichlet row is set last.
            for idx, nrm, method, v0, v1 in flux_rows:
                grad = gradient(u, self.domain, method=method)  # (N, 2), differentiable
                flux = jnp.sum(grad[idx] * nrm, axis=1)  # ∇u·n at the edge nodes
                b = v0(u)
                a = v1(u) - b
                r = r.at[idx].set(a[idx] * flux + b[idx])
            for secondary, main in periodic_rows:  # periodic: the redundant secondary face ≡ the main face
                r = r.at[secondary].set(u[secondary] - u[main])
            if extra_pins is not None:  # interface pin (a coupled subdomain's complement) — before the
                pidx, pvals = extra_pins  # authored Dirichlet, so the physical outer BC still wins on ∂Ω
                r = r.at[pidx].set(u[pidx] - pvals)
            for k, idx, gvals in rows:  # Dirichlet: pin field k's DOF block at its region nodes
                base = k * N
                r = r.at[base + idx].set(u[base + idx] - gvals)
            return r

        u0 = jnp.zeros(self._Ntot) if x0 is None else jnp.asarray(x0).reshape(-1)
        driver = nonlinear or _solve.newton(**_fd_newton_tolerances(residual_with_bc, u0))
        sol = driver(residual_with_bc, u0, linear_solve=_structured_linear_solve(self.domain) if single else None)
        return sol if single else sol.reshape(self._nf, N)  # coupled: (nf, N), one row per field

    def pinned_solver(self, node_ids, *, nonlinear=None):
        """A **reusable** ``f(values) -> field`` that solves the subdomain with ``node_ids`` pinned to
        ``values`` (the interface Dirichlet data from a neighbour) on top of the authored BCs. Built
        ONCE and JIT-compiled, so the Newton solve compiles a single time and is reused across Schwarz
        iterations — a fresh per-call closure would recompile every step and exhaust device memory."""
        import jax

        residual_fn = self._pde_residual_fn()
        rows = self._dirichlet_rows()
        flux_rows = self._flux_rows()
        pin_idx = jnp.asarray(node_ids)
        driver = nonlinear or _solve.newton()

        @jax.jit
        def solve(values):
            pv = jnp.asarray(values)

            def residual_with_bc(u):
                r = residual_fn(u)
                for idx, nrm, method, v0, v1 in flux_rows:
                    grad = gradient(u, self.domain, method=method)
                    flux = jnp.sum(grad[idx] * nrm, axis=1)
                    b = v0(u)
                    a = v1(u) - b
                    r = r.at[idx].set(a[idx] * flux + b[idx])
                r = r.at[pin_idx].set(u[pin_idx] - pv)  # interface pin — before the authored Dirichlet
                for _k, idx, gvals in rows:  # single-field (domain-decomposition) path ⇒ block 0
                    r = r.at[idx].set(u[idx] - gvals)
                return r

            return driver(residual_with_bc, jnp.zeros(self._N))

        return solve

    def solve_pinned(self, node_ids, values, *, nonlinear=None):
        """One-shot: solve with ``node_ids`` pinned to ``values`` (see :meth:`pinned_solver`, which the
        Schwarz driver builds once and reuses)."""
        return self.pinned_solver(node_ids, nonlinear=nonlinear)(values)

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

    def _march(self, *, nonlinear=None, save_ts=None, time=None):
        """Method-of-lines march of ``u̇ = -R_spatial(u)`` reusing jNO's solver-agnostic
        :class:`SemidiscreteTimeBlock` integrator (``custom_root`` differentiable). ``M = I`` on interior
        nodes; **Dirichlet and Neumann/Robin flux nodes carry a zero mass row** — Dirichlet pins to ``g``,
        a flux node imposes the SAME ``a·(∇u·n) + b`` the steady solve folds in, as an index-1 DAE
        constraint the boundary value satisfies at each instant (its value is determined by the interior via
        the flux). ``t_span``/``dt`` come from ``domain.time``; ``time=`` picks the scheme (backward Euler
        by default)."""
        import jax.experimental.sparse as jsparse

        from .utils.solver.backend_blocks import SemidiscreteTimeBlock, _block_time_grid
        from .utils.solver.time_route import _infer_time_window

        t0, t1, dt = _infer_time_window(self.domain)
        if dt is None:
            raise ValueError("jno.fdm([...]): domain.time must specify n_steps >= 2 for a transient march.")

        rows = self._dirichlet_rows()
        flux_rows = self._flux_rows()
        bmask = np.zeros(self._N, dtype=bool)
        bvals = np.zeros(self._N)
        for _k, idx, gv in rows:  # single-field transient ⇒ block 0
            bmask[np.asarray(idx)] = True
            bvals[np.asarray(idx)] = np.asarray(gv)
        algebraic = bmask.copy()  # Dirichlet + flux nodes are algebraic (zero mass row)
        for row in flux_rows:
            algebraic[np.asarray(row[0])] = True
        bmask, bvals, algebraic = jnp.asarray(bmask), jnp.asarray(bvals), jnp.asarray(algebraic)

        spatial_res = self._pde_residual_fn(spatial=True)
        c_nodes = self._mass_coefficient()  # u.t coefficient: 1 for a plain u.t, c(x) for ρcₚ(x)·u.t
        diag = jnp.stack([jnp.arange(self._N), jnp.arange(self._N)], axis=1)
        M = jsparse.BCOO((jnp.where(algebraic, 0.0, c_nodes), diag), shape=(self._N, self._N))  # 0: Dirichlet+flux

        def residual(wn, t, args):  # M u̇ + R = 0 → interior u̇ = -R_spatial; flux/Dirichlet rows algebraic
            r = spatial_res(wn)
            for idx, nrm, method, v0, v1 in flux_rows:  # a·(∇u·n) + b — the same folding as _steady_solve
                grad = gradient(wn, self.domain, method=method)
                flux = jnp.sum(grad[idx] * nrm, axis=1)
                b = v0(wn)
                a = v1(wn) - b
                r = r.at[idx].set(a[idx] * flux + b[idx])
            return jnp.where(bmask, wn - bvals, r)  # Dirichlet wins over flux on an overlapping node

        block = SemidiscreteTimeBlock(
            mass=lambda t, args: M,
            residual=residual,
            state0=self._initial_state(),
            t0=float(t0),
            t1=float(t1),
            dt=float(dt),
        )
        ts = _block_time_grid(block) if save_ts is None else jnp.asarray(save_ts)
        return _integrate_transient(block, ts, time)


def fdm(constraints):
    """Finite-difference PDE solver — the strong-form sibling of :func:`jno.fem`.

    The problem is a **constraint list**, exactly as in ``jno.fem``: ``jno.fdm([residual, u(xb, yb) - g])``
    authored with ``u = domain.unknown()`` and strong-form derivatives. Constraints are classified by
    region — interior terms are the PDE residual, boundary terms are Dirichlet conditions — and a
    ``u.t`` term makes it transient, taking its grid from ``domain.time``.

    For the accurate whole-Laplacian stencil write ONE term,
    ``u.laplacian(x, y, scheme="finite_difference:cotangent")``; per-axis ``d2`` refuses that
    sub-scheme, because summing one per axis would multiply the Laplacian by the dimension."""
    if not isinstance(constraints, (list, tuple)):
        raise TypeError(
            "jno.fdm expects a constraint LIST, e.g. jno.fdm([-u.laplacian(x, y) - f, u(xb, yb) - 0.0]) "
            f"with u = domain.unknown(); got {type(constraints).__name__}. The old function form "
            "jno.fdm(domain, residual=..., dirichlet=...) has been removed — write the residual and the "
            "boundary condition as terms instead."
        )
    return _TraceFDM(list(constraints))


fdm.laplacian = laplacian  # convenience: jno.fdm.laplacian(u, domain)
fdm.gradient = gradient
