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
as ``M = diag(c)``, a nonlinear ``c(u)`` fails loud), and **second order in time** with a ``u.tt`` term
(the augmented ``[u; v]`` march, θ = ½ by default; optional damping ``c(x)·u.t`` and initial velocity
``ui0.t - v0``) — all with linear + nonlinear residuals, and with
the time scheme selectable via ``.solve(time=…)`` exactly as ``fem.solve(time=…)`` (``jno.solve.theta``
for backward Euler / Crank–Nicolson, ``jno.solve.adaptive``; backward Euler by default — the exponential
integrator needs a linear block the matrix-free residual doesn't assemble, so it fails loud). **Flux BCs
compose with transient too** — a flux node becomes an algebraic zero-mass-row constraint imposing the
same ``a·(∇u·n) + b`` at each instant (its value determined by the interior via the flux). **Periodic**
boundaries are a tie constraint ``u(left) - u(right)`` (opposite faces, exactly as ``jno.fem``): on a
**structured grid** it wraps that axis (the ``jnp.roll`` stencil gives the true periodic Laplacian, not a
one-sided edge), structured-only since a strong-form stencil must wrap. A pure-Neumann
problem (no Dirichlet node) is singular (solution up to a constant) and is solved as-is.

**Structured grid.** ``jno.shape.rect(x0, y0, x1, y1, size=h).structured().domain()`` (2-D) or
``jno.shape.box(x0, y0, z0, x1, y1, z1, size=h).structured().domain()`` (3-D) builds a regular
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

    History, because the docstring used to claim the opposite: the **5.3e-08** floor this was built
    for was not the discretization at all. ``jno.np.parameter`` hardcoded ``float32``, so the strong-
    form residual rounded its unknown to single precision on every evaluation
    (``_pde_residual_fn`` casts the DOF vector to the unknown module's dtype). Both stencils measured
    ~6e-08 through the trace while both measured ~2e-16 called directly — the gap was the cast, not
    the nesting. With the dtype following ``jax_enable_x64`` every stencil measured here sits at
    machine precision (cotangent 1.7e-16, nested ``d2(x)+d2(y)`` 2.1e-16, ``:lsq`` 3.2e-16 on mesh
    0.08), so :func:`_fd_newton_tolerances` returns the driver's own defaults. The probe stays as a
    measured safety valve for an operator that genuinely is noisy.
    """
    import jax

    # EAGER-ONLY, like every other measurement guard in the library: under a trace (the parametric /
    # crux inverse path, where the residual closes over tracers) the norms below cannot concretise.
    # Returning 0.0 leaves the driver's own defaults untouched rather than fabricating a floor.
    try:
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        v1 = jax.random.normal(k1, u0.shape, u0.dtype)
        v2 = jax.random.normal(k2, u0.shape, u0.dtype)
        # One compiled call: three separate eager JVPs cost 2 s of dispatch at 16k nodes.
        tangents = jnp.stack([v1, v2, v1 + v2])
        a, b, c = jax.jit(lambda u, V: jax.vmap(lambda v: jax.jvp(residual_fn, (u,), (v,))[1])(V))(u0, tangents)
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

    The floor this was written for turned out to be a ``float32`` leak, now fixed at its source (see
    :func:`_fd_operator_noise`), and **every stencil measured since sits at machine precision** — so
    on the problems in the test suite this returns ``{}`` and Newton keeps its own ``1e-8`` gate. It
    stays because the rule is sound where a floor is real: an operator that cannot reach ``1e-8``
    makes Newton burn its whole step budget, and the convergence guard then reports a genuine
    non-convergence. The guard is right; the *request* was wrong.

    This does **not** paper over a bad solve: the floor is measured per problem, the gate is only ever
    *loosened*, and an operator with no noise keeps the full 1e-8.

    ``safety`` is 1000 rather than 1: the probe measures the floor of a **single** operator
    evaluation, while Newton's achievable residual is that noise amplified through the inner Krylov
    solve and the outer iteration. Measured back when a floor existed, one evaluation's floor sat
    3-30x below where Newton actually stalled, so three orders of margin covers the chain without
    being open-ended — and the gate is still *derived from a measurement of this problem*, not a
    constant.
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


def _integrate_transient(block, ts, time, linear_solve=None, nonlinear_solve=None):
    """March the semidiscrete ``block`` over the save-times ``ts`` with the chosen **time scheme** — a
    ``jno.solve.theta`` / ``adaptive`` / ``exponential`` slot, via its ``.integrate`` — or the default
    backward-Euler ``lax.scan`` when ``time is None``. This is the FDM analogue of ``fem.solve(time=…)``:
    the scheme is the *same* slot object ``jno.fem`` uses, so θ / Crank–Nicolson, adaptive step size, and
    the exponential integrator all compose onto the strong-form method-of-lines march."""
    from .utils.solver.backend_blocks import _default_transient_integrate

    if time is None:
        return _default_transient_integrate(block, {}, ts, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve)
    return time.integrate(block, {}, ts, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve)


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


def _two_ring(cells, n_nodes, idx):
    """Padded ``(len(idx), K)`` array of each node's two-ring neighbours (``-1`` = padding), from the
    mesh cells. Host-side and structural only; the numeric fit on top of it stays in JAX."""
    import scipy.sparse as sp

    cells = np.asarray(cells)
    k = cells.shape[1]
    rows = np.repeat(cells, k, axis=1).ravel()
    cols = np.tile(cells, (1, k)).ravel()
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes))
    ring2 = (adj @ adj)[np.asarray(idx)].tolil().rows
    nbrs = [[j for j in r if j != i] for r, i in zip(ring2, np.asarray(idx))]
    out = np.full((len(nbrs), max(len(r) for r in nbrs)), -1, dtype=int)
    for r, lst in enumerate(nbrs):
        out[r, : len(lst)] = lst
    return out


def _quadratic_gradient(u, pts, idx, nbrs):
    """``∇u`` at the nodes ``idx`` from a least-squares **quadratic** fit over their two-ring ``nbrs``.

    The one-ring area-weighted gradient is only first order at a boundary node, where the ring is
    one-sided, and a flux boundary condition built on it capped the whole solve at first order. Fitting
    ``u(x) ≈ u_i + g·d + ½ dᵀHd`` over the two-ring is second order in ``g`` on any mesh (measured on
    the unit square: boundary gradient error 8.2e-2 → 2.2e-2 → 5.4e-3 for h = 0.1 → 0.05 → 0.025,
    against 2.2e-1 → 1.2e-1 → 6.4e-2 area-weighted). Differentiable in ``u`` and in the coordinates."""
    mask = nbrs >= 0
    nb = np.where(mask, nbrs, 0)
    d = pts[nb] - pts[idx][:, None, :]  # (B, K, dim) offsets
    scale = jnp.max(jnp.linalg.norm(d, axis=-1) * mask, axis=1)[:, None, None]
    d = d / scale  # conditioning: fit in units of the local stencil radius
    dim = d.shape[-1]
    quad = [d[..., a] * d[..., b] for a in range(dim) for b in range(a, dim)]
    V = jnp.concatenate([d, jnp.stack(quad, axis=-1)], axis=-1) * mask[..., None]
    du = (u[nb] - u[idx][:, None]) * mask
    coef = jnp.einsum("bij,bj->bi", jnp.linalg.pinv(V), du)
    return coef[:, :dim] / scale[:, 0]


def _stencil_pattern(cells, n_nodes, radius, n_fields=1, extra_pairs=()):
    """Host-side sparsity candidate: node pairs within ``radius`` mesh edges of each other (plus any
    ``extra_pairs``, e.g. periodic partners), expanded to every field pair of a coupled system."""
    import scipy.sparse as sp

    cells = np.asarray(cells)
    k = cells.shape[1]
    rows = [np.repeat(cells, k, axis=1).ravel()]
    cols = [np.tile(cells, (1, k)).ravel()]
    for a, b in extra_pairs:
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        rows += [a, b]
        cols += [b, a]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    adj = sp.csr_matrix((np.ones(rows.size, dtype=np.int32), (rows, cols)), shape=(n_nodes, n_nodes))
    adj.data[:] = 1
    pattern = adj
    for _ in range(radius - 1):
        pattern = (pattern @ adj).tocsr()
        pattern.data[:] = 1
    if n_fields > 1:
        pattern = sp.kron(np.ones((n_fields, n_fields), dtype=np.int32), pattern)
    pattern = pattern.tocsr()
    pattern.data[:] = 1
    pattern.sort_indices()
    return pattern


def _color_columns(pattern):
    """Greedy colouring of the columns of ``pattern`` such that no two columns of one colour share a
    row, so one JVP per colour recovers every entry (Curtis, Powell & Reid, IMA J. Appl. Math. 13,
    1974). Host-side and structural."""
    conflict = (pattern.T @ pattern).tocsr()
    n = pattern.shape[1]
    color = np.full(n, -1, dtype=int)
    indptr, indices = conflict.indptr, conflict.indices
    for j in range(n):
        taken = color[indices[indptr[j] : indptr[j + 1]]]
        taken = taken[taken >= 0]
        if taken.size == 0:
            color[j] = 0
            continue
        free = np.ones(taken.max() + 2, dtype=bool)
        free[taken] = False
        color[j] = int(np.argmax(free))
    return color, int(color.max()) + 1


def _assemble_sparse(fun, u, pattern, color, n_colors):
    """The Jacobian of ``fun`` at ``u`` as a BCOO on ``pattern``: one JVP per colour, batched.
    Traceable and differentiable in whatever ``fun`` closes over (the pattern is a host constant)."""
    import jax
    import jax.experimental.sparse as jsp

    coo = pattern.tocoo()
    rows, cols = coo.row, coo.col
    u = jnp.asarray(u)
    seeds = jnp.asarray((color[None, :] == np.arange(n_colors)[:, None]).astype(np.float64), dtype=u.dtype)
    _, lin = jax.linearize(fun, u)
    products = jax.vmap(lin)(seeds)  # (n_colors, n)
    data = products[jnp.asarray(color[cols]), jnp.asarray(rows)]
    return jsp.BCOO((data, jnp.asarray(np.stack([rows, cols], axis=1))), shape=pattern.shape)


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
    subdomain. A ``jno.shape`` uses the analytic, shapely-free :meth:`shape.contains` (2-D and 3-D — the
    primary path); a shapely geometry falls back to shapely (2-D mesh-conforming regions in
    ``polygon_domain``, which stay shapely by scope)."""
    from .geometry import shape

    p = np.asarray(pts)
    if isinstance(geom, shape):
        mask = np.asarray(geom.contains(p[:, : geom.dim]))
    else:
        import shapely

        mask = np.asarray(shapely.contains_xy(geom.buffer(1e-9), p[:, 0], p[:, 1]))
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
    its structure, exactly as ``_set_normal`` handles a flux. (``u.tt`` used to be replaced by ``val`` as
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
        self._pde, self._dirichlet, self._neumann, self._ic, self._vel_ic = [], [], [], [], []
        self._periodic_axes = []  # grid axes tied by a `u(A) - u(B)` periodic constraint (structured only)
        for c in constraints:
            # Classify by structure (not by which region tag), so a value-only pin works on ANY region —
            # a boundary edge OR a geometric sub-region (`domain.region(name, geom)`, used by coupled /
            # domain-decomposition solves to pin a subdomain's complement to a neighbour's field):
            #   * a periodic tie `u(A) - u(B)` (opposite faces)  → wrap the grid axis (check first);
            #   * a normal derivative `ui.d(n, ...)`           → a Neumann/Robin flux row;
            #   * `u.t` on the `initial` region                → the initial velocity (u_tt problems);
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
            elif _region_tag(c) == "initial" and _has_temporal(c):
                self._vel_ic.append(c)
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
        self._time_order = max(_temporal_order(c) for c in self._pde)
        if self._time_order > 2:
            raise NotImplementedError(
                f"jno.fdm([...]): a time derivative of order {self._time_order} was found; only `u.t` "
                "(first order) and `u.tt` (second order) are supported."
            )
        if self._vel_ic and self._time_order != 2:
            raise ValueError(
                "jno.fdm([...]): an initial velocity `ui0.t - v0` was given, but the PDE has no `u.tt` "
                "term. An initial velocity only belongs to a second-order-in-time problem."
            )
        if self._vel_ic and not self._ic:
            raise ValueError(
                "jno.fdm([...]): an initial velocity `ui0.t - v0` was given without an initial displacement "
                "`u(xi, yi) - u0` — a second-order-in-time problem needs both (the velocity defaults to 0)."
            )
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
                    "domain with `jno.shape.rect(...).structured().domain()`. Periodic on an unstructured mesh is "
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
        context = self._eval_context(spatial_tags)
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

    def _eval_context(self, spatial_tags):
        """Evaluation context for a strong-form term: every spatial tag collocates at the mesh nodes, and
        ``domain.cell_size`` resolves to the per-node spacing :meth:`_node_spacing`."""
        context = {t: self._pts for t in spatial_tags}
        context["cell_size"] = self._node_spacing()[:, None]
        return context

    def _node_spacing(self):
        """``domain.cell_size`` in the strong form: the **node spacing** ``h``, per node the mean over its
        incident cells of ``(d!·|K|)^(1/d)``.

        On a structured grid this is exactly the grid spacing, in 2-D (right triangles, ``|K| = h²/2``) and
        3-D (Kuhn tetrahedra, ``|K| = h³/6``), so textbook stencil identities hold as written: first-order
        upwinding is ``b*ui.x - abs(b)*h/2*ui.xx`` with ``h = domain.cell_size``. On an unstructured mesh
        it is the leg of the right simplex with the same size (0.93·a for an equilateral triangle of side
        ``a``); on an anisotropic grid it is the geometric mean ``(hx·hy)^(1/2)``. **Differs from
        ``jno.fem``**, where ``cell_size`` is ``|K|^(1/d)`` at each quadrature point (``h/√2`` on the same
        right triangles). Differentiable in the mesh coordinates."""
        if getattr(self, "_h_nodes", None) is None:
            import jax

            # A constant of the mesh: computed concretely even when first asked for inside a trace (the
            # parametric solve), or the cached value would be a leaked tracer.
            with jax.ensure_compile_time_eval():
                self._h_nodes = self._node_spacing_now()
        return self._h_nodes

    def _node_spacing_now(self):
        """Per-node mean of ``(d!·|K|)^(1/d)`` over incident cells (see :meth:`_node_spacing`)."""
        pts, cells = _mesh(self.domain)
        dim = pts.shape[1]
        e = pts[cells[:, 1:]] - pts[cells[:, :1]]  # (C, dim, dim) edge vectors from vertex 0
        size = (jnp.abs(jnp.linalg.det(e))) ** (1.0 / dim)  # (d!·|K|)^(1/d), since |det| = d!·|K|
        total = jnp.zeros(self._N).at[cells.reshape(-1)].add(jnp.repeat(size, cells.shape[1]))
        count = jnp.zeros(self._N).at[cells.reshape(-1)].add(1.0)
        return total / jnp.maximum(count, 1.0)

    def _sparsity(self, key, fun, u):
        """``(pattern, colour, n_colours)`` for assembling ``fun``'s Jacobian, found once per problem.

        The stencil width depends on the operators (one ring for ``cotangent`` and the structured grid,
        two for a gradient of a gradient or the quadratic flux fit), so candidate patterns of growing
        radius are tried and each is **verified**: the assembled matrix must reproduce the matrix-free
        JVP on a random vector to 1e-10. Raises if none does, rather than solving with a wrong matrix."""
        cache = self.__dict__.setdefault("_sparsity_cache", {})
        if key in cache:
            return cache[key]
        import jax

        cells = _mesh(self.domain)[1]
        extra = [(np.asarray(sec), np.asarray(main)) for sec, main in self._periodic_rows()]
        n = int(np.asarray(u).size)
        n_fields = n // self._N
        v = jnp.asarray(np.random.default_rng(0).standard_normal(n), dtype=jnp.asarray(u).dtype)
        ref = jax.jvp(fun, (jnp.asarray(u),), (v,))[1]
        scale = float(jnp.linalg.norm(ref)) or 1.0
        for radius in (1, 2, 3):
            pattern = _stencil_pattern(cells, self._N, radius, n_fields=n_fields, extra_pairs=extra)
            color, n_colors = _color_columns(pattern)
            A = _assemble_sparse(fun, u, pattern, color, n_colors)
            if float(jnp.linalg.norm(A @ v - ref)) <= 1e-10 * scale:
                cache[key] = (pattern, color, n_colors)
                return cache[key]
        raise ValueError(
            "jno.fdm: could not assemble this strong-form operator as a sparse matrix (no stencil radius up "
            "to 3 reproduces its matrix-free action), so the linear=/precond= slots that need a matrix "
            "cannot be used. Leave linear=/precond= unset to keep the matrix-free default."
        )

    def _sparse_operator(self, key, fun, u):
        """``fun``'s Jacobian at ``u`` as a sparse BCOO (see :meth:`_sparsity`)."""
        pattern, color, n_colors = self._sparsity(key, fun, u)
        return _assemble_sparse(fun, u, pattern, color, n_colors)

    def _is_affine(self, key, fun, n):
        """Is ``fun`` affine in the DOF vector? Its JVP must be the same at two different states.
        Decided once per problem, eagerly, with the parameters at their current values."""
        cache = self.__dict__.setdefault("_affine_cache", {})
        if key not in cache:
            import jax

            rng = np.random.default_rng(1)
            v = jnp.asarray(rng.standard_normal(n))
            w = jnp.asarray(rng.standard_normal(n))
            j0 = jax.jvp(fun, (jnp.zeros(n),), (v,))[1]
            j1 = jax.jvp(fun, (w,), (v,))[1]
            cache[key] = bool(jnp.linalg.norm(j1 - j0) <= 1e-10 * (float(jnp.linalg.norm(j0)) or 1.0))
        return cache[key]

    def _mass_coefficient(self):
        """Per-node coefficient ``c`` on ``u.t`` (the diagonal mass ``M = diag(c)``), via the two-probe
        ``c = F(u.t=1) − F(u.t=0)`` (:func:`_set_temporal`) — the spatial residual cancels between the
        probes, leaving ``c``. A plain ``ui.t - 𝒩(u)`` gives ``c = 1``; a ``ρcₚ(x)·ui.t`` term gives the
        node values of ``ρcₚ(x)``. ``c`` must be constant in ``u`` (a nonlinear mass ``c(u)·u.t`` raises).
        Single-field only (the transient march is)."""
        return self._time_coefficient(1.0, 0.0, "`u.t` mass coefficient", "nonlinear mass `c(u)·u.t`")

    def _time_coefficient(self, t_val, tt_val, what, example):
        """Per-node coefficient of the time derivative selected by the probe ``(u.t, u.tt) = (t_val,
        tt_val)``: ``F(t_val, tt_val) − F(0, 0)``. Raises if it depends on ``u`` (probed at two constant
        states), because the diagonal-mass march cannot carry a state-dependent coefficient."""
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
        context = self._eval_context(spatial_tags)
        lid, base = self.unknown.layer_id, self.unknown.module

        def probe(u_val):
            def at(tv, ttv):
                mod = eqx.tree_at(lambda m: m.value, base, jnp.full(self._N, u_val, base.value.dtype))
                ev = TraceEvaluator(params={lid: mod})
                return jnp.asarray(ev.evaluate(_set_temporal(expr, tv, ttv), context=context, var_bindings={})).reshape(-1)

            return at(t_val, tt_val) - at(0.0, 0.0)

        c0 = probe(0.0)
        if not bool(jnp.allclose(c0, probe(1.0), atol=1e-6, rtol=1e-6)):  # u-dependence ⇒ nonlinear mass
            raise ValueError(
                f"jno.fdm([...]): the {what} depends on u (a {example}) — only a constant "
                "or coordinate-dependent coefficient is supported."
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
        mc = self.domain.mesh_connectivity
        # `boundary_edges` indexes the boundary-node list, not the mesh. A gmsh mesh numbers its boundary
        # nodes first, so the two coincide there by accident; on a structured grid they do not, and reading
        # the local indices as global ones silently dropped every flux condition.
        local = np.asarray(mc["boundary_edges"], dtype=int)
        edges = np.asarray(mc["boundary_indices"], dtype=int)[local]  # (E, 2) global node pairs
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
        context = self._eval_context(spatial_tags)
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
            scheme = getattr(jac, "scheme", None) or "finite_difference"
            _, grad_method, _ = _D.parse_fd_scheme(scheme)
            v0, v1 = self._flux_value_fn(c, 0.0, extra_params), self._flux_value_fn(c, 1.0, extra_params)
            f0, f1, f2 = v0(probe), v1(probe), self._flux_value_fn(c, 2.0, extra_params)(probe)
            if not bool(jnp.allclose(f2 - f0, 2.0 * (f1 - f0), atol=1e-6)):
                raise ValueError(
                    "jno.fdm([...]): a flux boundary condition must be affine in the normal derivative "
                    "∂u/∂n — e.g. Neumann `ui.d(n) - h` or Robin `ui.d(n) + α*(u - u∞)`. A condition "
                    "nonlinear in ∂u/∂n is not supported."
                )
            rows.append((jnp.asarray(idx), nrm, self._flux_gradient_fn(idx, scheme, grad_method), v0, v1))
        return rows

    def _interior_is_five_point(self):
        """Does every second derivative in the PDE use a five-point-type stencil — the ``cotangent``
        Laplacian, or any Hessian on a structured grid — rather than a gradient of the area-weighted
        gradient (the unstructured ``.d2`` default, or nested partials such as ``(κ * ui.x).x``)?

        The flux closure has to match. A gradient-of-gradient stencil reads the area-weighted gradient at
        the boundary nodes, so imposing ``∂u/∂n`` on exactly that gradient is its consistent (second-order)
        closure. A five-point-type stencil never reads it, and then a first-order one-sided gradient caps
        the solve at first order; it needs the quadratic fit instead. Measured on the unit square with a
        Neumann edge (rel. error at h = 0.1 / 0.05 / 0.025):

        =====================  ==========================  ==========================
        interior               area-weighted closure       quadratic closure
        =====================  ==========================  ==========================
        ``.d2`` (default)      2.9e-2 / 7.5e-3 / 1.8e-3    3.6e-2 / 1.5e-2 / 5.7e-3
        ``cotangent``          8.2e-3 / 3.6e-3 / 2.2e-3    5.1e-3 / 2.3e-3 / 7.4e-4
        =====================  ==========================  ==========================
        """
        from .trace import Hessian, Jacobian

        structured = self.domain.mesh_connectivity.get("grid") is not None
        five_point, other = False, False

        def walk(node):
            nonlocal five_point, other
            n = _unwrap(node)
            if isinstance(n, Hessian):
                if structured or ":cotangent" in str(getattr(n, "scheme", "")):
                    five_point = True
                else:
                    other = True
            elif isinstance(n, Jacobian) and _has_jacobian(n.target):
                other = True  # a nested partial is a gradient of the area-weighted gradient
            for c in _iter(n):
                walk(c)

        def _has_jacobian(node):
            n = _unwrap(node)
            return isinstance(n, Jacobian) or any(_has_jacobian(c) for c in _iter(n))

        for c in self._pde:
            walk(c)
        return not other and (five_point or structured)

    def _flux_gradient_fn(self, idx, scheme, grad_method):
        """``u ↦ ∇u`` at the flux nodes ``idx``, chosen to match the interior stencil (see
        :meth:`_interior_is_five_point`): the second-order quadratic fit (:func:`_quadratic_gradient`) for a
        five-point-type interior, the area-weighted gradient for a gradient-of-gradient interior. An
        explicitly chosen sub-scheme (``":lsq"``, ``":uniform"``, …) keeps the gradient it names."""
        idx = np.asarray(idx)
        if ":" in scheme or not self._interior_is_five_point():
            return lambda u: gradient(u, self.domain, method=grad_method)[idx]
        cells = _mesh(self.domain)[1]
        nbrs = _two_ring(cells, self._N, idx)
        pts, jidx = self._pts, jnp.asarray(idx)
        return lambda u: _quadratic_gradient(u, pts, jidx, nbrs)

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

    def _initial_velocity(self):
        """Initial velocity ``v0`` (shape ``(N,)``) from the ``ui0.t - v0`` condition(s) on the ``initial``
        region, as :func:`jno.fem` reads it for ``u_tt``; zero when none is given."""
        v0 = jnp.zeros(self._N)
        for c in self._vel_ic:
            idx = self._region_nodes(_region_tag(c))
            inner = _unwrap(c)
            if getattr(inner, "op", None) != "-":
                raise ValueError(f"jno.fdm([...]): write an initial velocity as `ui0.t - v0`; got {c!r}.")
            g_node = inner.right if _has_temporal(inner.left) else inner.left  # v0 is the side without u.t
            v0 = v0.at[jnp.asarray(idx)].set(self._eval_g(g_node, idx))
        return v0

    def solve(self, nonlinear=None, x0=None, profile=False, time=None, *, linear=None, precond=None):
        """Solve the strong-form system. **Steady** problems fold the Dirichlet rows into the residual
        (``u - g`` on the region) and hand it to the same ``jno.solve`` Newton–Krylov + ``custom_root``
        machinery ``jno.fem`` uses (linear/nonlinear uniform, differentiable for inverse problems).
        **Transient** problems (an ``u(initial) - u0`` condition is present) march by method-of-lines —
        ``t_span`` and the step count come from ``domain.time`` and the initial state from the IC — and
        return the trajectory (``(n_save, N)``); ``x0`` is rejected (the IC owns the initial state).

        ``linear=`` and ``precond=`` are the same solver slots as ``fem.solve``'s: any ``jno.solve``
        linear solver (``cg``, ``bicgstab``, ``gmres``, ``lu``, …) and any ``jno.precond`` spec (``jacobi``,
        ``amg``, ``gmg``, …). Setting either assembles the operator as a sparse matrix once — for a linear
        problem the whole system, for a nonlinear one the tangent — so matrix-based solvers and
        preconditioners apply, in steady solves and in every time step. Left unset, the matrix-free
        default is unchanged.

        ``time=`` selects the time scheme exactly as ``fem.solve(time=…)`` does — ``jno.solve.theta(θ)``
        (Crank–Nicolson at θ=0.5), ``jno.solve.adaptive(…)`` (step-doubling adaptive step size), or
        ``jno.solve.exponential(…)`` — defaulting to backward Euler. ``profile=True`` runs the (eager,
        non-parametric) solve inside a JAX Perfetto trace and writes it to ``./jno_traces``."""

        def _run():
            if self._transient:
                if x0 is not None:
                    raise ValueError("jno.fdm([...]): x0= is rejected for a transient problem — the IC owns the state.")
                return self._march(nonlinear=nonlinear, time=time, linear=linear, precond=precond)
            trainable = self._trainable_params()
            if trainable:
                return self._parametric_node(trainable, nonlinear=nonlinear, x0=x0, linear=linear, precond=precond)
            return self._steady_solve(nonlinear=nonlinear, x0=x0, linear=linear, precond=precond)

        if not profile:
            import jax

            from .utils.solver.solver_api import clear_gate_failures, raise_if_gate_failed

            # A linear solve inside a compiled march or Newton loop refuses non-convergence through a
            # callback, whose raise can be lost; it is also recorded, and drained here once the result is
            # concrete — the same boundary fem.solve drains at. A deferred (crux) node is drained by the
            # caller that evaluates it.
            clear_gate_failures()
            result = _run()
            if isinstance(result, jax.Array):
                jax.block_until_ready(result)
                raise_if_gate_failed()
            return result
        from .utils.profiling import profile_solve

        return profile_solve(_run, label=f"fdm profile · {self._N} nodes · {'transient' if self._transient else 'steady'}")

    def _steady_solve(self, *, nonlinear=None, x0=None, extra_params=None, extra_pins=None, linear=None, precond=None):
        """The steady solve: fold the flux and Dirichlet rows into the residual and hand it to the
        ``jno.solve`` Newton–Krylov driver. ``extra_params`` carries the current values of any trainable
        ``jno.np.parameter`` (from :meth:`_parametric_node`). ``extra_pins`` is an ``(idx, values)`` pair
        of nodes pinned to given values on top of the authored BCs — the interface pin a coupled /
        domain-decomposition Schwarz step applies (:meth:`solve_pinned`). The plain eager solve is
        compiled once per problem (:meth:`_compiled_steady`)."""
        import jax

        N, single = self._N, self._nf == 1
        u0 = jnp.zeros(self._Ntot) if x0 is None else jnp.asarray(x0).reshape(-1)
        if linear is not None or precond is not None:
            sol = self._slot_steady(nonlinear, linear, precond, x0, u0, extra_params, extra_pins)
        elif extra_params is None and extra_pins is None and not isinstance(u0, jax.core.Tracer):
            sol = self._compiled_steady(nonlinear, u0)
        else:
            residual_with_bc = self._steady_residual(extra_params, extra_pins)
            driver = nonlinear or _solve.newton(**_fd_newton_tolerances(residual_with_bc, u0))
            sol = driver(residual_with_bc, u0, linear_solve=_structured_linear_solve(self.domain) if single else None)
        return sol if single else sol.reshape(self._nf, N)  # coupled: (nf, N), one row per field

    def _slot_steady(self, nonlinear, linear, precond, x0, u0, extra_params, extra_pins):
        """The steady solve through ``fem.solve``'s ``linear=`` / ``precond=`` slots, on the assembled
        sparse operator. A linear problem is one ``(A, b)`` solve composed exactly as ``jno.fem`` composes
        it; a nonlinear one runs the composed Newton, whose direct variant gets the assembled tangent."""
        import jax

        from .utils.solver.solver_api import compose_linear_solve_fn, compose_nonlinear_solve_fn

        self._check_precond_shape(precond, self._Ntot)
        residual = self._steady_residual(extra_params, extra_pins)
        zeros = jnp.zeros(self._Ntot)
        # Structure (linearity, sparsity) is decided on a CONCRETE residual: the trainable parameters at
        # their current values, so this works when the solve itself runs inside a crux trace.
        with jax.ensure_compile_time_eval():
            probe = self._steady_residual({lid: n.model.module for lid, n in self._trainable_params().items()})
            linear_problem = self._is_affine("steady", probe, self._Ntot)
            self._sparsity("steady", probe, jnp.zeros(self._Ntot))  # the pattern search needs values
        if linear_problem:
            if nonlinear is not None:
                raise ValueError(
                    "jno.fdm: nonlinear= given, but this problem is linear — there is no Newton loop to "
                    "configure. Drop nonlinear=, or pick the linear solver with linear=."
                )
            A = self._sparse_operator("steady", residual, zeros)
            return compose_linear_solve_fn(linear, precond, x0, fem=self)(A, -residual(zeros))
        tangent = lambda w: self._sparse_operator("steady", residual, w)  # noqa: E731
        if nonlinear is None:
            # FDM can always assemble its tangent, so the Newton that uses it is the default here: it is
            # what lets jacobi / amg / gmg precondition a nonlinear strong-form solve at all (the
            # matrix-free JVP has no diagonal or matrix to give them).
            nonlinear = _solve.newton(direct=True)
        precond = self._frozen_precond(precond, lambda: tangent(u0 if not isinstance(u0, jax.core.Tracer) else zeros))
        driver = compose_nonlinear_solve_fn(nonlinear, linear, precond, fem=self)
        return driver(residual, u0, jacobian=tangent)

    def _frozen_precond(self, precond, representative):
        """A preconditioner that must see a concrete matrix (amg, gmg, ilu) cannot be set up inside the
        traced Newton loop, so it is set up once, here, on the assembled tangent at the initial guess —
        the frozen-preconditioner trade the transient march makes too (it changes how fast Krylov
        converges, never what it converges to). Traceable ones (jacobi) are left per-linearization."""
        if precond is None:
            return None
        from .utils.solver.solver_api import (
            LinearOperator,
            PrecondContext,
            _FrozenMarchPrecond,
            _specs_in,
            materialize_precond,
            prepare_precond,
        )

        leaves = [s for s in _specs_in(precond) if not getattr(s, "pairs", None) and getattr(s, "spec", None) is None]
        if all(bool(getattr(s, "traceable", True)) for s in leaves):
            return precond
        prepare_precond(precond, self)
        return _FrozenMarchPrecond(
            materialize_precond(precond, PrecondContext(LinearOperator(representative()), self)), precond
        )

    def _check_precond_shape(self, precond, n):
        """``gmg`` is a V-cycle on ONE scalar grid field. A coupled system or the augmented ``[u; v]`` state
        of a ``u.tt`` problem is several fields long, and the V-cycle would reshape it onto the grid."""
        if precond is None:
            return
        from .precond import _GMG
        from .utils.solver.solver_api import _specs_in

        if n != self._N and any(isinstance(s, _GMG) for s in _specs_in(precond)):
            raise ValueError(
                f"jno.precond.gmg() preconditions a single scalar field on the grid ({self._N} nodes), but "
                f"this system has {n} unknowns (a coupled system, or the [u; v] state of a u.tt problem). "
                "Use jno.precond.amg(), which works on any assembled operator."
            )

    def _steady_residual(self, extra_params=None, extra_pins=None):
        """The steady residual with every boundary row folded in, as a function of the DOF vector."""
        import jax

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
            for idx, nrm, grad_fn, v0, v1 in flux_rows:
                flux = jnp.sum(grad_fn(u) * nrm, axis=1)  # ∇u·n at the edge nodes, differentiable
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

        # Jitted so the trace evaluator runs once per shape: Newton, the Krylov tangent and the adjoint
        # all call this, and each un-jitted call re-walked the whole expression tree in Python.
        return jax.jit(residual_with_bc)

    def _data_fingerprint(self):
        """Identity and current values of every model in the constraints other than the unknowns — the
        data a compiled solve bakes in (a known nodal field, a network coefficient). Part of the
        :meth:`_compiled_steady` cache key, so changing such a value recompiles instead of silently
        solving with the old one."""
        import hashlib

        import jax

        from .trace import ModelCall

        seen, parts = set(), []

        def walk(n):
            n = _unwrap(n)
            if isinstance(n, ModelCall) and all(n.model is not u for u in self.unknowns) and id(n.model) not in seen:
                seen.add(id(n.model))
                digest = hashlib.blake2b(digest_size=16)
                for leaf in jax.tree_util.tree_leaves(getattr(n.model, "module", None)):
                    if hasattr(leaf, "shape"):
                        digest.update(np.asarray(leaf).tobytes())
                parts.append((id(n.model), digest.hexdigest()))
            for c in _iter(n):
                walk(c)

        for c in self._pde + self._dirichlet + self._neumann + self._ic:
            walk(c)
        return tuple(sorted(parts))

    def _compiled_steady(self, nonlinear, u0):
        """The eager steady solve, **compiled once per problem** and reused.

        Uncompiled, every ``.solve()`` re-traced Newton, the Krylov solve and (on a structured grid) the
        multigrid V-cycle, then ran them one primitive at a time: measured at 16 641 nodes, 4.3 s of
        tracing and 3.7 s of dispatch for a solve whose arithmetic takes milliseconds. The driver's own
        convergence guard is blind under ``jit``, so the verdict is taken here on the concrete result,
        against the tolerances the ``nonlinear=`` spec carries, and a stall raises exactly as before."""
        import jax

        from .utils.solver.solver_api import record_nonlinear_verdict

        cache = self.__dict__.setdefault("_steady_cache", {})
        key = (id(nonlinear), self._data_fingerprint())
        entry = cache.get(key)
        if entry is None or entry[0] is not nonlinear:
            residual = self._steady_residual()
            driver = nonlinear or _solve.newton(**_fd_newton_tolerances(residual, u0))
            linear = _structured_linear_solve(self.domain) if self._nf == 1 else None
            fn = jax.jit(lambda u_init: driver(residual, u_init, linear_solve=linear))
            entry = cache[key] = (nonlinear, driver, residual, fn)
        _, driver, residual, fn = entry
        sol = fn(u0)
        who = getattr(driver, "name", None) or "newton"
        r_end, bound, ok = record_nonlinear_verdict(residual, sol, u0, driver, who)
        if ok is False:
            raise RuntimeError(
                f"jno.fdm: {who} did not converge: residual norm {r_end:.3e} against the tolerance "
                f"{bound:.3e}. The last iterate is NOT a root -- raise max_steps, loosen atol/rtol, "
                "globalize the iteration (jno.solve.newton(line_search=True) or damping<1), or start "
                "from a better x0."
            )
        return sol

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
                for idx, nrm, grad_fn, v0, v1 in flux_rows:
                    flux = jnp.sum(grad_fn(u) * nrm, axis=1)
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

    def _parametric_node(self, trainable, *, nonlinear=None, x0=None, linear=None, precond=None):
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
            return self._steady_solve(nonlinear=nonlinear, x0=x0, extra_params=extra, linear=linear, precond=precond)

        node = FunctionCall(_solve, param_nodes, name="fdm_solve")
        node._domain = self.domain  # so jno.core infers the domain from the graph (no explicit domain= needed)
        return node

    def _march(self, *, nonlinear=None, save_ts=None, time=None, linear=None, precond=None):
        """Method-of-lines march of ``u̇ = -R_spatial(u)`` reusing jNO's solver-agnostic
        :class:`SemidiscreteTimeBlock` integrator (``custom_root`` differentiable). ``M = I`` on interior
        nodes; **Dirichlet and Neumann/Robin flux nodes carry a zero mass row** — Dirichlet pins to ``g``,
        a flux node imposes the SAME ``a·(∇u·n) + b`` the steady solve folds in, as an index-1 DAE
        constraint the boundary value satisfies at each instant (its value is determined by the interior via
        the flux). ``t_span``/``dt`` come from ``domain.time``; ``time=`` picks the scheme (backward Euler
        by default)."""
        import jax.experimental.sparse as jsparse

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

        def boundary_rows(wn, r):  # overwrite the flux and Dirichlet rows of r with their algebraic constraints
            for idx, nrm, grad_fn, v0, v1 in flux_rows:  # a·(∇u·n) + b — the same folding as _steady_solve
                flux = jnp.sum(grad_fn(wn) * nrm, axis=1)
                b = v0(wn)
                a = v1(wn) - b
                r = r.at[idx].set(a[idx] * flux + b[idx])
            return jnp.where(bmask, wn - bvals, r)  # Dirichlet wins over flux on an overlapping node

        slots = dict(nonlinear=nonlinear, linear=linear, precond=precond)
        if self._time_order == 2:
            return self._march_second_order(spatial_res, boundary_rows, algebraic, t0, t1, dt, save_ts, time, slots)

        c_nodes = self._mass_coefficient()  # u.t coefficient: 1 for a plain u.t, c(x) for ρcₚ(x)·u.t
        diag = jnp.stack([jnp.arange(self._N), jnp.arange(self._N)], axis=1)
        M = jsparse.BCOO((jnp.where(algebraic, 0.0, c_nodes), diag), shape=(self._N, self._N))  # 0: Dirichlet+flux

        def residual(wn, t, args):  # M u̇ + R = 0 → interior u̇ = -R_spatial; flux/Dirichlet rows algebraic
            return boundary_rows(wn, spatial_res(wn))

        return self._run_block(M, residual, self._initial_state(), (t0, t1, dt), {}, save_ts, time, slots)

    def _run_block(self, M, residual, state0, window, metadata, save_ts, time, slots):
        """Build the semidiscrete block ``M ẏ + R(y) = 0`` and march it.

        With no solver slots it is a residual block, stepped by the matrix-free Newton–Krylov (the
        historic default). With ``linear=`` / ``precond=`` it is composed exactly as ``fem.solve`` composes
        a transient: an affine ``R`` becomes a **linear** block (``A`` assembled once as a sparse matrix,
        ``c = −R(0)``), so every step is one preconditioned linear solve and an AMG or LU setup is built
        once before the march; a nonlinear ``R`` keeps the Newton step, with the assembled tangent
        available to a direct solver. A ``nonlinear=`` slot configures the per-step Newton."""
        from .utils.solver.backend_blocks import SemidiscreteTimeBlock, _block_time_grid
        from .utils.solver.solver_api import compose_transient_step_solvers

        t0, t1, dt = window
        nonlinear, linear, precond = slots["nonlinear"], slots["linear"], slots["precond"]
        common = dict(state0=state0, t0=float(t0), t1=float(t1), dt=float(dt), metadata=dict(metadata))
        frozen = lambda y: residual(y, float(t0), {})  # noqa: E731  (R does not depend on t)
        n = int(state0.size)
        self._check_precond_shape(precond, n)
        if linear is not None or precond is not None:
            import jax

            zeros = jnp.zeros(n)
            with jax.ensure_compile_time_eval():  # structure is decided on concrete values
                self._sparsity("march", frozen, jnp.zeros(n))
                linear_problem = self._is_affine("march", frozen, n)
            if linear_problem:
                A = self._sparse_operator("march", frozen, zeros)
                block = SemidiscreteTimeBlock(M=M, A=A, affine_bias=-frozen(zeros), **common)
            else:
                block = SemidiscreteTimeBlock(
                    mass=lambda t, args: M,
                    residual=residual,
                    jacobian=lambda w, t, args: self._sparse_operator("march", lambda y: residual(y, t, args), w),
                    **common,
                )
        else:
            block = SemidiscreteTimeBlock(mass=lambda t, args: M, residual=residual, **common)
        from .utils.solver.timeschemes import _ExponentialScheme

        if isinstance(time, _ExponentialScheme):
            # Measured: with a slot set the block is linear and the exponential scheme RUNS, but it came
            # back 4.8e-3 off a converged reference where Crank-Nicolson at the same step was 2.0e-5. It
            # forms exp(-dt M⁻¹A), and an FDM block is a DAE: its Dirichlet and flux rows are algebraic,
            # with a zero mass. Refused rather than returned.
            raise NotImplementedError(
                "jno.fdm: the exponential time scheme integrates a LINEAR block with an invertible mass "
                "matrix, and a strong-form "
                "march is a DAE — its boundary rows are algebraic constraints with zero mass. Use a "
                "θ-scheme: jno.solve.theta(0.5) (Crank–Nicolson) or the backward-Euler default."
            )
        steppers = (None, None)
        if linear is not None or precond is not None or nonlinear is not None:
            steppers = compose_transient_step_solvers(nonlinear, linear, precond, self, block, scheme=time, state=state0)
        ts = _block_time_grid(block) if save_ts is None else jnp.asarray(save_ts)
        return _integrate_transient(block, ts, time, *steppers)

    def _march_second_order(self, spatial_res, boundary_rows, algebraic, t0, t1, dt, save_ts, time, slots):
        """March ``m·u_tt + c·u_t + R_spatial(u) = 0`` as the first-order augmented system in
        ``y = [u; v]`` with ``v = u_t`` — the same reduction :func:`jno.fem` makes for ``u_tt``:

            u̇ − v = 0,        m·v̇ + c·v + R_spatial(u) = 0.

        ``m`` (inertia) and ``c`` (damping) are per-node coefficients found by probing (:func:`_set_temporal`).
        Dirichlet and flux nodes stay algebraic: their ``u`` row is the boundary constraint and their ``v``
        row pins ``v = 0`` (``v`` there feeds no other equation, and only ``u`` is returned). The default
        scheme is θ = ½ (trapezoidal, Newmark average acceleration; Newmark 1959, J. Eng. Mech. Div. 85),
        which conserves the energy of an undamped linear wave — backward Euler would damp it. Returns the
        ``u`` trajectory ``(n_save, N)``."""
        import jax.experimental.sparse as jsparse

        N = self._N
        m_nodes = self._time_coefficient(0.0, 1.0, "`u.tt` inertia coefficient", "nonlinear inertia `m(u)·u.tt`")
        c_nodes = self._time_coefficient(1.0, 0.0, "`u.t` damping coefficient", "nonlinear damping `c(u)·u.t`")
        mass = jnp.concatenate([jnp.where(algebraic, 0.0, 1.0), jnp.where(algebraic, 0.0, m_nodes)])
        diag = jnp.stack([jnp.arange(2 * N), jnp.arange(2 * N)], axis=1)
        M = jsparse.BCOO((mass, diag), shape=(2 * N, 2 * N))

        def residual(y, t, args):
            u, v = y[:N], y[N:]
            ru = boundary_rows(u, -v)  # interior: u̇ = v; boundary: the algebraic constraint on u
            rv = jnp.where(algebraic, v, c_nodes * v + spatial_res(u))
            return jnp.concatenate([ru, rv])

        v0 = jnp.where(algebraic, 0.0, self._initial_velocity())
        state0 = jnp.concatenate([self._initial_state(), v0])
        metadata = {"theta": 0.5, "second_order": True}
        return self._run_block(M, residual, state0, (t0, t1, dt), metadata, save_ts, time, slots)[:, :N]


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
