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

Scope: scalar fields, vector fields (``domain.unknown(value_shape=(c,))``), or a **coupled system** of several
``domain.unknown()`` fields (one PDE equation per unknown, equation *k* driving unknown *k*, steady or first order
in time, Dirichlet and flux conditions; ``.solve()`` returning ``(nf, N)``) — on a **2-D triangular or 3-D
tetrahedral mesh** or a structured grid. Real fields only: a complex value raises. The interior operators
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

import contextlib

import jax.numpy as jnp
import numpy as np

from . import precond as jno_precond
from . import solve as _solve
from .differential_operators import DifferentialOperators as _D
from .trace_evaluator import fd_grid

__all__ = ["fdm", "laplacian", "gradient"]


def _rtol(dtype=None) -> float:
    """The relative tolerance to which two floating-point results that should be equal are compared (an
    affine probe, a symmetry probe, a node on a lattice): ``√ε`` of ``dtype`` (the default float if ``None``),
    1.5e-8 in float64 and 3.5e-4 in float32. It replaces fixed 1e-10 / 1e-9 / 1e-6 thresholds, which were
    below float32's resolution (every float32 operator probed "nonlinear" and "non-symmetric")."""
    return float(np.sqrt(np.finfo(jnp.result_type(float) if dtype is None else dtype).eps))


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


def _pcg(matvec, b, precond, tol, maxiter=1000):
    """Preconditioned conjugate gradients (Hestenes & Stiefel, J. Res. Nat. Bur. Stand. 49 (1952) 409) to a
    relative residual ``tol``; returns ``(x, ‖r‖)``. Four vectors of state."""
    import jax

    stop = tol * jnp.linalg.norm(b)
    z = precond(b)
    state = (jnp.zeros_like(b), b, z, z, b @ z, 0)

    def cond(s):
        return (jnp.linalg.norm(s[1]) > stop) & (s[5] < maxiter)

    def body(s):
        x, r, _, p, rz, k = s
        Ap = matvec(p)
        a = rz / (p @ Ap)
        x, r = x + a * p, r - a * Ap
        z = precond(r)
        rz_new = r @ z
        return x, r, z, z + (rz_new / rz) * p, rz_new, k + 1

    x, r, *_ = jax.lax.while_loop(cond, body, state)
    return x, jnp.linalg.norm(r)


def _gmres_incremental(matvec, b, precond, tol, restart=30, maxiter=50, cycles=8):
    """GMRES that checks its residual every iteration (JAX's ``incremental`` method) -- the ``batched``
    method ``jno.solve.gmres`` uses finishes each 30-vector restart cycle whatever the convergence -- and
    whose tolerance is measured on the TRUE residual.

    JAX's ``gmres`` applies ``M`` on the LEFT, so it stops on ``‖M(b − Ax)‖``. Where ``M`` is not spectrally
    equivalent to the operator -- a ``jno.fd(order=4)`` stencil preconditioned by the second-order V-cycle --
    the two differ: measured 1e-10 preconditioned against 2.7e-8 true, which the caller's own gate then
    refused (`test_higher_order_poisson`). The outer loop restarts from the iterate and re-forms ``b − Ax``,
    so the tolerance means the same thing whatever ``M`` does; it costs one matvec per cycle. It stops at the
    tolerance, when a cycle no longer halves the residual (``M`` cannot do better), or after ``cycles``.
    Right-preconditioned FGMRES (:func:`jno.utils.solver.krylov.fgmres`) would measure the true residual
    directly, but keeps a second ``(restart, n)`` basis -- 4 GB more at 16.8M nodes."""
    import jax
    from jax.scipy.sparse.linalg import gmres

    nb = jnp.maximum(jnp.linalg.norm(b), jnp.finfo(b.dtype).tiny)

    def cond(s):
        _, rn, prev, k = s
        return (rn > tol * nb) & (rn <= 0.5 * prev) & (k < cycles)

    def body(s):
        x, rn, _, k = s
        dx, _ = gmres(
            matvec, b - matvec(x), tol=tol, restart=restart, maxiter=maxiter, M=precond, solve_method="incremental"
        )
        x = x + dx
        return x, jnp.linalg.norm(b - matvec(x)), rn, k + 1

    zero = jnp.zeros_like(b)
    state = (zero, nb * jnp.ones((), b.dtype), jnp.array(jnp.inf, b.dtype), jnp.array(0))
    x, rn, *_ = jax.lax.while_loop(cond, body, state)
    return x, rn


def _structured_linear_solve(domain, periodic=False, vcycle=None):
    """Inner linear solve for the matrix-free Newton–Krylov on a **structured grid**: GMRES rather than
    the driver's default BiCGStab. The reduced-Dirichlet 5-/7-point operator is nonsymmetric, and BiCGStab
    can break down on it (a strong-form ``u.d2(x)+u.d2(y)`` returns NaN), whereas GMRES is robust for
    nonsymmetric systems while staying matrix-free and differentiable (the driver firewalls it in
    ``custom_linear_solve``, so the reverse pass runs GMRES on ``Aᵀ``).

    The GMRES is preconditioned by the **operator-dependent V-cycle**
    (:func:`jno.utils.solver.lattice_mg.build`), built from ``vcycle_of`` -- a representative tangent, taken
    at the initial guess and frozen for the whole Newton loop, the same trade the march makes. A
    preconditioner changes how fast the Krylov solve converges, never what it converges to. Without one
    (an unstructured mesh, or no representative tangent) this is plain GMRES."""
    from .utils.solver.solver_api import LinearOperator

    # Unstructured: GMRES too. The strong-form operator is not symmetric, and the driver's BiCGStab broke
    # down on it: a linear 3-D cotangent problem with one Neumann face diverged to a Newton residual of
    # 5e24, where a direct solve gives 2.4e-3.
    if vcycle is None:
        gmres = _solve.gmres()
        return lambda mv, rhs: gmres(LinearOperator.from_matvec(mv), rhs)
    # Preconditioned, the iteration count is small, and `jno.solve.gmres` (JAX's batched method) would
    # finish a whole 30-vector cycle whatever the convergence -- 30 V-cycles where 3 suffice. Measured on a
    # 50-step heat march at 263k nodes: 34 s against 1.7 s for the incremental method, same answer.
    # Stops when the TRUE residual meets the tolerance, however many cycles that takes. A fixed budget was
    # tried instead (two restart cycles, on the inexact-Newton argument that a step needs only enough
    # progress to keep the outer iteration converging): at 66k nodes it left the step so far from solved
    # that Newton diverged on Bratu, residual 6.2e2 against a 5e-6 tolerance.
    return lambda mv, rhs: _gmres_incremental(mv, rhs, vcycle, _solve.gmres().tolerance)[0]


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
    grid = fd_grid(domain)  # structured-grid fast path, with the evaluated problem's periodic axes; else None
    if dim == 3:
        return _D.compute_fd_laplacian_3d_simple(u, pts, cells, dims=(0, 1, 2), method=method, grid=grid)
    return _D.compute_fd_laplacian_2d_simple(u, pts, cells, dims=(0, 1), method=method, grid=grid)


def gradient(u, domain, method: str = "area_weighted"):
    """FD gradient ``∇u`` of the nodal field ``u`` — shape ``(N, dim)``. ``method`` selects the stencil
    (``"area_weighted"`` default, ``"uniform"``, ``"inverse_distance"``, ``"least_squares"``); the same
    names apply on a 2-D triangular or 3-D tetrahedral mesh."""
    pts, cells = _mesh(domain)
    dim = int(getattr(domain, "dimension", 2))
    grid = fd_grid(domain)  # structured-grid fast path, with the evaluated problem's periodic axes; else None
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


def _fuse_fd_laplacian(expr, dim):
    """Rewrite ``c·(∂²u/∂x² + ∂²u/∂y² [+ ∂²u/∂z²])`` spelled with plain per-axis finite-difference second
    derivatives (``ui.xx + ui.yy``, ``-ui.d2(x) - ui.d2(y)``) into ONE cotangent Laplacian.

    On an unstructured mesh the per-axis default is a gradient of the area-weighted gradient, and that
    operator has a spurious oscillating mode (its lowest Dirichlet eigenvalue on the unit square is ~5.4,
    not 2π², and it does not refine away): advection–diffusion came out 2.09 off and Helmholtz near 5.4
    blew up. The cotangent Laplacian has no such mode. A sum is fused only when it names every spatial
    axis exactly once, with the same target and the same coefficient on each term; anything else (an
    anisotropic ``a·u_xx + b·u_yy``, a partial sum in 3-D, an explicit sub-scheme) is left as written."""
    from .trace import BinaryOp, Hessian, Literal

    def atom(node):  # (coefficient, target, variable) of `c * ∂²u/∂v²`, else None
        coef = 1.0
        n = _unwrap(node)
        if isinstance(n, BinaryOp) and n.op == "*":
            if isinstance(n.left, Literal) and isinstance(_unwrap(n.right), Hessian):
                coef, n = float(n.left.value), _unwrap(n.right)
            elif isinstance(n.right, Literal) and isinstance(_unwrap(n.left), Hessian):
                coef, n = float(n.right.value), _unwrap(n.left)
        if not isinstance(n, Hessian) or len(n.variables) != 1 or (n.scheme or "finite_difference") != "finite_difference":
            return None
        var = n.variables[0]
        if getattr(var, "axis", "spatial") != "spatial":
            return None
        return coef, n.target, var

    def signed_terms(node, sign=1.0):
        n = _unwrap(node)
        if isinstance(n, BinaryOp) and n.op in ("+", "-"):
            return signed_terms(n.left, sign) + signed_terms(n.right, sign if n.op == "+" else -sign)
        return [(sign, n)]

    def fuse_chain(node):
        terms = signed_terms(node)
        if len(terms) < dim:
            return node
        groups = {}
        for i, (sign, term) in enumerate(terms):
            a = atom(term)
            if a is not None:
                coef, target, var = a
                groups.setdefault((id(target), sign * coef, getattr(var, "tag", None)), []).append((i, target, var))
        replace, drop = {}, set()
        for (_tid, coef, _tag), members in groups.items():
            axes = [int(v.dim[0]) for _i, _t, v in members]
            if sorted(axes) != list(range(dim)):
                continue
            lap = Hessian(members[0][1], [v for _i, _t, v in members], "finite_difference:cotangent", trace=True)
            replace[members[0][0]] = (1.0, lap if coef == 1.0 else BinaryOp("*", Literal(coef), lap))
            drop.update(i for i, _t, _v in members[1:])
        if not replace:
            return node
        kept = [replace.get(i, t) for i, t in enumerate(terms) if i not in drop]
        sign0, out = kept[0]
        out = out if sign0 > 0 else BinaryOp("*", Literal(-1.0), out)
        for sign, term in kept[1:]:
            out = BinaryOp("+" if sign > 0 else "-", out, term)
        return out

    def visit(node):
        n = _unwrap(node)
        if (
            isinstance(n, Hessian)
            and getattr(n, "trace", False)
            and (n.scheme or "finite_difference") == "finite_difference"
            and sorted(int(v.dim[0]) for v in n.variables if getattr(v, "axis", "spatial") == "spatial") == list(range(dim))
            and len(n.variables) == dim
        ):
            # `ui.laplacian(x, y)` with the default scheme: the same operator, already in one node.
            return Hessian(n.target, list(n.variables), "finite_difference:cotangent", trace=True)
        if isinstance(n, BinaryOp):
            if n.op in ("+", "-"):
                fused = fuse_chain(n)
                if fused is not n:
                    return fused
            left, right = visit(n.left), visit(n.right)
            if left is not n.left or right is not n.right:
                return BinaryOp(n.op, left, right)
        return n

    return visit(expr)


def _has_time_variable(node):
    """Does a value expression use the temporal Variable (``tb`` in ``g(xb, yb, tb)``)?"""
    from .trace import Variable

    n = _unwrap(node) if not isinstance(node, (int, float)) else None
    if n is None:
        return False
    if isinstance(n, Variable):
        return getattr(n, "axis", None) == "temporal"
    return any(_has_time_variable(c) for c in _iter(n))


def _find_unknowns(constraints):
    """All ``domain.unknown()`` fields (nodal-field-parameter ModelCalls' Models) in the constraints, in
    **declaration** order — a **coupled** system has several, and ``.solve()`` returns one row per field in
    this order. (It used to be first-appearance order: listing the v-equation first silently swapped the
    returned ``u`` and ``v``.) At least one is required. The k-th PDE equation fills the k-th DOF block's
    interior rows, and each ``u_k(region) - g`` BC the boundary rows of its own field's block; the joint
    system is the same whichever order the equations are listed in."""
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
        and not str(v.tag).startswith(("n_", "fdm_bgrad_"))
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


class _TraceFDM:
    """Finite-difference system authored as a fem-style constraint list with ``u = domain.unknown()``:
    ``jno.fdm([-u.d2(x) - u.d2(y) - f, u(xb, yb) - g]).solve()``. Constraints are classified by the
    region their coordinate variables carry — the ``interior`` → the strong-form PDE residual, a
    boundary tag → a Dirichlet condition ``u(region) - g``, the ``initial`` region → the initial
    condition ``u(initial) - u0`` (exactly as in :func:`jno.fem`), and a boundary condition that
    **differentiates** the unknown (the field bound to the edge's tags ``ur = u.bind(x=xr, y=yr)``) → a
    boundary row at that edge's nodes, evaluated as written with boundary-accurate first derivatives
    (:meth:`_flux_row_fn`). Neumann ``ur.d(n) - h`` (``n = domain.variable(region, normals=True)``, or
    ``ur.d((nx, ny))``, or ``nx*ur.x + ny*ur.y``), Robin ``ur.d(n) + α(u - u∞)``, ``(κ*ur).d(n)``, an
    oblique or tangential derivative, a nonlinear flux — the whole edge equation is written with that
    edge's boundary tags. (Flux BCs are authored differently
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
        self.domain = self.unknown._fem_field_domain
        self._N = int(np.asarray(self.domain.mesh_connectivity["points"]).shape[0])  # nodes per field
        # A vector unknown (`domain.unknown(value_shape=(2,))`) is one DOF block per component, so the DOF
        # vector is [u_0 components…, u_1 components…] in declaration order; `_nf` counts BLOCKS.
        self._ncomp = [int(np.prod(np.shape(w.module.value)[1:], dtype=int)) for w in self.unknowns]
        self._block0 = [int(b) for b in np.cumsum([0] + self._ncomp)[:-1]]
        self._nf = int(sum(self._ncomp))
        self._Ntot = self._nf * self._N  # blocked DOF vector [block_0 (N), …, block_{nf-1} (N)]
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
                ax = _periodic_axis(*(np.asarray(self._pts)[self._region_nodes(t)] for t in tie))
                if ax is None:
                    raise ValueError(
                        f"jno.fdm([...]): a periodic tie `u(A) - u(B)` must connect two OPPOSITE faces "
                        f"(left/right, bottom/top, or front/back); got {tie}."
                    )
                self._periodic_axes.append(ax)
            elif _normal_jacobian(c) is not None or self._is_boundary_derivative_condition(c):
                self._neumann.append(c)  # a boundary row: Neumann, Robin, oblique, nonlinear flux, …
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
        if len(self.unknowns) > 1:  # coupled (multi-field)
            if len(self._pde) != len(self.unknowns):
                raise ValueError(
                    f"jno.fdm([...]): a coupled system needs exactly one PDE equation per unknown — got "
                    f"{len(self.unknowns)} unknowns but {len(self._pde)} PDE equation(s). Author one equation per "
                    "field, in the order the unknowns are declared (equation k drives unknown k)."
                )
            for k, (eq, own) in enumerate(zip(self._pde, self.unknowns)):
                other = [j for j, w in enumerate(self.unknowns) if w is not own and _has_temporal_of(eq, w)]
                if other:
                    raise NotImplementedError(
                        f"jno.fdm([...]): equation {k} of the coupled system carries the time derivative of "
                        f"unknown {other[0]}, not of its own unknown {k}. Equation k drives unknown k and may only "
                        "carry that unknown's `u.t` (a diagonal mass); reorder the equations, or solve for the "
                        "combination that appears differentiated."
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
            per = [False] * len(grid["shape"])
            for ax in self._periodic_axes:
                per[ax] = True
            # This problem's own view of the grid. The domain's descriptor is shared by every problem built on
            # it, so the wrap is applied only while this problem is evaluated (`_fd_scope`); writing it into the
            # domain made every later problem there wrap too.
            self._grid = {**grid, "periodic": tuple(per)}
        # The sub-domain this problem owns, if its PDE coordinates carry a named region
        # (`domain.region(name, poly)`). Used by `jno.core([...])` to couple subdomains automatically.
        self.region, self.region_geometry = self._pde_region()

    def _fd_scope(self):
        """Context under which this problem's terms are evaluated: the finite-difference kernels wrap its
        periodic axes (:func:`jno.trace_evaluator.fd_periodic`), and no others."""
        import contextlib

        from .trace_evaluator import fd_periodic

        if self._grid is None:
            return contextlib.nullcontext()
        return fd_periodic(self.domain.mesh_connectivity["grid"], self._grid["periodic"])

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
        if tag == "boundary" or tag is None:
            return np.asarray(self.domain.mesh_connectivity["boundary_indices"], dtype=int)
        pool = getattr(self.domain, "_mesh_pool", {}).get(tag)
        if pool is not None:  # a region sampled at mesh nodes (`domain.point_region(name, xy)`, …)
            from scipy.spatial import cKDTree

            dim = self._pts.shape[1]
            # A time-dependent domain stores the pool once per time step, (n_time, n, D): the same points.
            pts = np.unique(np.asarray(pool).reshape(-1, np.asarray(pool).shape[-1])[:, :dim], axis=0)
            dist, nodes = cKDTree(np.asarray(self._pts)).query(pts)
            extent = float(np.ptp(np.asarray(self._pts), axis=0).max())
            if len(pts) and np.all(dist <= _rtol() * extent):  # relative to the domain's size, not 1e-9
                return np.unique(nodes).astype(int)
        # An unknown tag used to fall through to the WHOLE boundary: `p(xg, yg) - 0` on a
        # `domain.point_region` pinned p on every wall node, and a lid-driven cavity came out 0.016 off
        # Ghia et al. instead of 0.0019 — plausible, and wrong. Refused instead.
        raise ValueError(
            f"jno.fdm([...]): no mesh nodes found for region {tag!r}. Use a boundary tag of the domain, "
            "'boundary', 'interior', a `domain.region(name, shape)` sub-region, or a "
            "`domain.point_region(name, xy)` node."
        )

    def _pde_exprs(self):
        """The PDE residual expressions as solved, one per field: summed for a single field, and with the
        per-axis default Laplacian fused into the cotangent one on an unstructured mesh
        (:func:`_fuse_fd_laplacian`)."""
        if len(self.unknowns) == 1:
            expr = self._pde[0]
            for c in self._pde[1:]:
                expr = expr + c
            exprs = [_unwrap(expr)]
        else:
            exprs = [_unwrap(self._pde[k]) for k in range(len(self.unknowns))]
        # Not on a sub-region: a domain-decomposition subdomain exports its interface flux from the
        # area-weighted gradient, which is consistent with the per-axis stencil but not the cotangent one;
        # fused, the FEM/FDM Dirichlet–Neumann iteration diverged (1e103). A cotangent-consistent interface
        # flux (the P1 reaction, as for a FEM subdomain) would lift this.
        if self.domain.mesh_connectivity.get("grid") is None and getattr(self, "region", None) is None:
            exprs = [_fuse_fd_laplacian(e, int(self.domain.dimension)) for e in exprs]
        return exprs

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
            value = getattr(module, "value", None)
            if value is not None and np.ndim(value) == 1 and np.shape(value)[0] == self._N:
                # one value per node: a per-node scalar field, shaped (N, 1) like every other one in the
                # strong form. As (N,) it broadcast against an (N, 1) derivative to (N, N).
                import equinox as eqx

                module = eqx.tree_at(lambda m: m.value, module, jnp.reshape(value, (self._N, 1)))
            if module is not None:
                found[lid] = module
        return found

    def _params_scope(self, extra_params=None):
        """The module every trainable parameter resolves to in an evaluation: its current value, then the
        values a traced solve injects (:attr:`_override`, set while a crux-driven march is traced), then
        ``extra_params``. Every evaluator this solver builds reads it, so a parameter works wherever it is
        written — in the PDE, a boundary value, a flux condition, a time coefficient, an initial value."""
        scope = self._referenced_models()  # data fields and networks, at their current values
        scope.update({lid: n.model.module for lid, n in self._trainable_params().items()})
        scope.update(getattr(self, "_override", None) or {})
        scope.update(extra_params or {})
        return scope

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
        visits, while a zero moves off zero."""
        import equinox as eqx

        rng = np.random.default_rng(11)
        probe = {}
        for lid, m in self._current_params().items():
            p = np.asarray(m.value)
            a, b = rng.uniform(0.1, 0.2, p.shape), rng.uniform(0.1, 0.2, p.shape)
            probe[lid] = eqx.tree_at(lambda mm: mm.value, m, jnp.asarray((1.0 + a) * p + b, dtype=m.value.dtype))
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

    def _pde_residual_fn(self, *, spatial=False, extra_params=None):
        """Differentiable residual over the nodal DOF vector, collocated at the mesh nodes. With
        ``spatial=True`` the ``u.t`` terms are dropped (:func:`_zero_temporal`) to give the
        method-of-lines spatial residual ``R_spatial`` for the semidiscrete march. ``extra_params``
        (``{layer_id: module}``) injects the current value of any **trainable** ``jno.np.parameter`` in
        the residual — how a ``crux``-driven inverse reaches the solve (see :meth:`_parametric_node`)."""

        from .trace_evaluator import TraceEvaluator

        exprs = self._pde_exprs()  # one equation per field (summed for a single field)
        if spatial:
            exprs = [_zero_temporal(e) for e in exprs]
        spatial_tags = set().union(*(_spatial_tags(c) for c in self._pde))  # collocated at the mesh nodes
        context = self._eval_context(spatial_tags)
        N = self._N
        scope = self._params_scope(extra_params)

        def residual_fn(dofs, t=None):
            """``t``: the time a source ``f(x, t)`` is evaluated at (the march passes each step's own
            time); ``None`` keeps the start time."""
            dofs = jnp.asarray(dofs)
            ev = TraceEvaluator(params={**scope, **self._live_params(), **self._inject(dofs)})
            ctx = context if t is None else {**context, "__time__": jnp.full((N, 1), t, dtype=dofs.dtype)}
            with self._fd_scope():
                blocks = [
                    self._as_blocks(
                        ev.evaluate(e, context=ctx, var_bindings={}), self._ncomp[k] if len(exprs) > 1 else self._nf
                    )
                    for k, e in enumerate(exprs)
                ]
            out = blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks)
            return self._require_real(out, "the PDE residual (a complex coefficient or source)")

        return residual_fn

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
                f"jno.fdm([...]): an equation has {c} component(s) but drives an unknown with {ncomp}. A vector "
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
                f"jno.fdm([...]): a condition on a {c}-component unknown gives {v.shape[1]} value(s) per node. Give "
                "every component, e.g. `U(xb, yb) - jnn.stack([gx, gy], axis=-1)`."
            )
        return [v[:, j] for j in range(c)]

    def _eval_context(self, spatial_tags):
        """Evaluation context for a strong-form term: every spatial tag collocates at the mesh nodes, and
        ``domain.cell_size`` resolves to the per-node spacing :meth:`_node_spacing`."""
        context = {t: self._pts for t in spatial_tags}
        if self._uses_cell_size():  # built on demand: on a mesh it walks every cell (1.8 GB at 0.9M nodes)
            context["cell_size"] = self._node_spacing()[:, None]
        # The temporal Variable reads `__time__`. A source `f(x, t)` used to raise KeyError here; it now
        # sees the start time unless the march passes the step's own (see `residual_fn(dofs, t)`).
        context["__time__"] = jnp.full((self._N, 1), self._start_time())
        return context

    def _start_time(self):
        window = getattr(self.domain, "time", None)
        return float(window[0]) if window is not None and self._transient else 0.0

    def _dirichlet_values_at(self, t):
        """``(mask, values)`` of the Dirichlet rows at time ``t``. A value written with the time variable,
        ``u(xb, yb) - g(xb, yb, tb)``, is evaluated at ``t``; it used to be evaluated once with the time
        column read from the coordinates, and the march then held the boundary fixed (a heat solve with
        g = e^{-π² t} cos(πx) stayed at 1.0 on the boundary; error 3.5 at T)."""
        mask = np.zeros(self._Ntot, dtype=bool)  # over the blocked DOF vector (field k at k·N …)
        vals = jnp.zeros(self._Ntot)
        for c in self._dirichlet:
            nodes = np.asarray(self._region_nodes(_region_tag(c)), dtype=int)
            k = self._field_index(c)
            for b in self._blocks_of(k):
                mask[b * self._N + nodes] = True
            g_node = self._value_side(c)
            if self._is_nodal_data(g_node):
                g = self._eval_g(g_node, nodes)  # a known nodal field: gathered at the region's nodes
            else:
                g = self._eval_value(g_node, nodes, getattr(self, "_override", None), t)
            for b, gb in zip(self._blocks_of(k), self._split_components(k, g, len(nodes))):
                vals = vals.at[jnp.asarray(b * self._N + nodes)].set(gb)
        return mask, vals

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

            grid = self.domain.mesh_connectivity.get("grid") if self.domain.mesh_connectivity else None
            if grid is not None and self._nodes_are_the_grid(grid):
                # The cell formula gives exactly the geometric mean of the axis spacings on this lattice; read
                # it off the grid instead of walking its 6 tetrahedra per voxel (1.8 GB transient at 0.9M
                # nodes, measured). A structured grid's nodes are not design variables, so no gradient is lost.
                h = float(np.prod(np.asarray(grid["spacing"], float)) ** (1.0 / len(grid["spacing"])))
                self._h_nodes = jnp.full(self._N, h)
                return self._h_nodes
            # A constant of the mesh: computed concretely even when first asked for inside a trace (the
            # parametric solve), or the cached value would be a leaked tracer.
            with jax.ensure_compile_time_eval():
                self._h_nodes = self._node_spacing_now()
        return self._h_nodes

    def _nodes_are_the_grid(self, grid):
        """The structured shortcut applies only while the nodes still sit on the lattice (a moved or
        relocated structured mesh falls back to the cell formula)."""
        shape = tuple(int(s) for s in grid["shape"])
        if int(np.prod(shape)) != self._N or "origin" not in grid:
            return False
        pts = np.asarray(self._pts)[:, : len(shape)]
        k = (pts - np.asarray(grid["origin"], float)) / np.asarray(grid["spacing"], float)
        return bool(np.max(np.abs(k - np.round(k))) <= _rtol(pts.dtype) * max(shape))  # in lattice units

    def _uses_cell_size(self):
        """Does any constraint read ``domain.cell_size``?"""
        if getattr(self, "_cell_size_used", None) is None:
            self._cell_size_used = any(
                getattr(v, "tag", None) == "cell_size" for c in self._constraints for v in _iter_variables(c)
            )
        return self._cell_size_used

    def _node_spacing_now(self):
        """Per-node mean of ``(d!·|K|)^(1/d)`` over incident cells (see :meth:`_node_spacing`)."""
        pts, cells = _mesh(self.domain)
        dim = pts.shape[1]
        e = pts[cells[:, 1:]] - pts[cells[:, :1]]  # (C, dim, dim) edge vectors from vertex 0
        size = (jnp.abs(jnp.linalg.det(e))) ** (1.0 / dim)  # (d!·|K|)^(1/d), since |det| = d!·|K|
        total = jnp.zeros(self._N).at[cells.reshape(-1)].add(jnp.repeat(size, cells.shape[1]))
        count = jnp.zeros(self._N).at[cells.reshape(-1)].add(1.0)
        return total / jnp.maximum(count, 1.0)

    def _sparsity(self, key, fun, u, make=None):
        """``(pattern, colour, n_colours)`` for assembling ``fun``'s Jacobian, found once per problem.

        The stencil width depends on the operators (one ring for ``cotangent`` and the structured grid,
        two for a gradient of a gradient or the quadratic flux fit), so candidate patterns of growing
        radius are tried and each is **verified**: the assembled matrix must reproduce the matrix-free
        JVP on a random vector to 1e-10, for any value of the parameters (a coefficient that starts at 0 can
        switch a wider stencil on; ``make()`` rebuilds ``fun`` when it is jitted, see
        :meth:`_for_every_parameter`). Raises if none does, rather than solving with a wrong matrix."""
        cache = self.__dict__.setdefault("_sparsity_cache", {})
        if key in cache:
            return cache[key]
        import jax

        cells = _mesh(self.domain)[1]
        extra = [(np.asarray(sec), np.asarray(main)) for sec, main in self._periodic_rows()]
        n = int(np.asarray(u).size)
        n_fields = n // self._N
        v = jnp.asarray(np.random.default_rng(0).standard_normal(n), dtype=jnp.asarray(u).dtype)

        def reproduces(f, pattern, color, n_colors):
            ref = jax.jvp(f, (jnp.asarray(u),), (v,))[1]
            A = _assemble_sparse(f, u, pattern, color, n_colors)
            return float(jnp.linalg.norm(A @ v - ref)) <= _rtol(ref.dtype) * (float(jnp.linalg.norm(ref)) or 1.0)

        for radius in (1, 2, 3, 4, 5, 6):  # a jno.fd(order=/fit=) mesh fit can read several rings
            pattern = _stencil_pattern(cells, self._N, radius, n_fields=n_fields, extra_pairs=extra)
            color, n_colors = _color_columns(pattern)
            ok = self._for_every_parameter(
                lambda probe: reproduces((make() if make is not None and probe else fun), pattern, color, n_colors)
            )
            if all(ok):
                cache[key] = (pattern, color, n_colors)
                return cache[key]
        raise ValueError(
            "jno.fdm: could not assemble this strong-form operator as a sparse matrix (no stencil radius up "
            "to 6 reproduces its matrix-free action), so the linear=/precond= slots that need a matrix "
            "cannot be used. Leave linear=/precond= unset to keep the matrix-free default."
        )

    def _sparse_operator(self, key, fun, u):
        """``fun``'s Jacobian at ``u`` as a sparse BCOO (see :meth:`_sparsity`)."""
        pattern, color, n_colors = self._sparsity(key, fun, u)
        return _assemble_sparse(fun, u, pattern, color, n_colors)

    def _dirichlet_lift(self, key, K):
        """Symmetric Dirichlet elimination on an assembled operator: ``(K_lifted, lift_rhs)``.

        A Dirichlet row is a pure constraint ``s·(u_j − g_j)``, so ``u_j = rhs_j / K_jj`` is known, and its
        column can move to the right-hand side: ``rhs_i −= K_ij·u_j`` and ``K_ij = 0`` for every other row.
        The solution is unchanged, and the operator becomes symmetric wherever the stencil is (a structured
        grid). Without it the interior rows referenced the boundary while the boundary rows did not
        reference them back, and CG converged to an answer 2e-6 off, inside the residual gate. The pattern
        is the host one cached by :meth:`_sparsity`, so this also works on a traced operator."""
        import jax

        pattern = self._sparsity_cache[key][0].tocoo()
        rows, cols = pattern.row, pattern.col
        n = K.shape[0]
        is_d = self._dirichlet_mask() if n == self._Ntot else np.zeros(n, dtype=bool)
        if not is_d.any():
            return K, lambda rhs: rhs
        moved = jnp.asarray(is_d[cols] & ~is_d[rows])
        diag_pos = np.nonzero(is_d[rows] & (rows == cols))[0]
        d_nodes = jnp.asarray(rows[diag_pos])
        d_diag = K.data[jnp.asarray(diag_pos)]
        jrows, jcols = jnp.asarray(rows), jnp.asarray(cols)
        moved_data = jnp.where(moved, K.data, 0.0)
        K_lifted = type(K)((jnp.where(moved, 0.0, K.data), K.indices), shape=K.shape)

        def lift_rhs(rhs):
            known = jnp.zeros(n, dtype=rhs.dtype).at[d_nodes].set(rhs[d_nodes] / d_diag)
            return rhs - jax.ops.segment_sum(moved_data * known[jcols], jrows, num_segments=n)

        return K_lifted, lift_rhs

    @staticmethod
    def _eliminate(A, is_d):
        """Split an assembled BCOO ``A`` at the Dirichlet DOFs ``is_d``: ``(A_s, solve)``.

        ``A_s`` keeps the interior block and the Dirichlet diagonal ``d`` and drops the couplings between
        them, so it is symmetric wherever the interior stencil is. ``solve(inner, b)`` returns the exact
        ``A⁻¹b`` from one ``inner(A_s, rhs)`` call, provided the Dirichlet rows **or** the Dirichlet
        columns of ``A`` are pure (``A_DI = 0`` or ``A_ID = 0``):

            x_I = A_II⁻¹ (b_I − A_ID b_D/d),    x_D = (b_D − A_DI x_I)/d

        A Newton tangent ``J`` has pure Dirichlet rows and its transpose pure columns, so the same split
        serves the forward step and the adjoint. Pure JAX on the BCOO values (the mask is host-side)."""
        import jax
        import jax.experimental.sparse as jsp

        n = A.shape[0]
        r, c = A.indices[:, 0], A.indices[:, 1]
        mask = jnp.asarray(is_d)
        dr, dc = mask[r], mask[c]
        d = jnp.where(mask, jax.ops.segment_sum(jnp.where(dr & (r == c), A.data, 0.0), r, num_segments=n), 1.0)
        A_s = jsp.BCOO((jnp.where(dr ^ dc, 0.0, A.data), A.indices), shape=A.shape)
        to_d, from_d = jnp.where(dr & ~dc, A.data, 0.0), jnp.where(~dr & dc, A.data, 0.0)

        def solve(inner, b):
            known = jnp.where(mask, b / d, 0.0)
            z = inner(A_s, b - jax.ops.segment_sum(from_d * known[c], r, num_segments=n))
            back = jax.ops.segment_sum(to_d * z[c], r, num_segments=n)
            return jnp.where(mask, (b - back) / d, z)

        return A_s, solve

    def _eliminating(self, linear):
        """``cg`` / ``minres`` wrapped to solve an assembled Newton tangent through :meth:`_eliminate`.

        The tangent's Dirichlet rows are identity rows whose columns the interior rows still reference,
        so it is not symmetric and CG on it returned NaN (the steady linear path lifts those columns out,
        the Newton path did not). Other solvers, and a matrix-free tangent, are returned untouched."""
        if getattr(linear, "name", "") not in ("cg", "minres"):
            return linear
        from .utils.solver.solver_api import LinearOperator, LinearSolver

        is_d = self._dirichlet_mask()

        def fn(op, b, *, M, x0):
            A = op.bcoo
            if A is None or A.shape[0] != is_d.size:
                return linear(op, b, M=M, x0=x0)
            _, solve = self._eliminate(A, is_d)
            return solve(lambda A_s, rhs: linear(LinearOperator(A_s), rhs, M=M, x0=x0), b)

        key = None if linear.key is None else ("dirichlet-eliminated", linear.key)
        return LinearSolver(fn, name=linear.name, traits=linear.traits, key=key)

    def _newton_linear(self, linear, key, make, at):
        """The ``linear=`` slot for a Newton on an assembled tangent: ``cg`` / ``minres`` go through the
        Dirichlet elimination, after the eliminated tangent at ``at`` of the residual ``make()`` builds passes
        the symmetry guard."""
        if getattr(linear, "name", "") not in ("cg", "minres"):
            return linear
        import jax

        mask = self._dirichlet_mask()
        with jax.ensure_compile_time_eval():
            self._require_symmetric(
                linear, lambda: self._eliminate(self._sparse_operator(key, make(), at), mask)[0], key=key + "-newton"
            )
        return self._eliminating(linear)

    def _dirichlet_mask(self):
        """Boolean mask of the Dirichlet DOFs (host-side indices, so it also works inside a trace)."""
        is_d = np.zeros(self._Ntot, dtype=bool)
        for c in self._dirichlet:
            nodes = np.asarray(self._region_nodes(_region_tag(c)))
            for b in self._blocks_of(self._field_index(c)):
                is_d[b * self._N + nodes] = True
        return is_d

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

        out = all(self._for_every_parameter(symmetric))
        if key is not None:
            cache[key] = out
        return out

    def _require_symmetric(self, linear, make_K, key=None):
        """``cg`` and ``minres`` assume a symmetric operator; on a non-symmetric one they return a wrong answer
        that can still pass the residual gate. FDM operators are symmetric on a structured grid (after the
        Dirichlet lift) but not on an unstructured mesh, whose cotangent rows are divided by nodal areas.
        ``make_K()`` assembles the operator; it must be symmetric for every value of the parameters."""
        name = getattr(linear, "name", "")
        if name not in ("cg", "minres") or self._is_symmetric(make_K, key):
            return
        a, b = self._symmetry_probe(make_K())
        raise ValueError(
            f"jno.solve.{name} needs a symmetric operator, and this strong-form operator is not "
            f"(wᵀKv = {a:.6e} vs vᵀKw = {b:.6e}). Unstructured FDM stencils are divided by nodal areas, "
            "and flux rows are one-sided. Use jno.solve.gmres() or jno.solve.bicgstab()."
        )

    def _mass_varies_in_time(self, make, t0, t1):
        """Does the ``u.t`` coefficient (``make()`` builds ``mass_of(t)``) change in time, for any value of the
        parameters (:meth:`_for_every_parameter`)? Cached like the other structural decisions."""
        cache = self.__dict__.setdefault("_varies_cache", {})
        if "mass" not in cache:

            def varies(_):
                mass_of = make()
                return not bool(jnp.allclose(mass_of(t0).data, mass_of(t1).data))

            cache["mass"] = any(self._for_every_parameter(varies))
        return cache["mass"]

    def _operator_varies_in_time(self, make, n, t0, t1, key="march"):
        """Cached per ``key``: decided on concrete values (a warm-up solve), for any value of the parameters,
        and reused inside a traced solve. ``make()`` builds ``residual(y, t, args)``."""
        cache = self.__dict__.setdefault("_varies_cache", {})
        if key not in cache:
            cache[key] = any(self._for_every_parameter(lambda _: self._operator_varies_now(make(), n, t0, t1)))
        return cache[key]

    @staticmethod
    def _operator_varies_now(residual, n, t0, t1):
        """Does the Jacobian of ``residual(y, t)`` change between ``t0`` and ``t1`` (a coefficient κ(t))?
        Then the operator cannot be assembled once. Time-dependent DATA alone does not count."""
        import jax

        v = jnp.asarray(np.random.default_rng(3).standard_normal(n))
        z = jnp.zeros(n)
        a = jax.jvp(lambda y: residual(y, t0, {}), (z,), (v,))[1]
        b = jax.jvp(lambda y: residual(y, t1, {}), (z,), (v,))[1]
        return bool(jnp.linalg.norm(a - b) > _rtol(a.dtype) * (float(jnp.linalg.norm(a)) or 1.0))

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

            cache[key] = all(self._for_every_parameter(affine))
        return cache[key]

    def _mass_coefficient(self):
        """Per-node coefficient ``c`` on ``u.t`` (the diagonal mass ``M = diag(c)``), via the two-probe
        ``c = F(u.t=1) − F(u.t=0)`` (:func:`_set_temporal`) — the spatial residual cancels between the
        probes, leaving ``c``. A plain ``ui.t - 𝒩(u)`` gives ``c = 1``; a ``ρcₚ(x)·ui.t`` term gives the
        node values of ``ρcₚ(x)``. ``c`` must be constant in ``u`` (a nonlinear mass ``c(u)·u.t`` raises).
        A coupled system stacks one block per field: equation k's coefficient of ``u_k.t`` (zero for an
        equation without a time derivative, whose field is then algebraic)."""
        return self._time_coefficients(1.0, 0.0, "`u.t` mass coefficient", "nonlinear mass `c(u)·u.t`")

    def _time_coefficients(self, t_val, tt_val, what, example):
        """:meth:`_time_coefficient` over every equation, concatenated into the blocked DOF layout: equation
        k's coefficient of its own field's time derivative (a vector equation spans its components)."""
        coefs = [self._time_coefficient(t_val, tt_val, what, example, k=k) for k in range(len(self._pde_exprs()))]
        if len(coefs) == 1:
            return coefs[0]
        return lambda t=None: jnp.concatenate([c(t) for c in coefs])

    @staticmethod
    def _same(a, b):
        """Are two evaluations of the same quantity equal to floating-point accuracy (:func:`_rtol`), relative
        to their magnitude?"""
        a, b = jnp.asarray(a), jnp.asarray(b)
        # Compared inside JAX: the values can be tracers (a coefficient differentiated by jax.grad), whose
        # comparison resolves but whose magnitude does not (`float()` of a tracer raises).
        scale = jnp.maximum(jnp.maximum(jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b))), jnp.finfo(a.dtype).tiny)
        return bool(jnp.all(jnp.abs(a - b) <= _rtol(a.dtype) * scale))

    def _time_coefficient(self, t_val, tt_val, what, example, k=0):
        """Per-node coefficient of the time derivative selected by the probe ``(u.t, u.tt) = (t_val,
        tt_val)``: ``F(t_val, tt_val) − F(0, 0)``, returned as a function of time ``c(t)``. Raises if it
        depends on ``u`` (probed at two constant states), because the diagonal-mass march cannot carry a
        state-dependent coefficient."""
        import equinox as eqx

        from .trace_evaluator import TraceEvaluator

        expr = self._pde_exprs()[k]
        spatial_tags = set().union(*(_spatial_tags(c) for c in self._pde))
        context = self._eval_context(spatial_tags)
        scope = self._params_scope()

        def probe(u_val, t=None):
            ctx = context if t is None else {**context, "__time__": jnp.full((self._N, 1), t)}

            def at(tv, ttv):
                states = {  # every field at the same constant state
                    w.layer_id: eqx.tree_at(
                        lambda m: m.value, w.module, jnp.full(w.module.value.shape, u_val, w.module.value.dtype)
                    )
                    for w in self.unknowns
                }
                ev = TraceEvaluator(params={**scope, **self._live_params(), **states})
                with self._fd_scope():
                    return jnp.asarray(ev.evaluate(_set_temporal(expr, tv, ttv), context=ctx, var_bindings={}))

            ncomp = self._ncomp[k] if len(self.unknowns) > 1 else self._nf
            diff = jnp.asarray(at(t_val, tt_val) - at(0.0, 0.0))
            if diff.size == 1:  # a constant coefficient, the same on every node and component
                return jnp.broadcast_to(diff.reshape(()), (ncomp * self._N,))
            return self._as_blocks(diff, ncomp)

        checked = self.__dict__.setdefault("_coefficient_checked", set())  # decided once, on concrete values
        if (t_val, tt_val, k) not in checked and not self._same(probe(0.0), probe(1.0)):
            raise ValueError(
                f"jno.fdm([...]): the {what} depends on u (a {example}) — only a coefficient of the "
                "coordinates and time is supported."
            )
        checked.add((t_val, tt_val, k))
        # A function of TIME: `c(x, t)·u.t` and `m(t)·u.tt` are evaluated at each step's own time. They used
        # to be probed once at the start and then held (measured: (1 + t)·u.t stayed at 1·u.t).
        return lambda t=None: probe(0.0, t)

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

    def _require_real(self, value, what):
        """``value``, unless it is complex while every unknown is real: a real solve would keep only its real
        part (JAX casts complex to real with a warning), which is a wrong answer, not a supported one."""
        if jnp.iscomplexobj(value) and not any(jnp.iscomplexobj(w.module.value) for w in self.unknowns):
            raise NotImplementedError(
                f"jno.fdm: {what} is complex, but the unknown is real, and a real solve would keep only the real "
                "part. jno.fdm has no complex fields yet: write the real and imaginary parts as two real unknowns "
                "(a coupled system)."
            )
        return value

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
            out = jnp.asarray(inner.model.module.value).reshape(-1)[jnp.asarray(idx)]  # nodal data → gather
        else:
            out = jnp.asarray(_eval_value_node_at(g_node, np.asarray(self._pts)[idx])).reshape(-1)
        return self._require_real(out, "a condition's value")

    def _condition_value(self, constraint, idx):
        """Value ``g`` of an affine condition ``u(region) - g`` (Dirichlet or IC), evaluated at the
        region's nodes ``idx`` — reused for both boundary conditions and the initial state."""
        return self._eval_g(self._value_side(constraint), idx)

    def _field_index(self, constraint):
        """Which unknown's DOF block a value-only constraint (Dirichlet) pins — 0 for a single field."""
        for k, u in enumerate(self.unknowns):
            if _contains_unknown(constraint, u):
                return k
        return 0

    def _dirichlet_rows(self, extra_params=None):
        """Per-block Dirichlet rows ``(block, node_indices, values)``: ``block`` selects the DOF block (a
        field, or one component of a vector field; 0 for a single scalar field), ``node_indices`` the region's nodes, ``values`` the pinned g.
        With ``extra_params`` a trainable parameter in ``g`` takes its injected (possibly traced) value."""
        rows = []
        extra_params = {**(getattr(self, "_override", None) or {}), **(extra_params or {})}
        for c in self._dirichlet:
            idx = self._region_nodes(_region_tag(c))
            if extra_params and self._uses_params(c, extra_params):
                vals = self._eval_value(self._value_side(c), idx, extra_params)
            else:
                vals = self._condition_value(c, idx)
            k = self._field_index(c)
            for b, vb in zip(self._blocks_of(k), self._split_components(k, vals, len(np.asarray(idx)))):
                rows.append((b, jnp.asarray(idx), vb))  # one row set per DOF block (component)
        return rows

    def _is_nodal_data(self, g_node):
        """A value that is a known nodal field (a ``jno.np.parameter`` of one value per node, no optimizer)."""
        from .trace import ModelCall

        n = _unwrap(g_node) if not isinstance(g_node, (int, float)) else None
        return (
            isinstance(n, ModelCall)
            and getattr(n.model, "_is_parameter", False)
            and n.model.layer_id not in self._trainable_params()
        )

    def _value_side(self, constraint):
        """``g`` of a value condition, so that it reads ``u = g``: ``u(region) - g`` and ``g - u(region)`` give
        ``g``; ``u(region) + v`` and ``v + u(region)`` give ``−v``; a bare ``u(region)`` gives 0.

        Anything else raises. Only the ``-`` form used to be read: ``u(xr, yr) + 1.0`` was imposed as u = 0,
        silently, and a steady Burgers solve with u = −1 on its right wall converged to u = 0 there."""
        from .trace import BinaryOp, Literal, ModelCall

        inner = _unwrap(constraint)

        def bare(n):
            n = _unwrap(n)
            return isinstance(n, ModelCall) and any(n.model is w for w in self.unknowns)

        if bare(inner):
            return 0.0
        if isinstance(inner, BinaryOp) and inner.op in ("-", "+"):
            if bare(inner.left) and not any(_contains_unknown(inner.right, w) for w in self.unknowns):
                return inner.right if inner.op == "-" else BinaryOp("*", Literal(-1.0), inner.right)
            if bare(inner.right) and not any(_contains_unknown(inner.left, w) for w in self.unknowns):
                return inner.left if inner.op == "-" else BinaryOp("*", Literal(-1.0), inner.left)
        raise ValueError(
            f"jno.fdm([...]): a value condition must read `u(region) - g` (or `u(region) + v`, `g - u(region)`); "
            f"got {constraint!r}. Divide out any factor on the unknown: `2*u(xb, yb) - 2*g` is `u(xb, yb) - g`."
        )

    @staticmethod
    def _uses_params(node, params):
        from .trace import ModelCall

        n = _unwrap(node)
        if isinstance(n, ModelCall) and n.model.layer_id in params:
            return True
        return any(_TraceFDM._uses_params(c, params) for c in _iter(n))

    def _eval_value(self, g_node, idx, extra_params=None, t=None):
        """``g`` at the nodes ``idx``, evaluated by the trace evaluator with ``extra_params`` injected, so a
        trainable parameter in a Dirichlet value is differentiable. (It used to be read from the stored
        value: a crux inverse on a Dirichlet value never moved, 0.5 stayed 0.5 against a true 1.5.)"""
        from .trace import ModelCall, Variable
        from .trace_evaluator import TraceEvaluator

        if isinstance(g_node, (int, float)):
            return jnp.full((len(idx),), float(g_node))
        pts = self._pts[jnp.asarray(np.asarray(idx, dtype=int))]
        params, ctx = {}, {}

        def walk(n):
            n = _unwrap(n)
            if isinstance(n, ModelCall):
                params.setdefault(n.model.layer_id, n.model.module)
            if isinstance(n, Variable):
                temporal = getattr(n, "axis", None) == "temporal"
                ctx[n.tag] = jnp.full((pts.shape[0], 1), self._start_time() if t is None else t) if temporal else pts
            for c in _iter(n):
                walk(c)

        walk(g_node)
        params.update(extra_params or {})
        params.update(self._live_params())
        with self._fd_scope():
            out = jnp.asarray(TraceEvaluator(params=params).evaluate(_unwrap(g_node), context=ctx, var_bindings={}))
        self._require_real(out, "a condition's value")
        return jnp.broadcast_to(out.reshape(-1), (len(idx),)) if out.size == 1 else out.reshape(-1)

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

    def _normal_field(self, region):
        """``(N, dim)`` outward unit normals of ``region`` at its nodes (zero elsewhere): the values the
        components ``nx, ny`` of ``d.variable(region, normals=True, split=True)`` take in a flux condition.
        The same normals the flux rows use (:meth:`_node_normals`), so ``pb.d(n) - (nx*gx + ny*gy)`` is
        exactly ``∇p·n = g·n``. They used to be missing from the evaluation context (a ``KeyError``)."""
        idx, nrm = self._node_normals(region)
        return jnp.zeros((self._N, self._pts.shape[1])).at[jnp.asarray(np.asarray(idx, dtype=int))].set(jnp.asarray(nrm))

    def _node_normals(self, region):
        """Unit outward normals for the flux nodes of ``region``, aligned to those nodes.

        **2-D** — computed from the mesh **boundary segments** (``mesh_connectivity["boundary_edges"]``):
        each segment's exact perpendicular is averaged over the (two) segments meeting at a node, so an
        axis-aligned edge yields an exact ``(±1, 0)`` / ``(0, ±1)`` — much cleaner than the domain's
        smoothed per-point normal, which bleeds a tangential component near corners and would spoil the
        flux. Each segment is oriented outward by its owning triangle, and only the region's own segments
        are averaged, so a corner shared with another flux region carries both conditions (summed).

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
        # Orient every segment OUTWARD, away from the opposite vertex of the triangle that owns it. The
        # boundary segments are not stored with a consistent orientation, so two collinear segments
        # could get opposite perpendiculars, average to zero, and have their shared node dropped as a
        # "corner": a Neumann node at (1, 0.9) then kept its PDE row, and a linear solution that should be
        # exact came back 5e-3 off.
        apex_of = {}
        for tri in np.asarray(_mesh(self.domain)[1], dtype=int):
            for a, b, c in ((tri[0], tri[1], tri[2]), (tri[1], tri[2], tri[0]), (tri[2], tri[0], tri[1])):
                apex_of[(min(a, b), max(a, b))] = c
        apex = np.array([apex_of[(min(a, b), max(a, b))] for a, b in edges])
        inward = np.sum(seg_n * (pts[apex] - pts[edges[:, 0]]), axis=1) > 0
        seg_n[inward] *= -1
        # Average only this region's own segments. A node where two regions meet (a corner between a
        # Neumann and a Robin edge) then gets each region's own normal, and both flux conditions are
        # imposed there, summed (see :meth:`_apply_flux_rows`). The normal is NOT renormalised: on a
        # straight edge it is the unit normal, and at a corner inside one region it is the average of the
        # two edge normals, which makes the row the average of the two edge conditions. Corners used to be
        # dropped and keep their PDE row, which is not a boundary condition at all (all-Neumann problems
        # stalled at 0.12; a structured transient march with such a corner blew up to 1e33).
        idx = np.asarray(self._region_nodes(region), dtype=int)
        in_region = np.zeros(self._N, dtype=bool)
        in_region[idx] = True
        own = in_region[edges[:, 0]] & in_region[edges[:, 1]]
        node_n = np.zeros((self._N, dim))
        cnt = np.zeros(self._N)
        for (i, j), n_e in zip(edges[own], seg_n[own]):
            node_n[i] += n_e
            node_n[j] += n_e
            cnt[i] += 1
            cnt[j] += 1
        keep = cnt[idx] > 0
        idx = idx[keep]
        n = node_n[idx] / cnt[idx][:, None]
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

    def _flux_region(self, c):
        """The one boundary region a boundary row lives on (its coordinate tag, or its normal's)."""
        normals = {str(v.tag)[len("n_") :] for v in _iter_variables(c) if str(getattr(v, "tag", "")).startswith("n_")}
        boundary = set(getattr(self.domain, "_boundary_registry", {}) or {}) - {"boundary"}
        tags = normals or {t for t in _spatial_tags(c) if t in boundary}  # interior coords in a value are fine
        if len(tags) != 1:
            raise ValueError(
                f"jno.fdm([...]): a boundary condition must live on one boundary region; this one uses {sorted(tags)}. "
                "Bind its fields and normal to the same region (`ub = u.bind(x=xr, y=yr)`, "
                "`d.variable('right', normals=True)`)."
            )
        return next(iter(tags))

    def _flux_owner(self, c, idx):
        """Which field's equation the boundary row replaces at the region's nodes: of the fields the condition
        differentiates, the one not already fixed there by a Dirichlet condition. A cavity wall carries
        ``u = 0``, ``v = 0`` and ``∂p/∂n = n·(νΔu − u·∇u)``, which differentiates all three, and is p's row."""
        fields = [k for k, w in enumerate(self.unknowns) if _has_unknown_derivative(c, w)]
        pinned = self._dirichlet_mask()
        free = [
            k for k in fields if not all(pinned[b * self._N + np.asarray(idx, dtype=int)].all() for b in self._blocks_of(k))
        ]
        normal = {k for j in _normal_jacobians(c) for k, w in enumerate(self.unknowns) if _contains_unknown(j.target, w)}
        if len(free) == 1:
            return free[0]
        if len(normal) == 1 and (not free or next(iter(normal)) in free):
            return next(iter(normal))  # the field whose ∂/∂n it imposes (a Dirichlet on the same nodes wins)
        if len(fields) == 1:
            return fields[0]
        raise ValueError(
            "jno.fdm([...]): cannot tell which field this boundary condition is for: it differentiates unknowns "
            f"{fields}, and {free if free else 'none'} of them are free on its region. A boundary row replaces one "
            "field's equation there; give the others Dirichlet conditions on that region, or impose this "
            "condition through its normal derivative `ub.d(n)`."
        )

    def _flux_row_fn(self, skeleton, extra_params=None, ncomp=1):
        """``(dofs, t) ↦ residual`` of a boundary condition at every node, from its eagerly built
        :meth:`_flux_skeleton`; ``extra_params`` injects trainable-parameter values."""

        from .trace_evaluator import TraceEvaluator

        expr, context, jacs, grads, idx = skeleton
        scope = self._params_scope(extra_params)
        N, dim, jidx = self._N, int(self._pts.shape[1]), jnp.asarray(idx)

        def row(dofs, t=None):
            with self._fd_scope():
                return _row(dofs, t)

        def _row(dofs, t=None):
            ev = TraceEvaluator(params={**scope, **self._live_params(), **self._inject(jnp.asarray(dofs))})
            ctx = dict(context) if t is None else {**context, "__time__": jnp.full((N, 1), t)}
            for i, (j, g) in enumerate(zip(jacs, grads)):
                target = jnp.asarray(ev.evaluate(j.target, context=ctx, var_bindings={}))
                # A scalar target is (N,) or (N, 1); a vector one (the velocity in a wall pressure) is (N, c)
                # and is differentiated component by component.
                target = (
                    jnp.broadcast_to(target.reshape(-1), (N,))[:, None] if target.size in (1, N) else target.reshape(N, -1)
                )
                G = jnp.stack([g(target[:, k]) for k in range(target.shape[1])], axis=1)  # (len(idx), c, dim)
                for a in range(dim):
                    ctx[f"fdm_bgrad_{i}_{a}"] = jnp.zeros((N, target.shape[1]), target.dtype).at[jidx].set(G[:, :, a])
            out = self._require_real(jnp.asarray(ev.evaluate(expr, context=ctx, var_bindings={})), "a flux condition")
            if ncomp > 1:  # a vector condition (a traction) carries one row per component
                return jnp.broadcast_to(out.reshape(-1, ncomp), (N, ncomp))
            out = out.reshape(-1)
            return jnp.broadcast_to(out, (N,)) if out.shape[0] == 1 else out

        return row

    def _flux_skeleton(self, c, idx):
        """The structural part of the boundary row for ``c``, built once, eagerly: the rewritten condition,
        its evaluation context (with the region's normals) and a boundary gradient per derivative node.
        :meth:`_flux_row_fn` binds the parameters and returns ``(dofs, t) ↦ residual`` at every node.

        The condition is evaluated **as written**. Every first derivative in it of an expression of the
        unknowns — ``ub.x``, ``ub.d(n)``, ``ub.d((nx, ny))``, ``(κ*ub).x`` — is replaced by the
        boundary-accurate gradient of that expression at the region's nodes (:meth:`_flux_gradient_fn`: the
        one-sided difference on a grid, the quadratic fit on a mesh), and a normal derivative by that
        gradient dotted with the region's normals. So any condition works, affine in ∂u/∂n or not —
        Neumann, Robin, oblique, a nonlinear flux; Newton solves the rows. Second derivatives (``Δu`` in a
        wall pressure) keep the grid stencil. ``extra_params`` injects trainable-parameter values."""
        from .trace import Jacobian, TemporalDerivative

        region = self._flux_region(c)
        dim = int(self._pts.shape[1])
        jacs = []

        def is_normal(var):
            return str(getattr(var, "tag", "")).startswith("n_")

        def spatial_jacobian(n):
            if not isinstance(n, Jacobian) or isinstance(n, TemporalDerivative) or len(n.variables) != 1:
                return False
            var = n.variables[0]
            if not (is_normal(var) or getattr(var, "axis", "spatial") == "spatial"):
                return False
            return any(_contains_unknown(n.target, w) for w in self.unknowns)

        def rewrite(n):
            n = _unwrap(n)
            if spatial_jacobian(n):
                i = len(jacs)
                jacs.append(n)
                # ∂target/∂x_a for every component of the target: `fdm_bgrad_{i}_{a}` is (N, c)
                comp = lambda a: _context_leaf(f"fdm_bgrad_{i}_{a}", None, self.domain)  # noqa: E731
                var = n.variables[0]
                if is_normal(var):  # ∂/∂n = ∇·n with the region's normals
                    out = comp(0) * _context_leaf(var.tag, 0, self.domain)
                    for a in range(1, dim):
                        out = out + comp(a) * _context_leaf(var.tag, a, self.domain)
                    return out
                return comp(int(var.dim[0]))
            return _map_children(n, rewrite)

        expr = rewrite(_unwrap(c))
        spatial_tags = _spatial_tags(c) - {t for t in _spatial_tags(c) if str(t).startswith("n_")}
        context = self._eval_context(spatial_tags)
        context.update({f"n_{region}": self._normal_field(region)})
        context.update({v.tag: self._normal_field(str(v.tag)[len("n_") :]) for v in _iter_variables(expr) if is_normal(v)})
        grads = [self._jacobian_gradient_fn(j, idx) for j in jacs]
        return expr, context, jacs, grads, np.asarray(idx, dtype=int)

    def _jacobian_gradient_fn(self, jac, idx):
        """The boundary gradient for one derivative node, honouring its FD sub-scheme."""
        scheme = getattr(jac, "scheme", None) or "finite_difference"
        if "finite_difference" not in str(scheme):
            scheme = "finite_difference"  # `.d(v)` defaults to AD on a view; on nodal DOFs it is the FD stencil
        _, grad_method, _ = _D.parse_fd_scheme(scheme)
        return self._flux_gradient_fn(np.asarray(idx, dtype=int), scheme, grad_method)

    def _flux_rows(self, extra_params=None):
        """``[(rows, nodes, row_fn)]`` for every boundary condition that differentiates an unknown: the
        global rows it replaces -- one set per component of its field, ``(ncomp, n_nodes)`` -- the nodes, and
        :meth:`_flux_row_fn`. ``extra_params`` (a trainable α, say) enters the row functions; the structure
        comes from :meth:`_flux_structure`, built once on concrete values."""
        return [
            (
                np.stack([(b * self._N + idx) for b in self._blocks_of(k)]),  # (ncomp, n_nodes)
                idx,
                self._flux_row_fn(skeleton, extra_params, self._ncomp[k]),
            )
            for _c, idx, k, skeleton in self._flux_structure()
        ]

    def _flux_structure(self):
        """``[(constraint, node_indices, field, skeleton)]`` for the boundary rows — structural, computed
        once on concrete values (the normals and stencils are host-side mesh work, which a crux-traced
        inverse solve cannot redo)."""
        if getattr(self, "_flux_struct", None) is not None:
            return self._flux_struct
        import jax

        out = []
        with jax.ensure_compile_time_eval():
            for c in self._neumann:
                idx, _ = self._node_normals(self._flux_region(c))
                idx = np.asarray(idx, dtype=int)
                owner = self._flux_owner(c, idx)
                out.append((c, idx, owner, self._flux_skeleton(c, idx)))
        self._flux_struct = out
        return out

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

        for e in self._pde_exprs():
            walk(e)
        return not other and (five_point or structured)

    @staticmethod
    def _apply_flux_rows(u, r, flux_rows, t=None):
        """Replace the boundary rows of ``r`` by their conditions, SUMMING where several conditions of one
        field share a node (a corner), so every condition is imposed there rather than the last one."""
        acc = jnp.zeros_like(r)
        hit = jnp.zeros(r.shape[0], dtype=bool)
        for rows, idx, row_fn in flux_rows:  # `rows`: (ncomp, n_nodes) of the owning field's blocks
            out = row_fn(u, t)  # (N,) for a scalar condition, (N, ncomp) for a vector one (a traction)
            for j, rows_j in enumerate(rows):
                acc = acc.at[rows_j].add(out[idx] if out.ndim == 1 else out[idx, j])
                hit = hit.at[rows_j].set(True)
        return jnp.where(hit, acc, r)

    def _flux_gradient_fn(self, idx, scheme, grad_method):
        """``u ↦ ∇u`` at the flux nodes ``idx``, chosen to match the interior stencil (see
        :meth:`_interior_is_five_point`): the second-order quadratic fit (:func:`_quadratic_gradient`) for a
        five-point-type interior, the area-weighted gradient for a gradient-of-gradient interior. An
        explicitly chosen sub-scheme (``":lsq"``, ``":uniform"``, …) keeps the gradient it names."""
        idx = np.asarray(idx)
        if ":" in scheme or not self._interior_is_five_point():
            return lambda u: gradient(u, self.domain, method=grad_method)[idx]
        grid = self._grid or self.domain.mesh_connectivity.get("grid")  # the problem's periodic axes, if any
        if grid is not None:
            return self._grid_boundary_gradient(idx, grid)
        cells = _mesh(self.domain)[1]
        nbrs = _two_ring(cells, self._N, idx)
        pts, jidx = self._pts, jnp.asarray(idx)
        return lambda u: _quadratic_gradient(u, pts, jidx, nbrs)

    @staticmethod
    def _grid_boundary_gradient(idx, grid):
        """``u ↦ ∇u`` at the nodes ``idx`` of a structured grid, with the second-order three-point one-sided
        difference ``(−3u₀ + 4u₁ − u₂)/2h`` along every axis the node ends (central elsewhere). A box face
        is axis-aligned, so ``∇u·n`` there is exactly that one-sided difference.

        The quadratic fit the unstructured closure uses is also second order, but its constant is large
        on a grid: a Neumann flux error ε shifts the mean of the solution by ``∮ε`` whenever the PDE
        fixes the mean through a small reaction term (integrate ``−Δu + u = f``). Measured on
        ``−Δu + u = f`` with ``∂u/∂n = 0`` on all four sides, ``u = cos πx cos πy + ½``: 0.43 relative
        error at h = 0.1 (the mean off by 0.31), and 2.4e-3 with this stencil."""
        shape, spacing, periodic = tuple(grid["shape"]), grid["spacing"], grid.get("periodic") or ()
        jidx = jnp.asarray(idx)

        def grad(u):
            U = jnp.asarray(u).reshape(shape)
            comps = []
            for a, h in enumerate(spacing):
                V = jnp.moveaxis(U, a, 0)
                if a < len(periodic) and periodic[a]:  # wrap-central over the unique nodes
                    W = V[:-1]
                    c = (jnp.roll(W, -1, 0) - jnp.roll(W, 1, 0)) / (2.0 * h)
                    c = jnp.concatenate([c, c[:1]], axis=0)
                else:
                    c = jnp.gradient(V, h, axis=0)
                    c = c.at[0].set((-3.0 * V[0] + 4.0 * V[1] - V[2]) / (2.0 * h))
                    c = c.at[-1].set((3.0 * V[-1] - 4.0 * V[-2] + V[-3]) / (2.0 * h))
                comps.append(jnp.moveaxis(c, 0, a).reshape(-1)[jidx])
            return jnp.stack(comps, axis=1)

        return grad

    def _initial_state(self):
        """Initial nodal state ``u0`` (shape ``(N,)``) from the ``u(initial) - u0`` condition(s), the
        same way :func:`jno.fem` reads its IC — the IC is data found from the constraints, never a flag."""
        u0 = jnp.zeros(self._Ntot)  # a coupled field without an initial condition starts at 0
        allnodes = np.arange(self._N, dtype=int)
        for c in self._ic:
            idx = self._region_nodes(_region_tag(c))  # "initial" → all nodes
            scope = self._params_scope()
            vals = (
                self._eval_value(self._value_side(c), idx, scope)
                if self._uses_params(c, scope)
                else self._condition_value(c, idx)
            )
            k, nodes = self._field_index(c), np.asarray(idx if len(idx) else allnodes)
            for b, vb in zip(self._blocks_of(k), self._split_components(k, vals, len(nodes))):
                u0 = u0.at[jnp.asarray(b * self._N + nodes)].set(vb)
        return u0

    def _initial_velocity(self):
        """Initial velocity ``v0`` over the blocked DOF vector, from the ``ui0.t - v0`` condition(s) on the
        ``initial`` region, as :func:`jno.fem` reads it for ``u_tt``; zero when none is given."""
        v0 = jnp.zeros(self._Ntot)
        for c in self._vel_ic:
            idx = np.asarray(self._region_nodes(_region_tag(c)), dtype=int)
            inner = _unwrap(c)
            if getattr(inner, "op", None) != "-":
                raise ValueError(f"jno.fdm([...]): write an initial velocity as `ui0.t - v0`; got {c!r}.")
            g_node = inner.right if _has_temporal(inner.left) else inner.left  # v0 is the side without u.t
            scope = self._params_scope()
            vals = self._eval_value(g_node, idx, scope) if self._uses_params(g_node, scope) else self._eval_g(g_node, idx)
            k = self._field_index(c)
            for b, vb in zip(self._blocks_of(k), self._split_components(k, vals, len(idx))):
                v0 = v0.at[jnp.asarray(b * self._N + idx)].set(vb)
        return v0

    def solve(self, nonlinear=None, x0=None, profile=False, time=None, *, linear=None, precond=None, save_ts=None):
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
        non-parametric) solve inside a JAX Perfetto trace and writes it to ``./jno_traces``.

        ``save_ts=`` are the times a transient solve returns, exactly as ``fem.solve(save_ts=…)``: the march
        keeps its own step ``Δt`` from ``domain.time`` and the trajectory is sampled at ``save_ts`` (linear
        interpolation between steps). ``save_ts=ts[::k]`` keeps every k-th step."""
        if save_ts is not None and not self._transient:
            raise ValueError("jno.fdm([...]): save_ts= samples a transient march; this problem is steady.")

        def _run():
            trainable = self._trainable_params()
            if self._transient:
                if x0 is not None:
                    raise ValueError("jno.fdm([...]): x0= is rejected for a transient problem — the IC owns the state.")
                if trainable:
                    return self._parametric_node(
                        trainable, nonlinear=nonlinear, linear=linear, precond=precond, time=time, save_ts=save_ts
                    )
                return self._march(nonlinear=nonlinear, time=time, linear=linear, precond=precond, save_ts=save_ts)
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
        if linear is None and precond is None and nonlinear is None and self._default_is_assembled():
            defaults = self.__dict__.setdefault("_default_slots", (_solve.bicgstab(), jno_precond.jacobi()))
            linear, precond = defaults  # the same spec objects every call, so the assembled cache is reused
            # A LINEAR problem on an unstructured mesh gets jno.fem's linear default: Jacobi-preconditioned
            # BiCGStab on the assembled operator. Matrix-free and unpreconditioned, BiCGStab broke down on a
            # 3-D cotangent problem with a Neumann face (Newton residual 5e24) and GMRES stalled at 1.6e-2;
            # its flux rows (~1/h) and Laplacian rows (~1/h²) differ in scale, and Jacobi is what evens them.
        if (linear is not None or precond is not None) and self._grid_slots_matrix_free(
            nonlinear, linear, precond, extra_pins
        ):
            sol = self._grid_linear_steady(extra_params, linear=linear, precond=precond)
        elif linear is not None or precond is not None:
            sol = self._slot_steady(nonlinear, linear, precond, x0, u0, extra_params, extra_pins)
        elif nonlinear is None and extra_pins is None and self._grid_linear_ok():
            sol = self._grid_linear_steady(extra_params)
        elif extra_params is None and extra_pins is None and not isinstance(u0, jax.core.Tracer):
            sol = self._compiled_steady(nonlinear, u0)
        else:
            residual_with_bc = self._steady_residual(extra_params, extra_pins)
            driver = nonlinear or _solve.newton(**_fd_newton_tolerances(residual_with_bc, u0))
            sol = driver(
                residual_with_bc,
                u0,
                linear_solve=_structured_linear_solve(
                    self.domain, bool(self._periodic_axes), self._grid_vcycle(residual_with_bc, u0)
                )
                if single
                else None,
            )
        return sol if single else sol.reshape(self._nf, N)  # coupled: (nf, N), one row per field

    def _stencil_window(self):
        """The offset range this problem's finite-difference schemes read, per axis, from the specs
        themselves -- the window a stencil probe should try first.

        A ``jno.fd(order=k)`` spec knows its own offsets, one-sided closures included: order 2 spans ±3 for a
        second derivative, order 4 ±5, order 6 ±7. Searching for those blindly costs a probe pass per
        candidate (a width-15 window is 225 colours in 2-D), and the search stops before it reaches them.
        ``None`` when no spec says otherwise, which leaves the default candidates."""
        from .stencils import FDStencil
        from .trace import Hessian, Jacobian

        reach = 0
        stack, seen = [_unwrap(c) for c in self._constraints], set()
        while stack:
            node = _unwrap(stack.pop())
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.extend(_iter(node))
            if isinstance(node, (Jacobian, Hessian)):
                spec = getattr(node, "scheme", None)
                if isinstance(spec, FDStencil):
                    deriv = 2 if isinstance(node, Hessian) else 1
                    try:
                        per_node = spec.stencils(deriv, max(8, 2 * (spec.order or 2) + 3))
                    except Exception:  # a spec this grid cannot answer for: leave it to the search
                        continue
                    reach = max(reach, max(max(abs(o) for o in offs) for offs, _w in per_node))
        return (-reach, reach) if reach else None

    def _lattice_vcycle(self, matvec, n, **sweeps):
        """The operator-dependent V-cycle (:func:`jno.utils.solver.lattice_mg.build`) for the ``n``-unknown
        lattice operator ``matvec``, or ``None`` when this problem is not on a lattice (or the system is not
        a whole number of fields on it). Built on concrete values."""
        import jax

        from .utils.solver.lattice_mg import build

        grid = self.domain.mesh_connectivity.get("grid") if self.domain.mesh_connectivity else None
        if grid is None or not self._nodes_are_the_grid(grid) or n % self._N:
            return None
        with jax.ensure_compile_time_eval():
            try:
                vcycle, _levels = build(
                    jax.jit(matvec),  # jitted: the probe applies it once per colour per level
                    tuple(grid["shape"]),
                    nf=n // self._N,
                    periodic=(self._grid or {}).get("periodic", ()),
                    hint=self._stencil_window(),
                    **sweeps,
                )
            except ValueError as exc:  # no bounded stencil to read (a spectral axis): solve without one
                if "no window" not in str(exc):
                    raise
                from .utils.logger import get_logger

                get_logger().info(
                    f"jno.fdm: {exc} The solve runs without a multigrid preconditioner; pass "
                    "linear=/precond= to choose another."
                )
                return None
        return vcycle

    def _grid_vcycle(self, residual, at):
        """The V-cycle for this problem's tangent at ``at``. Built once and frozen for the Newton loop that
        uses it: the tangent drifts as Newton proceeds, and the hierarchy built here does not follow it,
        which changes how fast the inner solve converges, never what it converges to.

        Built on the tangent with its **Dirichlet rows and columns eliminated**, as the linear path solves it,
        and applied as identity there. A Dirichlet row of a Newton tangent is a unit row, scale 1, next to
        interior rows of scale 1/h²; a smoother reads those as one operator and the V-cycle then amplifies
        (measured: a standalone factor of 107 on the unmasked tangent, 0.28 on the eliminated one)."""
        import jax

        at = jnp.zeros(self._Ntot) if isinstance(at, jax.core.Tracer) else jnp.asarray(at)
        with jax.ensure_compile_time_eval():
            mask_host = np.ones(self._Ntot)
            for b, idx, _vals in self._dirichlet_rows():
                mask_host[b * self._N + np.asarray(idx, dtype=int)] = 0.0
            mask = jnp.asarray(mask_host)
        vcycle = self._lattice_vcycle(lambda v: jax.jvp(residual, (at,), (v * mask,))[1] * mask, self._Ntot)
        if vcycle is None:
            return None
        return lambda r: jnp.where(mask > 0, vcycle(r * mask), r)

    def _default_is_assembled(self):
        """Does the default steady solve go through the assembled operator? For a linear, single-field
        problem on an unstructured mesh, yes; a structured grid keeps its matrix-free GMRES + multigrid,
        and a nonlinear problem its matrix-free Newton."""
        if self._nf != 1 or self.domain.mesh_connectivity.get("grid") is not None:
            return False
        import jax

        with jax.ensure_compile_time_eval():
            return self._is_affine("steady", lambda: self._steady_residual(self._current_params()), self._Ntot)

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
        make_probe = lambda: self._steady_residual(self._current_params())  # noqa: E731  (jitted: rebuilt per probe)
        with jax.ensure_compile_time_eval():
            probe = make_probe()
            linear_problem = self._is_affine("steady", make_probe, self._Ntot)
            self._sparsity("steady", probe, jnp.zeros(self._Ntot), make=make_probe)  # the pattern search needs values
        if linear_problem:
            if nonlinear is not None:
                raise ValueError(
                    "jno.fdm: nonlinear= given, but this problem is linear — there is no Newton loop to "
                    "configure. Drop nonlinear=, or pick the linear solver with linear=."
                )
            with jax.ensure_compile_time_eval():
                self._require_symmetric(
                    linear,
                    lambda: self._dirichlet_lift("steady", self._sparse_operator("steady", make_probe(), zeros))[0],
                    key="steady",
                )
            eager = extra_params is None and extra_pins is None and not isinstance(u0, jax.core.Tracer)
            cache = self.__dict__.setdefault("_assembled_cache", {})
            # Keyed on the specs themselves: equal keyed specs (two `jno.solve.gmres()` calls) share an entry,
            # a keyless one matches only itself, and the key keeps it alive. `id()` missed equal specs.
            key = (linear, precond, self._data_fingerprint())
            if eager and key in cache:
                A, rhs = cache[key][2], cache[key][3]  # a repeat solve: the operator is a constant of the problem
            else:
                A, lift_rhs = self._dirichlet_lift("steady", self._sparse_operator("steady", residual, zeros))
                rhs = lift_rhs(-residual(zeros))
                if eager:
                    cache[key] = (linear, precond, A, rhs)
            return compose_linear_solve_fn(linear, precond, x0, fem=self)(A, rhs)
        tangent = lambda w: self._sparse_operator("steady", residual, w)  # noqa: E731
        if nonlinear is None:
            # FDM can always assemble its tangent, so the Newton that uses it is the default here: it is
            # what lets jacobi / amg / gmg precondition a nonlinear strong-form solve at all (the
            # matrix-free JVP has no diagonal or matrix to give them).
            nonlinear = _solve.newton(direct=True)
        precond = self._frozen_precond(precond, lambda: tangent(u0 if not isinstance(u0, jax.core.Tracer) else zeros))
        linear = self._newton_linear(linear, "steady", make_probe, u0 if not isinstance(u0, jax.core.Tracer) else zeros)
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
        """``gmg`` reads the operator's stencil on the grid, so the system must BE a lattice operator: a whole
        number of fields on this grid. A coupled system and the augmented ``[u; v]`` state of a ``u.tt``
        problem both are (the V-cycle probes every field pair); a system of some other size is not."""
        if precond is None:
            return
        from .precond import _GMG
        from .utils.solver.solver_api import _specs_in

        if n % self._N and any(isinstance(s, _GMG) for s in _specs_in(precond)):
            raise ValueError(
                f"jno.precond.gmg() preconditions fields on the grid ({self._N} nodes each), and this system "
                f"has {n} unknowns, which is not a whole number of them. Use jno.precond.amg(), which works on "
                "any assembled operator."
            )

    def _steady_residual(self, extra_params=None, extra_pins=None):
        """The steady residual with every boundary row folded in, as a function of the DOF vector."""
        import jax

        N = self._N
        if extra_pins is not None and self._nf != 1:
            raise NotImplementedError(
                f"jno.fdm: interface pins (a domain-decomposition subdomain) index the nodes of ONE field, and this "
                f"system has {self._nf} DOF blocks (a coupled system or a vector unknown). Only field 0 would be pinned. "
                "Couple a single scalar field per subdomain."
            )
        residual_fn = self._pde_residual_fn(extra_params=extra_params)
        rows = self._dirichlet_rows(extra_params)
        flux_rows = self._flux_rows(extra_params)
        periodic_rows = self._periodic_rows()  # (secondary, main) face DOF pairs per periodic axis

        def residual_with_bc(u):
            r = residual_fn(u)
            # Flux rows first (`a·(∇u·n) + b`, with a = F(1)-F(0), b = F(0) — Neumann/Robin/etc.), then
            # the periodic ties, then Dirichlet: a node carrying several (a 2-D corner, or a 3-D edge)
            # resolves to the essential Dirichlet value — the Dirichlet row is set last.
            r = self._apply_flux_rows(u, r, flux_rows) if flux_rows else r
            for secondary, main in periodic_rows:  # periodic: the redundant secondary face ≡ the main face
                r = r.at[secondary].set(u[secondary] - u[main])
            if extra_pins is not None:  # interface pin (a coupled subdomain's complement) — before the
                pidx, pvals = extra_pins  # authored Dirichlet, so the physical outer BC still wins on ∂Ω
                r = r.at[pidx].set(u[pidx] - pvals)  # node indices: the single field's block (checked above)
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
        key = (nonlinear, self._data_fingerprint())  # a nonlinear spec compares by identity; the key keeps it alive
        entry = cache.get(key)
        if entry is None:
            residual = self._steady_residual()
            driver = nonlinear or _solve.newton(**_fd_newton_tolerances(residual, u0))
            linear = (
                _structured_linear_solve(self.domain, bool(self._periodic_axes), self._grid_vcycle(residual, u0))
                if self._nf == 1
                else None
            )
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

    # ------------------------------------------------------------------------------------------------
    # A linear problem on a structured grid: one Krylov solve on the interior, no Newton
    # ------------------------------------------------------------------------------------------------
    def _grid_linear_ok(self):
        """Does the steady solve take the structured linear path? One scalar field on a lattice, with an
        affine residual: then ``R(u) = 0`` is one preconditioned Krylov solve rather than a Newton step
        around it.

        It used to need Dirichlet data on the whole boundary ring, no flux row and no periodic tie, because
        the V-cycle it preconditioned with was built for ``-Δ`` with a Dirichlet ring. The V-cycle is built
        from this operator now (:meth:`_grid_vcycle`), so those rows are just rows. Decided once per problem.
        """
        if getattr(self, "_grid_linear", None) is not None:
            return self._grid_linear
        ok = False
        grid = self.domain.mesh_connectivity.get("grid") if self.domain.mesh_connectivity else None
        if self._nf == 1 and not self._transient and grid is not None and self._nodes_are_the_grid(grid):
            import jax

            with jax.ensure_compile_time_eval():
                ok = self._is_affine("steady", lambda: self._steady_residual(self._current_params()), self._Ntot)
        self._grid_linear = ok
        return ok

    def _grid_slots_matrix_free(self, nonlinear, linear, precond, extra_pins):
        """Explicit ``linear=`` / ``precond=`` slots on a structured linear problem stay matrix-free when
        they can: a Krylov solver (cg, gmres, bicgstab) and the multigrid V-cycle or no preconditioner.
        Anything else (a direct solve, AMG, ILU, Jacobi) needs the matrix and takes the assembled route."""
        if nonlinear is not None or extra_pins is not None or not self._grid_linear_ok():
            return False
        krylov = linear is None or getattr(linear, "name", None) in ("cg", "gmres", "bicgstab")
        return krylov and (precond is None or type(precond).__name__ == "_GMG")

    def _dirichlet_nodes(self):
        """Host-side node indices of the Dirichlet rows, found once.

        They are structural -- which nodes a region holds -- but computing them means array operations, and
        inside a trace (a crux-driven inverse) every array operation becomes traced, so a recomputation there
        would hand a tracer to numpy."""
        cached = self.__dict__.get("_dirichlet_node_cache")
        if cached is None:
            import jax

            with jax.ensure_compile_time_eval():
                cached = np.unique(
                    np.concatenate([np.asarray(r[1], dtype=int) for r in self._dirichlet_rows()] or [np.zeros(0, int)])
                )
            self._dirichlet_node_cache = cached
        return cached

    def _grid_ring(self, grid):
        """The nodes on the boundary ring of the structured grid, by lattice index."""
        shape = np.asarray(grid["shape"], int)
        k = np.rint(
            (np.asarray(self._pts)[:, : len(shape)] - np.asarray(grid["origin"], float))
            / np.asarray(grid["spacing"], float)
        )
        return np.nonzero(np.any((k == 0) | (k == shape - 1), axis=1))[0]

    def _grid_linear_steady(self, extra_params=None, tol=1e-12, *, linear=None, precond=None):
        """Solve ``R(u) = 0`` for an affine ``R`` on a structured grid as ONE linear solve.

        The Dirichlet rows are eliminated -- ``u = u_D + x`` with ``x`` zero on the boundary ring -- which
        leaves ``A x = -R(u_D)`` on the interior, applied matrix-free as the JVP of ``R``. ``A`` is
        symmetric for a self-adjoint operator (a Laplacian, a diffusion with a coefficient, plus a
        reaction); a two-vector probe decides, and a symmetric system gets preconditioned conjugate
        gradients (Hestenes & Stiefel 1952) and any other one GMRES that stops at convergence, both
        preconditioned by the V-cycle built from **this** operator's stencil
        (:func:`jno.utils.solver.lattice_mg.build`). The Newton-GMRES path this replaces linearised
        the residual, ran a Newton step and a full 30-vector GMRES cycle per restart whatever the
        convergence: measured at 0.9M nodes (3-D Poisson), 380 MB and 0.170 s against 143 MB and 0.024 s
        with 8 CG iterations.

        Differentiable: ``lax.custom_linear_solve`` gives the adjoint (a trainable parameter in the
        coefficient, the source or the boundary data). The relative residual is checked on the concrete
        result, and a solve that did not reach ``tol`` raises."""
        import jax

        slots = linear is not None or precond is not None
        # the tolerance the solve was asked for: the user's Krylov spec's, or this path's own
        check_tol = ((getattr(linear, "tolerance", None) if linear is not None else 1e-8) if slots else tol) or tol
        N = self._N
        eager = extra_params is None
        cache = self.__dict__.setdefault("_grid_linear_cache", {})
        key = (self._data_fingerprint(), linear, precond)  # the specs themselves: equal keyed specs share a compile
        if eager and cache.get("key") == key:
            fn = cache["fn"]
        else:
            # built on a cache MISS only: a repeat solve reuses the compiled solve, V-cycle included
            residual = self._steady_residual(extra_params)
            rows = self._dirichlet_rows(extra_params)
            mask_host = np.ones(N)
            mask_host[self._dirichlet_nodes()] = 0.0
            mask = jnp.asarray(mask_host)
            # The V-cycle reads the eliminated operator itself, at the parameters' current values (a traced
            # parameter would otherwise put the whole probe inside every solve); it preconditions, so its
            # being one step behind the parameter changes nothing about the answer.
            sweeps = {"n_pre": precond.n_pre, "n_post": precond.n_post} if precond is not None else {}
            with jax.ensure_compile_time_eval():
                # every input the probe closes over is built HERE: inside a crux trace even
                # `jnp.asarray(numpy_array)` is a tracer, and one traced input makes the whole probe traced
                probe_residual = self._steady_residual(self._current_params())
                zeros = jnp.zeros(self._Ntot)
                probe_mask = jnp.asarray(mask_host)
                # jitted: the probe applies it once per colour per level, and an un-jitted jvp of a jitted
                # residual re-traces and re-compiles on every one of them (12 s of setup at 1M nodes)
                probe_mv = jax.jit(lambda v: jax.jvp(probe_residual, (zeros,), (v * probe_mask,))[1] * probe_mask)
                vcycle = self._lattice_vcycle(probe_mv, self._Ntot, **sweeps)
                symmetric = self._grid_linear_symmetric(lambda: self._steady_residual(self._current_params()), probe_mask)
            if slots and getattr(linear, "name", None) == "cg" and not symmetric:
                raise ValueError(
                    "jno.fdm: linear=jno.solve.cg() needs a symmetric operator, and this one is not (after "
                    "eliminating the Dirichlet rows, a random-probe test gives wᵀAv ≠ vᵀAw -- an advection "
                    "term, or a one-sided boundary stencil). Use linear=jno.solve.gmres() or bicgstab()."
                )

            linear_spec = linear if linear is not None else _solve.gmres()

            def fn(rows=rows):
                uD = jnp.zeros(N)
                for _, idx, vals in rows:
                    uD = uD.at[idx].set(jnp.broadcast_to(jnp.asarray(vals), (idx.shape[0],)))
                matvec = lambda v: jax.jvp(residual, (uD,), (v * mask,))[1] * mask  # noqa: E731
                b = -residual(uD) * mask
                # no V-cycle (an operator with no bounded stencil, e.g. a spectral axis): solve unpreconditioned
                M = (lambda r: vcycle(r * mask) * mask) if vcycle is not None else (lambda r: r)  # noqa: E731
                if slots:  # the user's Krylov spec (it carries its own custom_linear_solve and tolerance)
                    from .utils.solver.solver_api import LinearOperator

                    x = linear_spec(LinearOperator.from_matvec(matvec), b, M=M if precond is not None else None, x0=None)
                    rn = jnp.linalg.norm(matvec(x) - b) / jnp.maximum(jnp.linalg.norm(b), 1e-300)
                    return uD + x, rn
                krylov = _pcg if symmetric else _gmres_incremental

                def solve(mv, rhs):  # (x, relative residual the Krylov loop ended at) -- no extra matvec
                    x, rn = krylov(mv, rhs, M, tol)
                    return x, rn / jnp.maximum(jnp.linalg.norm(rhs), 1e-300)

                # the V-cycle is symmetric, so the same preconditioned Krylov solve serves Aᵀ (the adjoint)
                x, rel = jax.lax.custom_linear_solve(
                    matvec, b, solve, transpose_solve=solve, symmetric=symmetric, has_aux=True
                )
                return uD + x, rel

            if eager:
                fn = jax.jit(fn)
                cache.update(key=key, fn=fn)
        u, rel = fn()
        if not isinstance(rel, jax.core.Tracer) and not float(rel) <= 100 * check_tol:
            raise RuntimeError(
                f"jno.fdm: the structured linear solve stopped at relative residual {float(rel):.2e} against "
                f"{check_tol:.0e} ({'conjugate gradients' if self._grid_linear_symmetric_flag else 'GMRES'} with "
                "multigrid). The operator may be indefinite or badly scaled for this preconditioner; pick the "
                "solver explicitly with linear=jno.solve.gmres() / jno.solve.lu() and precond=."
            )
        return u

    def _grid_linear_symmetric(self, make, mask):
        """Is the eliminated operator of the residual ``make()`` builds symmetric? ``wᵀA v = vᵀA w`` for two
        random vectors: exact for a symmetric linear operator, and violated with probability one otherwise.
        Decided once, for any value of the parameters (:meth:`_for_every_parameter`)."""
        if getattr(self, "_grid_linear_symmetric_flag", None) is None:
            import jax

            def symmetric(_):
                residual = make()
                rng = np.random.default_rng(7)
                v, w = (jnp.asarray(rng.standard_normal(self._N)) * mask for _ in range(2))
                z = jnp.zeros(self._N)
                Av = jax.jvp(residual, (z,), (v,))[1] * mask
                Aw = jax.jvp(residual, (z,), (w,))[1] * mask
                a, b_ = float(w @ Av), float(v @ Aw)
                return abs(a - b_) <= _rtol() * max(abs(a) + abs(b_), np.finfo(float).tiny)

            with jax.ensure_compile_time_eval():
                self._grid_linear_symmetric_flag = all(self._for_every_parameter(symmetric))
        return self._grid_linear_symmetric_flag

    def pinned_solver(self, node_ids, *, nonlinear=None):
        """A **reusable** ``f(values) -> field`` that solves the subdomain with ``node_ids`` pinned to
        ``values`` (the interface Dirichlet data from a neighbour) on top of the authored conditions. It is
        the step the Schwarz driver (:mod:`jno.dd`) takes, :meth:`_steady_solve` with ``extra_pins``, so every
        flux row, periodic tie, trainable parameter and cached compilation of the plain solve applies, from one
        code path rather than a second hand-written Newton."""
        pin = jnp.asarray(np.asarray(node_ids, dtype=int))
        return lambda values: self._steady_solve(nonlinear=nonlinear, extra_pins=(pin, jnp.asarray(values)))

    def solve_pinned(self, node_ids, values, *, nonlinear=None):
        """One-shot: solve with ``node_ids`` pinned to ``values`` (see :meth:`pinned_solver`, which the
        Schwarz driver builds once and reuses)."""
        return self.pinned_solver(node_ids, nonlinear=nonlinear)(values)

    def _parametric_node(self, trainable, *, nonlinear=None, x0=None, linear=None, precond=None, time=None, save_ts=None):
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

        if self._transient:
            # A warm-up march at the parameters' current values makes every STRUCTURAL decision on concrete
            # values (linearity, sparsity pattern, symmetry, time-variance, the u-independence of the time
            # coefficients); each is cached, so the traced march below only reuses them. Costs one solve.
            self._march(nonlinear=nonlinear, time=time, linear=linear, precond=precond)

        def _solve(*values):  # values = the parameters' current (crux-trained) values
            extra = {
                lid: eqx.tree_at(lambda m: m.value, modules[lid], jnp.asarray(v).astype(modules[lid].value.dtype))
                for lid, v in zip(lids, values)
            }
            if not self._transient:
                return self._steady_solve(nonlinear=nonlinear, x0=x0, extra_params=extra, linear=linear, precond=precond)
            # A transient solve reads the parameters through `_override`, which every evaluator in the march
            # consults (the PDE, boundary and flux values, time coefficients, initial values).
            self._override = extra
            try:
                return self._march(nonlinear=nonlinear, time=time, linear=linear, precond=precond, save_ts=save_ts)
            finally:
                self._override = None

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

        flux_rows = self._flux_rows()
        bmask_np, _ = self._dirichlet_values_at(float(t0))
        algebraic = bmask_np.copy()  # Dirichlet + flux nodes are algebraic (zero mass row)
        for row in flux_rows:
            algebraic[np.asarray(row[0]).reshape(-1)] = True  # every component's rows
        bmask, algebraic = jnp.asarray(bmask_np), jnp.asarray(algebraic)

        spatial_res = self._pde_residual_fn(spatial=True)

        def boundary_rows(wn, r, t):  # overwrite the flux and Dirichlet rows of r with their algebraic constraints
            r = self._apply_flux_rows(wn, r, flux_rows, t) if flux_rows else r  # the same folding as the steady solve
            return jnp.where(bmask, wn - self._dirichlet_values_at(t)[1], r)  # Dirichlet wins over flux

        slots = dict(nonlinear=nonlinear, linear=linear, precond=precond)
        if self._time_order == 2:
            traj = self._march_second_order(spatial_res, boundary_rows, algebraic, t0, t1, dt, save_ts, time, slots)
            # (steps, field, node) for a vector or coupled field, as the first-order march returns
            return traj if self._nf == 1 else traj.reshape(traj.shape[0], self._nf, self._N)

        c_of = self._mass_coefficient()  # u.t coefficient c(x, t): 1 for a plain u.t, ρcₚ(x) for ρcₚ(x)·u.t
        n = self._Ntot  # a coupled system marches the blocked vector [u_0; …; u_{nf-1}]
        diag = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)

        def M(t=None):  # 0 on Dirichlet + flux rows, and on a field whose equation has no u.t
            return jsparse.BCOO((jnp.where(algebraic, 0.0, c_of(t)), diag), shape=(n, n))

        def residual(wn, t, args):  # M u̇ + R = 0 → interior u̇ = -R_spatial; flux/Dirichlet rows algebraic
            return boundary_rows(wn, spatial_res(wn, t), t)

        traj = self._run_block(M, residual, self._initial_state(), (t0, t1, dt), {}, save_ts, time, slots)
        return traj if self._nf == 1 else traj.reshape(traj.shape[0], self._nf, self._N)  # (steps, field, node)

    def _run_block(self, M, residual, state0, window, metadata, save_ts, time, slots):
        """Build the semidiscrete block ``M ẏ + R(y) = 0`` and march it.

        With no solver slots it is a residual block, stepped by the matrix-free Newton–Krylov (the
        historic default). With ``linear=`` / ``precond=`` it is composed exactly as ``fem.solve`` composes
        a transient: an affine ``R`` becomes a **linear** block (``A`` assembled once as a sparse matrix,
        ``c = −R(0)``), so every step is one preconditioned linear solve and an AMG or LU setup is built
        once before the march; a nonlinear ``R`` keeps the Newton step, with the assembled tangent
        available to a direct solver. A ``nonlinear=`` slot configures the per-step Newton."""
        import jax

        from .utils.solver.backend_blocks import SemidiscreteTimeBlock, _block_time_grid
        from .utils.solver.solver_api import compose_transient_step_solvers

        t0, t1, dt = window
        nonlinear, linear, precond = slots["nonlinear"], slots["linear"], slots["precond"]
        common = dict(state0=state0, t0=float(t0), t1=float(t1), dt=float(dt), metadata=dict(metadata))
        mass_of = M if callable(M) else (lambda t=None: M)  # the mass may carry a time-dependent coefficient
        M = mass_of(float(t0))
        frozen = lambda y: residual(y, float(t0), {})  # noqa: E731  (the operator at the start time)
        n = int(state0.size)
        self._check_precond_shape(precond, n)
        if linear is not None or precond is not None:
            zeros = jnp.zeros(n)
            with jax.ensure_compile_time_eval():  # structure is decided on concrete values
                # the march's residual and mass are not jitted, so they read a probe's parameter values at
                # call time (`_live_params`) and need no rebuilding
                self._sparsity("march", frozen, jnp.zeros(n))
                linear_problem = (
                    self._is_affine("march", lambda: frozen, n)
                    and not self._operator_varies_in_time(lambda: residual, n, float(t0), float(t1), key="march")
                    and not self._mass_varies_in_time(lambda: mass_of, float(t0), float(t1))
                )
            if linear_problem:
                # Time-dependent DATA (a source f(x, t), a boundary value g(x, t)) rides the block's forcing
                # f(t) = −R(0, t); only the operator must be constant, which was just checked. Freezing
                # −R(0, t0) as a constant bias silently held the data at the start time.
                A, lift_rhs = self._dirichlet_lift("march", self._sparse_operator("march", frozen, zeros))
                if self._time_order == 1:  # the lift knows the fields' Dirichlet rows, not the [u; v] layout
                    self._require_symmetric(
                        linear,
                        lambda: self._dirichlet_lift("march", self._sparse_operator("march", frozen, zeros))[0],
                        key="march",
                    )
                    forcing = lambda t, args: lift_rhs(-residual(jnp.zeros(n), t, args))  # noqa: E731
                else:
                    A = self._sparse_operator("march", frozen, zeros)
                    forcing = lambda t, args: -residual(jnp.zeros(n), t, args)  # noqa: E731
                block = SemidiscreteTimeBlock(
                    M=M, A=A, affine_bias=jnp.zeros(n), forcing_vector_fn=forcing, forcing_mode="user_callback", **common
                )
            else:
                block = SemidiscreteTimeBlock(
                    mass=lambda t, args: mass_of(t),
                    residual=residual,
                    jacobian=lambda w, t, args: self._sparse_operator("march", lambda y: residual(y, t, args), w),
                    **common,
                )
                # The block carries its assembled tangent, so the per-step Newton uses it (as the steady
                # solve does): a matrix-free JVP has no diagonal for jacobi, nor a matrix for amg or lu.
                nonlinear = nonlinear or _solve.newton(direct=True)
                if self._time_order == 1:
                    with jax.ensure_compile_time_eval():
                        linear = self._newton_linear(linear, "march", lambda: frozen, state0)
        else:
            # The per-step solve stays the driver's matrix-free Newton-Krylov. A V-cycle on the step operator
            # was tried and is slower here: the step operator is the mass plus θΔt times the spatial one, so
            # at a usable Δt it is strongly diagonally dominant and an unpreconditioned Krylov solve already
            # takes a handful of iterations, while nesting a V-cycle inside the Newton inside the scan costs
            # a large compile and 7 GB. Measured, 50 heat steps: 0.55 s at 66k nodes and 1.6 s at 263k
            # (Δt = 1e-2), against 6.8 s and 16 s for the assembled `precond=gmg()` slot.
            block = SemidiscreteTimeBlock(mass=lambda t, args: mass_of(t), residual=residual, **common)
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

        N = self._Ntot  # the blocked vector: a vector or coupled field marches every component
        m_nodes = self._time_coefficients(0.0, 1.0, "`u.tt` inertia coefficient", "nonlinear inertia `m(u)·u.tt`")
        c_nodes = self._time_coefficients(1.0, 0.0, "`u.t` damping coefficient", "nonlinear damping `c(u)·u.t`")
        diag = jnp.stack([jnp.arange(2 * N), jnp.arange(2 * N)], axis=1)

        def M(t=None):
            mass = jnp.concatenate([jnp.where(algebraic, 0.0, 1.0), jnp.where(algebraic, 0.0, m_nodes(t))])
            return jsparse.BCOO((mass, diag), shape=(2 * N, 2 * N))

        def residual(y, t, args):
            u, v = y[:N], y[N:]
            ru = boundary_rows(u, -v, t)  # interior: u̇ = v; boundary: the algebraic constraint on u
            rv = jnp.where(algebraic, v, c_nodes(t) * v + spatial_res(u, t))
            return jnp.concatenate([ru, rv])

        v0 = jnp.where(algebraic, 0.0, self._initial_velocity())
        # The Newmark shortcut eliminates v and solves for the new displacement alone. For ONE field that is
        # half the unknowns and a well-conditioned step operator; for a vector or coupled field its step
        # operator comes out rank-deficient (measured: rank 152 of 162 on a two-component wave) and CG
        # returns NaN, so such a system marches the augmented [u; v] block instead -- the same answer, twice
        # the unknowns per step. Making Newmark work for systems is open.
        if time is None and self._nf == 1:
            traj = self._newmark(spatial_res, boundary_rows, algebraic, m_nodes, c_nodes, v0, (t0, t1, dt), slots)
            if save_ts is None:
                return traj
            from .utils.solver.backend_blocks import _resample_trajectory

            grid_ts = jnp.linspace(float(t0), float(t1), traj.shape[0], dtype=traj.dtype)
            return _resample_trajectory(traj, grid_ts, save_ts, traj.dtype)
        state0 = jnp.concatenate([self._initial_state(), v0])
        metadata = {"theta": 0.5, "second_order": True}
        return self._run_block(M, residual, state0, (t0, t1, dt), metadata, save_ts, time, slots)[:, :N]

    def _newmark(self, spatial_res, boundary_rows, algebraic, m_nodes, c_nodes, v0, window, slots):
        """The default ``u.tt`` march: Newmark average acceleration (Newmark 1959, J. Eng. Mech. Div. 85),
        solved for the new displacement ALONE.

        It is the same trapezoidal step as the augmented ``[u; v]`` march — ``u⁺ = u + Δt(v + v⁺)/2`` and
        ``m(v⁺ − v)/Δt + c(v + v⁺)/2 + (R(u⁺) + R(u))/2 = 0`` — with ``v⁺`` eliminated, which leaves one
        equation of the original size per step:

            (2m/Δt² + c/Δt)(u⁺ − u) − (2m/Δt)·v + ½(R(u⁺) + R(u)) = 0,      v⁺ = 2(u⁺ − u)/Δt − v.

        Half the unknowns, and with Δt ≈ h the step operator ``2m/Δt² + ½∂R`` is well conditioned (its
        spectrum spans a small factor), so a Krylov solve converges in a few iterations. The augmented
        system was twice the size and non-symmetric: BiCGStab broke down on it (NaN at step 0, 263k nodes)
        and GMRES took 53 ms/step. Boundary rows stay algebraic constraints on ``u⁺``. The per-step Newton
        runs inside one ``lax.scan``; its convergence is judged after the scan, where it is concrete."""
        import jax

        from .utils.solver.history_march import _check_march_converged
        from .utils.solver.solver_api import LinearOperator, compose_nonlinear_solve_fn

        t0, t1, dt = (float(w) for w in window)
        n_steps = int(round((t1 - t0) / dt))

        def alpha(t):  # inertia m(x, t) and damping c(x, t) at the step time
            return jnp.where(algebraic, 0.0, 2.0 * m_nodes(t) / dt**2 + c_nodes(t) / dt)

        def beta(t):
            return jnp.where(algebraic, 0.0, 2.0 * m_nodes(t) / dt)

        u0 = self._initial_state()
        r0 = spatial_res(u0, t0)

        # The interior rows carry 2m/Δt² (~1e4–1e5); the boundary constraint rows are O(1). Left as is, a
        # Jacobi-preconditioned GMRES stops on the preconditioned residual while the true one sits at 1e-4
        # (measured: every damped / flux wave test tripped the solver's residual gate). Scaling a
        # constraint row does not move its solution, so the rows are brought to the interior's scale.
        row_scale = jnp.where(algebraic, jnp.max(alpha(t0)), 1.0)

        def step_residual(w, u, v, r_now, t_next):
            g = alpha(t_next) * (w - u) - beta(t_next) * v + 0.5 * (spatial_res(w, t_next) + r_now)
            return row_scale * boundary_rows(w, g, t_next)

        nonlinear, linear, precond = slots["nonlinear"], slots["linear"], slots["precond"]
        self._check_precond_shape(precond, self._Ntot)
        probe = lambda w: step_residual(w, u0, v0, r0, t0 + dt)  # noqa: E731
        with jax.ensure_compile_time_eval():
            linear_problem = (
                nonlinear is None
                and self._is_affine("newmark", lambda: probe, self._Ntot)
                and not self._operator_varies_in_time(
                    lambda: lambda w, t, args: step_residual(w, u0, v0, r0, t), self._Ntot, t0 + dt, t1, key="newmark"
                )
            )
        if linear_problem:
            return self._newmark_linear(
                step_residual, probe, algebraic, u0, v0, r0, spatial_res, t0, dt, n_steps, linear, precond
            )
        if linear is not None or precond is not None:
            with jax.ensure_compile_time_eval():
                self._sparsity("newmark", probe, u0)
            spec = nonlinear or _solve.newton(direct=True)
            precond = self._frozen_precond(precond, lambda: self._sparse_operator("newmark", probe, u0))
            composed = compose_nonlinear_solve_fn(spec, linear, precond, fem=self)

            def solve_step(G, guess):
                return composed(G, guess, jacobian=lambda w: self._sparse_operator("newmark", G, w))

        else:
            spec = nonlinear or _solve.newton()
            gmres = _solve.gmres()
            inner = lambda mv, rhs: gmres(LinearOperator.from_matvec(mv), rhs)  # noqa: E731

            def solve_step(G, guess):
                return spec(G, guess, linear_solve=inner)

        def body(carry, _):
            u, v, r_now, t = carry
            t_next = t + dt
            G = lambda w: step_residual(w, u, v, r_now, t_next)  # noqa: E731
            guess = u + dt * v
            u_next = solve_step(G, guess)
            v_next = jnp.where(algebraic, 0.0, 2.0 * (u_next - u) / dt - v)
            norms = (jnp.linalg.norm(G(u_next)), jnp.linalg.norm(G(guess)))
            return (u_next, v_next, spatial_res(u_next, t_next), t_next), (u_next, *norms)

        carry0 = (u0, v0, r0, jnp.asarray(t0, dtype=u0.dtype))
        _, (traj, r_end, r_start) = jax.lax.scan(body, carry0, None, length=n_steps)

        class _Tolerances:  # what the march check judges against: the tolerances the Newton spec carries
            tolerances = (float(spec.traits.get("rtol", 1e-8)), float(spec.traits.get("atol", 1e-8)))

        grid = t0 + dt * np.arange(1, n_steps + 1)
        _check_march_converged(r_end, r_start, grid, _Tolerances, what="u_tt march", coord="t")
        return jnp.concatenate([u0[None], traj])

    def _newmark_linear(self, step_residual, probe, algebraic, u0, v0, r0, spatial_res, t0, dt, n_steps, linear, precond):
        """The Newmark march for a LINEAR problem: the step operator ``K = diag(2m/Δt² + c/Δt) + ½A`` does
        not change, so it is assembled once and its preconditioner set up once, and every step is one
        linear solve for the correction to the predictor ``u + Δt·v``. The nonlinear route re-linearised
        the same matrix inside a Newton loop at every step (measured, 1M nodes: 43 ms/step before).
        Default solver: CG with Jacobi when ``K`` is verified symmetric (a structured grid, after the
        Dirichlet lift), else GMRES with Jacobi; a ``linear=`` / ``precond=`` slot replaces either. A failed step is caught by the solver's residual
        gate, drained in :meth:`solve`."""
        import jax

        from .utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond, prepare_precond

        with jax.ensure_compile_time_eval():
            self._sparsity("newmark", probe, u0)
            K, lift_rhs = self._dirichlet_lift("newmark", self._sparse_operator("newmark", probe, u0))
            make_K = lambda: self._dirichlet_lift("newmark", self._sparse_operator("newmark", probe, u0))[0]  # noqa: E731
            self._require_symmetric(linear, make_K, key="newmark")
            symmetric = self._is_symmetric(make_K, key="newmark")
            coo = self._sparsity_cache["newmark"][0].tocoo()
            on_diag = np.nonzero(coo.row == coo.col)[0]
            diag = jnp.zeros(K.shape[0]).at[jnp.asarray(coo.row[on_diag])].add(K.data[jnp.asarray(on_diag)])
        is_d = jnp.asarray(self._dirichlet_mask())
        # The Krylov iterations use the STENCIL, not the assembled matrix: on a 1M-node structured grid a
        # BCOO matvec cost 1.12 ms on GPU against 0.12 ms for the stencil's JVP (the assembled pattern also
        # carries structural zeros). The lifted operator acts as K on v with its Dirichlet entries zeroed,
        # plus K's diagonal on the Dirichlet rows. The assembled K stays for what needs a matrix: the
        # preconditioner setup and a direct solver.
        _, jvp = jax.linearize(probe, u0)
        fast = LinearOperator.from_matvec(
            jax.jit(lambda w: jvp(jnp.where(is_d, 0.0, w)) + jnp.where(is_d, diag * w, 0.0)),
            diag_fn=lambda: diag,
            shape=K.shape,
        )
        # Default: CG where the step operator is verified symmetric (a structured grid, after the lift) —
        # measured 7.1 s against 27 s for GMRES on 1M nodes x 300 steps, same answer — else GMRES.
        solver = linear or (_solve.cg() if symmetric else _solve.gmres())
        direct = bool(getattr(solver, "direct", False))
        op = LinearOperator(K) if direct else fast
        spec = precond if precond is not None else (None if direct else jno_precond.jacobi())
        M = None
        if spec is not None:
            prepare_precond(spec, self)
            M = materialize_precond(spec, PrecondContext(LinearOperator(K), self))

        def body(carry, _):
            u, v, r_now, t = carry
            t_next = t + dt
            guess = u + dt * v
            # Solve for the CORRECTION to the predictor, K·Δ = −G(guess): the solver's relative tolerance
            # then applies to the step residual itself, not to a right-hand side dominated by (2m/Δt²)·u,
            # where 1e-8 of it let different solvers drift apart by ~3e-6 over 20 steps.
            rhs = lift_rhs(-step_residual(guess, u, v, r_now, t_next))
            delta = solver(op, rhs) if direct else solver(op, rhs, M=M)
            u_next = guess + delta
            v_next = jnp.where(algebraic, 0.0, 2.0 * (u_next - u) / dt - v)
            return (u_next, v_next, spatial_res(u_next, t_next), t_next), u_next

        _, traj = jax.lax.scan(body, (u0, v0, r0, jnp.asarray(t0, dtype=u0.dtype)), None, length=n_steps)
        return jnp.concatenate([u0[None], traj])


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
