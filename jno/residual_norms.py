"""Function-space norms of fields on a domain: the backends behind ``expr.norm(space)``.

A residual's loss can be measured in a norm other than the mean square over points. Negative Sobolev norms
(``H^-1``) are the Riesz-preconditioned residual losses of the variational PINN literature: for a second-order
operator the strong L2 residual loss has curvature ~|k|^4 per Fourier mode, the H^-1 norm ~|k|^2 (SV-PINNs,
Marcondes 2026, arXiv:2605.03542, Sec. 3-4; the Deep Fourier Residual method, Taylor et al. 2022,
arXiv:2210.14129; RVPINNs, Rojas et al. 2024). The norm is ``||r||_s^2 = <r, (scale - Laplace)^s r>``.

Two backends, both built once per (domain, region, space, options) and cached on the domain:

- :class:`GridSobolev`, a structured lattice (``mesh_connectivity["grid"]``): the per-axis eigenvectors of the
  discrete Laplacian (sine for Dirichlet, cosine for Neumann) as dense transforms, so any real order ``s``.
  Dense per-axis matrices cost O(n^(d+1)) per evaluation; an FFT path for large grids is not implemented.
- :class:`MeshSobolev`, a simplex mesh in 1D/2D/3D: P1 mass ``M`` and stiffness ``K`` assembled on the host
  once. ``H^1``: ``r^T (scale M + K) r`` of the P1 interpolant. ``H^-1``: ``R^T (scale M + K)^-1 R`` with
  ``R = M r`` the residual tested against the hat functions, one linear solve per evaluation (default CG
  with Jacobi; any ``jno.solve`` solver and ``jno.precond`` preconditioner, the latter built once from the
  constant matrix; differentiable through ``lax.custom_linear_solve``). ``bc="dirichlet"`` drops the boundary nodes from the test space
  (the dual of ``H^1_0``), ``bc="neumann"`` keeps them (the dual of ``H^1``).

``M`` and ``K`` come from the fixed mesh, so the norm is differentiable in the field values, not in the mesh
coordinates.
"""

from __future__ import annotations

import math
import re

import jax
import jax.numpy as jnp
import numpy as np

KNOWN_SPACES = "'l2' (root mean square over the points), 'L2', 'L1', 'Linf', 'H1', 'H-1', 'H^s' (s real, e.g. 'H^-0.5')"
_BCS = ("dirichlet", "neumann")
_METHODS = (None, "grid", "mesh")
_SIMPLEX_KEY = {1: "lines", 2: "triangles", 3: "tetrahedra"}


def parse_space(space: str):
    """``(kind, s)`` for a space name: kind is ``"l2" | "L2" | "L1" | "Linf" | "H"``, ``s`` the Sobolev order."""
    if space in ("l2", "L2", "L1", "Linf"):
        return space, None
    m = re.fullmatch(r"H\^?([+-]?\d+(?:\.\d*)?)", str(space))
    if m is None:
        raise ValueError(f"norm: unknown space {space!r}. Known: {KNOWN_SPACES}.")
    s = float(m.group(1))
    return ("L2", None) if s == 0 else ("H", s)


def check_options(kind, s, bc, scale, method):
    """Refuse what this first cut does not do, naming the way out (rule: never fail silently)."""
    if method in ("points", "sample"):
        raise NotImplementedError(
            f"norm(method={method!r}) is planned, not implemented: the point-cloud (Green's kernel) and the "
            "sampled (SV-PINN Monte-Carlo) estimators. Use a mesh or a structured grid (method=None picks one)."
        )
    if method not in _METHODS:
        raise ValueError(f"norm: method must be one of None, 'grid', 'mesh'; got {method!r}.")
    if bc == "periodic":
        raise NotImplementedError("norm(bc='periodic') is planned, not implemented. Use 'dirichlet' or 'neumann'.")
    if bc not in _BCS:
        raise ValueError(f"norm: bc must be 'dirichlet' or 'neumann'; got {bc!r}.")
    if scale < 0:
        raise ValueError(f"norm: scale must be >= 0 (the 'scale' in (scale - Laplace)^s); got {scale}.")
    if kind == "H" and s < 0 and scale == 0 and bc == "neumann":
        raise ValueError(
            "norm: a negative order with scale=0 and bc='neumann' is singular (constants have zero energy). "
            "Use scale > 0, or bc='dirichlet'."
        )


def _components(vals):
    """``(N, C)``: one row per point, the field's components (vector or matrix entries) flattened."""
    return vals.reshape(vals.shape[0], -1)


def pointwise_sq(vals):
    """|r(x)|^2 per point, summed over the components (the Frobenius / Euclidean fibre norm)."""
    return jnp.sum(jnp.square(_components(vals)), axis=-1)


# ---- structured lattice ----------------------------------------------------------------------------------------


class GridSobolev:
    """``sum_k (scale + lambda_k)^s c_k^2 ||v_k||^2`` with ``v_k`` the tensor-product eigenvectors of the
    discrete Laplacian on the lattice, ``c_k`` the coefficients of the field in the trapezoid inner product."""

    name = "grid"

    def __init__(self, grid: dict, points: np.ndarray, s: float, bc: str, scale: float):
        shape = tuple(int(n) for n in grid["shape"])
        spacing = tuple(float(h) for h in grid["spacing"])
        origin = tuple(float(o) for o in grid["origin"])
        d = len(shape)
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, d)
        expect = np.asarray(origin) + idx * np.asarray(spacing)
        pts = np.asarray(points)[:, :d]
        if pts.shape != expect.shape or not np.allclose(pts, expect, rtol=0, atol=1e-9 * max(spacing)):
            raise ValueError(
                "norm: the lattice's nodes are not in grid order (idx = ((i Ny + j) Nz + k)), so the grid backend "
                "cannot read them. Use method='mesh'."
            )
        self.shape = shape
        P, lam, nv = [], [], []
        for n_nodes, h in zip(shape, spacing):
            n = n_nodes - 1
            i = np.arange(n_nodes)
            k = np.arange(1, n) if bc == "dirichlet" else np.arange(0, n + 1)
            V = (np.sin if bc == "dirichlet" else np.cos)(np.pi * np.outer(k, i) / n)
            w = np.full(n_nodes, h)
            w[[0, -1]] = h / 2  # the trapezoid rule; for Dirichlet the sine is zero there anyway
            norms = (V**2) @ w
            P.append(jnp.asarray(V * w / norms[:, None]))
            lam.append(4.0 / h**2 * np.sin(np.pi * k / (2 * n)) ** 2)
            nv.append(norms)
        lam_sum = sum(np.meshgrid(*lam, indexing="ij"))
        nv_prod = math.prod(np.meshgrid(*nv, indexing="ij"))
        base = scale + lam_sum
        if s < 0 and np.any(base == 0):
            raise ValueError("norm: (scale + lambda)^s is infinite for the constant mode. Use scale > 0.")
        self.P = P
        self.weight = jnp.asarray(base**s * nv_prod)

    def __call__(self, vals):
        c = _components(vals)
        n_comp = c.shape[-1]
        c = c.reshape(*self.shape, n_comp)
        for axis, P in enumerate(self.P):  # coefficients, one axis at a time
            c = jnp.moveaxis(jnp.tensordot(P, c, axes=([1], [axis])), 0, axis)
        return jnp.sum(self.weight[..., None] * c**2)


# ---- simplex mesh, P1 ------------------------------------------------------------------------------------------


def p1_mass_stiffness(points: np.ndarray, cells: np.ndarray):
    """Consistent P1 mass and stiffness matrices (scipy CSR) on simplices of any dimension, vectorised."""
    import scipy.sparse as sp

    d = cells.shape[1] - 1
    X = points[cells][:, :, :d]  # (E, d+1, d)
    D = X[:, 1:, :] - X[:, :1, :]  # edge vectors as rows
    vol = np.abs(np.linalg.det(D)) / math.factorial(d)
    G = np.linalg.inv(D).transpose(0, 2, 1)  # rows: gradients of the barycentric coordinates 1..d
    G = np.concatenate([-G.sum(axis=1, keepdims=True), G], axis=1)  # (E, d+1, d)
    Ke = vol[:, None, None] * np.einsum("eid,ejd->eij", G, G)
    Me = vol[:, None, None] / ((d + 1) * (d + 2)) * (np.ones((d + 1, d + 1)) + np.eye(d + 1))
    rows = np.repeat(cells, d + 1, axis=1).ravel()
    cols = np.tile(cells, (1, d + 1)).ravel()
    n = points.shape[0]
    M = sp.coo_matrix((Me.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    return M, K


class MeshSobolev:
    """``H^1`` and ``H^-1`` on a simplex mesh with P1 hat functions (see the module docstring)."""

    name = "mesh"

    def __init__(self, mc: dict, s: float, bc: str, scale: float, solver=None, precond=None):
        from jax.experimental import sparse

        if s not in (-1.0, 1.0):
            raise NotImplementedError(
                f"norm: order s={s:g} on a mesh is not implemented (only H1 and H-1). Fractional orders need a "
                "structured grid: jno.shape.rect(...).structured(n=...)."
            )
        dim = int(mc["dimension"])
        cells = mc.get(_SIMPLEX_KEY.get(dim, ""))
        if cells is None or len(cells) == 0:
            raise NotImplementedError(
                f"norm: the H^s mesh backend needs a simplex mesh ({_SIMPLEX_KEY.get(dim)}); this {dim}-D mesh has "
                "none (a quad/hex mesh?). Use a triangle/tetrahedron mesh, or a structured lattice."
            )
        points = np.asarray(mc["points"], dtype=np.float64)
        M, K = p1_mass_stiffness(points, np.asarray(cells, dtype=np.int64))
        A = scale * M + K
        self.s = s
        if s > 0:
            self.A = sparse.BCOO.from_scipy_sparse(A)
            return
        free = np.ones(points.shape[0], dtype=bool)
        if bc == "dirichlet":
            free[np.asarray(mc["boundary_indices"], dtype=np.int64)] = False
        self.free = jnp.asarray(np.flatnonzero(free))
        self.M = sparse.BCOO.from_scipy_sparse(M)
        self.A = sparse.BCOO.from_scipy_sparse(A[free][:, free].tocsr())
        import jno

        self.solver = jno.solve.cg(tol=1e-10) if solver is None else solver
        if getattr(self.solver, "direct", False):
            self.precond = None
        else:
            self.precond = _default_precond() if precond is None else precond
            if hasattr(self.precond, "build"):  # host setup (AMG hierarchy, FSAI pattern) once, eagerly
                self.precond.build(self.A)

    def __call__(self, vals):
        c = _components(vals)
        total = 0.0
        for j in range(c.shape[-1]):
            r = c[:, j]
            if self.s > 0:
                total = total + r @ (self.A @ r)
            else:
                R = (self.M @ r)[self.free]
                z = self.solver(self.A, R) if self.precond is None else self.solver(self.A, R, M=self.precond)
                total = total + R @ z
        return total


def _default_precond():
    """AMG where pyamg is installed (an optional dependency), else Jacobi, said in the log.

    Measured per training step on a 4x128 MLP, H^-1 on a 2-D mesh of ~5k nodes (RTX 3070, shared): AMG-CG
    1.1-1.4x the cost of ``r.mse``; Jacobi-CG and FSAI-CG between 1.2x and 10x. On the CPU all three are within
    20 % of ``r.mse``, the network dominates there."""
    import importlib.util
    import logging

    import jno

    if importlib.util.find_spec("pyamg") is not None:
        return jno.precond.amg()
    logging.getLogger("jno").info(
        "norm('H-1') on a mesh: pyamg is not installed, so the solve uses Jacobi-CG; install pyamg for the AMG "
        "preconditioner (up to 10x faster per step on a GPU), or pass precond=."
    )
    return jno.precond.jacobi()


def sobolev_operator(domain, tag: str, mc: dict, backend: str, s: float, bc: str, scale: float, solver=None, precond=None):
    """The cached backend for one (region, space, options)."""
    if not hasattr(domain, "_norm_operator_cache"):
        domain._norm_operator_cache = {}
    key = (tag, backend, s, bc, float(scale), id(solver), id(precond))
    cache = domain._norm_operator_cache
    if key not in cache:
        # Built while a step is being traced, but kept across traces (eval, then solve, recompiles): the
        # arrays must be concrete, or the next trace reads a leaked tracer from the first one.
        with jax.ensure_compile_time_eval():
            if backend == "grid":
                cache[key] = GridSobolev(mc["grid"], mc["points"], s, bc, scale)
            else:
                cache[key] = MeshSobolev(mc, s, bc, scale, solver, precond)
    return cache[key]
