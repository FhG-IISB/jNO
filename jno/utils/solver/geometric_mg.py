"""Geometric multigrid (GMG) V-cycle for a **structured grid** — a matrix-free, differentiable
preconditioner for the constant-coefficient Poisson/Helmholtz-type operators ``jno.fdm`` produces on a
regular grid (see :func:`jno.shape.rect(...).structured().domain()`).

The V-cycle approximates ``A⁻¹`` for ``A = -Δ`` with homogeneous Dirichlet on the interior:

  smooth (damped Jacobi) → restrict residual (full-weighting) → recurse on the 2×-coarser grid →
  prolong the correction (multilinear interpolation) → smooth again,

with the coarse operators **rediscretised** on each grid (exact for the constant-coefficient Laplacian;
a Galerkin ``RAP`` coarse operator, needed for variable coefficients, is future work). Boundary DOFs are
passed through (identity rows), so it preconditions the reduced-Dirichlet system ``jno.fdm`` assembles.

Everything is roll/tensordot stencils on the reshaped grid — ``jit``-friendly and reverse-mode
differentiable (the coarsest level is a small dense solve). Isotropic coarsening: every axis halves
together, and the hierarchy stops when any axis can no longer halve (odd cell count or ``min_size``),
so a grid too small to coarsen yields a single level (the caller falls back to an un-preconditioned solve).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _prolong_matrix(nc: int) -> np.ndarray:
    """1-D linear-interpolation prolongation ``(nf, nc)``, fine ``nf = 2·(nc−1)+1`` nodes: coarse nodes
    inject to the even fine nodes, odd fine nodes are the average of their two coarse neighbours."""
    nf = 2 * (nc - 1) + 1
    P = np.zeros((nf, nc))
    for i in range(nc):
        P[2 * i, i] = 1.0
    for i in range(nc - 1):
        P[2 * i + 1, i] = 0.5
        P[2 * i + 1, i + 1] = 0.5
    return P


def _hierarchy(shape, spacing, min_size):
    """Fine→coarse ``[(shape, spacing), …]``. Halve **every** axis together while each axis has an even
    cell count ``(n−1) % 2 == 0`` and stays above ``min_size``; stop at the first axis that can't."""
    levels = [(tuple(int(n) for n in shape), tuple(float(h) for h in spacing))]
    while True:
        sh, sp = levels[-1]
        if all((n - 1) % 2 == 0 and n > min_size for n in sh):
            levels.append((tuple((n - 1) // 2 + 1 for n in sh), tuple(h * 2.0 for h in sp)))
        else:
            return levels


def _interior_mask(shape) -> jnp.ndarray:
    """1.0 on interior grid nodes, 0.0 on the boundary (any axis at index 0 or n−1)."""
    m = np.ones(shape)
    for ax, n in enumerate(shape):
        sl = [slice(None)] * len(shape)
        sl[ax] = 0
        m[tuple(sl)] = 0.0
        sl[ax] = n - 1
        m[tuple(sl)] = 0.0
    return jnp.asarray(m)


def _apply_axis(M: jnp.ndarray, X: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Apply the 1-D operator ``M`` (out×in) along ``axis`` of the tensor ``X`` (``X.shape[axis] == in``)."""
    Y = jnp.tensordot(M, X, axes=([1], [axis]))  # (out, …X without `axis`…)
    return jnp.moveaxis(Y, 0, axis)


def _restrict_axis(X: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Full-weighting restriction ``½Pᵀ`` along ``axis`` as a stencil: coarse node ``i`` takes
    ``½·x[2i] + ¼·(x[2i−1] + x[2i+1])``. Identical to ``_apply_axis(0.5 * P.T, X, axis)`` with ``P`` from
    :func:`_prolong_matrix`, but O(N) instead of a dense ``(n_c × n_f)`` product along the axis."""
    X = jnp.moveaxis(X, axis, 0)
    even, odd = X[0::2], X[1::2]  # n_c and n_c − 1 entries
    pad = [(0, 0)] * (X.ndim - 1)
    out = 0.5 * even + 0.25 * (jnp.pad(odd, [(0, 1)] + pad) + jnp.pad(odd, [(1, 0)] + pad))
    return jnp.moveaxis(out, 0, axis)


def _prolong_axis(X: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Linear-interpolation prolongation ``P`` along ``axis`` as a stencil: fine node ``2i`` is coarse
    node ``i`` and fine node ``2i+1`` is the mean of coarse nodes ``i`` and ``i+1``. Identical to
    ``_apply_axis(P, X, axis)``, O(N)."""
    X = jnp.moveaxis(X, axis, 0)
    mids = 0.5 * (X[:-1] + X[1:])
    pairs = jnp.stack([X[:-1], mids], axis=1).reshape((2 * (X.shape[0] - 1),) + X.shape[1:])
    return jnp.moveaxis(jnp.concatenate([pairs, X[-1:]], axis=0), 0, axis)


def _neg_laplacian(u_grid: jnp.ndarray, spacing, interior: jnp.ndarray) -> jnp.ndarray:
    """``(-Δu)`` via the 5-/7-point stencil, zeroed on the boundary (homogeneous-Dirichlet rows).

    One pass over the interior block, then a single zero pad for the boundary ring, which the mask
    zeroes anyway. It used to be written with ``jnp.roll``, whose wrap-around only ever reached the masked
    ring, so the result is bit-identical; but XLA fuses slices, and on CPU the rolls' concatenations were
    not fused: 1.64 ms against 0.11 ms for a plain copy of the same 1M-node grid."""
    u = u_grid * interior
    dim = u.ndim
    core = tuple(slice(1, -1) for _ in range(dim))
    centre = u[core]
    lap = jnp.zeros_like(centre)
    for ax, h in enumerate(spacing):
        plus = tuple(slice(2, None) if a == ax else slice(1, -1) for a in range(dim))
        minus = tuple(slice(None, -2) if a == ax else slice(1, -1) for a in range(dim))
        lap = lap + (u[plus] + u[minus] - 2.0 * centre) / (h * h)
    return jnp.pad(-lap, 1) * interior


def build_vcycle(shape, spacing, *, n_pre: int = 2, n_post: int = 2, omega: float | None = None, min_size: int = 5):
    """Build a one-V-cycle applier ``M⁻¹: r_flat → e_flat`` for ``-Δ`` (homogeneous Dirichlet interior) on
    the structured grid ``(shape, spacing)``. Returns ``(apply, n_levels)``; ``n_levels == 1`` means the
    grid can't be coarsened (the caller should skip GMG). Damped-Jacobi smoothing (``omega`` defaults to
    the model-problem optimum ``2d/(2d+1)``), full-weighting restriction ``½ᵈ Pᵀ``, rediscretised coarse
    operators, dense solve at the coarsest level."""
    dim = len(shape)
    if omega is None:
        omega = 2.0 * dim / (2.0 * dim + 1.0)  # 2/3 (1-D), 4/5 (2-D), 6/7 (3-D)
    levels = _hierarchy(shape, spacing, min_size)

    per = []  # per-level: (shape, spacing, interior, inv_diag, [P_axis], [R_axis])
    for lev, (sh, sp) in enumerate(levels):
        interior = _interior_mask(sh)
        diag = 2.0 * sum(1.0 / (h * h) for h in sp)  # diag of -Δ
        Ps = Rs = None
        if lev + 1 < len(levels):
            csh = levels[lev + 1][0]
            Ps = [jnp.asarray(_prolong_matrix(csh[ax])) for ax in range(dim)]
            Rs = [0.5 * P.T for P in Ps]  # full-weighting = ½ Pᵀ per axis → ½ᵈ Pᵀ tensor product
        per.append((sh, sp, interior, 1.0 / diag, Ps, Rs))

    # Coarsest level: dense interior solve. Assemble A = -Δ (homogeneous Dirichlet) by jacfwd on a unit
    # basis, then solve the interior sub-block directly (small; differentiable).
    csh, csp, cint, _, _, _ = per[-1]
    n_c = int(np.prod(csh))
    int_flat = np.asarray(cint).reshape(-1) > 0.5
    int_idx = jnp.asarray(np.nonzero(int_flat)[0])

    def _coarse_matvec(u_flat):
        return _neg_laplacian(u_flat.reshape(csh), csp, cint).reshape(-1)

    A_coarse = jax.jacfwd(_coarse_matvec)(jnp.zeros(n_c))
    A_int = A_coarse[jnp.ix_(int_idx, int_idx)]

    def _coarse_solve(r_grid):
        r_int = r_grid.reshape(-1)[int_idx]
        e_int = jnp.linalg.solve(A_int, r_int)
        return jnp.zeros(n_c).at[int_idx].set(e_int).reshape(csh)

    def _smooth(u, r, sp, interior, inv_diag, n):
        for _ in range(n):
            u = u + omega * inv_diag * (r - _neg_laplacian(u, sp, interior)) * interior
        return u

    # The transfers used to be dense 1-D matrices applied with `tensordot`: a (n_c × n_f) product along
    # each axis, O(n³) in 2-D rather than O(n²). On a 2049² grid that was ~17 GFLOP per V-cycle against
    # ~40 MFLOP for the smoothing, and a solve cost ~4000 residual evaluations instead of ~100.
    def _restrict(r_grid, Rs):
        out = r_grid
        for ax in range(len(Rs)):
            out = _restrict_axis(out, ax)
        return out

    def _prolong(e_grid, Ps):
        out = e_grid
        for ax in range(len(Ps)):
            out = _prolong_axis(out, ax)
        return out

    def _vcycle(r_grid, lev):
        sh, sp, interior, inv_diag, Ps, Rs = per[lev]
        if lev == len(levels) - 1:
            return _coarse_solve(r_grid)
        e = _smooth(jnp.zeros(sh), r_grid, sp, interior, inv_diag, n_pre)
        resid = (r_grid - _neg_laplacian(e, sp, interior)) * interior
        e = e + _prolong(_vcycle(_restrict(resid, Rs), lev + 1), Ps)
        e = _smooth(e, r_grid, sp, interior, inv_diag, n_post)
        return e * interior

    def apply(r_flat):
        r = jnp.asarray(r_flat)
        # Identity boundary rows (Dirichlet): pass r through there; V-cycle solves the interior.
        e_int = _vcycle((r.reshape(shape)) * per[0][2], 0)
        return jnp.where(per[0][2].reshape(-1) > 0.5, e_int.reshape(-1), r)

    # Jitted so the unrolled recursion (levels x smoothing sweeps, all Python) is traced once per shape:
    # a Krylov solve traces its preconditioner for the forward and the transpose solve, and re-tracing
    # it every time was most of a structured FDM solve's 8 s.
    return jax.jit(apply), len(levels)
