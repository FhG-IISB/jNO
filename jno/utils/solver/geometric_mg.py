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
differentiable (the coarsest level is factorised once when small, Chebyshev iteration when not). Isotropic coarsening: every axis halves
together, and the hierarchy stops when any axis can no longer halve (odd cell count or ``min_size``),
so a grid too small to coarsen yields a single level (the caller falls back to an un-preconditioned solve).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg
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


#: Largest coarsest-level interior the V-cycle factorises densely. The factor is baked into the compiled
#: V-cycle as a constant, so it is kept small: 1024² f64 is 8 MB. Above it the coarse level is solved
#: matrix-free by Chebyshev iteration -- see :func:`_build_vcycle`.
DENSE_COARSE_MAX = 1024
#: Error reduction the Chebyshev coarse solve guarantees on the coarse operator's whole spectrum.
COARSE_REDUCTION = 1e-2


def _laplacian_bounds(shape, spacing, scale, shift):
    """Exact extreme eigenvalues of ``scale·(−Δ_h) + shift`` with homogeneous Dirichlet on the grid: the
    1-D Dirichlet second difference on ``n`` nodes has eigenvalues ``(4/h²) sin²(kπ / 2(n−1))``, k = 1..n−2,
    and the stencil is their Kronecker sum."""
    lo = sum(4.0 / h**2 * np.sin(np.pi / (2 * (n - 1))) ** 2 for n, h in zip(shape, spacing))
    hi = sum(4.0 / h**2 * np.sin((n - 2) * np.pi / (2 * (n - 1))) ** 2 for n, h in zip(shape, spacing))
    return scale * lo + shift, scale * hi + shift


def _chebyshev_degree(lo, hi, reduction):
    """Steps for Chebyshev iteration on ``[lo, hi]`` to cut the error by ``reduction`` (Saad, *Iterative
    Methods for Sparse Linear Systems*, 2nd ed., §12.3): ``2·ρᵏ <= reduction`` with ``ρ = (√κ−1)/(√κ+1)``."""
    sk = np.sqrt(hi / lo)
    return max(1, int(np.ceil(np.log(2.0 / reduction) / np.log((sk + 1.0) / (sk - 1.0)))))


def build_vcycle(shape, spacing, **kwargs):
    """Build a one-V-cycle applier; see :func:`_build_vcycle`. The setup depends only on the grid, so it
    runs on concrete values even when called inside a trace (a crux-driven inverse solve used to hit a
    traced coarse matrix here)."""
    with jax.ensure_compile_time_eval():
        return _build_vcycle(shape, spacing, **kwargs)


def _build_vcycle(
    shape,
    spacing,
    *,
    n_pre: int = 2,
    n_post: int = 2,
    omega: float | None = None,
    min_size: int = 5,
    scale: float = 1.0,
    shift: float = 0.0,
):
    """Build a one-V-cycle applier ``M⁻¹: r_flat → e_flat`` for ``scale·(−Δ) + shift·I`` (homogeneous
    Dirichlet interior) on the structured grid ``(shape, spacing)``. Returns ``(apply, n_levels)``;
    ``n_levels == 1`` means the grid can't be coarsened (the caller should skip GMG). Damped-Jacobi
    smoothing (``omega`` defaults to the model-problem optimum ``2d/(2d+1)``), full-weighting restriction
    ``½ᵈ Pᵀ``, rediscretised coarse operators (the shift is the same on every level). The shift is what a
    time step adds: ``I + θΔt(−Δ)`` is ``scale = θΔt, shift = 1``.

    **The coarsest level.** Coarsening stops at the first axis with an odd cell count, so its size depends
    on the grid's arithmetic, not its size: 1024 cells a side coarsen to 5×5, 1000 to 126×126, 1200 to 76×76.
    It used to be assembled as a dense matrix and re-factorised on EVERY V-cycle, so the cost was set by
    that accident: on an 8 GB card a 1000² Poisson problem ran out of memory building a 2 GB coarse matrix,
    and a 1200² one solved 74x slower than 1024² (3.4 s against 0.046 s). Now a coarse interior of at most
    :data:`DENSE_COARSE_MAX` unknowns is factorised ONCE (Cholesky; the operator is SPD) and back-substituted
    per cycle; a larger one is solved matrix-free by Chebyshev iteration on the exact spectral interval
    (:func:`_laplacian_bounds`) to :data:`COARSE_REDUCTION`. That polynomial is fixed, so the V-cycle stays a
    linear, symmetric preconditioner -- which CG requires -- and its memory stays O(N). The route taken is
    logged, with the fix: a cell count of the form m·2ᵏ, small m, coarsens all the way."""
    dim = len(shape)
    if omega is None:
        omega = 2.0 * dim / (2.0 * dim + 1.0)  # 2/3 (1-D), 4/5 (2-D), 6/7 (3-D)
    levels = _hierarchy(shape, spacing, min_size)

    per = []  # per-level: (shape, spacing, interior, inv_diag, [P_axis], [R_axis])
    for lev, (sh, sp) in enumerate(levels):
        interior = _interior_mask(sh)
        diag = scale * 2.0 * sum(1.0 / (h * h) for h in sp) + shift  # diag of scale·(−Δ) + shift
        # The transfers are stencils (`_restrict_axis`, `_prolong_axis`); the dense 1-D matrices they
        # replaced used to be built here all the same, unused, ~360 MB of device memory on a 4097² grid.
        per.append((sh, sp, interior, 1.0 / diag))

    # Coarsest level (see the docstring): factorised once if small, Chebyshev iteration if not.
    csh, csp, cint, _ = per[-1]
    n_c = int(np.prod(csh))
    int_flat = np.asarray(cint).reshape(-1) > 0.5
    int_idx = jnp.asarray(np.nonzero(int_flat)[0])
    n_int = int(int_flat.sum())

    def _op(u, sp, interior):  # scale·(−Δ) + shift on the interior, zero on the boundary ring
        return scale * _neg_laplacian(u, sp, interior) + shift * u * interior

    from ..logger import get_logger

    if n_int <= DENSE_COARSE_MAX:

        def _coarse_int(v_int):  # the operator restricted to the interior unknowns
            u = jnp.zeros(n_c).at[int_idx].set(v_int).reshape(csh)
            return _op(u, csp, cint).reshape(-1)[int_idx]

        chol = jax.scipy.linalg.cho_factor(jax.jacfwd(_coarse_int)(jnp.zeros(n_int)))

        def _coarse_solve(r_grid):
            e_int = jax.scipy.linalg.cho_solve(chol, r_grid.reshape(-1)[int_idx])
            return jnp.zeros(n_c).at[int_idx].set(e_int).reshape(csh)

    else:
        lo, hi = _laplacian_bounds(csh, csp, scale, shift)
        degree = _chebyshev_degree(lo, hi, COARSE_REDUCTION)
        theta, delta = 0.5 * (hi + lo), 0.5 * (hi - lo)
        sigma = theta / delta
        get_logger().info(
            f"GMG: coarsening stopped at {'x'.join(map(str, csh))} nodes (an axis with an odd cell count), so "
            f"its {n_int} coarse unknowns are solved by Chebyshev iteration, {degree} steps per V-cycle. A "
            "cell count per axis of the form m*2^k with small m coarsens further and solves faster."
        )

        def _coarse_solve(r_grid):
            # Chebyshev iteration (Saad, Alg. 12.1) from zero: a fixed polynomial in the operator, so linear.
            def body(_, st):
                x, r, d, rho = st
                x = x + d
                r = r - _op(d, csp, cint)
                rho_new = 1.0 / (2.0 * sigma - rho)
                return x, r, rho_new * rho * d + (2.0 * rho_new / delta) * r, rho_new

            r0 = r_grid * cint
            x, *_ = jax.lax.fori_loop(0, degree, body, (jnp.zeros(csh), r0, r0 / theta, 1.0 / sigma))
            return x * cint

    def _smooth(u, r, sp, interior, inv_diag, n):
        for _ in range(n):
            u = u + omega * inv_diag * (r - _op(u, sp, interior)) * interior
        return u

    # The transfers used to be dense 1-D matrices applied with `tensordot`: a (n_c × n_f) product along
    # each axis, O(n³) in 2-D rather than O(n²). On a 2049² grid that was ~17 GFLOP per V-cycle against
    # ~40 MFLOP for the smoothing, and a solve cost ~4000 residual evaluations instead of ~100.
    def _restrict(r_grid, dim):
        out = r_grid
        for ax in range(dim):
            out = _restrict_axis(out, ax)
        return out

    def _prolong(e_grid, dim):
        out = e_grid
        for ax in range(dim):
            out = _prolong_axis(out, ax)
        return out

    def _vcycle(r_grid, lev):
        sh, sp, interior, inv_diag = per[lev]
        if lev == len(levels) - 1:
            return _coarse_solve(r_grid)
        e = _smooth(jnp.zeros(sh), r_grid, sp, interior, inv_diag, n_pre)
        resid = (r_grid - _op(e, sp, interior)) * interior
        e = e + _prolong(_vcycle(_restrict(resid, dim), lev + 1), dim)
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
