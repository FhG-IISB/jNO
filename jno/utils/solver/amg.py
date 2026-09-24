"""Hybrid algebraic multigrid: **pyamg setup on the host, pure-JAX V-cycle apply**.

The split that makes AMG compatible with ``jit``/``vmap``/AD: the *setup* (strength graphs,
aggregation, building the prolongators) is dynamic-sparsity graph work — hopeless to trace — so
it runs once, eagerly, through pyamg's smoothed-aggregation solver [2]. What it produces is a
frozen hierarchy of **fixed-pattern** sparse operators (`A_l`, `P_l`, `R_l`) plus a dense
coarse-grid inverse. The *cycle* is then nothing but SpMVs, Chebyshev polynomial smoothing [3],
and one small dense matmul — pure JAX, ``jit``- and ``vmap``-native (the hierarchy is closure
data: a shared preconditioner across a batch is legitimate because a preconditioner only affects
convergence speed, never the converged solution), and needs no gradient path (the
``custom_linear_solve`` firewall).

The frozen hierarchy is built from one *representative* concrete matrix. Reusing it while the
operator values change (Picard iterations, parameter updates during an inverse solve) is the
standard frozen-preconditioner trade: convergence degrades gracefully with the distance from the
setup matrix, correctness never does. Rebuild (``spec.build(A)``) when it drifts too far.

pyamg is an **optional** dependency (lazily imported here only); everything else is jNO + JAX.

References
----------
[1] P. Vaněk, J. Mandel, M. Brezina, *Algebraic Multigrid by Smoothed Aggregation for Second and
    Fourth Order Elliptic Problems*, Computing 56, 1996 — the SA-AMG setup this delegates to.
[2] N. Bell, L. N. Olson, J. Schroder, B. Zaman, *PyAMG: Algebraic Multigrid Solvers in Python*,
    J. Open Source Software 8(87):5495, 2023.
[3] M. Adams, M. Brezina, J. Hu, R. Tuminaro, *Parallel Multigrid Smoothing: Polynomial versus
    Gauss-Seidel*, J. Comput. Phys. 188, 2003 — polynomial (Chebyshev) smoothing as the
    parallel/GPU-friendly substitute for Gauss-Seidel; smoothing window ``[lmax/30, 1.1 lmax]``
    per pyamg's convention.
"""

from __future__ import annotations

from typing import Any, List

import jax
import jax.numpy as jnp
import numpy as np

from .krylov import chebyshev_apply, power_iteration_bound

__all__ = ["build_hierarchy", "vcycle_apply"]


def _require_pyamg():
    try:
        import pyamg  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - exercised only without the optional dep
        raise ImportError(
            "jno.precond.amg needs the optional dependency `pyamg` for its host-side setup "
            "(the apply is pure JAX). Install it with `pip install pyamg` (or use the pixi "
            "`fem`/`dev` environment, which includes it)."
        ) from e
    return pyamg


def _to_scipy_csr(A):
    """Concrete scipy sparse / BCOO / dense -> scipy CSR for the pyamg setup."""
    import scipy.sparse as sp

    # scipy FIRST: a scipy CSR also has `.todense` and `.indices` (its 1-D column array), so the
    # duck-typed BCOO test below matches it and then `idx[:, 0]` raises "too many indices". Already
    # being the target type is not a case to fall through on.
    if sp.issparse(A):
        return A.tocsr()
    if hasattr(A, "todense") and hasattr(A, "indices"):  # BCOO: 2-D (row, col) index array
        data = np.asarray(A.data)
        idx = np.asarray(A.indices)
        return sp.coo_matrix((data, (idx[:, 0], idx[:, 1])), shape=A.shape).tocsr()
    return sp.csr_matrix(np.asarray(A))


_tiny = 1e-300


def _smoother_lmax(A_host, A_dev, *, safety: float, iters: int, degree: int, lmin_ratio: float) -> float:
    """Upper bound on a level's spectrum for the Chebyshev smoother -- verified, never assumed.

    Chebyshev damps the band ``[lmin, lmax]`` and AMPLIFIES whatever lies above ``lmax``, so a bound
    that is too small does not merely smooth poorly, it makes the V-cycle divergent. Power iteration
    returns a Rayleigh quotient, which approaches the dominant eigenvalue **from below**, and on a
    nonsymmetric operator it can STALL far short of it: measured on the Newton tangent of
    ``(1 + u^2) grad u . grad phi`` (4751 dofs) it sat at ``0.54 * rho`` at 20 iterations and again at
    40 -- so "two runs agree" reads as converged there, and is worthless as a check -- while 200
    iterations were needed to reach ``rho``. One V-cycle then amplified a random residual **7.5x**,
    surfacing only as a stalled Krylov solve several frames away. The same problem's LINEAR step
    operator estimates to ``1.007 * rho``, which is why this went unnoticed.

    So the estimate is USED only if the smoother it implies is measured to damp; otherwise the level
    falls back to Gershgorin's disc theorem, ``rho(A) <= max_i sum_j |a_ij|``, which holds for any
    matrix and costs one pass over a matrix already assembled here. The probe is one Chebyshev apply.
    Measured: the broken case 7.5 -> 0.36, and the case that already worked keeps its tight bound and
    its 0.33 unchanged.
    """
    gershgorin = float(abs(A_host).sum(axis=1).max())
    tight = safety * float(power_iteration_bound(lambda v: A_dev @ v, A_dev.shape[0], dtype=A_dev.data.dtype, iters=iters))
    if not (0.0 < tight < gershgorin):  # already at or above the guaranteed bound: nothing to gain
        return gershgorin
    r = jax.random.normal(jax.random.PRNGKey(0), (A_dev.shape[0],), dtype=A_dev.data.dtype)
    x = chebyshev_apply(lambda v: A_dev @ v, r, lmin=lmin_ratio * tight, lmax=tight, degree=degree)
    damped = float(jnp.linalg.norm(r - A_dev @ x) / jnp.maximum(jnp.linalg.norm(r), _tiny))
    return tight if damped < 1.0 else gershgorin


def build_hierarchy(
    A: Any,
    *,
    max_levels: int = 10,
    coarse_size: int = 100,
    smoother_degree: int = 3,
    lmin_ratio: float = 1.0 / 30.0,
    safety: float = 1.1,
    bound_iters: int = 20,
) -> List[dict]:
    """One-time host-side setup: pyamg smoothed aggregation [1,2] -> frozen JAX level data.

    ``A`` must be **concrete** (this runs eagerly, never under a trace). Returns a list of level
    dicts — fine to coarse: ``{"A", "P", "R", "lmin", "lmax"}`` as BCOO/scalars, the last level
    ``{"Ainv"}`` a dense inverse — a valid JAX pytree, so a closure over it jits and vmaps.
    """
    import jax.experimental.sparse as jsp

    pyamg = _require_pyamg()

    data = getattr(A, "data", None)
    if isinstance(data, jax.core.Tracer) or isinstance(A, jax.core.Tracer):
        raise TypeError(
            "AMG setup needs a concrete matrix but got a traced one (inside jit/grad/vmap or a "
            "parametric solve). Pre-build eagerly from a representative operator: "
            "spec = jno.precond.amg(); spec.build(fem.A); then reuse the spec."
        )

    # Smoothed aggregation assumes a POSITIVE diagonal: its strength-of-connection graph and its
    # smoothers both do. On an operator with negative diagonal entries the aggregation degenerates and
    # the Galerkin coarse operator comes out NaN, or exactly zero -- and a zero coarse matrix does not
    # raise. `pinv` of it "succeeds", the V-cycle silently contributes nothing, and the outer Krylov
    # just converges slowly for no visible reason. Measured on the velocity block of a Navier-Stokes
    # tangent: 213 of 538 diagonal entries negative, coarse operator all zeros.
    _A_csr = _to_scipy_csr(A)
    _diag = np.asarray(_A_csr.diagonal())
    _neg = int((_diag < 0).sum())
    if _neg:
        raise ValueError(
            f"jno.precond.amg(): this operator has {_neg} negative diagonal entries out of {_diag.size}. "
            "Smoothed aggregation assumes a positive diagonal, and on an indefinite operator it builds a "
            "degenerate hierarchy (a NaN or all-zero coarse grid) that fails silently rather than loudly. "
            "This is the usual shape of a SADDLE-POINT or Newton-tangent block -- precondition the "
            "structure you have instead: jno.precond.saddle() / jno.precond.lsc() for the saddle system, "
            "or jno.precond.jacobi() / inner(...) on the block."
        )
    ml = pyamg.smoothed_aggregation_solver(_A_csr, max_levels=max_levels, max_coarse=coarse_size)
    levels: List[dict] = []
    for lvl in ml.levels[:-1]:
        A_l = jsp.BCOO.from_scipy_sparse(lvl.A.tocoo())
        P = jsp.BCOO.from_scipy_sparse(lvl.P.tocoo())
        R = jsp.BCOO.from_scipy_sparse(lvl.R.tocoo())
        lmax = _smoother_lmax(lvl.A, A_l, safety=safety, iters=bound_iters, degree=smoother_degree, lmin_ratio=lmin_ratio)
        levels.append({"A": A_l, "P": P, "R": R, "lmin": lmin_ratio * lmax, "lmax": lmax, "degree": smoother_degree})
    # Each level's A / P / R in the storage MEASURED fastest for it on this device (CSR or split COO,
    # see jno.utils.solver.matvec_format), converted here once. A V-cycle is ~10 fine-level products
    # (Chebyshev pre + post + residuals): measured 65% of its time on a 3-D P1 operator, and CSR levels
    # made a 303k-DOF 3-D AMG-PCG solve 1.20x faster, a 142k-DOF P2 one 1.44x.
    from . import matvec_format as _mf

    formats = []
    for lv in levels:
        lv["fast"] = {k: _mf.prepare(lv[k], log=False) for k in ("A", "P", "R")}
        formats.append("/".join(next(iter(lv["fast"][k])) for k in ("A", "P", "R")))
    from ..logger import get_logger

    get_logger().info(
        f"jno AMG hierarchy: {len(levels)} levels + coarse solve; level storage (A/P/R, measured on each level "
        f"matrix): {', '.join(formats)}"
    )
    A_c = np.asarray(ml.levels[-1].A.todense())
    if not np.isfinite(A_c).all() or not np.abs(A_c).max() > 0:
        raise ValueError(
            "jno.precond.amg(): the coarsest-grid operator came out "
            f"{'non-finite' if not np.isfinite(A_c).all() else 'identically zero'} ({A_c.shape}), so the "
            "coarse-grid correction would contribute nothing. The aggregation did not find usable "
            "structure in this operator -- see the diagonal check above for the usual cause."
        )
    levels.append({"Ainv": jnp.asarray(np.linalg.pinv(A_c))})  # pinv: robust to a gauge null space
    return levels


def vcycle_apply(levels: List[dict], r):
    """One V-cycle on residual ``r`` from a zero initial guess — pure JAX, and a fixed **linear**
    map in ``r`` (all constituents are linear), so it may precondition CG/MINRES."""

    def _cycle(i, r):
        lv = levels[i]
        if "Ainv" in lv:
            return lv["Ainv"] @ r
        if "fast" in lv:  # the storage measured fastest for this level (built by `build_hierarchy`)
            from .matvec_format import apply_prepared

            fast = lv["fast"]
            mv_A = lambda v: apply_prepared(fast["A"], lv["A"].shape, v)  # noqa: E731
            mv_P = lambda v: apply_prepared(fast["P"], lv["P"].shape, v)  # noqa: E731
            mv_R = lambda v: apply_prepared(fast["R"], lv["R"].shape, v)  # noqa: E731
        else:
            mv_A, mv_P, mv_R = (lambda v: lv["A"] @ v), (lambda v: lv["P"] @ v), (lambda v: lv["R"] @ v)
        smooth = lambda rhs: chebyshev_apply(mv_A, rhs, lmin=lv["lmin"], lmax=lv["lmax"], degree=lv["degree"])
        x = smooth(r)  # pre-smooth (from zero)
        x = x + mv_P(_cycle(i + 1, mv_R(r - mv_A(x))))  # coarse-grid correction
        return x + smooth(r - mv_A(x))  # post-smooth

    return _cycle(0, jnp.asarray(r).reshape(-1))
