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
    """Concrete BCOO / dense -> scipy CSR for the pyamg setup."""
    import scipy.sparse as sp

    if hasattr(A, "todense") and hasattr(A, "indices"):  # BCOO
        data = np.asarray(A.data)
        idx = np.asarray(A.indices)
        return sp.coo_matrix((data, (idx[:, 0], idx[:, 1])), shape=A.shape).tocsr()
    return sp.csr_matrix(np.asarray(A))


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

    ml = pyamg.smoothed_aggregation_solver(_to_scipy_csr(A), max_levels=max_levels, max_coarse=coarse_size)
    levels: List[dict] = []
    for lvl in ml.levels[:-1]:
        A_l = jsp.BCOO.from_scipy_sparse(lvl.A.tocoo())
        P = jsp.BCOO.from_scipy_sparse(lvl.P.tocoo())
        R = jsp.BCOO.from_scipy_sparse(lvl.R.tocoo())
        lmax = float(
            safety * power_iteration_bound(lambda v: A_l @ v, A_l.shape[0], dtype=A_l.data.dtype, iters=bound_iters)
        )
        levels.append({"A": A_l, "P": P, "R": R, "lmin": lmin_ratio * lmax, "lmax": lmax, "degree": smoother_degree})
    A_c = np.asarray(ml.levels[-1].A.todense())
    levels.append({"Ainv": jnp.asarray(np.linalg.pinv(A_c))})  # pinv: robust to a gauge null space
    return levels


def vcycle_apply(levels: List[dict], r):
    """One V-cycle on residual ``r`` from a zero initial guess — pure JAX, and a fixed **linear**
    map in ``r`` (all constituents are linear), so it may precondition CG/MINRES."""

    def _cycle(i, r):
        lv = levels[i]
        if "Ainv" in lv:
            return lv["Ainv"] @ r
        A = lv["A"]
        smooth = lambda rhs: chebyshev_apply(lambda v: A @ v, rhs, lmin=lv["lmin"], lmax=lv["lmax"], degree=lv["degree"])
        x = smooth(r)  # pre-smooth (from zero)
        x = x + (lv["P"] @ _cycle(i + 1, lv["R"] @ (r - A @ x)))  # coarse-grid correction
        return x + smooth(r - A @ x)  # post-smooth

    return _cycle(0, jnp.asarray(r).reshape(-1))
