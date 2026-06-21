#!/usr/bin/env python
"""How far is BCOO's matvec from a hand-written CUDA SpMV kernel?

The work inside CG is one sparse mat-vec per iteration, y = A @ v. This times
that single op two ways on the *same* Poisson stiffness matrix:

  * BCOO  -- jax.experimental.sparse, lowered by XLA to gather/segment-sum
  * cuSPARSE -- NVIDIA's hand-tuned CUDA SpMV kernel, via CuPy's CSR matvec

Same matrix, same vector, same timing protocol (chained power-iteration so XLA
can't constant-fold the repeats; one device sync per batch). The ratio is the
headroom a custom kernel could recover over BCOO.

Run:  uv run --no-sync python bench_spmv_kernels.py
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from shapely.geometry import box

import cupy as cp
import cupyx.scipy.sparse as cusp
import jno

MESH_SIZES = [0.02, 0.0123, 0.008, 0.0052]   # ~3k, 8k, 18k, 43k DOF
CHAIN = 50          # matvecs per timed batch (amortizes launch/dispatch overhead)
BATCHES = 20        # batches -> median


def poisson_bcoo(mesh_size):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=2)
    A, _ = fem.operator
    return A, int(A.shape[0]), int(A.nse)


def time_jax(A, n):
    # chained matvec with renormalisation -> not constant-foldable, no overflow
    @jax.jit
    def chain(v):
        def body(_, x):
            y = A @ x
            return y / jnp.linalg.norm(y)
        return jax.lax.fori_loop(0, CHAIN, body, v)
    v = jnp.ones((n,), jnp.float64)
    chain(v).block_until_ready()           # warmup / compile
    ts = []
    for _ in range(BATCHES):
        t0 = time.perf_counter()
        chain(v).block_until_ready()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return float(np.median(ts))


def time_cusparse(A, n):
    coo = sp.coo_matrix(
        (np.asarray(A.data), (np.asarray(A.indices)[:, 0], np.asarray(A.indices)[:, 1])),
        shape=(n, n),
    )
    A_cp = cusp.csr_matrix(coo.tocsr())     # cuSPARSE-backed CSR
    v = cp.ones((n,), cp.float64)
    for _ in range(CHAIN):                  # warmup
        v = A_cp @ v
        v /= cp.linalg.norm(v)
    cp.cuda.runtime.deviceSynchronize()
    ts = []
    for _ in range(BATCHES):
        v = cp.ones((n,), cp.float64)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(CHAIN):
            v = A_cp @ v
            v /= cp.linalg.norm(v)
        cp.cuda.runtime.deviceSynchronize()
        ts.append((time.perf_counter() - t0) / CHAIN)
    return float(np.median(ts))


def main():
    print(f"{'dofs':>8} {'nnz':>9} | {'BCOO us':>9} {'cuSPARSE us':>12} {'speedup':>8}")
    print("-" * 56)
    rows = []
    for ms in MESH_SIZES:
        A, n, nnz = poisson_bcoo(ms)
        t_j = time_jax(A, n)
        t_c = time_cusparse(A, n)
        rows.append((n, nnz, t_j, t_c))
        print(f"{n:8d} {nnz:9d} | {t_j*1e6:9.1f} {t_c*1e6:12.1f} {t_j/t_c:7.2f}x")
    return rows


if __name__ == "__main__":
    main()
