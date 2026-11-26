"""``lu(backend="pardiso")`` keeps one thread count per factorization, whichever thread calls it.

MKL PARDISO fixes its parallel schedule in the symbolic analysis (phase 11). A numeric factorization
(phase 22) entered with a different thread count writes outside its buffers. Measured with an analysis
on 14 threads and a refactorization on 1: a 1,431-unknown elasticity system came back with a relative
residual of order 1 and no error, or the process died with a glibc heap message; the 13,824-unknown
Laplacian used here came back as NaN. Small systems (a few thousand unknowns) are factored serially
and do not show it.

jNO reaches that state by itself. The thread count is a property of the CALLING thread, and the callback
runs on a pool thread under ``jax.jit`` but on the main thread when the solve is not compiled. Anything
that lowers the main thread's count (a mesh generator that sets its own OpenMP thread count, a
``threadpoolctl`` limit, ``mkl_set_num_threads_local``) therefore splits one cached factorization
across two counts: compile a solve, then call it un-compiled.

The scenario runs in a subprocess because the unfixed outcome is memory corruption, which can end the
interpreter rather than fail an assertion.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

SCENARIO = textwrap.dedent(
    """
    import sys
    import jax; jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp, numpy as np, scipy.sparse as sp
    from jax.experimental import sparse as jsp
    from pypardiso import PyPardisoSolver
    from jno.utils.solver.linear import pardiso_lu_solve

    mkl = PyPardisoSolver().libmkl
    if mkl.MKL_Get_Max_Threads() < 2:
        print("ONE THREAD ONLY"); sys.exit(0)

    def laplacian(m):  # 7-point stencil on an m^3 grid: symmetric, so PARDISO factors it as LDL^T
        I, T = sp.identity(m), sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(m, m))
        A = (sp.kron(sp.kron(T, I), I) + sp.kron(sp.kron(I, T), I) + sp.kron(sp.kron(I, I), T)).tocoo()
        return A, jnp.asarray(np.c_[A.row, A.col]), jnp.asarray(A.data)

    def residual(A, scale, x, b):
        r = float(np.linalg.norm(scale * (A @ np.asarray(x)) - b) / np.linalg.norm(b))
        return r if np.isfinite(r) else float("inf")  # a NaN solution must not vanish in max()

    worst = 0.0
    for m, compiled_first in ((24, True), (26, False)):  # large enough for a parallel schedule
        A, indices, values = laplacian(m)
        b = np.cos(np.arange(m**3, dtype=np.float64))
        solve = lambda scale: pardiso_lu_solve(jsp.BCOO((scale * values, indices), shape=A.shape), jnp.asarray(b))
        compiled = jax.jit(solve)
        # the main thread asks MKL for one thread from here on; the pool threads keep the default
        mkl.MKL_Set_Num_Threads_Local(1)
        order = (compiled, solve, compiled, solve) if compiled_first else (solve, compiled, solve, compiled)
        for k, f in enumerate(order):
            scale = 1.0 + 0.25 * k  # new values on the cached pattern: a numeric refactorization
            worst = max(worst, residual(A, scale, f(scale), b))
        mkl.MKL_Set_Num_Threads_Local(0)
        assert mkl.MKL_Get_Max_Threads() > 1, "the solve left the caller's own thread setting changed"
    print(f"WORST RESIDUAL {worst:.3e}")
    """
)


def test_compiled_and_uncompiled_solves_share_one_factorization():
    """A compiled solve and an un-compiled one on the same pattern, in both orders, with the main thread
    limited to one MKL thread: every solution satisfies its system, and the process survives."""
    from jno.utils.solver.linear import _pardiso_available

    if not _pardiso_available():
        pytest.skip("pypardiso (with the private phase hooks this backend drives) is not installed")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    for name in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "MKL_THREADING_LAYER"):
        env.pop(name, None)  # a one-thread or sequential MKL cannot show the defect
    r = subprocess.run([sys.executable, "-c", SCENARIO], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, (
        f"the process ended with exit {r.returncode} (a negative value is a signal: memory corruption "
        f"inside MKL PARDISO):\n{r.stdout[-300:]}\n{r.stderr[-900:]}"
    )
    if "ONE THREAD ONLY" in r.stdout:
        pytest.skip("MKL runs single-threaded on this machine, so the thread count cannot differ")
    worst = float(r.stdout.split("WORST RESIDUAL")[1].split()[0])
    assert worst < 1e-9, f"a PARDISO solve returned a vector that does not satisfy its system: residual {worst:.2e}"
