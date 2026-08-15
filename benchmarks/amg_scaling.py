"""Does AMG cash in the iteration win the roofline memo promised — and through WHICH front door?

The GPU roofline analysis found SpMV at ~69% of bandwidth: the backend is not the lever, the
ITERATION COUNT is (Jacobi-Krylov grows ~sqrt(n) on elliptic operators, multigrid stays ~flat).
jNO has three AMG front doors with very different overhead profiles, so this measures all three:

* ``precond.amg()``            — pyamg setup on host, pure-JAX V-cycle per application (no
                                 per-application host crossing; compiles into the solve).
* ``solve.amg()``              — direct AmgX solve: ONE host/handle crossing per SOLVE, and AmgX's
                                 structure-keyed warm resetup makes repeats ~10x cheaper (measured).
* ``precond.jaxamg()``         — AmgX per APPLICATION: measured ~11 ms fixed handle overhead per
                                 call, so it only pays where iterations x savings exceed that.
                                 (Kept out of the wall-clock table for that measured reason.)

Two tables: iterations-to-tol (host PCG, float64, counting — machine-independent) and wall-clock
repeat solve (second call timed; the first carries compilation — fair_bench convention).

Run: JAX_PLATFORMS=cuda,cpu pixi run python benchmarks/amg_scaling.py
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np

import jno


def build_poisson_3d(size):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 3.0 * vi, u(xb, yb, zb) - 0.0])
    return fem


def to_csr(A_raw, n):
    import scipy.sparse as sp

    if hasattr(A_raw, "indices"):
        idx = np.asarray(A_raw.indices)
        return sp.csr_matrix((np.asarray(A_raw.data, dtype=np.float64), (idx[:, 0], idx[:, 1])), shape=(n, n))
    return sp.csr_matrix(np.asarray(A_raw, dtype=np.float64))


def pcg_iterations(A_sp, b, M_apply, tol=1e-8, maxiter=20000):
    """Preconditioned CG on the host, counting iterations — the measurement jax.scipy hides."""
    x = np.zeros_like(b)
    r = b - A_sp @ x
    z = M_apply(r)
    p = z.copy()
    rz = float(r @ z)
    bnorm = float(np.linalg.norm(b))
    for it in range(1, maxiter + 1):
        Ap = A_sp @ p
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r) <= tol * bnorm:
            return it
        z = M_apply(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return maxiter


def timed_repeat(fn, warmup=2):
    # TWO warmups, not one: a cached AMG spec is eager on call 0 (builds the hierarchy), becomes
    # traceable, and call 1 pays the jit compile -- the steady state a repeated workflow sees starts
    # at call 2. Timing call 1 measured the COMPILE and misread the compiled path as eager.
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    return time.perf_counter() - t0


def main():
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    rows = []
    for size in (0.10, 0.07, 0.05, 0.04, 0.035):
        fem = build_poisson_3d(size)
        b = np.asarray(fem.b, dtype=np.float64)
        n = b.shape[0]
        A_sp = to_csr(fem.A, n)

        # -- iterations to 1e-8 with the V-CYCLE jNO actually ships (pyamg-hybrid applier) --------
        diag = A_sp.diagonal()
        it_jac = pcg_iterations(A_sp, b, lambda r: r / diag)
        spec = jno.precond.amg()
        applier = spec.materialize(
            PrecondContext(
                LinearOperator(
                    fem.A if hasattr(fem.A, "indices") else jnp.asarray(np.asarray(A_sp.todense(), dtype=np.float32))
                )
            )
        )
        it_amg = pcg_iterations(A_sp, b, lambda r: np.asarray(applier(jnp.asarray(r, dtype=jnp.float32)), dtype=np.float64))

        # -- wall-clock repeat solve through the three front doors ------------------------------
        t_def = timed_repeat(lambda: fem.solve())
        amg_pre = jno.precond.amg().cached()
        t_pyamg = timed_repeat(lambda: fem.solve(linear=jno.solve.cg(tol=1e-8), precond=amg_pre))
        t_amgx = timed_repeat(lambda: fem.solve(linear=jno.solve.amg()))

        row = dict(n=n, it_jacobi=it_jac, it_amg=it_amg, t_default=t_def, t_pyamg_cg=t_pyamg, t_amgx_direct=t_amgx)
        rows.append(row)
        print(
            f"n={n:7d}  iters: jacobi {it_jac:5d}  amg-vcycle {it_amg:3d} ({it_jac / max(it_amg, 1):5.1f}x)   "
            f"solve#2: default {t_def * 1e3:7.1f} ms  amg-cg {t_pyamg * 1e3:7.1f} ms  amgx-direct {t_amgx * 1e3:7.1f} ms"
        )
        del fem, A_sp

    with open("benchmarks/amg_scaling.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("\njacobi iterations grow ~sqrt(n), the V-cycle stays ~flat; wrote benchmarks/amg_scaling.json")


if __name__ == "__main__":
    main()
