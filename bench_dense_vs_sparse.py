#!/usr/bin/env python
"""Dense direct vs. sparse matrix-free solve, benchmarked on the GPU.

A 2D Poisson Dirichlet problem (manufactured solution u = x(1-x)y(1-y)) is
assembled through jNO/feax into a sparse BCOO stiffness matrix, then solved two
ways at increasing DOF count:

  * dense   -- jnp.linalg.solve on the densified (N x N) matrix  [the current
               fem.solve() default; cuSolver LU]
  * sparse  -- lineax CG, matrix-free, wrapping the BCOO matvec v -> A @ v
               (the path that never forms the N x N matrix)

Both are timed solve-only (densification is treated purely as the memory
ceiling, not as solve cost). CG rtol is tight so its accuracy matches the dense
solve -- the two time-lines are compared at *equal accuracy*. CG iteration count
is logged because unpreconditioned 2D-Poisson CG grows ~O(sqrt(N)), so the
sparse line bends upward; that is reported, not hidden.

Each (mesh_size, method) runs in its own subprocess (``--worker``) so a dense
out-of-memory cannot take down the shared desktop GPU: dense is *skipped* (never
attempted) above a memory budget computed from the actually-free VRAM.

Run:  uv run --no-sync python bench_dense_vs_sparse.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# mesh sizes -> roughly dofs ~ 1.2 / mesh_size**2  (513 @ .05 ... ~45k @ .0052)
MESH_SIZES = [0.05, 0.028, 0.02, 0.0155, 0.0123, 0.01, 0.008, 0.0065, 0.0052]
REPS = 5            # timed repetitions (median reported), after 1 warmup
CG_RTOL = 1e-8     # tight -> CG accuracy tracks the dense direct solve
CG_MAX_STEPS = 60000
RESULT_TAG = "RESULT "


# ===========================================================================
# worker: one (mesh_size, method) measurement in an isolated process
# ===========================================================================
def run_worker(mesh_size: float, method: str) -> None:
    # let XLA grow rather than grab the whole card (shared with the desktop)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import jax
    jax.config.update("jax_enable_x64", True)  # feax assembly is float64
    import jax.numpy as jnp
    import numpy as np
    from shapely.geometry import box

    import jno

    out: dict = {"method": method, "mesh_size": mesh_size, "status": "ok"}

    def emit(d):
        print(RESULT_TAG + json.dumps(d), flush=True)

    # -- assemble the Poisson system -> sparse BCOO (A, b) -------------------
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=2)
    A, b = fem.operator                      # A is jax.experimental.sparse.BCOO
    b = jnp.asarray(b).reshape(-1)
    n = int(b.shape[0])
    out["dofs"] = n
    out["nnz"] = int(getattr(A, "nse", 0))

    # exact nodal solution for the equal-accuracy check
    pts = np.asarray(fem.points)[:, :2]
    u_exact = jnp.asarray(pts[:, 0] * (1 - pts[:, 0]) * pts[:, 1] * (1 - pts[:, 1]))

    def rel_err(uh):
        return float(jnp.linalg.norm(uh - u_exact) / jnp.linalg.norm(u_exact))

    def timed(fn):
        fn()  # warmup / compile
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = fn()
            jax.block_until_ready(r)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), float(np.min(ts))

    try:
        if method == "dense":
            A_dense = jnp.asarray(A.todense())
            # one-time symmetry assertion: CG/SPD validity, fail loudly otherwise
            assert bool(jnp.allclose(A_dense, A_dense.T, atol=1e-8)), "stiffness not symmetric"
            solve = jax.jit(jnp.linalg.solve)
            t_med, t_min = timed(lambda: solve(A_dense, b))
            out.update(time_med_s=t_med, time_min_s=t_min,
                       rel_err=rel_err(solve(A_dense, b)), iters=None)

        elif method in ("sparse", "dense_cg"):
            # Same iterative algorithm (CG); the ONLY difference is the matvec
            # operand -- BCOO A (sparse) vs the densified A (dense_cg). Comparing
            # these two isolates the effect of *storage*; comparing dense_cg to
            # the LU "dense" line isolates the effect of *algorithm*.
            import lineax as lx
            tags = (lx.positive_semidefinite_tag, lx.symmetric_tag)
            struct = jax.ShapeDtypeStruct(b.shape, b.dtype)
            mat = jnp.asarray(A.todense()) if method == "dense_cg" else A

            @jax.jit
            def solve_cg(b_, _m=mat):
                op = lx.FunctionLinearOperator(lambda v: _m @ v, struct, tags=tags)
                sol = lx.linear_solve(
                    op, b_, lx.CG(rtol=CG_RTOL, atol=1e-12, max_steps=CG_MAX_STEPS)
                )
                return sol.value, sol.stats["num_steps"]

            t_med, t_min = timed(lambda: solve_cg(b))
            uh, iters = solve_cg(b)
            out.update(time_med_s=t_med, time_min_s=t_min,
                       rel_err=rel_err(uh), iters=int(iters))
        else:
            raise ValueError(method)
    except Exception as e:  # noqa: BLE001 -- report, don't crash the orchestrator
        out["status"] = f"error: {type(e).__name__}: {e}"

    emit(out)


# ===========================================================================
# orchestrator
# ===========================================================================
def free_vram_mb() -> float:
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        return float(q.stdout.strip().splitlines()[0])
    except Exception:
        return 4000.0


def dense_fits(dofs: int, budget_bytes: float) -> bool:
    # N*N float64 matrix + ~1x LU workspace -> ~2.2x the matrix
    return 2.2 * 8 * dofs * dofs < budget_bytes


def launch(mesh_size: float, method: str) -> dict | None:
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", TF_CPP_MIN_LOG_LEVEL="3")
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "--worker", "--mesh-size", str(mesh_size), "--method", method],
        capture_output=True, text=True, env=env,
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
    sys.stderr.write(proc.stderr[-2000:] + "\n")
    return None


def main() -> None:
    free = free_vram_mb()
    # leave ~1.8 GB headroom for the live desktop/compositor; cap dense at <=3 GB
    budget = max(1.5e9, min(3.0e9, (free - 1800) * 1e6))
    print(f"free VRAM ~{free:.0f} MB  ->  dense memory budget {budget/1e9:.2f} GB\n")

    def ms_(r):  # solve time in ms or "--"
        return f"{r['time_med_s']*1e3:9.2f}" if r and r.get("time_med_s") else f"{'--':>9}"

    results: list[dict] = []
    print(f"{'dofs':>8} {'nnz':>9} | {'LU-dense':>9} {'CG-dense':>9} {'CG-sparse':>9} | "
          f"{'rel_err':>9} {'cg_it':>6}")
    print("-" * 76)

    for ms in MESH_SIZES:
        sp = launch(ms, "sparse")
        if sp is None:
            continue
        dofs = sp["dofs"]
        fits = dense_fits(dofs, budget)
        dn = launch(ms, "dense") if fits else \
            {"method": "dense", "dofs": dofs, "status": "skipped: exceeds mem budget"}
        dc = launch(ms, "dense_cg") if fits else \
            {"method": "dense_cg", "dofs": dofs, "status": "skipped: exceeds mem budget"}

        results.append({"dofs": dofs, "nnz": sp.get("nnz"),
                        "dense": dn, "dense_cg": dc, "sparse": sp})

        serr = f"{sp['rel_err']:9.2e}" if sp.get("rel_err") is not None else f"{'--':>9}"
        print(f"{dofs:8d} {sp.get('nnz', 0):9d} | {ms_(dn)} {ms_(dc)} {ms_(sp)} | "
              f"{serr} {sp.get('iters', '--'):>6}")

    with open("bench_dense_vs_sparse.json", "w") as fh:
        json.dump({"free_vram_mb": free, "budget_bytes": budget, "results": results}, fh, indent=2)
    print("\nsaved bench_dense_vs_sparse.json")
    make_plot(results, budget)


def make_plot(results: list[dict], budget: float) -> None:
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(
        style="ticks", context="paper", palette="colorblind", font_scale=1.15,
        rc={
            "figure.figsize": (6, 4), "figure.dpi": 300, "savefig.dpi": 300,
            "savefig.bbox": "tight", "legend.frameon": False, "axes.grid": True,
            "grid.alpha": 0.25, "grid.linestyle": "--",
            "font.family": "sans-serif",
            "font.sans-serif": ["Frutiger 45 Light", "Frutiger", "FreeSans", "DejaVu Sans"],
            "font.weight": "light", "axes.titleweight": "bold", "axes.labelweight": "bold",
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        },
    )

    def line(key):
        xs = [r["dofs"] for r in results if r[key].get("time_med_s")]
        ys = [r[key]["time_med_s"] * 1e3 for r in results if r[key].get("time_med_s")]
        return xs, ys

    d_dof, d_t = line("dense")
    dc_dof, dc_t = line("dense_cg")
    s_dof, s_t = line("sparse")
    s_it = [(r["dofs"], r["sparse"]["iters"]) for r in results
            if r["sparse"].get("iters") is not None]
    dense_ceiling = (budget / (2.2 * 8)) ** 0.5  # max DOF dense fits in budget

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- panel 1: solve time. Two axes of variation, disentangled:
    #   LU-dense  vs  CG-dense  -> effect of ALGORITHM (direct vs iterative)
    #   CG-dense  vs  CG-sparse -> effect of STORAGE   (and the memory ceiling)
    ax.plot(d_dof, d_t, "o-", label="LU,  dense storage  (direct)")
    ax.plot(dc_dof, dc_t, "^--", label="CG,  dense storage  (iterative)")
    ax.plot(s_dof, s_t, "s-", label="CG,  sparse/BCOO     (iterative, matrix-free)")
    ax.axvline(dense_ceiling, color="0.4", lw=1, ls=":")
    ax.axvspan(dense_ceiling, max(s_dof) * 1.15, color="0.5", alpha=0.08)
    ax.text(dense_ceiling, ax.get_ylim()[0], "  dense exceeds GPU memory",
            rotation=90, va="bottom", ha="left", fontsize=8, color="0.35")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom  $N$")
    ax.set_ylabel("solve time  [ms]")
    ax.set_title("Solve time vs problem size (8 GB GPU, float64)")
    ax.legend(loc="upper left", fontsize=8)
    sns.despine(ax=ax)

    # -- panel 2: CG iteration growth ---------------------------------------
    if s_it:
        xi = np.array([d for d, _ in s_it]); yi = np.array([i for _, i in s_it])
        ax2.plot(xi, yi, "s-", color=sns.color_palette("colorblind")[1], label="CG iterations")
        ref = yi[0] * np.sqrt(xi / xi[0])  # O(sqrt N) reference
        ax2.plot(xi, ref, "k--", lw=1, alpha=0.7, label=r"$\propto\sqrt{N}$")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("degrees of freedom  $N$")
        ax2.set_ylabel("CG iterations to converge")
        ax2.set_title(f"Unpreconditioned CG cost (rtol={CG_RTOL:g})")
        ax2.legend(loc="upper left", fontsize=8)
        sns.despine(ax=ax2)

    fig.savefig("dense_vs_sparse_solve_time.png")
    print("saved dense_vs_sparse_solve_time.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--mesh-size", type=float)
    p.add_argument("--method", choices=["dense", "dense_cg", "sparse"])
    a = p.parse_args()
    if a.worker:
        run_worker(a.mesh_size, a.method)
    elif a.plot_only:
        blob = json.load(open("bench_dense_vs_sparse.json"))
        make_plot(blob["results"], blob["budget_bytes"])
    else:
        main()
