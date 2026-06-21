#!/usr/bin/env python
"""Dense direct vs. sparse matrix-free solve -- BiCGStab edition.

Same study as ``bench_dense_vs_sparse.py`` but with **BiCGStab** (the new
``fem.solve`` default) as the iterative method instead of CG, so the picture
matches what jNO now actually runs. Three matrix-free-comparable paths on the
same 2D Poisson BCOO system at growing DOF:

  * dense       -- jnp.linalg.solve on the densified matrix (cuSolver LU, direct)
  * BiCGStab-dense  -- BiCGStab on a *dense* matvec  (iterative, dense storage)
  * BiCGStab-sparse -- BiCGStab on the *BCOO* matvec (iterative, sparse storage)

LU-dense vs BiCGStab-dense isolates algorithm (direct vs iterative); BiCGStab-dense
vs BiCGStab-sparse isolates storage (and the memory ceiling). All at equal accuracy
(driven to residual rtol 1e-8; achieved rel-err + iteration count reported).

Each (mesh_size, method) runs in its own subprocess (``--worker``); dense is skipped
(never attempted) above a VRAM budget so a real OOM cannot freeze the desktop.

Run:  uv run --no-sync python bench_dense_vs_sparse_bicgstab.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

MESH_SIZES = [0.05, 0.028, 0.02, 0.0155, 0.0123, 0.01, 0.008, 0.0065, 0.0052]
REPS = 5
TOL = 1e-8
MAXIT = 40000
RESULT_TAG = "RESULT "


# ---------------------------------------------------------------------------
# custom BiCGStab -> returns (x, iters); matrix-free on any matvec
# ---------------------------------------------------------------------------
def bicgstab(matvec, b, *, tol=TOL, maxit=MAXIT):
    import jax
    import jax.numpy as jnp

    bnorm = jnp.linalg.norm(b)
    eps = 1e-300

    def cond(s):
        _, r, *_ , k = s
        return (jnp.linalg.norm(r) > tol * bnorm) & (k < maxit)

    def body(s):
        x, r, rhat, rho, alpha, omega, v, p, k = s
        rho_new = rhat @ r
        beta = (rho_new / (rho + eps)) * (alpha / (omega + eps))
        p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho_new / (rhat @ v + eps)
        sv = r - alpha * v
        t = matvec(sv)
        omega = (t @ sv) / (t @ t + eps)
        x = x + alpha * p + omega * sv
        r = sv - omega * t
        return x, r, rhat, rho_new, alpha, omega, v, p, k + 1

    x0 = jnp.zeros_like(b)
    r0 = b - matvec(x0)
    z = jnp.zeros_like(b)
    one = jnp.array(1.0, b.dtype)
    state = (x0, r0, r0, one, one, one, z, z, 0)
    x, r, *_, k = jax.lax.while_loop(cond, body, state)
    return x, k


# ===========================================================================
# worker
# ===========================================================================
def run_worker(mesh_size: float, method: str) -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    from shapely.geometry import box
    import jno

    out: dict = {"method": method, "mesh_size": mesh_size, "status": "ok"}

    def emit(d):
        print(RESULT_TAG + json.dumps(d), flush=True)

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=2)
    A, b = fem.operator
    b = jnp.asarray(b).reshape(-1)
    n = int(b.shape[0])
    out["dofs"] = n
    out["nnz"] = int(getattr(A, "nse", 0))

    pts = np.asarray(fem.points)[:, :2]
    u_exact = jnp.asarray(pts[:, 0] * (1 - pts[:, 0]) * pts[:, 1] * (1 - pts[:, 1]))

    def rel_err(uh):
        return float(jnp.linalg.norm(uh - u_exact) / jnp.linalg.norm(u_exact))

    def timed(fn):
        r = fn(); jax.block_until_ready(r)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            r = fn(); jax.block_until_ready(r)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), float(np.min(ts))

    try:
        if method == "dense":
            A_dense = jnp.asarray(A.todense())
            assert bool(jnp.allclose(A_dense, A_dense.T, atol=1e-8)), "stiffness not symmetric"
            solve = jax.jit(jnp.linalg.solve)
            t_med, t_min = timed(lambda: solve(A_dense, b))
            out.update(time_med_s=t_med, time_min_s=t_min, rel_err=rel_err(solve(A_dense, b)), iters=None)

        elif method in ("dense_bicg", "sparse_bicg"):
            mat = jnp.asarray(A.todense()) if method == "dense_bicg" else A
            solve = jax.jit(lambda b_, _m=mat: bicgstab(lambda v: _m @ v, b_))
            t_med, t_min = timed(lambda: solve(b)[0])
            uh, iters = solve(b)
            out.update(time_med_s=t_med, time_min_s=t_min, rel_err=rel_err(uh), iters=int(iters))
        else:
            raise ValueError(method)
    except Exception as e:  # noqa: BLE001
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
    budget = max(1.5e9, min(3.0e9, (free - 1800) * 1e6))
    print(f"free VRAM ~{free:.0f} MB  ->  dense budget {budget/1e9:.2f} GB\n")

    def ms_(r):
        return f"{r['time_med_s']*1e3:10.2f}" if r and r.get("time_med_s") else f"{'--':>10}"

    print(f"{'dofs':>7} {'nnz':>9} | {'LU-dense':>10} {'BiCG-dense':>10} {'BiCG-sparse':>11} | "
          f"{'rel_err':>9} {'bicg_it':>7}")
    print("-" * 80)

    results: list[dict] = []
    for ms in MESH_SIZES:
        sp = launch(ms, "sparse_bicg")
        if sp is None:
            continue
        dofs = sp["dofs"]
        fits = dense_fits(dofs, budget)
        dn = launch(ms, "dense") if fits else {"status": "skipped"}
        dc = launch(ms, "dense_bicg") if fits else {"status": "skipped"}
        results.append({"dofs": dofs, "nnz": sp.get("nnz"), "dense": dn, "dense_bicg": dc, "sparse": sp})
        serr = f"{sp['rel_err']:9.2e}" if sp.get("rel_err") is not None else f"{'--':>9}"
        print(f"{dofs:7d} {sp.get('nnz', 0):9d} | {ms_(dn)} {ms_(dc)} {ms_(sp)} | {serr} {sp.get('iters','--'):>7}")

    with open("bench_dense_vs_sparse_bicgstab.json", "w") as fh:
        json.dump({"free_vram_mb": free, "budget_bytes": budget, "results": results}, fh, indent=2)
    print("\nsaved bench_dense_vs_sparse_bicgstab.json")
    plot(results, budget)


def plot(results, budget):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(
        style="ticks", context="paper", palette="colorblind", font_scale=1.15,
        rc={"figure.figsize": (6, 4), "figure.dpi": 300, "savefig.dpi": 300,
            "savefig.bbox": "tight", "legend.frameon": False, "axes.grid": True,
            "grid.alpha": 0.25, "grid.linestyle": "--", "font.family": "sans-serif",
            "font.sans-serif": ["Frutiger 45 Light", "FreeSans", "DejaVu Sans"],
            "font.weight": "light", "axes.titleweight": "bold", "axes.labelweight": "bold",
            "xtick.direction": "in", "ytick.direction": "in"},
    )

    def line(key):
        xs = [r["dofs"] for r in results if r[key].get("time_med_s")]
        ys = [r[key]["time_med_s"] * 1e3 for r in results if r[key].get("time_med_s")]
        return xs, ys

    d_dof, d_t = line("dense")
    dc_dof, dc_t = line("dense_bicg")
    s_dof, s_t = line("sparse")
    s_it = [(r["dofs"], r["sparse"]["iters"]) for r in results if r["sparse"].get("iters") is not None]
    ceiling = (budget / (2.2 * 8)) ** 0.5

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(d_dof, d_t, "o-", label="LU,  dense storage  (direct)")
    ax.plot(dc_dof, dc_t, "^--", label="BiCGStab, dense storage")
    ax.plot(s_dof, s_t, "s-", label="BiCGStab, sparse/BCOO")
    ax.axvline(ceiling, color="0.4", lw=1, ls=":")
    ax.axvspan(ceiling, max(s_dof) * 1.15, color="0.5", alpha=0.08)
    ax.text(ceiling, ax.get_ylim()[0], "  dense exceeds GPU memory",
            rotation=90, va="bottom", ha="left", fontsize=8, color="0.35")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom  $N$"); ax.set_ylabel("solve time  [ms]")
    ax.set_title("Solve time: dense vs sparse (BiCGStab)")
    ax.legend(loc="upper left", fontsize=8); sns.despine(ax=ax)

    if s_it:
        xi = np.array([d for d, _ in s_it]); yi = np.array([i for _, i in s_it])
        ax2.plot(xi, yi, "s-", color=sns.color_palette("colorblind")[2], label="BiCGStab iterations")
        ax2.plot(xi, yi[0] * np.sqrt(xi / xi[0]), "k--", lw=1, alpha=0.7, label=r"$\propto\sqrt{N}$")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("degrees of freedom  $N$"); ax2.set_ylabel("BiCGStab iterations")
        ax2.set_title(f"Iteration growth (rtol={TOL:g})")
        ax2.legend(loc="upper left", fontsize=8); sns.despine(ax=ax2)

    fig.tight_layout()
    fig.savefig("dense_vs_sparse_bicgstab.png")
    print("saved dense_vs_sparse_bicgstab.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--mesh-size", type=float)
    p.add_argument("--method", choices=["dense", "dense_bicg", "sparse_bicg"])
    a = p.parse_args()
    if a.worker:
        run_worker(a.mesh_size, a.method)
    elif a.plot_only:
        blob = json.load(open("bench_dense_vs_sparse_bicgstab.json"))
        plot(blob["results"], blob["budget_bytes"])
    else:
        main()
