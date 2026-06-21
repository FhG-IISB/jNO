#!/usr/bin/env python
"""Linear-FEM solve: library solvers vs hand-rolled custom code.

Same 2D Poisson BCOO system, four matrix-free conjugate-gradient paths, timed
end-to-end at equal accuracy (all driven to residual rtol 1e-8; achieved rel-err
and iteration count reported so the comparison is honest):

  1. lineax        -- lx.CG via lx.linear_solve            (library)
  2. jax.scipy     -- jax.scipy.sparse.linalg.cg           (native JAX)
  3. custom        -- a CG written here in ~12 lines, one fused jax.jit
  4. custom+Jacobi -- the same CG with a diagonal preconditioner

Everything is matrix-free on A @ v (BCOO), so there is no dense matrix and no
memory wall -- this runs single-process across sizes.

What to watch (the "dynamics"):
  * library vs custom -> per-solve overhead of the library (lineax is heaviest)
  * custom vs custom+Jacobi -> whether diagonal scaling helps (for constant-
    coefficient Poisson on a quasi-uniform mesh it barely does -- the diagonal
    is nearly constant; Jacobi pays off on heterogeneous / graded problems)

Run:  uv run --no-sync python bench_solvers.py
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
from shapely.geometry import box

import lineax as lx
import jno

MESH_SIZES = [0.02, 0.0123, 0.008, 0.0052]   # ~3k ... ~43k DOF
TOL = 1e-8
MAXIT = 100_000
REPS = 5


# ---------------------------------------------------------------------------
# custom matrix-free CG (this is the "implement it in jNO" core -- ~12 lines)
# ---------------------------------------------------------------------------
def cg_solve(matvec, b, *, minv=None, tol=TOL, maxit=MAXIT):
    """Preconditioned CG. minv=None -> identity (plain CG). Fully traceable."""
    bnorm = jnp.linalg.norm(b)
    apply_minv = (lambda r: r) if minv is None else (lambda r: minv * r)

    def cond(state):
        _, r, _, _, k = state
        return (jnp.linalg.norm(r) > tol * bnorm) & (k < maxit)

    def body(state):
        x, r, p, rz, k = state
        Ap = matvec(p)
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        z = apply_minv(r)
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        return x, r, p, rz_new, k + 1

    x0 = jnp.zeros_like(b)
    r0 = b - matvec(x0)
    z0 = apply_minv(r0)
    state = (x0, r0, z0, r0 @ z0, 0)
    x, r, p, rz, k = jax.lax.while_loop(cond, body, state)
    return x, k


def poisson(mesh_size):
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
    idx = A.indices
    diag = jnp.zeros(n, b.dtype).at[idx[:, 0]].add(
        jnp.where(idx[:, 0] == idx[:, 1], A.data, 0.0))   # diag(A) without densifying
    pts = np.asarray(fem.points)[:, :2]
    u_exact = jnp.asarray(pts[:, 0] * (1 - pts[:, 0]) * pts[:, 1] * (1 - pts[:, 1]))
    return A, b, n, 1.0 / diag, u_exact


def make_solvers(A, b, minv):
    mv = lambda v: A @ v
    struct = jax.ShapeDtypeStruct(b.shape, b.dtype)
    tags = (lx.positive_semidefinite_tag, lx.symmetric_tag)

    @jax.jit
    def s_lineax(b_):
        op = lx.FunctionLinearOperator(mv, struct, tags=tags)
        sol = lx.linear_solve(op, b_, lx.CG(rtol=TOL, atol=1e-14, max_steps=MAXIT))
        return sol.value, sol.stats["num_steps"]

    @jax.jit
    def s_jsp(b_):
        x, _ = jax.scipy.sparse.linalg.cg(mv, b_, tol=TOL, atol=0.0, maxiter=MAXIT)
        return x, jnp.array(-1)            # jax.scipy.cg doesn't expose iters

    s_custom = jax.jit(lambda b_: cg_solve(mv, b_))
    s_jacobi = jax.jit(lambda b_: cg_solve(mv, b_, minv=minv))
    return {"lineax": s_lineax, "jax.scipy": s_jsp,
            "custom": s_custom, "custom+Jacobi": s_jacobi}


def time_solver(fn, b, u_exact):
    x, it = fn(b); jax.block_until_ready(x)            # warmup / compile
    rel = float(jnp.linalg.norm(x - u_exact) / jnp.linalg.norm(u_exact))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        x, _ = fn(b); jax.block_until_ready(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), int(it), rel


def main():
    names = ["lineax", "jax.scipy", "custom", "custom+Jacobi"]
    print(f"{'dofs':>7} | " + " | ".join(f"{n:>16}" for n in names))
    print(f"{'':>7} | " + " | ".join(f"{'ms':>7} {'it':>4} {'err':>3}" for _ in names))
    print("-" * 95)

    results = []
    for ms in MESH_SIZES:
        A, b, n, minv, u_exact = poisson(ms)
        solvers = make_solvers(A, b, minv)
        row = {"dofs": n}
        cells = []
        for name in names:
            t, it, rel = time_solver(solvers[name], b, u_exact)
            row[name] = {"ms": t * 1e3, "iters": it, "rel_err": rel}
            itxt = f"{it:4d}" if it >= 0 else "  --"
            cells.append(f"{t*1e3:7.1f} {itxt} {rel:6.0e}".replace("e-0", "e-"))
        results.append(row)
        print(f"{n:7d} | " + " | ".join(cells))

    plot(results, names)


def plot(results, names):
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
    dof = [r["dofs"] for r in results]
    markers = {"lineax": "o-", "jax.scipy": "v-", "custom": "s-", "custom+Jacobi": "D-"}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in names:
        ax.plot(dof, [r[name]["ms"] for r in results], markers[name], label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("degrees of freedom  $N$"); ax.set_ylabel("full solve time  [ms]")
    ax.set_title("Solve time: library vs custom CG")
    ax.legend(loc="upper left", fontsize=8); sns.despine(ax=ax)

    for name in ("lineax", "custom", "custom+Jacobi"):
        its = [r[name]["iters"] for r in results]
        if all(i >= 0 for i in its):
            ax2.plot(dof, its, markers[name], label=name)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("degrees of freedom  $N$"); ax2.set_ylabel("CG iterations")
    ax2.set_title("CG iterations (Jacobi ~ no-op here)")
    ax2.legend(loc="upper left", fontsize=8); sns.despine(ax=ax2)
    fig.tight_layout()
    fig.savefig("solver_comparison.png")
    print("\nsaved solver_comparison.png")


if __name__ == "__main__":
    main()
