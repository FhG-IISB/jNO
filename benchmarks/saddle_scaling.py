"""Where does the saddle-point preconditioner overtake a sparse direct solve on 3-D Stokes?

A direct factorisation is unbeatable until its fill-in catches up with it. In 3-D that happens
early: the elimination tree of a tetrahedral Taylor-Hood system densifies fast, so LU time grows
much faster than the dof count, while a preconditioned Krylov solve grows closer to linearly.
This measures the crossover on the same problem, same machine, same tolerance -- the number that
decides which solver a 3-D fluid workflow should actually reach for.

Two solvers over one mesh sweep:

* ``lu``     -- ``jno.solve.lu(backend="host")``, sparse direct (SuperLU). The baseline, and
                exact to ~1e-13, which is the honest thing to compare against.
* ``saddle`` -- ``jno.solve.fgmres(tol=1e-10, restart=150)`` with ``jno.precond.saddle()``:
                AMG on the momentum block, weighted pressure mass as the Schur approximation.

**One process per point, deliberately.** The quantity being measured IS the memory a 3-D
factorisation commits, and several live factorisations in one interpreter measure the interpreter's
allocator instead. The default run re-execs this file once per ``(size, solver)`` pair and parses
the ``RESULT`` line each child prints; ``--point SIZE SOLVER`` is that single-measurement mode.

**Restart matters more than it looks.** ``fgmres``'s default ``restart=30`` stagnates on this
preconditioner -- quietly, still returning, just slower and less accurate. An earlier version of
this benchmark ran that default and reported the saddle path ~4.7x slower than it is. The
accuracy column is here so that failure cannot hide again: if ``err`` for ``saddle`` drifts to
1e-5 while ``lu`` sits at 1e-13, the Krylov subspace is being thrown away, not the preconditioner
failing.

Run: JAX_PLATFORMS=cpu pixi run python benchmarks/saddle_scaling.py
"""

import json
import subprocess
import sys
import time

SIZES = (0.17, 0.14, 0.12, 0.105, 0.092, 0.082)
SOLVERS = ("lu", "saddle")
RESTART = 150
TOL = 1e-10


def build_stokes_3d(size):
    """Taylor-Hood P2/P1 Stokes on the unit cube, with an exact solution IN the discrete space.

        u = (y^2+z^2, z^2+x^2, x^2+y^2)   div u == 0,  Delta u = (4,4,4)
        p = x + y + z - 3/2               zero-mean over the cube
        f = -mu Delta u + grad p = (1 - 4 mu) (1,1,1)

    Both fields are representable exactly (P2 velocity, P1 pressure), so any error in the table is
    solver error and never discretisation error -- which is what makes the accuracy column readable.
    """
    import jno

    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    mu = 1.0

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    gu, gv = grad(u, [xi, yi, zi]), grad(v, [xi, yi, zi])
    pp, vv = p.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    qq = q.bind(x=xi, y=yi, z=zi)
    f = 1.0 - 4.0 * mu
    fem = jno.fem(
        [
            mu * inner(gu, gv, n_contract=2) - pp * trace(gv) - f * (vv[0] + vv[1] + vv[2]),
            -qq * trace(gu),
            u(xb, yb, zb)[0] - (yb**2 + zb**2),
            u(xb, yb, zb)[1] - (zb**2 + xb**2),
            u(xb, yb, zb)[2] - (xb**2 + yb**2),
            p.pin(mean=True),
        ]
    )
    return fem, mu


def run_point(size, which):
    """One solve; prints the RESULT line the parent parses. Runs in its own process."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import numpy as np

    import jno

    t0 = time.perf_counter()
    fem, mu = build_stokes_3d(size)
    t_build = time.perf_counter() - t0

    pv = np.asarray(fem.field_points[0])
    exact = np.stack([pv[:, 1] ** 2 + pv[:, 2] ** 2, pv[:, 2] ** 2 + pv[:, 0] ** 2, pv[:, 0] ** 2 + pv[:, 1] ** 2], axis=1)

    if which == "lu":
        kw = {"linear": jno.solve.lu(backend="host")}
    else:
        kw = {
            "linear": jno.solve.fgmres(tol=TOL, restart=RESTART),
            "precond": jno.precond.saddle(mass_weight=1.0 / mu),
        }

    t0 = time.perf_counter()
    try:
        sol = fem.solve(**kw)
        dt = time.perf_counter() - t0
        off = fem.offsets
        vel = np.asarray(sol)[off[0] : off[1]].reshape(-1, 3)
        err = float(np.linalg.norm(vel - exact) / np.linalg.norm(exact))
        tail = f"solve={dt:.2f} err={err:.3e}"
    except Exception as exc:  # a failure is a data point, not a crash -- record and let the sweep go on
        dt = time.perf_counter() - t0
        tail = f"solve={dt:.2f} err=nan FAILED={type(exc).__name__}:{str(exc)[:60]}"
    print(f"RESULT size={size} solver={which} dofs={int(fem.dofs)} build={t_build:.1f} {tail}", flush=True)


def parse(line):
    out = {}
    for tok in line.split()[1:]:
        k, _, v = tok.partition("=")
        out[k] = v
    return out


def main():
    rows = {}
    for size in SIZES:
        for which in SOLVERS:
            proc = subprocess.run([sys.executable, __file__, "--point", str(size), which], capture_output=True, text=True)
            hit = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
            if not hit:
                # Never silently drop a point: an OOM-killed child exits without printing, and a
                # sweep that quietly skips its largest mesh reads exactly like a sweep that ran.
                print(f"  size={size} {which}: NO RESULT (exit={proc.returncode}) {proc.stderr.strip()[-200:]}")
                continue
            rec = parse(hit[-1])
            rows.setdefault(int(rec["dofs"]), {})[which] = rec
            print(f"  {hit[-1]}")

    table = []
    print(f"\n{'dofs':>8}  {'lu (s)':>9}  {'saddle (s)':>11}  {'speedup':>8}  {'lu err':>10}  {'saddle err':>10}")
    for dofs in sorted(rows):
        r = rows[dofs]
        if set(SOLVERS) - set(r):
            continue
        t_lu, t_sd = float(r["lu"]["solve"]), float(r["saddle"]["solve"])
        table.append(
            dict(
                dofs=dofs,
                t_lu=t_lu,
                t_saddle=t_sd,
                speedup=t_lu / t_sd,
                err_lu=float(r["lu"]["err"]),
                err_saddle=float(r["saddle"]["err"]),
            )
        )
        print(
            f"{dofs:8d}  {t_lu:9.2f}  {t_sd:11.2f}  {t_lu / t_sd:7.2f}x  "
            f"{float(r['lu']['err']):10.1e}  {float(r['saddle']['err']):10.1e}"
        )

    with open("benchmarks/saddle_scaling.json", "w") as f:
        json.dump(dict(restart=RESTART, tol=TOL, rows=table), f, indent=1)
    under = [r for r in table if r["speedup"] <= 1.0]
    over = [r for r in table if r["speedup"] > 1.0]
    where = f"between {under[-1]['dofs']} and {over[0]['dofs']} dofs" if under and over else "not bracketed by this sweep"
    print(f"\ncrossover {where}; wrote benchmarks/saddle_scaling.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--point":
        run_point(float(sys.argv[2]), sys.argv[3])
    else:
        main()
