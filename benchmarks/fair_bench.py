"""One problem, measured fairly: 3-D Poisson on the unit cube.

`compare_libs.py` runs all three libraries **in one process**, jNO first. That hands jNO the entire
one-time JAX/XLA initialisation inside its `build_s` and lets whichever library runs second inherit a
warm XLA runtime — measured here at ~2.1 s, which is most of jNO's reported build at small sizes. The
`elem_map` docstring warns about exactly this trap ("running both in one process lets whichever goes
second inherit the other's warm XLA cache, which is how an earlier comparison flattered itself"), and
the existing benchmark walks into it.

This harness fixes three things and changes nothing else:

1. **One library per PROCESS.** No shared XLA state, no ordering advantage.
2. **Init is timed separately** from build, so a one-time cost cannot masquerade as per-problem work.
3. **Build and solve are each run TWICE.** The repeat is the number a real workflow sees — a transient
   march, a Newton loop or an optimiser builds once and solves many times. Reporting only the cold
   number measures compilation; reporting only the warm one hides it. Both are printed.

Deliberately ONE problem and one size: the point is a trustworthy number, not a sweep.

    pixi run python benchmarks/fair_bench.py              # driver: one subprocess per library
    pixi run python benchmarks/fair_bench.py --lib jno    # one library, emits JSON
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# JAX preallocates ~75% of the device by default, which fails outright when anything else (a browser,
# another run) already holds a few GB -- measured here as a RESOURCE_EXHAUSTED on a `jit_true_divide`.
# Allocating on demand costs a little speed but makes the benchmark runnable on a shared desktop GPU;
# it applies equally to jNO and JAX-FEM, so it does not tilt the comparison.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

MESH_SIZE = 0.03  # ~31k nodes in 3-D, matching compare_libs.json's "large" case
DIM = 3


def _time(fn):
    t0 = time.perf_counter()
    out = fn()
    return time.perf_counter() - t0, out


# ---------------------------------------------------------------------------------------- jNO
def run_jno():
    t_import, _ = _time(lambda: __import__("jno"))
    import jax
    import numpy as np

    import jno

    # force backend initialisation so it cannot land inside the build measurement
    t_init, _ = _time(lambda: jax.block_until_ready(jax.numpy.ones(1) * 2))

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=MESH_SIZE).domain()
    nodes = int(np.asarray(d.built_mesh.points).shape[0])

    def build():
        c = d.variable("interior", split=True)
        cb = d.variable("boundary", split=True)
        u, v = d.fem_symbols()
        ui, vi = u.bind(x=c[0], y=c[1], z=c[2]), v.bind(x=c[0], y=c[1], z=c[2])
        return jno.fem(
            [ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(cb[0], cb[1], cb[2]) - 0.0],
            element_type="TET4",
        )

    b1, fem1 = _time(build)
    b2, fem2 = _time(build)
    s1, sol = _time(lambda: np.asarray(jax.block_until_ready(fem1.solve())).reshape(-1))
    s2, _ = _time(lambda: np.asarray(jax.block_until_ready(fem2.solve())).reshape(-1))
    return dict(
        lib="jNO",
        solver="Jacobi-BiCGStab (GPU)",
        nodes=nodes,
        import_s=t_import,
        init_s=t_init,
        build1_s=b1,
        build2_s=b2,
        solve1_s=s1,
        solve2_s=s2,
        unorm=float(np.linalg.norm(sol)),
    )


# ------------------------------------------------------------------------------------ JAX-FEM
def run_jax_fem():
    t_import, _ = _time(lambda: __import__("jax_fem"))
    import jax
    import numpy as np
    from compare_libs import build_mesh

    t_init, _ = _time(lambda: jax.block_until_ready(jax.numpy.ones(1) * 2))
    _, points, cells = build_mesh(MESH_SIZE, dim=DIM)

    import jax.numpy as jnp
    from jax_fem.generate_mesh import Mesh
    from jax_fem.problem import Problem
    from jax_fem.solver import solver

    class Poisson(Problem):
        def get_tensor_map(self):
            return lambda x: x

        def get_mass_map(self):
            return lambda _u, _x: jnp.array([-1.0])

    # jax-fem calls the location fn on TRACED points, so it must be jnp: numpy raises
    # TracerArrayConversionError. p is a single point of shape (dim,).
    def boundary(p):
        return jnp.any((p < 1e-8) | (p > 1.0 - 1e-8))

    mesh = Mesh(points, cells, ele_type="TET4")

    def build():
        return Poisson(
            mesh=mesh,
            vec=1,
            dim=DIM,
            ele_type="TET4",
            dirichlet_bc_info=[[boundary], [0], [lambda _p: 0.0]],
        )

    b1, p1 = _time(build)
    b2, p2 = _time(build)
    s1, sol = _time(lambda: np.asarray(solver(p1)[0]).reshape(-1))
    s2, _ = _time(lambda: np.asarray(solver(p2)[0]).reshape(-1))
    return dict(
        lib="JAX-FEM",
        solver="jax_solver, preconditioned (GPU)",
        nodes=int(points.shape[0]),
        import_s=t_import,
        init_s=t_init,
        build1_s=b1,
        build2_s=b2,
        solve1_s=s1,
        solve2_s=s2,
        unorm=float(np.linalg.norm(sol)),
    )


# --------------------------------------------------------------------------------- scikit-fem
def run_skfem():
    t_import, _ = _time(lambda: __import__("skfem"))
    import numpy as np
    from compare_libs import build_mesh, on_boundary
    from skfem import Basis, ElementTetP1, MeshTet, asm, condense, solve
    from skfem.models.poisson import laplace, unit_load

    _, points, cells = build_mesh(MESH_SIZE, dim=DIM)
    t_init = 0.0  # CPU/numpy: no accelerator to warm

    mesh = MeshTet(points.T, cells.T)

    def build():
        basis = Basis(mesh, ElementTetP1())
        return asm(laplace, basis), asm(unit_load, basis), basis

    b1, (A1, f1, basis1) = _time(build)
    b2, (A2, f2, _basis2) = _time(build)
    dofs = np.nonzero(on_boundary(points))[0]
    s1, sol = _time(lambda: solve(*condense(A1, f1, D=dofs)))
    s2, _ = _time(lambda: solve(*condense(A2, f2, D=dofs)))
    return dict(
        lib="scikit-fem",
        solver="scipy sparse direct (CPU)",
        nodes=int(points.shape[0]),
        import_s=t_import,
        init_s=t_init,
        build1_s=b1,
        build2_s=b2,
        solve1_s=s1,
        solve2_s=s2,
        unorm=float(np.linalg.norm(sol)),
    )


RUNNERS = {"jno": run_jno, "jax-fem": run_jax_fem, "scikit-fem": run_skfem}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=sorted(RUNNERS))
    args = ap.parse_args()

    if args.lib:  # child: one library, this process only
        print("@@RESULT@@" + json.dumps(RUNNERS[args.lib]()))
        return

    rows = []
    for lib in ("jno", "jax-fem", "scikit-fem"):
        print(f"--- {lib} (own process) ---", flush=True)
        env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
        p = subprocess.run([sys.executable, __file__, "--lib", lib], capture_output=True, text=True, env=env)
        line = next((ln for ln in p.stdout.splitlines() if ln.startswith("@@RESULT@@")), None)
        if line is None:
            reason = (p.stderr.strip().splitlines() or ["(no stderr)"])[-1]
            print(f"    SKIPPED: {reason[:120]}")
            continue
        rows.append(json.loads(line[len("@@RESULT@@") :]))

    if not rows:
        print("\nno library ran.")
        return
    print(f"\n=== 3-D Poisson, unit cube, mesh_size={MESH_SIZE} — one process each ===")
    hdr = f"{'lib':>11} {'nodes':>8} {'import':>8} {'init':>8} {'build#1':>9} {'build#2':>9} {'solve#1':>9} {'solve#2':>9}"
    print(hdr)
    for r in rows:
        print(
            f"{r['lib']:>11} {r['nodes']:8,} {r['import_s']:7.2f}s {r['init_s']:7.2f}s "
            f"{r['build1_s']:8.3f}s {r['build2_s']:8.3f}s {r['solve1_s']:8.3f}s {r['solve2_s']:8.3f}s"
        )
    print("\nbuild#2 / solve#2 are the steady state a repeated workflow sees; #1 carries compilation.")
    print("`init` is one-time per process and belongs to NEITHER build nor solve.")
    with open("benchmarks/fair_bench.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    print("wrote benchmarks/fair_bench.json")


if __name__ == "__main__":
    main()
