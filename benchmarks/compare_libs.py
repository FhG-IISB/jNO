"""Same problem, same mesh, same machine: jNO against JAX-FEM and scikit-fem.

    <cmpenv>/bin/python benchmarks/compare_libs.py 0.08 0.06 0.05

WHY THIS EXISTS. Published FEM benchmarks cannot be compared against this suite's numbers. The
JAX-FEM paper (arXiv:2212.00964) reports a Quadro RTX 8000; TensorGalerkin (arXiv:2602.05052) an
H200. These solves are bandwidth-bound, so hardware alone is a ~10x confound, and neither paper
separates assembly from solve the way this suite does. The only way to get a number that means
something is to run the comparators on THIS card.

THE PROBLEM is the literature convention, not this suite's hardened one: constant-coefficient
3-D Poisson, ``-div(grad u) = 1`` on the unit cube with ``u = 0`` on the boundary, P1 tetrahedra.
The hardened cases deliberately carry a high-contrast coefficient to make the solve measurable;
that makes them harder than what everyone else reports, so they are the wrong thing to compare.

FAIRNESS. One mesh is generated once and handed to all three libraries, so mesh generation and
element count are identical by construction and only assembly + solve differ. Every library keeps
its OWN default solver -- that is the thing being compared -- so the solver used is reported next
to each number rather than being equalised away. The solutions are compared against each other at
the end: if they do not agree, the timings are not measuring the same problem and nothing else on
the row means anything.
"""

from __future__ import annotations

import json
import sys
import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def build_mesh(size, dim=3):
    """One mesh of the unit cube/square, generated once, shared by all three libraries."""
    import os

    import jno

    # Set JNO_BENCH_CACHE=1 to measure jNO's build with the persistent compilation cache on. Run
    # twice: the first run populates it, the second is the steady state a repeated workflow sees.
    if os.getenv("JNO_BENCH_CACHE"):
        jno.enable_compile_cache()

    if dim == 2:
        from shapely.geometry import box

        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=size)
        m = d.built_mesh
        return d, np.asarray(m.points, float)[:, :2], np.asarray(m.cells_dict["triangle"], int)
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    m = d.built_mesh
    return d, np.asarray(m.points, float), np.asarray(m.cells_dict["tetra"], int)


def on_boundary(pts, tol=1e-8):
    return np.any((pts < tol) | (pts > 1.0 - tol), axis=1)


# ---------------------------------------------------------------------------- jNO
def run_jno(d, dim=3):
    import jno

    t0 = time.perf_counter()
    if dim == 2:
        u, v = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
        fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        sol = np.asarray(fem.solve()).reshape(-1)
        return {
            "lib": "jNO",
            "solver": "Jacobi-BiCGStab (GPU)",
            "build_s": build,
            "solve_s": time.perf_counter() - t0,
            "u": sol,
        }
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    cb = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    fem = jno.fem(
        [ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(cb[0], cb[1], cb[2]) - 0.0],
        element_type="TET4",
    )
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol = np.asarray(fem.solve()).reshape(-1)
    solve = time.perf_counter() - t0
    return {"lib": "jNO", "solver": "Jacobi-BiCGStab (GPU)", "build_s": build, "solve_s": solve, "u": sol}


# ------------------------------------------------------------------------- JAX-FEM
def run_jax_fem(points, cells, dim=3):
    import jax.numpy as jnp
    from jax_fem.generate_mesh import Mesh
    from jax_fem.problem import Problem
    from jax_fem.solver import solver

    class Poisson(Problem):
        def get_tensor_map(self):
            return lambda x: x  # flux = grad u

        def get_mass_map(self):
            return lambda u, x: -jnp.ones(1)  # source f = 1

    # jax-fem calls the location fn on TRACED points, so it has to be jnp -- numpy conversion
    # raises TracerArrayConversionError. p is a single point of shape (dim,).
    def boundary(p):
        return jnp.any((p < 1e-8) | (p > 1.0 - 1e-8))

    ele = "TET4" if dim == 3 else "TRI3"
    t0 = time.perf_counter()
    mesh = Mesh(points, cells, ele_type=ele)
    prob = Poisson(
        mesh=mesh,
        vec=1,
        dim=dim,
        ele_type=ele,
        dirichlet_bc_info=[[boundary], [0], [lambda p: 0.0]],
    )
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol = solver(prob)
    solve = time.perf_counter() - t0
    return {
        "lib": "JAX-FEM",
        "solver": "jax_solver, preconditioned (GPU)",
        "build_s": build,
        "solve_s": solve,
        "u": np.asarray(sol[0]).reshape(-1),
    }


# ----------------------------------------------------------------------- scikit-fem
def run_skfem(points, cells, dim=3):
    import skfem
    from skfem.helpers import dot, grad
    from skfem.models.poisson import unit_load

    t0 = time.perf_counter()
    mesh = skfem.MeshTet(points.T, cells.T) if dim == 3 else skfem.MeshTri(points.T, cells.T)
    basis = skfem.Basis(mesh, skfem.ElementTetP1() if dim == 3 else skfem.ElementTriP1())

    @skfem.BilinearForm
    def laplace(u, v, _):
        return dot(grad(u), grad(v))

    A, b = laplace.assemble(basis), unit_load.assemble(basis)
    dofs = basis.get_dofs()
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    sol = skfem.solve(*skfem.condense(A, b, D=dofs))
    solve = time.perf_counter() - t0
    return {"lib": "scikit-fem", "solver": "scipy sparse direct (CPU)", "build_s": build, "solve_s": solve, "u": sol}


def main(sizes, dim=3):
    rows = []
    for size in sizes:
        d, points, cells = build_mesh(size, dim)
        n, ncell = len(points), len(cells)
        kind = "tets" if dim == 3 else "triangles"
        print(f"\n[{dim}-D] mesh size {size}:  {n:,} nodes, {ncell:,} {kind}  (generated once, shared)")
        got = {}
        for fn, args in (
            (run_jno, (d, dim)),
            (run_jax_fem, (points, cells, dim)),
            (run_skfem, (points, cells, dim)),
        ):
            try:
                r = fn(*args)
            except Exception as e:
                print(f"  {fn.__name__:<12} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:90]}")
                continue
            got[r["lib"]] = r["u"]
            print(
                f"  {r['lib']:<11} build {r['build_s']:7.2f}s  solve {r['solve_s']:7.2f}s"
                f"  total {r['build_s'] + r['solve_s']:7.2f}s   [{r['solver']}]"
            )
            rows.append({k: v for k, v in r.items() if k != "u"} | {"nodes": n, "cells": ncell, "size": size})
        # the gate: different answers mean the rows above are not comparable
        ref = got.get("scikit-fem")
        if ref is not None:
            for lib, u in got.items():
                if lib == "scikit-fem" or u.size != ref.size:
                    if u.size != ref.size:
                        print(f"    {lib}: {u.size} dofs vs scikit-fem {ref.size} -- NOT comparable")
                    continue
                rel = np.linalg.norm(u - ref) / max(np.linalg.norm(ref), 1e-300)
                verdict = "agrees" if rel < 1e-6 else "DISAGREES"
                print(f"    {lib} vs scikit-fem: relative difference {rel:.2e}  ({verdict})")
    out = f"benchmarks/compare_libs_{dim}d.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dim = 2 if args and args[0] == "2d" else 3
    sizes = [float(a) for a in (args[1:] if args and args[0] in ("2d", "3d") else args)]
    main(sizes or [0.08], dim)
