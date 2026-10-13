"""The development problem for the harness: a 3-D Poisson gradient on the GPU-resident cuDSS path, a few thousand
tets, every stage the planar gradient has (assemble, factor, solve, adjoint, readout) in under a minute.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python dev/examples/small3d.py [--size 0.12] [--label small3d]
    python dev/kernels.py dev/runs/small3d --stage jno/solve -- python dev/examples/small3d.py

The design variable is every vertex x-coordinate (as in the planar problem), the objective the compliance u.b."""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
jax.config.update("jax_enable_x64", True)
import jno  # noqa: E402
import jno.utils.solver.cudss_sym as cs  # noqa: E402
from dev import prof  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=float, default=0.12, help="mesh size on the unit cube (0.12 -> ~4 k tets)")
ap.add_argument("--label", default="small3d")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--cpu", action="store_true", help="PARDISO on the host (run with JAX_PLATFORMS=cpu) instead of cuDSS")
a = ap.parse_args()

d = jno.shape.box(0, 0, 0, 1, 1, 1, size=a.size).name("box").attach(k=1.0).domain()
xi, yi, zi, *_ = d.variable("interior", split=True)
xb, yb, zb, *_ = d.variable("boundary", split=True)
xi.trainable(name="X")
u, v = d.fem_symbols()
ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
fem = jno.fem([d.k * (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z) - 1.0 * vi, u(xb, yb, zb) - 0.0])
op = fem.operator
P = np.asarray(d.mesh.points)
n_tets = int(len(d.mesh.cells_dict["tetra"]))
x0 = jnp.asarray(P[:, 0])
A0, b0 = jax.jit(lambda x: op.evaluate({"X": x}))(x0)
n = A0.shape[0]
plan = None if a.cpu else cs.plan(np.asarray(A0.indices[:, 0]), np.asarray(A0.indices[:, 1]), n)
print(f"{n_tets:,} tets, {n:,} DOFs -> {a.label}", flush=True)
if a.cpu:
    from jno.utils.solver.linear import pardiso_lu_solve


def compliance(x):
    A, b = op.evaluate({"X": x})
    uu = pardiso_lu_solve(A, b) if a.cpu else cs.solve(plan, A.data, b)
    with jax.named_scope("jno/readout"):
        return jnp.dot(uu, b)


from jno.utils import profiling as _p  # noqa: E402

_p.collect_stats(True)
r = prof.run(
    jax.jit(jax.value_and_grad(compliance)),
    x0,
    label=a.label,
    reps=a.reps,
    backend_stats=lambda: _p.stats() + ([] if a.cpu else [dict(backend="cudss_ffi", n=plan.n, **plan.stats())]),
    sizes=dict(cells=n_tets, nodes=len(P), nnz=int(A0.nse), n=n, nrhs=1, complex=False),
)
print(r.table())
print("saved", r.save())
