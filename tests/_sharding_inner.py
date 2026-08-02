"""Inner assertions for the sharded-operator tests, run in a subprocess.

``XLA_FLAGS=--xla_force_host_platform_device_count=N`` is read when the JAX backend initialises, so
it must be set before the first ``import jax``. ``tests/conftest.py`` already imports jax, which
makes an in-module ``os.environ[...]`` useless in a full-suite run — hence a separate process.
Invoked by ``tests/test_fem_sharding.py``; not collected directly (no ``test_`` filename prefix).

Usage: ``python tests/_sharding_inner.py <n_devices>``
"""

from __future__ import annotations

import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.experimental.sparse as jsp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jno.utils.solver import krylov  # noqa: E402
from jno.utils.solver.sharding import jacobi_from_diagonal, pad_triplets, sharded_solve  # noqa: E402


def _spd_triplets(n=1500):
    """Diagonally dominant tridiagonal, emitted as unsorted triplets with duplicates — the shape a
    FEM assembler actually produces, not a tidy sorted matrix."""
    rows, cols, data = [], [], []
    for k in (-1, 0, 1):
        i = np.arange(max(0, -k), n - max(0, k))
        rows.append(i)
        cols.append(i + k)
        data.append(np.where(k == 0, 4.0, -1.0) * np.ones(len(i)))
    return (
        np.concatenate(rows).astype(np.int32),
        np.concatenate(cols).astype(np.int32),
        np.concatenate(data),
    )


def main(n_dev: int) -> None:
    devices = jax.devices()[:n_dev]
    assert len(devices) == n_dev, f"expected {n_dev} devices, saw {len(jax.devices())}"

    n = 1500
    rows, cols, data = _spd_triplets(n)
    A = jsp.BCOO((jnp.asarray(data), jnp.asarray(np.stack([rows, cols], 1))), shape=(n, n))
    rng = np.random.default_rng(0)
    b = jnp.asarray(rng.standard_normal(n))

    dense = np.zeros((n, n))
    for r, c, d in zip(rows, cols, data):
        dense[r, c] += d
    x_ref = np.linalg.solve(dense, np.asarray(b))
    rel = lambda x: float(np.linalg.norm(np.asarray(x) - x_ref) / np.linalg.norm(x_ref))  # noqa: E731

    # --- 1. every solver converges through a sharded operator, unmodified -----------------------
    solvers = {
        "cg": lambda mv, rhs, M, x0: jax.scipy.sparse.linalg.cg(mv, rhs, x0=x0, tol=1e-10, maxiter=3000, M=M)[0],
        "bicgstab": lambda mv, rhs, M, x0: jax.scipy.sparse.linalg.bicgstab(mv, rhs, x0=x0, tol=1e-10, maxiter=3000, M=M)[
            0
        ],
        "gmres": lambda mv, rhs, M, x0: jax.scipy.sparse.linalg.gmres(mv, rhs, x0=x0, tol=1e-10, maxiter=3000, M=M)[0],
        "minres": lambda mv, rhs, M, x0: krylov.minres(mv, rhs, M=M, x0=x0, tol=1e-10, maxiter=3000),
        "fgmres": lambda mv, rhs, M, x0: krylov.fgmres(mv, rhs, M=M, x0=x0, tol=1e-10, maxiter=3000),
    }
    for name, fn in solvers.items():
        x = sharded_solve(A, b, fn, devices, precond_fn=jacobi_from_diagonal)
        assert rel(x) < 1e-8, f"{name} on {n_dev} device(s): rel {rel(x):.3e}"
        assert x.sharding.spec == jax.sharding.PartitionSpec(), f"{name}: solution must stay replicated"

    # --- 2. the operator is genuinely distributed, never gathered -------------------------------
    # Values alone cannot show this: a closed-over (hence constant-folded, replicated) operator
    # gives identical answers with zero collectives and zero memory saving.
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from jno.utils.solver.sharding import SHARD_AXIS, shard_triplets

    mesh = Mesh(np.asarray(devices).reshape(-1), (SHARD_AXIS,))
    d_pad, i_pad, n_pad = pad_triplets(A.data, A.indices, n_dev)
    d_s, i_s = shard_triplets(d_pad, i_pad, mesh)
    per_device = d_s.addressable_shards[0].data.shape[0]
    assert per_device == d_pad.shape[0] // n_dev, f"expected an even split, got {per_device}"

    def _mv(d, idx, v):
        return jax.ops.segment_sum(d * v[idx[:, 1]], idx[:, 0], num_segments=n)

    repl = NamedSharding(mesh, P())
    shard_spec = NamedSharding(mesh, P(SHARD_AXIS))
    f = jax.jit(_mv, in_shardings=(shard_spec, shard_spec, repl), out_shardings=repl)
    hlo = f.lower(d_s, i_s, jax.device_put(b, repl)).compile().as_text()
    if n_dev > 1:
        assert "all-reduce" in hlo, "the partial scatter-adds must be combined by an all-reduce"
        assert "all-gather" not in hlo, "no device may reconstitute the whole operator"

    # --- 3. padding is invisible ---------------------------------------------------------------
    assert (n_pad == 0) or (int(d_pad.shape[0]) % n_dev == 0)
    got = np.asarray(f(d_s, i_s, jax.device_put(b, repl)))
    assert np.max(np.abs(got - dense @ np.asarray(b))) < 1e-10, "padded triplets perturbed the matvec"

    print(f"OK n_devices={n_dev} per_device_nnz={per_device} pad={n_pad}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
