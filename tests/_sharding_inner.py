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

    # --- 4. the SLOT-COMPOSED path shards too, and the refuse-set falls back silently ------------
    # `compose_linear_solve_fn` is the single LinearOperator construction point for every
    # slot-composed linear solve, so one change there covers all the Krylov solvers. It could not be
    # a decorator: `composed` closes over the operator, while sharding needs data/indices as jit
    # ARGUMENTS -- so the call structure is inverted and the matvec rebuilt as a segment_sum inside.
    import jno
    from jno.utils.solver.solver_api import LinearOperator, _shardable, compose_linear_solve_fn

    op = LinearOperator(A)
    for label, linear, precond in (
        ("no precond", jno.solve.bicgstab(tol=1e-10, maxiter=3000), None),
        ("jacobi", jno.solve.bicgstab(tol=1e-10, maxiter=3000), jno.precond.jacobi()),
        ("cg", jno.solve.cg(tol=1e-10, maxiter=3000), jno.precond.jacobi()),
        ("gmres", jno.solve.gmres(tol=1e-10, maxiter=3000), None),
    ):
        assert _shardable(op, linear, precond), f"{label} should be shardable"
        x = compose_linear_solve_fn(linear, precond, None, None, shard=devices)(A, b)
        assert rel(x) < 1e-8, f"slot-composed {label} on {n_dev} device(s): rel {rel(x):.3e}"
        # Values alone CANNOT show the operator was distributed -- a closed-over (hence replicated)
        # operator gives identical answers with zero collectives. The output's device set can: the
        # sharded path returns on the multi-device mesh, the fallback on a single device.
        assert len(x.sharding.device_set) == n_dev, (
            f"slot-composed {label} produced a {len(x.sharding.device_set)}-device result on "
            f"{n_dev} devices -- it silently took the unsharded path"
        )

    # the refuse-set: each must fall back to the ordinary path, never raise
    assert not _shardable(op, jno.solve.lu(), None), "a direct solver has nothing to distribute"
    assert not _shardable(op, jno.solve.bicgstab(), jno.precond.chebyshev()), (
        "a preconditioner that closes over the operator would replicate it -- refuse"
    )
    assert not _shardable(LinearOperator.from_matvec(lambda v: v, shape=(n, n)), jno.solve.cg(), None), (
        "a matvec-only operator has no triplet axis"
    )
    x_direct = compose_linear_solve_fn(jno.solve.lu(), None, None, None, shard=devices)(A, b)
    assert rel(x_direct) < 1e-8, "the refuse-set must fall back silently, not fail"
    if n_dev > 1:
        assert len(x_direct.sharding.device_set) == 1, "a direct solve must NOT claim to be sharded"

    # --- 5. TRACED (parametric / differentiate-through) operators -------------------------------
    # `device_put` cannot place a tracer, so this route uses `with_sharding_constraint` from inside
    # the trace. Two traps here were invisible in the answers and only showed up in the HLO, so this
    # asserts on collectives: padding the triplet axis to a multiple of the device count made XLA
    # `all-gather` the whole operator to feed the concatenate, and constraining an UNEVEN axis made it
    # gather the index array. Hence the divisible-prefix + replicated-tail split.
    import jax.experimental.sparse as _jsp

    from jno.utils.solver.solver_api import compose_linear_solve_fn as _compose

    for nnz_off in (0, 1, 2, 3):  # exercise every remainder against the device count
        rr, cc, dd = _spd_triplets(n)
        keep = len(dd) - nnz_off
        idx_t = jnp.asarray(np.stack([rr[:keep], cc[:keep]], 1))
        dat_t = jnp.asarray(dd[:keep])
        dense_t = np.zeros((n, n))
        np.add.at(dense_t, (rr[:keep], cc[:keep]), dd[:keep])
        b_t = jnp.asarray(rng.standard_normal(n))
        ref_t = np.linalg.solve(dense_t, np.asarray(b_t))

        solver = _compose(jno.solve.bicgstab(tol=1e-12, maxiter=5000), jno.precond.jacobi(), None, None, shard=devices)
        f = lambda th, _i=idx_t, _d=dat_t, _b=b_t, _s=solver: _s(_jsp.BCOO((th * _d, _i), shape=(n, n)), _b)  # noqa: E731
        got = np.asarray(jax.jit(f)(jnp.asarray(1.0, dat_t.dtype)))
        assert np.linalg.norm(got - ref_t) / np.linalg.norm(ref_t) < 1e-8, f"traced solve, remainder {nnz_off}"

        hlo_t = jax.jit(f).lower(jnp.asarray(1.0, dat_t.dtype)).compile().as_text()
        if n_dev > 1:
            assert "all-reduce" in hlo_t, f"traced operator not partitioned (remainder {nnz_off})"
        assert "all-gather" not in hlo_t, (
            f"remainder {nnz_off}: the operator is being reconstituted on every device -- correct "
            f"answers, zero memory saving, which is the failure this whole module guards against"
        )
        # and the gradient must still reach the parameter through the sharded solve
        g = jax.grad(lambda th, _f=f: jnp.sum(_f(th)))(jnp.asarray(1.0, dat_t.dtype))
        assert np.isfinite(float(g)), f"gradient lost through the sharded traced solve (remainder {nnz_off})"

    # --- 6. AUTOMATIC must leave traced operators alone -----------------------------------------
    # Inside a trace we are a guest: a sharding constraint must agree with the device commitments of
    # every other value in that jit, and under `crux` it does not (parameters arrive committed to one
    # device). That conflict is undetectable in advance and surfaces when the OUTER jit compiles, so
    # there is no fallback to write -- automatic placement must therefore not touch traced operators.
    auto = _compose(jno.solve.bicgstab(tol=1e-12, maxiter=5000), jno.precond.jacobi(), None, None, shard=None)
    rr, cc, dd = _spd_triplets(n)
    idx_a, dat_a = jnp.asarray(np.stack([rr, cc], 1)), jnp.asarray(dd)
    b_a = jnp.asarray(rng.standard_normal(n))
    fa = lambda th: auto(_jsp.BCOO((th * dat_a, idx_a), shape=(n, n)), b_a)  # noqa: E731
    hlo_a = jax.jit(fa).lower(jnp.asarray(1.0, dat_a.dtype)).compile().as_text()
    assert "all-reduce" not in hlo_a, "automatic must not shard a TRACED operator -- it can break crux"

    print(f"OK n_devices={n_dev} per_device_nnz={per_device} pad={n_pad}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
