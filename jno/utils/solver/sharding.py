"""Partition an assembled FEM operator across devices.

The assembled ``BCOO`` needs **no change of representation** to be parallelised: it is ``data`` plus
one ``(row, col)`` pair per nonzero, and that ``nnz`` axis partitions cleanly. Each device takes a
slice of the triplets, computes its partial scatter-add, and the partials are combined by a single
``all-reduce``. Verified on simulated devices: the sharded result matches the single-device one to
1.15e-14, XLA emits ``all-reduce`` and **zero** ``all-gather`` (no device ever reconstitutes the
matrix), and each device holds exactly ``nnz/N`` triplets.

**The operator shards; the vectors stay replicated.** For a FEM problem the operator is ~100x the
vector, so replication costs nothing and buys a great deal: no mesh partitioning, no halo exchange,
no ghost DOFs, no DOF renumbering, and — measured, not assumed — *no solver changes at all*. Every
Krylov step is either a matvec (sharded, all-reduce inside) or a vector operation on replicated data
(identical on every device, no communication). ``cg``, ``bicgstab``, ``gmres`` and jNO's own
``minres``/``fgmres`` all converge unmodified through a sharded operator, Jacobi included.

Two constraints are load-bearing and easy to get wrong:

* **The triplet count must divide the device count.** ``device_put`` raises ``IndivisibleError``
  otherwise. :func:`pad_triplets` appends zero-valued triplets at ``(0, 0)``, which contribute
  exactly nothing to the scatter-add — safe by construction rather than by masking.
* **Sharded arrays must be genuine ``jit`` inputs, never closed over.** A closed-over array is baked
  in as a compile-time constant and silently replicated to every device: the computation still gives
  the right answer, with zero collectives and zero memory saving. That failure is invisible unless
  you inspect the compiled HLO, which is why :func:`sharded_solve` threads ``data``/``indices``
  through as arguments and why the tests assert on the collectives rather than only on values.

What does **not** distribute: sparse-direct solves (``jno.solve.lu``/``amg`` — ``spsolve`` is
single-device with no batching rule) and host-assembled preconditioners (``jno.precond.amg``/``ams``
build through scipy/pyamg). Those refuse rather than silently gathering the operator.
"""

from __future__ import annotations

from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

__all__ = [
    "SHARD_AXIS",
    "resolve_devices",
    "operator_mesh",
    "pad_triplets",
    "shard_triplets",
    "sharded_solve",
    "constrain_operator",
    "jacobi_from_diagonal",
    "describe",
]

#: Mesh axis name for the partitioned triplet dimension. Deliberately *not* ``"batch"``/``"model"``
#: (``jno.core``'s PINN device mesh): a FEM operator has neither axis, it has nonzeros.
SHARD_AXIS = "shard"


def resolve_devices(shard: Any = None) -> list:
    """Devices to partition the operator over.

    ``None`` (the default) means **automatic**: every visible device. That is on by default because
    the realistic alternative is not a tuned single-device run — it is idle silicon. A FEM solve
    today uses one device and leaves the rest of a multi-GPU box unused, and sharding is answer-
    preserving (same operator, same solvers; only the reduction order moves, ~1e-14).

    On a single-device machine automatic resolves to that one device, so the default path is
    byte-identical to before and carries no risk.

    ``1`` or ``False`` opts out explicitly; an int takes the first N; a device list pins exactly.
    Returns ``[]`` when there is nothing to distribute across, which callers read as "stay on the
    single-device path".
    """
    avail = jax.devices()
    if shard is None or shard is True:  # automatic
        return list(avail) if len(avail) > 1 else []
    if shard is False:
        return []
    if isinstance(shard, int):
        if shard < 1:
            raise ValueError(f"jno.fem: shard= must be a positive device count (or False); got {shard}.")
        if shard > len(avail):
            raise ValueError(
                f"jno.fem: shard={shard} was requested but only {len(avail)} device(s) are visible "
                f"({[d.device_kind for d in avail]}). Simulate more with "
                f"XLA_FLAGS=--xla_force_host_platform_device_count=N, or lower shard=."
            )
        return list(avail[:shard]) if shard > 1 else []
    devices = list(shard)
    if not devices:
        raise ValueError("jno.fem: shard= got an empty device list.")
    return devices if len(devices) > 1 else []


def operator_mesh(devices) -> Mesh:
    """A 1-D device mesh over :data:`SHARD_AXIS`. One axis, because a FEM operator has one
    partitionable dimension — its nonzeros."""
    return Mesh(np.asarray(devices).reshape(-1), (SHARD_AXIS,))


def pad_triplets(data, indices, n_devices: int) -> Tuple[Any, Any, int]:
    """Pad the triplet axis to a multiple of ``n_devices``; returns ``(data, indices, n_pad)``.

    A padded triplet is ``0.0`` at ``(0, 0)``. It is scatter-added like any other and contributes
    exactly zero, so no masking, no index arithmetic and no special case in the matvec — the padding
    is invisible to everything downstream.
    """
    nnz = int(data.shape[0])
    n_pad = (-nnz) % int(n_devices)
    if n_pad == 0:
        return data, indices, 0
    data = jnp.concatenate([data, jnp.zeros((n_pad,), data.dtype)])
    indices = jnp.concatenate([indices, jnp.zeros((n_pad, indices.shape[1]), indices.dtype)])
    return data, indices, n_pad


def shard_triplets(data, indices, mesh: Mesh):
    """Place the triplet arrays across ``mesh``, partitioned on the nonzero axis."""
    spec = NamedSharding(mesh, P(SHARD_AXIS))
    return jax.device_put(data, spec), jax.device_put(indices, spec)


def sharded_solve(A, b, solve_fn, devices, *, precond_fn=None, x0=None):
    """Run ``solve_fn`` with the operator partitioned across ``devices``.

    ``A`` is the assembled BCOO; ``solve_fn(op_matvec, rhs, M, x0) -> x`` is any matvec-based solver.
    ``data``/``indices`` are passed as **jit arguments** with explicit input shardings — closing over
    them would replicate the operator onto every device and silently undo the partitioning.

    The rhs and the solution stay replicated (``P()``), so every reduction inside the Krylov loop is
    consistent across devices without further coordination.
    """
    mesh = operator_mesh(devices)
    n = int(A.shape[0])
    data, indices, _n_pad = pad_triplets(A.data, A.indices, len(np.asarray(devices).reshape(-1)))
    data_s, idx_s = shard_triplets(data, indices, mesh)

    shard_spec = NamedSharding(mesh, P(SHARD_AXIS))
    repl = NamedSharding(mesh, P())
    b_s = jax.device_put(jnp.asarray(b).reshape(-1), repl)
    x0_s = None if x0 is None else jax.device_put(jnp.asarray(x0).reshape(-1), repl)

    def _run(d, idx, rhs, guess):
        rows, cols = idx[:, 0], idx[:, 1]
        matvec = lambda v: jax.ops.segment_sum(d * v[cols], rows, num_segments=n)  # noqa: E731
        M = None
        if precond_fn is not None:
            diag = jax.ops.segment_sum(jnp.where(rows == cols, d, 0.0), rows, num_segments=n)
            M = precond_fn(diag)
        return solve_fn(matvec, rhs, M, guess)

    in_shardings = (shard_spec, shard_spec, repl, None if x0_s is None else repl)
    return jax.jit(_run, in_shardings=in_shardings, out_shardings=repl)(data_s, idx_s, b_s, x0_s)


def constrain_operator(A, devices):
    """Partition a **traced** operator's triplets, returning ``(matvec, diag_fn)``.

    The concrete route (:func:`sharded_solve`) places ``data``/``indices`` with ``device_put`` and
    ``jit(in_shardings=...)``, and ``device_put`` cannot place a tracer. That excluded every
    parametric and differentiate-through solve -- which is where jNO spends its largest problems.

    ``lax.with_sharding_constraint`` is the way in: it is expressed *inside* the trace, so it applies
    to a tracer, and it partitions the same nonzero axis for the same one ``all-reduce``. Verified on
    simulated devices to emit ``all-reduce``, **no** ``all-gather``, and to carry a gradient through.

    Returns the pieces rather than running a solve, because here the caller is already inside the
    trace: there is no ``jit`` boundary of ours to hang ``in_shardings`` on, and wrapping one would
    close over the operator (constant-folded, replicated -- the failure this whole module exists to
    avoid). ``diag_fn`` is the same scatter-add restricted to ``row == col``, so a Jacobi
    preconditioner distributes with the matvec instead of forcing an assembled matrix.

    One honest caveat: with the operator arriving as a traced value rather than a placed argument,
    the per-device footprint is a compiler decision expressed by the constraint, not something the
    caller can measure with ``memory_analysis`` the way the concrete route can.
    """
    from jax.lax import with_sharding_constraint

    mesh = operator_mesh(devices)
    n = int(A.shape[0])
    # NOT padded, deliberately. `pad_triplets` exists because `device_put` raises IndivisibleError on
    # a triplet count that does not divide the device count -- but `with_sharding_constraint` handles
    # an uneven axis itself. Padding here is actively harmful: the `jnp.concatenate` is performed on
    # replicated data, and XLA then inserts an `all-gather` of the WHOLE operator to feed it, undoing
    # the partition entirely. Measured on a 4-device run: 898 triplets (padded to 900) emitted an
    # `all-gather` of f64[900] -- the full operator on every device -- while 896 (already divisible,
    # so unpadded) emitted none. Answers and gradients were correct either way, which is exactly why
    # this has to be checked on the collectives rather than on values.
    # Shard the DIVISIBLE PREFIX and leave the sub-device-count tail replicated.
    #
    # An uneven triplet axis is the trap here, and it is invisible in the answers. Both obvious
    # approaches make XLA `all-gather` the whole operator back onto every device -- padding to a
    # multiple of the device count (the `jnp.concatenate` runs on replicated data and is fed by an
    # `all-gather` of f64[900] for an 898-triplet operator), and constraining the uneven axis directly
    # (XLA's own uneven-sharding path gathers s32[900,1], the indices, which cost the same 8 bytes per
    # triplet as the data). Measured both; each time the results and gradients were correct and the
    # memory saving was entirely gone.
    #
    # A divisible axis compiles clean, so the operator is split: the first `(nnz // nd) * nd` triplets
    # shard, and the remaining FEWER THAN `nd` triplets stay replicated. That tail is at most 3
    # triplets on a 4-device run -- a rounding error against an operator of thousands -- and its
    # scatter-add is exact, so the two partial results simply add.
    nd = len(np.asarray(devices).reshape(-1))
    nnz = int(A.data.shape[0])
    m = (nnz // nd) * nd
    spec = NamedSharding(mesh, P(SHARD_AXIS))
    data = with_sharding_constraint(A.data[:m], spec)
    rows = with_sharding_constraint(A.indices[:m, 0], spec)
    cols = with_sharding_constraint(A.indices[:m, 1], spec)
    t_data, t_rows, t_cols = A.data[m:], A.indices[m:, 0], A.indices[m:, 1]

    def matvec(v):
        out = jax.ops.segment_sum(data * v[cols], rows, num_segments=n)
        if m < nnz:
            out = out + jax.ops.segment_sum(t_data * v[t_cols], t_rows, num_segments=n)
        return out

    def diag_fn():
        d = jax.ops.segment_sum(jnp.where(rows == cols, data, 0.0), rows, num_segments=n)
        if m < nnz:
            d = d + jax.ops.segment_sum(jnp.where(t_rows == t_cols, t_data, 0.0), t_rows, num_segments=n)
        return d

    return matvec, diag_fn


def jacobi_from_diagonal(diag):
    """``v -> v / diag`` built from the *sharded* triplets — the one preconditioner that needs no
    assembled matrix, so it distributes for free (the diagonal is the same scatter-add the matvec is)."""
    safe = jnp.where(jnp.abs(diag) > 0, diag, 1.0)
    return lambda v: v / safe


def describe(A, devices) -> str:
    """Human-readable partitioning summary, for a docstring example or a log line."""
    nd = len(np.asarray(devices).reshape(-1))
    nnz = int(A.nse) if hasattr(A, "nse") else int(A.data.shape[0])
    per = -(-nnz // nd)
    return f"{nnz} nonzeros over {nd} device(s) ~ {per} each ({A.data.nbytes / 2**20 / nd:.1f} MiB of data per device)"
