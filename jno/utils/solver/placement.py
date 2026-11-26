"""Where a finished operator lives: on the device that solves with it, moved there ONCE.

Assembly runs on the host by default (``_fem._host_assembly_scope``: the element loop's temporaries are
far larger than the matrix it produces, and would exhaust the card). The finished operator used to stay
there, so every solve copied it to the device again -- measured on a 3-D P1 Poisson problem (87k DOF,
RTX 3070): 11 of the 38 ms of a warm linear solve, and 30 of the 224 ms of a 69k-DOF, 20-step heat march,
were that host->device copy of arrays that had not changed since the previous call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["share_program_constants", "solve_device", "to_solve_device"]


def solve_device():
    """The device an uncommitted computation runs on: ``jax.default_device`` if set, else the first device."""
    d = jax.config.jax_default_device
    if isinstance(d, str):  # a platform name
        d = jax.devices(d)[0]
    return d if d is not None else jax.devices()[0]


def _movable(x, target) -> bool:
    # Only UNCOMMITTED arrays: that is how host assembly leaves them, and it means "no placement was
    # asked for". An array the caller committed to a device stays exactly where it was put.
    if not isinstance(x, jax.Array) or isinstance(x, jax.core.Tracer):
        return False
    if getattr(x, "committed", True):
        return False
    try:
        devs = x.devices()
    except Exception:  # a deleted array: leave it alone
        return False
    # Host arrays only: that is what host assembly produces, and it keeps `np.asarray` below zero-copy.
    return len(devs) == 1 and next(iter(devs)).platform == "cpu"


#: Above this, an array goes to the device a chunk at a time (see `upload`).
_UPLOAD_CHUNK_BYTES = 128 << 20


def upload(x):
    """``jnp.asarray(x)`` for a host array, in the least device memory: uncommitted, under the current default device.

    A transfer from the host takes a staging buffer as large as the array, rounded up to a power of two, out of the same
    pool: 80 MB took 214 MB at its peak (RTX 3070, measured with `memory_stats`). So it is finished before anything
    else is sent, or the staging buffers of a whole operator stack up, and a large array goes in chunks written in
    place into its final buffer, which caps the staging at one chunk."""
    a = np.asarray(x)
    if a.nbytes <= _UPLOAD_CHUNK_BYTES or a.ndim == 0:
        return jax.block_until_ready(jnp.asarray(a))
    step = max(1, _UPLOAD_CHUNK_BYTES // (a.nbytes // a.shape[0]))
    out = jnp.zeros(a.shape, a.dtype)
    write = jax.jit(jax.lax.dynamic_update_slice_in_dim, donate_argnums=0, static_argnums=3)
    for start in range(0, a.shape[0], step):
        out = jax.block_until_ready(write(out, jnp.asarray(a[start : start + step]), start, 0))
    return out


def to_solve_device(tree, memo: dict | None = None, *, numpy: bool = False):
    """``tree`` with every uncommitted HOST array moved to :func:`solve_device` (still uncommitted).

    A no-op on a CPU-only run. ``numpy=True`` moves numpy arrays too, for the constants of a traced program
    (``make_jaxpr`` hands back numpy for everything a closure captured as numpy -- measured: 27 of the 28
    constants of an 87k-DOF nonlinear solve, 10.6 MB copied from the host on every call). ``memo`` (``id(source) -> (source, moved)``) lets several cached programs
    that close over the same host arrays share ONE device copy instead of one each; the source is kept
    in it so its ``id`` cannot be reused while the entry lives.
    """
    target = solve_device()
    if target.platform == "cpu":
        return tree

    def move(x):
        is_np = numpy and isinstance(x, np.ndarray) and x.size > 1
        if not is_np and not _movable(x, target):
            return x
        if memo is not None:
            hit = memo.get(id(x))
            if hit is not None and hit[0] is x:
                return hit[1]
        # Through numpy, not `jax.device_put`: without a device that is a no-op on an uncommitted array,
        # and with one it COMMITS the result, after which a computation the caller scopes to another
        # device fails on mixed placement. A host array converts to numpy without a copy, and
        # `asarray` under the target scope makes an uncommitted array there -- placement stays implicit.
        with jax.default_device(target):
            y = upload(x)
        if memo is not None:
            memo[id(x)] = (x, y)
        return y

    return jax.tree_util.tree_map(move, tree)


def share_program_constants() -> None:
    """One copy on the device of each constant that compiled programs take as an argument.

    A program's constants reach it as arguments (``JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS``), each placed where the program
    runs. Assembly runs on the host (``_fem._host_assembly_scope``), so the tables its closures hold are host arrays,
    and JAX places them anew for every program: one 12 MB index table stood five times on the card (136 of 222 MB
    live in a design loop was such copies, 169 of 387 MB in another; shared, the same loops use 0.6 and 1.0 GiB less
    of the card). So a host array is placed once per content and
    placement, and every program that takes it shares that buffer; it goes with the last program that holds it. An array
    already on the program's devices is left to JAX, which reuses its buffer. Keyed by content, not identity: a numpy
    array can change between two compilations, and two arrays of one mesh table are one table."""
    import hashlib
    import weakref

    from jax._src.interpreters import pxla
    from jaxlib import xla_client as xc

    if getattr(pxla.MeshExecutable.shard_const_args, "_jno_shared", False):
        return
    placed = weakref.WeakValueDictionary()
    original = pxla.MeshExecutable.shard_const_args

    def on_host(x):
        if isinstance(x, np.ndarray):
            return True
        return isinstance(x, jax.Array) and all(d.platform == "cpu" for d in x.devices())

    def shard_const_args(self, const_args):
        # JAX asks again at every call of a program with a host callback, which skips its fast path: each array was
        # hashed each time, 35 of a 170 ms solve. A program's constants are fixed when it is traced.
        memo = getattr(self, "_jno_placed", None)
        if memo is not None and memo[0] is const_args:
            return memo[1]
        n = len(const_args)
        shardings, layouts = self._in_shardings[:n], self._xla_in_layouts[:n]
        if all(next(iter(s.device_set)).platform == "cpu" for s in shardings):  # the program is on the host itself
            return original(self, const_args)
        out = list(const_args)
        rest = []
        for i, (x, sharding, layout) in enumerate(zip(const_args, shardings, layouts)):
            if not on_host(x):
                rest.append(i)
                continue
            a = np.asarray(x)  # no copy: it is on the host
            key = (
                a.shape,
                a.dtype.str,
                bool(getattr(x, "weak_type", False)),
                jax.config.jax_enable_x64,
                sharding,
                str(layout),
                hashlib.sha1(np.ascontiguousarray(a).reshape(-1).view(np.uint8)).hexdigest(),
            )
            y = placed.get(key)
            if y is None:
                y = pxla.shard_args([sharding], [layout], [xc.ArrayCopySemantics.REUSE_INPUT], [x])[0]
                placed[key] = jax.block_until_ready(y)  # one staging buffer at a time (see `upload`)
            out[i] = y
        if rest:
            semantics = [xc.ArrayCopySemantics.REUSE_INPUT] * len(rest)
            others = pxla.shard_args(
                [shardings[i] for i in rest], [layouts[i] for i in rest], semantics, [const_args[i] for i in rest]
            )
            for i, y in zip(rest, others):
                out[i] = y
        self._jno_placed = (const_args, out)
        return out

    shard_const_args._jno_shared = True
    pxla.MeshExecutable.shard_const_args = shard_const_args
