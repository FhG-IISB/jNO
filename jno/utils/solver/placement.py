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

__all__ = ["solve_device", "to_solve_device"]


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
            y = jax.device_put(x) if is_np else jnp.asarray(np.asarray(x))
        if memo is not None:
            memo[id(x)] = (x, y)
        return y

    return jax.tree_util.tree_map(move, tree)
