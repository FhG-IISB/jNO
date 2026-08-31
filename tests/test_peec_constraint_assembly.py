"""The constraint block is assembled once, not once per node.

Current balance writes a row per free node, so ``vv`` carries thousands of entries on any real
lattice. Sending each through ``jnp.asarray(v).reshape(-1)`` costs two eager dispatches apiece --
and, on a GPU, two host-to-device transfers. Measured at 8,640 nodes over three warm solves that was
26,337 ``jnp.asarray`` calls and about 90 % of the solve, spent assembling a vector rather than
solving with it.

It is also why the GPU was no faster than the CPU: Python dispatch does not care which device it is
dispatching to. With the runs fused, the same solve went 0.801 s -> 0.317 s on GPU and started
beating the CPU 2x rather than losing to it.

The invariant worth pinning is not a time, which is machine-dependent, but the SCALING: the number
of eager array constructions must not grow with the node count.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import _fuse_triplets

jax.config.update("jax_enable_x64", True)

SIG, LX, WY, TZ = 5.8e7, 0.040, 0.004, 0.002


def _naive(parts):
    """What it used to do, kept as the oracle rather than as the implementation."""
    return jnp.concatenate([jnp.asarray(v, dtype=complex).reshape(-1) for v in parts])


def test_fusing_reproduces_the_per_piece_concatenation():
    rng = np.random.default_rng(0)
    parts = [
        np.array([1.0, -1.0]),
        np.array([2.5]),
        jnp.asarray([3.0 + 1j]),  # a traced-shaped piece: must stay on the jax side
        rng.normal(size=4),
        np.array([[7.0], [8.0]]),  # not already flat -- the reshape has to survive
        jnp.asarray([9.0]),
    ]
    assert np.allclose(np.asarray(_fuse_triplets(parts)), np.asarray(_naive(parts)))


def test_an_all_numpy_list_becomes_exactly_one_array():
    """The common case -- no device impedance, no placement weight -- must not touch jax per piece."""
    calls = {"n": 0}
    real = jnp.asarray

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    parts = [np.array([1.0, -1.0]) for _ in range(500)]
    jnp.asarray = counted
    try:
        out = _fuse_triplets(parts)
    finally:
        jnp.asarray = real
    assert calls["n"] == 1, f"{calls['n']} array constructions for 500 concrete pieces"
    assert out.shape == (1000,)


def test_an_empty_list_is_an_empty_vector_not_a_crash():
    out = _fuse_triplets([])
    assert out.shape == (0,) and jnp.iscomplexobj(out)


@pytest.mark.parametrize("pitch", [0.004, 0.002])
def test_a_solve_does_not_construct_arrays_per_NODE(pitch):
    """The scaling claim, asserted end to end: refining the mesh must not multiply the dispatches.

    A 2x refinement is at least 4x the nodes here -- the bar is thin, so the third direction runs
    out of cells before the other two do. If assembly were still per-row the count would follow that;
    fused, it is flat in everything but the handful of port rows.
    """
    counts = {}
    real = jnp.asarray

    for p in (pitch, pitch / 2):
        bar = jno.Shape.box(0, 0, 0, LX, WY, TZ, size=p).attach(sigma=SIG).name("bar")
        d = bar.domain()
        d.tag("A", lambda x, y, z: x < p)
        d.tag("B", lambda x, y, z: x > LX - p)
        _i, v = d.peec_symbols()
        at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
        built = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).build()
        built.solve().R.block_until_ready()  # warm the compile and the caches

        n = {"c": 0}

        def counted(*a, **k):
            n["c"] += 1
            return real(*a, **k)

        jnp.asarray = counted
        try:
            built.solve().R.block_until_ready()
        finally:
            jnp.asarray = real
        counts[p] = (n["c"], int(np.asarray(built.fil.nodes).shape[0]))

    (c_coarse, n_coarse), (c_fine, n_fine) = counts[pitch], counts[pitch / 2]
    assert n_fine >= 4 * n_coarse  # the refinement really is the one this reasons about
    # per-row assembly would grow with the nodes; fused, the count is flat
    assert c_fine < 2 * c_coarse, (
        f"{c_coarse} array constructions at {n_coarse} nodes but {c_fine} at {n_fine} "
        "-- assembly is scaling with the mesh again"
    )
