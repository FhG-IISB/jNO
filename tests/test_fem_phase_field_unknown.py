"""A phase-field march written with ``domain.unknown()`` fields, and a crack seeded as a history field.

``domain.unknown()`` is meant to stand in for ``fem_symbols`` everywhere in ``jno.fem``: the same march
written either way must give the same trajectory. Two places did not lower it -- a bound
(``d.bounds``) and a state update (``H.evolves`` reading the displacement) -- and a coordinate inside an
``.evolves`` formula carried a phantom component axis, which is what an initial crack seeded as a
history field ``H0(x)`` needs (Borden, Verhoosel, Scott, Hughes & Landis, CMAME 217-220 (2012) 77-95).

The oracles are relationships between two spellings of one problem, not stored numbers.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    """Float64, restored afterwards (the session default is x64-off, see tests/conftest.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _march(spelling, *, bounds=True, load=0.01, seed=None):
    """A tiny AT2 march (unit square, bottom clamped, top pulled) -> its trajectory.

    ``spelling``: ``"symbols"`` (``fem_symbols``) or ``"unknown"`` (``domain.unknown``).
    ``seed``: ``None``; ``"evolves"`` puts an initial history ``H0(y)`` into the state update;
    ``"form"`` drives the damage by ``max(H.i(-1), H0)`` in the weak form instead."""
    n = jno.np
    dom = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(tau=(0.0, 1.0, 3))
    dom.tag("bot", lambda x, y: y < 1e-9)
    dom.tag("top", lambda x, y: y > 1 - 1e-9)
    co, cb, ct = (dom.variable(r, split=True) for r in ("interior", "bot", "top"))
    X = [co[0], co[1]]
    if spelling == "unknown":
        u, dm, H = dom.unknown.vector(2, name="u"), dom.unknown.scalar(name="d"), dom.unknown.scalar(name="H")
        phi, q = u.test(), dm.test()
    else:
        u, phi = dom.fem_symbols(value_shape=(2,), names=("u", "v_u"))
        dm, q = dom.fem_symbols(names=("d", "v_d"))
        H, _ = dom.fem_symbols(names=("H", "v_H"))
    e = n.sym(n.grad(u, X))
    tr = n.trace(e)
    dev = e - tr / 2 * n.identity(2)
    psi = 0.5 * n.maximum(tr, 0.0) ** 2 + n.inner(dev, dev, 2)
    sig = n.diff(((1.0 - dm) ** 2 + 1e-6) * psi, e)
    H0 = 10.0 * n.maximum(0.2 - n.abs(co[1] - 0.5), 0.0)
    drive = n.maximum(H.i(-1), H0) if seed == "form" else H.i(-1)
    update = n.maximum(H.i(-1), psi + H0) if seed == "evolves" else n.maximum(H.i(-1), psi)
    terms = [
        n.inner(sig, n.sym(n.grad(phi, X)), 2),
        dm * q + 0.01 * n.inner(n.grad(dm, X), n.grad(q, X), 1) - 2.0 * (1.0 - dm) * drive * q,
        H.evolves(update),
        u(*cb)[0] - 0.0,
        u(*cb)[1] - 0.0,
        u(*ct)[0] - 0.0,
        u(*ct)[1] - load * ct[-1],
    ]
    if bounds:
        terms.append(dm.bounds(0.0, 1.0))
    fem = jno.fem(terms)
    traj = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, dm], max_sweeps=30, rtol=1e-10, atol=1e-12)))
    return traj, fem.block_index(dm), fem


def test_unknown_with_bounds_matches_fem_symbols():
    """`d.bounds(0, 1)` on a `domain.unknown()` field is the same box as on a `fem_symbols` field."""
    ref, _, _ = _march("symbols")
    got, _, _ = _march("unknown")
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_state_update_reading_an_unknown_matches_fem_symbols():
    """`H.evolves(...)` reading a `domain.unknown()` displacement advances the same history."""
    ref, _, _ = _march("symbols", bounds=False)
    got, _, _ = _march("unknown", bounds=False)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)


def test_coordinate_in_a_state_update_seeds_the_history():
    """An initial history `H0(y)` written in the update equals the same `H0` written in the form.

    With no load the displacement stays zero, so `psi = 0` and the stored history is exactly `H0` after
    the first step: from the second step on, both spellings drive the damage by `H0` alone. (At step 0
    the update has not run yet, so only the form spelling sees `H0` there.)"""
    a, i_d, fem = _march("symbols", bounds=False, load=0.0, seed="evolves")
    b, _, _ = _march("symbols", bounds=False, load=0.0, seed="form")
    blk = fem.blocks[i_d]
    assert np.abs(a[1:, blk]).max() > 1e-3, "the seed must damage the band, or the test shows nothing"
    np.testing.assert_allclose(a[1:], b[1:], rtol=1e-10, atol=1e-12)
