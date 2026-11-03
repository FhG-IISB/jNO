"""A repeated load-path march reuses its compiled program.

Each ``fem.solve()`` of a history march used to trace and compile the whole ``lax.scan`` (and the
adaptive pilot's step) again, so a second identical solve cost about as much as the first. The oracle is
structural: the number of XLA compilations a repeated solve triggers, and that the answer is unchanged.
A solver that differs in any setting must not pick up the first one's program.

Repeated solves are compared to round-off, not bit for bit: under the test session's allocator two
identical eager marches already differ in the last bits (about 1e-17 on a 0.05 displacement).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest
from jax._src import monitoring

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


class _Compiles:
    """Counts XLA backend compilations while active."""

    def __init__(self):
        self.n = 0
        self.on = False
        monitoring.register_event_duration_secs_listener(self._hear)

    def _hear(self, name, _duration, **_kw):
        if self.on and "backend_compile" in name:
            self.n += 1

    def __call__(self, fn):
        self.n, self.on = 0, True
        try:
            out = np.asarray(jax.block_until_ready(fn()))
        finally:
            self.on = False
        return out, self.n


_COUNT = _Compiles()


def _fem():
    """A tiny AT2 march: elasticity + damage + history + a box, bottom clamped, top pulled."""
    n = jno.np
    dom = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(tau=(0.0, 1.0, 4))
    dom.tag("bot", lambda x, y: y < 1e-9)
    dom.tag("top", lambda x, y: y > 1 - 1e-9)
    co, cb, ct = (dom.variable(r, split=True) for r in ("interior", "bot", "top"))
    X = [co[0], co[1]]
    u, dm, H = dom.unknown.vector(2, name="u"), dom.unknown.scalar(name="d"), dom.unknown.scalar(name="H")
    phi, q = u.test(), dm.test()
    e = n.sym(n.grad(u, X))
    tr = n.trace(e)
    dev = e - tr / 2 * n.identity(2)
    psi = 0.5 * n.maximum(tr, 0.0) ** 2 + n.inner(dev, dev, 2)
    sig = n.diff(((1.0 - dm) ** 2 + 1e-6) * psi, e)
    fem = jno.fem(
        [
            n.inner(sig, n.sym(n.grad(phi, X)), 2),
            dm * q + 0.01 * n.inner(n.grad(dm, X), n.grad(q, X), 1) - 2.0 * (1.0 - dm) * H.i(-1) * q,
            H.evolves(n.maximum(H.i(-1), psi)),
            dm.bounds(0.0, 1.0),
            u(*cb)[0] - 0.0,
            u(*cb)[1] - 0.0,
            u(*ct)[0] - 0.0,
            u(*ct)[1] - 0.05 * ct[-1],
        ]
    )
    return fem, u, dm


@pytest.mark.parametrize("adaptive", [False, True], ids=["fixed", "adaptive"])
def test_a_repeated_march_compiles_nothing_and_returns_the_same_trajectory(adaptive):
    fem, u, dm = _fem()
    alt = jno.solve.staggered([u, dm], max_sweeps=30)
    kw = {"tau": jno.solve.adaptive(limit=[(dm, 0.5)], max_steps=50)} if adaptive else {}
    first, n_first = _COUNT(lambda: fem.solve(nonlinear=alt, **kw))
    again, n_again = _COUNT(lambda: fem.solve(nonlinear=alt, **kw))
    assert n_first > 0, "the first solve must compile, or the counter hears nothing"
    assert n_again == 0, f"the repeated march compiled {n_again} program(s) again"
    np.testing.assert_allclose(again, first, rtol=0, atol=1e-14)
    # An equal spec built anew is the same solver (its value identity), so it reuses the program too.
    fresh, n_fresh = _COUNT(lambda: fem.solve(nonlinear=jno.solve.staggered([u, dm], max_sweeps=30), **kw))
    assert n_fresh == 0
    np.testing.assert_allclose(fresh, first, rtol=0, atol=1e-14)


def test_a_different_solver_setting_gets_its_own_program():
    """A cache keyed too loosely would hand the second solver the first one's program: a silent wrong
    answer. A loose tolerance stops the sweeps early, so its trajectory must differ from the tight one."""
    fem, u, dm = _fem()
    full, _ = _COUNT(lambda: fem.solve(nonlinear=jno.solve.staggered([u, dm], max_sweeps=30)))
    loose, n_loose = _COUNT(lambda: fem.solve(nonlinear=jno.solve.staggered([u, dm], rtol=1e-1, atol=1e-1)))
    assert n_loose > 0, "a different solver setting must compile its own program"
    assert np.abs(loose - full).max() > 1e-4, "the loose solver returned the tight march: its program was reused"
    again, n_again = _COUNT(lambda: fem.solve(nonlinear=jno.solve.staggered([u, dm], max_sweeps=30)))
    assert n_again == 0
    np.testing.assert_allclose(again, full, rtol=0, atol=1e-14)


def test_a_reordered_or_renamed_sweep_is_a_different_solver():
    """`staggered` names its fields in its value identity. A `domain.unknown()` field used to print as a
    bound method (the same for every field) and two fields of one name printed alike, so
    `staggered([u, d])` and `staggered([d, u])` shared a key -- and the second sweep order ran the first's
    compiled program."""
    fem, u, dm = _fem()
    assert repr(jno.solve.staggered([u, dm])) != repr(jno.solve.staggered([dm, u]))
    dom = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
    a, _ = dom.fem_symbols(names=("w", "v_a"))
    b, _ = dom.fem_symbols(names=("w", "v_b"))
    assert repr(jno.solve.staggered([a, b])) != repr(jno.solve.staggered([b, a]))
    _COUNT(lambda: fem.solve(nonlinear=jno.solve.staggered([u, dm], max_sweeps=30)))
    _, n_swapped = _COUNT(lambda: fem.solve(nonlinear=jno.solve.staggered([dm, u], max_sweeps=30)))
    assert n_swapped > 0, "the swapped sweep order ran the first order's compiled march"
