"""Under-relaxing the contact search — ``jno.solve.contact(relax=...)``.

The re-pair loop iterates ``u_{k+1} = G(u_k)`` with ``G = solve . repair``. That is a contraction only
while the pairing barely feeds back into the solution. A **follower** contact normal
(``variable(..., follow_normals=True)``) closes the loop — the normal is a function of ``u``, and the
traction it carries moves ``u`` — and the iteration can then stop contracting outright. Measured on a
sheet drawn over a die radius, undamped: the pairing had *settled* (0 slots re-paired for four rounds
running) while ``|du|/|u|`` sat at 5.0e-3, 1.2e-2, 7.0e-3, 8.2e-3 with no downward trend, so no number
of extra rounds could have helped.

``relax`` damps the update to ``u <- (1-relax) u_k + relax G(u_k)``. This does **not** move the fixed
point — where ``G(u) = u`` the blend is the identity — so the tests below pin exactly that: the same
answer, reached under damping when the undamped iteration cannot reach it at all.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver.contact_search import ContactSpec, run_contact_solve
from tests.test_fem_contact_search import _stacked_bars


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ----------------------------------------------------------------------------------------------
# The knob itself
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_a_relax_outside_the_unit_interval_is_refused(bad):
    """0 would freeze the search and >1 over-shoots past the solved iterate; both are named, not
    silently clamped."""
    _, fem = _stacked_bars()
    with pytest.raises(ValueError, match="relax must lie in"):
        fem.solve(contact=jno.solve.contact(relax=bad))


def test_the_default_is_bit_identical_to_not_passing_it():
    """`relax` must be inert until asked for: this is an existing-behaviour guard, so it asserts bit
    equality rather than closeness."""
    _, fem_a = _stacked_bars()
    _, fem_b = _stacked_bars()
    a = np.asarray(fem_a.solve(contact=jno.solve.contact())).reshape(-1)
    b = np.asarray(fem_b.solve(contact=jno.solve.contact(relax=1.0))).reshape(-1)
    assert a.shape == b.shape
    assert np.array_equal(a, b), f"relax=1.0 moved the answer by {np.abs(a - b).max():.3e}"


def test_damping_does_not_move_the_fixed_point():
    """The physical oracle. Where the undamped iteration converges, the damped one must converge to the
    SAME field -- damping changes the path, not the root."""
    _, fem_a = _stacked_bars()
    _, fem_b = _stacked_bars()
    full = np.asarray(fem_a.solve(contact=jno.solve.contact(tol=1e-6, rounds=40))).reshape(-1)
    half = np.asarray(
        fem_b.solve(contact=jno.solve.contact(tol=1e-6, rounds=40, relax=0.5))
    ).reshape(-1)
    scale = max(float(np.abs(full).max()), 1e-30)
    assert np.abs(full - half).max() / scale < 1e-4, (
        f"damping moved the fixed point by {np.abs(full - half).max() / scale:.3e} relative"
    )


# ----------------------------------------------------------------------------------------------
# The oracle with an exact answer: a round map that is NOT a contraction
# ----------------------------------------------------------------------------------------------
class _FakeOp:
    """The minimum surface `run_contact_solve` drives, wrapping a chosen scalar round map."""

    contact_pairs = (("s", "m"),)
    residual = staticmethod(lambda u, args=None: u)
    jacobian = None

    def __init__(self, n):
        self.size = n
        # a constant payload: the PAIRING never moves, so convergence rests purely on |du| -- which is
        # the quantity `relax` acts on, and the only one this test is about.
        self._tb = {"s": {"ids_full": np.zeros((1, 1, 1), int), "w_full": np.ones((1, 1, 1))}}

    def repair_contact(self, u, capture=None):
        return {k: dict(v) for k, v in self._tb.items()}


class _FakeFem:
    """`u <- A u + b` as the round map. |A| > 1 makes the undamped iteration diverge; relaxing by r
    turns it into `(1-r) + r A`, which contracts once r is small enough."""

    def __init__(self, A, b, n=4):
        self._op = _FakeOp(n)
        self._A, self._b, self._n = A, b, n
        self.calls = 0

    def _solve_dispatch(self, solve_fn=None, x0=None, **kw):
        self.calls += 1
        u_prev = np.zeros(self._n) if x0 is None else np.asarray(x0).reshape(-1)
        return self._A * u_prev + self._b


def test_a_non_contracting_round_map_converges_only_when_damped():
    """`A = -1.5` has |A| > 1, so the plain iteration walks away from a root that plainly exists.

    The fixed point is exact: u* = b/(1-A) = 1.0/2.5 = 0.4. Undamped the loop must RAISE rather than
    return; at relax=0.5 the effective multiplier is (1-0.5) + 0.5*(-1.5) = -0.25, and it converges to
    that same 0.4.
    """
    A, b = -1.5, 1.0
    exact = b / (1.0 - A)

    undamped = _FakeFem(A, b)
    with pytest.raises(RuntimeError):
        run_contact_solve(undamped, ContactSpec(rounds=25, tol=1e-8, relax=1.0))

    damped = _FakeFem(A, b)
    u = run_contact_solve(damped, ContactSpec(rounds=200, tol=1e-10, relax=0.5))
    assert np.allclose(u, exact, atol=1e-6), f"converged to {np.asarray(u)[:1]}, exact is {exact}"


def test_the_damped_iterate_is_the_stated_blend():
    """The algebra, pinned on the sequence itself rather than only on its limit.

    Round 1 has no previous iterate to blend against, so it is necessarily UNDAMPED: `u_1 = G(0) = b`.
    Every round after it is `u_{k+1} = (1-r) u_k + r (A u_k + b)`. The test reproduces that recurrence
    and demands the loop's own trajectory match it term by term, so a sign slip, an off-by-one, or a
    blend applied to the wrong operand is caught -- none of which the limit alone would reveal.
    """
    A, b, r = -1.5, 1.0, 0.5
    seen = []
    fem = _FakeFem(A, b)
    orig = fem._op.repair_contact
    fem._op.repair_contact = lambda u, capture=None, _o=orig, _s=seen: (_s.append(np.asarray(u).copy()), _o(u, capture))[1]

    run_contact_solve(fem, ContactSpec(rounds=60, tol=1e-12, relax=r))

    # seen[0] is the pre-round seed at u = 0; the rounds proper start at seen[1]
    expect, traj = np.zeros(4), []
    for k in range(len(seen) - 1):
        g = A * expect + b
        expect = g if k == 0 else (1.0 - r) * expect + r * g   # round 1 has nothing to blend against
        traj.append(expect.copy())
    for k, (got, want) in enumerate(zip(seen[1:], traj)):
        assert np.allclose(got, want, atol=1e-12), f"round {k + 1}: {got[:1]} vs {want[:1]}"


# ----------------------------------------------------------------------------------------------
# The LOAD PATH runs its own copy of the loop -- relax has to reach that one too
# ----------------------------------------------------------------------------------------------
def test_the_load_path_march_honours_relax():
    """A march does not go through `run_contact_solve`; `_march_eager_contact` re-implements the loop
    per load step. That duplication is exactly where a new knob gets silently dropped -- and did: the
    first version of `relax` reached the steady driver only, and a damped sheet-forming march came back
    with round-for-round BIT-IDENTICAL numbers to the undamped one.

    So this asserts the march's own trajectory actually changes, and still lands on the same answer.
    """
    from tests.test_fem_contact_search import _al_contact_march

    _, fem_a, _ = _al_contact_march()
    _, fem_b, _ = _al_contact_march()
    full = np.asarray(fem_a.solve(contact=jno.solve.contact(rounds=30, tol=1e-5)))
    half = np.asarray(fem_b.solve(contact=jno.solve.contact(rounds=30, tol=1e-5, relax=0.5)))
    assert full.shape == half.shape
    assert not np.array_equal(full, half), "relax=0.5 left the march bit-identical: it never arrived"
    scale = max(float(np.abs(full).max()), 1e-30)
    assert np.abs(full - half).max() / scale < 1e-3, (
        f"damping moved the marched answer by {np.abs(full - half).max() / scale:.3e} relative"
    )


@pytest.mark.parametrize("bad", [0.0, 1.5])
def test_the_march_refuses_a_relax_outside_the_unit_interval(bad):
    from tests.test_fem_contact_search import _al_contact_march

    _, fem, _ = _al_contact_march()
    with pytest.raises(ValueError, match="relax must lie in"):
        fem.solve(contact=jno.solve.contact(relax=bad))


# ----------------------------------------------------------------------------------------------
# The oscillation diagnostic must not fire on slow-but-monotone convergence
# ----------------------------------------------------------------------------------------------
def test_slow_monotone_convergence_is_not_reported_as_oscillation():
    """`_stalled` asked only whether the recent best beat the earlier best by 2x. A geometric decay
    slower than that ratio -- r = 0.90 over 12 rounds, which still reduces |du| by 0.31x and never once
    goes back up -- was therefore reported as a LIMIT CYCLE.

    That matters most for exactly the case `relax=` exists to fix: damping multiplies the contraction
    factor, so a correctly damped solve converges monotonically but slowly, and the old test would tell
    a caller who had already damped to "damp the iteration with relax=0.5". A cycle is distinguished by
    going back UP, not by going down slowly.
    """
    from jno.utils.solver.contact_search import _stalled

    for r in (0.90, 0.95, 0.99):
        hist = [r**k for k in range(12)]
        assert not _stalled(hist, 1e-4), f"monotone decay at r={r} reported as oscillating"


def test_a_real_limit_cycle_is_still_caught():
    """The other side of the same coin -- loosening the test must not blind it. Both sequences here
    were measured, not invented: the first from a sheet-forming march, the second from the gear sweep."""
    from jno.utils.solver.contact_search import _stalled

    forming = [5.0e-3, 1.2e-2, 7.0e-3, 8.2e-3, 5.0e-3, 1.2e-2, 7.0e-3, 8.2e-3]
    gears = [2.2e-2, 1.3e-2, 1.4e-2, 2.0e-2, 2.2e-2, 1.3e-2, 1.4e-2, 2.0e-2]
    assert _stalled(forming, 1e-4), "the measured forming limit cycle is no longer caught"
    assert _stalled(gears, 1e-4), "the measured gear limit cycle is no longer caught"


def test_a_flat_history_is_still_caught():
    """Not contracting at all, and not rising either -- the degenerate case between the two above. It
    is not converging, so it must still raise rather than run out the rounds silently."""
    from jno.utils.solver.contact_search import _stalled

    assert _stalled([1e-2] * 10, 1e-4), "a perfectly flat history is not converging"
