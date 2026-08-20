"""Bound constraints — ``u.bounds(lo, hi)`` as a term in the ``jno.fem`` list.

A bound on an unknown is part of the *problem statement*, the inequality sibling of a Dirichlet
condition, so it goes in the term list and ``fem.solve()`` still takes nothing. The solve becomes a
**variational inequality**: the KKT conditions replace ``R(u) = 0`` with

    lo < u < hi  ->  R = 0        u = lo  ->  R >= 0        u = hi  ->  R <= 0

which is exactly the root of the min-map ``min(max(R, u - hi), u - lo)`` (Facchinei & Pang 2003).

Oracles:
* **the classic obstacle problem, solved against its ANALYTIC free boundary.** ``-u'' = -1`` on
  ``(0,1)`` with ``u(0)=u(1)=0`` and ``u >= -c`` has the closed-form solution: the membrane leaves the
  obstacle at ``a = sqrt(2c)``, is the parabola ``x²/2 - a x`` on ``[0, a]``, and rests flat at ``-c``
  on ``[a, 1-a]``. The free boundary is what makes this a real test — see the control below.
* **a solve is not a clip.** The same problem solved unconstrained and then clipped to ``[-c, inf)``
  puts the contact set in a *visibly different place*: clipping detaches where the parabola crosses
  ``-c`` (``x = (1-sqrt(1-8c))/2``), the true solution where it does so *tangentially* (``x =
  sqrt(2c)``). At ``c = 1/18`` that is 0.127 vs 0.333 — a factor 2.6 no rounding can explain.
* **complementarity** — at the converged solution the residual vanishes off the contact set and has
  the correct one sign on it (the KKT multiplier), checked directly through ``fem.eval``.
* **coupled** — a box on one block leaves the other alone, each keeping its own oracle.
* **irreversibility** — ``u.bounds(u.i(-1), None)`` on a ``domain(tau=...)`` march ratchets: the field
  holds its peak while the load falls, where the unbounded control returns to zero. The bound is an
  *expression* re-read every step, which is the case the whole design exists for.
* **fail-loud** — an unsupported assembly route, a contradictory box, a bound with no sides, a bound
  on something that is not a solved field, and ``u.i(-1)`` without a load path.

Deliberately NOT tested here: a bound on the phase-field damage energy. That energy is non-convex in
``(u, d)`` jointly and a monolithic Newton is not expected to converge on it — that is the staggered
driver's job, and the test belongs with it. Measured while writing this: the divergence a bounded
damage form shows is caused by the *degradation floor*, not the bound — at ``d = 1`` exactly,
``(1-d)²`` makes the displacement block singular, so the floor is a well-posedness requirement that a
bound does not remove. Both are needed.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    """These tests run in float64. The session default is x64-off (see tests/conftest.py), and this
    flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


import jno

C = 1.0 / 18.0  # obstacle depth -> free boundary at a = sqrt(2c) = 1/3
A_FREE = np.sqrt(2.0 * C)


def _aliases():
    n = jno.np
    return n.grad, n.inner


def _obstacle_exact(x, c=C):
    """Analytic solution of ``-u'' = -1``, ``u(0)=u(1)=0``, ``u >= -c``.

    Off the contact set the membrane is the parabola through the origin that meets ``-c``
    *tangentially* at ``a = sqrt(2c)``; between ``a`` and ``1-a`` it rests on the obstacle."""
    a = np.sqrt(2.0 * c)
    x = np.asarray(x)
    left = 0.5 * x**2 - a * x
    right = 0.5 * (1 - x) ** 2 - a * (1 - x)
    return np.where(x < a, left, np.where(x > 1 - a, right, -c))


def _obstacle_fem(*, bounded, ny=0.25, nx=0.05):
    """``-Δu = -1`` on a strip, clamped at x=0,1 and NATURAL (zero-flux) on y — so the solution is
    x-only and equals the 1-D obstacle problem. ``bounded`` adds ``u >= -C``.

    Returns ``(fem, domain, equilibrium_term)`` — the term is handed back so a readout can assemble
    the very same expression the solve used (a fresh ``fem_symbols()`` would be a different field)."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, ny, size=nx).domain()
    d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    equilibrium = inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi  # -Δu = -1
    terms = [equilibrium, u(*ce) - 0.0]
    if bounded:
        terms.append(u.bounds(-C, None))  # one-sided: an obstacle from below
    return jno.fem(terms), d, equilibrium


# --------------------------------------------------------------------------------------------------
# Oracle 1 — the obstacle problem against its analytic solution AND its analytic free boundary.
# --------------------------------------------------------------------------------------------------
def test_obstacle_problem_matches_the_analytic_free_boundary():
    fem, _d, _eq = _obstacle_fem(bounded=True)
    sol = np.asarray(fem.solve())
    x = np.asarray(fem.points)[:, 0]

    # The bound is respected to machine precision — a solve, not a penalty.
    assert sol.min() > -C - 1e-12, f"the obstacle was violated by {(-C - sol.min()):.3e}"

    # The solution matches the closed form everywhere.
    exact = _obstacle_exact(x)
    err = np.abs(sol - exact).max()
    assert err < 2e-3, f"max |u - u_exact| = {err:.3e}"

    # The FREE BOUNDARY is where the answer is decided: the contact set must be [a, 1-a].
    on = np.abs(sol + C) < 1e-9  # nodes resting on the obstacle
    assert on.any(), "nothing made contact — the bound never activated"
    lo_edge, hi_edge = x[on].min(), x[on].max()
    assert abs(lo_edge - A_FREE) < 0.03, f"left detachment at {lo_edge:.4f}, analytic {A_FREE:.4f}"
    assert abs(hi_edge - (1 - A_FREE)) < 0.03, f"right detachment at {hi_edge:.4f}, analytic {1 - A_FREE:.4f}"


# --------------------------------------------------------------------------------------------------
# Oracle 2 — the control that makes oracle 1 mean something: a solve is NOT a clip. The unconstrained
# solution clipped to the box satisfies the bound just as exactly, and is the WRONG answer — it
# detaches where the parabola CROSSES the obstacle rather than where it meets it tangentially.
# --------------------------------------------------------------------------------------------------
def test_the_bound_is_solved_not_clipped():
    fem_b, _d, _eq = _obstacle_fem(bounded=True)
    fem_u, _d2, _eq2 = _obstacle_fem(bounded=False)
    x = np.asarray(fem_b.points)[:, 0]
    solved = np.asarray(fem_b.solve())
    clipped = np.maximum(np.asarray(fem_u.solve()), -C)

    # Both respect the bound...
    assert solved.min() > -C - 1e-12 and clipped.min() > -C - 1e-12
    # ...but the clip's contact set starts where the parabola CROSSES -c, which is a different place.
    cross = 0.5 * (1.0 - np.sqrt(1.0 - 8.0 * C))  # x²/2 - x/2 = -c
    assert abs(cross - A_FREE) > 0.15, "the two predictions must differ enough to discriminate"
    c_on, s_on = np.abs(clipped + C) < 1e-9, np.abs(solved + C) < 1e-9
    assert abs(x[c_on].min() - cross) < 0.03, "the clipped control did not detach at the crossing"
    assert abs(x[s_on].min() - A_FREE) < 0.03, "the solve did not detach at the tangency"
    # And they disagree by far more than discretization error.
    assert np.abs(solved - clipped).max() > 0.2 * C


# --------------------------------------------------------------------------------------------------
# Oracle 3 — complementarity. The bound is enforced through the KKT conditions, so at the solution the
# free residual must vanish off the contact set and carry one consistent sign on it (the multiplier —
# the obstacle's upward reaction). Read with `fem.eval`, which assembles the term un-eliminated.
# --------------------------------------------------------------------------------------------------
def test_kkt_complementarity_holds_at_the_solution():
    fem, _d, equilibrium = _obstacle_fem(bounded=True)
    sol = np.asarray(fem.solve())
    x = np.asarray(fem.points)[:, 0]
    R = np.asarray(fem.eval(equilibrium, sol))  # the SAME term the solve used, un-eliminated

    on = np.abs(sol + C) < 1e-9
    ends = (x < 1e-9) | (x > 1 - 1e-9)  # Dirichlet rows carry their own reaction — excluded
    free = ~on & ~ends
    scale = np.abs(R).max()
    assert scale > 1e-6, "the residual is identically zero — nothing to check"
    assert np.abs(R[free]).max() < 1e-9 * max(scale, 1.0), "equilibrium is violated OFF the contact set"
    # On the contact set the obstacle pushes back: the multiplier has one sign, and it is nonzero.
    assert (R[on] > -1e-9).all(), "the contact reaction changed sign — that is not a KKT multiplier"
    assert R[on].max() > 1e-6, "the contact set carries no reaction at all"


# --------------------------------------------------------------------------------------------------
# Oracle 4 — the bound on a COUPLED system. The box must apply to exactly one block and leave the other
# alone, so each block keeps an independent oracle: the bounded field is the obstacle solution above,
# and the field it drives is that same solution's own downstream solve. Kept convex on purpose — the
# non-convex phase-field energy is a *solver* problem (monolithic Newton is not expected to converge on
# it; that is what the staggered driver is for), not a statement about the bound.
# --------------------------------------------------------------------------------------------------
def test_a_bound_on_one_block_of_a_coupled_system():
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.05).domain()
    d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
    X = [co[0], co[1]]
    a, phi = d.fem_symbols()
    b, chi = d.fem_symbols()
    fem = jno.fem(
        [
            inner(grad(a, X), grad(phi, X), 1) + 1.0 * phi,  # the obstacle problem, on `a`
            inner(grad(b, X), grad(chi, X), 1) - 20.0 * a * chi,  # `b` is driven by `a`, one way
            a.bounds(-C, None),
            a(*ce) - 0.0,
            b(*ce) - 0.0,
        ]
    )
    sol = np.asarray(fem.solve())
    sa, sb = fem.blocks[fem.block_index(a)], fem.blocks[fem.block_index(b)]
    ua, ub = sol[sa], sol[sb]
    x = np.asarray(fem.field_points[fem.block_index(a)])[:, 0]

    # The bounded block is exactly the single-field obstacle solution, free boundary and all.
    assert ua.min() > -C - 1e-12
    assert np.abs(ua - _obstacle_exact(x)).max() < 2e-3
    assert abs(x[np.abs(ua + C) < 1e-9].min() - A_FREE) < 0.03

    # The UNBOUNDED block is untouched by the box — it runs well past `-C`, which it could not do if
    # the bound had leaked across blocks. (The coupling is scaled so this is not vacuous.)
    assert ub.min() < -C, "the second block picked up the first block's bound"
    # b solves -Δb = 20a with b=0 at the ends. Its discrete residual must vanish at every free DOF.
    Rb = np.asarray(fem.eval(inner(grad(b, X), grad(chi, X), 1) - 20.0 * a * chi, sol))
    free_b = np.ones(Rb.shape[0], dtype=bool)
    free_b[sa] = False
    ends = np.abs(np.asarray(fem.field_points[fem.block_index(b)])[:, 0] - 0.5) > 0.5 - 1e-9
    free_b[sb] = ~ends
    assert np.abs(Rb[free_b]).max() < 1e-9, "the driven block is not in equilibrium"


# --------------------------------------------------------------------------------------------------
# Oracle 5 — ``u.bounds(u.i(-1), None)`` on a load-path march: bound-constrained IRREVERSIBILITY, the
# motivating case. The bound is an *expression* re-read every step (the previous load level), not a
# constant. Ramp the load up then back down: bounded, the field ratchets and holds its peak; the
# control follows the load back down. Convex, so the march converges and the A/B is about the bound.
# --------------------------------------------------------------------------------------------------
def _ratchet_march(*, bounded, nstep=9):
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.08).domain(tau=(0.0, 1.0, nstep))
    d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    ramp = 1.0 - jno.np.abs(2 * tau - 1.0)  # 0 -> 1 -> 0
    terms = [
        inner(grad(u, X), grad(phi, X), 1) - 10.0 * ramp * phi + 0.0 * s.i(-1) * phi,
        s.evolves(s.i(-1)),  # an inert state: its only job is to trigger the march
        u(*ce) - 0.0,
    ]
    if bounded:
        terms.append(u.bounds(u.i(-1), None))  # u may never decrease
    return np.asarray(jno.fem(terms).solve())


def test_bound_constrained_irreversibility_on_a_march():
    ratchet, control = _ratchet_march(bounded=True), _ratchet_march(bounded=False)
    peak_r = np.abs(ratchet).max(axis=1)
    peak_c = np.abs(control).max(axis=1)
    assert peak_c.max() > 1e-3, "the control never deflected — nothing to compare"

    # Rising branch: the bound is slack (u grows), so the two must agree.
    mid = ratchet.shape[0] // 2
    assert np.abs(ratchet[mid] - control[mid]).max() / peak_c.max() < 1e-8, "the bound bit while loading"
    # Falling branch: the control returns to zero, the ratchet holds its peak.
    assert peak_c[-1] / peak_c.max() < 1e-6, "the control did not unload"
    assert peak_r[-1] / peak_r.max() > 0.999, "the ratchet did not hold its peak"
    # Monotone non-decreasing — the defining property of the constraint.
    assert np.all(np.diff(peak_r) > -1e-9)


def test_a_history_bound_needs_a_load_path():
    """`u.i(-1)` means *the previous load step*, so on a plain domain it refers to nothing."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.1).domain()  # NO tau grid
    d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(*ce) - 0.0, u.bounds(u.i(-1), None)])
    with pytest.raises(ValueError, match="tau|previous load step"):
        fem.solve()


# --------------------------------------------------------------------------------------------------
# Oracle 5 — a bound makes the problem a VARIATIONAL INEQUALITY even when the operator is linear, so a
# linear form carrying a bound must route to the residual path rather than assembling as a matrix pair.
# The obstacle problem above is exactly that case; this pins the routing directly.
# --------------------------------------------------------------------------------------------------
def test_a_bound_makes_a_linear_form_solve_as_a_variational_inequality():
    fem_b, _d, _eq = _obstacle_fem(bounded=True)
    fem_u, _d2, _eq2 = _obstacle_fem(bounded=False)
    assert fem_u.is_linear, "the unbounded control must still be a plain linear solve"
    assert not fem_b.is_linear, "a bound must take the form off the linear assembly route"


# --------------------------------------------------------------------------------------------------
# Oracle 6 — extremes and fail-loud.
# --------------------------------------------------------------------------------------------------
def test_a_slack_bound_changes_nothing():
    """A box far outside the solution range must reproduce the unconstrained answer exactly — the
    min-map has to be the identity wherever no constraint is active."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.05).domain()
    d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    base = [inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi, u(*ce) - 0.0]
    free = np.asarray(jno.fem(base).solve())
    slack = np.asarray(jno.fem(base + [u.bounds(-1e3, 1e3)]).solve())
    assert np.abs(free - slack).max() / np.abs(free).max() < 1e-8


def test_a_fully_active_bound_pins_the_whole_field():
    """The degenerate extreme: a lower bound above the unconstrained solution everywhere. Every DOF is
    active, so the answer is the bound itself (the Dirichlet ends excepted)."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.05).domain()
    d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    sol = np.asarray(jno.fem([inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi, u(*ce) - 0.0, u.bounds(0.0, None)]).solve())
    assert sol.min() > -1e-12
    assert sol.max() < 1e-9, "the fully-active solution must be the bound itself"


def test_a_coordinate_expression_bound_matches_its_constant_equivalent():
    """A bound may be a coordinate expression, evaluated at the field's DOF points like a Dirichlet
    value. Spelled so it *is* the constant, it must produce the identical answer — which pins the
    evaluation path itself rather than a second analytic solution."""
    grad, inner = _aliases()

    def run(spatial):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.05).domain()
        d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
        co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
        X = [co[0], co[1]]
        u, phi = d.fem_symbols()
        lo = (-C + 0.0 * X[0]) if spatial else -C  # identical value, one built as an expression
        return np.asarray(
            jno.fem([inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi, u(*ce) - 0.0, u.bounds(lo, None)]).solve()
        )

    const, expr = run(False), run(True)
    assert np.abs(const - expr).max() < 1e-12, "the expression path disagreed with the constant it equals"
    assert np.abs(const + C).min() < 1e-9, "the bound never activated — the comparison is vacuous"


def test_a_tilted_obstacle_shifts_the_contact_set():
    """A genuinely varying obstacle, ``psi = -C(0.5 + x)``: it sits HIGHER on the left (``-0.5C``) than
    on the right (``-1.5C``), so it obstructs the membrane more there and the contact set shifts left.
    Checks feasibility pointwise against the *expression's own* values, not a single number."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.05).domain()
    d.tag("ends", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    co, ce = d.variable("interior", split=True), d.variable("ends", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) + 1.0 * phi, u(*ce) - 0.0, u.bounds(-C * (0.5 + X[0]), None)])
    sol = np.asarray(fem.solve())
    x = np.asarray(fem.points)[:, 0]
    psi = -C * (0.5 + x)
    assert (sol - psi).min() > -1e-12, f"the tilted obstacle was violated by {-(sol - psi).min():.3e}"
    on = np.abs(sol - psi) < 1e-9
    assert on.any(), "no contact with the tilted obstacle"
    assert x[on].mean() < 0.5 - 0.02, "contact did not shift towards the side the obstacle protrudes on"
    # ...and the asymmetry is genuine: the symmetric-obstacle contact set is centred by construction.
    fem_c, _d, _eq = _obstacle_fem(bounded=True)
    sc = np.asarray(fem_c.solve())
    xc = np.asarray(fem_c.points)[:, 0]
    assert abs(xc[np.abs(sc + C) < 1e-9].mean() - 0.5) < 0.02


def test_a_bound_may_not_depend_on_the_live_unknown():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.1).domain()
    u, _phi = d.fem_symbols()
    v, _chi = d.fem_symbols()
    with pytest.raises(ValueError, match="complementarity|live unknown"):
        u.bounds(v, None)
    with pytest.raises(ValueError, match="complementarity|live unknown"):
        u.bounds(None, 2.0 * u)


def test_bounds_on_something_that_is_not_a_field_fails_loud():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.1).domain()
    co = d.variable("interior", split=True)
    u, _phi = d.fem_symbols()
    with pytest.raises(TypeError, match="fem_symbols|field"):
        (u * u).bounds(0.0, 1.0)
    with pytest.raises(TypeError, match="fem_symbols|field"):
        co[0].bounds(0.0, 1.0)


def test_contradictory_box_fails_loud():
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.1).domain()
    co = d.variable("interior", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    with pytest.raises(ValueError, match="lo|below|above|empty"):
        jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u.bounds(1.0, 0.0)])


def test_bounds_needs_at_least_one_side():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 0.25, size=0.1).domain()
    u, _phi = d.fem_symbols()
    with pytest.raises(ValueError, match="lo|hi|at least"):
        u.bounds(None, None)
