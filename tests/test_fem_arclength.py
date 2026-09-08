"""``fem.solve(tau=jno.solve.arclength(...))`` — continuation that can pass a limit point.

Load control cannot follow an equilibrium path around a fold: past the peak there is no equilibrium at
a higher load, so no step size finds one, and the march fails. Arc-length advances along the path
instead of along the load, so the load factor may *decrease* while the solution keeps growing.

The oracle is analytic. Bratu's problem ``-Delta u = lam e^u`` on ``(0, 1)`` with ``u(0) = u(1) = 0``
has a closed-form load-deflection curve, parametrically in ``th``::

    lam(th) = th^2 / (2 cosh^2(th/4)),    u_max(th) = 2 ln cosh(th/4)

so its fold is ``max_th lam(th) = 3.5138307...`` (Bratu; Gelfand; Frank-Kamenetskii). The tests
*derive* that constant from the formula rather than hard-coding it, which is what makes it an
independent oracle rather than a restatement.

The strip is thin and the long sides carry the natural condition, so the solution is the 1-D one and
the 1-D constant applies.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

n = jno.np


@pytest.fixture(autouse=True)
def _x64():
    """x64: the bordered system is solved to 1e-8 and the fold is where the un-bordered tangent is
    singular — float32 cannot resolve either. The session default is x64-off (tests/conftest.py)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _analytic_bratu_fold():
    """``max_th lam(th)`` for the 1-D Bratu problem — computed from the closed form, not looked up."""
    th = np.linspace(1e-9, 20.0, 400_001)
    return float(np.max(th**2 / (2.0 * np.cosh(th / 4.0) ** 2)))


def _bratu(nsteps, *, hi=12.0, size=0.08):
    """``-Delta u = tau e^u`` on a thin strip, as a term list.

    The inert state field is what triggers the load-path march at all (`.i(k)` history plus a `tau=`
    grid); it is deliberately multiplied by zero so it cannot influence the answer — the same device
    `tests/test_fem_adaptive_load_path.py` uses.
    """
    d = jno.shape.rect(0.0, 0.0, 1.0, 0.1, size=size).domain(tau=(0.0, hi, nsteps))
    u, phi = d.fem_symbols()
    s, _sp = d.fem_symbols()
    co = d.variable("interior", split=True)
    xi, yi, ti = co[0], co[1], co[-1]
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    si = s.bind(x=xi, y=yi)
    return jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - ti * n.exp(ui) * vi + 0.0 * si.i(-1) * vi,
            s.evolves(si.i(-1)),
            u(xl, yl) - 0.0,
            u(xr, yr) - 0.0,
        ]
    )


def _linear_march(nsteps, *, hi=1.0, size=0.25):
    """A LINEAR load path: ``-Delta u = tau`` with the same inert-state march trigger."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(tau=(0.0, hi, nsteps))
    u, phi = d.fem_symbols()
    s, _sp = d.fem_symbols()
    co = d.variable("interior", split=True)
    xi, yi, ti = co[0], co[1], co[-1]
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    si = s.bind(x=xi, y=yi)
    return jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - ti * vi + 0.0 * si.i(-1) * vi,
            s.evolves(si.i(-1)),
            u(xb, yb) - 0.0,
        ]
    )


# ----------------------------------------------------------------------------------------------
# The headline: the fold is reached, and passed
# ----------------------------------------------------------------------------------------------


def test_arclength_traverses_the_bratu_fold():
    """The peak load factor must match the analytic fold, and then the path must TURN AROUND — the load
    factor decreasing while the solution keeps growing. That non-monotone lambda is the whole point:
    it is the thing load control provably cannot produce."""
    lam_c = _analytic_bratu_fold()
    fem = _bratu(40, hi=12.0)
    sol = np.asarray(fem.solve(tau=jno.solve.arclength()))
    lam = np.asarray(fem.tau_schedule)
    umax = np.abs(sol).max(axis=1)
    k = int(np.argmax(lam))

    assert abs(lam.max() - lam_c) / lam_c < 0.02, f"peak {lam.max():.5f} vs analytic {lam_c:.5f}"
    assert k < len(lam) - 1, "the march must continue PAST the peak, not stop at it"
    assert lam[-1] < lam.max() - 1e-3, "the load factor must come back down on the far branch"
    assert umax[-1] > umax[k], "...while the solution keeps growing"
    assert np.all(np.diff(umax) > 0), "the solution norm advances monotonically along the path"


def test_load_control_cannot_pass_the_fold():
    """The control that makes the test above mean something. The SAME problem over a declared span that
    contains the fold, marched under ordinary load control, must fail rather than return a path — there
    is no equilibrium up there to find."""
    fem = _bratu(40, hi=4.0)
    with pytest.raises(RuntimeError):
        fem.solve()


def test_the_peak_converges_to_the_analytic_fold_under_refinement():
    """A single mesh matching to 2% could be luck. The discretization error must SHRINK when the mesh
    is refined — that is what makes the analytic constant an oracle rather than a coincidence."""
    lam_c = _analytic_bratu_fold()
    errs = []
    for size in (0.16, 0.08):
        fem = _bratu(40, hi=12.0, size=size)
        fem.solve(tau=jno.solve.arclength())
        errs.append(abs(np.asarray(fem.tau_schedule).max() - lam_c))
    assert errs[1] < 0.6 * errs[0], f"peak error did not improve under refinement: {errs}"


# ----------------------------------------------------------------------------------------------
# It degrades gracefully
# ----------------------------------------------------------------------------------------------


def test_arclength_reproduces_the_uniform_grid_on_a_linear_problem():
    """With the default calibrated `ds`, a linear load path has a straight equilibrium path, so equal
    arcs are equal load increments: arc-length must reproduce the declared uniform grid, and the same
    trajectory the ordinary march gives. Exercises the calibration, the predictor and the constraint at
    once, against an exact answer."""
    nsteps = 6
    plain = np.asarray(_linear_march(nsteps).solve())
    fem = _linear_march(nsteps)
    arc = np.asarray(fem.solve(tau=jno.solve.arclength()))
    lam = np.asarray(fem.tau_schedule)

    assert np.allclose(lam, np.linspace(0.0, 1.0, nsteps), atol=1e-8), lam
    assert np.abs(arc - plain).max() < 1e-8


def test_the_load_factors_are_recorded_for_the_force_displacement_curve():
    """`fem.tau_schedule` is the observability channel: one load factor per output row, so it pairs
    directly with a reaction read off the trajectory."""
    fem = _bratu(12, hi=6.0, size=0.16)
    sol = np.asarray(fem.solve(tau=jno.solve.arclength()))
    lam = np.asarray(fem.tau_schedule)
    assert lam.shape == (sol.shape[0],)
    assert lam[0] == 0.0, "step 0 is the declared start, solved under load control"


# ----------------------------------------------------------------------------------------------
# Fail loud
# ----------------------------------------------------------------------------------------------


def test_staggered_is_refused_and_says_why():
    """Freezing the load factor while sweeping a block IS load control, which has no equilibrium past
    the fold — so this is a correctness refusal, not a missing feature."""
    fem = _bratu(6, hi=2.0, size=0.25)
    with pytest.raises(NotImplementedError, match="load control"):
        fem.solve(tau=jno.solve.arclength(), nonlinear=jno.solve.staggered([0]))


def test_a_direct_driver_is_refused_because_the_bordered_tangent_is_not_assembled():
    fem = _bratu(6, hi=2.0, size=0.25)
    with pytest.raises(NotImplementedError, match="direct"):
        fem.solve(tau=jno.solve.arclength(), nonlinear=jno.solve.newton(direct=True))


def test_too_few_steps_is_refused_with_the_reason():
    """Arc-length needs a load-controlled start and a step to set the direction before it can border
    anything, so a two-point path cannot work."""
    fem = _bratu(2, hi=2.0, size=0.25)
    with pytest.raises(ValueError, match="at least 3"):
        fem.solve(tau=jno.solve.arclength())


def test_a_degenerate_declared_span_is_refused_even_with_an_explicit_ds():
    """The declared span does two jobs: it calibrates the default arc length AND it establishes which
    way along the path to travel. A zero-width span leaves the direction undefined, and `ds=` cannot
    supply it — so the refusal must not be conditional on `ds` being absent. (It was, at first: with an
    explicit `ds` the secant predictor divided by a zero first increment and produced NaN.)"""
    for spec in (jno.solve.arclength(), jno.solve.arclength(ds=0.05)):
        fem = _bratu(6, hi=0.0, size=0.25)
        with pytest.raises(ValueError, match="zero width"):
            fem.solve(tau=spec)


def test_the_form_is_staged_once_across_the_arclength_march():
    """Arc length walks its steps on the HOST, so it can re-stage the form per step if it hands the
    solver a fresh closure each time — which is exactly what the contact march did (22-26 XLA
    compilations per round, each retained) and what `continuation` did before its step was jitted.
    Neither is visible in an answer; both only show up as time and memory.

    The invariant is that staging cannot scale with the number of steps. Asserted by comparing two
    step counts rather than against a magic number, so it stays true if the Newton body changes how
    many times it stages internally.
    """
    counts = []
    for nsteps in (5, 15):
        fem = _bratu(nsteps)
        op = fem._op
        real = op.residual
        tr = {"n": 0}

        def counting(u, *a, _r=real, _t=tr, **kw):
            if isinstance(u, jax.core.Tracer):  # a traced u means the form is being staged out
                _t["n"] += 1
            return _r(u, *a, **kw)

        op.residual = counting
        try:
            fem.solve(tau=jno.solve.arclength(ds=0.35))
        finally:
            op.residual = real
        counts.append(tr["n"])

    assert counts[0] > 0, "the residual was never staged — the counter is not wired to the solve"
    assert counts[1] <= counts[0], f"staging scales with the march: {counts[0]} traces for 5 steps, {counts[1]} for 15"
