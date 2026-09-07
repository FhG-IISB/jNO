"""A transient march must refuse a step whose Newton solve did not converge.

The per-step driver already knows how to refuse a stalled solve -- ``newton_krylov``'s
``_convergence_check`` -- but it needs a *concrete* residual, so it disables itself under a trace, and
every step of the march runs inside ``lax.scan``. That is precisely where it cannot fire, so until
this guard the march returned the capped iterate as a perfectly plausible-looking trajectory.

Two things make the silence worse than the usual "no verdict" case, and both were measured on a
coupled melt-pool model:

* the failure is not reproducible -- two runs of the same command gave peak melt velocities of
  2.98 and 33.5 m/s, because an unconverged iterate is whatever the step cap happened to leave;
* a DIVERGED march is *faster* than a healthy one. Once the residual is NaN the loop condition
  ``||r|| > tol`` is False, so Newton exits on iteration one. The usual "it got slow, something is
  wrong" signal is inverted.

``jno/utils/solver/history_march.py`` already does this for the load-path march; this is the same
check, on the same numbers, for the time march.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nonlinear=True, mesh_size=0.34, time=(0.0, 0.1, 6)):
    """Transient heat with a u^3 reaction -- nonlinear, so the step is a Newton solve."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=time)
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    react = ui**3 * vi if nonlinear else 0.0
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y + react, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


@pytest.mark.parametrize("direct", [False, True], ids=["matrix-free", "sparse-direct"])
def test_a_step_capped_transient_march_raises(direct):
    """One Newton step against a 1e-14 tolerance cannot be a root. The march must say so."""
    fem = _heat()
    capped = jno.solve.newton(max_steps=1, rtol=1e-14, atol=1e-14, direct=direct)
    with pytest.raises(RuntimeError, match=r"did not converge"):
        np.asarray(fem.solve(nonlinear=capped).fn())


def test_the_refusal_names_the_step_and_the_numbers():
    """The message has to be actionable: which step, what residual, against what bound."""
    fem = _heat()
    with pytest.raises(RuntimeError) as e:
        np.asarray(fem.solve(nonlinear=jno.solve.newton(max_steps=1, rtol=1e-14, atol=1e-14)).fn())
    msg = str(e.value)
    assert "step" in msg and "residual" in msg, msg
    assert "t=" in msg, f"the failing time must be named: {msg}"


def test_a_converged_march_is_untouched():
    """A preconditioner-style rule: the guard changes what is REFUSED, never what is returned."""
    fem = _heat()
    ref = np.asarray(fem.solve().fn())
    got = np.asarray(fem.solve(nonlinear=jno.solve.newton(rtol=1e-10, atol=1e-10)).fn())
    assert np.isfinite(got).all()
    assert np.abs(got - ref).max() < 1e-7, "the guard must not perturb a healthy march"


def test_a_linear_transient_march_is_not_judged():
    """A linear step is a linear solve, not a Newton loop -- it has its own guard and must not
    acquire a second one that could refuse a perfectly good trajectory."""
    fem = _heat(nonlinear=False)
    got = np.asarray(fem.solve().fn())
    assert np.isfinite(got).all() and np.abs(got).max() > 1e-3


def test_the_guard_self_disables_under_a_trace():
    """Under an outer transform the residual norms are themselves traced, so the test cannot
    concretise. It must no-op rather than raise a ConcretizationTypeError -- the same trade the rest
    of jNO makes: under a transform the solver's own iteration cap is all there is.

    Pinned with ``jit`` because that is the cheapest way to hand the check a tracer; ``grad`` of a
    runtime-parametric march reaches it by the same route.
    """
    fem = _heat()
    got = np.asarray(jax.jit(lambda: fem.solve().fn())())
    assert np.isfinite(got).all(), "a traced march must still run"


def test_a_capped_march_under_a_trace_is_not_silently_rescued():
    """The flip side, stated so it is not mistaken for a promise: inside a trace the guard is off, so
    a capped solve still returns its non-root. This is the documented limit, not an oversight."""
    fem = _heat()
    capped = jno.solve.newton(max_steps=1, rtol=1e-14, atol=1e-14)
    got = np.asarray(jax.jit(lambda: fem.solve(nonlinear=capped).fn())())
    ref = np.asarray(fem.solve(nonlinear=jno.solve.newton(rtol=1e-10, atol=1e-10)).fn())
    # Not just "it did not raise": the returned trajectory must actually BE the non-root, or this
    # test would pass equally well if the traced path had quietly converged and there were no limit
    # to document. Same capped solve raises eagerly (test above); under the trace it comes back.
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() > 1e-6, (
        "the traced capped march returned something indistinguishable from the converged answer; "
        "the documented limit would then not be real"
    )


def test_the_bdf2_march_is_judged_too():
    """BDF2 has its OWN integrate (a two-level carry, and a backward-Euler startup step outside the
    scan), so it does not inherit the theta march's guard and needs its own."""
    fem = _heat()
    capped = jno.solve.newton(max_steps=1, rtol=1e-14, atol=1e-14)
    with pytest.raises(RuntimeError, match=r"did not converge"):
        np.asarray(fem.solve(nonlinear=capped, time=jno.solve.bdf2()).fn())


def test_the_bdf2_startup_step_is_covered_by_the_drivers_own_guard():
    """BDF2's backward-Euler startup step runs OUTSIDE the scan, so its iterate is concrete and the
    DRIVER's eager check fires on it directly -- the march needs no second guard there. Pinned
    because it is the reason the startup step is not judged alongside the scan's steps."""
    fem = _heat()
    capped = jno.solve.newton(max_steps=1, rtol=1e-14, atol=1e-14)
    with pytest.raises(RuntimeError, match=r"newton_krylov did not converge in max_steps=1"):
        np.asarray(fem.solve(nonlinear=capped, time=jno.solve.bdf2()).fn())


def test_bdf2_on_a_converged_problem_is_untouched():
    fem = _heat()
    got = np.asarray(fem.solve(nonlinear=jno.solve.newton(rtol=1e-10, atol=1e-10), time=jno.solve.bdf2()).fn())
    ref = np.asarray(fem.solve().fn())
    assert np.isfinite(got).all()
    # BDF2 is second order and backward Euler first, so at dt = 0.02 they differ by O(dt) -- measured
    # 0.039 against a peak of 0.79. The point here is only that the guard let both through unchanged;
    # the order study itself lives in tests/test_fem_time_schemes.py.
    assert np.abs(got - ref).max() < 0.06, "BDF2 and backward Euler should agree to their order gap"
