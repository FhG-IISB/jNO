"""Time-integration schemes via ``fem.solve(time=...)`` — ``jno.solve.theta`` and ``jno.solve.exponential``.

Oracle is transient heat with a fundamental-mode IC (``sin πx sin πy``), which decays as ``e^{-2π²t}``.
The defining property of the exponential integrator is that it is **exact in time**: its answer is
independent of the number of steps, where backward-Euler's is not — and it beats backward-Euler at a
fixed (coarse) step count. The θ-scheme test checks the override (θ=1 ≡ default, θ=½ differs, both valid).
"""

import importlib.util

import jax
import numpy as np
import pytest
from shapely.geometry import box

import jno

_HAS_MATFREE = importlib.util.find_spec("matfree") is not None
PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nsteps, T=0.03, h=0.11):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h, time=(0.0, T, nsteps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def _final(fem, **kw):
    return np.asarray(fem.solve(**kw).fn())[-1]


def test_theta_override():
    """θ=1 reproduces the default (backward-Euler); θ=½ (Crank–Nicolson) gives a different, valid answer."""
    default = _final(_heat(6))
    be = _final(_heat(6), time=jno.solve.theta(1.0))
    cn = _final(_heat(6), time=jno.solve.theta(0.5))
    assert np.allclose(be, default, atol=1e-10)  # θ=1 ≡ the assembly default
    assert not np.allclose(cn, default, atol=1e-4)  # θ=½ is a genuinely different scheme
    assert np.isfinite(cn).all()


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_is_exact_in_time():
    """The exponential integrator is *exact in time*: its answer does not depend on the step count, where
    backward-Euler's does. exp(2 steps) ≈ exp(8 steps); BE(2) is far from BE(8)."""
    e2 = _final(_heat(2), time=jno.solve.exponential(order=40))
    e8 = _final(_heat(8), time=jno.solve.exponential(order=40))
    assert np.linalg.norm(e2 - e8) / np.linalg.norm(e8) < 1e-5  # step-independent ⇒ exact in time

    b2, b8 = _final(_heat(2)), _final(_heat(8))
    assert np.linalg.norm(b2 - b8) / np.linalg.norm(b8) > 5e-2  # backward-Euler depends strongly on the step


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_beats_backward_euler():
    """At a fixed coarse step count the exponential integrator is much closer to the time-converged answer."""
    ref = _final(_heat(400))  # time-converged reference
    exp_err = np.linalg.norm(_final(_heat(4), time=jno.solve.exponential(order=40)) - ref) / np.linalg.norm(ref)
    be_err = np.linalg.norm(_final(_heat(4)) - ref) / np.linalg.norm(ref)
    assert exp_err < 0.5 * be_err  # exact-in-time wins at coarse steps


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_consistent_mass_is_more_accurate():
    """``mass='consistent'`` (Cholesky-factored M, no lumping error) is closer to the time-converged
    reference than ``mass='lumped'`` — and is still exact in time (step-independent)."""
    ref = _final(_heat(400))
    lump = np.linalg.norm(_final(_heat(4), time=jno.solve.exponential(mass="lumped")) - ref) / np.linalg.norm(ref)
    cons = np.linalg.norm(_final(_heat(4), time=jno.solve.exponential(mass="consistent")) - ref) / np.linalg.norm(ref)
    assert cons < lump  # consistent mass removes the lumping error

    c4 = _final(_heat(4), time=jno.solve.exponential(mass="consistent"))
    c8 = _final(_heat(8), time=jno.solve.exponential(mass="consistent"))
    assert np.linalg.norm(c4 - c8) / np.linalg.norm(c8) < 1e-6  # still exact in time


def test_time_scheme_rejects_a_steady_problem():
    """``time=`` selects a TIME integrator, so it is an error on a non-transient problem."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi])
    with pytest.raises(ValueError, match="transient"):
        fem.solve(time=jno.solve.theta(0.5))
