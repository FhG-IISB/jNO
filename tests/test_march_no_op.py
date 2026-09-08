"""A transient march whose tolerance sits above the form's residual scale must refuse, not return
its initial condition dressed as an answer.

The trap this pins: `atol` is an ABSOLUTE floor. A weak form written in SI units can carry a step
residual far below an `atol` its author would call loose -- and then Newton's FIRST convergence test
passes at the incoming iterate, the driver returns without updating anything, and the march hands
back a finite, plausibly shaped trajectory that is exactly the initial state.

Reduced from a 2-D melt pool driven by a thermocapillary traction, where it presented as a converged
solve reporting a peak melt velocity of exactly 0.0 m/s.
"""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


inner, grad = jno.np.inner, jno.np.grad


def _tiny_residual_fem(nsteps=6):
    """A NONLINEAR transient diffusion whose step residual is ~1e-9 -- small purely by its units.

    `SCALE` multiplies the whole equation, so the root is untouched and only the residual NORM moves.
    That is the point: the physics is scale-free, the convergence test is not.
    """
    SCALE = 1e-9
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 1.0, nsteps + 1))
    u, w = d.fem_symbols(names=("u", "w"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, wi = u.bind(x=xi, y=yi, t=ti), w.bind(x=xi, y=yi, t=ti)
    # (1 + u^2) makes it nonlinear, so the march runs the Newton driver and the judge is active.
    return jno.fem(
        [
            SCALE * ((1.0 + ui**2) * ui.t * wi + inner(grad(u, [xi, yi]), grad(w, [xi, yi]), n_contract=1) - wi),
            u(xb, yb) - 0.0,
            u(*ci) - 0.0,
        ]
    )


def test_march_refuses_when_no_step_ever_solves():
    """atol above the residual scale -> every step "converges" on entry -> refuse by name."""
    fem = _tiny_residual_fem()
    assert not fem.is_linear, "the guard only runs on the nonlinear march path"
    with pytest.raises(RuntimeError, match="returned its INITIAL STATE unchanged"):
        fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-6, atol=1e-2), linear=jno.solve.lu(backend="host")).fn()


def test_the_same_march_solves_when_the_tolerance_fits_the_scale():
    """The control: the ONLY change is atol. It must now march, and to a non-trivial answer.

    Without this the guard could be passing for the wrong reason -- a march that refuses everything
    also "catches" the defect.
    """
    fem = _tiny_residual_fem()
    traj = np.asarray(
        fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-8, atol=1e-14), linear=jno.solve.lu(backend="host")).fn()
    )
    assert np.isfinite(traj).all()
    assert np.abs(traj[-1]).max() > 1e-3, "the source should have driven u well away from its zero IC"


def test_a_march_that_solves_anywhere_is_not_refused():
    """A loose-but-usable tolerance still marches: the guard keys on NO step moving, not on a
    generous atol. Pinned so the guard cannot creep into refusing ordinary loose solves."""
    fem = _tiny_residual_fem()
    traj = np.asarray(
        fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-6, atol=1e-13), linear=jno.solve.lu(backend="host")).fn()
    )
    assert np.abs(traj[-1]).max() > 1e-3
