"""Parameter continuation — the INTERNAL ``run_continuation`` driver.

The public spelling is deliberately absent: the agreed home (2026-08-08) is a warm-started
``sequence`` axis in the ``tune`` space, wired to this driver. These tests pin the engine that
wiring will call — a ``fem.solve`` kwarg for it was built, reviewed, and removed.

One driver under three names: an EM material/frequency sweep (``keep="all"``), mechanics load
stepping, numerical homotopy. Every solve warm-starts from the previous one; a failing step names
its parameter values and index instead of returning NaN downstream.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
import jno.jnp_ops as J
from jno.utils.solver.solver_api import ContinuationSpec, run_continuation


def _spec(keep="last", **params):
    return ContinuationSpec(params=dict(params), keep=keep)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson_kap():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    u, v = d.fem_symbols()
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    kap = jno.np.parameter((1,), name="kap")
    q = jno.np.parameter((1,), name="q")
    return jno.fem([kap * (ui.x * vi.x + ui.y * vi.y) - (q + 0.0) * vi, u(xb, yb) - 0.0])


def test_a_linear_sweep_matches_independent_solves():
    """``keep="all"`` returns the family, and each member equals a cold solve at that value (a
    single-step continuation IS a cold solve, so the reference uses the same public entry)."""
    fem = _poisson_kap()
    ks = jnp.linspace(0.5, 2.0, 5)
    U = run_continuation(fem, _spec(kap=ks, keep="all"), kwargs={"q": jnp.array([1.0])})
    assert U.shape[0] == 5 and bool(jnp.isfinite(U).all())
    # member equality is checked under the DIRECT solve, where warm starts cannot move the answer
    # (the default Krylov agrees only to its tolerance, which is the solver's business, not the sweep's)
    U_lu = run_continuation(fem, _spec(kap=ks, keep="all"), kwargs={"q": jnp.array([1.0])}, linear=jno.solve.lu())
    for i in (0, 2, 4):
        ref = run_continuation(fem, _spec(kap=ks[i : i + 1]), kwargs={"q": jnp.array([1.0])}, linear=jno.solve.lu())
        assert np.asarray(U_lu[i]) == pytest.approx(np.asarray(ref), abs=1e-12), f"family member {i} != cold solve"
    # -div(kap grad u) = q: u scales as 1/kap, so the ends must sit at a 4x ratio
    assert float(U[0].max() / U[-1].max()) == pytest.approx(4.0, rel=1e-6)
    last = run_continuation(fem, _spec(kap=ks), kwargs={"q": jnp.array([1.0])})
    # abs=1e-12, not 0.0: on GPU two runs of the same march differ by reduction order (measured 1.4e-17,
    # one ULP) -- bit-exactness is a CPU accident, not a property of the driver
    assert np.asarray(last) == pytest.approx(np.asarray(U[-1]), abs=1e-12), 'keep="last" != keep="all"[-1]'


def test_two_parameters_march_together_and_fixed_kwargs_hold():
    """Zipped sequences march in lockstep; a fixed keyword holds its value across the sweep."""
    fem = _poisson_kap()
    U = run_continuation(fem, _spec(kap=jnp.array([1.0, 2.0]), q=jnp.array([1.0, 2.0]), keep="all"))
    # kap and q double together, so u = (q/kap)*shape is IDENTICAL at both steps
    assert np.asarray(U[0]) == pytest.approx(np.asarray(U[1]), abs=1e-9), "zipped march broke the q/kap ratio"


def _neo_hookean_load():
    """Finite-strain cantilever with the load as a runtime parameter — the homotopy fixture.
    Cold DEFAULT Newton (no line search) fails at load=0.1 and converges at 0.05 (measured)."""
    mu, lam = 1.0, 1.0
    d = jno.Shape.rect(0.0, 0.0, 2.0, 0.5, size=0.15).domain()
    d.tag("left", lambda x, n, names: x[:, 0] < 1e-6)
    u, v = d.fem_symbols(value_shape=(2,))
    xi, yi = d.variable("interior", split=True)[:2]
    xl, yl = d.variable("left", split=True)[:2]
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    load = jno.np.parameter((1,), name="load")
    F11, F12, F21, F22 = 1.0 + ui[0].x, ui[0].y, ui[1].x, 1.0 + ui[1].y
    Jd = F11 * F22 - F12 * F21
    iT11, iT12, iT21, iT22 = F22 / Jd, -F21 / Jd, -F12 / Jd, F11 / Jd
    c = lam * J.log(Jd)
    P11 = mu * (F11 - iT11) + c * iT11
    P12 = mu * (F12 - iT12) + c * iT12
    P21 = mu * (F21 - iT21) + c * iT21
    P22 = mu * (F22 - iT22) + c * iT22
    weak = P11 * vi[0].x + P12 * vi[0].y + P21 * vi[1].x + P22 * vi[1].y + load * vi[1]
    return jno.fem([weak, u(xl, yl)[0] - 0.0, u(xl, yl)[1] - 0.0])


def test_homotopy_reaches_a_load_the_cold_solve_cannot():
    """The measured teeth. Cold default-Newton at load=0.1 diverges; the four-step ramp reaches it
    with the SAME default solver — no line search, no tuning — and the answer matches a line-search
    cold solve at the target load."""
    fem = _neo_hookean_load()
    with pytest.raises(RuntimeError, match=r"step 1/1 at load="):
        run_continuation(fem, _spec(load=jnp.array([0.1])))

    ramped = run_continuation(fem, _spec(load=jnp.linspace(0.025, 0.1, 4)))
    assert bool(jnp.isfinite(ramped).all())

    ref = run_continuation(fem, _spec(load=jnp.array([0.1])), nonlinear=jno.solve.newton(line_search=True))
    assert np.asarray(ramped) == pytest.approx(np.asarray(ref), abs=1e-6), (
        f"ramped default-Newton disagrees with line-search cold: {np.abs(np.asarray(ramped - ref)).max():.2e}"
    )
    assert float(jnp.abs(ramped).max()) > 1.0, "the target load should be well into the finite-strain regime"


def test_a_complex_sweep_returns_the_complex_family():
    """The EM shape of the driver, on the 1D form where parametric complex is wired today: sweep the
    loss ``sig`` in ``c = 1/(1 + i sig)`` and get the complex family, each member matching the
    constant-coefficient complex solve. (2D/3D nodal parametric-complex is a known wall — the
    ComplexPair x parameter algebra is not wired — so the sweep is exercised where the spelling
    exists; the driver itself is dimension-agnostic.)"""
    from jno.domain import domain as _domain_mod  # noqa: F401  (1D line ctor lives on jno.domain)

    def _build(sig_value=None):
        d = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
        u, phi = d.fem_symbols()  # the 1j in the coefficient is what makes the system complex
        xi = d.variable("interior", split=True)[0]
        xb = d.variable("boundary", split=True)[0]
        ui, vi = u.bind(x=xi), phi.bind(x=xi)
        sig = jno.np.parameter((1,), name="sig") if sig_value is None else sig_value
        c = 1.0 / (1.0 + 1j * sig)
        return d, jno.fem([c * ui.x * vi.x - (np.pi**2) * jno.np.sin(np.pi * xi) * vi, u(xb) - 0.0])

    _d, fem = _build()
    sigs = jnp.linspace(0.1, 1.0, 4)
    U = run_continuation(fem, _spec(sig=sigs, keep="all"))
    assert np.iscomplexobj(np.asarray(U)), "a complex sweep must return complex solutions"
    assert U.shape[0] == 4
    for i in (0, 3):
        _d2, fem_c = _build(float(sigs[i]))
        ref = np.asarray(fem_c.solve()).reshape(-1)
        assert np.abs(np.asarray(U[i]) - ref).max() < 1e-9, f"sweep member {i} != constant-coefficient solve"
    assert float(np.abs(np.asarray(U[3]).imag).max()) > 0.1, "the swept family lost its imaginary part"


def test_a_failing_step_names_the_value_and_index():
    """Marching kappa through zero makes the operator singular. The direct solve returns NaN rather
    than raising, so without the driver's finite check this would surface three steps downstream as
    silent garbage. It must instead raise, naming the value and the step."""
    fem = _poisson_kap()
    with pytest.raises(RuntimeError, match=r"step 3/4 at kap=.*non-finite|step 3/4 at kap="):
        run_continuation(
            fem,
            _spec(kap=jnp.array([1.0, 0.5, 0.0, -0.5]), keep="all"),
            kwargs={"q": jnp.array([1.0])},
            linear=jno.solve.lu(),
        )


def test_validation_fails_loud():
    fem = _poisson_kap()
    with pytest.raises(TypeError, match="unknown runtime parameter"):
        run_continuation(fem, _spec(nope=jnp.array([1.0])))
    with pytest.raises(ValueError, match="share one length"):
        run_continuation(fem, _spec(kap=jnp.array([1.0, 2.0]), q=jnp.array([1.0])))
    with pytest.raises(ValueError, match="either marched or held"):
        run_continuation(fem, _spec(kap=jnp.array([1.0])), kwargs={"kap": jnp.array([2.0])})

    # transient problems refuse: there is no warm start to transfer between trajectories
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.1, 3))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    kap = jno.np.parameter((1,), name="kap")
    fem_t = jno.fem([ui.t * vi + kap * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])
    with pytest.raises(NotImplementedError, match="steady"):
        run_continuation(fem_t, _spec(kap=jnp.array([1.0, 2.0])))


def test_the_slots_compose_with_the_sweep():
    """Each step's solve goes through the composed slots — a CG+Jacobi sweep must match the default."""
    fem = _poisson_kap()
    ks = jnp.linspace(0.5, 2.0, 3)
    a = run_continuation(fem, _spec(kap=ks, keep="all"), kwargs={"q": jnp.array([1.0])})
    b = run_continuation(
        fem,
        _spec(kap=ks, keep="all"),
        kwargs={"q": jnp.array([1.0])},
        linear=jno.solve.cg(tol=1e-12),
        precond=jno.precond.jacobi(),
    )
    assert np.asarray(a) == pytest.approx(np.asarray(b), abs=1e-7)
