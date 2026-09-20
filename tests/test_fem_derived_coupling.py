"""``jno.derived``: a nodal field computed from the state by a rule that is not a local integrand.

A weak term is *local* -- an integrand at one quadrature point. Nonlocal physics is not, and jNO already
carries the *residual* side of it (:class:`jno.Coupling`, ``R(u) += c(u)``). ``jno.derived`` is the *value*
side: ``d = f(u)`` on the nodes, usable anywhere a field is. The distinction is not cosmetic -- a beam's
attenuation ``exp(-tau(u))`` MULTIPLIES a source, so it can never be written as a load.

The semantics are lagged: the values are produced inside the residual from ``stop_gradient(u)``, so the
linearization sees them as data and the nonlinear solver *is* the Picard loop. Three claims follow, and
this file is those three claims plus the refusals:

* the tangent does not see the coupling, but it does see its VALUE -- so a derived field really is a
  coefficient, just a lagged one (:func:`test_the_tangent_is_lagged_but_its_value_is_live`);
* the assembled tangent and the matrix-free JVP are the SAME operator. This is the one that must not
  drift: without the ``stop_gradient`` they would differ by exactly the coupling, and
  ``newton(direct=True)`` and JFNK would silently solve different problems;
* the converged root is EXACT. Lagging changed the path, not the answer. The oracle is a nonlocal but
  linear coupling -- a source proportional to the solution's own mean -- whose closed form is
  ``u = w / (1 - c * mean(w))`` with ``w`` the uncoupled solution.

The last test is the honest one: past a critical coupling strength the fixed point does not converge, and
the solve must SAY so rather than return a plausible wrong field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
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


SIZE = 0.2  # one mesh for the whole file: 44 P1 nodes on the unit square


def _mean_source(c, *, size=SIZE):
    """``-div(grad u) = 1 + c*mean(u)`` on the unit square, u = 0 on the boundary.

    The source is nonlocal -- every node's load depends on every other node's value -- but LINEAR in u,
    which is what buys a closed form: with ``w`` the ``c = 0`` solution, ``u = (1 + c*mean(u)) w``, so
    ``mean(u) = mean(w)/(1 - c*mean(w))`` and ``u = w/(1 - c*mean(w))``. The fixed point contracts at
    rate ``c*mean(w)``, so the same knob sets both the answer and the difficulty.
    """
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    src = jno.derived(lambda T: c * jnp.mean(T) * jnp.ones_like(T), inputs=[u], on=u)
    return jno.fem([ui.x * vi.x + ui.y * vi.y - (1.0 + src) * vi, u(xb, yb) - 0.0]), d


def _mean_conductivity(rule, *, size=SIZE):
    """``-div((1 + d(u)) grad u) = 1``, u = 0 on the boundary -- the derived field as a COEFFICIENT.

    Here the field multiplies the unknown, so it appears in the tangent: this is the form that can tell a
    lagged Jacobian from a frozen one at all (in a pure source term the value never reaches the tangent).
    """
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    k = 1.0 + jno.derived(rule, inputs=[u], on=u)
    return jno.fem([k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0]), d


def _dense(J):
    return jnp.asarray(J.todense() if hasattr(J, "todense") else J)


def _admissible(d, n, seed):
    """A random state that SATISFIES the Dirichlet data, so the operator does not lift it out from under
    the comparison -- the residual pins those DOFs before evaluating, which would otherwise feed the rule
    a different vector than the one the test froze at."""
    pinned = [int(p[0]) for p in (getattr(d, "_fem_native_dirichlet_pairs", None) or [])]
    x = np.array(jax.random.uniform(jax.random.PRNGKey(seed), (n,)), copy=True)
    x[pinned] = 0.0
    return jnp.asarray(x)


K_RULE = lambda T: 0.7 * jnp.mean(T**2) * jnp.ones_like(T)  # noqa: E731 -- nonlocal, nonlinear, closed form free


def test_the_tangent_is_lagged_but_its_value_is_live():
    """The Jacobian is the one a FROZEN field of the same values would give -- and it moves when they do.

    Both halves matter. Agreement with the frozen field at the SAME state is the lagging (the derivative
    of the rule never enters); disagreement with the frozen field at a DIFFERENT state is the proof that
    the value is genuinely used, i.e. that the coupling is real and not a zero placeholder.
    """
    fem, d = _mean_conductivity(K_RULE)
    n = int(fem.dofs)
    u0, u1 = _admissible(d, n, 0), _admissible(d, n, 7)

    J = _dense(fem.operator.jacobian(u0, {}))
    J_here = _dense(_mean_conductivity(lambda T, c=K_RULE(u0): c)[0].operator.jacobian(u0, {}))
    J_there = _dense(_mean_conductivity(lambda T, c=K_RULE(u1): c)[0].operator.jacobian(u0, {}))

    assert float(jnp.abs(J - J_here).max()) < 1e-13, "the tangent picked up the rule's derivative -- not lagged"
    moved = float(jnp.abs(J - J_there).max())
    assert moved > 1e-3, f"the tangent ignores the derived VALUE (moved by {moved:.2e}) -- the field is inert"


def test_the_assembled_tangent_and_the_matrix_free_jvp_are_one_operator():
    """``J v`` from the assembled Jacobian equals ``jvp(residual)`` -- the test the ``stop_gradient`` exists for.

    The assembled tangent differentiates per element and so cannot see args-borne data at all; the
    matrix-free tangent differentiates the WHOLE residual and would see the rule. ``stop_gradient`` is what
    keeps them the same operator. Without it ``newton(direct=True)`` and JFNK converge to the same root by
    different iterates -- and every preconditioner built from the assembled matrix is wrong for the
    operator actually being solved.
    """
    fem, d = _mean_conductivity(K_RULE)
    n = int(fem.dofs)
    u0 = _admissible(d, n, 0)
    J = _dense(fem.operator.jacobian(u0, {}))
    w = jax.random.normal(jax.random.PRNGKey(3), (n,))
    _, jvp = jax.jvp(lambda z: jnp.asarray(fem.operator.residual(z, {})).reshape(-1), (u0,), (w,))
    gap = float(jnp.abs(J @ w - jvp).max()) / float(jnp.abs(jvp).max())
    assert gap < 1e-13, f"assembled and matrix-free tangents differ by {gap:.2e} relative -- check the stop_gradient"


@pytest.mark.parametrize("c", [0.3, 1.0, 2.0])
def test_the_converged_root_is_exact(c):
    """Lagging changes the path, not the answer: ``u = w / (1 - c*mean(w))`` to solver tolerance."""
    w = np.asarray(_mean_source(0.0)[0].solve()).reshape(-1)  # the uncoupled solution
    T = np.asarray(_mean_source(c)[0].solve()).reshape(-1)
    exact = w / (1.0 - c * w.mean())
    err = np.abs(T - exact).max()
    assert err < 1e-8, f"c={c}: converged root is off its closed form by {err:.2e}"
    assert abs(T.max() / w.max() - 1.0) > 1e-3, "the coupling did nothing -- the oracle would pass on an inert field"


def test_the_same_physics_as_a_coupling_reaches_the_same_root():
    """Cross-check against the mechanism jNO already had, on physics written both ways.

    The source ``c*mean(u)`` is spatially constant, so its consistent load is ``c*mean(u)`` times the load
    vector of ``1``: the identical physics fits :class:`jno.Coupling`'s residual signature exactly. Same
    root, reached through a tangent that carries the coupling instead of lagging it -- which is the whole
    difference between the two mechanisms, and it must not show up in the answer.
    """
    c = 1.0
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=SIZE).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    base = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    b1 = -jnp.asarray(base.operator[1] if isinstance(base.operator, tuple) else base.operator).reshape(-1)

    d2 = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=SIZE).domain()
    u2, v2 = d2.fem_symbols()
    xi2, yi2, _ = d2.variable("interior", split=True)
    xb2, yb2, _ = d2.variable("boundary", split=True)
    ui2, vi2 = u2.bind(x=xi2, y=yi2), v2.bind(x=xi2, y=yi2)
    coupled = jno.fem([ui2.x * vi2.x + ui2.y * vi2.y - 1.0 * vi2, lambda w: c * jnp.mean(w) * b1, u2(xb2, yb2) - 0.0])

    T_derived = np.asarray(_mean_source(c)[0].solve()).reshape(-1)
    T_coupling = np.asarray(coupled.solve()).reshape(-1)
    err = np.abs(T_derived - T_coupling).max() / np.abs(T_coupling).max()
    assert err < 1e-8, f"derived and Coupling disagree by {100 * err:.3e} % on the same physics"


def test_a_coupling_too_strong_to_iterate_fails_loudly():
    """Past ``c*mean(w) = 1`` the fixed point does not converge -- and the solve must say so.

    This is the honest cost of lagging, and it is why :class:`jno.Coupling` is not being retired. The
    solution also blows through a pole there (``1 - c*mean(w) -> 0``), so there is no right answer to
    return: silence would be the only bad outcome.
    """
    w = np.asarray(_mean_source(0.0)[0].solve()).reshape(-1)
    c_crit = 1.0 / w.mean()
    with pytest.raises(Exception) as excinfo:
        np.asarray(_mean_source(1.5 * c_crit)[0].solve())
    assert "converge" in str(excinfo.value).lower() or "residual" in str(excinfo.value).lower(), (
        f"the failure did not name the convergence problem: {excinfo.value}"
    )


def _transient_mean_source(c, every, n_steps, *, size=0.34, t_end=0.2):
    """``u_t - div(grad u) = c*mean(u)``, u = 0 on the boundary, from a sine hump.

    Both cadences march the same physics; they differ only in WHEN the nonlocal source is evaluated --
    inside every Newton residual, or once per step from the state the step starts at.
    """
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size), time=(0.0, t_end, n_steps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    src = jno.derived(lambda T: c * jnp.mean(T) * jnp.ones_like(T), inputs=[u], on=u, every=every)
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y - src * vi, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def test_evaluating_once_per_step_costs_a_first_order_splitting_error():
    """``every="step"`` is an operator splitting, and the test is the ORDER, not mere closeness.

    Lagging the source by one step cannot change where the march is heading -- at a fixed point the
    previous state IS the current one -- but it changes the path by O(dt). Asserting a small difference
    would pass for a field that is simply inert; asserting the observed RATE is what actually pins the
    claim in the docstring.
    """
    c = 1.0
    ends = {}
    for n in (8, 16, 32):
        ends[n] = {
            e: np.asarray(_transient_mean_source(c, e, n).solve().fn())[-1].reshape(-1) for e in ("residual", "step")
        }
    gaps = [np.abs(ends[n]["step"] - ends[n]["residual"]).max() for n in (8, 16, 32)]
    assert gaps[0] > 1e-8, f"the cadences are indistinguishable ({gaps[0]:.2e}) -- the test would pass on nothing"
    rates = [np.log2(gaps[i] / gaps[i + 1]) for i in range(len(gaps) - 1)]
    assert all(0.7 < r < 1.4 for r in rates), f"splitting error is not first order in dt: gaps {gaps}, rates {rates}"


def test_a_parameter_inside_the_rule_threads_and_is_differentiable_as_the_picard_adjoint():
    """A ``jno.np.parameter`` used only inside the rule is invisible to the trace walk -- the rule is a
    plain function, not an expression -- so it is declared with ``params=[...]`` and merged into the
    operator's runtime parameters, exactly as a :class:`jno.Coupling`'s is.

    The gradient is the honest part. The parameter itself is NOT lagged, so the direct sensitivity through
    it is exact; the path through ``u`` uses the lagged Jacobian. What comes out is the standard Picard
    adjoint: right sign, right magnitude, descent-worthy -- and measurably not the true derivative. The
    test asserts the *measured* gap rather than pretending it is 1e-6.
    """
    from jno.utils.solver.newton_krylov import newton_krylov

    p = jno.np.parameter((1,), name="cstr")
    p.initialize(jax.nn.initializers.constant(0.5))
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=SIZE).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    rule = lambda T, pp: pp["cstr"].reshape(()) * jnp.mean(T) * jnp.ones_like(T)  # noqa: E731
    src = jno.derived(rule, inputs=[u], on=u, params=[p])
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - (1.0 + src) * vi, u(xb, yb) - 0.0])
    op, n = fem.operator, int(fem.dofs)
    assert "cstr" in (op.runtime_parameter_exprs or {}), "a params=[...] value never reached the solve's args"

    def qoi(c):  # the mean temperature of the coupled solve, as a function of the coupling strength
        res = lambda w: jnp.asarray(op.residual(w, {"cstr": jnp.atleast_1d(c)})).reshape(-1)  # noqa: E731
        return jnp.mean(newton_krylov(res, jnp.zeros(n)))

    # the forward answer is the exact coupled root, so the closed form still holds
    w = np.asarray(_mean_source(0.0)[0].solve()).reshape(-1)
    assert float(qoi(0.5)) == pytest.approx(float(np.mean(w / (1.0 - 0.5 * w.mean()))), rel=1e-6)

    g = float(jax.grad(qoi)(0.5))
    fd = float((qoi(0.5 + 1e-5) - qoi(0.5 - 1e-5)) / 2e-5)
    assert np.isfinite(g) and abs(g) > 1e-8, f"no sensitivity to a parameter inside the rule (grad {g:.3e})"
    assert g * fd > 0, f"the Picard adjoint has the wrong SIGN: grad {g:.3e} vs finite difference {fd:.3e}"
    gap = abs(g - fd) / abs(fd)
    assert gap < 0.05, f"Picard adjoint is {100 * gap:.1f} % off the true derivative -- expected a few percent"


def _periodic_mean_source(c, *, size=0.25):
    """The same nonlocal source, now on a form with a periodic tie ``u(left) = u(right)``."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    src = jno.derived(lambda T: c * jnp.mean(T) * jnp.ones_like(T), inputs=[u], on=u)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - (1.0 + src) * vi, u(xl, yl) - u(xr, yr), u(xb, yb) - 0.0])
    r = fem.solve()
    return np.asarray(r.fn() if hasattr(r, "fn") else r).reshape(-1)


def test_a_steady_periodic_form_composes_with_a_derived_field():
    """Periodic ties reduce LAZILY on the steady nonlinear path (``P^T r(P.)``), so the rule still sees
    full nodal values and the same closed form holds. Worth checking rather than assuming: the reduced
    state is shorter than the nodal vector, and a rule fed the wrong one returns a plausible field."""
    c = 0.5
    w = _periodic_mean_source(0.0)
    T = _periodic_mean_source(c)
    exact = w / (1.0 - c * w.mean())
    assert np.abs(T - exact).max() < 1e-7, f"periodic + derived misses its closed form by {np.abs(T - exact).max():.2e}"


def test_a_periodic_transient_refuses_the_step_cadence_instead_of_reading_the_reduced_state():
    """The one place the two features genuinely do not compose, refused by name.

    A periodic transient block marches the REDUCED main-DOF state. Evaluated inside the residual that is
    harmless -- the wrapper prolongs its input back to the nodal space first -- but the ``every="step"``
    driver sees the reduced vector and would slice it with the form's full-space offsets. Measured before
    the guard: a 17-DOF reduced state handed to a rule expecting 20 nodes, and a plausible field returned.
    """
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.34), time=(0.0, 0.1, 4))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    terms = lambda q: [  # noqa: E731
        ui.t * vi + ui.x * vi.x + ui.y * vi.y - q * vi,
        u(xl, yl) - u(xr, yr),
        u(xb, yb) - 0.0,
        u(ci[0], ci[1]) - 1.0,
    ]
    rule = lambda T: 0.2 * jnp.mean(T) * jnp.ones_like(T)  # noqa: E731
    with pytest.raises(NotImplementedError, match="every='step'"):
        jno.fem(terms(jno.derived(rule, inputs=[u], on=u, every="step")))
    # ...and the default cadence composes, marching to a full-length nodal field
    marched = np.asarray(jno.fem(terms(jno.derived(rule, inputs=[u], on=u))).solve().fn())
    assert marched.ndim == 2 and np.all(np.isfinite(marched)), f"periodic transient + derived gave {marched.shape}"


# ---------------------------------------------------------------------------
# refusals -- each names the fix, and each is a mistake that would otherwise be silent
# ---------------------------------------------------------------------------
def _symbols():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    u, v = d.fem_symbols()
    return d, u, v


def test_a_rule_that_is_not_jax_is_refused_at_build_with_the_fix_named():
    """A host/numpy rule cannot run inside the residual. The probe traces abstractly, so it fails at
    ``jno.fem`` build -- naming the fix -- instead of somewhere unrecognisable inside a Newton step."""
    d, u, v = _symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    bad = jno.derived(lambda T: np.asarray(T) * 2.0, inputs=[u], on=u)  # np.asarray on a tracer
    with pytest.raises(ValueError, match="not traceable"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - bad * vi, u(xb, yb) - 0.0])


def test_a_rule_that_returns_the_wrong_number_of_nodes_is_refused():
    d, u, v = _symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    bad = jno.derived(lambda T: T[:3], inputs=[u], on=u)
    with pytest.raises(ValueError, match="nodes"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - bad * vi, u(xb, yb) - 0.0])


def test_every_step_on_a_steady_problem_is_refused_rather_than_silently_ignored():
    """There is no march to step, so nothing would supply the per-step values and the field would quietly
    fall back to per-residual evaluation -- a different problem than the one that was asked for."""
    d, u, v = _symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    bad = jno.derived(lambda T: T * 0.5, inputs=[u], on=u, every="step")
    with pytest.raises(ValueError, match="no march to step"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - bad * vi, u(xb, yb) - 0.0])


def test_the_constructor_refuses_what_it_can_see_immediately():
    """Cadence, inputs and chaining are decidable at the call site, so they fail there, not at build."""
    _, u, _ = _symbols()
    ok = jno.derived(lambda T: T, inputs=[u], on=u)
    with pytest.raises(ValueError, match="not a cadence"):
        jno.derived(lambda T: T, inputs=[u], on=u, every="round")
    with pytest.raises(ValueError, match="`inputs` is empty"):
        jno.derived(lambda T: T, inputs=[], on=u)
    with pytest.raises(TypeError, match="must be a list"):
        jno.derived(lambda T: T, inputs=u, on=u)
    with pytest.raises(TypeError, match="must be a trial function"):
        jno.derived(lambda T: T, inputs=[1.0], on=u)
    with pytest.raises(ValueError, match="Chained derived fields"):
        jno.derived(lambda T: T, inputs=[ok], on=u)
    with pytest.raises(TypeError, match="must be a callable"):
        jno.derived("not a rule", inputs=[u], on=u)
