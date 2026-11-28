"""``jno.fem(matrix_free=True)`` on a nonlinear form: Newton-Krylov, the tangent never assembled.

Each Newton step applies the tangent as a JVP of the (sum-factorised) residual and preconditions it with the
tangent's diagonal, read by sum factorisation inside the compiled solve. The root is the assembled Newton's.
"""

import jax
import numpy as np
import pytest
from jno.numpy import cos, sin

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _problem(matrix_free, *, g=0.0, order=5, n=3):
    """``-div((1 + u^2) grad u) = f`` on the unit square, Q_k on n x n quads, manufactured ``u = sin(pi x) sin(pi y)``
    (plus ``g`` on the boundary and inside: a non-zero prescribed value takes the lift step)."""
    d = jno.shape.rect(0, 0, 1, 1).structured(n=n).quad().domain()
    u, v = d.fem_symbols(order=order)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    pi = np.pi
    U = sin(pi * x) * sin(pi * y) + g
    Ux, Uy = pi * cos(pi * x) * sin(pi * y), pi * sin(pi * x) * cos(pi * y)
    f = (1 + U * U) * 2 * pi**2 * (U - g) - 2 * U * (Ux * Ux + Uy * Uy)
    form = (1 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - f * vi
    return jno.fem([form, u(xb, yb) - g], quadrature="gll", matrix_free=matrix_free)


def _exact(fem, g=0.0):
    P = np.asarray(fem.points)
    return np.sin(np.pi * P[:, 0]) * np.sin(np.pi * P[:, 1]) + g


@pytest.mark.parametrize("g", [0.0, 0.5])
def test_the_default_solve_of_a_matrix_free_nonlinear_form_is_newton_krylov_at_the_assembled_root(g):
    mf, asm = _problem(True, g=g), _problem(False, g=g)
    a = np.asarray(mf.solve()).reshape(-1)
    assert mf.stats["nonlinear"]["driver"] == "newton_krylov"
    assert mf.stats["linear_iterations"]["by"] == ["the newton_krylov inner Jacobi-BiCGStab"]
    b = np.asarray(asm.solve()).reshape(-1)
    assert asm.stats["nonlinear"]["driver"] == "newton_direct"  # the assembled default is unchanged
    assert np.abs(a - b).max() < 1e-7  # both to a relative residual of 1e-8
    assert np.abs(a - _exact(mf, g)).max() < 1e-4


def test_newton_direct_none_on_a_matrix_free_form_stays_matrix_free():
    mf = _problem(True)
    mf.solve(nonlinear=jno.solve.newton())
    assert mf.stats["nonlinear"]["driver"] == "newton_krylov"


def test_a_tangent_only_option_on_a_matrix_free_form_says_so():
    with pytest.raises(ValueError, match="matrix_free"):
        _problem(True).solve(nonlinear=jno.solve.newton(ptc=1.0))


def test_the_diagonal_inside_the_newton_trace_is_the_sum_factorised_one(monkeypatch):
    """The fast diagonal's check against the exact pass runs once per term, at build and on concrete values, so a
    Newton step (whose state is a tracer, and whose first diagonal is asked for inside its compiled loop) keeps the
    fast route. Decided in the trace it answered no, and the exact pass ran: 1.9 s a diagonal at Q8, 530k DOFs."""
    from jno.utils.solver import fem_native as fn

    checks = []
    real_check = fn._fast_diag_agrees
    monkeypatch.setattr(fn, "_fast_diag_agrees", lambda *a, **k: (lambda r: checks.append(r) or r)(real_check(*a, **k)))
    fem = _problem(True)
    assert checks == [True]  # decided at build
    op = fem._op
    u = jax.numpy.asarray(np.random.default_rng(1).uniform(-1, 1, int(op.size)))
    traced = np.asarray(jax.jit(op.jacobian_diag)(u))  # first use inside a trace
    eager = np.asarray(op.jacobian_diag(u))
    assert checks == [True]  # kept, not re-decided
    np.testing.assert_allclose(traced, eager, rtol=1e-12, atol=1e-12)
    # and it is the tangent's diagonal at u
    J = np.asarray(jax.jacfwd(op.residual)(u))
    np.testing.assert_allclose(eager, np.diag(J), rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------------------------------- transient forms


def _nl_heat(matrix_free, *, steps=11, quadrature="gll"):
    d = jno.shape.rect(0, 0, 1, 1).structured(n=2).quad().domain(time=(0.0, 0.02, steps))
    u, v = d.fem_symbols(order=4)
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    form = ui.t * vi + (1 + ui * ui) * (ui.x * vi.x + ui.y * vi.y)
    u0 = sin(np.pi * ci[0]) * sin(np.pi * ci[1])
    return jno.fem([form, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0], quadrature=quadrature, matrix_free=matrix_free)


def _march(fem, scheme):
    out = fem.solve() if scheme is None else fem.solve(time=scheme)
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


_SCHEMES = {"backward-euler": None, "crank-nicolson": jno.solve.theta(0.5), "bdf2": jno.solve.bdf2()}


@pytest.mark.parametrize("quadrature", ["gll", None])
@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_a_matrix_free_nonlinear_march_steps_by_newton_krylov_to_the_assembled_trajectory(scheme, quadrature):
    mf, asm = _nl_heat(True, quadrature=quadrature), _nl_heat(False, quadrature=quadrature)
    assert mf._op.jacobian is None and asm._op.jacobian is not None  # no step tangent is ever assembled
    if quadrature == "gll":
        assert int(mf._op.mass(0.0).nse) == int(mf.points.shape[0])  # the diagonal mass, n entries
    a, b = _march(mf, _SCHEMES[scheme]), _march(asm, _SCHEMES[scheme])
    assert np.abs(b[-1]).max() > 0.5 and np.abs(a - b).max() < 1e-7  # both to their Newton tolerance


def test_an_explicit_matrix_free_nonlinear_march_is_the_assembled_one():
    rk4 = jno.solve.erk("rk4")
    a, b = _march(_nl_heat(True, steps=401), rk4), _march(_nl_heat(False, steps=401), rk4)
    assert np.abs(a - b).max() < 1e-12


def test_a_direct_step_solve_on_a_matrix_free_march_says_so():
    with pytest.raises(ValueError, match="matrix_free"):
        out = _nl_heat(True, steps=3).solve(nonlinear=jno.solve.newton(direct=True))
        np.asarray(out.fn() if hasattr(out, "fn") else out)
