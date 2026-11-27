"""``jno.fem(matrix_free=True)`` on a time domain: the march applies the stiffness and stores no matrix.

A first-order (``u.t``) and a second-order (``u.tt``) linear form march on every time scheme exactly as their
assembled twins do (to round-off), with the stiffness an unstored operator and the mass stored as its diagonal
when it is diagonal (the GLL rule at a Q_k field's own nodes) or applied like the stiffness when it is not.
"""

from math import pi, sqrt

import jax
import numpy as np
import pytest
from jno.numpy import exp, sin

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _domain(T, steps, n=2):
    return jno.shape.rect(0, 0, 1, 1).structured(n=n).quad().domain(time=(0.0, T, steps))


def _symbols(d, order):
    u, v = d.fem_symbols(order=order)
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    return u, u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t), (x, y, t), (xb, yb), ci


def _heat(matrix_free, quadrature, *, source=False, T=0.02, steps=21, order=4):
    d = _domain(T, steps)
    u, ui, vi, (x, y, t), (xb, yb), ci = _symbols(d, order)
    form = ui.t * vi + ui.x * vi.x + ui.y * vi.y
    if source:
        form = form - exp(-t) * sin(pi * x) * sin(pi * y) * vi
    u0 = sin(pi * ci[0]) * sin(pi * ci[1])
    return jno.fem([form, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0], quadrature=quadrature, matrix_free=matrix_free)


def _wave(matrix_free, quadrature, *, T=0.2, steps=401, order=6):
    d = _domain(T, steps)
    u, ui, vi, _X, (xb, yb), ci = _symbols(d, order)
    u0 = sin(pi * ci[0]) * sin(pi * ci[1])
    return jno.fem(
        [ui.tt * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0],
        quadrature=quadrature,
        matrix_free=matrix_free,
    )


def _march(fem, scheme=None):
    out = fem.solve() if scheme is None else fem.solve(time=scheme)
    return np.asarray(out.fn() if hasattr(out, "fn") else out)


SCHEMES = {"backward-euler": None, "crank-nicolson": jno.solve.theta(0.5), "rk4": jno.solve.erk("rk4")}


@pytest.mark.parametrize("quadrature", ["gll", None])
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_a_matrix_free_heat_march_is_the_assembled_one(scheme, quadrature):
    steps = 401 if scheme == "rk4" else 21  # the explicit scheme at its stable step
    mf, asm = _heat(True, quadrature, steps=steps), _heat(False, quadrature, steps=steps)
    assert getattr(mf._op.A, "matrix_free", False)
    # the GLL rule at the nodes gives a diagonal mass, stored as its n entries; the Gauss rule's is applied
    if quadrature == "gll":
        assert int(mf._op.M.nse) == int(mf._op.M.shape[0])
    else:
        assert getattr(mf._op.M, "matrix_free", False)
    a, b = _march(mf, SCHEMES[scheme]), _march(asm, SCHEMES[scheme])
    assert np.abs(a - b).max() < 1e-12


def test_a_time_dependent_source_reaches_the_matrix_free_march():
    a, b = _march(_heat(True, "gll", source=True)), _march(_heat(False, "gll", source=True))
    assert np.abs(b[-1]).max() > 1e-3 and np.abs(a - b).max() < 1e-12


@pytest.mark.parametrize("quadrature", ["gll", None])
def test_a_matrix_free_wave_march_is_the_assembled_one(quadrature):
    mf, asm = _wave(True, quadrature, steps=41), _wave(False, quadrature, steps=41)
    assert getattr(mf._op.A, "matrix_free", False)
    a, b = _march(mf), _march(asm)  # the u.tt default: trapezoidal
    assert np.abs(a - b).max() < 1e-12


def test_the_spectral_element_wave_reaches_the_exact_standing_wave_explicitly():
    """Q6 on 2 x 2 cells, the diagonal GLL mass, RK4: one stiffness apply per stage and no solve."""
    fem = _wave(True, "gll")
    assert int(fem._op.M.nse) == int(fem._op.M.shape[0])  # [M2, M2] on the diagonal, nothing else
    P = np.asarray(fem.points)
    uT = _march(fem, jno.solve.erk("rk4"))[-1].reshape(-1)[: P.shape[0]]
    exact = np.cos(sqrt(2.0) * pi * 0.2) * np.sin(pi * P[:, 0]) * np.sin(pi * P[:, 1])
    assert np.abs(uT - exact).max() < 1e-6


def test_a_matrix_free_march_plans_no_sparsity_pattern(monkeypatch):
    from jno.utils.solver import fem_native as fn

    calls = []
    real = fn.compress_plan_blocks
    monkeypatch.setattr(fn, "compress_plan_blocks", lambda *a, **k: calls.append(1) or real(*a, **k))
    _march(_wave(True, "gll", steps=11))
    _march(_heat(True, "gll", steps=5))
    assert not calls


def test_a_driven_boundary_on_a_matrix_free_wave_says_so():
    d = _domain(0.1, 11)
    u, ui, vi, (x, y, t), (xb, yb), ci = _symbols(d, 4)
    tb = d.variable("boundary", split=True)[-1]
    with pytest.raises(NotImplementedError, match="matrix_free"):
        jno.fem(
            [ui.tt * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - sin(tb), u(ci[0], ci[1]) - 0.0],
            quadrature="gll",
            matrix_free=True,
        )


def test_an_explicit_matrix_free_march_sets_its_dirichlet_rows_instead_of_solving_them(monkeypatch):
    """The constraint projection of an explicit stage is a residual evaluation per Newton-Jacobi iteration, 8 per
    stage: on an unstored operator 8 stiffness applies for rows whose value is known (measured 7x the step).
    The block names them as held rows, and the march sets them; the trajectory is the assembled one's."""
    from jno.utils.solver.timeschemes import _ERKScheme

    calls = []
    real = _ERKScheme._project
    monkeypatch.setattr(_ERKScheme, "_project", lambda self, *a: calls.append(1) or real(self, *a))
    rk4 = jno.solve.erk("rk4")
    a = _march(_wave(True, "gll", steps=41), rk4)
    assert not calls
    b = _march(_wave(False, "gll", steps=41), rk4)
    assert calls  # the assembled block still projects
    assert np.abs(a - b).max() < 1e-12
