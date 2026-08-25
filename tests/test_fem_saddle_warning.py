"""Saying so when the matrix-free default is pointed at a saddle-point system.

The default steady-linear solve is Jacobi-preconditioned BiCGStab. A saddle block has a ZERO
diagonal, so the preconditioner is the identity exactly where the system is hardest. That is not
always a loud failure: the convergence guard accepts any relative residual under ``1e-4`` -- on a
Taylor-Hood system that can still be a pressure with no correct digits -- and the guard steps aside
entirely under ``jit``/``vmap``/``grad``, where it cannot concretise a residual.

Detection is structural and happens at BUILD time: a field whose own test function never meets its
own trial function contributes no ``(i, i)`` block. Read from the terms rather than an assembled
matrix, that holds in every mode and needs no tangent -- and being known before the solve is what
lets the warning fire under ``jit``, which is the case the residual guard cannot cover.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import jno

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
SADDLE = "saddle-point system"


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stokes(size=0.4):
    """Taylor-Hood P2/P1: the canonical saddle system."""
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            inner(gu, gv, n_contract=2) - pp * trace(gv),
            -qq * trace(gu),
            u(xb, yb)[0] - 0.0,
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )
    return u, p, pp, qq, fem


def _two_coupled_scalars():
    """Two fields, genuinely coupled off-diagonal, but BOTH diagonal blocks present.

    The discriminating case: coupling alone is not a saddle, and a detector that only looked for
    off-diagonal terms would cry wolf on every multiphysics problem in the suite.
    """
    d = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    s1, t1 = d.fem_symbols(names=("s1", "t1"), order=1)
    s2, t2 = d.fem_symbols(names=("s2", "t2"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    a1, b1 = s1.bind(x=xi, y=yi), t1.bind(x=xi, y=yi)
    a2, b2 = s2.bind(x=xi, y=yi), t2.bind(x=xi, y=yi)
    return jno.fem(
        [
            a1.x * b1.x + a1.y * b1.y + a2 * b1 - 1.0 * b1,  # block (0,0) and (0,1)
            a2.x * b2.x + a2.y * b2.y - 1.0 * b2,  # block (1,1)
            s1(xb, yb) - 0.0,
            s2(xb, yb) - 0.0,
        ]
    )


def _saddle_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if SADDLE in str(w.message)]


def test_the_default_on_a_saddle_warns_and_names_the_field():
    _, _, _, _, fem = _stokes()
    assert fem._saddle_blocks == ("p",), f"expected the pressure block, got {fem._saddle_blocks}"
    msgs = _saddle_warnings(fem.solve)
    assert len(msgs) == 1
    assert "'p'" in msgs[0] or " p " in msgs[0]
    # it must point at the paths that actually work, not merely complain
    assert "jno.solve.lu" in msgs[0] and "precond" in msgs[0]


@pytest.mark.parametrize(
    "slot",
    [
        pytest.param({"linear": jno.solve.lu(backend="host")}, id="linear"),
        pytest.param(
            {"solve_fn": lambda A, b: np.linalg.solve(np.asarray(A.todense()), np.asarray(b).reshape(-1))}, id="solve_fn"
        ),
    ],
)
def test_choosing_a_solver_silences_it(slot):
    """Passing a solver IS the deliberate choice the warning asks for -- advising anyway is noise."""
    _, _, _, _, fem = _stokes()
    assert _saddle_warnings(lambda: fem.solve(**slot)) == []


def test_a_problem_with_no_saddle_block_is_silent():
    d = jno.Shape.rect(0, 0, 1, 1, size=0.4).domain()
    a, b = d.fem_symbols(names=("a", "b"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    ai, bi = a.bind(x=xi, y=yi), b.bind(x=xi, y=yi)
    fem = jno.fem([ai.x * bi.x + ai.y * bi.y - 1.0 * bi, a(xb, yb) - 0.0])
    assert fem._saddle_blocks == ()
    assert _saddle_warnings(fem.solve) == []


def test_off_diagonal_coupling_alone_is_not_a_saddle():
    """The false-positive guard: every coupled multiphysics problem must stay quiet."""
    fem = _two_coupled_scalars()
    assert fem._saddle_blocks == (), f"cried wolf on a coupled non-saddle: {fem._saddle_blocks}"
    assert _saddle_warnings(fem.solve) == []


def test_it_fires_under_jit_where_the_residual_guard_cannot():
    """The reason detection is structural and happens at build: `_residual_check` needs a concrete
    residual and steps aside on a tracer, so under `jit` it is exactly the case with NO guard."""
    _, _, _, _, fem = _stokes()
    assert len(_saddle_warnings(lambda: jax.jit(lambda: fem.solve())())) == 1


def test_it_warns_once_per_problem():
    _, _, _, _, fem = _stokes()
    assert len(_saddle_warnings(lambda: (fem.solve(), fem.solve()))) == 1
