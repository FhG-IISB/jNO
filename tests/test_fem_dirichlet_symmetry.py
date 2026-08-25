"""Symmetric Dirichlet elimination on the residual (Newton) paths.

The linear path has always eliminated symmetrically — lift the known columns to the RHS, zero the
constrained rows *and* columns, unit diagonal. The residual paths did row replacement instead:
``R[d] = u[d] − g`` with the constrained **columns left populated**, which makes the tangent
non-symmetric for a problem whose operator is symmetric. That is not free: jNO tests symmetry
*bitwise* before letting cuDSS/PARDISO take LDLᵀ instead of a general LU, and ``cg``/``minres``
are unusable on a non-symmetric operator — measured, ``linear=jno.solve.cg()`` returned **NaN
without raising** on a nonlinear Dirichlet problem.

The fix composes the free residual with the projection onto the constraint set:

    R(u) = M · R_free(P(u)) + (I − M) · (u − g),      P(u) = M·u + (I − M)·g

whose derivative is ``M·J(P(u))·M + (I − M)``. The root is unchanged — it is the same constrained
solution — so this is behaviour-preserving with a strictly better tangent.

Oracles:
* **the constrained columns are gone** — the structural claim, read straight off the assembled tangent.
* **a symmetric form gives a symmetric tangent**, and a genuinely non-symmetric one does NOT get
  quietly symmetrized (the control: advection keeps its O(1) asymmetry).
* **cg works** where it previously returned NaN, and agrees with the default driver.
* **no conforming start is needed** — the projection *is* the lift, so an ``x0`` that violates the
  boundary values converges to the same answer with the pins landing exactly on ``g``.
* **the solution did not move** — checked against an analytic oracle, not against itself.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
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

EPS = float(np.finfo(np.float64).eps)
ULPS = 64.0 * EPS  # "symmetric to within a few ulps" — measured round-off is ~0.25 ulps


def _aliases():
    n = jno.np
    return n.inner, n.grad, n.sym, n.trace, n.identity


def _poisson(g=0.35, size=0.2):
    """Scalar nonlinear Poisson with an INHOMOGENEOUS Dirichlet value (g != 0 is what exercises the
    lift: with g = 0 the projection is the identity and the test would be vacuous)."""
    inner, grad, _s, _t, _i = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    return jno.fem([inner(grad(u, X), grad(phi, X), 1) + 0.5 * u * u * phi - 8.0 * phi, u(*cb) - g]), d


def _elasticity(size=0.5):
    """Vector, and VARIATIONAL: the stress is the derivative of a stored energy, so the tangent is that
    energy's Hessian and is symmetric by construction. (A nonlinear form that is not the gradient of a
    potential has no reason to give a symmetric tangent — that is a property of the physics.)"""
    inner, grad, sym, trace, ident = _aliases()
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    d.tag("bot", lambda x, y, z: z < 1e-6)
    co, cb = d.variable("interior", split=True), d.variable("bot", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(value_shape=(3,))
    e = sym(grad(u, X))
    sig = jno.np.diff(40.0 * trace(e) ** 2 + 60.0 * inner(e, e, 2) + 200.0 * inner(e, e, 2) ** 2, e)
    zhat = jno.np.asarray([0.0, 0.0, 1.0])
    return jno.fem(
        [inner(sig, sym(grad(phi, X)), 2) - 4.0 * inner(zhat, phi, 1)]
        + [u(cb[0], cb[1], cb[2])[i] - 0.01 * (i + 1) for i in range(3)]
    ), d


def _advection(b=1.0, size=0.25):
    """A genuinely NON-symmetric operator — the control."""
    inner, grad, _s, _t, _i = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    return jno.fem(
        [inner(grad(u, X), grad(phi, X), 1) + b * u.d(X[0]) * phi + 0.1 * u * u * phi - 1.0 * phi, u(*cb) - 0.0]
    ), d


def _dense_tangent(fem, u):
    J = fem._op.jacobian(jnp.asarray(u))
    return np.asarray(J.todense()) if hasattr(J, "todense") else np.asarray(J)


def _asymmetry(A):
    return float(np.abs(A - A.T).max() / max(np.abs(A).max(), 1e-30))


# --------------------------------------------------------------------------------------------------
# Oracle 1 — the structural claim: the constrained COLUMNS are eliminated, not just the rows.
# --------------------------------------------------------------------------------------------------
def test_constrained_columns_are_eliminated():
    fem, d = _poisson()
    sol = np.asarray(fem.solve())
    A = _dense_tangent(fem, sol)
    x = np.asarray(fem.points)[:, 0]
    y = np.asarray(fem.points)[:, 1]
    pinned = (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9)
    assert pinned.sum() > 3, "no pinned DOFs — the test would be vacuous"
    free = ~pinned
    # Row replacement kept these; symmetric elimination zeroes them.
    assert np.abs(A[np.ix_(free, pinned)]).max() == 0.0, "constrained columns still couple into the free rows"
    assert np.abs(A[np.ix_(pinned, free)]).max() == 0.0, "constrained rows are not identity rows"
    assert np.array_equal(A[np.ix_(pinned, pinned)], np.eye(int(pinned.sum()))), "pinned block must be the identity"


# --------------------------------------------------------------------------------------------------
# Oracle 2 — a symmetric form gives a symmetric tangent; a non-symmetric one is left alone.
# --------------------------------------------------------------------------------------------------
def test_a_symmetric_form_now_has_a_symmetric_tangent():
    fem, _d = _poisson()
    A = _dense_tangent(fem, np.asarray(fem.solve()))
    # A scalar element block computes K[a,b] and K[b,a] as the identical float expression, so with the
    # columns eliminated the whole tangent is symmetric BITWISE — what cuDSS/PARDISO require for LDLᵀ.
    assert np.array_equal(A, A.T), f"scalar tangent is not bitwise symmetric: rel {_asymmetry(A):.3e}"


def test_a_vector_form_is_symmetric_to_roundoff():
    """A vector element block contracts components in a different order for (a,i),(b,j) than for
    (b,j),(a,i), so it differs by an ulp — assembly round-off, not the Dirichlet structure. Measured
    ~0.25 ulps, against ~1e-1 before this change."""
    fem, _d = _elasticity()
    A = _dense_tangent(fem, np.asarray(fem.solve()))
    rel = _asymmetry(A)
    assert rel < ULPS, f"vector tangent asymmetry {rel:.3e} ({rel / EPS:.1f} ulps) exceeds a few ulps"
    assert rel > 0.0, "bitwise symmetry here would mean the element block changed — see the docstring"


def test_a_genuinely_nonsymmetric_form_is_not_symmetrized():
    """The control. Nothing in this change may quietly average a non-symmetric operator."""
    fem, _d = _advection(b=1.0)
    A = _dense_tangent(fem, np.asarray(fem.solve()))
    rel = _asymmetry(A)
    assert rel > 1e-3, f"an advection tangent must stay non-symmetric, got {rel:.3e}"


# --------------------------------------------------------------------------------------------------
# Oracle 3 — cg becomes usable. Before this change it returned NaN, silently, on exactly this problem.
# --------------------------------------------------------------------------------------------------
def test_cg_solves_a_dirichlet_nonlinear_problem():
    for maker in (_poisson, _elasticity):
        fem, _d = maker()
        ref = np.asarray(fem.solve())  # matrix-free BiCGStab default
        got = np.asarray(fem.solve(linear=jno.solve.cg()))
        assert np.all(np.isfinite(got)), f"{maker.__name__}: cg returned a non-finite solution"
        rel = np.abs(got - ref).max() / max(np.abs(ref).max(), 1e-30)
        assert rel < 1e-8, f"{maker.__name__}: cg disagrees with the default driver, rel {rel:.3e}"


# --------------------------------------------------------------------------------------------------
# Oracle 4 — the projection IS the lift, so no conforming starting iterate is required.
# --------------------------------------------------------------------------------------------------
def test_a_starting_iterate_violating_the_boundary_values_converges_the_same():
    g = 0.35
    fem, _d = _poisson(g=g)
    base = np.asarray(fem.solve())
    bad = jnp.full((fem.dofs,), -5.0)  # nowhere near the pinned value, nor near the solution
    got = np.asarray(fem.solve(x0=bad))
    rel = np.abs(got - base).max() / np.abs(base).max()
    assert rel < 1e-8, f"a non-conforming x0 changed the answer: rel {rel:.3e}"
    pts = np.asarray(fem.points)
    pinned = (pts[:, 0] < 1e-9) | (pts[:, 0] > 1 - 1e-9) | (pts[:, 1] < 1e-9) | (pts[:, 1] > 1 - 1e-9)
    assert np.abs(got[pinned] - g).max() < 1e-12, "the pins must land exactly on g regardless of the start"


# --------------------------------------------------------------------------------------------------
# Oracle 5 — the answer itself, against an ANALYTIC reference rather than against the code's own output.
# --------------------------------------------------------------------------------------------------
def test_the_constrained_solution_is_unchanged_and_correct():
    """Linear Poisson with a manufactured solution: -Δu = 2π² sin(πx) sin(πy) + the pinned value, so
    u = sin(πx)sin(πy) + g solves it exactly with u = g on the boundary. Driven through the NONLINEAR
    (residual) path by a term that vanishes at the solution, so it is the projected residual under
    test and not the linear assembly."""
    inner, grad, _s, _t, _i = _aliases()
    g = 0.4
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.05).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    pi = float(np.pi)
    f = 2 * pi**2 * jno.np.sin(pi * X[0]) * jno.np.sin(pi * X[1])
    exact = jno.np.sin(pi * X[0]) * jno.np.sin(pi * X[1]) + g
    # `(u - exact)**3` is zero at the solution (so it does not perturb the answer) but is nonlinear in
    # u, which routes the form to the residual path this change touches.
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - f * phi + (u - exact) ** 3 * phi, u(*cb) - g])
    sol = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    ref = np.sin(pi * pts[:, 0]) * np.sin(pi * pts[:, 1]) + g
    err = np.abs(sol - ref).max()
    assert err < 5e-3, f"the constrained solution is wrong against the manufactured one: {err:.3e}"
