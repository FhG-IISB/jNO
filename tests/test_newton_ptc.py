"""``newton(ptc=dtau0)``: pseudo-transient continuation (Kelley & Keyes, SIAM J. Numer. Anal. 35(2), 1998, Sec. 2) --
each step solves ``(V / dtau_k + J) delta = -r`` with ``V = |diag J|`` and the switched evolution relaxation
``dtau_k = dtau0 * ||r0|| / ||r_k||``, so far from the root it is a damped, diagonally stabilised step and near it pure
Newton. The residual is unchanged, so is the root and its implicit gradient.

Oracles: ``atan(u) = 0`` from ``u0 = 3``, where plain Newton diverges (it does for ``|u0| > 1.3917``); a row with a zero
diagonal (a constraint, like a Taylor-Hood pressure row) that the pseudo-time term must leave alone; ``u^3 = p``, whose root
``p^(1/3)`` has the gradient ``p^(-2/3) / 3``; and a FEM reaction-diffusion form through ``jno.fem``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


import jno  # noqa: E402
from jno.utils.solver.newton_krylov import LAST_NEWTON_STATS, _ptc_shift, newton_direct  # noqa: E402

_dense = lambda A, b: jnp.linalg.solve(A, b)  # noqa: E731


def _atan(u):
    return jnp.arctan(jnp.asarray(u))


def _atan_jac(u):
    return jnp.diag(1.0 / (1.0 + jnp.asarray(u).reshape(-1) ** 2))


def test_plain_newton_diverges_on_atan_from_3_and_ptc_converges_to_the_root():
    with pytest.raises(RuntimeError):
        newton_direct(_atan, _atan_jac, jnp.array([3.0]), linear_solve=_dense, max_steps=50)
    u = newton_direct(
        _atan, _atan_jac, jnp.array([3.0]), linear_solve=_dense, max_steps=50, ptc=0.1, rtol=1e-12, atol=1e-14
    )
    assert abs(float(u[0])) < 1e-12
    assert LAST_NEWTON_STATS["converged"] and LAST_NEWTON_STATS["steps"] < 30


def test_a_step_that_raises_the_residual_is_rejected_and_dtau_shrinks():
    """atan from u0 = 3 with an over-large dtau0 = 10: the first steps are near-Newton and overshoot (|r| would grow);
    the safeguard rejects them, shrinks dtau 4x each time, and still converges. Without rejection (the earlier SER
    rule) dtau = 10 on atan from 3 diverges like plain Newton."""
    u = newton_direct(
        _atan, _atan_jac, jnp.array([3.0]), linear_solve=_dense, max_steps=60, ptc=10.0, rtol=1e-12, atol=1e-14
    )
    assert abs(float(u[0])) < 1e-12 and LAST_NEWTON_STATS["converged"]


def test_a_row_with_no_diagonal_gets_no_pseudo_time():
    """r = (atan(u) + lam, u - 1/2): the constraint row has J = (1, 0); its update must solve the constraint exactly, so
    after the first step u = 1/2 whatever dtau is. A time term on that row would leave u short of 1/2."""
    res = lambda w: jnp.stack([jnp.arctan(w[0]) + w[1], w[0] - 0.5])  # noqa: E731
    jac = lambda w: jnp.array([[1.0 / (1.0 + w[0] ** 2), 1.0], [1.0, 0.0]])  # noqa: E731
    shifted = _ptc_shift(jac(jnp.array([3.0, 0.0])), 1.0 / 1e-3)
    assert float(shifted[1, 1]) == 0.0 and np.isclose(float(shifted[0, 0]), 0.1 * 1001)
    w = newton_direct(res, jac, jnp.array([3.0, 0.0]), linear_solve=_dense, max_steps=60, ptc=1e-3, rtol=1e-12, atol=1e-14)
    assert np.allclose(np.asarray(w), [0.5, -np.arctan(0.5)], atol=1e-12)


def test_the_shift_of_a_sparse_tangent_counts_duplicate_diagonal_entries_once():
    from jax.experimental.sparse import BCOO

    # diag entry (0,0) stored twice (2 + 1 = 3), (1,1) = -4, row 2 has no diagonal entry
    J = BCOO((jnp.array([2.0, 1.0, -4.0, 5.0]), jnp.array([[0, 0], [0, 0], [1, 1], [2, 0]])), shape=(3, 3))
    S = np.asarray(_ptc_shift(J, 0.5).todense())
    assert np.allclose(S, np.asarray(J.todense()) + np.diag([1.5, 2.0, 0.0]))


def test_ptc_keeps_the_root_and_its_gradient():
    def root(p):
        res = lambda u: jnp.asarray(u) ** 3 - p  # noqa: E731
        jac = lambda u: jnp.diag(3.0 * jnp.asarray(u).reshape(-1) ** 2)  # noqa: E731
        return newton_direct(res, jac, jnp.array([5.0]), linear_solve=_dense, ptc=0.5, rtol=1e-13, atol=1e-14)[0]

    p = 2.0
    assert np.isclose(float(root(p)), p ** (1 / 3), rtol=1e-12)
    assert np.isclose(float(jax.grad(root)(p)), p ** (-2 / 3) / 3, rtol=1e-9)


def test_through_jno_fem_ptc_solves_what_plain_newton_cannot():
    """grad u . grad v + 50 (atan(u) - 1/2) v = 0 (natural edges, so u = tan(1/2) everywhere): a stiff atan reaction, from u0 = 4: the form's own Newton
    diverges, newton(direct=True, ptc=1) reaches the same root as a damped Newton that is given enough steps."""
    pytest.importorskip("meshio")
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain()
    u, v = d.fem_symbols(names=("u", "v"), order=1)
    x, y, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + 50.0 * (jno.np.arctan(ui) - 0.5) * vi])
    x0 = np.full(d.mesh.points.shape[0], 4.0)
    with pytest.raises(RuntimeError):
        fem.solve(x0=x0, nonlinear=jno.solve.newton(direct=True, max_steps=40))
    got = np.asarray(fem.solve(x0=x0, nonlinear=jno.solve.newton(direct=True, ptc=1.0, max_steps=40))).reshape(-1)
    steps = fem.stats["nonlinear"]["steps"]
    ref = np.asarray(fem.solve(x0=x0, nonlinear=jno.solve.newton(direct=True, damping=0.1, max_steps=600))).reshape(-1)
    assert np.allclose(got, np.tan(0.5), atol=1e-8) and np.allclose(got, ref, atol=1e-8)
    assert steps < 40


def test_ptc_is_refused_where_it_cannot_apply():
    with pytest.raises(ValueError, match="ptc"):
        jno.solve.newton(ptc=0.0)
    with pytest.raises(ValueError, match="matrix-free"):
        jno.solve.newton(ptc=1.0, direct=False)
    with pytest.raises(ValueError, match="reuse"):
        jno.solve.newton(ptc=1.0, reuse=True)
