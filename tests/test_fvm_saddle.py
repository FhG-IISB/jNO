"""``jno.fvm`` saddle pairs: incompressible Navier-Stokes and Darcy flow, with the Rhie-Chow mass flux."""

import jax
import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

PI = np.pi
GHIA_Y = np.array(
    [0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547]
)
GHIA_U_RE100 = np.array(
    [
        0.84123,
        0.78871,
        0.73722,
        0.68717,
        0.23151,
        0.00332,
        -0.13641,
        -0.20581,
        -0.21090,
        -0.15662,
        -0.10150,
        -0.06434,
        -0.04775,
        -0.04192,
        -0.03717,
    ]
)  # Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387, Table I


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _navier_stokes(d, nu, walls):
    """Momentum and continuity as written for jno.fdm; ``walls`` are the velocity conditions."""
    x, y, _ = d.variable("interior", split=True)
    U, P = d.unknown.vector(2), d.unknown()
    Ui, pi = U.bind(x=x, y=y), P.bind(x=x, y=y)
    u, v = Ui[0], Ui[1]
    s = jno.fv()
    momentum = (u * Ui).d(x, scheme=s) + (v * Ui).d(y, scheme=s) - nu * (Ui.xx + Ui.yy) + jnn.stack([pi.x, pi.y], axis=-1)
    return [momentum, Ui[0].x + Ui[1].y, *walls(U, P)]


def test_the_lid_driven_cavity_matches_ghia_at_re_100():
    """The classical cavity with a constant lid: a cell-centred scheme has no corner node, so the lid needs no
    regularisation."""
    n = 32
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=n).quad().domain()
    d.tag("lid", lambda x, y: y > 1 - 1e-9)
    d.tag("wall", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9))
    d.point_region("ppin", (0.0, 0.0))
    xl, yl, _ = d.variable("lid", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    xp, yp, _ = d.variable("ppin", split=True)

    def walls(U, P):
        return [U(xl, yl) - jnn.stack([1.0 + 0.0 * xl, 0.0 * xl], axis=-1), U(xw, yw) - 0.0, P(xp, yp) - 0.0]

    prob = jno.fvm(_navier_stokes(d, 0.01, walls))
    assert prob._saddle == {1: 0}
    sol = np.asarray(prob.solve())
    c = prob.centroids
    col = np.abs(c[:, 0] - 0.5) < 1.0 / n  # the two cell columns either side of x = 0.5
    ys = np.unique(np.round(c[col, 1], 12))
    us = np.array([sol[0][col][np.isclose(c[col, 1], t)].mean() for t in ys])
    assert np.abs(np.interp(GHIA_Y, ys, us) - GHIA_U_RE100).max() < 6e-3
    continuity = np.asarray(prob._residual_fn()(jax.numpy.asarray(sol.reshape(-1))))[2 * prob._N :]
    assert np.abs(continuity).max() < 1e-10


@pytest.mark.slow
def test_kovasznay_flow_converges():
    Re = 40.0
    lam = Re / 2 - np.sqrt(Re**2 / 4 + 4 * PI**2)
    errs = []
    for n in (16, 32):
        d = jno.shape.rect(-0.5, -0.5, 1.0, 1.5).structured(n=n).quad().domain()
        d.point_region("ppin", (-0.5, -0.5))
        xb, yb, _ = d.variable("boundary", split=True)
        xp, yp, _ = d.variable("ppin", split=True)

        def walls(U, P):
            ub = 1 - jnn.exp(lam * xb) * jnn.cos(2 * PI * yb)
            vb = lam / (2 * PI) * jnn.exp(lam * xb) * jnn.sin(2 * PI * yb)
            return [U(xb, yb) - jnn.stack([ub, vb], axis=-1), P(xp, yp) - 0.5 * (1 - np.exp(-lam))]

        prob = jno.fvm(_navier_stokes(d, 1 / Re, walls))
        sol, c = np.asarray(prob.solve()), prob.centroids
        ue = 1 - np.exp(lam * c[:, 0]) * np.cos(2 * PI * c[:, 1])
        ve = lam / (2 * PI) * np.exp(lam * c[:, 0]) * np.sin(2 * PI * c[:, 1])
        errs.append(np.sqrt(((sol[0] - ue) ** 2 + (sol[1] - ve) ** 2).mean()))
    assert errs[1] < 2e-2 and errs[0] / errs[1] > 3.5


def test_darcy_flow_is_a_second_saddle_pair():
    """``u + ∇p = 0, ∇·u = f`` with ``p = 0`` on the boundary: the same detection and the same Rhie-Chow
    weight (its ``a_P`` is the drag term here, there is no convection or diffusion), second order."""
    errs = []
    for n in (16, 32):
        d = jno.shape.rect(0, 0, 1, 1, size=1.0 / n).domain()
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        U, P = d.unknown.vector(2), d.unknown()
        Ui, pi = U.bind(x=x, y=y), P.bind(x=x, y=y)
        f = 2 * PI**2 * jnn.sin(PI * x) * jnn.sin(PI * y)
        prob = jno.fvm([Ui + jnn.stack([pi.x, pi.y], axis=-1), Ui[0].x + Ui[1].y - f, P(xb, yb) - 0.0])
        sol, c = np.asarray(prob.solve()), prob.centroids
        pe = np.sin(PI * c[:, 0]) * np.sin(PI * c[:, 1])
        errs.append(np.linalg.norm(sol[2] - pe) / np.linalg.norm(pe))
    assert errs[0] < 5e-3 and errs[0] / errs[1] > 3.5


def test_a_constraint_must_carry_one_unknown():
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain()
    x, y, _ = d.variable("interior", split=True)
    a, b, c = d.unknown(), d.unknown(), d.unknown()
    ai, bi, ci = a.bind(x=x, y=y), b.bind(x=x, y=y), c.bind(x=x, y=y)
    with pytest.raises(ValueError, match="constraint"):
        jno.fvm([ai.xx + bi.x, ai.x + ci.y, ci.xx + ai])  # equation 1 has no b, and carries a and c


def test_a_krylov_solve_with_the_simple_schur_saddle_preconditioner():
    """``jno.precond.saddle(schur="simple")`` builds ``D − C diag(A)⁻¹ B`` from the assembled operator: gmres
    with it reaches the direct solution of a Darcy saddle system."""
    d = jno.shape.rect(0, 0, 1, 1, size=1 / 16).domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    U, P = d.unknown.vector(2), d.unknown()
    Ui, pi = U.bind(x=x, y=y), P.bind(x=x, y=y)
    cons = [
        Ui + jnn.stack([pi.x, pi.y], axis=-1),
        Ui[0].x + Ui[1].y - 2 * PI**2 * jnn.sin(PI * x) * jnn.sin(PI * y),
        P(xb, yb) - 0.0,
    ]
    a = np.asarray(jno.fvm(cons).solve())
    b = np.asarray(jno.fvm(cons).solve(linear=jno.solve.gmres(tol=1e-12), precond=jno.precond.saddle(schur="simple")))
    np.testing.assert_allclose(a, b, atol=1e-10)
