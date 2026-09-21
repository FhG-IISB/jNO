"""Surface tension on a free surface: the capillary traction, written as terms, against exact oracles.

A liquid body in a void has no Dirichlet condition anywhere. Its surface states a TRACTION,
``T n = -sigma H n``, which on the weak form is the boundary term ``sigma * div_Gamma(v)`` -- and that
term alone fixes the pressure level, so the problem needs no ``p.pin()``. With an equal-order P1/P1
velocity-pressure pair the SUPG/PSPG stabilisation fills the pressure block, so the saddle structure is
regular too. Nothing here is new API: it is the ``jno.fem([...])`` list.

Oracles, all exact:

* **Laplace.** A drop at rest has ``p = sigma / R`` inside. The discrete mesh boundary is a regular
  polygon of ``N`` vertices, whose exact value is ``sigma / (R cos(pi/N))`` -- 0.18 % above ``sigma/R``
  at ``N = 53``, and that is what the discretisation must reproduce.
* **Curvature is local.** Two drops of different radii in one void each take their own ``sigma / R_i``.
* **Rigid rotation.** ``u = omega (-y, x)`` with ``p = sigma/R + rho omega^2 (r^2 - R^2) / 2`` is an
  exact free-surface Navier-Stokes solution (the viscous stress vanishes identically), so the spin must
  not decay.
* **The pseudo-traction trap.** Writing the viscous term ``eta grad u : grad v`` -- correct behind
  Dirichlet walls, and what the stabilised-flow tutorial and the CHNS tutorial use -- makes the natural
  boundary condition ``eta du/dn - p n`` instead of the true traction. At a free surface that is
  silently wrong physics: the same rotating drop loses a large part of its spin. Only
  ``2 eta D(u) : D(v)`` is right.
"""

import jax
import numpy as np
import pytest

import jno

RHO, ETA, SIGMA = 1.0, 0.1, 1.0
NU = ETA / RHO
C_I, DT, R, H = 36.0, 0.02, 0.25, 0.03


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _drop(shape, *, n_steps=3, ic=None, viscous="symgrad"):
    """The stabilised P1/P1 free-surface form on ``shape``: no Dirichlet, no pin, traction only."""
    inner, grad, trace, lap, symgrad = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian, jno.np.symgrad
    dot = lambda a, b: inner(a, b, n_contract=1)  # noqa: E731
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731

    d = shape.domain(time=(0.0, n_steps * DT, n_steps + 1))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xs, ys, _ts, nx, ny = d.variable("boundary", normals=True, split=True)
    x0, y0, _t0 = d.variable("initial", split=True)
    B = dict(x=xi, y=yi, t=ti)
    ub, vv, pp, qq = u.bind(**B), v.bind(**B), p.bind(**B), q.bind(**B)
    gu, gv, gp, gq = grad(u, [xi, yi]), grad(v, [xi, yi]), grad(p, [xi, yi]), grad(q, [xi, yi])
    vs = v.bind(x=xs, y=ys)
    adv = lambda gw, w: inner(gw, w, n_contract=1)  # noqa: E731  (w.grad)w
    D = lambda w: symgrad(w, [xi, yi])  # noqa: E731
    ndv = lambda f, i: nx * f.x[i] + ny * f.y[i]  # noqa: E731  (grad f_i).n
    div_G = lambda f: f.x[0] + f.y[1] - (nx * ndv(f, 0) + ny * ndv(f, 1))  # noqa: E731

    G = d.cell_metric
    gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
    tau = jno.lag(((2.0 / DT) ** 2 + gG(ub) + C_I * NU**2 * inner(G, G, n_contract=2)) ** -0.5)
    # The strong residual r_m = u_t + (u.grad)u - nu lap u + grad p / rho enters the SUPG/PSPG terms
    # SPLIT by temporal order: `tau*(a . r_m)` as one term would put r_m's spatial part in the mass
    # matrix, which the transient assembler refuses (weak_form_helpers.refuse_mixed_temporal_group).
    r_sp = adv(gu, ub) - NU * lap(u, [xi, yi]) + gp / RHO  # r_m without its u_t
    # The free-surface viscous term is 2*eta*D(u):D(v); `gradgrad` is the pseudo-traction this test
    # measures as wrong. Everything else is identical between the two.
    visc = 2.0 * ETA * ddot(D(ub), D(vv)) if viscous == "symgrad" else ETA * ddot(gu, gv)
    supg = tau * dot(adv(gv, ub), ub.t) + tau * dot(adv(gv, ub), r_sp)
    momentum = RHO * dot(ub.t, vv) + RHO * dot(adv(gu, ub), vv) + visc - pp * trace(gv) + supg
    continuity = -qq * trace(gu) - tau * dot(gq, ub.t) - tau * dot(gq, r_sp)
    capillary = SIGMA * div_G(vs)  # T n = -sigma H n, moved to the left-hand side
    u0 = (0.0, 0.0) if ic is None else ic(x0, y0)
    fem = jno.fem([momentum, continuity, capillary, u(x0, y0)[0] - u0[0], u(x0, y0)[1] - u0[1]])
    traj = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True)).fn())
    return fem, d, traj


def _blocks(fem, frame):
    """``(velocity (n_nodes, 2), pressure (n_nodes,), pressure node coordinates)``."""
    off = [int(o) for o in fem.offsets]  # block order is the order the fields were built in: u, then p
    uu = np.asarray(frame)[off[0] : off[1]].reshape(-1, 2)  # node-major interleaved
    pu = np.asarray(frame)[off[1] : off[2]]
    return uu, pu, np.asarray(fem.field_points[1])


def test_a_drop_at_rest_carries_the_laplace_jump():
    fem, d, traj = _drop(jno.shape.disk(0.0, 0.0, R, size=H))
    uu, pu, _pts = _blocks(fem, traj[-1])
    n_bnd = int(np.asarray(d.tag_node_mask("boundary", np.asarray(d.mesh.points))).sum())
    exact = SIGMA / (R * np.cos(np.pi / n_bnd))  # the regular polygon the mesh boundary is
    assert np.ptp(pu) < 1e-6 * exact, "the pressure is not uniform inside a drop at rest"
    assert pu.mean() == pytest.approx(exact, rel=1e-5), f"{pu.mean():.6f} against polygon {exact:.6f}"
    assert pu.mean() == pytest.approx(SIGMA / R, rel=3e-3), "not the Laplace jump sigma/R"
    assert np.abs(uu).max() < 1e-9, f"spurious currents {np.abs(uu).max():.2e}"


def test_each_drop_takes_its_own_curvature():
    """Two drops in one void, no pin: curvature is local, so p is 5/3 higher in the smaller one."""
    r2 = 0.15
    fem, _d, traj = _drop(jno.shape.disk(0.0, 0.0, R, size=H) | jno.shape.disk(0.6, 0.0, r2, size=H))
    _uu, pu, pts = _blocks(fem, traj[-1])
    big, small = pu[pts[:, 0] < 0.3], pu[pts[:, 0] > 0.3]
    assert np.ptp(big) < 1e-6 and np.ptp(small) < 1e-6
    assert big.mean() == pytest.approx(SIGMA / R, rel=3e-3)
    assert small.mean() == pytest.approx(SIGMA / r2, rel=6e-3)
    assert small.mean() / big.mean() == pytest.approx(R / r2, rel=5e-3)


def _omega(uu, pts):
    """Least-squares rigid-rotation rate of a velocity field: (x uy - y ux) / r^2 away from the centre."""
    r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
    keep = r2 > (0.5 * R) ** 2
    return float(np.mean((pts[keep, 0] * uu[keep, 1] - pts[keep, 1] * uu[keep, 0]) / r2[keep]))


def test_a_rigidly_rotating_drop_keeps_its_spin():
    """An exact free-surface solution: the viscous stress of a rigid rotation vanishes identically."""
    fem, d, traj = _drop(jno.shape.disk(0.0, 0.0, R, size=H), n_steps=5, ic=lambda x, y: (-2.0 * y, 2.0 * x))
    uu, pu, pts = _blocks(fem, traj[-1])
    assert _omega(uu, pts) == pytest.approx(2.0, abs=5e-3), "the rotation decayed"
    # p = p_surface + rho omega^2 (r^2 - R^2)/2, with the SAME polygon-corrected surface value the static
    # drop takes -- so this is exact for the discretisation, not just to the 0.18 % polygon offset.
    n_bnd = int(np.asarray(d.tag_node_mask("boundary", np.asarray(d.mesh.points))).sum())
    centre = int(np.argmin(pts[:, 0] ** 2 + pts[:, 1] ** 2))
    r2 = pts[centre, 0] ** 2 + pts[centre, 1] ** 2
    p_exact = SIGMA / (R * np.cos(np.pi / n_bnd)) + RHO * 4.0 * (r2 - R**2) / 2.0
    assert pu[centre] == pytest.approx(p_exact, rel=1e-3), f"{pu[centre]:.5f} against {p_exact:.5f}"


def test_the_pseudo_traction_form_loses_the_spin():
    """`eta grad u : grad v` is fine behind Dirichlet walls and wrong at a free surface -- silently."""
    fem_s, _ds, traj_s = _drop(jno.shape.disk(0.0, 0.0, R, size=H), n_steps=5, ic=lambda x, y: (-2.0 * y, 2.0 * x))
    fem_g, _dg, traj_g = _drop(
        jno.shape.disk(0.0, 0.0, R, size=H), n_steps=5, ic=lambda x, y: (-2.0 * y, 2.0 * x), viscous="gradgrad"
    )
    w_sym = _omega(*_blocks(fem_s, traj_s[-1])[::2])
    w_pseudo = _omega(*_blocks(fem_g, traj_g[-1])[::2])
    assert w_sym == pytest.approx(2.0, abs=5e-3)
    assert w_pseudo < 1.6, f"the pseudo-traction kept the spin ({w_pseudo:.3f}); the trap is gone?"
