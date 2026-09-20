"""A drop rings at Lamb's frequency: the sharp-interface route against a DYNAMIC oracle.

The static tests (``test_fem_free_surface_tension.py``) pin the capillary traction at equilibrium. This
one pins the motion: an ellipse released from rest, on a mesh its own fluid carries, must oscillate about
the circle at the inviscid 2-D frequency and decay at the viscous rate,

    omega^2 = n(n^2-1) sigma / (rho R^3)   (Rayleigh 1879; n = 2)
    gamma   = 2 n(n-1) nu / R^2            (Lamb's dissipation method)

so the whole moving-mesh stack is under test at once: the capillary traction, the ALE convection
``u - w`` with the mesh velocity read from ``xi.d(ti)``, the kinematic laws reading the solved velocity
through a frozen vector field, and the nodes riding with the mesh.

The shape is measured as a Fourier mode of ``r(theta)`` and fitted with a damped sinusoid. A bounding-box
aspect ratio and extremum counting are NOT adequate: on this signal they returned +7.4/s and -1.6/s for
the SAME run over different spans.

**The stabilisation must be scaled for this regime.** ``tau`` as written for advection-dominated flow
(the Kovasznay/stabilised-flow recipe) contributes about ten times the physical damping here and drives a
spurious n = 4 mode, because a capillary drop is nearly inviscid and nearly stagnant. ``TAU_SCALE`` below
is what makes the physics come out; the second test pins the trap.
"""

import jax
import numpy as np
import pytest

import jno

RHO, SIGMA, ETA, C_I = 1.0, 10.0, 0.01, 36.0
NU = ETA / RHO
REQ, K, H, DT, NSTEP, NPTS = 0.2449, 1.05, 0.05, 5e-4, 400, 48
TAU_SCALE = 1e-4
W_LAMB = np.sqrt(6.0 * SIGMA / (RHO * REQ**3))  # n = 2
GAMMA_VISC = 4.0 * NU / REQ**2  # 2n(n-1) nu / R^2 at n = 2


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _march(tau_scale):
    """An ellipse of equal-area radius REQ, released from rest, marched on its own moving mesh."""
    inner, symgrad = jno.np.inner, jno.np.symgrad
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731
    ax, bx = REQ * np.sqrt(K), REQ / np.sqrt(K)
    pts = [(ax * np.cos(t), bx * np.sin(t)) for t in np.linspace(0, 2 * np.pi, NPTS, endpoint=False)]
    d = jno.shape.polygon(pts, size=H).domain(time=(0.0, NSTEP * DT, NSTEP + 1))
    nnode = int(np.asarray(d.mesh.points).shape[0])
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xs, ys, ts, nx, ny = d.variable("boundary", normals=True, split=True)
    x0, y0, _t0 = d.variable("initial", split=True)
    B = dict(x=xi, y=yi, t=ti)
    ub, vv, pp, qq = u.bind(**B), v.bind(**B), p.bind(**B), q.bind(**B)
    vs = v.bind(x=xs, y=ys)
    D = lambda w: symgrad(w, [xi, yi])  # noqa: E731
    ndv = lambda f, i: nx * f.x[i] + ny * f.y[i]  # noqa: E731
    div_G = lambda f: f.x[0] + f.y[1] - (nx * ndv(f, 0) + ny * ndv(f, 1))  # noqa: E731
    G = d.cell_metric
    gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
    # u - w: the ALE convective velocity. Componentwise, because a vector cannot be built from two scalar
    # expressions (jno.np.stack gives a layout `inner` will not broadcast against a gradient).
    c0, c1 = ub[0] - xi.d(ti), ub[1] - yi.d(ti)
    conv = lambda i: c0 * ub.x[i] + c1 * ub.y[i]  # noqa: E731
    tau = jno.lag(tau_scale * ((2.0 / DT) ** 2 + gG(ub) + C_I * NU**2 * inner(G, G, n_contract=2)) ** -0.5)
    r0, r1 = ub.t[0] + conv(0) + pp.x / RHO, ub.t[1] + conv(1) + pp.y / RHO
    momentum = (
        RHO * (ub.t[0] * vv[0] + ub.t[1] * vv[1])
        + RHO * (conv(0) * vv[0] + conv(1) * vv[1])
        + 2.0 * ETA * ddot(D(ub), D(vv))  # the free-surface viscous term; grad:grad would be wrong
        - pp * (vv.x[0] + vv.y[1])
        + tau * ((c0 * vv.x[0] + c1 * vv.y[0]) * r0 + (c0 * vv.x[1] + c1 * vv.y[1]) * r1)
    )
    continuity = -qq * (ub.x[0] + ub.y[1]) - tau * (qq.x * r0 + qq.y * r1)
    uf = u.bind(x=xs, y=ys).freeze(np.zeros((nnode, 2)))  # the solved velocity, delivered each step
    fem = jno.fem(
        [
            momentum,
            continuity,
            SIGMA * div_G(vs),  # the capillary traction sets the pressure level: no pin, no Dirichlet
            u(x0, y0)[0] - 0.0,
            u(x0, y0)[1] - 0.0,
            xs.d(ts) - uf[0],  # the surface rides with the fluid
            ys.d(ts) - uf[1],
        ]
    )
    return fem.solve(nonlinear=jno.solve.newton(direct=True))


def _mode2(traj):
    """The n = 2 Fourier amplitude of r(theta) on the free surface, frame by frame."""
    from jno.utils.solver.fem_adapt import _boundary_edges_from_triangles

    bnd = np.unique(np.asarray(_boundary_edges_from_triangles(np.asarray(traj.meshes[0][1]))).reshape(-1))
    a2 = []
    for m in traj.meshes:
        X = np.asarray(m[0])[bnd]
        Xc = X - X.mean(0)
        th = np.arctan2(Xc[:, 1], Xc[:, 0])
        r = np.hypot(Xc[:, 0], Xc[:, 1])
        a2.append(2.0 * np.mean(r * np.cos(2.0 * th)) / r.mean())
    return np.asarray(traj.times), np.asarray(a2)


def _fit(t, a2):
    """``(omega, gamma, relative residual)`` of a damped sinusoid fitted to the whole series."""
    from scipy.optimize import curve_fit

    model = lambda tt, A, g, w, ph, c: A * np.exp(-g * tt) * np.cos(w * tt + ph) + c  # noqa: E731
    popt, _ = curve_fit(model, t, a2, p0=[a2[0], GAMMA_VISC, W_LAMB, 0.0, 0.0], maxfev=40000)
    resid = float(np.sqrt(np.mean((model(t, *popt) - a2) ** 2)) / np.abs(a2).max())
    return abs(float(popt[2])), float(popt[1]), resid


def test_a_drop_rings_at_lambs_frequency_and_decays_at_the_viscous_rate():
    w, g, resid = _fit(*_mode2(_march(TAU_SCALE)))
    assert resid < 0.05, f"not a damped sinusoid (residual {resid:.3f})"
    assert w == pytest.approx(W_LAMB, rel=0.03), f"omega {w:.2f} against Lamb {W_LAMB:.2f}"
    assert g == pytest.approx(GAMMA_VISC, rel=0.30), f"gamma {g:.3f} against viscous {GAMMA_VISC:.3f}"


def test_the_advection_stabilisation_over_damps_a_capillary_drop():
    """The same drop with the stabilised-flow recipe's ``tau``: it does not ring at all.

    Measured here: ``omega`` collapses to 0.23 against Lamb's 63.91 -- the fit degenerates to a pure decay
    -- and the amplitude creeps through zero (0.0732 -> -0.0099) instead of oscillating about it, at 4.5x
    the viscous rate. A capillary drop is nearly inviscid and nearly stagnant, the opposite of the regime
    that ``tau`` was built for. Nothing warns; only the oracle catches it.
    """
    w, g, _resid = _fit(*_mode2(_march(1.0)))
    assert w < 0.5 * W_LAMB, f"omega {w:.2f}: the over-damping trap seems gone (Lamb {W_LAMB:.2f})"
    assert g > 2.0 * GAMMA_VISC, f"gamma {g:.3f} against viscous {GAMMA_VISC:.3f}"
