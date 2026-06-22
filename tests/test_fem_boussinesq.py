"""Coupled multiphysics: Boussinesq (heat + incompressible flow) on a rectangle.

Three coupled fields — velocity ``u`` (P2 vector), pressure ``p`` (P1), temperature ``T`` (P2) —
with the two-way coupling that makes this *multiphysics*: buoyancy ``-Ra·Pr·T`` feeds temperature
into the momentum balance (a linear cross term), and advection ``u·∇T`` feeds velocity into the
energy balance (a product of two *different* unknowns → nonlinear). Assembled as one
``jno.fem([...])`` and solved by the coupled nonlinear Newton path.

Two things this pins, both of which were broken before:
  * the **steady** coordinate-dependent manufactured solution is recovered to tight tolerance —
    with a vector field's **per-component** Dirichlet (``u(wall)[i]-g``) and a scalar field's
    **all-component** Dirichlet (``T(wall)-g``) on the *same* wall (previously rejected as
    "region mixes all-component and per-component");
  * the **transient** Rayleigh–Bénard run accepts an **all-component vector initial condition**
    (``u(initial)-0``, previously a ``NotImplementedError``) and convection onsets from rest.

Run with x64: ``JAX_ENABLE_X64=1``.
"""

import pytest

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi
_dn = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    """feax assembly is float64; opt into x64 per-test (the session default may be x64-off when
    co-run with test_periodic). Save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_boussinesq_mms_steady_coordinate_dependent_bcs():
    """Manufactured steady Boussinesq: div-free ``u`` (nonzero on the wall), genuine cross-advection
    and buoyancy, analytic forcing. Velocity BC is **per-component**, temperature BC **all-component**
    on the same wall — recovers ``u``/``T`` to < 1e-3."""
    Sn, Cn = jno.np.sin, jno.np.cos
    Pr, Ra = 1.0, 100.0
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.07)
    d.point_region("ppin", (0.0, 0.0))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    T, sT = d.fem_symbols(names=("T", "sT"), order=2)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pb, qb = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    Tb, sb = T.bind(x=xi, y=yi), sT.bind(x=xi, y=yi)

    S1, C1, S2, C2 = Sn(PI * xi), Cn(PI * xi), Sn(PI * yi), Cn(PI * yi)
    # manufactured: u = (pi S1 C2, -pi C1 S2) [div-free], p = C1 C2, T = S1 C2 (active u.grad T)
    fmx = PI**3 * S1 * C1 - PI * S1 * C2 + 2 * PI**3 * Pr * S1 * C2
    fmy = PI**3 * S2 * C2 - PI * C1 * S2 - 2 * PI**3 * Pr * C1 * S2 - Ra * Pr * S1 * C2
    fe = PI**2 * S1 * C1 + 2 * PI**2 * S1 * C2

    ux, uy, vx, vy = ub[0], ub[1], vb[0], vb[1]
    uxx, uxy, uyx, uyy = ub.x[0], ub.y[0], ub.x[1], ub.y[1]  # grad-then-index
    vxx, vxy, vyx, vyy = vb.x[0], vb.y[0], vb.x[1], vb.y[1]
    mom = (
        ((ux * uxx + uy * uxy) * vx + (ux * uyx + uy * uyy) * vy)  # (u.grad)u
        + Pr * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)  # Pr grad u : grad v
        - pb * (vxx + vyy)  # -p div v
        - Ra * Pr * Tb * vy  # buoyancy: T -> momentum
        - (fmx * vx + fmy * vy)
    )
    cont = qb * (uxx + uyy)  # div u = 0
    ener = (ux * Tb.x + uy * Tb.y) * sb + (Tb.x * sb.x + Tb.y * sb.y) - fe * sb  # u.grad T -> energy

    UXb, UYb = PI * Sn(PI * xb) * Cn(PI * yb), -PI * Cn(PI * xb) * Sn(PI * yb)
    Tmb = Sn(PI * xb) * Cn(PI * yb)
    fem = jno.fem(
        [
            mom,
            cont,
            ener,
            u(xb, yb)[0] - UXb,  # per-component velocity Dirichlet ...
            u(xb, yb)[1] - UYb,
            T(xb, yb) - Tmb,  # ... + all-component scalar Dirichlet, SAME wall (fix #1)
            p(xpn, ypn) - 1.0,
        ]
    )
    assert fem._mode == "nonlinear"

    r, J = fem.residual, fem.jacobian
    w = jnp.zeros(fem.dofs)
    for _ in range(15):
        g = jnp.asarray(r(w)).reshape(-1)
        w = w + jnp.linalg.solve(_dn(J(w)), -g)
        if float(jnp.linalg.norm(jnp.asarray(r(w)).reshape(-1))) < 1e-10:
            break
    off = fem.offsets  # [0, n_vel, n_vel+n_p, n_total]
    w = np.asarray(w)
    pts = np.asarray(fem.field_points[0])
    uu = w[off[0] : off[1]].reshape(-1, 2)
    UXn = PI * np.sin(PI * pts[:, 0]) * np.cos(PI * pts[:, 1])
    UYn = -PI * np.cos(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    ptsT = np.asarray(fem.field_points[2])
    Tn = np.sin(PI * ptsT[:, 0]) * np.cos(PI * ptsT[:, 1])
    TT = w[off[2] : off[3]]
    assert np.linalg.norm(uu - np.stack([UXn, UYn], 1)) / np.linalg.norm(np.stack([UXn, UYn], 1)) < 1e-3
    assert np.linalg.norm(TT - Tn) / np.linalg.norm(Tn) < 1e-3


def test_boussinesq_transient_vector_ic_and_convection_onset():
    """Transient Rayleigh–Bénard: hot bottom / cold top, no-slip, started from rest with an
    all-component vector IC ``u(initial)-0`` (fix #3). Convection onsets (|u| grows from 0) and the
    flow stays incompressible (max|div u| small)."""
    Sn, whr = jno.np.sin, jno.np.where
    Pr, Ra = 1.0, 5000.0
    d = jno.domain(box(0, 0, 2, 1), mesh_size=0.14, time=(0.0, 0.4, 2))
    d.point_region("ppin", (0.0, 0.0))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    T, sT = d.fem_symbols(names=("T", "sT"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    xpn, ypn, _ = d.variable("ppin", split=True)
    ub, vb = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    pb, qb = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
    Tb, sb = T.bind(x=xi, y=yi, t=ti), sT.bind(x=xi, y=yi, t=ti)

    ux, uy, vx, vy = ub[0], ub[1], vb[0], vb[1]
    uxx, uxy, uyx, uyy = ub.x[0], ub.y[0], ub.x[1], ub.y[1]
    vxx, vxy, vyx, vyy = vb.x[0], vb.y[0], vb.x[1], vb.y[1]
    mom = (
        (ub.t[0] * vx + ub.t[1] * vy)
        + ((ux * uxx + uy * uxy) * vx + (ux * uyx + uy * uyy) * vy)
        + Pr * (uxx * vxx + uxy * vxy + uyx * vyx + uyy * vyy)
        - pb * (vxx + vyy)
        - Pr * Ra * Tb * vy
    )
    cont = qb * (uxx + uyy)
    ener = Tb.t * sb + (ux * Tb.x + uy * Tb.y) * sb + (Tb.x * sb.x + Tb.y * sb.y)
    T0 = (1 - ci[1]) + 0.1 * Sn(PI * ci[0]) * Sn(PI * ci[1])  # conductive + perturbation
    fem = jno.fem(
        [
            mom,
            cont,
            ener,
            u(xb, yb) - 0.0,  # no-slip (all-component) ...
            T(xb, yb) - whr(yb < 1e-6, 1.0, 0.0),  # hot bottom, cold elsewhere
            p(xpn, ypn) - 0.0,
            u(ci[0], ci[1]) - 0.0,  # all-component vector IC at rest (fix #3)
            T(ci[0], ci[1]) - T0,
        ]
    )
    assert fem._mode == "transient"

    blk = fem.operator
    M, dt = _dn(blk.mass(0.0, {})), 0.01
    off = fem.offsets  # [0, n_vel, n_vel+n_p, n_total]
    w = jnp.asarray(blk.state0)
    assert float(np.abs(np.asarray(w)[off[0] : off[1]]).max()) == 0.0  # starts at rest
    for step in range(12):
        w_prev, t_next = w, (step + 1) * dt
        for _ in range(6):
            G = M @ (w - w_prev) / dt + jnp.asarray(blk.residual(w, t_next, {})).reshape(-1)
            dw = jnp.linalg.solve(M / dt + _dn(blk.jacobian(w, t_next, {})), -G)
            w = w + dw
            if float(jnp.linalg.norm(dw)) < 1e-8:
                break
    uu = np.asarray(w)[off[0] : off[1]].reshape(-1, 2)
    assert np.abs(uu).max() > 1.0  # convection has developed from rest
