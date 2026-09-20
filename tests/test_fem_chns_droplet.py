"""Two-phase flow as terms: the Cahn–Hilliard–Navier–Stokes (diffuse-interface) droplet.

No new API -- the model is the term list. A phase field ``phi`` (+1 liquid, -1 gas) and its chemical
potential ``mu`` ride beside a Taylor–Hood velocity/pressure pair:

    rho (u_t + (u.grad)u) = -grad p + eta lap u + mu grad(phi),        div u = 0
    phi_t + u.grad(phi)   = div(M grad mu),     mu = lam (-lap phi + (phi^3 - phi)/eps^2)

with surface tension ``sigma = (2 sqrt2 / 3) lam / eps`` (Jacqmin, J. Comput. Phys. 155 (1999) 96;
Yue, Feng, Liu & Shen, J. Fluid Mech. 515 (2004) 293).

Oracles:
  * Laplace: a static drop carries the pressure jump ``sigma / R`` (2-D), to O(eps/R);
  * Gibbs–Thomson: its chemical potential settles at ``sigma / (2R)``;
  * a static drop stays static -- the capillary force is balanced, so the spurious currents die out;
  * conservation: ``int phi`` is conserved to solver precision on a fixed mesh. Summing the phi equation
    over the P1 partition of unity leaves only ``int phi div u``, and phi lies in the P1 pressure-test
    space, so the discrete continuity equation makes that term exactly zero;
  * two nearby drops merge into one, and relax toward a circle of the combined area.
"""

import jax
import numpy as np
import pytest

import jno

SIGMA, ETA, RHO = 1.0, 0.1, 1.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _chns(phi0, *, h, eps, mobility, t_end, n_steps):
    """Build the CHNS system on the unit square with no-slip walls; ``phi0(x, y)`` is the initial phase."""
    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    dot = lambda a, b: inner(a, b, n_contract=1)  # noqa: E731
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731
    lam = 3.0 * SIGMA * eps / (2.0 * np.sqrt(2.0))

    d = jno.shape.rect(0, 0, 1, 1, size=h).domain(time=(0.0, t_end, n_steps + 1))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    c, s = d.fem_symbols(names=("phi", "psi"), order=1)
    m, w = d.fem_symbols(names=("mu", "chi"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    X, B = [xi, yi], dict(x=xi, y=yi, t=ti)
    u_, v_, p_, q_ = u.bind(**B), v.bind(**B), p.bind(**B), q.bind(**B)
    phi, psi, mu, chi = c.bind(**B), s.bind(**B), m.bind(**B), w.bind(**B)
    grad_u, grad_v = grad(u, X), grad(v, X)

    momentum = (
        RHO * dot(u_.t, v_)
        + RHO * dot(dot(grad_u, u_), v_)
        + ETA * ddot(grad_u, grad_v)
        - p_ * trace(grad_v)
        - mu * (phi.x * v_[0] + phi.y * v_[1])  # the capillary force mu grad(phi)
    )
    continuity = -q_ * trace(grad_u)
    cahn_hilliard = phi.t * psi + (u_[0] * phi.x + u_[1] * phi.y) * psi + mobility * (mu.x * psi.x + mu.y * psi.y)
    chemical_potential = mu * chi - lam * (phi * phi * phi - phi) / eps**2 * chi - lam * (phi.x * chi.x + phi.y * chi.y)
    terms = [
        momentum,
        continuity,
        cahn_hilliard,
        chemical_potential,
        u(xb, yb)[0] - 0.0,
        u(xb, yb)[1] - 0.0,
        p.pin(),
        u(x0, y0)[0] - 0.0,
        u(x0, y0)[1] - 0.0,
        c(x0, y0) - phi0(x0, y0),
    ]
    fem = jno.fem(terms)
    sol = fem.solve(nonlinear=jno.solve.newton(direct=True))
    traj = np.asarray(jno.core([sol.mse]).eval([sol]))
    from jno._fem import _field_names

    names = [_field_names(terms)[k] for k in fem._block_field_keys]
    off = list(fem.offsets)
    fields = {nm: (slice(off[i], off[i + 1]), np.asarray(fem.field_points[i])[:, :2]) for i, nm in enumerate(names)}
    return traj, fields, d


def _tanh_drop(cx, cy, R, eps):
    return lambda x, y: jno.np.tanh((R - jno.np.sqrt((x - cx) ** 2 + (y - cy) ** 2)) / (np.sqrt(2.0) * eps))


def _p1_integral(d, values):
    """int of a P1 nodal field by the vertex rule (exact for P1)."""
    pts = np.asarray(d.mesh.points)[:, :2]
    tri = np.asarray(d.mesh.cells_dict["triangle"])
    P = pts[tri]
    area = 0.5 * np.abs(
        (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1])
    )
    return float(np.sum(area * values[tri].mean(axis=1)))


def test_a_static_drop_obeys_laplace_and_gibbs_thomson():
    R, eps = 0.25, 0.03
    traj, f, _ = _chns(_tanh_drop(0.5, 0.5, R, eps), h=0.02, eps=eps, mobility=1e-2, t_end=1.0, n_steps=10)
    final = traj[-1]
    (sp, pp), (sm, _), (su, _) = f["p"], f["mu"], f["u"]
    r = np.hypot(pp[:, 0] - 0.5, pp[:, 1] - 0.5)
    dp = final[sp][r < R - 4 * eps].mean() - final[sp][r > R + 4 * eps].mean()
    mu0 = final[sm].mean()
    assert abs(dp / (SIGMA / R) - 1.0) < 0.06, f"Laplace jump {dp:.4f} vs sigma/R = {SIGMA / R:.4f}"
    assert abs(mu0 / (SIGMA / (2 * R)) - 1.0) < 0.06, f"Gibbs–Thomson mu0 {mu0:.4f} vs sigma/2R = {SIGMA / (2 * R):.4f}"
    assert abs(dp - 2.0 * mu0) < 0.02 * dp, "the pressure jump must equal mu0 times the phase jump of 2"
    # the spurious currents die out as the drop equilibrates: small, and falling over the last frames
    speed = [np.abs(traj[j][su]).max() for j in range(len(traj))]
    assert speed[-1] < 1e-4 * (SIGMA / ETA) and speed[-1] < speed[-3], (
        f"spurious currents {speed[-3]:.1e} -> {speed[-1]:.1e}"
    )


def test_two_drops_merge_and_conserve_phase():
    R, eps = 0.18, 0.04
    # rims 0.08 apart -- about 1.4 interface widths -- so the drops start separate (phi < 0 in the gap)
    phi0 = lambda x, y: _tanh_drop(0.28, 0.5, R, eps)(x, y) + _tanh_drop(0.72, 0.5, R, eps)(x, y) + 1.0  # noqa: E731
    traj, f, d = _chns(phi0, h=0.03, eps=eps, mobility=1e-2, t_end=0.6, n_steps=12)
    sc, pc = f["phi"]
    first, last = traj[0][sc], traj[-1][sc]
    # conservation: exact up to solver precision on a fixed mesh (see the module docstring for why)
    drift = abs(_p1_integral(d, last) - _p1_integral(d, first))
    assert drift < 1e-8, f"int phi drifted by {drift:.2e} on a fixed mesh"
    # the gap between the drops has filled -- one drop, not two
    gap = last[np.argmin(np.hypot(pc[:, 0] - 0.5, pc[:, 1] - 0.5))]
    assert first[np.argmin(np.hypot(pc[:, 0] - 0.5, pc[:, 1] - 0.5))] < 0.0 < 0.9 < gap, f"the gap reads phi = {gap:.3f}"
    liquid = pc[last > 0.0]
    aspect = np.ptp(liquid[:, 0]) / np.ptp(liquid[:, 1])
    assert aspect < np.ptp(pc[first > 0.0][:, 0]) / np.ptp(pc[first > 0.0][:, 1]), (
        "the merged drop is not relaxing toward a circle"
    )
