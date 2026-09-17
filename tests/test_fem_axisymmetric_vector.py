"""Axisymmetric VECTOR forms, written by hand — the hoop strain is the part that has no 2-D counterpart.

``tests/test_fem_axisymmetric.py`` pins the scalar case: multiply the Cartesian integrand by ``2πr`` and
the form is complete. For a vector field that is not enough, and `docs/fem/limitations.md` lists it as one
of only two SILENT limits: a displacement ``u = (u_r, u_z)`` on a meridian also strains the hoop direction,

    eps_rr = d u_r/dr,   eps_zz = d u_z/dz,   eps_rz = (d u_r/dz + d u_z/dr)/2,   eps_qq = u_r / r

and that last one is invisible to a 2-D form. Multiplying by ``r`` is arithmetic the assembler cannot tell
from a legitimate radial coefficient, so nothing raises -- the answer is simply wrong, and plausibly so.

The oracle is the Lamé thick-walled cylinder under internal pressure (plane strain), whose radial
displacement is exact:

    u_r(r) = (1+nu) p a^2 / (E (b^2 - a^2)) * [ (1 - 2 nu) r + b^2 / r ],    u_z = 0

It is the right test because the hoop stiffness is what carries the load: drop ``eps_qq`` and the cylinder
is far too compliant, which the second test measures rather than asserts by eye. A pipe-flow test then
does the same for the axisymmetric Stokes operator, on a domain that includes the axis r = 0.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

TWO_PI = 2.0 * np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _lame_exact(r, a, b, p, E, nu):
    return (1.0 + nu) * p * a**2 / (E * (b**2 - a**2)) * ((1.0 - 2.0 * nu) * r + b**2 / r)


def _cylinder(a, b, h, p, E, nu, *, hoop=True, size=0.06, order=2):
    """Thick-walled cylinder under internal pressure, meridian ``[a, b] x [0, h]``, plane strain.

    ``hoop=False`` writes the SAME form without ``eps_qq`` -- the mistake the docs warn about.
    """
    lam, mu = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))
    d = jno.shape.rect(a, 0.0, b, h, size=size).domain()
    u, v = d.fem_symbols(value_shape=(2,), order=order)  # (u_r, u_z) on the meridian
    r, z, _ = d.variable("interior", split=True)
    ra, za, _ = d.variable("left", split=True)  # r = a, the pressurised bore
    rt, zt, _ = d.variable("top", split=True)
    rb, zb, _ = d.variable("bottom", split=True)
    ub, vb = u.bind(x=r, y=z), v.bind(x=r, y=z)

    def strains(w):
        e_rr, e_zz = w.x[0], w.y[1]
        e_rz = 0.5 * (w.y[0] + w.x[1])
        e_qq = w[0] / r if hoop else 0.0 * w[0]  # THE term with no 2-D counterpart
        return e_rr, e_zz, e_qq, e_rz

    err_u, ezz_u, eqq_u, erz_u = strains(ub)
    err_v, ezz_v, eqq_v, erz_v = strains(vb)
    tr_u, tr_v = err_u + ezz_u + eqq_u, err_v + ezz_v + eqq_v
    energy = lam * tr_u * tr_v + 2.0 * mu * (err_u * err_v + ezz_u * ezz_v + eqq_u * eqq_v + 2.0 * erz_u * erz_v)

    # Internal pressure pushes outward (+e_r) on the bore; the boundary term carries its own ring measure.
    work = -p * v.bind(x=ra, y=za)[0] * (TWO_PI * ra)
    fem = jno.fem(
        [
            energy * (TWO_PI * r),
            work,
            u(rt, zt)[1] - 0.0,  # plane strain: the ends are held, u_z = 0
            u(rb, zb)[1] - 0.0,
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2)
    return sol, np.asarray(fem.points)


def test_the_hoop_strain_gives_the_lame_cylinder():
    a, b, h, p, E, nu = 1.0, 2.0, 0.4, 5.0, 1000.0, 0.3
    sol, pts = _cylinder(a, b, h, p, E, nu)
    exact = _lame_exact(pts[:, 0], a, b, p, E, nu)
    err = np.abs(sol[:, 0] - exact).max() / np.abs(exact).max()
    assert err < 0.01, f"u_r is off the Lame solution by {100 * err:.2f} %"
    assert np.abs(sol[:, 1]).max() < 0.02 * np.abs(exact).max(), "plane strain: u_z should vanish"


def test_forgetting_the_hoop_strain_is_silently_wrong():
    """The same mesh, the same 2*pi*r measure, the same solver -- and no error of any kind."""
    a, b, h, p, E, nu = 1.0, 2.0, 0.4, 5.0, 1000.0, 0.3
    good, pts = _cylinder(a, b, h, p, E, nu, hoop=True)
    bad, _ = _cylinder(a, b, h, p, E, nu, hoop=False)
    exact = _lame_exact(pts[:, 0], a, b, p, E, nu)
    err_bad = np.abs(bad[:, 0] - exact).max() / np.abs(exact).max()
    assert err_bad > 0.10, f"dropping eps_qq should be badly wrong, but it was {100 * err_bad:.1f} %"
    assert np.abs(bad[:, 0]).max() > 1.2 * np.abs(good[:, 0]).max(), "without hoop stiffness it must be softer"


def test_axisymmetric_stokes_gives_poiseuille_flow_through_the_axis():
    """A pipe of radius R driven by a body force: u_z = G (R^2 - r^2) / (4 eta), exact for P2.

    The profile is imposed at both ends, because a traction-free end is NOT what fully developed flow
    satisfies: the parabola carries a shear traction eta u_z'(r) on a z-face. With the ends prescribed,
    a P2 velocity can represent this quadratic exactly, so a correct operator has to return it to machine
    precision -- a much sharper oracle than a tolerance.

    The meridian includes the axis r = 0, where the hoop term u_r/r would be singular if u_r were not
    zero there; the quadrature points sit strictly inside the cells, and u_r = 0 is imposed on the axis.
    """
    R, L, G, eta = 1.0, 2.0, 4.0, 1.5
    d = jno.shape.rect(0.0, 0.0, R, L, size=0.12).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)  # Taylor-Hood: no stabilisation needed
    r, z, _ = d.variable("interior", split=True)
    r_ax, z_ax, _ = d.variable("left", split=True)  # r = 0, the axis
    r_w, z_w, _ = d.variable("right", split=True)  # r = R, the wall
    r_b, z_b, _ = d.variable("bottom", split=True)  # z = 0, inlet
    r_t, z_t, _ = d.variable("top", split=True)  # z = L, outlet
    ub, vb, pb, qb = u.bind(x=r, y=z), v.bind(x=r, y=z), p.bind(x=r, y=z), q.bind(x=r, y=z)

    def strains(w):
        return w.x[0], w.y[1], w[0] / r, 0.5 * (w.y[0] + w.x[1])

    err_u, ezz_u, eqq_u, erz_u = strains(ub)
    err_v, ezz_v, eqq_v, erz_v = strains(vb)
    div_u, div_v = err_u + ezz_u + eqq_u, err_v + ezz_v + eqq_v  # the cylindrical divergence
    viscous = 2.0 * eta * (err_u * err_v + ezz_u * ezz_v + eqq_u * eqq_v + 2.0 * erz_u * erz_v)
    momentum = (viscous - pb * div_v - G * vb[1]) * (TWO_PI * r)
    continuity = -qb * div_u * (TWO_PI * r)
    fem = jno.fem(
        [
            momentum,
            continuity,
            u(r_w, z_w) - (0.0, 0.0),  # no slip at the wall
            u(r_ax, z_ax)[0] - 0.0,  # symmetry: no flow through the axis
            u(r_b, z_b)[0] - 0.0,  # the developed profile, in and out
            u(r_b, z_b)[1] - G * (R**2 - r_b**2) / (4.0 * eta),
            u(r_t, z_t)[0] - 0.0,
            u(r_t, z_t)[1] - G * (R**2 - r_t**2) / (4.0 * eta),
            p.pin(),  # velocity is prescribed all round, so the pressure level needs fixing
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    off = [int(o) for o in fem.offsets]
    uu = sol[off[0] : off[1]].reshape(-1, 2)
    pts = np.asarray(fem.field_points[0])
    exact = G * (R**2 - pts[:, 0] ** 2) / (4.0 * eta)
    err = np.abs(uu[:, 1] - exact).max() / exact.max()
    assert err < 1e-8, f"u_z is off Poiseuille by {err:.2e} (P2 represents the parabola exactly)"
    assert np.abs(uu[:, 0]).max() < 1e-8 * exact.max(), "a fully developed pipe flow has no radial velocity"
