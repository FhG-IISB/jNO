"""3-D **Navier-Stokes** on Taylor-Hood tetrahedra -- the convective term in three dimensions.

`test_fem_stokes_3d.py` pins the 3-D mixed *Stokes* system. Convection was still untested in 3-D:
the term is dimension-agnostic in principle, but so was the mixed assembly, and that turned out to
carry a pressure gauge that did not converge. Measured rather than assumed.

Two exact solutions, both div-free by construction.

**Quadratic** -- lies in the P2/P1 space, so a correct assembler recovers it to machine precision
even with the nonlinearity present, Newton included. `(u.grad)u` is cubic there, so this exercises
the convective assembly on a form no linear test reaches:

    u = (y^2 + z^2, z^2 + x^2, x^2 + y^2)          div u == 0,  Delta u = (4, 4, 4)
    (u.grad)u = ( 2y(z^2+x^2) + 2z(x^2+y^2),
                  2x(y^2+z^2) + 2z(x^2+y^2),
                  2x(y^2+z^2) + 2y(z^2+x^2) )      [derived, not assumed]
    p = x + y + z - 3/2                            int p dx == 0

**Transcendental** -- nothing is captured exactly, so the discretisation error is genuinely nonzero
and the observed order is meaningful:

    u = (sin y, sin z, sin x)                      div u == 0,  Delta u = -u
    (u.grad)u = (sin z cos y, sin x cos z, sin y cos x)
    p = sin x sin y sin z - C,  C = (1 - cos 1)^3  int p dx == 0

The convective term is always written on the **unknown**, never on the exact field, so the Jacobian
comes from autodiff and the system is genuinely nonlinear.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
sin, cos = jno.np.sin, jno.np.cos
C_MEAN = float((1.0 - np.cos(1.0)) ** 3)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _build(size, nu, kind, quad_degree=6):
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, zi = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    gu, gv = grad(u, [xi, yi, zi]), grad(v, [xi, yi, zi])
    ub, vv = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    pp, qq = p.bind(x=xi, y=yi, z=zi), q.bind(x=xi, y=yi, z=zi)
    conv = inner(gu, ub, n_contract=1)  # (u.grad)u on the UNKNOWN -> autodiff Jacobian

    if kind == "quadratic":
        ue = (yi**2 + zi**2, zi**2 + xi**2, xi**2 + yi**2)
        ce = (
            2 * yi * (zi**2 + xi**2) + 2 * zi * (xi**2 + yi**2),
            2 * xi * (yi**2 + zi**2) + 2 * zi * (xi**2 + yi**2),
            2 * xi * (yi**2 + zi**2) + 2 * yi * (zi**2 + xi**2),
        )
        lap = (4.0, 4.0, 4.0)
        gp = (1.0, 1.0, 1.0)
        bcs = (yb**2 + zb**2, zb**2 + xb**2, xb**2 + yb**2)
    else:
        ue = (sin(yi), sin(zi), sin(xi))
        ce = (sin(zi) * cos(yi), sin(xi) * cos(zi), sin(yi) * cos(xi))
        lap = (-ue[0], -ue[1], -ue[2])
        gp = (
            cos(xi) * sin(yi) * sin(zi),
            sin(xi) * cos(yi) * sin(zi),
            sin(xi) * sin(yi) * cos(zi),
        )
        bcs = (sin(yb), sin(zb), sin(xb))

    f = sum((ce[k] - nu * lap[k] + gp[k]) * vv[k] for k in range(3))  # sum() over components
    fem = jno.fem(
        [
            inner(conv, vv, n_contract=1) + nu * inner(gu, gv, n_contract=2) - pp * trace(gv) - f,
            -qq * trace(gu),
            u(xb, yb, zb)[0] - bcs[0],
            u(xb, yb, zb)[1] - bcs[1],
            u(xb, yb, zb)[2] - bcs[2],
            p.pin(mean=True),
        ],
        quad_degree=quad_degree,
    )
    return d, fem, u, p


def _split(fem, sol):
    off = fem.offsets
    return np.asarray(sol)[off[0] : off[1]].reshape(-1, 3), np.asarray(sol)[off[1] :]


def test_the_convective_term_is_assembled_exactly():
    """A quadratic velocity is IN the P2 space, so the nonlinear system must reproduce it to machine
    precision -- Newton included. Any error here is the convective assembly or the nonlinear solve,
    never discretisation, which is what makes it a sharp test of the 3-D convective term.

    (Measured aside: the native assembler integrates this correctly at every `quad_degree` offered,
    including 1, so the recovery is not sensitive to that argument here.)"""
    _, fem, *_ = _build(0.35, nu=1.0, kind="quadratic")
    assert not fem.is_linear, "the convective term must make the system nonlinear"

    sol = fem.solve(linear=jno.solve.lu(backend="host"))
    vel, pre = _split(fem, sol)
    pv, pp_ = np.asarray(fem.field_points[0]), np.asarray(fem.field_points[1])
    x, y, z = pv[:, 0], pv[:, 1], pv[:, 2]
    ev = np.stack([y**2 + z**2, z**2 + x**2, x**2 + y**2], axis=1)
    ep = pp_[:, 0] + pp_[:, 1] + pp_[:, 2] - 1.5

    assert np.abs(vel - ev).max() < 1e-9, f"velocity off by {np.abs(vel - ev).max():.2e}"
    assert np.abs(pre - ep).max() < 1e-8, f"pressure off by {np.abs(pre - ep).max():.2e}"


@pytest.mark.parametrize("nu", [0.2, 0.05])
def test_a_transcendental_solution_is_recovered_at_discretisation_accuracy(nu):
    """Convection genuinely present (cell Peclet ~1.4 and ~5.6 at this mesh), no stabilisation.
    Thresholds are the measured errors with ~2x headroom, not invented tolerances."""
    d, fem, u, p = _build(0.28, nu=nu, kind="transcendental")
    sol = fem.solve(linear=jno.solve.lu(backend="host"))
    vel, pre = _split(fem, sol)
    pv, pp_ = np.asarray(fem.field_points[0]), np.asarray(fem.field_points[1])
    ev = np.stack([np.sin(pv[:, 1]), np.sin(pv[:, 2]), np.sin(pv[:, 0])], axis=1)
    ep = np.sin(pp_[:, 0]) * np.sin(pp_[:, 1]) * np.sin(pp_[:, 2]) - C_MEAN

    # Gates are the MEASURED max nodal errors with ~2x headroom, not tolerances picked to pass:
    #   nu=0.2   velocity 2.16e-3, pressure 3.29e-2      nu=0.05  velocity 7.91e-3, pressure 3.13e-2
    # Nodal maxima are much larger than the L2 errors (5.8e-4 / 4.1e-3) because P1 pressure error
    # peaks at corners -- gating on the wrong norm is how these first came out too tight.
    assert np.abs(vel - ev).max() < 1.5e-2
    assert np.abs(pre - ep).max() < 6e-2
    # the pressure is gauged by its integral, so it must carry no spurious constant (measured 1.2e-3)
    assert abs(float(np.mean(pre - ep))) < 5e-3
