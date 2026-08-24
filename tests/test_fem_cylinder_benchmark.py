"""DFG benchmark 2D-1: steady Navier-Stokes past a cylinder at Re = 20.

The first case here checked against an EXTERNAL reference rather than a manufactured solution or
against jNO itself, and the first fluid case with a natural (do-nothing) outflow.

Configuration and reference values, Schaefer & Turek 1996 (the DFG/Featflow benchmark):

    domain     [0, 2.2] x [0, 0.41] \\ B_0.05(0.2, 0.2)
    inflow     u = (4 U y (0.41 - y) / 0.41^2, 0),  U = 0.3   -> U_mean = 2/3 U = 0.2
    viscosity  nu = 1e-3,  D = 0.1   ->   Re = U_mean D / nu = 20
    cD, cL     2 F / (U_mean^2 D)
    dP         p(0.15, 0.2) - p(0.25, 0.2)

    cD = 5.57953523384    cL = 0.010618948146    dP = 0.11752016697

Two things this exercises that nothing else in the suite does:

* the **outflow is natural** -- nothing is written for the downstream face, and that is also what
  fixes the pressure level, so the problem carries no `p.pin()`;
* the forces come from `fem.eval` + `region_dofs` -- the **reaction** conjugate to the cylinder's
  no-slip constraint, which is the accurate way to get a force out of a finite element solution
  (John, *Int. J. Numer. Meth. Fluids* 44, 2004), not an integral of stress over the surface.

**On lift.** ``cL`` is deliberately NOT held to a tight tolerance. The cylinder centre sits at
``y = 0.2`` in a channel of height ``0.41``, i.e. 0.005 off the axis, so the lift is a small residue
of two nearly-cancelling forces and is acutely sensitive to the random asymmetry of an unstructured
mesh. Measured across four refinements it does not settle: 6.87%, 1.93%, 3.77%, 6.66% error, while
drag over the same meshes goes 0.39%, 0.10%, 0.04%, 0.02%. A tight lift gate would be a flaky test
wearing a benchmark's clothes; the sign and order of magnitude are what is actually reproducible
here.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace

L, H = 2.2, 0.41
CX, CY, RR = 0.2, 0.2, 0.05
UMAX, NU = 0.3, 1e-3
UMEAN, DIA = 2.0 / 3.0 * UMAX, 2.0 * RR
EPS = 1e-9

CD_REF = 5.57953523384
CL_REF = 0.010618948146
DP_REF = 0.11752016697


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _solve(h_far, h_cyl):
    """Steady NS past the cylinder; returns (cD, cL, dP, probe_offset, fem, classification)."""
    shape = jno.Shape.rect(0, 0, L, H, size=h_far) - jno.Shape.disk(CX, CY, RR, size=h_cyl)
    d = shape.domain()
    d.tag("inlet", lambda x, y: x < EPS)
    d.tag("walls", lambda x, y: (y < EPS) | (y > H - EPS))
    d.tag("cyl", lambda x, y: (x - CX) ** 2 + (y - CY) ** 2 < (RR + 1e-4) ** 2)

    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
    p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure
    xi, yi = d.variable("interior", split=True)[:2]
    xin, yin = d.variable("inlet", split=True)[:2]
    xw, yw = d.variable("walls", split=True)[:2]
    xc, yc = d.variable("cyl", split=True)[:2]

    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    conv = inner(gu, ub, n_contract=1)  # (u.grad)u -- the convective nonlinearity
    momentum = inner(conv, vv, n_contract=1) + NU * inner(gu, gv, n_contract=2) - pp * trace(gv)
    profile = 4.0 * UMAX * yin * (H - yin) / H**2

    fem = jno.fem(
        [
            momentum,
            -qq * trace(gu),
            u(xin, yin)[0] - profile,
            u(xin, yin)[1] - 0.0,
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
            u(xc, yc)[0] - 0.0,
            u(xc, yc)[1] - 0.0,
            # the downstream face carries NO term: untagged boundary is natural (do-nothing), and
            # that is also what determines the pressure level -- hence no p.pin() anywhere here
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))

    free = np.asarray(fem.eval(momentum, sol))  # residual with no essential elimination
    scale = 2.0 / (UMEAN**2 * DIA)
    cD = -scale * float(free[fem.region_dofs("cyl", field=u, component=0)].sum())
    cL = -scale * float(free[fem.region_dofs("cyl", field=u, component=1)].sum())

    pre = sol[fem.offsets[1] :]
    ppts = np.asarray(fem.field_points[1])

    def probe(pt):
        i = int(np.argmin(np.sum((ppts - np.asarray(pt)) ** 2, axis=1)))
        return float(pre[i]), float(np.linalg.norm(ppts[i] - np.asarray(pt)))

    pf, off_f = probe((CX - RR, CY))
    pb, off_b = probe((CX + RR, CY))
    return cD, cL, pf - pb, max(off_f, off_b), fem


def test_drag_and_pressure_drop_match_the_published_reference():
    """~10k dofs, about 5 s. Drag is the quantity this benchmark actually pins down."""
    cD, cL, dP, probe_off, fem = _solve(0.035, 0.006)

    assert not fem.is_linear, "Re=20 past a cylinder must carry the convective nonlinearity"
    # no gauge: the natural outflow fixes the pressure level, so nothing pins a pressure node
    assert not any("_gauge_pin" in c for c in fem.classification)

    assert abs(cD - CD_REF) / CD_REF < 5e-3, f"cD {cD:.5f} vs reference {CD_REF:.5f}"
    # dP is read at the nearest PRESSURE NODE to each probe point, so the gate has to cover the
    # node offset as well as the solution -- measured 0.73% at this resolution with the nearest
    # node 0.0031 away, and 0.00% on meshes that happen to put a node exactly on the point.
    assert probe_off < 0.1 * RR, f"pressure probe landed {probe_off:.4f} from the benchmark point"
    assert abs(dP - DP_REF) / DP_REF < 1.5e-2, f"dP {dP:.5f} vs reference {DP_REF:.5f}"

    # Lift: sign and order of magnitude only -- see the module docstring for why.
    assert 0.0 < cL < 5.0 * CL_REF, f"cL {cL:.6f} is not even the right size (reference {CL_REF:.6f})"


@pytest.mark.slow
def test_drag_converges_towards_the_reference_under_refinement():
    """The claim that this is a real benchmark match and not a lucky mesh."""
    errs = []
    for h_far, h_cyl in [(0.05, 0.010), (0.035, 0.006), (0.025, 0.004)]:
        cD, *_ = _solve(h_far, h_cyl)
        errs.append(abs(cD - CD_REF) / CD_REF)
    assert np.all(np.diff(errs) < 0), f"drag error not decreasing under refinement: {errs}"
    assert errs[-1] < 1e-3, f"finest drag error {errs[-1]:.2e} should be under 0.1%"
