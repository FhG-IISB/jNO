"""Natural convection in a differentially heated square cavity (Boussinesq), Ra = 10³ and 10⁴.

    ∂u/∂t + (u·∇)u + ∇p − Pr Δu − Ra Pr T ŷ = 0,   ∇·u = 0,   ∂T/∂t + u·∇T − ΔT = 0
    no slip everywhere; T = 1 on the left, T = 0 on the right, ∂T/∂n = 0 on bottom and top; Pr = 0.71

Oracle: G. de Vahl Davis, *Int. J. Numer. Methods Fluids* 3 (1983) 249, Table I (benchmark solution):
    Ra = 10³: Nu_avg = 1.118, u_max(x=½) = 3.649, v_max(y=½) = 3.697
    Ra = 10⁴: Nu_avg = 2.243, u_max(x=½) = 16.178, v_max(y=½) = 19.617
Three coupled fields (vector velocity, pressure, temperature) with the buoyancy coupling, a wall pressure
from the momentum balance and adiabatic walls, all written as flux conditions.

The steady state is reached by MARCHING (backward Euler from rest, T linear in x). A steady Newton solve
from a zero initial guess did not converge (residual stalled at 5.5, direct or matrix-free): the Jacobian's
pressure mode is only weakly determined (smallest singular value 5.7e-4 on 9²). The march reports how much
each field still changes over its last half, and the pressure is the field that keeps drifting.

Run: python benchmarks/fdm/boussinesq.py
"""

import numpy as np
from _common import table

import jno
import jno.jnp_ops as jnn

Pr = 0.71
DAVIS = {1e3: (1.118, 3.649, 3.697), 1e4: (2.243, 16.178, 19.617)}


def march(Ra, n, T_end=1.0, steps=101):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured(), time=(0.0, T_end, steps))
    x, y, t = d.variable("interior", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    U, p, T = d.unknown(value_shape=(2,)), d.unknown(), d.unknown()
    u, pi, th = U.vector.bind(x=x, y=y, t=t), p.bind(x=x, y=y, t=t), T.bind(x=x, y=y, t=t)
    ux, uy = u[0], u[1]
    buoyancy = lambda s: jnn.stack([0.0 * s, Ra * Pr * s], axis=-1)  # noqa: E731
    terms = [
        u.t + ux * u.x + uy * u.y + jnn.stack([pi.x, pi.y], axis=-1) - Pr * (u.xx + u.yy) - buoyancy(th),
        ux.x + uy.y - 0.05 * d.cell_size**2 * (pi.xx + pi.yy),
        th.t + ux * th.x + uy * th.y - (th.xx + th.yy),
        U(x0, y0) - jnn.stack([0.0 * x0, 0.0 * x0], axis=-1),
        T(x0, y0) - (1.0 - x0),
    ]
    d.point_region("gauge", (0.5, 0.5))
    xg, yg, _ = d.variable("gauge", split=True)
    terms.append(p(xg, yg) - 0.0)
    for wall in ("left", "right", "bottom", "top"):
        xw, yw, _, nx, ny = d.variable(wall, normals=True, split=True)
        n_ = d.variable(wall, normals=True)
        uw, pw, tw = U.vector.bind(x=xw, y=yw), p.bind(x=xw, y=yw), T.bind(x=xw, y=yw)
        m = Pr * (uw.xx + uw.yy) - (uw[0] * uw.x + uw[1] * uw.y) + buoyancy(tw)  # the wall momentum balance
        terms += [U(xw, yw) - jnn.stack([0.0 * xw, 0.0 * xw], axis=-1), pw.d(n_) - (nx * m[0] + ny * m[1])]
        if wall in ("left", "right"):
            terms.append(T(xw, yw) - (1.0 if wall == "left" else 0.0))
        else:
            terms.append(tw.d(n_) - 0.0)  # adiabatic
    traj = np.asarray(jno.fdm(terms).solve(save_ts=[0.0, T_end / 2, T_end]))
    nxg, nyg = d.mesh_connectivity["grid"]["shape"]
    h = 1.0 / n
    Ug, Vg, Tg = (traj[-1, k].reshape(nxg, nyg) for k in (0, 1, 3))
    nu_avg = float(np.trapezoid((3 * Tg[0] - 4 * Tg[1] + Tg[2]) / (2 * h), dx=h))  # Nu = −∂T/∂x at the hot wall
    change = np.abs(traj[-1] - traj[-2]).max(axis=1)
    return {
        "Ra": f"{Ra:.0e}",
        "grid": f"{n + 1}²",
        "Nu_avg": nu_avg,
        "u_max(x=½)": float(Ug[nxg // 2].max()),
        "v_max(y=½)": float(Vg[:, nyg // 2].max()),
        "Δu last half": float(change[:2].max()),
        "Δp last half": float(change[2]),
    }


if __name__ == "__main__":
    import sys

    grids = [int(g) for g in sys.argv[1:]] or [32, 64]  # e.g. `python boussinesq.py 32` for the coarse grid only
    cols = ["Ra", "grid", "Nu_avg", "Nu ref", "u_max(x=½)", "u ref", "v_max(y=½)", "v ref", "Δu last half", "Δp last half"]
    for Ra in (1e3, 1e4):
        for n in grids:
            row = march(Ra, n)
            ref = DAVIS[Ra]
            row.update({"Nu ref": ref[0], "u ref": ref[1], "v ref": ref[2]})
            table(f"Boussinesq cavity (marched to steady state), Ra = {Ra:.0e}, {n + 1}²", [row], cols)
            sys.stdout.flush()
