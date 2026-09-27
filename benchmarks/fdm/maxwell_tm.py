"""Time-domain Maxwell, 2-D TM mode in a PEC square cavity, as a first-order system on collocated nodes.

    ∂H_x/∂t = −∂E_z/∂y,   ∂H_y/∂t = ∂E_z/∂x,   ∂E_z/∂t = ∂H_y/∂x − ∂H_x/∂y     (c = 1)
    E_z = 0 on the walls (perfect conductor)

Oracle: the cavity mode (m, n) = (1, 1): E_z = sin πx sin πy cos ωt, H_x = −(π/ω) sin πx cos πy sin ωt,
H_y = (π/ω) cos πx sin πy sin ωt, ω = π√2.
Question: the standard FD method for Maxwell is Yee's staggered grid (K. S. Yee, *IEEE Trans. Antennas
Propag.* 14 (1966) 302), because central differences on ONE grid decouple odd and even nodes — spurious
modes. Does the collocated scheme converge on a smooth mode anyway, and how does it compare with the
second-order-in-time wave equation E_z.tt = ΔE_z on the same grid?

Run: python benchmarks/fdm/maxwell_tm.py
"""

import numpy as np
from _common import nodes, rates, rel, table

import jno
import jno.jnp_ops as jnn

π = np.pi
ω = π * np.sqrt(2.0)
T = 1.0


def first_order(n):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured(), time=(0.0, T, 2 * n + 1))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    Hx, Hy, Ez = d.unknown(), d.unknown(), d.unknown()
    hx, hy, ez = Hx.bind(x=x, y=y, t=t), Hy.bind(x=x, y=y, t=t), Ez.bind(x=x, y=y, t=t)
    traj = np.asarray(
        jno.fdm(
            [
                hx.t + ez.y,
                hy.t - ez.x,
                ez.t - (hy.x - hx.y),
                Ez(xb, yb) - 0.0,
                Hx(x0, y0) - 0.0,
                Hy(x0, y0) - 0.0,
                Ez(x0, y0) - jnn.sin(π * x0) * jnn.sin(π * y0),
            ]
        ).solve(time=jno.solve.theta(0.5))  # Crank–Nicolson: energy-conserving for this skew system
    )
    X = nodes(d)
    exact = np.sin(π * X[:, 0]) * np.sin(π * X[:, 1]) * np.cos(ω * T)
    return rel(traj[-1, 2], exact)


def wave(n):
    d = jno.domain(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured(), time=(0.0, T, 2 * n + 1))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    Ez = d.unknown()
    ez = Ez.bind(x=x, y=y, t=t)
    traj = np.asarray(
        jno.fdm([ez.tt - (ez.xx + ez.yy), Ez(xb, yb) - 0.0, Ez(x0, y0) - jnn.sin(π * x0) * jnn.sin(π * y0)]).solve()
    )
    X = nodes(d)
    return rel(traj[-1], np.sin(π * X[:, 0]) * np.sin(π * X[:, 1]) * np.cos(ω * T))


if __name__ == "__main__":
    rows = [{"n": n, "first-order E,H": first_order(n), "wave E_z.tt": wave(n)} for n in (10, 20, 40)]
    table(
        "TM cavity mode (1, 1) at T = 1, relative L2 error of E_z (Δt = h/2)", rows, ["n", "first-order E,H", "wave E_z.tt"]
    )
    for k in ("first-order E,H", "wave E_z.tt"):
        print(f"rate {k}: {', '.join(f'{r:.2f}' for r in rates([row[k] for row in rows]))}")
