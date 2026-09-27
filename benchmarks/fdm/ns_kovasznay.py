"""Steady incompressible Navier–Stokes: Kovasznay flow, Re = 40, velocity as one vector unknown.

Oracle: the exact Kovasznay solution (L. I. G. Kovasznay, *Proc. Camb. Phil. Soc.* 44 (1948) 58).
Question: does the pressure-stabilised collocated scheme converge in u and p, written as the math?

Run: python benchmarks/fdm/ns_kovasznay.py
"""

import numpy as np
from _common import nodes, rates, rel, table

import jno
import jno.jnp_ops as jnn

Re = 40.0
nu = 1.0 / Re
lam = Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)
Ux = lambda x, y, m=np: 1 - m.exp(lam * x) * m.cos(2 * np.pi * y)  # noqa: E731
Uy = lambda x, y, m=np: lam / (2 * np.pi) * m.exp(lam * x) * m.sin(2 * np.pi * y)  # noqa: E731
P = lambda x, y, m=np: 0.5 * (1 - m.exp(2 * lam * x))  # noqa: E731


def solve(h):
    d = jno.shape.rect(-0.5, -0.5, 1.0, 1.5, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    U, p = d.unknown(value_shape=(2,)), d.unknown()
    u, pi = U.vector.bind(x=x, y=y), p.bind(x=x, y=y)
    ux, uy = u[0], u[1]
    sol = np.asarray(
        jno.fdm(
            [
                ux * u.x + uy * u.y + jnn.stack([pi.x, pi.y], axis=-1) - nu * (u.xx + u.yy),  # (u·∇)u + ∇p − νΔu
                ux.x + uy.y - 0.05 * d.cell_size**2 * (pi.xx + pi.yy),  # ∇·u, pressure-stabilised
                U(xb, yb) - jnn.stack([Ux(xb, yb, jnn), Uy(xb, yb, jnn)], axis=-1),
                p(xb, yb) - P(xb, yb, jnn),
            ]
        ).solve()
    )
    X = nodes(d)
    return {
        "h": h,
        "u_x": rel(sol[0], Ux(X[:, 0], X[:, 1])),
        "u_y": rel(sol[1], Uy(X[:, 0], X[:, 1])),
        "p": rel(sol[2], P(X[:, 0], X[:, 1])),
    }


if __name__ == "__main__":
    rows = [solve(h) for h in (0.1, 0.05, 0.025)]
    table("Kovasznay, relative L2 errors", rows, ["h", "u_x", "u_y", "p"])
    for k in ("u_x", "u_y", "p"):
        print(f"rate {k}: {', '.join(f'{r:.2f}' for r in rates([row[k] for row in rows]))}")
