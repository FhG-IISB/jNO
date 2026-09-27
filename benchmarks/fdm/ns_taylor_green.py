"""Time-dependent incompressible Navier–Stokes: the Taylor–Green vortex on [0, π]², BDF2.

Oracle: the exact decaying vortex u = (−cos x sin y, sin x cos y)·e^{−2νt}, p = −¼(cos 2x + cos 2y)·e^{−4νt}
(Taylor & Green, *Proc. R. Soc. A* 158 (1937) 499), with its time-dependent values on the boundary.
Question: second order in u and a converging pressure with Δt ∝ h? (Crank–Nicolson's pressure does not
converge — see docs/fdm.md.)

Run: python benchmarks/fdm/ns_taylor_green.py
"""

import numpy as np
from _common import nodes, rates, rel, table

import jno
import jno.jnp_ops as jnn

nu, T = 0.1, 1.0


def solve(n):
    d = jno.domain(jno.shape.rect(0.0, 0.0, np.pi, np.pi, size=np.pi / n).structured(), time=(0.0, T, n + 1))
    x, y, t = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    U, p = d.unknown(value_shape=(2,)), d.unknown()
    u, pi = U.vector.bind(x=x, y=y, t=t), p.bind(x=x, y=y, t=t)
    ux, uy = u[0], u[1]
    E = lambda s, k=2: jnn.exp(-k * nu * s)  # noqa: E731
    Uex = lambda X, Y, s: jnn.stack([-jnn.cos(X) * jnn.sin(Y) * E(s), jnn.sin(X) * jnn.cos(Y) * E(s)], axis=-1)  # noqa: E731
    Pex = lambda X, Y, s: -0.25 * (jnn.cos(2 * X) + jnn.cos(2 * Y)) * E(s, 4)  # noqa: E731
    traj = np.asarray(
        jno.fdm(
            [
                u.t + ux * u.x + uy * u.y + jnn.stack([pi.x, pi.y], axis=-1) - nu * (u.xx + u.yy),
                ux.x + uy.y - 0.05 * d.cell_size**2 * (pi.xx + pi.yy),
                U(xb, yb) - Uex(xb, yb, tb),
                p(xb, yb) - Pex(xb, yb, tb),
                U(x0, y0) - Uex(x0, y0, 0.0),
            ]
        ).solve(time=jno.solve.bdf2())
    )
    X = nodes(d)
    ex_u = -np.cos(X[:, 0]) * np.sin(X[:, 1]) * np.exp(-2 * nu * T)
    ex_p = -0.25 * (np.cos(2 * X[:, 0]) + np.cos(2 * X[:, 1])) * np.exp(-4 * nu * T)
    return {"n": n, "u_x": rel(traj[-1, 0], ex_u), "p": rel(traj[-1, 2], ex_p)}


if __name__ == "__main__":
    rows = [solve(n) for n in (10, 20, 40)]
    table("Taylor–Green at T = 1, relative L2 errors (h = π/n, Δt = T/n)", rows, ["n", "u_x", "p"])
    for k in ("u_x", "p"):
        print(f"rate {k}: {', '.join(f'{r:.2f}' for r in rates([row[k] for row in rows]))}")
