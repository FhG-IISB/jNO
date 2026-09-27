"""Lid-driven cavity, Re = 100, against Ghia, Ghia & Shin (1982).

Oracle: u(0.5, y) from Ghia, Ghia & Shin, *J. Comput. Phys.* 48 (1982) 387, Table I.
The wall pressure comes from the momentum balance, ∂p/∂n = n·(νΔu − (u·∇)u), and the pressure constant is
fixed at one interior node (`domain.point_region`).
Question: how close is the collocated scheme to the benchmark, and where does it stop converging in Re?

Run: python benchmarks/fdm/ns_cavity.py
"""

import numpy as np
from _common import table

import jno
import jno.jnp_ops as jnn

GHIA_Y = np.array([0.0547, 0.1719, 0.2813, 0.4531, 0.5, 0.6172, 0.7344, 0.8516, 0.9531, 0.9766])
GHIA_U100 = np.array([-0.03717, -0.10150, -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.84123])


def cavity(n, Re=100.0):
    nu = 1.0 / Re
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=1.0 / n).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    U, p = d.unknown(value_shape=(2,)), d.unknown()
    u, pi = U.vector.bind(x=x, y=y), p.bind(x=x, y=y)
    ux, uy = u[0], u[1]
    terms = [
        ux * u.x + uy * u.y + jnn.stack([pi.x, pi.y], axis=-1) - nu * (u.xx + u.yy),
        ux.x + uy.y - 0.05 * d.cell_size**2 * (pi.xx + pi.yy),
    ]
    d.point_region("gauge", (0.5, 0.5))
    xg, yg, _ = d.variable("gauge", split=True)
    terms.append(p(xg, yg) - 0.0)
    for wall in ("left", "right", "bottom", "top"):
        xw, yw, _, nx, ny = d.variable(wall, normals=True, split=True)
        uw, pw = U.vector.bind(x=xw, y=yw), p.bind(x=xw, y=yw)
        m = nu * (uw.xx + uw.yy) - (uw[0] * uw.x + uw[1] * uw.y)  # νΔu − (u·∇)u at the wall
        lid = 1.0 if wall == "top" else 0.0
        terms += [
            U(xw, yw) - jnn.stack([lid + 0.0 * xw, 0.0 * xw], axis=-1),
            pw.d(d.variable(wall, normals=True)) - (nx * m[0] + ny * m[1]),
        ]
    sol = np.asarray(jno.fdm(terms).solve())
    nxg, nyg = d.mesh_connectivity["grid"]["shape"]
    centre = sol[0].reshape(nxg, nyg)[nxg // 2]
    return float(np.abs(np.interp(GHIA_Y, np.linspace(0, 1, nyg), centre) - GHIA_U100).max())


if __name__ == "__main__":
    rows = [{"grid": f"{n + 1}²", "max |u - Ghia|": cavity(n)} for n in (32, 64)]
    table("Cavity, Re = 100: u(0.5, y) against Ghia et al. (1982)", rows, ["grid", "max |u - Ghia|"])
