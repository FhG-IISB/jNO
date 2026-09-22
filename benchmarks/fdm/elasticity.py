"""Linear elasticity (plane strain, Navier–Cauchy form) with a vector displacement, Dirichlet boundary.

    μ Δu + (λ + μ) ∇(∇·u) + f = 0,     u = 0 on ∂Ω,  Ω = (0, 1)²

Oracle: a manufactured solution u = (S, S), S = sin πx sin πy, with f worked out by hand:
Δu_i = −2π² S and ∂_x(∇·u) = π²(C − S), ∂_y(∇·u) = π²(C − S), C = cos πx cos πy.
Question: does a vector unknown carry a solid-mechanics operator with mixed derivatives, at second order?
(Traction — a derivative condition on the vector field — is refused today, so this is Dirichlet only.)

Run: python benchmarks/fdm/elasticity.py
"""

import numpy as np
from _common import nodes, rates, rel, table

import jno
import jno.jnp_ops as jnn

E_mod, nu_p = 1.0, 0.3
mu = E_mod / (2 * (1 + nu_p))
lmbda = E_mod * nu_p / ((1 + nu_p) * (1 - 2 * nu_p))
π = np.pi


def solve(h):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    U = d.unknown(value_shape=(2,))
    u = U.vector.bind(x=x, y=y)
    ux, uy = u[0], u[1]
    S, C = jnn.sin(π * x) * jnn.sin(π * y), jnn.cos(π * x) * jnn.cos(π * y)
    f = jnn.stack([2 * π**2 * mu * S - (lmbda + mu) * π**2 * (C - S)] * 2, axis=-1)  # body force, both components
    grad_div = jnn.stack([ux.xx + uy.xy, ux.xy + uy.yy], axis=-1)  # ∇(∇·u)
    sol = np.asarray(
        jno.fdm(
            [mu * (u.xx + u.yy) + (lmbda + mu) * grad_div + f, U(xb, yb) - jnn.stack([0.0 * xb, 0.0 * xb], axis=-1)]
        ).solve()
    )
    X = nodes(d)
    exact = np.sin(π * X[:, 0]) * np.sin(π * X[:, 1])
    return {"h": h, "u_x": rel(sol[0], exact), "u_y": rel(sol[1], exact)}


if __name__ == "__main__":
    rows = [solve(h) for h in (0.1, 0.05, 0.025)]
    table("Elasticity, relative L2 errors", rows, ["h", "u_x", "u_y"])
    print(f"rate u_x: {', '.join(f'{r:.2f}' for r in rates([r['u_x'] for r in rows]))}")
