"""Inverse problem: recover the viscosity of a Navier–Stokes flow from its velocity field.

A trainable ν inside the Kovasznay momentum equation (vector velocity, pressure), fitted through `jno.core`
to the velocity the forward solve produces at the true ν = 1/40. The gradient reaches ν through the
implicit-function theorem on the converged Newton solve.
Oracle: the true value, ν = 0.025. Question: does a trainable parameter work in a coupled, nonlinear,
vector system?

The inner solve is `linear=jno.solve.lu()`. With the default (matrix-free, unpreconditioned BiCGStab)
the forward solve converges, but the ADJOINT solve, which gives the gradient, does not: it stalls at
relative residual 0.87 and raises. A Navier–Stokes tangent is a saddle-point system, where an
unpreconditioned Krylov method is known to struggle.

Run: python benchmarks/fdm/inverse_viscosity.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from _common import table

import jno
import jno.jnp_ops as jnn

Re = 40.0
lam = Re / 2 - np.sqrt(Re**2 / 4 + 4 * np.pi**2)
Ux = lambda x, y, m=np: 1 - m.exp(lam * x) * m.cos(2 * np.pi * y)  # noqa: E731
Uy = lambda x, y, m=np: lam / (2 * np.pi) * m.exp(lam * x) * m.sin(2 * np.pi * y)  # noqa: E731
P = lambda x, y, m=np: 0.5 * (1 - m.exp(2 * lam * x))  # noqa: E731

d = jno.shape.rect(-0.5, -0.5, 1.0, 1.5, size=0.1).structured().domain()
x, y, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
U, p = d.unknown(value_shape=(2,)), d.unknown()
u, pi = U.vector.bind(x=x, y=y), p.bind(x=x, y=y)
ux, uy = u[0], u[1]


def problem(nu):
    return jno.fdm(
        [
            ux * u.x + uy * u.y + jnn.stack([pi.x, pi.y], axis=-1) - nu * (u.xx + u.yy),
            ux.x + uy.y - 0.05 * d.cell_size**2 * (pi.xx + pi.yy),
            U(xb, yb) - jnn.stack([Ux(xb, yb, jnn), Uy(xb, yb, jnn)], axis=-1),
            p(xb, yb) - P(xb, yb, jnn),
        ]
    )


if __name__ == "__main__":
    observed = jnp.asarray(problem(1.0 / Re).solve())
    nu = jno.np.parameter((1,), name="nu")
    nu.dtype(jnp.float64)
    nu.initialize(jax.nn.initializers.constant(0.05))
    nu.optimizer(optax.adam(2e-3))
    crux = jno.core([(problem(nu).solve(linear=jno.solve.lu()) - observed).mse])
    t0 = time.perf_counter()
    crux.solve(300)
    got = float(np.asarray(crux.eval([nu])).reshape(-1)[0])
    table(
        "Inverse: ν from the velocity field",
        [
            {
                "true ν": 1 / Re,
                "start": 0.05,
                "recovered": got,
                "rel. error": abs(got - 1 / Re) * Re,
                "time (s)": time.perf_counter() - t0,
            }
        ],
        ["true ν", "start", "recovered", "rel. error", "time (s)"],
    )
