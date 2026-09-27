"""Electrostatics with a piecewise permittivity: ∇·(ε ∇φ) = 0, two dielectric layers.

    ε = ε₁ for x < ½, ε₂ for x > ½;   φ = 0 at x = 0, φ = 1 at x = 1;  ∂φ/∂n = 0 at y = 0, 1

Oracle: φ is piecewise linear with a continuous flux D = −ε ∂φ/∂x = −q, q = 1 / (½/ε₁ + ½/ε₂).
Question: how does the collocated divergence form treat a jump in the coefficient (the interface sits on a
grid line)? A discontinuous ε is where naive FD loses order. Chaining two central differences was first
order here (0.14 at h = 0.1, only the even nodes wrong); `(ε·φ.x).x` is now the conservative compact flux
difference with ε evaluated at the half-points, which is exact for layers (`jno.fd(average=...)`).

Run: python benchmarks/fdm/electrostatics.py
"""

import numpy as np
from _common import nodes, rates, table

import jno
import jno.jnp_ops as jnn

eps1, eps2 = 1.0, 10.0
q = 1.0 / (0.5 / eps1 + 0.5 / eps2)


def exact(X):
    return np.where(X < 0.5, q * X / eps1, q * 0.5 / eps1 + q * (X - 0.5) / eps2)


def solve(h):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=h).structured().domain()
    x, y, _ = d.variable("interior", split=True)
    phi = d.unknown()
    ph = phi.bind(x=x, y=y)
    eps = jnn.where(x < 0.5, eps1, eps2)
    terms = [(eps * ph.x).x + (eps * ph.y).y]  # ∇·(ε∇φ)
    (xl, yl, _), (xr, yr, _) = d.variable("left", split=True), d.variable("right", split=True)
    terms += [phi(xl, yl) - 0.0, phi(xr, yr) - 1.0]
    for wall in ("bottom", "top"):
        xw, yw, _ = d.variable(wall, split=True)
        terms.append(phi.bind(x=xw, y=yw).d(d.variable(wall, normals=True)) - 0.0)
    sol = np.asarray(jno.fdm(terms).solve()).reshape(-1)
    X = nodes(d)
    err = np.abs(sol - exact(X[:, 0])).max()
    D = -q * np.ones_like(X[:, 0])
    return {
        "h": h,
        "max |φ − φ_exact|": float(err),
        "φ(½)": float(sol[np.argmin(np.abs(X[:, 0] - 0.5) + np.abs(X[:, 1] - 0.5))]),
        "exact φ(½)": q * 0.5 / eps1,
        "_D": D,
    }


if __name__ == "__main__":
    rows = [solve(h) for h in (0.1, 0.05, 0.025)]
    table(f"Electrostatics, ε₁ = {eps1}, ε₂ = {eps2}", rows, ["h", "max |φ − φ_exact|", "φ(½)", "exact φ(½)"])
    print(f"rate: {', '.join(f'{r:.2f}' for r in rates([r['max |φ − φ_exact|'] for r in rows]))}")
