# --8<-- [start:code]
"""**Stabilised equal-order Navier-Stokes** -- P1 velocity and P1 pressure, which is *not* an
inf-sup-stable pair, made to work by residual-based stabilisation written as ordinary weak-form terms.

    (u.grad)u - nu lap u + grad p = 0,   div u = 0

The exact solution is Kovasznay (1948), so every number below is checked against a closed form.

Two things make this possible, and neither is a "stabilisation feature" -- they are the two pieces the
formula is built out of:

  * `jno.np.laplacian` on a **vector** field, so the momentum strong residual can carry `nu*lap(u)`;
  * `dom.cell_metric`, the element metric `G = J^-T J^-1`, which is what `tau` needs in order to be
    direction-aware. `dom.cell_size` is an isotropic scalar and cannot see a stretched cell.

Everything else is the FEM contract: the term list is the whole problem.

Why bother with equal order? Taylor-Hood needs P2 velocity. In 3-D that is roughly four times the
velocity DOFs of P1 for the same mesh. PSPG buys the cheaper pair.

References:
  tau, and the metric it is built on -- Tezduyar & Osawa, *CMAME* **190** (2000), Sec. 3
  PSPG                              -- Hughes, Franca & Balestra, *CMAME* **59** (1986)
  the exact solution                -- Kovasznay, *Proc. Camb. Phil. Soc.* **44** (1948) 58
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

inner, grad, trace, lap = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.laplacian

NU = 0.05
C_I = 36.0  # inverse-estimate constant for linear elements (Tezduyar & Osawa, Sec. 3)
RE = 1.0 / NU
LAM = RE / 2.0 - np.sqrt(RE**2 / 4.0 + 4.0 * np.pi**2)

# The exact Kovasznay fields, for the boundary data and the final check.
u_ex = lambda x, y: 1.0 - np.exp(LAM * x) * np.cos(2 * np.pi * y)  # noqa: E731
v_ex = lambda x, y: LAM / (2 * np.pi) * np.exp(LAM * x) * np.sin(2 * np.pi * y)  # noqa: E731
p_ex = lambda x, y: 0.5 * (1.0 - np.exp(2 * LAM * x))  # noqa: E731

d = jno.domain(box(-0.5, -0.5, 1.0, 1.5), mesh_size=0.075)
d.point_region("ppin", (-0.5, -0.5))  # one pressure gauge DOF

u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)  # P1 velocity  -- equal order,
p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure  -- NOT inf-sup stable
xi, yi = d.variable("interior", split=True)[:2]
xb, yb = d.variable("boundary", split=True)[:2]
xpn, ypn = d.variable("ppin", split=True)[:2]

ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
gp, gq = grad(p, [xi, yi]), grad(q, [xi, yi])
pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)

# Notation first, so the weak form below reads like the paper.
div = lambda gw: trace(gw)  # noqa: E731  -- div w
adv = lambda gw, w: inner(gw, w, n_contract=1)  # noqa: E731  -- (w.grad)w

# --- Galerkin ------------------------------------------------------------------------------------
momentum = inner(adv(gu, ub), vv, n_contract=1) + NU * inner(gu, gv, n_contract=2) - pp * div(gv)
continuity = -qq * div(gu)

# --- stabilisation -------------------------------------------------------------------------------
G = d.cell_metric  # the element metric, per quadrature point
gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731  -- a^T G a

# tau is LAGGED: it is a coefficient, not part of the equation, and differentiating through `u.G u`
# gives Newton a tangent it does not need to follow.
tau = jno.lag((gG(ub) + C_I * NU**2 * inner(G, G, n_contract=2)) ** -0.5)

r_m = adv(gu, ub) - NU * lap(u, [xi, yi]) + gp  # the momentum STRONG residual

# SUPG and PSPG are the same strong residual against two different test perturbations. They are two
# TERMS, not one: an additive term carries exactly one test field, because the test field is what
# names the equation block. And each carries the sign of the equation it joins -- momentum is written
# `+(u.grad u, v)` so SUPG is `+`; continuity is written `-(q, div u)` so PSPG is `-`.
momentum = momentum + tau * inner(adv(gv, ub), r_m, n_contract=1)  # SUPG  -> momentum block
continuity = continuity - tau * inner(gq, r_m, n_contract=1)  # PSPG  -> continuity block

bx = 1.0 - jno.np.exp(LAM * xb) * jno.np.cos(2 * np.pi * yb)
by = LAM / (2 * np.pi) * jno.np.exp(LAM * xb) * jno.np.sin(2 * np.pi * yb)

fem = jno.fem(
    [
        momentum,
        continuity,
        u(xb, yb)[0] - bx,
        u(xb, yb)[1] - by,
        p(xpn, ypn) - float(p_ex(-0.5, -0.5)),
    ]
)
assert not fem.is_linear, "the convective term keeps this nonlinear"
print(f"\nStabilised P1/P1 Kovasznay flow (Re={RE:.0f}): dofs={fem.dofs}")

# Assembled-tangent Newton: the matrix-free default goes NaN on this cold start from rest.
sol = np.asarray(fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-8, atol=1e-8)))

off = fem.offsets
uv = sol[off[0] : off[1]].reshape(-1, 2)
ph = sol[off[1] :].reshape(-1)
pts = np.asarray(d.mesh.points)[:, :2]
x, y = pts[:, 0], pts[:, 1]

e_u = np.sqrt(np.mean((uv[:, 0] - u_ex(x, y)) ** 2 + (uv[:, 1] - v_ex(x, y)) ** 2))
pe = p_ex(x, y)
e_p = np.sqrt(np.mean(((ph - ph.mean()) - (pe - pe.mean())) ** 2))  # pressure is gauge-free

print(f"  velocity RMS error vs Kovasznay: {e_u:.3e}")
print(f"  pressure RMS error vs Kovasznay: {e_p:.3e}")
assert e_u < 5e-2, f"velocity error {e_u:.3e} is too large -- the stabilised pair should resolve this"
assert e_p < 1.5e-1, f"pressure error {e_p:.3e} is too large -- PSPG should be controlling it"
print("\nEqual-order P1/P1 converged against a closed-form solution.")
# --8<-- [end:code]
