"""06 - Linear-elastic cantilever beam under an end load (vector FEM) via ``jno.fem``.

Isotropic plane-stress elasticity on a slender beam (length L = 10, height H = 1) clamped at
the root and loaded by a downward shear traction on the tip. The unknown is the vector
displacement u = (u_x, u_y) -- ``fem_symbols(value_shape=(2,))`` -- with P2 elements
(``order=2``; constant-strain TRI3 is too stiff in bending). The weak form is the isotropic
elasticity bilinear form  lambda (div u)(div phi) + 2 mu eps(u):eps(phi).

The tip deflection is checked against Euler-Bernoulli beam theory  delta = P L^3 / (3 E I).
EB is a slender-beam approximation (it ignores shear deformation and the clamped-end boundary
layer), yet a P2 solve on a 10:1 beam matches it to ~1% (Timoshenko & Goodier, *Theory of
Elasticity*, 1970). Linearity in the load is inherent to the linear operator.
"""

import jax.numpy as jnp
import numpy as np

import jno

E, nu, L, H = 1000.0, 0.3, 10.0, 1.0
lam, mu = E * nu / (1.0 - nu**2), E / (2.0 * (1.0 + nu))  # plane-stress Lame parameters
Iz = H**3 / 12.0  # second moment of area (unit thickness)
inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

d = jno.Shape.rect(0.0, 0.0, L, H, size=0.5).domain()
u, phi = d.fem_symbols(value_shape=(2,), order=2)  # P2 vector displacement
xi, yi, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)  # clamped root
xr, yr, _ = d.variable("right", split=True)  # loaded tip
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)


q = 0.1  # downward shear traction (0, -q) on the tip; total load P = q * H
traction = -1.0 * inner(jnp.array([0.0, -q]), phi.bind(x=xr, y=yr), n_contract=1)
fem = jno.fem([weak, u(xl, yl) - (0.0, 0.0), traction])
# The slender-beam P2 stiffness matrix is too ill-conditioned for Jacobi-CG in float32 -> sparse-direct slot.
sol = jnp.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2)  # (n_nodes, 2)
tip = np.asarray(fem.points)[:, 0] > L - 1e-6
delta, eb = float(-jnp.mean(sol[tip, 1])), (q * H) * L**3 / (3.0 * E * Iz)
print(
    f"\nCantilever (P2 elasticity): dofs={fem.dofs}  FEM tip={delta:.4f}  Euler-Bernoulli={eb:.4f}  ratio={delta / eb:.3f}"
)
assert fem.is_linear and abs(delta - eb) / eb < 0.05  # matches beam theory to ~1%
