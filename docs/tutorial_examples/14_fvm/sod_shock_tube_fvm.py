# --8<-- [start:code]
"""04 - The Sod shock tube: the Euler equations as the flux of a vector unknown.

    q = (rho, rho u, E),  q_t + F(q)_x = 0,  F = (rho u, rho u^2 + p, u (E + p)),  p = (gamma - 1)(E - rho u^2/2).

No Euler-specific code: the numerical fluxes read the Jacobian of the flux you write. The generic Roe flux
(|A| = sqrt(A^2 + delta^2 I)) resolves the contact sharper than Rusanov.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

GAMMA, T = 1.4, 0.2
d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.005).domain(time=(0.0, T, 21))
x, t = d.variable("interior", split=True)[:2]
xi = d.variable("initial", split=True)[0]
q = d.unknown.vector(3)
qi = q.bind(x=x, t=t)
rho, mom, E = qi[0], qi[1], qi[2]
vel = mom / rho
p = (GAMMA - 1) * (E - 0.5 * mom * vel)
F = jnn.stack([mom, mom * vel + p, vel * (E + p)], axis=-1)  # axis=-1: the components of every point
left = (1.0 - jnn.tanh(1e4 * (xi - 0.5))) / 2
q0 = jnn.stack([0.125 + 0.875 * left, 0.0 * xi, (0.1 + 0.9 * left) / (GAMMA - 1)], axis=-1)

for flux in ("rusanov", "roe"):
    prob = jno.fvm([qi.t + F.d(x, scheme=jno.fv(flux=flux, limiter="minmod")), q(xi) - q0])
    traj = np.asarray(prob.solve(time=jno.solve.erk("ssprk3", cfl=0.5)))
    mass = traj[:, 0] @ prob.volumes
    print(
        f"{flux:8s} density in [{traj[-1, 0].min():.3f}, {traj[-1, 0].max():.3f}], mass drift {abs(mass[-1] - mass[0]):.1e}"
    )
    assert abs(mass[-1] - mass[0]) < 1e-12
# --8<-- [end:code]
