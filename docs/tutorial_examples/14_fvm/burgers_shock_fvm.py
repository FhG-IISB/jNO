# --8<-- [start:code]
"""03 - An inviscid Burgers shock, marched explicitly with ``jno.solve.erk``.

    u_t + (u^2/2)_x = 0,  u = 1 for x < 0.2, 0 beyond: a shock moving at (1 + 0)/2.

Conservation form ``(ui*ui/2).d(x, ...)`` puts the shock where the Rankine-Hugoniot condition says;
SSPRK3 with a minmod limiter keeps it sharp and monotone, and ``cfl=`` chooses the substeps from the
spectral radius of the first-order operator at every step.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

T = 0.6
d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.005).domain(time=(0.0, T, 121))
x, t = d.variable("interior", split=True)[:2]
xl = d.variable("left", split=True)[0]
xi = d.variable("initial", split=True)[0]
u = d.unknown()
ui = u.bind(x=x, t=t)

prob = jno.fvm(
    [
        ui.t + (ui * ui / 2).d(x, scheme=jno.fv(limiter="minmod")),
        u(xl) - 1.0,
        u(xi) - (1.0 - jnn.tanh(1e4 * (xi - 0.2))) / 2,
    ]
)
traj = np.asarray(prob.solve(time=jno.solve.erk("ssprk3", cfl=0.5)))
c = prob.centroids[:, 0]
front = c[np.argmin(np.abs(traj[-1] - 0.5))]
print(f"shock at x = {front:.3f}  (Rankine-Hugoniot: {0.2 + 0.5 * T:.3f});  range [{traj.min():.3f}, {traj.max():.3f}]")
assert abs(front - (0.2 + 0.5 * T)) < 0.01
# --8<-- [end:code]
