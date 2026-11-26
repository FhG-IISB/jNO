# --8<-- [start:code]
"""05 - The lid-driven cavity at Re = 100: incompressible Navier-Stokes as a saddle pair.

Momentum and continuity are written as for ``jno.fdm``. ``jno.fvm`` finds the constraint by structure (the
continuity equation has no term in its own unknown p) and gives it the Rhie-Chow mass flux; the classical
constant lid needs no regularisation, since a cell-centred scheme has no corner node. Compared with Ghia,
Ghia & Shin, J. Comput. Phys. 48 (1982) 387, Table I.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

N, NU = 32, 0.01
d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=N).quad().domain()
d.tag("lid", lambda x, y: y > 1 - 1e-9)
d.tag("wall", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9))
d.point_region("ppin", (0.0, 0.0))  # the pressure gauge
x, y, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("lid", split=True)
xw, yw, _ = d.variable("wall", split=True)
xp, yp, _ = d.variable("ppin", split=True)

U, P = d.unknown.vector(2), d.unknown()
Ui, pi = U.bind(x=x, y=y), P.bind(x=x, y=y)
u, v = Ui[0], Ui[1]
s = jno.fv()
momentum = (u * Ui).d(x, scheme=s) + (v * Ui).d(y, scheme=s) - NU * (Ui.xx + Ui.yy) + jnn.stack([pi.x, pi.y], axis=-1)
prob = jno.fvm(
    [
        momentum,
        Ui[0].x + Ui[1].y,
        U(xl, yl) - jnn.stack([1.0 + 0.0 * xl, 0.0 * xl], axis=-1),
        U(xw, yw) - 0.0,
        P(xp, yp) - 0.0,
    ]
)
sol = np.asarray(prob.solve())

ghia_y = np.array(
    [0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547]
)
ghia_u = np.array(
    [
        0.84123,
        0.78871,
        0.73722,
        0.68717,
        0.23151,
        0.00332,
        -0.13641,
        -0.20581,
        -0.21090,
        -0.15662,
        -0.10150,
        -0.06434,
        -0.04775,
        -0.04192,
        -0.03717,
    ]
)
c = prob.centroids
col = np.abs(c[:, 0] - 0.5) < 1.0 / N
ys = np.unique(np.round(c[col, 1], 12))
us = np.array([sol[0][col][np.isclose(c[col, 1], t)].mean() for t in ys])
err = np.abs(np.interp(ghia_y, ys, us) - ghia_u).max()
print(f"centre-line u: max |u - Ghia| = {err:.2e};  min u = {us.min():.4f} (Ghia -0.2109)")
assert err < 6e-3
# --8<-- [end:code]
