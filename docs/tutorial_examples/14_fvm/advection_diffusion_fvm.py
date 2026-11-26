# --8<-- [start:code]
"""02 - Convection-dominated transport: numerical fluxes and limiters with ``jno.fv``.

    u' - eps u'' = 0 on (0, 1), u(0) = 0, u(1) = 1,  eps = 0.01: a boundary layer at x = 1.

Write the convection in divergence form, ``ui.d(x, scheme=jno.fv(...))``: the scheme rides on the outer
derivative, as a ``jno.fd(...)`` stencil does. At a cell Peclet number of 2.5 the unlimited second-order
reconstruction oscillates; every limiter keeps the solution in [0, 1].
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

EPS = 0.01
d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.05).domain()
x = d.variable("interior", split=True)[0]
xl, xr = d.variable("left", split=True)[0], d.variable("right", split=True)[0]
u = d.unknown()
ui = u.bind(x=x)

for name, scheme in [
    ("first order", jno.fv(recon="none")),
    ("unlimited", jno.fv(limiter=None)),
    ("van Albada", jno.fv()),
    ("minmod", jno.fv(limiter="minmod")),
    ("HLL + van Leer", jno.fv(flux="hll", limiter="vanleer")),
]:
    sol = np.asarray(jno.fvm([ui.d(x, scheme=scheme) - EPS * ui.xx, u(xl) - 0.0, u(xr) - 1.0]).solve())
    print(f"{name:15s} range [{sol.min():+.3f}, {sol.max():.3f}]")
    if name != "unlimited":
        assert sol.min() > -1e-3 and sol.max() < 1 + 1e-3
# --8<-- [end:code]
