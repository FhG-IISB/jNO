"""08 - Transient heat diffusion via ``jno.fem`` (time-dependent forward solve).

    u_t = nu Delta u      on the unit square,   u = 0 on the boundary,   u(t=0) = sin(pi x) sin(pi y)

Passing ``time=(t0, t1, n)`` to the domain turns the weak form into a semidiscrete system
``M u_dot + A u = 0``: ``jno.fem`` then exposes the mass matrix ``fem.M``, the operator
``fem.operator.A`` and the initial state ``fem.state0``. The initial condition is a single
Laplacian eigenmode, so the exact solution decays as ``exp(-2 nu pi^2 t)`` -- backward Euler
over (M, A) reproduces that decay. (For a *parametric* transient inverse, train through
``fem.solve()`` instead -- see ``transient_inverse_heat.py``.)
"""

import jax.numpy as jnp
import numpy as np
from shapely.geometry import box

import jno

nu = 1.0
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08, time=(0.0, 0.05, 26))
u, phi = d.fem_symbols()
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
xi0, yi0, _ = d.variable("initial", split=True)
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
ic = u(xi0, yi0) - jno.fn(lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y), [xi0, yi0])
fem = jno.fem([ui.t * vi + nu * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, ic])
assert fem.is_transient

M, A, dt = dense(fem.M), dense(fem.operator.A), float(fem.dt)
w = jnp.asarray(fem.state0)
for _ in range(round((fem.t1 - fem.t0) / dt)):  # backward Euler: (M + dt A) w_next = M w
    w = jnp.linalg.solve(M + dt * A, M @ w)

pts = np.asarray(fem.points)
analytic = np.exp(-2.0 * nu * np.pi**2 * fem.t1) * np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
rel_l2 = float(jnp.linalg.norm(analytic - w) / jnp.linalg.norm(analytic))
print(f"\nTransient heat (backward Euler over jno.fem M, A): dofs={fem.dofs}  rel_L2 at t={fem.t1}: {rel_l2:.3e}")
assert rel_l2 < 5e-2
