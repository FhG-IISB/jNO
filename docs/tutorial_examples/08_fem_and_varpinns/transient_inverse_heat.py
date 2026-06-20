"""10 - Transient inverse: recover a diffusion rate from a time series via differentiable ``fem.solve()``.

    Forward:  u_t = alpha Delta u,   u = 0 on the boundary,   u(t=0) = sin(pi x) sin(pi y).

Given the observed trajectory ``u(t)``, recover the unknown scalar diffusion rate ``alpha`` by
differentiating the *time integration* itself: for a transient weak form ``fem.solve()`` returns
the trajectory ``u(save_ts)`` (default: backward Euler over the assembled ``dt``), and the
gradient flows through the integrator back to ``alpha``. ``crux`` then fits it to the data.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from shapely.geometry import box

import jno

alpha_true = 1.0
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12, time=(0.0, 0.08, 17))
u, phi = d.fem_symbols()
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ci = d.variable("initial", split=True)
ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
u0 = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
alpha = jno.np.parameter((1,), name="alpha")
fem = jno.fem([ui.t * vi + alpha * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - u0])

# Observed trajectory at the true rate (backward Euler over the assembled block) -- the "data".
blk, dt = fem.operator, float(fem.operator.dt)
M, ts = dense(blk.M), jnp.linspace(float(blk.t0), float(blk.t1), round((blk.t1 - blk.t0) / dt) + 1)
w, traj = jnp.asarray(blk.state0).reshape(-1), [jnp.asarray(blk.state0).reshape(-1)]
for tn in ts[1:]:
    w = jnp.linalg.solve(M + dt * jnp.asarray(blk.operator_fn(tn, {"alpha": alpha_true})), M @ w)
    traj.append(w)
u_obs = jnp.stack(traj)

# Recover alpha from the trajectory through the differentiable transient solve.
alpha.dtype(jnp.float64)
alpha.initialize(jax.nn.initializers.constant(2.0))  # start far from the truth
alpha.optimizer(optax.adam(5e-2))
crux = jno.core([(fem.solve() - u_obs).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
crux.solve(200)

rec = float(np.asarray(crux.eval([alpha])).reshape(-1)[0])
print(f"\nTransient inverse: recovered alpha={rec:.4f} (truth {alpha_true})  rel-err={abs(rec - alpha_true):.2%}")
assert abs(rec - alpha_true) < 0.05
