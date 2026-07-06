"""04 - Differentiable inverse through ``jno.fdm`` + ``jno.core``: recover an unknown source amplitude.

When the constraint list carries a trainable ``jno.np.parameter``, ``jno.fdm([...]).solve()`` returns
a differentiable **trace node** (not an array) -- exactly as ``fem.solve()`` does -- so it composes
straight into ``jno.core``. We run a twin experiment: generate a synthetic observation from the
forward solve at the true amplitude ``s = 1``, then recover ``s`` from a deliberately wrong start by
minimising the data misfit, with the parameter's own attached optimizer driving the fit.

    -Delta u = s * f_base,  u = 0 on the boundary.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
x, y, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
f_base = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)
u = d.unknown()
ui = u.bind(x=x, y=y)

# Synthetic observation: the forward solve at the true amplitude s = 1 (a plain float -> eager array).
observed = jnp.asarray(jno.fdm([-ui.d2(x) - ui.d2(y) - 1.0 * f_base, u(xb, yb) - 0.0]).solve()).reshape(-1)

# Recover s: a trainable parameter with an attached optimizer, driven by crux through the data misfit.
s = jno.np.parameter((1,), name="s")
s.dtype(jnp.float64)
s.initialize(jax.nn.initializers.constant(2.5))  # deliberately wrong start
s.optimizer(optax.adam(1e-1))
solve = jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()  # a differentiable trace node
crux = jno.core([(solve - observed).mse])  # domain inferred from the graph — no explicit domain= needed
crux.solve(150)

rec = float(np.asarray(crux.eval([s])).reshape(-1)[0])  # the recovered amplitude (do NOT index [0] on a field)
print(f"\nInverse via jno.fdm + jno.core: recovered source amplitude s={rec:.4f}  (true 1.0)")
assert abs(rec - 1.0) < 1e-2, f"did not recover the source amplitude: s={rec:.4f}"
