"""04 - Differentiable inverse through ``jno.fdm``: recover an unknown source amplitude.

The strong-form solve differentiates through ``custom_root``, so ``jax.grad`` flows to any parameter
that appears in the constraint list -- exactly like ``fem.solve()``. We run a twin experiment:
generate a synthetic observation from the model at the true amplitude ``s = 1``, then recover ``s``
from a wrong initial guess by plain gradient descent on the data-misfit loss. This is the mechanism
that lets ``jno.fdm`` compose into gradient-based inversion.

    -Delta u = s * f_base,  u = 0 on the boundary.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
x, y, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
f_base = 2 * np.pi**2 * jnn.sin(np.pi * x) * jnn.sin(np.pi * y)


def solve_scale(s):  # the one entry: a differentiable jno.fdm([...]).solve() parameterized by s
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    return jno.fdm([-ui.d2(x) - ui.d2(y) - s * f_base, u(xb, yb) - 0.0]).solve()


observed = jax.lax.stop_gradient(solve_scale(1.0))  # synthetic data from the model at the true s = 1
grad_loss = jax.grad(lambda s: jnp.mean((solve_scale(s) - observed) ** 2))  # grad through custom_root

s = 2.5  # deliberately wrong initial guess
for _ in range(15):
    s = s - 2.0 * float(grad_loss(s))  # gradient descent through the differentiable solve

print(f"\nInverse via jno.fdm: recovered source amplitude s={s:.4f}  (true 1.0)")
assert abs(s - 1.0) < 1e-2, f"did not recover the source amplitude: s={s:.4f}"
