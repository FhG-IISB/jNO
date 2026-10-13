"""Smoke test of the harness on a small jNO problem: a 2-D Poisson solve on the GPU through fem.solve
(assembly stage from jNO) and a dense gradient (a user scope)."""

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
jax.config.update("jax_enable_x64", True)
import jno  # noqa: E402
from dev import prof  # noqa: E402

h = 0.02
d = jno.shape.rect(0, 0, 1, 1).name("box").attach(k=1.0).sized(h).domain()
u, v = d.fem_symbols(names=("u", "v"))
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
fem = jno.fem([d.k * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])
op = fem.operator
A, b = op if not hasattr(op, "evaluate") else op.evaluate({})


def energy(scale):
    with jax.named_scope("jno/readout"):
        Ad = (A * scale).todense() if hasattr(A, "todense") else A * scale
        uu = jnp.linalg.solve(Ad + 1e-9 * jnp.eye(Ad.shape[0]), b)
        return 0.5 * uu @ b


r = prof.run(jax.jit(jax.value_and_grad(energy)), jnp.asarray(1.0), label="toy-poisson", reps=3)
print(r.table())
print("saved", r.save())
