"""Smoke test for the release image: run inside the built container before it is pushed.

The full suite already runs in ci.yml and nightly.yml; this checks what the image itself can break --
the installed version, the system libraries gmsh needs to mesh, and an end-to-end FEM solve.
Usage: smoke_test.py [expected_version]
"""

import sys

import jax
import jax.numpy as jnp

import jno

if len(sys.argv) > 1 and jno.__version__ != sys.argv[1]:
    sys.exit(f"image has jno {jno.__version__}, expected {sys.argv[1]}")

# -Δu = 1 on the unit square, u = 0 on the boundary (README example, coarse mesh).
d = jno.shape.rect(0, 0, 1, 1, size=0.2).domain()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
u, v = d.fem_symbols()
ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
sol = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0]).solve()

values = jnp.asarray(jax.tree_util.tree_leaves(sol)[0])
if not bool(jnp.all(jnp.isfinite(values))) or float(jnp.max(jnp.abs(values))) == 0.0:
    sys.exit("FEM solve returned non-finite or all-zero values")

print(f"jno {jno.__version__} smoke test passed on {jax.devices()}; max |u| = {float(jnp.max(jnp.abs(values))):.4f}")
