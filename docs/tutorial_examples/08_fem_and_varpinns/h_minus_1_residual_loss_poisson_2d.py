# --8<-- [start:code]
"""The residual measured in H^-1: a better-conditioned PINN loss with one changed line.

The usual PINN loss is the mean square of the strong residual, ``r.mse``. For a second-order operator its
curvature grows like |k|^4 with the frequency k of the error, so plain gradient descent clears the high
frequencies at once and crawls along the smooth error. Measuring the same residual in the dual norm H^-1,
``r.norm("H-1")**2``, weights mode k by 1 / (1 + |k|^2) and the curvature grows like |k|^2 instead
(SV-PINNs, Marcondes 2026, arXiv:2605.03542; the Deep Fourier Residual method, Taylor et al. 2022).

Poisson  -Δu = f  on the unit square, u = 0 on the boundary, with the exact solution of the SV-PINN paper's
experiment 2, u = sin(πx) sin(πy) [sin(a(x + y)) + sin(2πx) + cos(3πy)], a = 10. The same network, optimiser
and step budget are trained twice: on ``r.mse`` and on ``r.norm("H-1")**2``. On this structured lattice the
norm takes its grid backend (sine transforms); on a triangle mesh the same line takes the P1 backend.
"""

import foundax
import jax
import numpy as np
import optax

import jno

jax.config.update("jax_enable_x64", True)
run_dir = jno.setup(__file__)
print(jno.info())

π, sin, cos = np.pi, jno.np.sin, jno.np.cos
A_FREQ, STEPS = 10.0, 2000

dom = jno.shape.rect(0, 0, 1, 1).structured(n=(32, 32)).domain()
print(jno.info(dom))
x, y, _ = dom.variable("interior", split=True)
u_exact = sin(π * x) * sin(π * y) * (sin(A_FREQ * (x + y)) + sin(2 * π * x) + cos(3 * π * y))
f = -u_exact.laplacian(x, y)


def train(loss):
    """Train a fresh network on ``loss(residual)``; return the relative L2 error against the exact solution."""
    net = jno.nn(foundax.mlp(2, 1, hidden_dims=128, num_layers=4, key=jax.random.PRNGKey(0)))
    u = net(x, y) * x * (1 - x) * y * (1 - y)  # hard-BC ansatz: vanishes on the boundary
    r = -u.laplacian(x, y) - f  # the strong residual
    net.optimizer(optax.adam(optax.exponential_decay(1e-3, 100, 0.9)))  # the SV-PINN paper's schedule
    crux = jno.core([loss(r)], domain=dom)
    crux.solve(STEPS)
    pred, ref = (np.asarray(v).reshape(-1) for v in crux.eval([u, u_exact], domain=dom))
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


err_l2 = train(lambda r: r.mse)  # the usual loss
err_h1 = train(lambda r: r.norm("H-1") ** 2)  # the same residual, measured in H^-1
print(f"\nPoisson a = {A_FREQ:g}, {STEPS} Adam steps, 33 x 33 lattice:")
print(f"  r.mse                 rel-L2 = {err_l2:.3e}")
print(f"  r.norm('H-1')**2      rel-L2 = {err_h1:.3e}")

assert err_h1 < 0.5 * err_l2, (err_l2, err_h1)
# --8<-- [end:code]
