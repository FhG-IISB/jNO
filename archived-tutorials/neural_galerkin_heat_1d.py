"""Neural Galerkin: evolve a network's *weights* in time by projecting the PDE onto ∂u/∂θ.

Where a Deep-Ritz / VPINN network is trained once for a *steady* solution, Neural Galerkin makes the
network **time-dependent through its weights** θ(t): the PDE ``u_t = 𝓝(u)`` is projected onto the
tangent space of the parametrisation, giving an ODE for the weights

    M(θ) θ̇ = F(θ),   M = ∫ ∂_θu (∂_θu)ᵀ dx,   F = ∫ ∂_θu · 𝓝(u) dx ,

which is then marched with any ODE integrator. The parameter-Jacobian ``J = ∂u/∂θ`` (N points × P
weights) is the whole engine — here via ``jax.jacfwd`` over the network weights — and the weight
step is a regularised least-squares projection of the spatial operator onto ``J``.

Method: **Neural Galerkin** — Bruna, Peherstorfer & Vanden-Eijnden, *J. Comput. Phys.* 496 (2024),
§2–3 (arXiv:2203.01360).

Test problem: 1-D heat ``u_t = ν u_xx`` on ``[0,1]``, ``u(x,0)=sin(πx)``, ``u=0`` at the ends, whose
exact solution is ``u(x,t) = e^{−νπ²t} sin(πx)``. The IC fit is a normal jNO ``crux.solve`` regression;
the time evolution is a bring-your-own-integrator loop (the transient-FEM pattern), since Neural
Galerkin marches an ODE for θ rather than doing gradient training.

Two practical points that make or break the scheme (both verified while building this):
  * **N ≥ P.** More collocation points than trainable weights, else θ̇ is underdetermined and the
    dynamics are garbage. We use a modest network (P≈140) and ~400 points.
  * **Regularise the projection.** The Gram ``M`` is notoriously ill-conditioned; a raw least-squares
    solve blows up. A small Tikhonov term ``(JᵀJ + λI)`` keeps θ̇ bounded and the march stable.
"""

import os

os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path  # noqa: E402

import equinox as eqx  # noqa: E402
import foundax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

jax.config.update("jax_enable_x64", True)  # the weight ODE + projection accumulate in float64

NU = 0.05  # diffusivity

# ---- domain, network, hard-BC ansatz -------------------------------------------------------
domain = jno.domain.line(mesh_size=0.0025)  # ~400 interior collocation points (N ≫ P)
x, _ = domain.variable("interior")
net = jno.nn.wrap(
    foundax.mlp(in_features=1, hidden_dims=10, num_layers=2, activation=jax.nn.tanh, key=jax.random.PRNGKey(0))
)
ansatz = x * (1 - x)  # vanishes at x=0,1  → hard Dirichlet u=0 at both ends
u = (ansatz * net(x)).scalar.bind(x=x)  # the network trial u(x; θ)

# ---- 1. initial condition: fit θ(0) so u(x;θ₀) ≈ sin(πx), through crux.solve ---------------
u_ic = jnn.sin(np.pi * x)
net.optimizer(optax.adam(optax.exponential_decay(3e-3, 3000, 0.5, end_value=2e-5)))
crux = jno.core([(u - u_ic).mse])
crux.solve(8000)
xs = np.asarray(crux.eval([x])).reshape(-1)  # concrete collocation points
ic_rel = float(
    np.linalg.norm(np.asarray(crux.eval([u])).reshape(-1) - np.sin(np.pi * xs)) / np.linalg.norm(np.sin(np.pi * xs))
)
print(f"\nNeural Galerkin heat 1D:  N={xs.size} collocation points")
print(f"  initial-condition fit  u(·,0) ≈ sin(πx):  rel-L2 = {ic_rel:.3e}")

# ---- 2. weight evolution: march  (JᵀJ + λI) θ̇ = Jᵀ (ν u_xx)  in time ------------------------
trainable, static = eqx.partition(net.module, eqx.is_array)  # same partition jax.jacfwd uses for J
theta, unravel = ravel_pytree(trainable)
P = theta.size
xs_j = jnp.asarray(xs)


def u_point(th, xp):  # scalar x → scalar u(x; θ), the SAME ansatz used above
    mod = eqx.combine(unravel(th), static)
    return xp * (1.0 - xp) * jnp.ravel(mod(jnp.reshape(xp, (1,))))[0]


DT, T, LAM = 1e-4, 0.5, 1e-4


@jax.jit
def ng_step(th):
    J = jax.vmap(lambda xp: jax.jacfwd(lambda t: u_point(t, xp))(th))(xs_j)  # (N, P) parameter-Jacobian
    u_xx = jax.vmap(lambda xp: jax.grad(jax.grad(lambda z: u_point(th, z)))(xp))(xs_j)  # (N,)
    theta_dot = jnp.linalg.solve(J.T @ J + LAM * jnp.eye(P), J.T @ (NU * u_xx))  # Tikhonov projection
    return th + DT * theta_dot


nsteps = round(T / DT)
for _ in range(nsteps):
    theta = ng_step(theta)

# ---- 3. verify the marched network against the analytic solution ---------------------------
pred = np.asarray(jax.vmap(lambda xp: u_point(jnp.asarray(theta), xp))(xs_j)).reshape(-1)
exact = np.exp(-NU * np.pi**2 * T) * np.sin(np.pi * xs)
rel = float(np.linalg.norm(pred - exact) / np.linalg.norm(exact))
print(f"  marched to t={T} ({nsteps} steps): Neural-Galerkin vs e^(−νπ²t)sin(πx):  rel-L2 = {rel:.3e}")

# ---- plot the marched profile vs analytic --------------------------------------------------
u0 = np.sin(np.pi * xs)
order = np.argsort(xs)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(xs[order], u0[order], "k--", lw=1, label="u(x,0) = sin(πx)")
ax.plot(xs[order], exact[order], color="tab:blue", lw=2.4, alpha=0.5, label=f"exact, t={T}")
ax.plot(xs[order], pred[order], color="tab:red", lw=1.2, label="Neural Galerkin")
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title(f"Neural Galerkin — 1D heat (ν={NU}), weights evolved in time")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "neural_galerkin_heat_1d.png", dpi=90)

assert ic_rel < 1e-2, f"IC fit failed: rel-L2={ic_rel:.3e}"
assert rel < 1e-2, f"Neural Galerkin march failed: rel-L2={rel:.3e}"
