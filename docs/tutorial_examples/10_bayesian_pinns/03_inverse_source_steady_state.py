"""03 — Bayesian PINN inverse: mixed-mode source recovery (elliptic PDE)

Problem
-------
Steady-state heat equation with an unknown forcing amplitude ``A``:

    α u''(x) + A · sin(πx) = 0,    x ∈ [0, 1],    u(0) = u(1) = 0.

The analytical solution is

    u_exact(x) = (A / (α π²)) · sin(πx),

so the data ``u_obs = u_exact(x; A_true)`` carries direct information about
``A``.  We treat ``A`` as an unknown and recover its posterior with NUTS.

This is the canonical Bayesian inverse-PDE pattern in 1-D: one PDE
residual, one data term, one unknown coefficient, and a posterior at the
end.  A transient version (`u_t = α u_xx + A sin(πx)`) follows exactly the
same recipe with `jno.domain.line(..., time=(0, T_end, N))` and an
ansatz that hard-enforces the initial condition — at the cost of a
substantially slower JIT compile.

Compile time
------------
The first run JIT-compiles the NUTS-in-jit step function and takes
~10-30 s on CPU; subsequent runs of the *same script* reload the compiled
artifact from JAX's persistent cache (``~/.cache/jno/xla_cache``, enabled
automatically by ``jno.core.solve``) and start in ~1 s.  Knobs that cut
compile cost:
* swap ``blackjax.nuts`` for ``blackjax.mala`` (single leapfrog) or
  ``blackjax.hmc(num_integration_steps=4)``,
* lower ``max_num_doublings`` on NUTS (default 10 → try 3 or 4),
* shrink the surrogate net / mesh while prototyping.

Technique
---------
* ``A`` is a scalar trainable parameter, sampled with NUTS.
* ``net(x)`` is a small MLP, *optimised* with Adam — provides the surrogate
  ``u(x)`` whose Laplacian appears in the PDE residual.
* Loss = PDE-residual MSE + data-fit MSE against the analytical observation.

Reference
---------
Hoffman, M. D., & Gelman, A. (2014).  *The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo.*  JMLR 15(1), 1593-1623.
"""

from pathlib import Path

import blackjax
import foundax
import jax
import jax.numpy as jnp
import optax

import jno

π = jno.np.pi

# ── Physical setup ────────────────────────────────────────────────────────────
α = 0.1
A_true = 2.0

# ── Domain (steady-state, 1-D) ────────────────────────────────────────────────
domain = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
x, _ = domain.variable("interior")
xb, _ = domain.variable("boundary")

# ── Trainable source amplitude — Bayesian ────────────────────────────────────
A = jno.np.parameter((1,), key=jax.random.PRNGKey(0), name="A")
A.bayesian(
    blackjax.nuts,
    step_size=2e-2,
    warmup=80,
    keep=150,
    max_num_doublings=4,
    # Mixed mode (A is Bayesian, net is optax) — disable window adaptation
    # because the adapter would run once at the START against the
    # *untrained* MLP surrogate, producing a step_size tuned for the wrong
    # logdensity. With adapt=False, `warmup=80` reverts to the classic
    # "discard first N samples" meaning, which is the safe choice for
    # mixed-mode runs.
    adapt=False,
)

# ── Surrogate u(x) — deterministic optax ─────────────────────────────────────
net = jno.nn.wrap(
    foundax.mlp(
        in_features=1,
        hidden_dims=16,
        num_layers=2,
        key=jax.random.PRNGKey(1),
    )
)
net.optimizer(optax.adam(1e-3))

# Hard-enforce BC u(0) = u(1) = 0 via ansatz.
u = net(x) * x * (1 - x)

# ── Constraints ───────────────────────────────────────────────────────────────
# PDE residual: α u'' + A sin(πx) = 0
u_xx = u.d2(x, scheme="finite_difference")
pde = α * u_xx + A * jno.np.sin(π * x)

# Synthetic noiseless observation: closed-form solution at A_true.
u_obs = (A_true / (α * π**2)) * jno.np.sin(π * x)
data = u - u_obs

# ── Solve ─────────────────────────────────────────────────────────────────────
crux = jno.core([pde.mse, data.mse], domain)
crux.solve(230)

# ── Posterior summary ────────────────────────────────────────────────────────
A_chain = A.posterior_samples  # shape (150, 1)
A_mean = float(jnp.mean(A_chain))
A_lo, A_hi = (float(v) for v in jnp.quantile(A_chain, jnp.array([0.05, 0.95])))

print(f"A = {A_mean:.3f}  90% CI = [{A_lo:.3f}, {A_hi:.3f}]   truth = {A_true}")

rel_A = abs(A_mean - A_true) / abs(A_true)

results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    f.write(
        f"10_bayesian_pinns/03_inverse_source_steady_state.py | epochs=230 | "
        f"rel_A={rel_A:.4f} | CI_width={A_hi - A_lo:.4f}\n"
    )

assert rel_A < 0.5, f"posterior-mean A off by {rel_A:.2%}"
