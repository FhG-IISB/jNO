"""12 — Laplace warm-start via ``.initialize()``

Demonstrates jno's *logdensity-aware initializer* hook with **Laplace
approximation** — a second concrete initializer that plugs into the
same :meth:`Model.initialize` API pathfinder uses:

.. code-block:: python

    a.initialize(jno.bayesian.laplace(map_steps=300,
                                      map_optimizer=optax.adam(1e-1),
                                      hessian_strategy="diagonal"))
    a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)

Algorithm
---------

1. **Find the MAP.**  Optimise ``-log p(theta | data)`` with an optax
   optimiser (Adam by default).  Implemented as a JIT-compiled
   ``jax.lax.scan`` over ``map_steps`` iterations.
2. **Compute the Hessian** ``H = -∇²log p`` at the MAP.  Two
   strategies:

   * ``hessian_strategy="full"`` materialises the full ``(D, D)``
     Hessian via :func:`jax.hessian`.  Numerically clean; memory cost
     grows as ``D²``.
   * ``hessian_strategy="diagonal"`` (default) computes only the
     diagonal of the Hessian via ``D`` Hessian-vector probes.
     Memory cost ``O(D)`` — required for BNN-scale problems.

3. **Posterior** ≈ ``N(MAP, H⁻¹)``.  For ``num_chains=1`` the warm
   position is the MAP; for ``num_chains>1`` we sample K i.i.d. warm
   positions from this Gaussian.  The diagonal of ``H⁻¹`` is returned
   as the kernel's ``inverse_mass_matrix``.

Side-by-side comparison
-----------------------

Same harmonic-regression problem as Tutorials 02 / 11.  Two
side-by-side runs, identical seeds:

============ ================================================= ================================
Run          ``.initialize(...)``                              ``.bayesian(warmup, adapt)``
============ ================================================= ================================
baseline     none                                              ``warmup=300, adapt=True``
laplace      ``laplace(map_steps=300, optax.adam(1e-1))``       ``warmup=0,   adapt=False``
============ ================================================= ================================

References
----------
MacKay, D. J. C. (1992).  *A Practical Bayesian Framework for
Backpropagation Networks.*  §6 (Laplace approximation around the
posterior mode).  Neural Computation, 4(3), 448-472.
https://doi.org/10.1162/neco.1992.4.3.448

Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M.,
& Hennig, P. (2021).  *Laplace Redux — Effortless Bayesian Deep
Learning.*  §2 (Laplace approximations for neural networks).
NeurIPS 2021.  https://arxiv.org/abs/2106.14806

Magnani, E., Krämer, N., Pförtner, M., & Hennig, P. (2024).
*Linearization Turns Neural Operators into Function-Valued Gaussian
Processes.*  §3.  https://arxiv.org/abs/2406.05072
"""

import os

# Two sequential solve() calls share device memory; pin CPU for
# portability (remove on hosts with enough VRAM).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import blackjax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402

π = jno.np.pi
# T02-scale truth — Laplace's Adam-based MAP search converges in
# ``map_steps=300`` for this magnitude.  For larger-scale truths
# (e.g. T11's (50, -30)), bump ``map_steps`` or use ``optax.sgd``
# with a tuned learning-rate schedule.
A_true, B_true = 3.14, -2.71


def _build_problem():
    domain = jno.domain(constructor=jno.domain.line(mesh_size=0.02))
    x, _ = domain.variable("interior")
    target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    a = jno.np.parameter((1,), key=k1, name="a")
    b = jno.np.parameter((1,), key=k2, name="b")

    residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
    return domain, a, b, residual


def _run(label, configure_a, configure_b, total_epochs):
    domain, a, b, residual = _build_problem()
    configure_a(a)
    configure_b(b)
    crux = jno.core([residual.mse], domain)
    t0 = time.perf_counter()
    crux.solve(total_epochs)
    wall = time.perf_counter() - t0

    a_chain = a.posterior_samples
    b_chain = b.posterior_samples
    rhat_a = float(jno.bayesian.rhat(a_chain)[0])
    rhat_b = float(jno.bayesian.rhat(b_chain)[0])
    A_mean = float(jnp.mean(a_chain))
    B_mean = float(jnp.mean(b_chain))
    print(f"[{label:10s}] A={A_mean:+.3f}  B={B_mean:+.3f}  R-hat A={rhat_a:.3f}  R-hat B={rhat_b:.3f}  wall={wall:.2f}s")
    return {
        "label": label,
        "A_mean": A_mean,
        "B_mean": B_mean,
        "rhat_a": rhat_a,
        "rhat_b": rhat_b,
        "wall": wall,
    }


# ── Run 1: baseline window adaptation from default zero init ────────────────
baseline = _run(
    label="baseline",
    configure_a=lambda p: p.bayesian(blackjax.nuts, step_size=1e-2, warmup=300, keep=300, num_chains=2, adapt=True),
    configure_b=lambda p: p.bayesian(blackjax.nuts, step_size=1e-2, warmup=300, keep=300, num_chains=2, adapt=True),
    total_epochs=600,
)


# ── Run 2: Laplace warm-start, no window adaptation ─────────────────────────
def _laplace_only(p):
    # Laplace finds the MAP via 300 Adam steps then forms a Gaussian
    # approximation; the diagonal of H^{-1} becomes the IMM.
    p.initialize(
        jno.bayesian.laplace(
            map_steps=300,
            map_optimizer=optax.adam(1e-1),
            hessian_strategy="diagonal",
        )
    )
    p.bayesian(
        blackjax.nuts,
        step_size=1e-2,
        inverse_mass_matrix=jnp.ones(1),
        warmup=0,
        keep=300,
        num_chains=2,
        adapt=False,
    )


laplace_only = _run(label="laplace", configure_a=_laplace_only, configure_b=_laplace_only, total_epochs=300)

# ── Append summary to tutorial_results.txt ──────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    for run in (baseline, laplace_only):
        rel_A = abs(run["A_mean"] - A_true) / abs(A_true)
        rel_B = abs(run["B_mean"] - B_true) / abs(B_true)
        f.write(
            f"10_bayesian_pinns/12_laplace_init.py | run={run['label']:10s} | "
            f"rel_A={rel_A:.4f} | rel_B={rel_B:.4f} | "
            f"rhat_a={run['rhat_a']:.3f} | rhat_b={run['rhat_b']:.3f} | "
            f"wall={run['wall']:.2f}s\n"
        )

# ── Asserts ─────────────────────────────────────────────────────────────────
for run in (baseline, laplace_only):
    rel_A = abs(run["A_mean"] - A_true) / abs(A_true)
    rel_B = abs(run["B_mean"] - B_true) / abs(B_true)
    assert rel_A < 0.3, f"{run['label']}: A off by {rel_A:.2%}"
    assert rel_B < 0.3, f"{run['label']}: B off by {rel_B:.2%}"

# Laplace warm-start should produce R-hat at least as good as the
# baseline (warm-start can't hurt mixing).  Loose tolerance — multichain
# R-hat with K=2 on a short chain is intrinsically noisy.
assert laplace_only["rhat_a"] <= baseline["rhat_a"] + 0.5, (
    f"laplace R-hat A worse than baseline: {laplace_only['rhat_a']:.3f} vs {baseline['rhat_a']:.3f}"
)
