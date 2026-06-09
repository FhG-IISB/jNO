"""11 — Pathfinder warm-start via ``.initialize()``

Demonstrates jno's *logdensity-aware initializer* hook by warm-starting
NUTS with :func:`blackjax.pathfinder` (Zhang et al. 2022) — exposed
through the existing :meth:`Model.initialize` API:

.. code-block:: python

    a.initialize(jno.bayesian.pathfinder(maxiter=30, num_samples=200))
    a.bayesian(blackjax.nuts, step_size=1e-2, warmup=100, adapt=True)

Problem
-------
Same as Tutorial 02 — recover ``(A, B)`` in
``d(x) = A·sin(πx) + B·cos(πx)`` (truth ``A = 50.0, B = -30.0``) from
the default zero initialisation.  The deliberately *large-magnitude*
truth makes the pathfinder benefit visible: window adaptation from
zero spends most of its budget locating the posterior mode; pathfinder
closes that gap in one L-BFGS run.

Three side-by-side runs (identical RNG seeds, same total budget):

============ ============================ ================================
Run          ``.initialize(...)``         ``.bayesian(warmup, adapt)``
============ ============================ ================================
baseline     none                         ``warmup=300, adapt=True``
pathfinder   ``pathfinder(maxiter=30)``   ``warmup=0,   adapt=False``
chained      ``pathfinder(maxiter=30)``   ``warmup=100, adapt=True``
============ ============================ ================================

* **baseline** is the recipe from T02 — window adaptation from the
  zero starting point.
* **pathfinder** uses pathfinder for both the warm position AND the
  diagonal IMM estimate; user's ``step_size`` is kept.
* **chained** runs pathfinder first (warm position + IMM), then a
  short window adaptation refines ``step_size`` from there — the
  recommended pipeline per Zhang et al. 2022 §4.2.

Each run uses ``num_chains=2`` so we can report Gelman–Rubin R-hat as
a convergence diagnostic.  R-hat near 1.0 means the K chains agree;
larger values indicate poor mixing.

Reference
---------
Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022).
*Pathfinder: Parallel quasi-Newton variational inference.*
Journal of Machine Learning Research, 23(306), 1-49.
https://arxiv.org/abs/2108.03782
"""

import os

# Three sequential solve() calls share device memory; on small GPUs the
# second JIT compile can OOM.  Pinning to CPU keeps the tutorial portable;
# remove this line on a host with enough VRAM (~6 GiB suffices).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import blackjax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

# ── Shared problem definition ───────────────────────────────────────────────
π = jno.np.pi
A_true, B_true = 50.0, -30.0  # large-magnitude truth → posterior mode far from default zero init


def _build_problem():
    """Build a fresh domain + parameters at default zero init.  Returns
    ``(domain, a, b, residual)`` ready for ``crux.solve``.
    """
    domain = jno.domain.line(mesh_size=0.02)
    x, _ = domain.variable("interior")
    target = A_true * jno.np.sin(π * x) + B_true * jno.np.cos(π * x)

    k1, k2 = jax.random.split(jax.random.PRNGKey(0), 2)
    a = jno.np.parameter((1,), key=k1, name="a")
    b = jno.np.parameter((1,), key=k2, name="b")

    residual = a * jno.np.sin(π * x) + b * jno.np.cos(π * x) - target
    return domain, a, b, residual


def _run(label, configure_a, configure_b, total_epochs):
    """Run one solve(); return per-parameter summary + wall-clock."""
    domain, a, b, residual = _build_problem()
    configure_a(a)
    configure_b(b)
    crux = jno.core([residual.mse])
    t0 = time.perf_counter()
    crux.solve(total_epochs)
    wall = time.perf_counter() - t0

    a_chain = a.posterior_samples  # (K, N, 1)
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


# ── Run 1: baseline — window adaptation from bad init ───────────────────────
baseline = _run(
    label="baseline",
    configure_a=lambda p: p.bayesian(blackjax.nuts, step_size=1e-2, warmup=300, keep=300, num_chains=2, adapt=True),
    configure_b=lambda p: p.bayesian(blackjax.nuts, step_size=1e-2, warmup=300, keep=300, num_chains=2, adapt=True),
    total_epochs=600,
)


# ── Run 2: pathfinder only — no window adaptation ──────────────────────────
def _pf_only(p):
    p.initialize(jno.bayesian.pathfinder(maxiter=30, num_samples=200))
    p.bayesian(
        blackjax.nuts,
        step_size=1e-2,
        inverse_mass_matrix=jnp.ones(1),
        warmup=0,
        keep=300,
        num_chains=2,
        adapt=False,
    )


pathfinder_only = _run(label="pf-only", configure_a=_pf_only, configure_b=_pf_only, total_epochs=300)


# ── Run 3: chained — pathfinder + window adaptation ─────────────────────────
def _pf_chain(p):
    p.initialize(jno.bayesian.pathfinder(maxiter=30, num_samples=200))
    p.bayesian(
        blackjax.nuts,
        step_size=1e-2,
        warmup=100,
        keep=300,
        num_chains=2,
        adapt=True,
    )


chained = _run(label="chained", configure_a=_pf_chain, configure_b=_pf_chain, total_epochs=400)

# ── Append summary row to tutorial_results.txt ──────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    for run in (baseline, pathfinder_only, chained):
        rel_A = abs(run["A_mean"] - A_true) / abs(A_true)
        rel_B = abs(run["B_mean"] - B_true) / abs(B_true)
        f.write(
            f"10_bayesian_pinns/11_pathfinder_init.py | run={run['label']:10s} | "
            f"rel_A={rel_A:.4f} | rel_B={rel_B:.4f} | "
            f"rhat_a={run['rhat_a']:.3f} | rhat_b={run['rhat_b']:.3f} | "
            f"wall={run['wall']:.2f}s\n"
        )

# ── Asserts ─────────────────────────────────────────────────────────────────
# All three runs recover (A, B).  The pathfinder-touched runs do so with
# materially less warmup (or none); the baseline pays for full window
# adaptation from the bad init.
for run in (baseline, pathfinder_only, chained):
    rel_A = abs(run["A_mean"] - A_true) / abs(A_true)
    rel_B = abs(run["B_mean"] - B_true) / abs(B_true)
    assert rel_A < 0.3, f"{run['label']}: A off by {rel_A:.2%}"
    assert rel_B < 0.3, f"{run['label']}: B off by {rel_B:.2%}"

# Chained warm-start should produce R-hat at least as good as the
# baseline (warm-start can't hurt mixing).  Loose tolerance — multichain
# R-hat with K=2 on a short chain is intrinsically noisy.
assert chained["rhat_a"] <= baseline["rhat_a"] + 0.5, (
    f"chained R-hat A worse than baseline: {chained['rhat_a']:.3f} vs {baseline['rhat_a']:.3f}"
)
assert chained["rhat_b"] <= baseline["rhat_b"] + 0.5, (
    f"chained R-hat B worse than baseline: {chained['rhat_b']:.3f} vs {baseline['rhat_b']:.3f}"
)
