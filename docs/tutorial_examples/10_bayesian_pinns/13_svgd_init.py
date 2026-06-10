"""13 — SVGD warm-start via ``.initialize()``"""

import os

# Two sequential solve() calls share device memory; pin CPU for
# portability (remove on hosts with enough VRAM).
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import blackjax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

π = jno.np.pi
A_true, B_true = 3.14, -2.71


def _build_problem():
    domain = jno.domain.line(mesh_size=0.02)
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
    crux = jno.core([residual.mse])
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


# ── Run 2: SVGD warm-start ──────────────────────────────────────────────────
def _svgd_only(p):
    # 32 particles, 300 SVGD iterations, init spread = 2.  The
    # particle ensemble approximates the posterior and serves as the
    # warm-start (mean for K=1, K distinct particles for K>1).
    p.initialize(
        jno.bayesian.svgd(
            num_iters=300,
            num_particles=32,
            init_jitter=2.0,
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


svgd_only = _run(label="svgd", configure_a=_svgd_only, configure_b=_svgd_only, total_epochs=300)

# ── Append summary to tutorial_results.txt ──────────────────────────────────
results_file = Path(__file__).parent.parent.parent / "tutorial_results.txt"
with open(results_file, "a") as f:
    for run in (baseline, svgd_only):
        rel_A = abs(run["A_mean"] - A_true) / abs(A_true)
        rel_B = abs(run["B_mean"] - B_true) / abs(B_true)
        f.write(
            f"10_bayesian_pinns/13_svgd_init.py | run={run['label']:10s} | "
            f"rel_A={rel_A:.4f} | rel_B={rel_B:.4f} | "
            f"rhat_a={run['rhat_a']:.3f} | rhat_b={run['rhat_b']:.3f} | "
            f"wall={run['wall']:.2f}s\n"
        )

# ── Asserts ─────────────────────────────────────────────────────────────────
for run in (baseline, svgd_only):
    rel_A = abs(run["A_mean"] - A_true) / abs(A_true)
    rel_B = abs(run["B_mean"] - B_true) / abs(B_true)
    assert rel_A < 0.3, f"{run['label']}: A off by {rel_A:.2%}"
    assert rel_B < 0.3, f"{run['label']}: B off by {rel_B:.2%}"

# SVGD warm-start should produce R-hat at least as good as the
# baseline; loose tolerance since K=2 R-hat is noisy.
assert svgd_only["rhat_a"] <= baseline["rhat_a"] + 0.5, (
    f"svgd R-hat A worse than baseline: {svgd_only['rhat_a']:.3f} vs {baseline['rhat_a']:.3f}"
)
