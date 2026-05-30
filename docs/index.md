# jNO Documentation

**jNO** (jax Neural Operators) is a JAX-native library for training
neural operators, physics-informed networks, and PDE foundation models.
Everything — the PDE residual, the supervised loss, the FEM weak form,
the sensitivity diagnostic — is written in one symbolic *tracing*
language and compiled into one optimisation pipeline.

---

## Get started

<div class="grid cards" markdown>

-   :material-download-box-outline: **[Install](Installation.md)**

    Pip, Pixi, or Docker — under 30 seconds. GPU support is on by
    default.

-   :material-rocket-launch-outline: **[30-min Quickstart](Getting-Started.md)**

    A 2-D Poisson problem with random diffusion, trained end-to-end.
    Teaches the full pipeline.

-   :material-book-open-page-variant-outline: **[Tutorials](#tutorials)**

    40 worked examples across elliptic, parabolic, hyperbolic, coupled,
    inverse, integral, stochastic, and FEM problems.

-   :material-language-python: **[API Reference](API.md)**

    Auto-generated reference for `jno.core`, `jno.domain`, `jno.nn`,
    `jno.np`, and `jno.fn`.

</div>

---

## Capability matrix

What works today, what is experimental, what is not yet supported.

| Capability | Status | Notes |
|------------|--------|-------|
| Forward PINNs (residual minimisation) | ✅ stable | Hard or soft BC enforcement |
| Operator learning (DeepONet, FNO via [foundax](https://github.com/FhG-IISB/foundax)) | ✅ stable | Combine with PDE residual or train purely data-driven |
| Inverse problems (parameter recovery, surrogate inversion) | ✅ stable | See [Inverse Problems](inverse-problems.md) |
| FEM / Variational PINNs | ✅ stable | TRI3/TRI6/QUAD4 elements, weak-form assembly |
| Adaptive resampling (RAD, RARD, CR3 causal) | ✅ stable | See [Adaptive Resampling](adaptive/resampling.md) |
| Hybrid data + model parallelism | ✅ stable | `core(mesh=(batch, model))` |
| W&B logging, gradient-conflict explainability | ✅ stable | See [Explainability](training/explainability.md) |
| Stochastic PDEs (Fokker–Planck, noisy forcing) | ✅ stable | See [`07_stochastic`](tutorials/07-stochastic/fokker-planck-2d.md) |
| Hyperparameter / architecture search | 🟡 beta | Grid + Nevergrad; see [Tuning](Hyperparameter-Tuning.md) |
| Multi-physics coupling | 🟡 beta | HyCo Poisson tutorial; broader patterns developing |
| Pretrained operator model zoo | ⛔ planned | Tracked in roadmap |
| Bayesian / ensemble UQ helpers | ⛔ planned | Ensemble runs already possible via multiple seeds |

---

## Common terminology

A few jNO-specific terms appear throughout the docs. See the
[Glossary](Glossary.md) for full definitions:

- **Trace / tracing system** — the unified symbolic graph that holds
  domain points, network calls, residuals, and losses.
- **Placeholder** — any symbolic node in that graph (a Variable, a
  network call, an operator).
- **Constraint** — a single optimisable expression passed to
  `jno.core([…], dom)`. Typically `pde.mse`, but can be any scalar.
- **Model controls** — fine-grained per-parameter knobs (freeze, LoRA,
  initialisation, dtype) configured on a wrapped network.
- **Mesh** — overloaded: a spatial mesh for the PDE domain *and* a
  device mesh `(batch, model)` for parallelism.

---

## Tutorials

Forty worked examples, ordered from simplest to most involved. Start at
01 if you are new; jump into 05 (inverse) or 08 (FEM) if you have a
specific goal.

| Group | Topic |
|-------|-------|
| 01 Basics | 1-D Laplace, Poisson, Biharmonic |
| 02 Elliptic | Anisotropic / variable-coefficient Poisson, Helmholtz, mixed BC |
| 03 Parabolic | Heat 1-D/2-D, Reaction-Diffusion, Allen–Cahn |
| 04 Hyperbolic | Advection–Diffusion, Burgers, Telegraph, Wave |
| 05 Coupled & Inverse | Coupled systems, parameter recovery, surrogate inversion, HyCo |
| 06 Integration | Boundary flux, divergence theorem, Fredholm, integro-differential |
| 07 Stochastic | Fokker–Planck, stochastic forcing |
| 08 FEM & Variational PINNs | Poisson 2-D, Allen–Cahn, Robin BC, Helmholtz 3-D |
| 09 W&B logging | Reproducible training runs with Weights & Biases |

The full nav is in the left sidebar; the source scripts live under
[`docs/tutorial_examples/`](https://github.com/FhG-IISB/jno/tree/main/docs/tutorial_examples).

---

## Cite jNO

Citing helps us justify continued development. See the
[`Cite this repository`](https://github.com/FhG-IISB/jno) button on
GitHub (reads [`CITATION.cff`](https://github.com/FhG-IISB/jno/blob/main/CITATION.cff))
or copy the BibTeX entry from the [README](https://github.com/FhG-IISB/jno#citation).
