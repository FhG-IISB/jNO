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

    ~50 worked examples across elliptic, parabolic, hyperbolic, coupled,
    inverse, integral, stochastic, FEM, and Bayesian problems.

-   :material-language-python: **[API Reference](API.md)**

    Auto-generated reference for `jno.core`, `jno.domain`, `jno.nn`,
    `jno.np`, and `jno.fn`.

</div>

---

## Capability matrix

What works today, what is experimental, what is not yet supported.

Each status badge links to the tutorial (or, where no dedicated tutorial exists, the closest reference page) that demonstrates the capability.

| Capability | Status | Notes |
|------------|--------|-------|
| Forward PINNs (residual minimisation) | [✅ stable](tutorials/01-basics/poisson-1d.md) | Hard or soft BC enforcement |
| Operator learning (DeepONet, FNO, PROSE via [foundax](https://github.com/FhG-IISB/foundax)) | [✅ stable](Getting-Started.md) | Combine with PDE residual or train purely data-driven |
| Inverse problems (parameter recovery, surrogate inversion) | [✅ stable](tutorials/05-coupled-and-inverse/inverse-parameter.md) | See [Inverse Problems](inverse-problems.md) |
| FEM / Variational PINNs | [✅ stable](tutorials/08-fem-and-varpinns/poisson-2d-fem.md) | TRI3 / TRI6 / QUAD4 elements, weak-form assembly |
| Adaptive resampling (RAD, RARD, CR3, R3, pinnfluence) | [✅ stable](adaptive/resampling.md) | See [Adaptive Resampling](adaptive/resampling.md) |
| Stochastic PDEs and noise nodes (gaussian / uniform / laplace) | [✅ stable](tutorials/07-stochastic/fokker-planck-2d.md) | Fokker–Planck, stochastic forcing |
| Bayesian PINNs (NUTS, HMC, MALA, SGLD, SGHMC, VI) | [✅ stable](tutorials/10-bayesian-pinns/index.md) | 14 worked tutorials — see [Bayesian Sampling](training/bayesian.md) |
| Parameter-efficient fine-tuning (LoRA, DoRA, rsLoRA, PiSSA, VeRA, LoKr, OFT, IA3) | [✅ stable](model-controls/lora.md) | Chain `.lora(...)` on any wrapped model |
| Training explainability (gradient conflict, NTK, Hessian, loss landscape, input sensitivity, residual stats) | [✅ stable](tutorials/07-analysis/gradient-conflict.md) | See [Explainability](training/explainability.md) |
| Foundation-model integration ([foundax](https://github.com/FhG-IISB/foundax) MLPs, transformers, DeepONet, FNO, PROSE) | [✅ stable](foundation_models/index.md) | Wrap any Equinox module via `jno.nn.wrap(...)` |
| Hybrid data + model parallelism | [✅ stable](training/parallelism.md) | `core(mesh=(batch, model))` |
| W&B logging + Orbax checkpointing | [✅ stable](tutorials/09-wandb/wandb-integration.md) | |
| IREE / MLIR compiled inference for deployment | [✅ stable](model-controls/iree.md) | |
| Hyperparameter / architecture search | [🟡 beta](Hyperparameter-Tuning.md) | Grid + Nevergrad |
| Multi-physics coupling | [🟡 beta](tutorials/05-coupled-and-inverse/hyco-poisson-1d.md) | HyCo Poisson tutorial; broader patterns developing |

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

Roughly fifty worked examples, ordered from simplest to most involved.
Start at 01 if you are new; jump into 05 (inverse), 08 (FEM), or 10
(Bayesian) if you have a specific goal.

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
| 10 Bayesian PINNs | NUTS / HMC chains, VI, inverse problems, BNN regression, warm-start initialisations (Pathfinder, Laplace, SVGD) |

The full nav is in the left sidebar; the source scripts live under
[`docs/tutorial_examples/`](https://github.com/FhG-IISB/jno/tree/main/docs/tutorial_examples).

---

## Cite jNO

Citing helps us justify continued development. See the
[`Cite this repository`](https://github.com/FhG-IISB/jno) button on
GitHub (reads [`CITATION.cff`](https://github.com/FhG-IISB/jno/blob/main/CITATION.cff))
or copy the BibTeX entry from the [README](https://github.com/FhG-IISB/jno#citation).
