# 11 — Operator Learning

Operator learning trains a network to map **inputs** (parameters, forcing functions, initial conditions) to **outputs** (PDE solutions) for an entire family of problems, instead of solving a single instance. jNO supports two complementary patterns:

- **PDE-residual operator learning** — the network sees one parametric instance per batch sample; the PDE residual is enforced at the collocation points. The network never sees ground-truth solutions, only the physics. Closest to "PINN with a parameter".
- **Data-driven operator learning** — the network is supervised on a dataset of `(input, solution)` pairs. No PDE residual is computed during training; the solution operator is learnt purely from examples.

The two architecture tutorials showcase **two foundax architectures** on the **same Poisson problem**, so the only variable is the architecture itself:

| Tutorial | Architecture | Pattern | What it teaches |
|---|---|---|---|
| [DeepONet — parametric Poisson](deeponet-poisson-2d.md) | `foundax.deeponet` | PDE-residual | Branch/trunk decomposition; the canonical operator-learning architecture |
| [FNO2D — supervised Poisson](fno-poisson-2d.md) | `foundax.fno2d` | Data-driven | Spectral convolutions in Fourier space; resolution-independent in principle |

Both plug into the same `jno.nn.wrap(...)` interface, so the rest of your training pipeline (callbacks, schedules, checkpointing, evaluation) is identical. Foundation models (PROSE, Poseidon, PDEformer-2) plug in the same way but are out of scope here — see the [Foundation Models](../../foundation_models/index.md) page for the fine-tuning workflow.
