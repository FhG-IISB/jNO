<p align="center">
  <img src="assets/logo.png" alt="jNO logo" width="500"/>
</p>

<p align="center">
    <a href="https://fhg-iisb.github.io/jNO/">
        <img src="https://img.shields.io/badge/docs-GitHub%20Pages-0aa?style=for-the-badge" alt="Dev Docs"/>
    </a>
    <a href="https://github.com/FhG-IISB/jno/actions/workflows/ci.yml">
        <img src="https://img.shields.io/github/actions/workflow/status/FhG-IISB/jno/ci.yml?branch=main&style=for-the-badge&label=tests" alt="Tests"/>
    </a>
    <a href="https://codecov.io/gh/FhG-IISB/jno">
        <img src="https://img.shields.io/codecov/c/github/FhG-IISB/jno/main?style=for-the-badge&label=coverage" alt="Coverage"/>
    </a>
    <a href="LICENSE">
        <img src="https://img.shields.io/badge/license-EPL--2.0-2ea44f?style=for-the-badge" alt="License"/>
    </a>
    <a href="CITATION.cff">
        <img src="https://img.shields.io/badge/cite-CITATION.cff-6b5b95?style=for-the-badge" alt="Citation"/>
    </a>
    <img src="https://img.shields.io/badge/docker-image%20available-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker image available"/>
    <a href="https://arxiv.org/abs/2605.10159">
        <img src="https://img.shields.io/badge/arXiv-2605.10159-b31b1b?style=for-the-badge" alt="arXiv Paper"/>
    </a>
</p>

**[Features](#what-you-can-do-with-jno)** · [Install](#install) · [Example](#example) · [Docs](https://fhg-iisb.github.io/jNO/) · [Citation](#citation)

**jNO** (jax Numerical Operators) is a JAX-native library for **differentiable
numerical methods**. Classical solvers — finite elements, finite differences,
and spectral (RCWA) — and scientific machine learning — PINNs, neural operators,
Bayesian inference — are two pillars on **one substrate**: you write the math
(a weak form, a strong-form stencil, a PDE residual, a data loss), and it lowers
to a single GPU-ready, end-to-end reverse-mode-differentiable, `jit`-compiled graph.

Because every solve is differentiable, the things that are usually separate
frameworks are the *same tool* here: an inverse problem, a PDE-constrained
optimization, and a neural-network coefficient are one composition away from a
forward solve — no glue code, no finite differences, no leaving JAX.

> [!NOTE]
> Research-level repository under active development. The public API is stabilising but may change between minor versions. Parts of the numerical-methods stack are marked *experimental* below — the scope and known limitations are stated on each docs page.

## What you can do with jNO

### Pillar 1 — Differentiable numerical methods

| Capability | Maturity | Notes |
|------------|----------|-------|
| **FEM, nodal** — `jno.fem` | [stable](https://fhg-iisb.github.io/jNO/fem/) | Lagrange P1 / P2 / P3+, **2-D & 3-D**; steady (linear + Newton), transient (θ-method), **second-order-in-time** (wave / elastodynamics), complex, periodic, coupled multifield |
| **FEM, non-nodal** — H(div) / H(curl) / C¹ | [experimental](https://fhg-iisb.github.io/jNO/fem/#non-nodal-element-families-hdiv-and-hcurl) | **Raviart–Thomas** (H(div)) and **Nédélec edge** (H(curl) — Maxwell, eddy currents) elements; **C¹ Hermite / Argyris / Morley** (plates, biharmonic) |
| **FDM** — `jno.fdm` | [stable](https://fhg-iisb.github.io/jNO/fdm/) | Strong-form collocation from a term list; **structured grids + geometric multigrid**; unstructured meshes; periodic, coupled, flux BCs; 2-D & 3-D |
| **Spectral / RCWA** — `jno.rcwa` | [stable](https://fhg-iisb.github.io/jNO/rcwa/) | Vector-Maxwell RCWA, anisotropic media, Jones / polarization readout |
| **Linear & nonlinear solvers** — `jno.solve` / `jno.precond` | [stable](https://fhg-iisb.github.io/jNO/) | Sparse-direct LU, Jacobi-BiCGStab, GMRES, CG, MINRES, Chebyshev, geometric multigrid, optional GPU **AMG** — matrix-free and differentiable |
| **Generalized eigenproblems** — `fem.eigs` | [beta](https://fhg-iisb.github.io/jNO/) | `K x = λ M x`, differentiable, M-orthonormal |
| **Time integration** | [stable](https://fhg-iisb.github.io/jNO/) | θ-method (backward-Euler / Crank–Nicolson), exponential integrators, adaptive step size |
| **Adaptive meshing** — `AdaptSpec` / `MovingBoundary` | [beta](https://fhg-iisb.github.io/jNO/) | Hessian-metric remeshing (AFEM), moving boundaries, r-adaptivity |
| **Differentiable inverse / PDE-constrained** | [stable](https://fhg-iisb.github.io/jNO/tutorials/05-coupled-and-inverse/inverse-parameter/) | Recover a scalar, a field `k(x)`, the geometry, or a **neural coefficient** through any solve — the gradient flows through the whole march |
| **Geometry** — `jno.Shape` / `jno.Path` | [stable](https://fhg-iisb.github.io/jNO/) | CSG via gmsh-OCC; conforming multi-material regions |

### Pillar 2 — Scientific machine learning

| Capability | Maturity | Notes |
|------------|----------|-------|
| Forward PINNs (residual minimisation) | [stable](https://fhg-iisb.github.io/jNO/tutorials/01-basics/poisson-1d/) | Hard or soft BC enforcement |
| Variational PINNs (weak-form losses) | [stable](https://fhg-iisb.github.io/jNO/tutorials/08-fem-and-varpinns/poisson-2d-fem/) | Network trial functions against the FEM weak form |
| Operator learning (DeepONet, FNO, U-Net, PROSE via [foundax](https://github.com/FhG-IISB/foundax)) | [stable](https://fhg-iisb.github.io/jNO/tutorials/11-operator-learning/) | PDE-residual or data-driven |
| Adaptive resampling (RAD, RARD, CR3, R3, pinnfluence) | [stable](https://fhg-iisb.github.io/jNO/adaptive/resampling/) | |
| Stochastic PDEs & noise nodes (gaussian / uniform / laplace) | [stable](https://fhg-iisb.github.io/jNO/tutorials/07-stochastic/fokker-planck-2d/) | Fokker–Planck, stochastic forcing |
| Bayesian PINNs (NUTS, HMC, MALA, SGLD, SGHMC, VI) | [stable](https://fhg-iisb.github.io/jNO/tutorials/10-bayesian-pinns/) | 14 worked tutorials |
| Parameter-efficient fine-tuning (LoRA, DoRA, rsLoRA, PiSSA, VeRA, LoKr, OFT, IA3) | [stable](https://fhg-iisb.github.io/jNO/model-controls/lora/) | Chain `.lora(...)` on any wrapped model |
| Training explainability (gradient conflict, NTK, Hessian, loss landscape, input sensitivity) | [stable](https://fhg-iisb.github.io/jNO/tutorials/07-analysis/gradient-conflict/) | |
| Foundation-model integration ([foundax](https://github.com/FhG-IISB/foundax) MLPs, transformers, DeepONet, FNO, PROSE) | [stable](https://fhg-iisb.github.io/jNO/foundation_models/) | Wrap any Equinox module via `jno.nn.wrap(...)` |
| Hybrid data + model parallelism | [stable](https://fhg-iisb.github.io/jNO/training/parallelism/) | `jno.core(..., mesh=(batch, model))` |
| W&B logging + Orbax checkpointing | [stable](https://fhg-iisb.github.io/jNO/tutorials/09-wandb/wandb-integration/) | |
| IREE / MLIR compiled inference for deployment | [stable](https://fhg-iisb.github.io/jNO/model-controls/iree/) | |

**One tracing language bridges the two.** A weak form, a strong-form stencil, a PDE residual for a network, and a supervised loss all lower to the same differentiable, `jit`-compiled graph — so a classical solve, a PINN, and an inverse problem *compose* rather than living in separate stacks. ~50 worked tutorials span elliptic, parabolic, hyperbolic, coupled, inverse, integral, stochastic, FEM / variational, Bayesian, and operator-learning problems — browse the [tutorials index](https://fhg-iisb.github.io/jNO/#tutorials).

## Install

```bash
pip install jax-numerical-operators
```

One install — FEM, FDM, the solver stack, PINNs, and the scientific-ML tooling all come in the box, with GPU support on by default (jNO depends on `jax[cuda]`). A couple of heavy, self-contained backends (the RCWA Fourier solver and GPU algebraic multigrid) stay optional; the [Installation guide](https://fhg-iisb.github.io/jNO/Installation/) covers those, plus Pixi, Docker, and pinning a specific CUDA build.

## Example

<details open>
<summary><strong>Differentiable FEM & FDM — one term list, weak or strong form</strong></summary>

```python
import jno

d = jno.Shape.rect(0, 0, 1, 1, size=0.05).domain()
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)

# The same BVP two ways — −Δu = 1 on the unit square, u = 0 on the boundary.

# FEM — the WEAK form is the term list (with a test function v):
u, v = d.fem_symbols()
ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
u_fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi,   # ∫∇u·∇v − ∫f·v = 0
                 u(xb, yb) - 0.0]).solve()

# FDM — the STRONG form: the same term list, no test function, collocated at the nodes:
w  = d.unknown()
wi = w.bind(x=xi, y=yi)
u_fdm = jno.fdm([-wi.d2(xi) - wi.d2(yi) - 1.0,           # −Δu = 1
                 w(xb, yb) - 0.0]).solve()

# Both are sparse, matrix-free, GPU-ready, and end-to-end differentiable —
# wrap either in an objective to recover a coefficient, a source, or the geometry.
```

</details>

<details>
<summary><strong>RCWA — a periodic metasurface, from the same constraint list</strong> (click to expand)</summary>

```python
import jno
import jax.numpy as jnp

# A periodic metasurface unit cell — a patterned high-index slab between two ambients.
K0 = 2 * jnp.pi                                            # vacuum wavenumber (wavelength λ = 1)
d = jno.Shape.box(0, 0, 0, 0.6, 0.6, 1.0, size=0.12).domain()
d.tag("bottom", lambda x, y, z: z < 0.01);  d.tag("top",   lambda x, y, z: z > 0.99)   # z ambients
d.tag("left",   lambda x, y, z: x < 0.01);  d.tag("right", lambda x, y, z: x > 0.59)   # x-periodic
d.tag("front",  lambda x, y, z: y < 0.01);  d.tag("back",  lambda x, y, z: y > 0.59)   # y-periodic

u, v = d.fem_symbols()
xi, yi, zi, _ = d.variable("interior", split=True)
ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
def on(tag):                                              # bind u, v on a named face
    s = d.variable(tag, split=True)
    return u.bind(x=s[0], y=s[1], z=s[2]), v.bind(x=s[0], y=s[1], z=s[2])
(ut, vt), (ub, vb) = on("top"), on("bottom")
ul, ur, uf, ubk = on("left")[0], on("right")[0], on("front")[0], on("back")[0]

slab = jno.fn(lambda x, y, z: jnp.where((0.4 < z) & (z < 0.6), 1.0, 0.0), [xi, yi, zi])
eps  = 1.0 + 10.0 * slab      # a patterned slab (swap in jno.np.parameter(...) for inverse design)

# the SAME scalar-Helmholtz list you'd hand jno.fem — rcwa infers period, layers, ε, and incidence:
sol = jno.rcwa([
    ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),   # ∇u·∇v − k₀²·ε·u·v
    -(1j * K0 * ut) * vt,                   # outgoing radiation (top ambient)
    -(1j * K0 * ub - 2j * K0) * vb,         # incident plane wave + radiation (bottom)
    ul - ur,  uf - ubk,                     # Floquet periodicity (x and y)
], orders=200).solve()

sol.efficiency("T")     # transmitted power fraction (needs the [rcwa] backend)
sol.order(+1, 0)        # a chosen diffraction order
```

</details>

<details>
<summary><strong>Operator learning — a DeepONet trained on a PDE residual</strong> (click to expand)</summary>

```python
import jno
import jax
import optax
import foundax

dir = jno.setup("./runs/test")

# Domain: `500 *` batches 500 random-coefficient samples of the same geometry
dom = 500 * jno.Shape.rect(0, 0, 2, 1, size=0.05).domain()
x, y, _ = dom.variable("interior")
xb, yb, _ = dom.variable("boundary")
k = dom.variable("k", jax.random.uniform(jax.random.PRNGKey(0), shape=(500, 1, 1), minval=0.5, maxval=1.5))

# Network + optimizer
fx = foundax.deeponet(n_sensors=1, coord_dim=2, basis_functions=32, hidden_dim=128, activation=jax.numpy.tanh)
net = jno.nn.wrap(fx)
net.optimizer(optax.adam(optax.schedules.cosine_decay_schedule(1e-3, 20_000, alpha=1e-5)))

# Hard BC enforcement via an output transform; the PDE residual is the loss
u = net(k, jno.np.concat([x, y], axis=-1)) * x * (2 - x) * y * (1 - y)
pde = k * (u.dd(x) + u.dd(y)) + 1.0

crux = jno.core(constraints=[pde.mse], domain=dom)
crux.solve(epochs=20_000, batchsize=32).plot(f"{dir}/training.png")
jno.save(crux, f"{dir}/model.pkl")
```

</details>

## Citation

If you use jNO in academic work, please cite:

```bibtex
@article{armbruster2026jno,
  title   = {jNO: A JAX Library for Neural Operator and Foundation Model Training},
  author  = {Armbruster, Leon and Ramesh, Rathan and Kruse, Georg and Straub, Christopher},
  journal = {arXiv preprint arXiv:2605.10159},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.10159},
  url     = {https://arxiv.org/abs/2605.10159}
}
```

## AI Disclosure

Parts of this codebase — including model ports, tests, and documentation — were developed with the assistance of AI coding tools. All contributions are reviewed and tested to the best of our ability, but mistakes may remain; please open an issue if you spot one.
