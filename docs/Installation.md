# Installation

Requires Python 3.11–3.13.

---

## PyPI

```bash
pip install jax-numerical-operators
```

That is the whole library: **FEM, FDM, the solver stack, PINNs and SciML all work from the core
install.** The finite-element assembler's `fenics-basix` backend (imports as `basix`) is a core
dependency, and `jno.fdm` has no optional dependency at all — neither needs an extra.

The core install is **CPU-only on purpose** — everything runs on a plain `jax` wheel, and a laptop,
CI runner, or macOS machine should not pay for multi-GB CUDA wheels it cannot use. An NVIDIA GPU
is one extra away:

```bash
pip install "jax-numerical-operators[cuda]"
```

## Optional extras

Everything below is imported **lazily**: the core install works without it, and each missing
backend raises a clear `ImportError` naming the extra that provides it. Install any of them as
`pip install "jax-numerical-operators[<extra>]"`; extras combine as `[fem,rcwa]`.

| Extra | Enables | Pulls in |
|---|---|---|
| `[cuda]` | **NVIDIA GPU support** — swaps in the CUDA-capable JAX build (same version pin as the core `jax`) | `jax[cuda]` |
| `[fem]` | **Everything `jno.fem` reaches for beyond the core** — the one-liner. Meta-extra = `[mesh]` + `[pardiso]` + `[cudss]`. | see the three below |
| `[mesh]` | Adaptive/anisotropic remeshing behind `fem.solve(adapt=...)` | `mmgpy` |
| `[pardiso]` | `jno.solve.lu(backend="pardiso")` — Intel MKL PARDISO, multithreaded CPU sparse-direct. The fastest factorization measured here, and the answer when a factorization exceeds GPU memory. x86-64 only. | `pypardiso` |
| `[cudss]` | `jno.solve.lu(backend="cudss")` — NVIDIA cuDSS, the fastest **repeated solve** (shift-invert eigensolves, constant-operator transients). Linux x86-64. | `nvmath-python`, `nvidia-cudss-cu12`, `cupy` |
| `[rcwa]` | The [RCWA solver](rcwa.md) (`jno.rcwa`, periodic-layered electromagnetics) | `fmmax` |
| `[amg]` | GPU algebraic multigrid — `jno.solve.amg` / `jno.precond.ams` | `jaxamg` (builds a CUDA extension against a prebuilt AmgX 2.5+; needs that toolchain) |
| `[iree]` | `model.to_iree(...)` export | `iree-base-compiler`, `iree-base-runtime` |
| `[dev]` | Test/lint tooling for working on jNO itself | pytest, ruff-era tooling, … |

```bash
pip install "jax-numerical-operators[fem]"        # FEM with remeshing + both direct backends
pip install "jax-numerical-operators[rcwa]"       # + the Fourier-modal EM solver
pip install "jax-numerical-operators[fem,rcwa]"   # combine freely
```

There is deliberately **no `[fdm]` extra** — finite differences ship in the core install.

The platform markers are part of the design: on a machine where a backend cannot run (PARDISO on
arm64, cuDSS on macOS) the extra installs cleanly and simply lands without it — no resolver
failure. A marker cannot detect whether a CUDA *device* is present, though: the cuDSS wheels
install on any Linux x86-64 host and raise at call time without a GPU, which is why
`jno.solve.lu()` never defaults to them.

The JAX pin is tight on purpose: jNO tracks a single JAX minor version per
release to avoid silently breaking on JAX API changes, and the `[cuda]`
extra carries the identical pin so CPU and GPU installs can never drift
apart.

If you need a specific CUDA toolkit version instead of the `[cuda]` extra,
install JAX from its own package index *before* installing jNO:

```bash
# CUDA 12 example
pip install --upgrade "jax[cuda12]>=0.10.1,<0.11"
pip install jax-numerical-operators
```

To pin a different JAX version locally for an experiment, see the
[JAX install matrix](https://jax.readthedocs.io/en/latest/installation.html).

---

## Clone + Pixi

For development or to run examples from source. Requires [pixi](https://pixi.sh).

```bash
git clone https://github.com/FhG-IISB/jNO.git
cd jNO
pixi install
```

Common tasks:

```bash
pixi run fmt     # format with ruff
pixi run lint    # lint and auto-fix
pixi run test    # run the test suite
```

The pip extras above map to pixi **environments**: `pixi run -e fem ...`, `-e rcwa`, `-e iree`,
and `-e dev` (adds matplotlib and the test tooling).

---

## Docker

CPU:

```bash
docker run --rm ghcr.io/fhg-iisb/jno:latest
```

GPU (requires NVIDIA drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker run --rm --gpus all ghcr.io/fhg-iisb/jno:latest
```

Build locally:

```bash
docker build -t jno:latest .
docker run --rm --gpus all jno:latest
```
