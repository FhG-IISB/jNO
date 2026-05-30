# Installation

Requires Python 3.11–3.13.

---

## PyPI

```bash
pip install "jax-neural-operators[fem]"
```

GPU support is included by default — `jax-neural-operators` depends on
`jax[cuda]>=0.10.1,<0.11`, so a standard `pip install` already pulls a
CUDA-capable JAX wheel. The pin is tight on purpose: jNO tracks a single
JAX minor version per release to avoid silently breaking on JAX API
changes.

If you need a specific CUDA toolkit version, install JAX from its own
package index *before* installing jNO:

```bash
# CUDA 12 example
pip install --upgrade "jax[cuda12]>=0.10.1,<0.11"
pip install "jax-neural-operators[fem]"
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
