# Installation

Requires Python 3.11–3.13.

---

## PyPI

```bash
pip install "jax-neural-operators[fem]"
```

Add `cuda` for GPU support:

```bash
pip install "jax-neural-operators[fem,cuda]"
```

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
