# Docker

jNO ships two Docker images: a CPU image based on Ubuntu 24.04 and a CUDA image for GPU-accelerated runs.

---

## Pre-built images

Images are published to the GitHub Container Registry automatically by CI on every release.

| Tag | Description |
|-----|-------------|
| `ghcr.io/<owner>/jno:latest` | CPU image (linux/amd64) |
| `ghcr.io/<owner>/jno:latest-cuda` | CUDA 12 image (linux/amd64, requires NVIDIA GPU) |
| `ghcr.io/<owner>/jno:edge` | Latest commit on the default branch (CPU) |

### Pull and run (CPU)

```bash
docker pull ghcr.io/<owner>/jno:latest
docker run --rm ghcr.io/<owner>/jno:latest
```

### Pull and run (CUDA)

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

```bash
docker pull ghcr.io/<owner>/jno:latest-cuda
docker run --rm --gpus all ghcr.io/<owner>/jno:latest-cuda
```

The CUDA image sets `JAX_PLATFORM_NAME=gpu` and prints the detected JAX devices on startup.

---

## Building locally

> **Note:** The images are built for `linux/amd64`. Do not build on ARM via QEMU — the heavy Python compilation steps are unstable under emulation. Use GitHub Actions or a native amd64 machine.

### CPU image (`Dockerfile`)

```bash
docker build -t jno:cpu .
docker run --rm jno:cpu
```

**Base image:** `ubuntu:24.04`  
**Python:** 3.12 (from the Ubuntu 24.04 system packages)  
**Install:** `uv sync --extra iree --extra dev` from the lockfile

### CUDA image (`Dockerfile.cuda`)

```bash
docker build -f Dockerfile.cuda -t jno:cuda .
docker run --rm --gpus all jno:cuda
```

**Base image:** `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`  
**Python:** 3.12 (added via the [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) since Ubuntu 22.04 ships 3.10)  
**Install:** two-step process explained below

#### Why a two-step install for CUDA?

`uv.lock` is generated on a CPU machine and does not contain CUDA platform wheels. A plain `uv sync --extra cuda` would fail inside the container. Instead:

1. **Step 1 — install from lockfile (CPU wheels):**
   ```dockerfile
   RUN uv sync --extra iree --extra dev --no-extra cuda
   ```
2. **Step 2 — overlay CUDA JAX (bypasses lockfile):**
   ```dockerfile
   RUN uv pip install --reinstall "jax[cuda12]"
   ```
   This fetches `jaxlib-cuda12`, `cublas`, `cusparse`, and related wheels directly from PyPI, compiled against CUDA 12.x.

---

## System dependencies

Both images install the following system libraries required by jNO's mesh and rendering backends:

| Library | Purpose |
|---------|---------|
| `libglu1-mesa` | OpenGL utilities (mesh processing) |
| `libgl1` | OpenGL runtime |
| `libxrender1`, `libxcursor1`, `libxft2`, `libxinerama1` | X11 display libraries |

---

## Environment variables

| Variable | Image | Value | Description |
|----------|-------|-------|-------------|
| `DEBIAN_FRONTEND` | both | `noninteractive` | Suppresses apt prompts during build |
| `PATH` | both | `/app/.venv/bin:...` | Puts the uv-managed venv first |
| `JAX_PLATFORM_NAME` | CUDA only | `gpu` | Tells JAX to prefer the GPU; falls back to CPU if no GPU is present |

---

## CI / GitHub Actions

Images are built and pushed by `.github/workflows/docker-release.yml`. The workflow:

- Triggers on new GitHub releases
- Builds the CPU image with `Dockerfile` and pushes to `ghcr.io/<owner>/jno:latest`
- Builds the CUDA image with `Dockerfile.cuda` and pushes to `ghcr.io/<owner>/jno:latest-cuda`
- Also pushes an `edge` tag on every push to the default branch

Authentication uses `GITHUB_TOKEN` with `packages: write` permission — no additional secrets are required.
