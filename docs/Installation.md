# Installation

This page is the single setup reference for jNO, including Python environments and Docker.

## Prerequisites

- Python 3.11 to 3.13
- One package manager: [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended), [micromamba/conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), or [pip](https://pip.pypa.io/en/stable/installation/)
- For CUDA profile: NVIDIA drivers on host

## Quick Install

Choose support profile, then Python version, then installer.

=== "CPU + FEM"

	=== "Python 3.11"

		=== "micromamba / conda"

			```bash
			micromamba create -n jno python=3.11 pip -y
			micromamba activate jno
			micromamba install -n jno -c conda-forge gmsh python-gmsh suitesparse -y
			pip install "jax-neural-operators[fem]"
			```

		=== "uv"

			```bash
			uv venv .jno --python 3.11
			source .jno/bin/activate
			uv pip install "jax-neural-operators[fem]"
			```

		=== "pip only"

			```bash
			python3.11 -m pip install "jax-neural-operators[fem]"
			```

	=== "Python 3.12"

		=== "micromamba / conda"

			```bash
			micromamba create -n jno python=3.12 pip -y
			micromamba activate jno
			micromamba install -n jno -c conda-forge gmsh python-gmsh suitesparse -y
			pip install "jax-neural-operators[fem]"
			```

		=== "uv"

			```bash
			uv venv .jno --python 3.12
			source .jno/bin/activate
			uv pip install "jax-neural-operators[fem]"
			```

		=== "pip only"

			```bash
			python3.12 -m pip install "jax-neural-operators[fem]"
			```

	=== "Python 3.13"

		=== "micromamba / conda"

			```bash
			micromamba create -n jno python=3.13 pip -y
			micromamba activate jno
			micromamba install -n jno -c conda-forge gmsh python-gmsh suitesparse -y
			pip install "jax-neural-operators[fem]"
			```

		=== "uv"

			```bash
			uv venv .jno --python 3.13
			source .jno/bin/activate
			uv pip install "jax-neural-operators[fem]"
			```

		=== "pip only"

			```bash
			python3.13 -m pip install "jax-neural-operators[fem]"
			```

=== "CUDA + FEM"

	=== "Python 3.11"

		=== "micromamba / conda"

			```bash
			micromamba create -n jno python=3.11 pip -y
			micromamba activate jno
			micromamba install -n jno -c conda-forge gmsh python-gmsh suitesparse -y
			pip install "jax-neural-operators[fem,cuda]"
			```

		=== "uv"

			```bash
			uv venv .jno --python 3.11
			source .jno/bin/activate
			uv pip install "jax-neural-operators[fem,cuda]"
			```

		=== "pip only"

			```bash
			python3.11 -m pip install "jax-neural-operators[fem,cuda]"
			```

	=== "Python 3.12"

		=== "micromamba / conda"

			```bash
			micromamba create -n jno python=3.12 pip -y
			micromamba activate jno
			micromamba install -n jno -c conda-forge gmsh python-gmsh suitesparse -y
			pip install "jax-neural-operators[fem,cuda]"
			```

		=== "uv"

			```bash
			uv venv .jno --python 3.12
			source .jno/bin/activate
			uv pip install "jax-neural-operators[fem,cuda]"
			```

		=== "pip only"

			```bash
			python3.12 -m pip install "jax-neural-operators[fem,cuda]"
			```

	=== "Python 3.13"

		=== "micromamba / conda"

			```bash
			micromamba create -n jno python=3.13 pip -y
			micromamba activate jno
			micromamba install -n jno -c conda-forge gmsh python-gmsh suitesparse -y
			pip install "jax-neural-operators[fem,cuda]"
			```

		=== "uv"

			```bash
			uv venv .jno --python 3.13
			source .jno/bin/activate
			uv pip install "jax-neural-operators[fem,cuda]"
			```

		=== "pip only"

			```bash
			python3.13 -m pip install "jax-neural-operators[fem,cuda]"
			```

## Docker

Use Docker when you want a reproducible runtime without managing local Python dependencies.
The image targets `linux/amd64` and includes JAX's bundled CUDA runtime — no NVIDIA base
image needed. You only need NVIDIA drivers and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
on the host.

### Pre-built image

Images are published to GitHub Container Registry on each release.

| Tag | Description |
|-----|-------------|
| `ghcr.io/<owner>/jno:latest` | linux/amd64, GPU-capable (falls back to CPU) |
| `ghcr.io/<owner>/jno:<version>` | pinned release, e.g. `0.2.1` |

### Run

CPU only:

```bash
docker run --rm ghcr.io/<owner>/jno:latest
```

With GPU:

```bash
docker run --rm --gpus all ghcr.io/<owner>/jno:latest
```

### HPC / Apptainer

Most HPC clusters use [Apptainer](https://apptainer.org) (formerly Singularity).
Convert the Docker image once and run it with GPU passthrough:

```bash
apptainer pull jno.sif docker://ghcr.io/<owner>/jno:latest
apptainer run --nv jno.sif
```

### Build locally

```bash
docker build -t jno:latest .
docker run --rm --gpus all jno:latest
```

### Environment variables

| Variable | Value | Description |
|----------|-------|-------------|
| `DEBIAN_FRONTEND` | `noninteractive` | Suppress apt prompts during image build |
| `JAX_PLATFORM_NAME` | `gpu` | Prefer GPU backend; falls back to CPU if no GPU present |

### CI note

Images are built and pushed by `.github/workflows/docker-release.yml` on each GitHub release.

## Development Setup

If you are contributing to jNO, use [pixi](https://pixi.sh) to get a fully reproducible environment from the lock file:

```bash
curl -fsSL https://pixi.sh/install.sh | bash   # install pixi once
git clone https://github.com/FhG-IISB/jNO.git
cd jNO
pixi install                                     # installs exact locked deps
```

Then use the built-in tasks for everyday work:

```bash
pixi run fmt     # auto-format with ruff
pixi run lint    # lint and auto-fix with ruff
pixi run test    # run the fast test suite
```

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full workflow.

## Next Step

Continue with [Getting Started](Getting-Started.md) to run your first example.
