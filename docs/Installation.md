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

### Pre-built images

Images are published to GitHub Container Registry on release and on default-branch edge builds.

| Tag | Description |
|-----|-------------|
| `ghcr.io/<owner>/jno:latest` | CPU image (linux/amd64) |
| `ghcr.io/<owner>/jno:latest-cuda` | CUDA image (linux/amd64) |
| `ghcr.io/<owner>/jno:edge` | Default-branch edge image (CPU) |

### Run pre-built images

CPU:

```bash
docker pull ghcr.io/<owner>/jno:latest
docker run --rm ghcr.io/<owner>/jno:latest
```

CUDA (pin to GPU 0):

```bash
docker pull ghcr.io/<owner>/jno:latest-cuda
docker run --rm --gpus '"device=0"' ghcr.io/<owner>/jno:latest-cuda
```

### Build locally

CPU amd64:

```bash
docker build -f Dockerfile.amd64 -t jno:cpu-amd64 .
docker run --rm jno:cpu-amd64
```

CPU arm64:

```bash
docker build -f Dockerfile.arm64 -t jno:cpu-arm64 .
docker run --rm jno:cpu-arm64
```

CUDA:

```bash
docker build -f Dockerfile.cuda -t jno:cuda .
docker run --rm --gpus '"device=0"' jno:cuda
```

### Environment variables

| Variable | Image | Value | Description |
|----------|-------|-------|-------------|
| `DEBIAN_FRONTEND` | both | `noninteractive` | Suppress apt prompts during image build |
| `PATH` | both | `/app/.venv/bin:...` | Prioritize uv-managed environment in container |
| `JAX_PLATFORM_NAME` | CUDA | `gpu` | Prefer GPU backend in JAX |

### CI note

Docker images are built by `.github/workflows/docker-release.yml` and pushed to GHCR.

## Next Step

Continue with [Getting Started](Getting-Started.md) to run your first example.
