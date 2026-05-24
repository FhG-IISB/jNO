# syntax=docker/dockerfile:1
# ------------------------------------------------------------
# jNO — linux/amd64, Ubuntu 24.04
#
# JAX bundles its own CUDA runtime (jax-cuda12-plugin[with-cuda]),
# so no NVIDIA base image is required. You only need NVIDIA drivers
# and the NVIDIA Container Toolkit on the host.
#
# Run with GPU:
#   docker run --gpus all ghcr.io/<owner>/jno:latest
#
# HPC / Apptainer:
#   apptainer pull jno.sif docker://ghcr.io/<owner>/jno:latest
#   apptainer run --nv jno.sif
# ------------------------------------------------------------
FROM --platform=linux/amd64 ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System libs required by gmsh (via pygmsh) at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libglu1-mesa \
        libgl1 \
        libxrender1 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

WORKDIR /app

# Copy lockfile and metadata before source for better layer caching
COPY pyproject.toml pixi.lock README.md ./
COPY jno/ ./jno/

# Install exact versions from the lock file — no network resolution at build time
RUN pixi install --frozen

# Prefer GPU; falls back to CPU if no GPU is present at runtime
ENV JAX_PLATFORM_NAME=gpu

CMD ["pixi", "run", "python", "-c", \
     "import jax; import jno; print('jNO ready — devices:', jax.devices())"]
