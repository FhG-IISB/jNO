# syntax=docker/dockerfile:1
# ------------------------------------------------------------
# jNO – CPU image (linux/amd64, Ubuntu 24.04)
#
# Built by GitHub Actions on real amd64 hardware.
# Do NOT build locally on ARM via QEMU — unstable for heavy Python builds.
#
# Publish:  create a GitHub release → CI builds & pushes automatically.
# Pull:     docker pull ghcr.io/<owner>/jno:latest
# Run:      docker run --rm ghcr.io/<owner>/jno:latest
# ------------------------------------------------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# ---- system dependencies + Python 3.12 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        git \
        build-essential \
        libglu1-mesa \
        libgl1 \
        libxrender1 \
        libxcursor1 \
        libxft2 \
        libxinerama1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1

# ---- install uv ----
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ---- copy source + metadata ----
COPY pyproject.toml README.md ./
COPY jno/ ./jno/

# ---- install package + all dependencies ----
RUN uv pip install --system ".[dev]"

# ---- default entrypoint ----
CMD ["python", "-c", "import jno; print('jNO ready')"]
