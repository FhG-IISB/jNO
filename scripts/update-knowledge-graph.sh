#!/usr/bin/env bash
# Rebuild the AST knowledge graph after commits (CPU-only, no API cost).
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v graphify >/dev/null 2>&1; then
  echo "graphify not on PATH; run: uv tool install graphifyy (see the graphify skill)" >&2
  exit 0
fi
graphify update .
