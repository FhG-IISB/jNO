#!/usr/bin/env bash
# CI test runner: one pytest process per test FILE, so memory (JAX device
# buffers, accumulated coverage state) is fully released between files.
#
# The all-in-one `pytest --cov` run accumulates memory across the growing FEM
# suite and gets OOM-killed on the GitHub runner (exit 137) ~35% of the way in,
# before finishing — this has failed CI on `main` and every FEM PR. Running each
# file in its own process bounds peak memory to a single file's footprint. A
# shared JAX compilation cache keeps the fresh per-file processes from
# recompiling the same kernels, so the wall-time cost stays modest.
#
# `-m 'not slow'` still skips slow-marked tests (heavy training-loop recoveries,
# tutorials). Coverage is appended across files into one `.coverage` and written
# to `coverage.xml` at the end for the codecov upload.
set -uo pipefail

export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax_cache}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

rm -f .coverage coverage.xml

# Optional explicit file list (for local testing); default is the whole suite.
if [ "$#" -gt 0 ]; then
  files=("$@")
else
  mapfile -t files < <(find tests -name 'test_*.py' | sort)
fi
echo "Running ${#files[@]} test files, one process each (memory released between files)."

failed=()
for f in "${files[@]}"; do
  echo "::group::${f}"
  pytest "$f" -m 'not slow' --cov=jno --cov-append --cov-report= --tb=short -q
  code=$?
  echo "::endgroup::"
  # exit 5 == "no tests collected" (every test in the file is slow-marked/skipped): not a failure.
  if [ "$code" -ne 0 ] && [ "$code" -ne 5 ]; then
    failed+=("$f")
    echo "::error::test file failed: ${f} (exit ${code})"
  fi
done

coverage xml -o coverage.xml 2>/dev/null || echo "warning: could not write coverage.xml"

if [ "${#failed[@]}" -ne 0 ]; then
  echo "==================================================================="
  echo "FAILED test files (${#failed[@]}):"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "All ${#files[@]} test files passed."
