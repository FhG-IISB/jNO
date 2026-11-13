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
# tutorials).
#
# CI_COVERAGE=1 (the nightly) appends coverage across files into one `.coverage`
# and writes `coverage.xml` at the end for the codecov upload. PR runs leave it
# off: nothing uploads their report, and coverage tracing made a test file ~15%
# slower (tests/test_fem_rosenbrock.py, 67 s -> 78 s, cache off).
#
# Every file's wall time goes to `ci-file-times.tsv` (path<TAB>seconds), which
# the workflows upload so `.github/ci_plan.py update` can rebalance the shards.
# `--durations` lists each file's slowest tests in the log.
set -uo pipefail

# The compile cache lives under ~/.cache/jno, which the workflows save and restore
# between runs: jNO's own default (`JNO_COMPILE_CACHE_DIR`), and the same directory
# for JAX's flag in case a test sets the cache up without jNO. The size cap keeps a
# cache that is restored and refilled run after run from growing without bound
# (least recently used entries go first).
export JNO_COMPILE_CACHE_DIR="${JNO_COMPILE_CACHE_DIR:-$HOME/.cache/jno/xla}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$JNO_COMPILE_CACHE_DIR}"
export JAX_COMPILATION_CACHE_MAX_SIZE="${JAX_COMPILATION_CACHE_MAX_SIZE:-1000000000}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
# MPI (petsc4py, for jno.precond.hypre) runs single-process here, so UCX needs shared memory only.
# Left to probe the network, MPI_Init fails on runners with an Azure MANA device
# (`uct_iface_open(ud_verbs/mana_0:1) failed`), and under output capture the test file hangs.
export UCX_TLS="${UCX_TLS:-self,sm}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

rm -f .coverage coverage.xml ci-file-times.tsv

cov=()
if [ "${CI_COVERAGE:-0}" = "1" ]; then cov=(--cov=jno --cov-append --cov-report=); fi

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
  start=$SECONDS
  pytest "$f" -m 'not slow' "${cov[@]}" --durations=15 --tb=short -q
  code=$?
  # Killed by a signal (exit > 128) before pytest could report anything: a native library crashed
  # (PETSc, for one, turns a crash into an MPI abort, i.e. SIGTERM, and the captured message is lost).
  # Run the file once more without output capture, so a repeat shows the native error; a pass is a
  # flake and is flagged as a warning, not hidden.
  if [ "$code" -gt 128 ]; then
    echo "::warning::${f} was killed by signal $((code - 128)); running it once more without capture"
    pytest "$f" -m 'not slow' "${cov[@]}" --tb=short -q -s
    code=$?
  fi
  printf '%s\t%s\n' "$f" "$((SECONDS - start))" >> ci-file-times.tsv
  echo "::endgroup::"
  # exit 5 == "no tests collected" (every test in the file is slow-marked/skipped): not a failure.
  if [ "$code" -ne 0 ] && [ "$code" -ne 5 ]; then
    failed+=("$f")
    echo "::error::test file failed: ${f} (exit ${code})"
  fi
done

if [ "${#cov[@]}" -ne 0 ]; then
  coverage xml -o coverage.xml 2>/dev/null || echo "warning: could not write coverage.xml"
fi

if [ "${#failed[@]}" -ne 0 ]; then
  echo "==================================================================="
  echo "FAILED test files (${#failed[@]}):"
  printf '  %s\n' "${failed[@]}"
  exit 1
fi
echo "All ${#files[@]} test files passed."
