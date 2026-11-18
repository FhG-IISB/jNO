#!/usr/bin/env bash
# Keep a compile cache directory under a size limit: delete its oldest files until it fits.
#
#     bash .github/ci_prune_cache.sh DIR MAX_MB
#
# Run once before the tests (.github/ci_test_per_file.sh), so the bound costs one directory scan per
# CI job instead of one per cache write. A directory that does not exist is left alone.
set -euo pipefail

dir="$1"
max_bytes=$(($2 * 1024 * 1024))
[ -d "$dir" ] || exit 0

# modification time, size, path -- oldest first; print paths until what is left fits
find "$dir" -type f -printf '%T@\t%s\t%p\n' | sort -n |
  awk -F'\t' -v max="$max_bytes" '
    { path[NR] = $3; size[NR] = $2; total += $2 }
    END { for (i = 1; i <= NR && total > max; i++) { print path[i]; total -= size[i] } }' |
  xargs -r -d '\n' rm -f
