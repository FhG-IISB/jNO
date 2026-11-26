#!/usr/bin/env python3
"""Split the test files into CI groups and balanced shards.

One place decides which test file belongs to which group and which shard of that group, so the PR
workflow (ci.yml) and the nightly workflow (nightly.yml) cut the suite the same way -- which also
makes their compile-cache keys line up, so a PR can start from the cache the nightly left behind.

    python3 .github/ci_plan.py matrix fem core      ->  {"include": [{"group": "fem", "shard": 1, "of": 9}, ...]}
    python3 .github/ci_plan.py files fem 3 9        ->  the test files of shard 3 of 9 of `fem`, one per line
    python3 .github/ci_plan.py update times.tsv...  ->  merge measured per-file times into ci_file_times.tsv
    python3 .github/ci_plan.py slow-matrix          ->  the nightly slow pass: {"include": [{"shard": 1, "of": 3}, ...]}
    python3 .github/ci_plan.py slow-files 2 3       ->  the files of shard 2 of 3 of the slow pass

A group gets as many shards as its measured run time needs to stay under TARGET_MINUTES per shard
(more shards cost nothing but a runner start on a public repository). Files are dealt out
longest-first, each to the shard with the least time so far (LPT scheduling). A file with no
measured time yet -- a new one -- counts as the median file, so it is covered by default and the
split stays roughly even until its first measurement is merged in with `update`.
"""

from __future__ import annotations

import fnmatch
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMES = ROOT / ".github" / "ci_file_times.tsv"
TARGET_MINUTES = 30
SLOW_TIMES = ROOT / ".github" / "ci_slow_times.tsv"  # per-file seconds of the `-m slow` tests (see slow_files)
SLOW_DEFAULT_SECONDS = 300.0  # a slow file with no measured time yet
SLOW_EXCLUDED = {"tests/test_tutorial_examples.py"}  # slow-marked, run by the manual tutorials workflow
MAX_SHARDS = 12

# Group -> test-file stems. `core` is the COMPLEMENT of the named groups, computed rather than listed,
# so a new test file is covered by default instead of silently belonging to nothing. The stems do NOT
# require a trailing underscore: `test_fdm_*.py` would miss the group's own headline `test_fdm.py`.
GROUPS = {
    "fem": ["test_fem*.py"],
    "geometry": ["test_domain*.py", "test_shape*.py", "test_polygon*.py", "test_mesh*.py"],
    "rcwa": ["test_rcwa*.py"],
    "fdm": ["test_fdm*.py"],
}
ALL_GROUPS = [*GROUPS, "core"]


def test_files() -> list[str]:
    return sorted(f"tests/{p.name}" for p in (ROOT / "tests").glob("test_*.py"))


def group_of(path: str) -> str:
    name = Path(path).name
    for group, stems in GROUPS.items():
        if any(fnmatch.fnmatch(name, s) for s in stems):
            return group
    return "core"


def slow_files() -> list[str]:
    """The test files that hold a slow-marked test (found in the source, no import), minus SLOW_EXCLUDED."""
    return [f for f in test_files() if f not in SLOW_EXCLUDED and "mark.slow" in (ROOT / f).read_text()]


def slow_timed() -> list[tuple[float, str]]:
    times = read_times(SLOW_TIMES)
    return [(times.get(f, SLOW_DEFAULT_SECONDS), f) for f in slow_files()]


def slow_n_shards() -> int:
    total = sum(t for t, _ in slow_timed())
    return max(1, min(MAX_SHARDS, math.ceil(total / (60 * TARGET_MINUTES))))


def slow_shard_files(shard: int, of: int) -> list[str]:
    bins: list[tuple[float, list[str]]] = [(0.0, []) for _ in range(of)]
    for t, f in sorted(slow_timed(), key=lambda x: (-x[0], x[1])):
        i = min(range(of), key=lambda k: (bins[k][0], k))
        bins[i] = (bins[i][0] + t, [*bins[i][1], f])
    return sorted(bins[shard - 1][1])


def read_times(path: Path = TIMES) -> dict[str, float]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f, s = line.split("\t")
            out[f] = float(s)
    return out


def timed_files(group: str) -> list[tuple[float, str]]:
    times = read_times()
    default = statistics.median(times.values()) if times else 30.0
    return [(times.get(f, default), f) for f in test_files() if group_of(f) == group]


def n_shards(group: str) -> int:
    total = sum(t for t, _ in timed_files(group))
    return max(1, min(MAX_SHARDS, math.ceil(total / (60 * TARGET_MINUTES))))


def shard_files(group: str, shard: int, of: int) -> list[str]:
    bins: list[tuple[float, list[str]]] = [(0.0, []) for _ in range(of)]
    for t, f in sorted(timed_files(group), key=lambda x: (-x[0], x[1])):
        i = min(range(of), key=lambda k: (bins[k][0], k))
        bins[i] = (bins[i][0] + t, [*bins[i][1], f])
    return sorted(bins[shard - 1][1])


def main(argv: list[str]) -> None:
    cmd, *rest = argv
    if cmd == "matrix":
        groups = rest or ALL_GROUPS
        unknown = set(groups) - set(ALL_GROUPS)
        if unknown:
            sys.exit(f"unknown group(s): {sorted(unknown)}")
        include = [{"group": g, "shard": k, "of": n} for g in groups for n in [n_shards(g)] for k in range(1, n + 1)]
        print(json.dumps({"include": include}))
    elif cmd == "files":
        group, shard, of = rest[0], int(rest[1]), int(rest[2])
        print("\n".join(shard_files(group, shard, of)))
    elif cmd == "slow-matrix":
        n = slow_n_shards()
        print(json.dumps({"include": [{"shard": k, "of": n} for k in range(1, n + 1)]}))
    elif cmd == "slow-files":
        print("\n".join(slow_shard_files(int(rest[0]), int(rest[1]))))
    elif cmd == "update":
        times = read_times()
        for tsv in rest:
            for line in Path(tsv).read_text().splitlines():
                if line.strip():
                    f, s = line.split("\t")
                    times[f] = float(s)
        live = set(test_files())
        TIMES.write_text("".join(f"{f}\t{times[f]:.0f}\n" for f in sorted(times) if f in live))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
