"""Drive the FEM benchmark matrix -- one subprocess per case, results to JSON.

    pixi run python benchmarks/run_bench.py            # everything
    pixi run python benchmarks/run_bench.py poisson2d eddy3d

One process per case is the point. Running the matrix in a single process lets later cases reuse
earlier XLA compilations, so whatever runs last looks fastest; the isolation keeps every number a
cold, comparable measurement. A case that dies (out of memory, a solver that will not converge at
that size) is recorded as a failure and does not stop the rest.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results.json"
TIMEOUT_S = 900


def run_case(case: str, idx: int) -> dict | None:
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    env.setdefault("JAX_PLATFORMS", "cuda,cpu")
    try:
        proc = subprocess.run(
            [sys.executable, str(HERE / "fem_bench.py"), case, str(idx)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"  {case}[{idx}]  TIMEOUT after {TIMEOUT_S}s")
        return {"case": case, "size_index": idx, "failed": "timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            rec = json.loads(line[len("RESULT ") :])
            rec["size_index"] = idx
            print(
                f"  {rec['case']:<11} ms={rec['mesh_size']:<6} dofs={rec['dofs']:>7}"
                f"  build {rec['build_ms']:>8.0f}  solve {rec['solve_ms']:>8.0f}  peak {rec['peak_mb']:>7.0f} MB"
            )
            return rec
    err = (proc.stderr or "").strip().splitlines()
    print(f"  {case}[{idx}]  FAILED: {err[-1][:100] if err else 'no output'}")
    return {"case": case, "size_index": idx, "failed": err[-1][:200] if err else "no output"}


def main(argv: list[str]) -> None:
    sys.path.insert(0, str(HERE))
    from fem_bench import CASES

    wanted = argv or list(CASES)
    t0 = time.perf_counter()
    records = []
    for case in wanted:
        if case not in CASES:
            print(f"unknown case {case!r}; known: {', '.join(CASES)}")
            continue
        print(f"{case}:")
        for idx in range(len(CASES[case][1])):
            rec = run_case(case, idx)
            if rec:
                records.append(rec)
    OUT.write_text(json.dumps(records, indent=2))
    ok = sum(1 for r in records if "failed" not in r)
    print(f"\n{ok}/{len(records)} cases succeeded in {time.perf_counter() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
