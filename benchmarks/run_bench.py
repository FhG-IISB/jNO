"""Drive the FEM benchmark matrix -- one subprocess per case, results to JSON.

    pixi run python benchmarks/run_bench.py            # everything
    pixi run python benchmarks/run_bench.py poisson2d eddy3d

One process per case is the point. Running the matrix in a single process lets later cases reuse
earlier XLA compilations, so whatever runs last looks fastest; the isolation keeps every number a
cold, comparable measurement. A case that dies (out of memory, a solver that will not converge at
that size) is recorded as a failure and does not stop the rest.

Each point runs ``REPEATS`` times and is stored as the MEDIAN with ``*_min``/``*_max`` alongside,
so the figure can show the spread rather than implying a precision the measurement does not have.
A point that fails only SOME repeats keeps ``failed_repeats`` on the record instead of being
quietly reported as though it were solid -- intermittent success is the finding.

Two conditions are called out in the summary because they invalidate a number rather than merely
widen it: a solve that did not converge (its timing measures nothing) and a point that was flaky.

NOT controlled here: thermal and clock drift. This card idles at 285 MHz against a 2100 MHz max
with persistence mode off, so every subprocess starts cold and a long matrix run heats the card as
it goes -- later cases are measured on a slower GPU. Process isolation fixes the XLA-cache bias and
introduces this one in its place. ``nvidia-smi -pm 1`` and ``nvidia-smi -lgc`` (both need root)
would remove it; the per-record ``sm_clock`` stamp at least makes the drift visible.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: Output file. Overridable so a CPU sweep does not overwrite a GPU one: the merge key is
#: (case, size_index), which carries no notion of platform, so both would collide in one file.
OUT = Path(os.environ.get("JNO_BENCH_OUT", HERE / "results.json"))
TIMEOUT_S = 900


#: repeats per point, each a FRESH PROCESS. In-process repeats would only re-time a warm solve;
#: the run-to-run variation that matters here is cold -- driver init, XLA compilation, and this
#: card's clock state (it idles at 285 MHz against a 2100 MHz max). Measured instability that
#: motivated this: the same vector case converged in one process and failed in another, and the
#: same 1.43M-DOF Poisson reported 8.01 s and 9.04 s on separate runs.
REPEATS = 3


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def repeat_case(case: str, idx: int) -> dict:
    """Run one point REPEATS times and aggregate to median with the observed spread."""
    runs = [run_case(case, idx) for _ in range(REPEATS)]
    ok = [r for r in runs if r and "failed" not in r]
    if not ok:
        return runs[0] or {"case": case, "size_index": idx, "failed": "no output"}

    agg = dict(ok[-1])
    for key in ("build_ms", "solve_ms", "peak_mb"):
        vals = [r[key] for r in ok if key in r]
        if vals:
            agg[key] = round(_median(vals), 1)
            agg[f"{key}_min"], agg[f"{key}_max"] = round(min(vals), 1), round(max(vals), 1)
    agg["repeats"] = len(ok)
    # a point that only sometimes runs is a finding, not a footnote -- keep it on the record
    agg["failed_repeats"] = REPEATS - len(ok)
    spread = 0.0
    if agg.get("solve_ms"):
        spread = 100.0 * (agg["solve_ms_max"] - agg["solve_ms_min"]) / max(agg["solve_ms"], 1e-9)
    flags = []
    if agg["failed_repeats"]:
        flags.append(f"{agg['failed_repeats']}/{REPEATS} FAILED")
    if agg.get("converged") is False:
        flags.append(f"NOT CONVERGED rel={agg.get('rel_residual')}")
    if spread > 20.0:
        flags.append(f"spread {spread:.0f}%")
    if flags:
        print(f"      ^ {'; '.join(flags)}")
    return agg


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
            rec = repeat_case(case, idx)
            if rec:
                records.append(rec)
    # MERGE with whatever is already on disk, keyed on (case, size). Overwriting would mean that
    # re-running one case to check a fix silently discards every other case's numbers -- and the file
    # still looks complete.
    merged = {}
    if OUT.exists():
        for r in json.loads(OUT.read_text()):
            merged[(r["case"], r["size_index"])] = r
    for r in records:
        merged[(r["case"], r["size_index"])] = r
    OUT.write_text(json.dumps(sorted(merged.values(), key=lambda r: (r["case"], r["size_index"])), indent=2))

    ok = sum(1 for r in records if "failed" not in r)
    bad = [r for r in records if r.get("converged") is False]
    flaky = [r for r in records if r.get("failed_repeats")]
    print(f"\n{ok}/{len(records)} points succeeded in {time.perf_counter() - t0:.0f}s -> {OUT}")
    if bad:
        names = ", ".join(f"{r['case']}[{r['size_index']}] rel={r.get('rel_residual')}" for r in bad)
        print(f"  NOT CONVERGED (timings on these measure nothing): {names}")
    if flaky:
        names = ", ".join(f"{r['case']}[{r['size_index']}] {r['failed_repeats']}/{REPEATS}" for r in flaky)
        print(f"  FLAKY (failed some repeats): {names}")


if __name__ == "__main__":
    main(sys.argv[1:])
