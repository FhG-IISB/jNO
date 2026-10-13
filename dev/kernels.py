"""L2: hardware counters per kernel with Nsight Compute, for ONE stage of a profiled run at a time, budgeted.

    python dev/kernels.py dev/runs/<label> --stage jno/assemble -- python the_script.py ARGS
    python dev/kernels.py RUN --stage jno/solve --top 5 --metrics flops,memory,occupancy -- python the_script.py

The run's trace says which kernels the stage launches and how long they take; ncu replays every selected launch once
per metric pass (`passes` below), so the cost is `launches x passes x (kernel time + ~1 ms overhead)`. The runner
refuses above `--budget` seconds unless `--force`; `--top N` keeps the N kernel kinds with the most time, `--launches`
caps the launches ncu profiles. The script runs with `JNO_PROF_NCU=1` so it can skip its own warm-up/trace (the
harness skips the trace and the transfer check then). Output: `RUN/kernels_<stage>.json` and `.csv`; the ladder
and the bind strip read the json."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time

# metric groups: (ncu metric, short name). FP64 FLOPs: fma counts 2. Bytes: DRAM and L2 (lts) traffic.
METRICS = {
    "time": [("gpu__time_duration.sum", "time_ns")],
    "flops": [
        ("sm__sass_thread_inst_executed_op_dfma_pred_on.sum", "dfma"),
        ("sm__sass_thread_inst_executed_op_dmul_pred_on.sum", "dmul"),
        ("sm__sass_thread_inst_executed_op_dadd_pred_on.sum", "dadd"),
        ("sm__sass_thread_inst_executed_op_ffma_pred_on.sum", "ffma"),
        ("sm__sass_thread_inst_executed_op_fmul_pred_on.sum", "fmul"),
        ("sm__sass_thread_inst_executed_op_fadd_pred_on.sum", "fadd"),
        ("sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active", "fp64_pipe_pct"),
    ],
    "memory": [
        ("dram__bytes_read.sum", "dram_read_B"),
        ("dram__bytes_write.sum", "dram_write_B"),
        ("lts__t_bytes.sum", "l2_B"),
        ("l1tex__t_bytes.sum", "l1_B"),
        ("lts__t_sector_hit_rate.pct", "l2_hit_pct"),
        ("l1tex__t_sector_hit_rate.pct", "l1_hit_pct"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram_pct"),
    ],
    "occupancy": [
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm_pct"),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "occupancy_pct"),
        ("launch__registers_per_thread", "regs"),
        ("launch__occupancy_limit_registers", "occ_limit_regs"),
        ("launch__waves_per_multiprocessor", "waves"),
        ("smsp__warps_issue_stalled_long_scoreboard.avg", "stall_long_scoreboard"),
        ("smsp__warps_issue_stalled_math_pipe_throttle.avg", "stall_math_pipe"),
        ("smsp__warps_issue_stalled_lg_throttle.avg", "stall_lg_throttle"),
    ],
}
PASSES = {"time": 1, "flops": 2, "memory": 3, "occupancy": 3}  # measured on the RTX 3070 (ncu 2026.2): passes per group
OVERHEAD_S = 1.5e-3  # per replayed launch, beyond the kernel's own time


def nvtx_regex(stage: str) -> str:
    """ncu's NVTX filter for every XLA range whose op path carries the stage. `/` separates the push/pop levels in
    ncu's syntax, so a `/` inside the scope path is written `.`; the trailing `/` selects push/pop ranges."""
    return "regex:TSL@Thunk.*" + re.escape(stage).replace("/", ".") + ".*/"


def estimate(run_json: str, stage: str, groups, top: int | None, launches_cap: int | None):
    with open(run_json) as fh:
        d = json.load(fh)
    ks = [k for k in d["kernels"] if k["stage"] == stage]
    if not ks:
        raise SystemExit(
            f"no kernels of stage {stage!r} in {run_json}; stages: {sorted({k['stage'] for k in d['kernels']})}"
        )
    ks.sort(key=lambda k: -k["duration_ns"] * k["count"])
    if top:
        ks = ks[:top]
    n = sum(k["count"] for k in ks)
    if launches_cap:
        n = min(n, launches_cap)
    t_kernel = sum(k["duration_ns"] * k["count"] for k in ks) / 1e9 * (n / max(1, sum(k["count"] for k in ks)))
    passes = sum(PASSES[g] for g in groups)
    return ks, n, passes, passes * (t_kernel + n * OVERHEAD_S)


UNIT = {
    "ms": 1e-3,
    "us": 1e-6,
    "ns": 1e-9,
    "s": 1.0,
    "byte": 1.0,
    "Kbyte": 1e3,
    "Mbyte": 1e6,
    "Gbyte": 1e9,
    "inst": 1.0,
    "%": 1.0,
    "register/thread": 1.0,
    "warp": 1.0,
    "": 1.0,
}
NVTX_COL = "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"


def parse_csv(text: str):
    """ncu --csv --page raw: one wide row per launch (row 2 holds the units); values are brought to base units
    (seconds, bytes, counts, percent)."""
    rows = list(csv.reader(io.StringIO(text[text.index('"ID"') :])))
    hdr, units = rows[0], rows[1]
    short = {m: s for g in METRICS.values() for m, s in g}
    col = {m: hdr.index(m) for m in short if m in hdr}
    recs = []
    for r in rows[2:]:
        if not r or not r[0]:
            continue
        nv = r[hdr.index(NVTX_COL)] if NVTX_COL in hdr else ""
        op = re.search(r"hlo_op=([\w.\-]+)#", nv.split("Thunk:#")[-1]) if "Thunk:#" in nv else None
        rec = {
            "id": int(r[0]),
            "kernel": r[hdr.index("Kernel Name")],
            "hlo_op": op.group(1) if op else "",
            "scope": (re.search(r"name=([^,#]+)", nv.split("Thunk:#")[-1]) or [None, ""])[1],
            "grid": r[hdr.index("Grid Size")],
            "block": r[hdr.index("Block Size")],
        }
        for m, i in col.items():
            v = r[i].replace(",", "")
            try:
                rec[short[m]] = float(v) * (UNIT.get(units[i], 1.0) if not short[m].endswith("pct") else 1.0)
            except ValueError:
                rec[short[m]] = v
        if "time_ns" in rec:
            rec["time_ns"] *= 1e9  # was brought to seconds
        if "dfma" in rec:
            rec["fp64_flop"] = 2 * rec.get("dfma", 0) + rec.get("dmul", 0) + rec.get("dadd", 0)
            rec["fp32_flop"] = 2 * rec.get("ffma", 0) + rec.get("fmul", 0) + rec.get("fadd", 0)
        if "dram_read_B" in rec:
            rec["dram_B"] = rec["dram_read_B"] + rec["dram_write_B"]
        if "fp64_flop" in rec and "dram_B" in rec:
            rec["intensity"] = rec["fp64_flop"] / max(rec["dram_B"], 1.0)
        for key in ("fp64_flop", "dram_B"):
            if key in rec and rec.get("time_ns"):
                rec[key + "_per_s"] = rec[key] / (rec["time_ns"] * 1e-9)
        recs.append(rec)
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="dev/runs/<label> of a profiled run (its run.json picks the kernels)")
    ap.add_argument("--stage", required=True, help="jno/<stage> or 'other'")
    ap.add_argument("--metrics", default="time,flops,memory,occupancy", help="groups: " + ",".join(METRICS))
    ap.add_argument("--top", type=int, default=None, help="keep the N kernel kinds with the most time")
    ap.add_argument("--launches", type=int, default=None, help="cap on launches ncu profiles (-c)")
    ap.add_argument("--budget", type=float, default=120.0, help="refuse an estimated replay above this many seconds")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry", action="store_true", help="only print the estimate and the command")
    ap.add_argument("--reparse", action="store_true", help="parse the saved kernels_<stage>.csv instead of running")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        ap.error("give the program to run after --")
    cmd = argv[argv.index("--") + 1 :]
    a = ap.parse_args(argv[: argv.index("--")])
    groups = a.metrics.split(",")
    tag = a.stage.replace("/", "_")
    rep = os.path.join(a.run, f"kernels_{tag}")
    if a.reparse:
        with open(rep + ".csv") as fh:
            recs = parse_csv(fh.read())
        _write(rep, a.stage, groups, recs, float("nan"))
        return
    ks, n, passes, est = estimate(os.path.join(a.run, "run.json"), a.stage, groups, a.top, a.launches)
    print(f"{a.stage}: {len(ks)} kernel kinds, {n} launches, {passes} passes -> estimated replay {est:.0f} s", flush=True)
    if est > a.budget and not a.force:
        raise SystemExit(f"above the budget of {a.budget:.0f} s: use --top, --launches, fewer --metrics, or --force")
    metrics = ",".join(m for g in groups for m, _ in METRICS[g])
    # no report file (-o): with --nvtx a report grows by tens of MB per launch (XLA registers thousands of NVTX
    # strings); the CSV on stdout is all the harness needs and is kept as kernels_<stage>.csv
    # XLA's own kernels (CUDA graphs off: one uniquely named kernel per HLO op) are selected by NAME, with no NVTX
    # at all: with --nvtx ncu's working report grows by tens of MB per launch (XLA registers thousands of NVTX
    # strings) and lands in TMPDIR. Library kernels under an FFI call (cuDSS: the same kernel names in factor and
    # solve) need the NVTX range of their stage.
    names = sorted({k["name"] for k in ks})
    xla_named = all("::" not in n and "<" not in n for n in names)
    if xla_named:
        ncu = [
            "ncu",
            "-k",
            "regex:^(" + "|".join(re.escape(n) for n in names) + ")$",
            "--metrics",
            metrics,
            "--csv",
            "--page",
            "raw",
        ]
    else:
        ncu = ["ncu", "--nvtx", "--nvtx-include", nvtx_regex(a.stage), "--metrics", metrics, "--csv", "--page", "raw"]
    if a.launches:
        ncu += ["-c", str(a.launches)]
    full = ncu + cmd
    print(" ".join(full), flush=True)
    if a.dry:
        return
    # no CUDA graphs under ncu: a kernel launched from a graph has no per-thunk NVTX range (only `command_buffer_N`),
    # so the stage filter cannot see it; no autotuning either (ncu makes XLA's candidate runs slow)
    xla_flags = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"
    tmp = os.path.join(
        os.environ.get("JNO_PROF_SCRATCH", "/var/tmp/jobs"), "ncu-tmp"
    )  # ncu's working report: on disk, never /tmp (RAM here)
    os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, JNO_PROF_NCU="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_FLAGS=xla_flags.strip(), TMPDIR=tmp)
    t = time.perf_counter()
    r = subprocess.run(full, env=env, capture_output=True, text=True)
    dt = time.perf_counter() - t
    import shutil

    shutil.rmtree(tmp, ignore_errors=True)
    sys.stderr.write(
        "\n".join(line for line in r.stderr.splitlines() if "==ERROR==" in line or "==WARNING==" in line) + "\n"
    )
    if r.returncode != 0 and '"ID"' not in r.stdout:
        raise SystemExit(f"ncu failed ({r.returncode}) after {dt:.0f} s:\n{r.stderr[-2000:]}")
    with open(rep + ".csv", "w") as fh:
        fh.write(r.stdout[r.stdout.index('"ID"') :])
    recs = parse_csv(r.stdout)
    _write(rep, a.stage, groups, recs, dt)


def _write(rep, stage, groups, recs, dt):
    with open(rep + ".json", "w") as fh:
        json.dump({"stage": stage, "groups": groups, "launches": len(recs), "wall_s": dt, "kernels": recs}, fh, indent=1)
    print(f"{len(recs)} launches profiled in {dt:.0f} s -> {rep}.json", flush=True)
    tot_t = sum(r.get("time_ns", 0) for r in recs) / 1e6
    tot_f = sum(r.get("fp64_flop", 0) for r in recs) / 1e9
    tot_b = sum(r.get("dram_B", 0) for r in recs) / 1e6
    print(
        f"  stage total: {tot_t:.1f} ms, fp64 {tot_f:.3f} GFLOP, dram {tot_b:.1f} MB "
        f"({tot_f * 1e9 / max(tot_b * 1e6, 1):.2f} flop/byte)",
        flush=True,
    )
    by = {}
    for r in recs:
        b = by.setdefault(
            r["kernel"].split("(")[0][:48],
            {"n": 0, "t": 0.0, "f": 0.0, "B": 0.0, "sm": [], "dram": [], "l2": [], "occ": []},
        )
        b["n"] += 1
        b["t"] += r.get("time_ns", 0)
        b["f"] += r.get("fp64_flop", 0)
        b["B"] += r.get("dram_B", 0)
        for key, src in (("sm", "sm_pct"), ("dram", "dram_pct"), ("l2", "l2_hit_pct"), ("occ", "occupancy_pct")):
            if src in r:
                b[key].append(r[src])
    avg = lambda v: sum(v) / len(v) if v else float("nan")
    print(
        f"  {'kernel':48s} {'n':>4s} {'ms':>7s} {'GFLOP':>7s} {'MB':>7s} {'sm%':>4s} {'dram%':>5s} {'l2hit':>5s} {'occ%':>4s}"
    )
    for k, b in sorted(by.items(), key=lambda kv: -kv[1]["t"])[:12]:
        print(
            f"  {k:48s} {b['n']:4d} {b['t'] / 1e6:7.2f} {b['f'] / 1e9:7.3f} {b['B'] / 1e6:7.1f} {avg(b['sm']):4.0f} {avg(b['dram']):5.0f} "
            f"{avg(b['l2']):5.0f} {avg(b['occ']):4.0f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
