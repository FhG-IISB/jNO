"""CPU hardware counters per stage inside one jitted XLA:CPU program, sampled with `perf`.

    JAX_PLATFORMS=cpu python dev/examples/small3d.py --cpu --label small3d-cpu      # the run (trace + HLO)
    JAX_PLATFORMS=cpu python dev/cpu.py dev/runs/small3d-cpu -- python dev/examples/small3d.py --cpu --label small3d-cpu

How: XLA:CPU compiles every HLO op into its own function and registers each as an in-memory ELF object with
LLVM's gdb JIT interface (`__jit_debug_descriptor` in libjax_common.so). `perf_map()` walks that list in the
running process and writes `/tmp/perf-<pid>.map`, so `perf` resolves the kernels by HLO op name; the compiled
HLO's metadata then maps the op to its `jno/<stage>` scope. The script is re-run with `JNO_PROF_PERF=<events>`;
the harness (`prof.run`) writes the map after compiling and runs the steady-state calls under `perf record -p`.
Library kernels the thunks call (MKL dgemm, Eigen gebp, oneDNN/ynn) are symbolised but carry no op name: they
are reported as "library" (unattributed to a stage).

Events (P-cores only for the FP ones: pin with `taskset -c 0-11`): cycles, instructions, cache-misses, LLC misses,
FP64 scalar / 128-bit / 256-bit packed. Sampled, so counts are statistical; the per-stage shares are the point."""

from __future__ import annotations

import argparse
import collections
import ctypes
import glob
import json
import os
import re
import struct
import subprocess
import sys
import time

EVENTS = {
    "cycles": "cycles",
    "instructions": "instructions",
    "cache_misses": "cache-misses",
    "llc_misses": "LLC-load-misses",
    "fp64_scalar": "fp_arith_inst_retired.scalar_double",
    "fp64_128b": "fp_arith_inst_retired.128b_packed_double",
    "fp64_256b": "fp_arith_inst_retired.256b_packed_double",
}
FLOP_WEIGHT = {"fp64_scalar": 1, "fp64_128b": 2, "fp64_256b": 4}


def perf_map(path: str | None = None) -> dict[str, tuple[int, int]]:
    """Write the perf map of XLA:CPU's compiled kernels for this process; returns {function: (addr, size)}."""
    import jaxlib

    so = glob.glob(os.path.join(os.path.dirname(jaxlib.__file__), "**", "libjax_common.so"), recursive=True)[0]
    sym = subprocess.run(["readelf", "-sW", so], capture_output=True, text=True).stdout
    off = int(next(line for line in sym.splitlines() if line.endswith(" __jit_debug_descriptor")).split()[1], 16)
    with open("/proc/self/maps") as fh:
        base = min(int(line.split("-")[0], 16) for line in fh if so in line)
    _version, _action, _relevant, entry = struct.unpack("IIQQ", ctypes.string_at(base + off, 24))
    out = {}
    while entry:
        nxt, _prev, addr, size = struct.unpack("QQQQ", ctypes.string_at(entry, 32))
        buf = ctypes.string_at(addr, size)
        shoff = struct.unpack_from("Q", buf, 0x28)[0]
        shentsize, shnum = struct.unpack_from("HH", buf, 0x3A)
        sh = [struct.unpack_from("IIQQQQIIQQ", buf, shoff + i * shentsize) for i in range(shnum)]
        for s in sh:
            if s[1] != 2:  # SHT_SYMTAB
                continue
            strtab = sh[s[6]]
            for j in range(s[5] // 24):
                name, info, _other, shndx, value, sz = struct.unpack_from("IBBHQQ", buf, s[4] + j * 24)
                if info & 0xF != 2 or sz == 0 or shndx >= shnum:  # STT_FUNC with a size
                    continue
                nm = buf[strtab[4] + name : buf.index(b"\0", strtab[4] + name)].decode()
                out[nm] = (sh[shndx][3] + value, sz)  # section address is already the final one
        entry = nxt
    path = path or f"/tmp/perf-{os.getpid()}.map"
    with open(path, "w") as fh:
        for nm, (addr, sz) in out.items():
            fh.write(f"{addr:x} {sz:x} {nm}\n")
    return out


class Recorder:
    """`perf record -p <this pid>` around a block of calls; `stop()` returns the perf.data path."""

    def __init__(self, events, out: str, freq: int = 4000):
        self.events, self.out, self.freq = events, out, freq
        self.p = None

    def start(self):
        ev = ",".join(EVENTS[e] for e in self.events)
        self.p = subprocess.Popen(
            ["perf", "record", "-q", "-F", str(self.freq), "-e", ev, "-p", str(os.getpid()), "-o", self.out],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.3)  # perf attaches asynchronously
        return self

    def stop(self):
        self.p.send_signal(2)
        self.p.wait(timeout=60)
        return self.out


_LINE = re.compile(r"^\s*(\d+)\s+(\S+?):\s+([0-9a-fA-F]+)\s+(.*)$")


def samples(perf_data: str):
    """{(event, symbol): period sum} from `perf script` (lines: `period event: ip symbol`; on a hybrid CPU the
    event reads `cpu_core/<name>/`)."""
    r = subprocess.run(
        ["perf", "script", "-i", perf_data, "-F", "event,period,ip,sym", "--no-demangle"], capture_output=True, text=True
    )
    agg = collections.Counter()
    for line in r.stdout.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        period, ev, _ip, sym = m.groups()
        ev = ev.split("/")[1] if "/" in ev else ev
        agg[(ev, sym.strip())] += int(period)
    return agg


def attribute(agg, hlo_meta: dict[str, str], events):
    """Per stage and event: the sampled count; XLA kernels through the HLO join, everything else as 'library'."""
    from dev.prof import _stage_of

    inv = {EVENTS[e]: e for e in events}
    inv.update({EVENTS[e].replace("-", "_"): e for e in events})
    by = collections.defaultdict(lambda: collections.Counter())
    syms = collections.defaultdict(lambda: collections.Counter())
    for (ev, sym), n in agg.items():
        e = inv.get(ev.split("/")[0], ev)
        base = re.sub(r"[+@].*$", "", sym)
        if base in hlo_meta:
            stage, _ = _stage_of(hlo_meta[base])
        elif base.startswith("[unknown]") or not base:
            stage = "unknown"
        else:
            stage = "library"
        by[stage][e] += n
        syms[stage][(base, e)] += n
    return by, syms


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--events", default="cycles,instructions,cache_misses,fp64_scalar,fp64_128b,fp64_256b")
    ap.add_argument("--cores", default="0-11", help="taskset list (P-cores on fleef; FP counters exist only there)")
    ap.add_argument("--reps", type=int, default=20, help="steady-state calls recorded")
    ap.add_argument("--reparse", action="store_true", help="re-read the run's perf.data instead of recording")
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv and "--reparse" not in argv:
        ap.error("give the program to run after --")
    cmd = argv[argv.index("--") + 1 :] if "--" in argv else []
    a = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
    env = dict(os.environ, JNO_PROF_PERF=a.events, JNO_PROF_PERF_REPS=str(a.reps), JAX_PLATFORMS="cpu")
    env["JNO_PROF_PERF_OUT"] = os.path.join(a.run, "perf.data")
    dt = float("nan")
    if not a.reparse:
        t = time.perf_counter()
        r = subprocess.run(["taskset", "-c", a.cores] + cmd, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"the program failed:\n{r.stderr[-3000:]}")
        dt = time.perf_counter() - t
    with open(os.path.join(a.run, "compiled.hlo")) as fh:
        from dev.prof import _hlo_meta

        meta = _hlo_meta(fh.read())
    events = a.events.split(",")
    by, syms = attribute(samples(env["JNO_PROF_PERF_OUT"]), meta, events)
    tot = collections.Counter()
    for st in by.values():
        tot.update(st)
    out = {
        "events": events,
        "reps": a.reps,
        "wall_s": dt,
        "stages": {s: dict(c) for s, c in by.items()},
        "total": dict(tot),
    }
    for s, c in by.items():
        f = sum(FLOP_WEIGHT[e] * c.get(e, 0) for e in FLOP_WEIGHT if e in c)
        out["stages"][s]["fp64_flop_sampled"] = f
        out["stages"][s]["top"] = [{"symbol": k[0], "event": k[1], "count": n} for k, n in syms[s].most_common(6)]
    with open(os.path.join(a.run, "cpu.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    hdr = f"  {'stage':16s}" + "".join(f"{e:>14s}" for e in events) + f"{'vector share':>14s}"
    print(f"{a.run}: {a.reps} calls under perf ({dt:.0f} s), shares of the sampled counts per stage")
    print(hdr)
    for s, c in sorted(by.items(), key=lambda kv: -kv[1].get("cycles", 0)):
        fp = {e: c.get(e, 0) for e in FLOP_WEIGHT}
        vec = (2 * fp["fp64_128b"] + 4 * fp["fp64_256b"]) / max(1, sum(FLOP_WEIGHT[e] * fp[e] for e in fp))
        print(
            f"  {s:16s}" + "".join(f"{100 * c.get(e, 0) / max(1, tot[e]):13.1f}%" for e in events) + f"{100 * vec:13.0f}%"
        )
    lib = syms.get("library", collections.Counter())
    if lib:
        print(
            "  library symbols (cycles): "
            + ", ".join(f"{k[0][:28]} {n:,}" for k, n in lib.most_common(12) if k[1] == "cycles")
        )
    print("->", os.path.join(a.run, "cpu.json"))


if __name__ == "__main__":
    main()
