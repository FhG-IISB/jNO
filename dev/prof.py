"""Developer profiling harness for jNO (not shipped; `dev/` is outside the `jno` package).

    from dev import prof
    r = prof.run(jax.jit(jax.value_and_grad(f)), x, label="planar44k-grad", reps=3)
    print(r.table()); r.save()            # dev/runs/<label>/run.json + figures

One run collects, per *stage* of the compiled program (a `jax.named_scope("jno/<stage>")` in jNO or in the
user's function; everything outside a scope is "other"):

  L0 (always)   compile vs steady-state wall time, XLA's static flops / bytes / temp memory, XLA pool peak,
                whole-card device memory delta, host peak RSS, an implicit-transfer check (`jax.transfer_guard`),
                the solver backends' own statistics (jno.utils.profiling.stats, when the backends report)
  L1 (trace)    one steady-state call inside `jax.profiler.trace`: every device kernel with its scope path,
                duration, occupancy, registers, plus memcpys; aggregated into a stage tree (forward / adjoint)
  L2 (kernels)  see `prof.kernels(...)`: ncu on one stage's kernels only, budgeted (not in this file yet)

The scope path of a kernel comes from the compiled HLO's metadata (`op_name`), joined on the kernel's
`hlo_op`; the trace's own `name` stat is the fallback (it is sometimes garbled)."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import resource
import subprocess
import sys
import time
from collections import defaultdict

import jax
import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
SCOPE_RE = re.compile(r"jno/([A-Za-z0-9_.]+)")
_META_RE = re.compile(r"^\s*(?:ROOT )?%?([\w.\-]+) = .*?metadata=\{[^}]*?op_name=\"([^\"]+)\"", re.M)


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _card_used_gb():
    """Whole-card device memory in use (nvidia-smi view): XLA's pool, cuDSS, the CUDA context, other processes."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.split()[0]
        return float(out) / 1024
    except Exception:  # noqa: BLE001
        return float("nan")


def _block(x):
    return jax.block_until_ready(x)


def _wall(fn, *args):
    t = time.perf_counter()
    out = _block(fn(*args))
    return time.perf_counter() - t, out


@dataclasses.dataclass
class Kernel:
    hlo_op: str
    name: str  # kernel symbol as the driver reports it
    scope: str  # full op_name path, e.g. jit(f)/transpose(jvp(jno/solve))/.../triangular_solve
    stage: str  # jno/<stage> or "other"
    adjoint: bool
    duration_ns: float
    count: int = 1
    occupancy_pct: float = float("nan")
    regs: int = -1
    grid: str = ""
    block: str = ""


@dataclasses.dataclass
class Memcpy:
    kind: str  # H2D, D2H, D2D
    bytes: int
    duration_ns: float
    count: int = 1
    stage: str = "other"
    scope: str = ""
    src: str = ""  # pageable, pinned, device
    dst: str = ""


@dataclasses.dataclass
class Result:
    label: str
    meta: dict
    l0: dict
    kernels: list[Kernel]
    memcpys: list[Memcpy]
    backend_stats: list[dict]
    trace_dir: str | None = None

    # ── bounds ────────────────────────────────────────────────────────────────────────────────────────
    def intrinsic_bounds(self, peak=None):
        """{stage: ms} the minimal work of each stage at the card's peak, from `meta["sizes"]` and the backend records.
        assemble: read the mesh and write the triplets once, the adjoint reads them and writes a gradient (x2);
        factor: sum_j nnz(L_j)^2 flops approximated as nnz(L)^2 / n (a lower bound for a balanced profile), at
        FP64 peak; solve: forward + backward sweep over nnz(L) per right-hand side, twice for the adjoint, at DRAM
        peak. Stages without a formula get none (the ladder leaves the rung out)."""
        from dev.figures import PEAK

        pk = peak or PEAK
        sz = self.meta.get("sizes") or {}
        if not sz:
            return {}
        w = 16 if sz.get("complex") else 8
        out = {}
        nnz, n, cells, nodes = sz.get("nnz"), sz.get("n"), sz.get("cells"), sz.get("nodes")
        nrhs = sz.get("nrhs", 1)
        if cells and nnz:
            mesh_B = cells * 4 * 4 + (nodes or cells // 5) * 3 * 8  # connectivity (int32) + coordinates
            trip_B = nnz * (w + 8)  # value + packed index
            out["jno/assemble"] = 2 * (mesh_B + trip_B) / pk["dram_B_s"] * 1e3
        lu = next((b.get("lu_nnz") for b in self.backend_stats if b.get("lu_nnz")), None)
        if lu and n:
            out["jno/factor"] = (lu**2 / n) / pk["fp64_flop_s"] * (4 if sz.get("complex") else 1) * 1e3
            out["jno/solve"] = 2 * 2 * nrhs * lu * w / pk["dram_B_s"] * 1e3
        return out

    def verdicts(self, kernel_bounds=None, launch_s: float = 4e-6):
        """One line per stage from the rungs: where the time can go and what would recover it.
        `kernel_bounds`: {stage: (kernel_bound_ms, _, n_profiled_launches)} as `figures.bounds_from_kernels` gives."""
        ib = self.intrinsic_bounds()
        out = {}
        total = sum(a["forward_ms"] + a["adjoint_ms"] for a in self.by_stage().values())
        for s, a in self.by_stage().items():
            m = a["forward_ms"] + a["adjoint_ms"]
            floor = a["launches"] * launch_s * 1e3
            if total > 0 and m < 0.02 * total:
                out[s] = f"negligible ({100 * m / total:.1f} % of the device time)"
                continue
            kb = None
            if kernel_bounds and s in kernel_bounds and kernel_bounds[s][0]:
                kb = kernel_bounds[s][0]
                if len(kernel_bounds[s]) > 2 and kernel_bounds[s][2]:
                    kb *= a["launches"] / kernel_bounds[s][2]
            i = ib.get(s)
            if m <= 0:
                out[s] = "-"
            elif floor > 0.5 * m:
                out[s] = f"launch-bound ({a['launches']:,} launches): fuse or batch, not a kernel"
            elif kb and m / kb < 1.5 and (not i or kb / i < 3):
                out[s] = "at the roof: leave it"
            elif kb and m / kb < 1.5 and i:
                out[s] = f"kernels at the roof, x{kb / i:.0f} above the minimal work: reformulate (fewer passes, less data)"
            elif kb and m / kb >= 1.5:
                out[s] = f"x{m / kb:.1f} above the roof at its own intensity: a better kernel" + (
                    f" (and x{kb / i:.0f} of algorithm above that)" if i and kb / i >= 3 else ""
                )
            elif i:
                out[s] = f"x{m / i:.0f} above the minimal work; run dev/kernels.py on it to split kernel from algorithm"
            else:
                out[s] = "no bound known (pass sizes=, run dev/kernels.py)"
        return out

    # ── aggregation ────────────────────────────────────────────────────────────────────────────────────
    def by_stage(self):
        """{stage: {"forward_ms", "adjoint_ms", "launches", "kernels": [Kernel...]}} sorted by total time."""
        agg = defaultdict(lambda: {"forward_ms": 0.0, "adjoint_ms": 0.0, "launches": 0, "kernels": []})
        for k in self.kernels:
            a = agg[k.stage]
            a["adjoint_ms" if k.adjoint else "forward_ms"] += k.duration_ns * k.count / 1e6
            a["launches"] += k.count
            a["kernels"].append(k)
        return dict(sorted(agg.items(), key=lambda kv: -(kv[1]["forward_ms"] + kv[1]["adjoint_ms"])))

    def tree(self):
        """Nested {name: {"ms": float, "n": launches, "children": {...}}} along the scope path, for the icicle."""
        root = {"ms": 0.0, "n": 0, "children": {}}
        for k in self.kernels:
            node = root
            node["ms"] += k.duration_ns * k.count / 1e6
            node["n"] += k.count
            for part in k.scope.split("/"):
                node = node["children"].setdefault(part, {"ms": 0.0, "n": 0, "children": {}})
                node["ms"] += k.duration_ns * k.count / 1e6
                node["n"] += k.count
        return root

    def table(self) -> str:
        l0 = self.l0
        lines = [
            f"{self.label}: {self.meta['device']}, jax {self.meta['jax']}, jno {self.meta.get('jno', '?')}",
            f"  compile {l0['compile_s']:.2f} s, steady {np.median(l0['steady_s']) * 1e3:.1f} ms "
            f"(reps {len(l0['steady_s'])}), XLA static {l0['flops'] / 1e9:.2f} GFLOP, {l0['bytes'] / 1e9:.3f} GB accessed, "
            f"temp {l0['temp_gb']:.3f} GB; pool peak {l0['pool_peak_gb']:.2f} GB, card delta {l0['card_delta_gb']:.2f} GB, "
            f"rss {l0['rss_gb']:.1f} GB",
            f"  transfer guard: {l0['transfer_guard']}",
        ]
        if self.kernels:
            tot = sum(k.duration_ns * k.count for k in self.kernels) / 1e6
            small = sum(k.count for k in self.kernels if k.duration_ns < 20e3)
            lines.append(
                f"  device: {tot:.1f} ms in {sum(k.count for k in self.kernels):,} launches "
                f"({small:,} under 20 us); memcpy {sum(m.bytes * m.count for m in self.memcpys) / 1e6:.1f} MB in "
                f"{sum(m.count for m in self.memcpys)} copies"
            )
            ib = self.intrinsic_bounds()
            lines.append(
                f"  {'stage':28s} {'fwd ms':>8s} {'adj ms':>8s} {'launches':>9s} {'memcpy':>14s} {'x intrinsic':>11s}  top kernel"
            )
            for s, a in self.by_stage().items():
                top = max(a["kernels"], key=lambda k: k.duration_ns * k.count)
                mc = [m for m in self.memcpys if m.stage == s]
                mtxt = f"{sum(m.bytes * m.count for m in mc) / 1e6:.1f} MB/{sum(m.count for m in mc)}" if mc else "-"
                itxt = f"x{(a['forward_ms'] + a['adjoint_ms']) / ib[s]:.0f}" if s in ib and ib[s] > 0 else "-"
                lines.append(
                    f"  {s:28s} {a['forward_ms']:8.1f} {a['adjoint_ms']:8.1f} {a['launches']:9,d} {mtxt:>14s} {itxt:>11s}  "
                    f"{top.hlo_op} ({top.duration_ns * top.count / 1e6:.1f} ms, occ {top.occupancy_pct:.0f} %)"
                )
            try:
                from dev.figures import bounds_from_kernels

                kb = bounds_from_kernels(os.path.join(RUNS, self.label))
            except Exception:  # noqa: BLE001
                kb = {}
            lines.append("  verdicts:")
            for s, v in self.verdicts(kb).items():
                lines.append(f"    {s:24s} {v}")
            h2d = [m for m in self.memcpys if m.kind != "D2D"]
            if h2d:
                lines.append(
                    "  host<->device: "
                    + "; ".join(
                        f"{m.kind} {m.bytes * m.count / 1e6:.2f} MB x{m.count} ({m.src}) in {m.stage}"
                        for m in sorted(h2d, key=lambda m: -m.bytes * m.count)[:6]
                    )
                )
        for b in self.backend_stats:
            lines.append(f"  backend {b}")
        return "\n".join(lines)

    def save(self, out_dir=None, figures=True):
        if os.environ.get("JNO_PROF_NCU") == "1" or os.environ.get("JNO_PROF_PERF"):
            return None  # a counter run must not overwrite the profiled run
        out_dir = out_dir or os.path.join(RUNS, self.label)
        os.makedirs(out_dir, exist_ok=True)
        d = dataclasses.asdict(self)
        with open(os.path.join(out_dir, "run.json"), "w") as fh:
            json.dump(d, fh, indent=1, default=float)
        with open(os.path.join(out_dir, "summary.txt"), "w") as fh:
            fh.write(self.table() + "\n")
        if figures and self.kernels:
            from dev import figures as _f

            _f.all_figures(self, out_dir)
        return out_dir


# ── the trace parse ─────────────────────────────────────────────────────────────────────────────────────
def _hlo_meta(hlo_text: str) -> dict[str, str]:
    """hlo instruction name -> op_name metadata (the scope path), fusions carrying their root's."""
    meta = {}
    for m in _META_RE.finditer(hlo_text):
        meta.setdefault(m.group(1), m.group(2))
    return meta


def _stage_of(scope: str):
    m = SCOPE_RE.search(scope)
    return (f"jno/{m.group(1)}" if m else "other"), "transpose(" in scope


def parse_trace(trace_dir: str, hlo_meta: dict[str, str], n_calls: int = 1):
    """Kernels and memcpys of the device planes of a JAX profiler trace, scope paths joined from the HLO.
    `n_calls` traced calls are averaged into one."""
    import glob

    from jax.profiler import ProfileData

    files = glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True)
    if not files:
        raise FileNotFoundError(f"no xplane.pb under {trace_dir}")
    p = ProfileData.from_file(max(files, key=os.path.getmtime))
    kernels: dict[tuple, Kernel] = {}
    memcpys: dict[tuple, Memcpy] = {}
    planes = list(p.planes)
    device = [pl for pl in planes if "/device:" in pl.name and any(True for ln in pl.lines for _ in ln.events)]
    if not device:  # XLA:CPU: the thunks run on the Eigen worker threads, one timed event per HLO op (no stats)
        for pl in planes:
            if "/host:CPU" not in pl.name:
                continue
            for line in pl.lines:
                if not line.name.startswith("tf_XLA"):
                    continue
                for e in line.events:
                    op = e.name
                    if op.startswith("end:") or op not in hlo_meta:
                        continue
                    scope = hlo_meta[op]
                    stage, adj = _stage_of(scope)
                    k = kernels.get((op, op))
                    if k is None:
                        kernels[(op, op)] = Kernel(op, op, scope, stage, adj, e.duration_ns, 1)
                    else:
                        k.duration_ns = (k.duration_ns * k.count + e.duration_ns) / (k.count + 1)
                        k.count += 1
        for k in kernels.values():
            k.count = max(1, round(k.count / n_calls))
        return list(kernels.values()), []
    for plane in device:
        for line in plane.lines:
            for e in line.events:
                st = {k: str(v) for k, v in e.stats}
                if e.name.startswith("Memcpy"):  # driver copies (memcpy32_post etc. are XLA kernels, counted below)
                    kind = e.name.split("(")[0][6:]
                    det = st.get("memcpy_details", "")
                    g = lambda key: (re.search(key + r":(\w+)", det) or [None, ""])[1]
                    nb = int(g("size") or 0)
                    scope = (hlo_meta.get(st.get("hlo_op", "")) or st.get("name", "")).split(",hlo_op=")[0]
                    stg, _ = _stage_of(scope)
                    m = memcpys.setdefault(
                        (kind, nb, stg), Memcpy(kind, nb, 0.0, 0, stg, scope, g("kind_src"), g("kind_dst"))
                    )
                    m.duration_ns += e.duration_ns
                    m.count += 1
                    continue
                hlo_op = st.get("hlo_op", "")
                if not hlo_op and "tf_op" not in st and "name" not in st:
                    continue  # not an XLA kernel event (e.g. a memset)
                # kernels launched from a CUDA graph report hlo_op=command_buffer_N; the event name is then the
                # real kernel, whose HLO instruction is the name with its trailing _N as .N
                cands = [hlo_op, e.name, re.sub(r"_(\d+)$", r".\1", e.name)]
                scope = next((hlo_meta[c] for c in cands if c in hlo_meta), None) or st.get("name", "") or hlo_op or e.name
                scope = scope.split(",hlo_op=")[0]
                if hlo_op.startswith("command_buffer"):
                    hlo_op = next((c for c in cands[1:] if c in hlo_meta), hlo_op)
                stage, adj = _stage_of(scope)
                kd = st.get("kernel_details", "")
                regs = int(re.search(r"regs:(\d+)", kd).group(1)) if "regs:" in kd else -1
                grid = re.search(r"grid:([\d,]+)", kd)
                block = re.search(r"block:([\d,]+)", kd)
                key = (hlo_op, e.name)
                k = kernels.get(key)
                if k is None:
                    kernels[key] = Kernel(
                        hlo_op,
                        e.name,
                        scope,
                        stage,
                        adj,
                        e.duration_ns,
                        1,
                        float(st.get("theoretical_occupancy_pct", "nan")),
                        regs,
                        grid.group(1) if grid else "",
                        block.group(1) if block else "",
                    )
                else:
                    k.duration_ns = (k.duration_ns * k.count + e.duration_ns) / (k.count + 1)
                    k.count += 1
    for k in kernels.values():
        k.count = max(1, round(k.count / n_calls))
    for m in memcpys.values():
        m.duration_ns /= max(1, m.count)
        m.count = max(1, round(m.count / n_calls))
    return list(kernels.values()), list(memcpys.values())


def reparse(run_dir: str, figures: bool = True) -> Result:
    """Rebuild a saved run's kernels from its trace and compiled HLO (after a change to the parse), and resave."""
    with open(os.path.join(run_dir, "run.json")) as fh:
        d = json.load(fh)
    kernels, memcpys = parse_trace(
        os.path.join(run_dir, "trace"), _hlo_meta(open(os.path.join(run_dir, "compiled.hlo")).read())
    )
    r = Result(d["label"], d["meta"], d["l0"], kernels, memcpys, d["backend_stats"], d["trace_dir"])
    r.save(run_dir, figures=figures)
    return r


# ── the run ─────────────────────────────────────────────────────────────────────────────────────────────
def run(
    fn,
    *args,
    label: str,
    reps: int = 3,
    trace: bool = True,
    out_dir: str | None = None,
    transfer_check: bool = True,
    backend_stats=None,
    sizes: dict | None = None,
) -> Result:
    """Profile one jitted callable at `args`. The first call compiles (timed apart); `reps` steady-state calls
    are timed; one more runs inside the profiler trace. `backend_stats`: a callable returning the solver
    backends' records (default: jno.utils.profiling.stats() if jNO carries the hook). `sizes`: what the
    program is made of, for the intrinsic bounds (`intrinsic_bounds`): cells, nodes, nnz (operator entries),
    n (unknowns), nrhs, complex (bool), dtype_bytes; factor nnz comes from the backend record (`lu_nnz`)."""
    try:
        import jno

        jno_ver = getattr(jno, "__version__", "?")
    except Exception:  # noqa: BLE001
        jno_ver = None
    dev = jax.local_devices()[0]
    meta = dict(
        label=label,
        jax=jax.__version__,
        jno=jno_ver,
        device=str(dev),
        device_kind=dev.device_kind,
        argv=sys.argv,
        time=time.strftime("%Y-%m-%d %H:%M"),
        cwd=os.getcwd(),
        sizes=dict(sizes or {}),
    )
    under_ncu = os.environ.get("JNO_PROF_NCU") == "1"  # dev/kernels.py re-runs the script under ncu: no trace, no guard
    trace, transfer_check, reps = (False, False, 1) if under_ncu else (trace, transfer_check, reps)
    card0 = _card_used_gb()
    if not hasattr(fn, "lower"):
        fn = jax.jit(fn)
    t = time.perf_counter()
    lowered = fn.lower(*args)
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t
    hlo_text = compiled.as_text()
    ca = compiled.cost_analysis()
    ca = ca[0] if isinstance(ca, list) else (ca or {})
    try:
        ma = compiled.memory_analysis()
        temp_gb = ma.temp_size_in_bytes / 1e9
        arg_gb, out_gb = ma.argument_size_in_bytes / 1e9, ma.output_size_in_bytes / 1e9
    except Exception:  # noqa: BLE001
        temp_gb = arg_gb = out_gb = float("nan")
    first_s, _ = _wall(fn, *args)  # first execution (cuDSS symbolic analysis, caches, ...)
    perf_events = os.environ.get("JNO_PROF_PERF")  # dev/cpu.py: record the steady-state calls with perf
    if perf_events:
        from dev.cpu import Recorder, perf_map

        n_map = len(perf_map())
        rec = Recorder(perf_events.split(","), os.environ.get("JNO_PROF_PERF_OUT", "perf.data")).start()
        steady = [_wall(fn, *args)[0] for _ in range(int(os.environ.get("JNO_PROF_PERF_REPS", "20")))]
        rec.stop()
        print(f"perf: {n_map} compiled kernels in the map, {len(steady)} calls recorded", flush=True)
        trace, transfer_check = False, False
    else:
        steady = [_wall(fn, *args)[0] for _ in range(reps)]
    guard = "not checked"
    if transfer_check:
        try:
            with jax.transfer_guard("disallow"):
                _block(fn(*args))
            guard = "no implicit transfer at the Python level (callbacks are not seen by the guard)"
        except Exception as e:  # noqa: BLE001
            guard = "IMPLICIT TRANSFER: " + str(e).splitlines()[0][:160]
    try:
        pool_peak = (dev.memory_stats() or {}).get("peak_bytes_in_use", float("nan")) / 1e9
    except Exception:  # noqa: BLE001
        pool_peak = float("nan")
    l0 = dict(
        compile_s=compile_s,
        first_s=first_s,
        steady_s=steady,
        flops=float(ca.get("flops", float("nan"))),
        bytes=float(ca.get("bytes accessed", float("nan"))),
        transcendentals=float(ca.get("transcendentals", 0.0)),
        temp_gb=temp_gb,
        arg_gb=arg_gb,
        out_gb=out_gb,
        pool_peak_gb=pool_peak,
        card_delta_gb=_card_used_gb() - card0,
        rss_gb=_rss_gb(),
        transfer_guard=guard,
    )
    kernels, memcpys, trace_dir = [], [], None
    if trace:
        trace_dir = os.path.join(out_dir or os.path.join(RUNS, label), "trace")
        os.makedirs(trace_dir, exist_ok=True)
        with jax.profiler.trace(trace_dir):
            _block(fn(*args))
        kernels, memcpys = parse_trace(trace_dir, _hlo_meta(hlo_text))
    stats = []
    if backend_stats is None:
        try:
            from jno.utils import profiling as _p

            backend_stats = getattr(_p, "stats", None)
        except Exception:  # noqa: BLE001
            backend_stats = None
    if backend_stats is not None:
        try:
            stats = list(backend_stats())
        except Exception as e:  # noqa: BLE001
            stats = [{"error": repr(e)}]
    if not under_ncu and not perf_events:
        os.makedirs(out_dir or os.path.join(RUNS, label), exist_ok=True)
        with open(os.path.join(out_dir or os.path.join(RUNS, label), "compiled.hlo"), "w") as fh:
            fh.write(hlo_text)
    return Result(label, meta, l0, kernels, memcpys, stats, trace_dir)
