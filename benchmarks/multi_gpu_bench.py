"""Multi-device FEM: time and per-device memory against the device count, for the cluster.

    python benchmarks/multi_gpu_bench.py                       # every case, sizes, 1/2/4/8 devices
    python benchmarks/multi_gpu_bench.py nonlinear3d --devices 1 8 --sizes 0 1
    python benchmarks/multi_gpu_bench.py _point nonlinear3d 1 8   # one point (what the driver runs)

Nothing here is measured on the development machine, which has one GPU: this script exists so the
claims in ``docs/fem/inverse.md`` (correctness, placement, collectives -- checked on simulated devices)
can be joined by the numbers only real hardware gives: strong scaling and per-device memory.

Every point runs in a FRESH PROCESS, for the reason ``run_bench.py`` records: in one process later points
inherit earlier compilations. A point is ``(case, size index, device count)``; the device count is passed
as ``fem.solve(shard=n)`` (``1`` = the single-device path), so every point sees the same visible devices
and only the placement changes. Each point records:

* ``first_s`` -- the first solve, compilation included (with jNO's persistent cache, a re-run is warm);
* ``repeat_s`` -- the median of three further solves: the cost a time loop or optimiser pays per call.
  An eager NONLINEAR solve re-stages its Newton loop on every call (``lax.while_loop`` traces its body
  per Python call), so its repeat carries a size-independent tracing cost -- ~0.4 s measured on a
  233-DOF problem on CPU -- that is not solver work; read scaling from the large sizes;
* ``peak_gb`` -- the peak memory of EVERY device, so a split that leaves one device holding everything
  (a replicated operator, a gathered residual) shows up as one large entry, not an average;
* ``max_diff`` -- against the one-device answer for the same case and size, computed in the same
  process, so a placement that changes the answer is caught next to its timing.

Cases (3-D, sized to fill an H200 at the largest index; lower the index on a smaller card):

* ``linear3d``    -- Poisson with a heterogeneous coefficient, default solve: the OPERATOR splits.
* ``nonlinear3d`` -- ``-div grad u + u^3 = f``, default Jacobian-free Newton: the CELLS split.
* ``march3d``     -- the same reaction-diffusion, transient, 10 backward-Euler Newton steps: the cells
  split inside the scan.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

#: mesh sizes per index; DOFs ~ (1/h)^3 on the unit cube. The largest is meant for 8 x H200.
SIZES = {
    "linear3d": (0.04, 0.025, 0.016, 0.011, 0.008),
    "nonlinear3d": (0.05, 0.03, 0.02, 0.014, 0.01),
    "march3d": (0.06, 0.04, 0.028, 0.02, 0.014),
}
DEVICES = (1, 2, 4, 8)
TIMEOUT_S = 7200


def _problem(case, h):
    import numpy as np

    import jno

    grad, inner = jno.np.grad, jno.np.inner
    sin = jno.np.sin
    if case == "march3d":
        d = jno.shape.box(0, 0, 0, 1, 1, 1, size=h).domain(time=(0.0, 0.01, 11))
    else:
        d = jno.shape.box(0, 0, 0, 1, 1, 1, size=h).domain()
    u, v = d.fem_symbols()
    xs = d.variable("interior", split=True)
    x, y, z = xs[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    B = dict(x=x, y=y, z=z) if case != "march3d" else dict(x=x, y=y, z=z, t=xs[3])
    U, V = u.bind(**B), v.bind(**B)
    gu, gv = grad(u, [x, y, z]), grad(v, [x, y, z])
    k = jno.np.exp(2.0 * sin(3 * np.pi * x) * sin(3 * np.pi * y) * sin(3 * np.pi * z))  # e^-2 .. e^2
    diff = k * inner(gu, gv, n_contract=1)
    bc = u(xb, yb, zb) - 0.0
    if case == "linear3d":
        return jno.fem([diff - 10.0 * V, bc])
    if case == "nonlinear3d":
        return jno.fem([diff + U**3 * V - 10.0 * V, bc])
    x0, y0, z0, _t0 = d.variable("initial", split=True)
    ic = u(x0, y0, z0) - sin(np.pi * x0) * sin(np.pi * y0) * sin(np.pi * z0)
    return jno.fem([U.t * V + diff + 5.0 * U**3 * V - 2.0 * V, bc, ic])


def _solve(fem, n):
    import jax

    s = fem.solve(shard=n if n > 1 else False)
    s = s.fn() if hasattr(s, "fn") else s
    return jax.block_until_ready(s)


def point(case, idx, n):
    """One (case, size, device count) point; prints a RESULT line."""
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    h = SIZES[case][idx]
    t0 = time.perf_counter()
    fem = _problem(case, h)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = _solve(fem, n)
    first_s = time.perf_counter() - t0
    reps = []
    for _ in range(3):
        t0 = time.perf_counter()
        _solve(fem, n)
        reps.append(time.perf_counter() - t0)
    peak = [round((d.memory_stats() or {}).get("peak_bytes_in_use", 0) / 1e9, 3) for d in jax.devices()]
    rec = {
        "case": case,
        "size_index": idx,
        "h": h,
        "dofs": int(fem.dofs),
        "devices": n,
        "visible": len(jax.devices()),
        "device_kind": jax.devices()[0].device_kind,
        "build_s": round(build_s, 2),
        "first_s": round(first_s, 3),
        "repeat_s": round(statistics.median(reps), 4),
        "peak_gb": peak,
        "placed_on": len(getattr(out, "sharding", None).device_set) if hasattr(out, "sharding") else 1,
    }
    if n > 1:  # same process, so this reference compiles cold here but does not affect the timings above
        ref = _solve(fem, 1)
        rec["max_diff"] = float(np.abs(np.asarray(out) - np.asarray(ref)).max())
    print("RESULT " + json.dumps(rec), flush=True)


def main(argv):
    if argv and argv[0] == "_point":
        point(argv[1], int(argv[2]), int(argv[3]))
        return
    cases = [a for a in argv if a in SIZES] or list(SIZES)
    devs = _opt(argv, "--devices", DEVICES)
    sizes = _opt(argv, "--sizes", None)
    out = os.environ.get("JNO_MULTI_GPU_OUT", "multi_gpu_results.json")
    results = json.load(open(out)) if os.path.exists(out) else []
    for case in cases:
        for idx in sizes if sizes is not None else range(len(SIZES[case])):
            for n in devs:
                cmd = [sys.executable, __file__, "_point", case, str(idx), str(n)]
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
                    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                    rec = (
                        json.loads(lines[-1][7:])
                        if lines
                        else {"case": case, "size_index": idx, "devices": n, "failed": (p.stderr or "").strip()[-400:]}
                    )
                except subprocess.TimeoutExpired:
                    rec = {"case": case, "size_index": idx, "devices": n, "failed": f"timeout {TIMEOUT_S}s"}
                results.append(rec)
                json.dump(results, open(out, "w"), indent=1)  # after every point: a crash loses nothing
                print(json.dumps(rec), flush=True)


def _opt(argv, flag, default):
    if flag not in argv:
        return default
    i = argv.index(flag) + 1
    vals = []
    while i < len(argv) and not argv[i].startswith("--") and argv[i] not in SIZES:
        vals.append(int(argv[i]))
        i += 1
    return vals


if __name__ == "__main__":
    main(sys.argv[1:])
