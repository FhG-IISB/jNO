"""The fan to the target size: per stage, device time and memory against the problem size over several profiled
runs, fitted exponents, and the extrapolation to a deployment size against the card's memory.

    python dev/fan.py dev/runs/small3d-s0.20 dev/runs/small3d-s0.15 ... --target 3e6 --out dev/runs/small3d-fan

Each run is one `prof.run(...)` with `sizes=` (the x-axis is `cells`; `--x n` picks the unknowns). Fits are least
squares in log-log over the runs given; an exponent from three points is a guide, not a measurement. Output: the
figure `fan.{pdf,png}` and `fan.json` (exponents, extrapolated ms and GB at the target)."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev import house_style as hs  # noqa: E402

CARD_GB = 8.0  # RTX 3070


def load(run_dir, x_key="cells"):
    with open(os.path.join(run_dir, "run.json")) as fh:
        d = json.load(fh)
    x = (d["meta"].get("sizes") or {}).get(x_key)
    if not x:
        raise SystemExit(f"{run_dir}: run.json has no sizes[{x_key!r}]; pass sizes= to prof.run")
    stages = {}
    for k in d["kernels"]:
        stages[k["stage"]] = stages.get(k["stage"], 0.0) + k["duration_ns"] * k["count"] / 1e6
    lu = next((b.get("lu_nnz") for b in d["backend_stats"] if b.get("lu_nnz")), None)
    dev_gb = next((b.get("device_peak") for b in d["backend_stats"] if b.get("device_peak")), 0) / 1e9
    return dict(
        x=x,
        label=d["label"],
        launches=sum(k["count"] for k in d["kernels"]),
        stages=stages,
        total_ms=float(np.median(d["l0"]["steady_s"]) * 1e3),
        pool_gb=d["l0"]["pool_peak_gb"],
        card_gb=d["l0"]["card_delta_gb"],
        solver_gb=dev_gb,
        lu_nnz=lu,
    )


def fit(xs, ys):
    """log-log slope and intercept; None when fewer than two positive points."""
    m = [(x, y) for x, y in zip(xs, ys) if x > 0 and y > 0]
    if len(m) < 2:
        return None
    lx, ly = np.log(np.array([a for a, _ in m])), np.log(np.array([b for _, b in m]))
    p, c = np.polyfit(lx, ly, 1)
    return float(p), float(c)


def fan(runs, target=None, x_label="Cells", ax_t=None, ax_m=None):
    runs = sorted(runs, key=lambda r: r["x"])
    xs = np.array([r["x"] for r in runs], float)
    stages = sorted({s for r in runs for s in r["stages"]}, key=lambda s: -max(r["stages"].get(s, 0) for r in runs))
    stages = [s for s in stages if s != "other"][:4] + (["other"] if any("other" in r["stages"] for r in runs) else [])
    fig = None
    if ax_t is None:
        fig, (ax_t, ax_m) = __import__("matplotlib.pyplot").pyplot.subplots(1, 2, figsize=(6.69, 2.5), layout="none")
        fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.8, wspace=0.35)
    out = {"x": xs.tolist(), "stages": {}, "memory": {}}
    xt = np.array([xs.min(), target or xs.max()])
    for i, s in enumerate(stages):
        ys = np.array([r["stages"].get(s, 0.0) for r in runs])
        col = hs.CYCLE[i] if i < 4 else hs.MUTED
        ax_t.plot(xs, ys, marker=hs.MARKERS[i % 6], color=col, lw=0.8, ms=3, mec="white", mew=0.4, label=s)
        f = fit(xs, ys)
        if f:
            p, c = f
            ax_t.plot(xt, np.exp(c) * xt**p, color=col, lw=0.6, ls=(0, (3, 2)))
            ax_t.text(xt[-1] * 1.08, np.exp(c) * xt[-1] ** p, f"n^{p:.2f}", fontsize=7, color=col, va="center")
            out["stages"][s] = dict(exponent=p, ms_at_target=float(np.exp(c) * (target or xs.max()) ** p))
    tot = np.array([r["total_ms"] for r in runs])
    ax_t.plot(xs, tot, color=hs.INK, lw=0.6, ls=(0, (1, 1.5)), label="wall")
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_xlabel(x_label)
    ax_t.set_ylabel("Time per call [ms]")
    hs.legend(ax_t)
    for j, (key, name) in enumerate(
        (("card_gb", "card delta"), ("pool_gb", "XLA pool peak"), ("solver_gb", "solver factors"))
    ):
        ys = np.array([r[key] for r in runs])
        ax_m.plot(xs, ys, marker=hs.MARKERS[j], color=hs.CYCLE[j], lw=0.8, ms=3, mec="white", mew=0.4, label=name)
        f = fit(xs, ys)
        if f:
            p, c = f
            ax_m.plot(xt, np.exp(c) * xt**p, color=hs.CYCLE[j], lw=0.6, ls=(0, (3, 2)))
            out["memory"][key] = dict(exponent=p, gb_at_target=float(np.exp(c) * (target or xs.max()) ** p))
    ax_m.axhline(CARD_GB, **hs.REFERENCE)
    ax_m.text(xs.min(), CARD_GB * 1.08, f"card {CARD_GB:g} GB", fontsize=7, color=hs.INK)
    if target:
        for ax in (ax_t, ax_m):
            ax.axvline(target, color=hs.MUTED, lw=0.6, ls=(0, (2, 2)))
        ax_m.text(
            target,
            ax_m.get_ylim()[0] * 1.3 if ax_m.get_yscale() == "log" else 0,
            "target",
            fontsize=7,
            color=hs.MUTED,
            ha="right",
        )
    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_xlabel(x_label)
    ax_m.set_ylabel("Device memory [GB]")
    hs.legend(ax_m)
    return fig, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--target", type=float, default=None, help="deployment size (in --x units) to extrapolate to")
    ap.add_argument("--x", default="cells", help="sizes key on the x-axis (cells, n, nnz)")
    ap.add_argument("--out", default=None, help="output directory (default: the first run's parent + '-fan')")
    a = ap.parse_args(argv)
    runs = [load(r, a.x) for r in a.runs]
    hs.apply("paper")
    fig, out = fan(runs, a.target, {"cells": "Cells", "n": "Unknowns $n$", "nnz": "Operator entries"}.get(a.x, a.x))
    out_dir = a.out or os.path.join(
        os.path.dirname(a.runs[0].rstrip("/")), os.path.basename(a.runs[0].rstrip("/")).split("-s")[0] + "-fan"
    )
    os.makedirs(out_dir, exist_ok=True)
    hs.save(fig, os.path.join(out_dir, "fan"))
    out["target"] = a.target
    with open(os.path.join(out_dir, "fan.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{len(runs)} runs, x = {[r['x'] for r in sorted(runs, key=lambda r: r['x'])]}")
    big = max(runs, key=lambda r: r["x"])
    floor_ms = big["launches"] * 4e-3
    if floor_ms > 0.3 * big["total_ms"]:
        print(
            f"  NOTE: the largest run is launch-dominated ({big['launches']:,} launches, floor ~{floor_ms:.1f} ms of "
            f"{big['total_ms']:.1f} ms): the exponents are below their asymptotic values; sweep larger sizes"
        )
        out["launch_dominated"] = True
    for s, v in out["stages"].items():
        print(f"  {s:20s} ~ n^{v['exponent']:.2f}" + (f", {v['ms_at_target']:.0f} ms at {a.target:g}" if a.target else ""))
    for k, v in out["memory"].items():
        print(f"  {k:20s} ~ n^{v['exponent']:.2f}" + (f", {v['gb_at_target']:.2f} GB at {a.target:g}" if a.target else ""))
    print("->", out_dir)


if __name__ == "__main__":
    main()
