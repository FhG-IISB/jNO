"""Figures of one profiling run (house style): the headroom ladder, the mirrored icicle, the launch spectrum."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from dev import house_style as hs

LAUNCH_S = 4e-6  # per-launch floor used for the launch-bound estimate (hollow marker; an estimate, not a measurement)
# RTX 3070 (GA104): FP64 at 1/64 of FP32 (46 SMs x 2 FP64 FMA/clk x 2 flop x 1.725 GHz boost); DRAM 448 GB/s
PEAK = {"fp64_flop_s": 46 * 2 * 2 * 1.725e9, "dram_B_s": 448e9}


def bounds_from_kernels(run_dir, peak=PEAK):
    """{stage: (kernel_bound_ms, None, n_profiled_launches)} from the ncu passes of a run: the roof at each kernel's
    MEASURED intensity, summed over the profiled launches (the ladder scales to one call by the launch count)."""
    import glob
    import json

    out = {}
    for f in glob.glob(os.path.join(run_dir, "kernels_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        recs = [r for r in d["kernels"] if "fp64_flop" in r and "dram_B" in r]
        if not recs:
            continue
        t = sum(max(r["fp64_flop"] / peak["fp64_flop_s"], r["dram_B"] / peak["dram_B_s"]) for r in recs)
        out[d["stage"]] = (t * 1e3, None, len(recs))
    return out


def _stage_colors(stages):
    """Up to four stages get Okabe-Ito colours in order of time; the rest muted."""
    return {s: (hs.CYCLE[i] if i < 4 else hs.MUTED) for i, s in enumerate(stages)}


def ladder(res, bounds=None, ax=None, verdict=None):
    """Per stage: measured time (filled), the launch floor (hollow square, estimate) and, when `bounds` gives
    them ({stage: (kernel_bound_ms, intrinsic_ms)}), the roof bounds. Sorted by measured time."""
    agg = res.by_stage()
    stages = list(agg)
    fig = None
    if ax is None:
        H = 1.1 + 0.4 * len(stages)
        fig = plt.figure(figsize=(6.69, H), layout="none")
        ax = fig.add_axes([0.24, 0.5 / H, 0.42, 1 - 1.05 / H])
    y = np.arange(len(stages))[::-1]
    lo = np.inf
    for yi, s in zip(y, stages):
        a = agg[s]
        m = a["forward_ms"] + a["adjoint_ms"]
        floor = a["launches"] * LAUNCH_S * 1e3
        b = (bounds or {}).get(s, (None, None))
        kb, ib = b[0], b[1]
        if kb and len(b) > 2 and b[2] and a["launches"]:
            kb *= a["launches"] / b[2]
        lo = min(lo, floor, *(v for v in (kb, ib) if v))
        if ib:
            ax.plot([ib, kb or m], [yi, yi], color=hs.MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
        if kb:
            ax.plot([kb, m], [yi, yi], color=hs.INK, lw=0.8, zorder=1)
            ax.plot(kb, yi, hs.MARKERS[1], color=hs.SKY, ms=4, mec="white", mew=0.5, zorder=3)
        if ib:
            ax.plot(ib, yi, hs.MARKERS[2], color=hs.GREEN, ms=4.5, mec="white", mew=0.5, zorder=3)
        ax.plot(floor, yi, hs.MARKERS[1], color=hs.MUTED, ms=4, mfc="white", mew=0.6, zorder=2)
        ax.plot(m, yi, hs.MARKERS[0], color=hs.ORANGE, ms=4.5, mec="white", mew=0.5, zorder=3)
        txt = f"{m:.1f} ms, {a['launches']:,} launches"
        if kb:
            txt = f"x{m / kb:.1f}" + (f" / x{m / ib:.0f}" if ib else "") + "   " + txt
        elif ib:
            txt = f"- / x{m / ib:.0f}   " + txt
        ax.text(m * 1.3, yi, txt, va="center", ha="left", fontsize=8, color=hs.INK)
        if verdict and s in verdict:
            ax.text(
                1.02,
                yi,
                verdict[s].split(":")[0],
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=7,
                color=hs.MUTED,
                clip_on=False,
            )
    ax.yaxis.set_major_locator(mt.FixedLocator(list(y)))
    ax.set_yticklabels(stages)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.set_xscale("log")
    hi = max(agg[s]["forward_ms"] + agg[s]["adjoint_ms"] for s in stages)
    ax.set_xlim(10 ** np.floor(np.log10(max(lo, 1e-3))), 10 ** np.ceil(np.log10(hi * 3)))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Device time per call [ms]")
    ax.tick_params(axis="y", length=0)
    h = [
        Line2D(
            [], [], marker=hs.MARKERS[0], color=hs.ORANGE, ls="", ms=4.5, mec="white", label="measured (forward + adjoint)"
        ),
        Line2D(
            [],
            [],
            marker=hs.MARKERS[1],
            color=hs.MUTED,
            ls="",
            ms=4,
            mfc="white",
            label="launch floor (launches x 4 us, estimate)",
        ),
    ]
    if bounds:
        h += [
            Line2D([], [], marker=hs.MARKERS[1], color=hs.SKY, ls="", ms=4, mec="white", label="kernel bound"),
            Line2D([], [], marker=hs.MARKERS[2], color=hs.GREEN, ls="", ms=4.5, mec="white", label="intrinsic bound"),
        ]
    ax.legend(
        handles=h,
        loc="lower left",
        bbox_to_anchor=(0, 1.03),
        ncols=2,
        frameon=False,
        borderaxespad=0,
        handletextpad=0.4,
        columnspacing=1.5,
        fontsize=8,
    )
    return fig


def _subtree(res, adjoint: bool):
    root = {"ms": 0.0, "n": 0, "children": {}}
    for k in res.kernels:
        if k.adjoint != adjoint:
            continue
        ms = k.duration_ns * k.count / 1e6
        parts = [p for p in k.scope.replace("jno/", "jno:").split("/") if p and not p.startswith("jit(")]
        parts = [p.replace("transpose(jvp(", "").replace("jvp(", "").rstrip(")").replace("jno:", "jno/") for p in parts]
        node = root
        node["ms"] += ms
        node["n"] += k.count
        for p in parts[:4]:
            node = node["children"].setdefault(p, {"ms": 0.0, "n": 0, "children": {}})
            node["ms"] += ms
            node["n"] += k.count
    return root


def icicle(res, ax=None, depth: int = 4):
    """Forward above the axis growing down, adjoint below growing up; width = time, level = scope depth,
    colour = stage. Labels only where a box is wide enough."""
    fwd, adj = _subtree(res, False), _subtree(res, True)
    total = max(fwd["ms"], adj["ms"], 1e-9)
    stages = list(res.by_stage())
    col = _stage_colors(stages)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(6.69, 2.6), layout="none")
        ax = fig.add_axes([0.06, 0.12, 0.9, 0.82])
    ax.set_xlim(0, total)
    ax.set_ylim(-depth - 0.3, depth + 0.3)
    ax.axis("off")

    def draw(node, x0, level, sign, stage="other"):
        for name, ch in sorted(node["children"].items(), key=lambda kv: -kv[1]["ms"]):
            st = name if name.startswith("jno/") else stage
            w = ch["ms"]
            y0 = sign * (level - 1) if sign > 0 else -level
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    w,
                    1,
                    facecolor=col.get(st, hs.MUTED),
                    edgecolor="white",
                    lw=0.5,
                    alpha=0.9 - 0.12 * (level - 1),
                )
            )
            box_pt = w / total * ax.get_window_extent().width * 72 / ax.figure.dpi
            if box_pt > 4.2 * min(len(name), 10):  # label only where it fits (7 pt type)
                room = int(box_pt / 4.2)
                ax.text(
                    x0 + w / 2,
                    y0 + 0.5,
                    name if len(name) <= room else name[: max(room - 1, 3)] + "…",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=hs.INK,
                    clip_on=True,
                )
            if level < depth:
                draw(ch, x0, level + 1, sign, st)
            x0 += w

    draw(fwd, 0, 1, +1)
    draw(adj, 0, 1, -1)
    ax.axhline(0, color=hs.INK, lw=0.6)
    ax.text(0, depth + 0.1, f"forward {fwd['ms']:.1f} ms", fontsize=8, va="bottom", color=hs.INK)
    ax.text(0, -depth - 0.1, f"adjoint {adj['ms']:.1f} ms", fontsize=8, va="top", color=hs.INK)
    h = [Line2D([], [], marker="s", color=col[s], ls="", ms=6, label=s) for s in stages[:5]]
    ax.legend(
        handles=h,
        loc="lower right",
        bbox_to_anchor=(1, 1.0),
        ncols=min(5, len(h)),
        frameon=False,
        borderaxespad=0,
        fontsize=8,
        handletextpad=0.4,
    )
    return fig


def spectrum(res, ax=None):
    """Kernel durations (log), the time-weighted histogram filled and the launch-count histogram as an outline:
    mass under 20 us means launch-bound."""
    d = np.array([k.duration_ns / 1e3 for k in res.kernels for _ in range(k.count)])  # us, one per launch
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(3.25, 2.3), layout="none")
        ax = fig.add_axes([0.2, 0.2, 0.75, 0.68])
    bins = np.logspace(np.floor(np.log10(d.min())), np.ceil(np.log10(d.max())), 25)
    ax.hist(d, bins, weights=d / 1e3, color=hs.ORANGE, alpha=0.6, label="time [ms]")
    ax2 = ax.twinx()
    ax2.hist(d, bins, histtype="step", color=hs.INK, lw=0.8, label="launches")
    ax.axvline(20, color=hs.INK, lw=0.6, ls=(0, (4, 2)))
    ax.text(20 * 1.15, ax.get_ylim()[1] * 0.95, "20 us", fontsize=8, va="top", color=hs.INK)
    ax.set_xscale("log")
    ax.set_xlabel("Kernel duration [us]")
    ax.set_ylabel("Time [ms]")
    ax2.set_ylabel("Launches")
    ax2.grid(False)
    ax2.yaxis.set_major_locator(mt.MaxNLocator(5))  # own locator: tidy leaves it alone (no scaled labels)
    ax2.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    h = [
        Line2D([], [], color=hs.ORANGE, lw=6, alpha=0.6, label="time"),
        Line2D([], [], color=hs.INK, lw=0.8, label="launches"),
    ]
    ax.legend(handles=h, loc="upper right", frameon=False, fontsize=8)
    return fig


def bind_strip(run_dir, ax=None, top: int = 12):
    """Per profiled kernel kind (rows, by time): fraction-of-peak cells for SM, DRAM, FP64 pipe, L2 hit, occupancy,
    and the measured flop/byte against the ridge. Reads every kernels_<stage>.json of the run."""
    import glob
    import json

    rows = []
    for f in sorted(glob.glob(os.path.join(run_dir, "kernels_*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        by = {}
        for r in d["kernels"]:
            k = r["kernel"].split("(")[0].replace("void ", "")[:34]
            b = by.setdefault(
                k,
                {
                    "stage": d["stage"],
                    "t": 0.0,
                    "n": 0,
                    "f": 0.0,
                    "B": 0.0,
                    "sm": [],
                    "dram": [],
                    "fp64": [],
                    "l2": [],
                    "occ": [],
                },
            )
            b["t"] += r.get("time_ns", 0)
            b["n"] += 1
            b["f"] += r.get("fp64_flop", 0)
            b["B"] += r.get("dram_B", 0)
            for key, src in (
                ("sm", "sm_pct"),
                ("dram", "dram_pct"),
                ("fp64", "fp64_pipe_pct"),
                ("l2", "l2_hit_pct"),
                ("occ", "occupancy_pct"),
            ):
                if src in r:
                    b[key].append(r[src])
        rows += list(by.items())
    rows.sort(key=lambda kb: -kb[1]["t"])
    rows = rows[:top]
    if not rows:
        return None
    cols = ["SM", "DRAM", "FP64 pipe", "L2 hit", "occupancy"]
    avg = lambda v: np.mean(v) / 100 if v else np.nan
    M = np.array([[avg(b["sm"]), avg(b["dram"]), avg(b["fp64"]), avg(b["l2"]), avg(b["occ"])] for _, b in rows])
    fig = None
    if ax is None:
        H = 0.9 + 0.26 * len(rows)
        fig = plt.figure(figsize=(6.69, H), layout="none")
        ax = fig.add_axes([0.42, 0.12 / H, 0.38, 1 - 0.75 / H])
    ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(
                    j,
                    i,
                    f"{M[i, j] * 100:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if M[i, j] < 0.6 else hs.INK,
                )
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [f"{k}  ({b['stage'][4:]}, {b['t'] / 1e6:.1f} ms)" for k, b in rows], fontsize=7)
    ax.tick_params(length=0)
    ax.grid(False)
    ridge = PEAK["fp64_flop_s"] / PEAK["dram_B_s"]
    for i, (_, b) in enumerate(rows):
        ai = b["f"] / max(b["B"], 1)
        ax.text(
            len(cols) - 0.4,
            i,
            f"{ai:5.2f} flop/B " + ("compute" if ai > ridge else "memory"),
            ha="left",
            va="center",
            fontsize=7,
            color=hs.MUTED,
        )
    ax.text(len(cols) - 0.4, -0.9, f"intensity (ridge {ridge:.2f})", ha="left", va="center", fontsize=7, color=hs.MUTED)
    return fig


def all_figures(res, out_dir, bounds=None):
    hs.apply("paper")
    if bounds is None:
        kb = bounds_from_kernels(out_dir)
        ib = res.intrinsic_bounds()
        bounds = {
            s: (kb.get(s, (None, None, None))[0], ib.get(s), kb.get(s, (None, None, None))[2]) for s in set(kb) | set(ib)
        } or None
    for name, f in (
        ("headroom_ladder", lambda: ladder(res, bounds, verdict=res.verdicts(bounds_from_kernels(out_dir)))),
        ("icicle", lambda: icicle(res)),
        ("launch_spectrum", lambda: spectrum(res)),
        ("bind_strip", lambda: bind_strip(out_dir)),
    ):
        fig = f()
        if fig is None:
            continue
        hs.save(fig, os.path.join(out_dir, name))
        plt.close(fig)
