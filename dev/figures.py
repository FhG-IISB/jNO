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


def _stage_colors(stages):
    """Up to four stages get Okabe-Ito colours in order of time; the rest muted."""
    return {s: (hs.CYCLE[i] if i < 4 else hs.MUTED) for i, s in enumerate(stages)}


def ladder(res, bounds=None, ax=None):
    """Per stage: measured time (filled), the launch floor (hollow square, estimate) and, when `bounds` gives
    them ({stage: (kernel_bound_ms, intrinsic_ms)}), the roof bounds. Sorted by measured time."""
    agg = res.by_stage()
    stages = list(agg)
    fig = None
    if ax is None:
        H = 1.1 + 0.4 * len(stages)
        fig = plt.figure(figsize=(6.69, H), layout="none")
        ax = fig.add_axes([0.24, 0.5 / H, 0.52, 1 - 1.05 / H])
    y = np.arange(len(stages))[::-1]
    lo = np.inf
    for yi, s in zip(y, stages):
        a = agg[s]
        m = a["forward_ms"] + a["adjoint_ms"]
        floor = a["launches"] * LAUNCH_S * 1e3
        kb, ib = (bounds or {}).get(s, (None, None))
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
        ax.text(m * 1.3, yi, txt, va="center", ha="left", fontsize=8, color=hs.INK)
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


def all_figures(res, out_dir, bounds=None):
    hs.apply("paper")
    for name, f in (
        ("headroom_ladder", lambda: ladder(res, bounds)),
        ("icicle", lambda: icicle(res)),
        ("launch_spectrum", lambda: spectrum(res)),
    ):
        fig = f()
        hs.save(fig, os.path.join(out_dir, name))
        plt.close(fig)
