"""Plot the FEM benchmark suite: what scales, what it costs to hold, and where the time goes.

    pixi run python benchmarks/plot_bench.py        # reads results.json, writes fem_benchmark.png

Three panels, because the suite answers three different questions.

LEFT plots SOLVE time against problem size, log-log -- deliberately not total time. Build is
dominated by a per-problem XLA compilation, so plotting the total collapses several cases into one
band; the solve is the part that tracks problem size. The line is the MEDIAN of repeated runs and
the shading is min-to-max across them, each repeat a fresh process, so the shading covers cold
variation -- driver init, XLA compilation and this card's clock state -- not just kernel jitter.
A point crossed with an X did not converge: its timing measures nothing and the curve through it
should not be read.

MIDDLE plots PEAK DEVICE MEMORY against the same axis, and is the panel most likely to be
misread, so: NOTHING here is near the card. The largest point is 3.1 GB against 8 GB nominal, and
3-D Poisson's ladder ends at 586 MB. Its "memory-walled" label is about where the curve WOULD land
at the ~1.3M tets a 10 s solve needs, extrapolated -- what stops it at 98k in practice is a 31 s
build. The measured wall is the transient's: ~1.5 GB at only 18k spatial DOFs, an order of magnitude
above anything else at that size, because the whole trajectory is stored -- and 6000 steps does OOM.
The value is the peak over the WHOLE case, build and solve together (``peak_bytes_in_use`` is a
running high-water mark and cannot be attributed to the solve alone); that is the right number for
"will this run", not for a solver's working set.

RIGHT is the share of wall time that is actually SOLVING, the rest being mesh + assembly + XLA
compilation. Bounded 0-100%, so the cases stay comparable by eye. Sizing the solves up to ~10 s did
not rescue the steady linear cases -- they sit near 5-7% and 3-D Poisson falls to ~1%, i.e. worse
with size, not better. Above half are the cases whose cost is not a single linear solve: the
transient, H(curl) at its larger sizes, and Newton. The adjoint case sits well above its forward
twin because a gradient does ~2x the work against the same one-time build -- for an inverse problem,
where the build is paid once and iterated on, build overhead largely stops mattering.

The black triangles are REFERENCE slopes, not fits. Measured least-squares exponents, for reading
against them: solve time 1.53 for the direct H(curl) solve (LU fill-in), 1.07 for 2-D Poisson,
0.08 for 3-D Poisson (flat -- it never becomes solve-bound); peak memory 0.97 for the vector and
transient cases, i.e. linear in DOFs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent

#: Okabe-Ito — the colourblind-safe qualitative palette (Okabe & Ito, "Color Universal Design", 2008).
#: Spelled out rather than pulled from seaborn so this stays runnable in the plain test env.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
TEAL, GRAY, INK = "#0D9488", "#94A3B8", "#1A202C"


#: A VARIANT of a case -- the same problem differentiated (``_adj``) or solved by a different
#: solver (``_amg``) -- is drawn in its parent's colour with its own dash pattern, so the cost of
#: the variation can be read directly against the thing it varies.
VARIANT_STYLES = {"_adj": "--", "_amg": ":"}


def pair_styles(order):
    """Map case -> (colour, linestyle), pairing ``<case><suffix>`` with ``<case>``."""

    def split(case):
        for suffix, dash in VARIANT_STYLES.items():
            if case.endswith(suffix):
                return case[: -len(suffix)], dash
        return case, "-"

    base = [c for c in order if split(c)[1] == "-"]
    palette = {c: OKABE_ITO[i % len(OKABE_ITO)] for i, c in enumerate(base)}
    styles = {}
    for c in order:
        parent, dash = split(c)
        if dash != "-" and parent in palette:
            styles[c] = (palette[parent], dash)
        else:
            styles[c] = (palette.get(c, OKABE_ITO[len(palette) % len(OKABE_ITO)]), "-")
    return styles


def curve(ax, x, y, color, label=None, n=250, ls="-", logy=True):
    """Draw the trend through measured points as a smooth curve, in log-log space.

    PCHIP rather than a cubic spline, deliberately: it is shape-preserving, so it cannot invent a
    local maximum or minimum that the measurements do not contain. These ladders are not perfectly
    monotone -- the transient dips at 6k DOFs and 3-D Poisson wiggles along its flat stretch, both
    run-to-run scatter on a latency-bound solve -- and a natural cubic would overshoot into
    structure that was never measured.

    The curve BETWEEN points is still interpolation. With markers off there is nothing showing
    where the six measurements per case actually sit; pass ``ms``-marked points back in if that
    distinction matters for a given audience.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        ax.plot(x, y, lw=1.6, color=color, label=label, ls=ls)
        return
    lx = np.log10(x)
    fine = np.linspace(lx[0], lx[-1], n)
    # interpolate in the space the AXIS uses, or the curve bends the wrong way between points
    ly = np.log10(y) if logy else y
    fy = PchipInterpolator(lx, ly)(fine)
    ax.plot(10**fine, 10**fy if logy else fy, lw=1.6, color=color, label=label, ls=ls)


def band(ax, x, lo, hi, color, n=250):
    """Shade min-to-max across repeats. Absent bounds mean a single run -- draw nothing."""
    x, lo, hi = np.asarray(x, float), np.asarray(lo, float), np.asarray(hi, float)
    if len(x) < 3 or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return
    lx = np.log10(x)
    fine = np.linspace(lx[0], lx[-1], n)
    f_lo = 10 ** PchipInterpolator(lx, np.log10(lo))(fine)
    f_hi = 10 ** PchipInterpolator(lx, np.log10(hi))(fine)
    ax.fill_between(10**fine, f_lo, f_hi, color=color, alpha=0.18, lw=0, zorder=2)


def mark_unconverged(ax, rs, key):
    """Cross out any point whose solve did not converge -- its timing measures nothing."""
    bad = [r for r in rs if r.get("converged") is False]
    if bad:
        ax.plot(
            [r["dofs"] for r in bad],
            [r[key] for r in bad],
            ls="none",
            marker="x",
            ms=7,
            mew=1.6,
            color=INK,
            zorder=6,
        )


def slope_triangle(ax, x0, y0, slope, decades=0.3, label=None):
    """Draw a REFERENCE slope on log-log axes: a right triangle of the given exponent.

    This is a ruler, not a fit -- it marks what ``t ~ n**slope`` looks like on these axes so the
    measured curves can be read against it. ``x0, y0`` are the lower-left corner in DATA units.
    """
    x1 = x0 * 10**decades
    y1 = y0 * 10 ** (slope * decades)
    ax.plot([x0, x1, x1, x0], [y0, y0, y1, y0], color=INK, lw=0.9, zorder=5)
    ax.fill([x0, x1, x1], [y0, y0, y1], color=INK, alpha=0.07, zorder=4, lw=0)
    if label:
        ax.text(x1 * 1.15, (y0 * y1) ** 0.5, label, fontsize=7.5, color=INK, va="center", ha="left")


plt.rcParams.update(
    {
        "figure.figsize": (6, 4),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "font.sans-serif": ["Frutiger 45 Light", "Frutiger", "FreeSans", "DejaVu Sans"],
        "font.weight": "light",
        "font.size": 10.5,
        "axes.titleweight": "light",
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
    }
)


def main() -> None:
    recs = [r for r in json.loads((HERE / "results.json").read_text()) if "failed" not in r]
    if not recs:
        raise SystemExit("no successful benchmark records in results.json")

    cases, order = {}, []
    for r in recs:
        cases.setdefault(r["case"], []).append(r)
        if r["case"] not in order:
            order.append(r["case"])
    for rs in cases.values():
        rs.sort(key=lambda r: r["dofs"])

    styles = pair_styles(order)
    fig, (ax_s, ax_m, ax_b) = plt.subplots(1, 3, figsize=(16.5, 4.4))

    # --- left: SOLVE time vs size ----------------------------------------------------------
    # Solve, not total. Total wall time is nearly flat in this range because build is dominated by
    # a per-problem XLA compilation that does not care how big the mesh is, so a total-vs-dofs plot
    # collapses every case into one narrow band and shows nothing. The solve is the part that
    # actually tracks problem size; the fixed cost it hides behind is the right panel's subject.
    for case in order:
        rs, (c, ls) = cases[case], styles[case]
        band(
            ax_s,
            [r["dofs"] for r in rs],
            [r.get("solve_ms_min", np.nan) / 1e3 for r in rs],
            [r.get("solve_ms_max", np.nan) / 1e3 for r in rs],
            c,
        )
        curve(ax_s, [r["dofs"] for r in rs], [r["solve_ms"] / 1e3 for r in rs], c, label=rs[0]["label"], ls=ls)
        mark_unconverged(ax_s, [dict(r, solve_s=r["solve_ms"] / 1e3) for r in rs], "solve_s")
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("Degrees of freedom")
    ax_s.set_ylabel("Solve time  [s]")
    # n and n^1.5 bracket what is here: the direct H(curl) solve fits 1.53 (LU fill-in), the
    # iterative 2-D Poisson 1.07, and 3-D Poisson 0.08 -- flat, because it is not solve-bound at all.
    slope_triangle(ax_s, 5.5e5, 0.30, 1.0, label="$n$")
    slope_triangle(ax_s, 5.5e5, 0.80, 1.5, label="$n^{3/2}$")

    # --- middle: peak device memory vs size ------------------------------------------------
    # Peak over the whole case (build AND solve), because that is what decides whether a problem
    # runs at all. It is why three ladders stop where they do.
    for case in order:
        rs, (c, ls) = cases[case], styles[case]
        curve(ax_m, [r["dofs"] for r in rs], [r["peak_mb"] for r in rs], c, ls=ls)
    ax_m.axhline(8192, color=GRAY, lw=1.0, ls=":")
    # x in AXES fraction, y in data units -- reading get_xlim() here would predate the log scaling
    # below and silently place the label off the axis
    ax_m.text(
        0.02,
        8192,
        "8 GB card (nominal)",
        transform=ax_m.get_yaxis_transform(),
        va="bottom",
        ha="left",
        fontsize=7,
        color=GRAY,
    )
    ax_m.set_xscale("log")
    ax_m.set_yscale("log")
    ax_m.set_xlabel("Degrees of freedom")
    ax_m.set_ylabel("Peak device memory  [MB]")
    slope_triangle(ax_m, 4.5e5, 130.0, 1.0, label="$n$")

    # --- right: how much of the wall clock is actually solving --------------------------------
    # A share rather than a ratio: it is bounded, so the cases stay comparable by eye, and 50% reads
    # directly as "half the time is the build". The rest is meshing, assembly and XLA compilation.
    for case in order:
        rs, (c, ls) = cases[case], styles[case]
        share = [100.0 * r["solve_ms"] / max(r["build_ms"] + r["solve_ms"], 1e-9) for r in rs]
        curve(ax_b, [r["dofs"] for r in rs], share, c, ls=ls, logy=False)
    ax_b.axhline(50.0, color=INK, lw=1.0, ls="--")
    ax_b.text(
        0.02, 50.0, " half the time is build", transform=ax_b.get_yaxis_transform(), va="bottom", fontsize=7, color=INK
    )
    ax_b.set_xscale("log")
    ax_b.set_ylim(0, 100)
    ax_b.set_xlabel("Degrees of freedom")
    ax_b.set_ylabel("Solve share of wall time  [%]")

    # --- one legend for all three panels ------------------------------------------------------
    handles, labels = ax_s.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))

    out = HERE / "fem_benchmark.png"
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out)
    print(f"wrote {out}")

    print(f"\n{'case':<26}{'dofs':>9}{'build s':>10}{'solve s':>10}{'peak MB':>10}")
    for case in order:
        for r in cases[case]:
            print(
                f"{r['label']:<26}{r['dofs']:>9,}{r['build_ms'] / 1e3:>10.2f}"
                f"{r['solve_ms'] / 1e3:>10.2f}{r['peak_mb']:>10.0f}"
            )


if __name__ == "__main__":
    main()
