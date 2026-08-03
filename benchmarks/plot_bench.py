"""Plot the FEM benchmark suite: what scales on the left, where the time goes on the right.

    pixi run python benchmarks/plot_bench.py        # reads results.json, writes fem_benchmark.png

Two panels, because the suite answers two different questions.

LEFT plots SOLVE time against problem size, log-log -- deliberately not total time. Total is nearly
flat across this whole range, since build is dominated by a per-problem XLA compilation that does not
care how big the mesh is; plotting it collapses every case into one narrow band and shows nothing.
The solve is the part that tracks problem size.

RIGHT is why that matters: at each problem's largest size, how much of the wall clock is BUILD
(meshing, assembly, compilation) versus SOLVE. The split is lopsided toward build almost everywhere,
which is the single most useful thing this figure says -- and the reason optimising the solver alone
moves the wall clock so little.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

#: Okabe-Ito — the colourblind-safe qualitative palette (Okabe & Ito, "Color Universal Design", 2008).
#: Spelled out rather than pulled from seaborn so this stays runnable in the plain test env.
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"]
TEAL, GRAY = "#0D9488", "#94A3B8"

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

    colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(order))]
    fig, (ax_s, ax_b) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # --- left: SOLVE time vs size ----------------------------------------------------------
    # Solve, not total. Total wall time is nearly flat in this range because build is dominated by
    # a per-problem XLA compilation that does not care how big the mesh is, so a total-vs-dofs plot
    # collapses every case into one narrow band and shows nothing. The solve is the part that
    # actually tracks problem size; the fixed cost it hides behind is the right panel's subject.
    for c, case in zip(colors, order):
        rs = cases[case]
        ax_s.plot(
            [r["dofs"] for r in rs],
            [r["solve_ms"] / 1e3 for r in rs],
            marker="o",
            ms=4,
            lw=1.2,
            color=c,
            label=rs[0]["label"],
        )
    ax_s.set_xscale("log")
    ax_s.set_yscale("log")
    ax_s.set_xlabel("Degrees of freedom")
    ax_s.set_ylabel("Solve time  [s]")
    ax_s.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.32))

    # --- right: build vs solve at the largest size of each problem --------------------------
    big = [cases[case][-1] for case in order]
    y = np.arange(len(big))
    build = np.array([r["build_ms"] / 1e3 for r in big])
    solve = np.array([r["solve_ms"] / 1e3 for r in big])
    ax_b.barh(y, build, color=TEAL, height=0.6, label="build  (mesh, assemble, compile)")
    ax_b.barh(y, solve, left=build, color=GRAY, height=0.6, label="solve")
    for i, r in enumerate(big):
        frac = 100 * r["build_ms"] / max(r["build_ms"] + r["solve_ms"], 1e-9)
        ax_b.text(build[i] + solve[i] + 0.08, i, f"{frac:.0f}% build", va="center", fontsize=7)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f"{r['label']}\n{r['dofs']:,} dof" for r in big], fontsize=7)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Wall time at the largest size  [s]")
    ax_b.set_xlim(0, (build + solve).max() * 1.30)
    ax_b.set_ylim(len(big) - 0.4, -1.15)  # headroom above the first bar for the legend
    ax_b.grid(axis="y", visible=False)
    ax_b.legend(fontsize=7, loc="upper right", ncol=2)

    out = HERE / "fem_benchmark.png"
    fig.tight_layout()
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
