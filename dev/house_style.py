"""House plot style (design skill). Copy next to the plotting script and `import house_style as hs`.

    hs.apply("slide")                         # or "paper" (whitepapers, reports); call before plotting
    fig, ax = plt.subplots(figsize=hs.SIZE["slide"])
    ...
    hs.save(fig, "convergence_p1")            # convergence_p1.pdf (vector, for LaTeX) + .png preview

Pure matplotlib (no seaborn). Fonts: Frutiger when installed, else Noto Sans, the same choice as the
slide theme and the whitepaper template. Rules and reasons: references/figures.md.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager

# ---- Leon's picks (style picker, 2026-10-12) -------------------------------------------------------
# Round 1: boxed axes, Okabe-Ito with a marker per series, fine weights, viridis + RdBu fields.
# Round 2: markers filled with a white edge, dotted major grid, direct line labels (row above the panel when
# crowded), 9 / 8 pt text, natural few ticks with the axis limits snapped onto them (tidy()).
ORANGE, SKY, GREEN, VERMILLION, BLUE, PURPLE = "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#0072B2", "#CC79A7"
TEAL = "#0D9488"  # the slides' accent; not a plot series colour
INK, MUTED, FAINT = "#1A202C", "#94A3B8", "#EDF2F4"
# Series order: Okabe-Ito. Under simulated deuteranopia/protanopia/tritanopia (Machado 2009) the first four
# keep a min pairwise CIEDE2000 >= 12. In GRAYSCALE orange/sky (L* 71/70) and green/vermillion (58/54) are
# nearly equal, so every series also gets its own marker shape (MARKERS, same order): the figure must survive
# black-and-white printing. Use at most four series; five and six exist for exceptions.
CYCLE = [ORANGE, SKY, GREEN, VERMILLION, BLUE, PURPLE]
MARKERS = ["o", "s", "^", "D", "v", "P"]
DASHES = ["-", (0, (4, 1.5)), (0, (1, 1.2)), (0, (5, 1.5, 1, 1.5)), (0, (2, 1.5)), (0, (6, 2))]  # extra cue if needed
REFERENCE = dict(color=INK, lw=0.6, ls=(0, (4, 2)), marker="")  # exact / analytic / reference curves
CONTEXT = dict(color=MUTED, lw=0.6, marker="")  # de-emphasised series behind a highlighted one
SEQUENTIAL, DIVERGING = "viridis", "RdBu_r"  # cmaps; see figures.md for the alternatives

# ---- size (inches) at FINAL size: include the PDF at 100 %, never scale it in LaTeX --------------------
# slide: beamer 16:9 8pt, \textwidth = 401 pt, figure height ~0.66 \textheight; paper: A4, 2 cm margins.
SIZE = {
    "slide": (5.55, 2.2),
    "slide-half": (2.65, 2.2),
    "slide-tall": (5.55, 2.9),
    "paper": (6.69, 2.4),
    "paper-half": (3.25, 2.3),
    "paper-tall": (6.69, 3.6),
}

# Fraunhofer's licensed family names first; "Frutiger" is the URW version (Regular + Bold only, basic Latin: no
# Light, no Italic, no Greek or maths symbols), installed from ~/Downloads/frutiger.zip on 2026-10-12.
FRUTIGER = ["Frutiger LT 45 Light", "Frutiger LT Com", "Frutiger LT Std", "Frutiger Next LT W1G", "Frutiger"]
FALLBACK = "Noto Sans"


def font() -> str:
    """The house font family that is actually installed: Frutiger if present, else Noto Sans."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in FRUTIGER:
        if name in have:
            return name
    return FALLBACK if FALLBACK in have else "DejaVu Sans"


def apply(medium: str = "paper") -> str:
    """Set rcParams for `medium` ("slide" or "paper"). Returns the font family in use."""
    fam = font()
    faces = [f for f in font_manager.fontManager.ttflist if f.name == fam]
    it_face = fam if any(f.style == "italic" for f in faces) and fam != "Frutiger" else FALLBACK
    import logging  # maths glyphs missing from the house font fall back

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # to DejaVu (no light weight)
    text, small = (9.0, 8.0)  # labels 9 pt, ticks and legends 8 pt (a little above the 8 pt body)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [fam, FALLBACK, "DejaVu Sans"],
            "font.weight": 300 if fam == FALLBACK else 400,  # Noto Sans Light; Frutiger LT 45 is light already
            "font.size": text,
            "axes.labelsize": text,
            "axes.titlesize": text,
            "xtick.labelsize": small,
            "ytick.labelsize": small,
            "legend.fontsize": small,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            # maths in the same sans face as the text; variables and Greek in italic from the fallback face when the
            # house font has no italic or no Greek (the URW Frutiger has neither)
            "mathtext.fontset": "custom",
            "mathtext.rm": fam,
            "mathtext.it": f"{it_face}:italic",
            "mathtext.bf": f"{fam}:bold",
            "mathtext.sf": fam,
            "mathtext.fallback": "stixsans",
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.prop_cycle": mpl.cycler(color=CYCLE),  # markers: hs.mark(ax) (bars reject marker keys)
            # boxed: all four spines, ticks inward on every side
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.top": True,
            "ytick.right": True,
            "axes.linewidth": 0.45,
            "axes.axisbelow": True,
            "axes.xmargin": 0.0,
            "axes.ymargin": 0.0,  # limits hug the data; tidy() snaps them onto ticks
            "axes.grid": True,
            "grid.color": INK,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "grid.linestyle": ":",
            "lines.linewidth": 0.8,
            "lines.markersize": 3.0,
            "lines.markeredgewidth": 0.0,
            "lines.solid_capstyle": "round",
            "scatter.marker": "o",
            "patch.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.minor.width": 0.35,
            "ytick.minor.width": 0.35,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.3,
            "figure.figsize": SIZE[medium],
            "figure.dpi": 150,
            "figure.constrained_layout.use": True,
            "savefig.format": "pdf",
            "savefig.bbox": "standard",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,  # embed real fonts: text stays text in the PDF
            "image.cmap": SEQUENTIAL,
            "image.interpolation": "nearest",
        }
    )
    return fam


def save(fig, stem: str, *, preview: bool = True) -> list[str]:
    """Save `stem`.pdf (vector, what LaTeX includes) and a `stem`.png preview (for the chat).
    Runs tidy(fig) first, so ticks are natural and the axis limits sit on them."""
    tidy(fig)
    out = [f"{stem}.pdf"]
    fig.savefig(out[0])
    if preview:
        fig.savefig(f"{stem}.png", format="png", dpi=200)
        out.append(f"{stem}.png")
    return out


def mark(ax, *, every: float = 0.08):
    """Give each series line its marker shape (MARKERS, in plot order) so the figure survives grayscale
    printing. Markers are filled with a thin white edge that separates them from lines they cross.
    Reference and context lines (ink / muted) are skipped. Dense lines get markers spaced `every`
    (fraction of the axes diagonal, first one offset from the frame); lines with <= 30 points get a marker
    on every point. Call after plotting, before legend()/label_lines()."""
    skip = {mpl.colors.to_hex(INK), mpl.colors.to_hex(MUTED)}
    i = 0
    for l in ax.get_lines():
        if l.get_label().startswith("_") or mpl.colors.to_hex(l.get_color()) in skip:
            continue
        l.set(
            marker=MARKERS[i % len(MARKERS)],
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markevery=None if len(l.get_xdata()) <= 30 else (every / 2, every),
        )
        l.set_clip_on(False)  # a marker on the frame stays whole; limits cover the data
        i += 1


def scatter(ax, x, y, i: int = 0, *, label=None, s: float = 10, alpha: float = 0.95, **kw):
    """Scatter group i: its colour and marker shape, with the same thin white edge as the line markers."""
    kw.setdefault("clip_on", False)  # a point on the frame stays whole (limits cover the data)
    return ax.scatter(
        x,
        y,
        s=s,
        marker=MARKERS[i % len(MARKERS)],
        facecolors=CYCLE[i % len(CYCLE)],
        edgecolors="white",
        linewidths=0.4,
        alpha=alpha,
        label=label,
        **kw,
    )


def _natural_ticks(lo: float, hi: float, max_expand: float):
    """Natural tick candidates (steps 1, 2, 5 x 10^n; 3 to 7 ticks) for the window [lo, hi]. Returns
    (locator, tick values, aligned): the candidate with the FEWEST ticks whose outer ticks lie within
    `max_expand` of the span outside the window (then the window can be moved onto them), else minimal
    natural ticks inside the unchanged window."""
    cands = []
    eps = 1e-9 * (hi - lo)  # 1.2 / 6 must not round up past the step 0.2
    for nb in (3, 4, 5, 6):
        loc = mticker.MaxNLocator(nbins=nb, steps=[1, 2, 5, 10], min_n_ticks=3)
        t = np.array([float(f"{v:.10g}") for v in loc.tick_values(lo + eps, hi - eps)])
        cands.append((len(t), ((lo - t[0]) + (t[-1] - hi)) / (hi - lo), loc, t))
    ok = [c for c in cands if c[1] <= max_expand + 1e-9 and c[0] <= 7]  # never more than 7 ticks
    if ok:
        _, _, _, t = min(ok, key=lambda c: (c[0], c[1]))
        return mticker.FixedLocator(t), t, True
    loc = mticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10], min_n_ticks=3)
    return loc, loc.tick_values(lo, hi), False


def _fmt(v, _):
    """Plain natural numbers: 0, 0.2, 1, 1.5, -0.4 (typographic minus)."""
    if abs(v) < 1e-12:
        return "0"
    return f"{v:g}".replace("-", "\u2212")


def tidy(target, *, max_expand: float = 0.2, max_expand_log: float = 0.25):
    """Natural, minimal ticks, and axis limits that sit ON the outer ticks, so the tick marks run to the
    frame on all four sides (boxed axes mirror them). The window is moved onto the next natural tick only
    when that costs at most `max_expand` of its span (`max_expand_log` in decades for log axes, whose limits
    go to whole decades); otherwise the limits stay and the ticks sit inside. Axes with their own locator,
    colorbars and axes with the axis off are left alone. save() calls this."""
    axes = [target] if hasattr(target, "get_xlim") else list(target.axes)
    for ax in axes:
        if ax.get_label() == "<colorbar>" or not ax.axison:
            continue
        for axis, get, setlim, auto in (
            (ax.xaxis, ax.get_xlim, ax.set_xlim, ax.get_autoscalex_on()),
            (ax.yaxis, ax.get_ylim, ax.set_ylim, ax.get_autoscaley_on()),
        ):
            a, b = get()
            lo, hi = min(a, b), max(a, b)
            if not hi > lo:
                continue
            kind = type(axis.get_major_locator())
            if axis.get_scale() == "log" and kind is mticker.LogLocator and lo > 0:
                llo, lhi = np.log10(lo), np.log10(hi)
                flo, fhi = np.floor(llo + 1e-9), np.ceil(lhi - 1e-9)
                if fhi > flo and ((llo - flo) + (fhi - lhi)) / (lhi - llo) <= max_expand_log:
                    new = (10.0**flo, 10.0**fhi)
                    setlim(new[::-1] if a > b else new)
                elif auto:  # not on decades: a little air so markers are not cut
                    pad = 0.03 * (lhi - llo)
                    new = (10.0 ** (llo - pad), 10.0 ** (lhi + pad))
                    setlim(new[::-1] if a > b else new)
                continue
            if axis.get_scale() != "linear" or kind is not mticker.AutoLocator:
                continue
            room = axis is ax.xaxis and getattr(ax, "_hs_room", False)
            loc, t, aligned = _natural_ticks(lo, hi, 1e-6 if room else max_expand)
            if room:
                auto = False
            if aligned:
                setlim((t[-1], t[0]) if a > b else (t[0], t[-1]))
            elif auto:  # not on ticks: a little air so markers are not cut
                pad = 0.03 * (hi - lo)
                setlim((hi + pad, lo - pad) if a > b else (lo - pad, hi + pad))
            axis.set_major_locator(loc)
            if 1e-3 <= max(abs(t[0]), abs(t[-1])) < 1e5:
                axis.set_major_formatter(mticker.FuncFormatter(_fmt))


def hist(ax, data, bins, i: int = 0, *, label=None, alpha: float = 0.45, **kw):
    """Histogram for series i: a light fill and a full-colour outline with the series' line style
    (DASHES[i]) as ONE shape, so overlapping histograms stay apart in grayscale and the legend shows the
    outline style too."""
    c = mpl.colors.to_rgb(CYCLE[i % len(CYCLE)])
    return ax.hist(
        data,
        bins=bins,
        histtype="stepfilled",
        facecolor=(*c, alpha),
        edgecolor=c,
        lw=0.8,
        ls=DASHES[i % len(DASHES)],
        label=label,
        **kw,
    )


def label_lines(ax, lines=None, *, dx: float = 4.0, max_shift: float = 2.0) -> bool:
    """Name each line at its right end, inside the frame: a short sample of the line (its colour, line style
    and marker) followed by the name in dark ink, so the label still identifies the line in black and white.
    The x-limit is widened to the next natural tick that makes room. Labels keep a minimum vertical gap and
    the stack stays centred on the line ends. Returns False and draws nothing when the ends are so crowded
    that a label would sit more than `max_shift` label heights from its line, or the x-axis is not linear:
    use a row above the panel then (legend() does this for you)."""
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea

    lines = lines or [l for l in ax.get_lines() if not l.get_label().startswith("_")]
    lines = [
        l
        for l in lines
        if np.isfinite(np.asarray(l.get_xdata(), float)).any() and np.isfinite(np.asarray(l.get_ydata(), float)).any()
    ]  # dummy legend lines have no points
    if not lines or ax.get_xscale() != "linear":
        return False
    ax.figure.canvas.draw()  # final limits/layout before converting to pixels
    ends = []
    for l in lines:
        x, y = np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)
        ok = np.isfinite(x) & np.isfinite(y)
        ends.append((ax.transData.transform((x[ok][-1], y[ok][-1]))[1], l, x[ok][-1]))
    ends.sort(key=lambda e: e[0])
    xs = [e[2] for e in ends]
    x0_, x1_ = ax.get_xlim()
    if max(xs) - min(xs) > 0.1 * abs(x1_ - x0_):  # lines end at different places: a label would sit
        return False  # among the other curves
    fs = mpl.rcParams["legend.fontsize"]
    gap = fs * 1.3 * ax.figure.dpi / 72
    target = np.array([e[0] for e in ends])
    placed = target.copy()
    for k in range(1, len(placed)):  # push up to keep the gap ...
        placed[k] = max(placed[k], placed[k - 1] + gap)
    placed += np.mean(target) - np.mean(placed)  # ... then re-centre the stack on the ends
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    if np.max(np.abs(placed - target)) > max_shift * gap:
        return False
    inv = ax.transData.inverted()
    boxes = []
    for ypix, (_, l, x) in zip(placed, ends):
        w, h = 1.3 * fs, fs  # sample size in points
        da = DrawingArea(w, h, 0, 0)
        da.add_artist(Line2D([0, w], [h / 2, h / 2], color=l.get_color(), lw=l.get_linewidth(), ls=l.get_linestyle()))
        if l.get_marker() not in ("", "None", None, " "):
            da.add_artist(
                Line2D(
                    [w / 2],
                    [h / 2],
                    ls="",
                    marker=l.get_marker(),
                    color=l.get_color(),
                    markersize=l.get_markersize(),
                    markeredgecolor=l.get_markeredgecolor(),
                    markeredgewidth=l.get_markeredgewidth(),
                )
            )
        box = HPacker(
            children=[da, TextArea(l.get_label(), textprops=dict(color=INK, fontsize=fs))], align="center", pad=0, sep=3
        )
        ab = AnnotationBbox(
            box,
            (x, inv.transform((0, ypix))[1]),
            xybox=(dx, 0),
            xycoords="data",
            boxcoords="offset points",
            box_alignment=(0, 0.5),
            frameon=False,
            pad=0,
            annotation_clip=False,
        )
        ax.add_artist(ab)
        boxes.append(ab)
    # room: widen the x-limit so the widest label sits inside the frame, with the lines ending short of it
    r = ax.figure.canvas.get_renderer()
    box_ax = ax.get_window_extent(r)
    widest = max(b.get_window_extent(r).width for b in boxes)
    if max(b.get_window_extent(r).x1 for b in boxes) + 18 > box_ax.x1:
        x0, x1 = ax.get_xlim()
        x_end = max(e[2] for e in ends)
        q = (widest + 18) / box_ax.width
        x_need = x0 + (x_end - x0) / (1 - q)
        # the new right edge is the smallest natural tick that still leaves the room, so ticks meet the
        # frame; when that tick would leave more than 12 % of the window empty, keep the exact room instead
        tops = []
        for nb in (3, 4, 5, 6):
            t = mticker.MaxNLocator(nbins=nb, steps=[1, 2, 5, 10], min_n_ticks=3).tick_values(x0, x_need)
            top = float(f"{t[-1]:.10g}")
            if top >= x_need - 1e-12 and len(t) <= 7:
                tops.append((top, len(t)))
        top = min(tops)[0] if tops else None
        ax.set_xlim(x0, top if top is not None and (top - x_need) / (top - x0) <= 0.12 else x_need)
        ax._hs_room = True  # tidy() must not re-snap the window sized for the labels
    return True


def legend(ax, **kw) -> str:
    """Name the series of one panel the house way. Up to four lines, ends not crowded: direct labels at the
    line ends ("direct"). Crowded, or more lines: a frameless row above the panel ("above"). Scatter groups,
    histograms and bars: a frameless legend in the emptiest corner ("corner"). Returns which was used."""
    from matplotlib.lines import Line2D

    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return "none"
    if all(isinstance(h, Line2D) for h in handles):
        if len(handles) <= 4 and label_lines(ax):
            return "direct"
        ax.figure.canvas.draw()
        r = ax.figure.canvas.get_renderer()
        width = ax.get_window_extent(r).width
        k = min(len(labels), 5)
        while True:  # as many columns as fit over the panel, else more rows
            lg = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=k, borderaxespad=0, **kw)
            if k == 1 or lg.get_window_extent(r).width <= width * 1.02:
                break
            lg.remove()
            k -= 1
        return "above"
    ax.legend(handles, labels, loc="best", **kw)
    return "corner"


def band(
    ax, x, lo, mid, hi, *, color=ORANGE, marker=None, label=None, smooth: bool = True, log: bool = False, n: int = 400
):
    """A thin centre line with its spread band (alpha 0.28). `smooth` resamples with a shape-preserving
    PCHIP interpolant (in log10 space if `log`), for statistics known only at discrete points."""
    x, lo, mid, hi = (np.asarray(a, float) for a in (x, lo, mid, hi))
    if smooth and len(x) > 2:
        from scipy.interpolate import PchipInterpolator

        xf = np.linspace(x.min(), x.max(), n)
        f = (lambda y: 10 ** PchipInterpolator(x, np.log10(y))(xf)) if log else (lambda y: PchipInterpolator(x, y)(xf))
        x, lo, mid, hi = xf, f(lo), f(mid), f(hi)
    ax.fill_between(x, lo, hi, color=color, alpha=0.28, lw=0)
    return ax.plot(x, mid, color=color, lw=0.8, marker=marker or "", markevery=0.08, label=label)[0]


def spread(ax, x, runs, i: int = 0, *, label=None, log: bool = False, smooth: bool = True, n: int = 400):
    """Median of repeated runs with two nested bands (chosen 2026-10-12): the interquartile range dark,
    minimum to maximum light, the median a thin line with series i's marker. `runs` is runs x len(x).
    `smooth` resamples with a shape-preserving PCHIP interpolant (in log10 space if `log`) for statistics
    known only at discrete points. Returns the median line."""
    x = np.asarray(x, float)
    q = np.percentile(np.asarray(runs, float), [0, 25, 50, 75, 100], axis=0)
    if smooth and len(x) > 2:
        from scipy.interpolate import PchipInterpolator

        xf = np.linspace(x.min(), x.max(), n)
        f = (lambda y: 10 ** PchipInterpolator(x, np.log10(y))(xf)) if log else (lambda y: PchipInterpolator(x, y)(xf))
        x, q = xf, np.array([f(r) for r in q])
    c = CYCLE[i % len(CYCLE)]
    ax.fill_between(x, q[0], q[4], color=c, alpha=0.13, lw=0)
    ax.fill_between(x, q[1], q[3], color=c, alpha=0.30, lw=0)
    return ax.plot(
        x,
        q[2],
        color=c,
        lw=0.8,
        marker=MARKERS[i % len(MARKERS)],
        markevery=(0.04, 0.1),
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.6,
        label=label,
    )[0]


def slope_triangle(ax, x0: float, y0: float, order: float, *, width: float = 0.25):
    """Reference slope `order` on log-log axes: a small right triangle with its hypotenuse from (x0, y0)
    rising over `width` decades in x, the order written beside its vertical leg. Use it on convergence
    plots instead of a fitted line; place it in empty space below/right of the data."""
    x1 = x0 * 10**width
    y1 = y0 * (x1 / x0) ** order
    ax.plot([x0, x1, x1, x0], [y0, y1, y0, y0], color=INK, lw=0.5, marker="", solid_joinstyle="miter")
    ax.annotate(
        f"{order:g}",
        (x1, (y0 * y1) ** 0.5),
        xytext=(3, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=INK,
        fontsize=mpl.rcParams["legend.fontsize"],
    )


def slope_triangle_near(ax, line, order: float, *, width: float = 0.25, side: str = "auto") -> str:
    """Draw the reference-slope triangle for `line` (a curve on log-log axes that converges with `order`)
    NEXT TO that curve: its hypotenuse parallel to the curve, a small gap away, the order written beside
    the vertical leg. The triangle goes into free space: candidate places along the curve, below and above
    it, are tested against every other line (and earlier triangles) and the first clear one is used.
    side: "below", "above" or "auto" (below first). Returns the side used. Call after all lines are
    plotted, before tidy()/save()."""
    from matplotlib.path import Path

    x = np.asarray(line.get_xdata(), float)
    y = np.asarray(line.get_ydata(), float)
    ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    order_ = np.argsort(x[ok])
    lx, ly = np.log10(x[ok][order_]), np.log10(y[ok][order_])
    span = lx[-1] - lx[0]
    w = min(width, 0.4 * span)
    ax.figure.canvas.draw()
    T = ax.transData
    box = ax.get_window_extent()

    def px(a, b):
        return T.transform(np.c_[10.0 ** np.asarray(a), 10.0 ** np.asarray(b)])

    obstacles = []
    for l in ax.get_lines():
        xd, yd = np.asarray(l.get_xdata(), float), np.asarray(l.get_ydata(), float)
        m = np.isfinite(xd) & np.isfinite(yd) & (xd > 0) & (yd > 0)
        if m.sum() < 2:
            continue
        pts = T.transform(np.c_[xd[m], yd[m]])
        ts = np.linspace(0, 1, 14)[:, None]
        obstacles.append(np.vstack([pts[i] + (pts[i + 1] - pts[i]) * ts for i in range(len(pts) - 1)]))
    obstacles = np.vstack(obstacles) if obstacles else np.empty((0, 2))

    def bad_points(P, pad=5.0):
        """Number of obstacle samples inside the polygon P or within `pad` px of its edges."""
        inside = Path(P).contains_points(obstacles)
        d = np.full(len(obstacles), np.inf)
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            ab = b - a
            t = np.clip(((obstacles - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
            d = np.minimum(d, np.linalg.norm(obstacles - (a + t[:, None] * ab), axis=1))
        return int((inside | (d < pad)).sum())

    sides = ("below", "above") if side == "auto" else (side,)
    best = None
    for f in (0.62, 0.42, 0.78, 0.25, 0.9, 0.1):
        for sd in sides:
            for gap in (0.2, 0.3, 0.42):
                a0 = lx[0] + f * (span - w)
                a1 = a0 + w
                b0 = np.interp(a0, lx, ly) + (-gap if sd == "below" else gap)
                b1 = b0 + order * w
                poly = [(a0, b0), (a1, b1), (a1, b0)] if sd == "below" else [(a0, b0), (a1, b1), (a0, b1)]
                P = px([q[0] for q in poly], [q[1] for q in poly])
                lab = P[:, 1].mean()
                xr = P[:, 0].max() if sd == "below" else P[:, 0].min()
                R = (
                    np.array([[xr + 2, lab - 7], [xr + 14, lab - 7], [xr + 14, lab + 7], [xr + 2, lab + 7]])
                    if sd == "below"
                    else np.array([[xr - 14, lab - 7], [xr - 2, lab - 7], [xr - 2, lab + 7], [xr - 14, lab + 7]])
                )
                inside_box = all(
                    box.x0 + 3 < q[0] < box.x1 - 3 and box.y0 + 3 < q[1] < box.y1 - 3 for q in np.vstack([P, R])
                )
                n = bad_points(P) + bad_points(R, pad=1.0) + (0 if inside_box else 10_000)
                if best is None or n < best[0]:
                    best = (n, poly, sd)
                if n == 0:
                    break
            if best[0] == 0:
                break
        if best[0] == 0:
            break
    _, poly, sd = best
    ax.plot(
        [10.0 ** q[0] for q in poly] + [10.0 ** poly[0][0]],
        [10.0 ** q[1] for q in poly] + [10.0 ** poly[0][1]],
        color=INK,
        lw=0.5,
        marker="",
        solid_joinstyle="miter",
        label="_triangle",
    )
    mid_y = 10.0 ** np.mean([q[1] for q in poly])
    if sd == "below":
        ax.annotate(
            f"{order:g}",
            (10.0 ** poly[1][0], mid_y),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=INK,
            fontsize=mpl.rcParams["legend.fontsize"],
        )
    else:
        ax.annotate(
            f"{order:g}",
            (10.0 ** poly[0][0], mid_y),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            color=INK,
            fontsize=mpl.rcParams["legend.fontsize"],
        )
    return sd


def order_triangles(ax, orders, *, side: str = "auto"):
    """slope_triangle_near for each series line of `ax`, in plot order, with the given convergence orders
    (reference and context lines, ink / muted, are skipped)."""
    skip = {mpl.colors.to_hex(INK), mpl.colors.to_hex(MUTED)}
    series = [
        l for l in ax.get_lines() if not l.get_label().startswith("_") and mpl.colors.to_hex(l.get_color()) not in skip
    ]
    return [slope_triangle_near(ax, l, p, side=side) for l, p in zip(series, orders)]


def converged_region(ax, tol: float, counts=None, *, ymin=None):
    """Shade the region below the stopping tolerance of a residual plot (ink at 8 %), write "converged"
    at its left end, and, when `counts` is given as {x_end: "12 it", ...}, write each count at the point
    where its curve enters the region: right and above the point; when that would leave the plot frame,
    inside the shaded region below and left of it (never outside the plot). Call after plotting and
    after setting the y-limits (ymin defaults to the lower y-limit)."""
    lo = ax.get_ylim()[0] if ymin is None else ymin
    ax.axhspan(lo, tol, color=INK, alpha=0.08, lw=0)
    mid = (lo * tol) ** 0.5 if ax.get_yscale() == "log" else 0.5 * (lo + tol)
    ax.annotate("converged", (0.02, mid), xycoords=("axes fraction", "data"), ha="left", va="center", color=INK)
    texts = [
        ax.annotate(text, (x, tol), xytext=(5, 4), textcoords="offset points", ha="left", va="bottom", color=INK)
        for x, text in (counts or {}).items()
    ]
    if texts:
        ax.figure.canvas.draw()
        r = ax.figure.canvas.get_renderer()
        edge = ax.get_window_extent(r).x1
        for t in texts:
            if t.get_window_extent(r).x1 > edge - 3:
                t.set_ha("right")
                t.set_va("top")
                t.xyann = (-4, -4)


def grouped_barh(
    ax, categories, values, series, *, sort_by: int | None = -1, annotate: int | None = -1, fmt: str = "{:g}", err=None
):
    """Horizontal grouped bars (chosen 2026-10-12). `values` is categories x series. One hatched bar per
    series (colour + hatch, so series stay apart in black and white), categories sorted by the series
    `sort_by` (largest first, drawn from the bottom), the value of series `annotate` written at the end of
    each group's bar. Axis starts at zero; no grid. `sort_by=None` keeps the categories in the given order,
    the first at the top (parameter sweeps). `err` (categories x series, same units) draws a thin whisker
    of +-err on each bar."""
    values = np.asarray(values, float)
    order = np.argsort(values[:, sort_by])[::-1] if sort_by is not None else np.arange(len(values))[::-1]
    n = values.shape[1]
    h = 0.8 / n
    hatches = ["", "///", "xxx", "...", "\\\\", "|||"]
    ax.grid(False)
    for j in range(n):
        ax.barh(
            np.arange(len(order)) + (j - (n - 1) / 2) * h,
            values[order, j],
            height=h,
            color=CYCLE[j % len(CYCLE)],
            edgecolor=INK,
            linewidth=0.4,
            hatch=hatches[j % len(hatches)],
            label=series[j],
        )
    if err is not None:
        err = np.asarray(err, float)
        for j in range(n):
            ax.errorbar(
                values[order, j],
                np.arange(len(order)) + (j - (n - 1) / 2) * h,
                xerr=err[order, j],
                fmt="none",
                ecolor=INK,
                elinewidth=0.5,
                capsize=1.5,
                capthick=0.5,
            )
    if annotate is not None:
        top = values.max()
        for k, o in enumerate(order):
            ax.text(
                values[o, annotate] + (0 if err is None else err[o, annotate]) + 0.012 * top,
                k + (annotate % n - (n - 1) / 2) * h,
                fmt.format(values[o, annotate]),
                va="center",
                fontsize=mpl.rcParams["legend.fontsize"],
                color=INK,
            )
    ax.set_yticks(np.arange(len(order)), [categories[o] for o in order])
    ax.set_xlim(
        0, 1.15 * (values + (0 if err is None else err)).max()
    )  # room for the written value; tidy() moves it onto a tick
    ax.tick_params(axis="y", length=0)


# ---- heatmaps -------------------------------------------------------------------------------------
_NICE = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)


def nice(v: float, direction: str = "up") -> float:
    """Round |v| to a nice number (up for a vmax, down for a positive vmin); sign kept."""
    if v == 0:
        return 0.0
    s, a = np.sign(v), abs(v)
    m = 10 ** np.floor(np.log10(a))
    steps = _NICE if direction == "up" else _NICE[::-1]
    for k in steps:
        if (a / m <= k) if direction == "up" else (a / m >= k):
            return float(s * k * m)
    return float(s * (10 if direction == "up" else 1) * m)


def colorbar_ticks(vmin: float, vmax: float):
    """~5 nice ticks with BOTH endpoints on the tick grid. Returns (ticks, vmin_snapped, vmax_snapped);
    pass the snapped range to the image AND the colorbar."""
    raw = (vmax - vmin) / 4.0
    m = 10.0 ** np.floor(np.log10(raw))
    step = next((k * m for k in (1.0, 1.25, 1.5, 2.0, 2.5, 5.0, 10.0) if raw / m <= k + 1e-9), 10 * m)
    lo, hi = np.floor(vmin / step + 1e-9) * step, np.ceil(vmax / step - 1e-9) * step
    return [round(lo + i * step, 10) for i in range(int(round((hi - lo) / step)) + 1)], float(lo), float(hi)


def heatmap(
    ax,
    data,
    *,
    extent=None,
    cmap=None,
    vmin=None,
    vmax=None,
    label=None,
    rasterized=True,
    contours: int = 0,
    axes: bool | None = None,
):
    """imshow with no frame, a thin colorbar exactly as tall as the image, round ticks on one grid,
    symmetric range + DIVERGING cmap when the data cross zero, 0-anchored when a non-negative range is wide.
    Pass vmin/vmax explicitly to share one scale across panels (pooled 95th percentile, see figures.md).
    contours=n overlays n thin ink contour levels per sign (negative levels dashed, zero omitted): use it for
    signed fields that may be printed in black and white, where a diverging map loses the sign.
    axes: boxed coordinate axes with natural ticks (chosen 2026-10-12; label them "$x$ [m]"); default True
    when `extent` is given, False for a bare image."""
    show_axes = (extent is not None) if axes is None else axes
    d = np.asarray(data, float)
    if vmin is None or vmax is None:
        lo, hi = np.nanmin(d), np.nanmax(d)
        if lo < 0 < hi:
            v = nice(max(-lo, hi))
            vmin, vmax = -v, v
        else:
            vmin = 0.0 if (lo >= 0 and lo < 0.4 * hi) else nice(lo, "down")
            vmax = nice(hi, "up")
    cmap = cmap or (DIVERGING if vmin < 0 < vmax else SEQUENTIAL)
    ticks, vmin, vmax = colorbar_ticks(vmin, vmax)
    im = ax.imshow(d, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", extent=extent, rasterized=rasterized)
    if show_axes:
        ax.grid(False)
    else:
        ax.set_axis_off()
    if contours:
        ys, xs = np.arange(d.shape[0]), np.arange(d.shape[1])
        if extent is not None:
            xs = np.linspace(extent[0], extent[1], d.shape[1])
            ys = np.linspace(extent[2], extent[3], d.shape[0])
        v = max(abs(vmin), abs(vmax))
        lv = np.linspace(-v, v, 2 * contours + 1)
        lv = lv[np.abs(lv) > 1e-12 * v]
        ax.contour(xs, ys, d, levels=lv, colors=INK, linewidths=0.4, linestyles=["--" if l < 0 else "-" for l in lv])
    if extent is None:
        ax.set_box_aspect(d.shape[0] / d.shape[1])
    else:
        ax.set_box_aspect(abs(extent[3] - extent[2]) / abs(extent[1] - extent[0]))
    cax = ax.inset_axes([1.025, 0.0, 0.035, 1.0])  # same height as the (aspect-pinned) image; the
    cb = ax.figure.colorbar(im, cax=cax)  # layout engine counts it, so labels never collide
    cax.grid(False)
    cb.outline.set_visible(False)
    cb.minorticks_off()
    cb.ax.tick_params(length=0, pad=2)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    tls = cb.ax.get_yticklabels()
    for t in tls:
        t.set_fontstyle("italic")
        t.set_color(INK)
    if tls:
        tls[0].set_verticalalignment("bottom")
        tls[-1].set_verticalalignment("top")
    if label:
        cb.set_label(label)
    return im, cb


def trifield(ax, tri, z, *, contours: int = 0, xlabel="$x$ [m]", ylabel="$y$ [m]", title=None):
    """A nodal field on a triangulation (matplotlib.tri.Triangulation), drawn the house way (chosen
    2026-10-12): Gouraud shading, symmetric limits on a natural step with the diverging map when the field
    changes sign (otherwise sequential from 0), boxed coordinate axes with natural ticks and units, a thin
    colour bar as tall as the field with ticks every second level. contours=n adds n thin ink contour lines
    per sign (negative dashed): use it when a signed field may be printed in black and white."""
    z = np.asarray(z, float)
    signed = z.min() < 0 < z.max()
    if signed:
        step = nice(float(np.abs(z).max()) / 4)
        levels = np.arange(-4, 5) * step
        vmin, vmax, cmap, ticks = -4 * step, 4 * step, DIVERGING, list(levels[::2])
    else:
        ticks, vmin, vmax = colorbar_ticks(
            0.0 if z.min() >= 0 and z.min() < 0.4 * z.max() else nice(z.min(), "down"), nice(z.max(), "up")
        )
        levels, cmap = np.linspace(vmin, vmax, 9), SEQUENTIAL
    im = ax.tripcolor(tri, z, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    if contours:
        lv = np.linspace(vmin, vmax, 2 * contours + 1)
        lv = lv[np.abs(lv) > 1e-12 * max(abs(vmin), abs(vmax))] if signed else lv[1:-1]
        ax.tricontour(tri, z, levels=lv, colors=INK, linewidths=0.4, linestyles=["--" if l < 0 else "-" for l in lv])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.margins(0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.text(0, 1.03, title, transform=ax.transAxes, fontsize=mpl.rcParams["legend.fontsize"])
    cax = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
    cb = ax.figure.colorbar(im, cax=cax)
    cax.grid(False)
    cb.outline.set_visible(False)
    cb.minorticks_off()
    cb.ax.tick_params(length=0, pad=2)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    for t in cb.ax.get_yticklabels():
        t.set_fontstyle("italic")
        t.set_color(INK)
    return im, cb


MAGNITUDE = mpl.colors.LinearSegmentedColormap.from_list(  # cividis reversed (chosen 2026-10-12): dark = large;
    "magnitude", mpl.colormaps["cividis_r"](np.linspace(0.15, 1.0, 256))
)  # palest end cut: small entries stay visible


def spy(ax, A, *, blocks=None, names=None, cmap=None, s: float = 4.5):
    """Sparsity pattern coloured by magnitude (chosen 2026-10-12): a square per nonzero, coloured by
    log10 |a_ij| (MAGNITUDE = reversed cividis, dark = large), row 0 at the top. blocks = block boundaries (e.g. [100] for
    a 2 x 2 saddle-point system) drawn as dashed lines; names = {(col, row): "$K$", ...} written at those
    positions; the number of nonzeros and a colour bar are added."""
    A = A.tocoo()
    nr, nc = A.shape
    ax.grid(False)
    sc = ax.scatter(
        A.col + 0.5,
        A.row + 0.5,
        c=np.log10(np.abs(A.data)),
        s=s,
        marker="s",
        cmap=cmap or MAGNITUDE,
        linewidths=0,
        clip_on=False,
    )
    ax.set_xlim(0, nc)
    ax.set_ylim(nr, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    for b in blocks or []:
        ax.axhline(b, color=MUTED, lw=0.6, ls=(0, (4, 2)))
        ax.axvline(b, color=MUTED, lw=0.6, ls=(0, (4, 2)))
    for (xx, yy), t in (names or {}).items():
        ax.text(
            xx,
            yy,
            t,
            ha="center",
            va="center",
            fontsize=mpl.rcParams["legend.fontsize"] + 2,
            color=INK,
            bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
        )
    ax.text(
        0.03 * nc, 0.42 * nr, f"nnz = {A.nnz}", ha="left", va="center", fontsize=mpl.rcParams["legend.fontsize"], color=INK
    )
    cax = ax.inset_axes([1.03, 0, 0.04, 1])
    cb = ax.figure.colorbar(sc, cax=cax)
    cax.grid(False)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=2)
    cb.set_label(r"$\log_{10}|a_{ij}|$")
    return sc, cb


def spectrum(ax, spectra, labels=None, *, density: bool = True, s: float = 6, bw: float = 0.18, grid: int = 200):
    """Eigenvalues in the complex plane, all in one panel (chosen 2026-10-12): one marker shape and colour
    per spectrum (hs.scatter), and, when `density`, tinted nested regions of each spectrum's kernel
    density estimate underneath, darker where it is denser, in the spectrum's own colour. `spectra` is a
    list of complex arrays; call hs.legend(ax) afterwards."""
    from scipy.stats import gaussian_kde

    spectra = [np.asarray(e, complex) for e in spectra]
    allv = np.concatenate(spectra)
    padx, pady = 0.08 * np.ptp(allv.real) + 1e-12, 0.12 * np.ptp(allv.imag) + 1e-12
    gx = np.linspace(allv.real.min() - padx, allv.real.max() + padx, grid)
    gy = np.linspace(allv.imag.min() - pady, allv.imag.max() + pady, grid)
    GX, GY = np.meshgrid(gx, gy)
    for i, ev in enumerate(spectra):
        if density and len(ev) > 5:
            z = gaussian_kde(np.vstack([ev.real, ev.imag]), bw_method=bw)(np.vstack([GX.ravel(), GY.ravel()])).reshape(
                GX.shape
            )
            z = z / z.max()
            rgb = mpl.colors.to_rgb(CYCLE[i % len(CYCLE)])
            ax.contourf(
                GX,
                GY,
                z,
                levels=[0.015, 0.05, 0.15, 0.4, 1.01],
                colors=[(*rgb, a) for a in (0.16, 0.28, 0.42, 0.58)],
                zorder=1,
            )
        scatter(ax, ev.real, ev.imag, i, s=s, label=None if labels is None else labels[i], zorder=3)
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    ax.set_xlabel(r"Re $\lambda$")
    ax.set_ylabel(r"Im $\lambda$")


def render_colorbar(raw, out, *, label="$T$", vmin=0.0, vmax=1.0, ticks=None, width: float = 3.25, cmap=None):
    """Crop a Blender render (RGBA on a transparent film) to its content on white and add the house
    colour bar, as a PNG at final size (`width` inches, 200 dpi). The colour map must be the one used for
    the vertex colours (default: the sequential map)."""
    import matplotlib.image as mpimg

    img = mpimg.imread(raw)
    alpha = None
    if img.shape[2] == 4:
        a = img[..., 3:4]
        alpha = img[..., 3]  # the object is opaque, its soft shadow is not
        img = img[..., :3] * a + (1 - a)
    img = img[..., :3]
    ys, xs = np.where(alpha > 0.5) if alpha is not None else np.where(img.min(axis=2) < 0.985)
    pad = 14
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
    crop = img[y0:y1, x0:x1]
    fig, ax = plt.subplots(figsize=(width, width * crop.shape[0] / crop.shape[1] * 0.88 + 0.1))
    ax.imshow(crop)
    ax.set_axis_off()
    cax = ax.inset_axes([1.01, 0.14, 0.035, 0.72])
    cax.grid(False)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(vmin, vmax), mpl.colormaps[cmap or SEQUENTIAL]), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=2)
    cb.set_ticks(ticks if ticks is not None else [vmin, 0.5 * (vmin + vmax), vmax])
    cb.ax.set_title(label, fontsize=mpl.rcParams["legend.fontsize"], pad=4)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _speed_bar(ax, mappable, vmax, label):
    ticks, _, _ = colorbar_ticks(0.0, vmax)
    cax = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
    cb = ax.figure.colorbar(mappable, cax=cax)
    cax.grid(False)
    cb.outline.set_visible(False)
    cb.minorticks_off()
    cb.ax.tick_params(length=0, pad=2)
    cb.set_ticks(ticks)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    for t in cb.ax.get_yticklabels():
        t.set_fontstyle("italic")
        t.set_color(INK)
    cb.ax.set_title(label, fontsize=mpl.rcParams["legend.fontsize"], pad=4)
    return cb


def vectors2d(
    ax,
    x,
    y,
    u,
    v,
    *,
    mask=None,
    vmax=None,
    density: float = 1.0,
    label=r"$|\mathbf{u}|$",
    xlabel="$x$ [m]",
    ylabel="$y$ [m]",
):
    """A 2-D vector field (chosen 2026-10-12): the speed as an image (sequential map, thin colour bar with
    `label`) with thin ink streamlines and direction arrows over it. x, y are 1-D grid coordinates, u, v
    arrays of shape (len(y), len(x)); `mask` (True where there is no fluid, e.g. inside a body) is left
    blank: draw the body on top yourself. Boxed coordinate axes with natural ticks and units."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    m = np.zeros(u.shape, bool) if mask is None else np.asarray(mask, bool)
    speed = np.ma.array(np.hypot(u, v), mask=m)
    vmax = nice(float(speed.max())) if vmax is None else vmax
    im = ax.imshow(
        speed,
        extent=(x[0], x[-1], y[0], y[-1]),
        origin="lower",
        cmap=SEQUENTIAL,
        vmin=0,
        vmax=vmax,
        rasterized=True,
        interpolation="bilinear",
    )
    ax.streamplot(
        x,
        y,
        np.ma.array(u, mask=m),
        np.ma.array(v, mask=m),
        color=INK,
        linewidth=0.5,
        density=density,
        arrowsize=0.6,
        arrowstyle="->",
    )
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _speed_bar(ax, im, vmax, label)
    return im


def deformed(
    ax, nodes0, nodes1, triangles, outline=None, *, vmax=None, label=r"$|\mathbf{u}|$", xlabel="$x$ [m]", ylabel="$y$ [m]"
):
    """A deformed shape (chosen 2026-10-12): the deformed mesh coloured by the displacement magnitude
    (Gouraud, mesh edges faint), the undeformed outline dashed in ink on top. nodes0/nodes1: (n, 2)
    undeformed and deformed positions; triangles: (m, 3) indices; outline: (k, 2) closed polygon of the
    undeformed shape (default: its convex hull is not drawn). Plot true scale unless the caption names
    a factor."""
    import matplotlib.tri as mtri

    nodes0, nodes1 = np.asarray(nodes0, float), np.asarray(nodes1, float)
    disp = np.linalg.norm(nodes1 - nodes0, axis=1)
    vmax = nice(float(disp.max())) if vmax is None else vmax
    tr = mtri.Triangulation(nodes1[:, 0], nodes1[:, 1], triangles)
    im = ax.tripcolor(tr, disp, shading="gouraud", cmap=SEQUENTIAL, vmin=0, vmax=vmax, rasterized=True)
    ax.triplot(tr, color=INK, lw=0.25, alpha=0.5)
    if outline is not None:
        ax.plot(np.asarray(outline)[:, 0], np.asarray(outline)[:, 1], **REFERENCE)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _speed_bar(ax, im, vmax, label)
    return im


def support(ax, p0, p1, *, side: str = "left", width: float = 0.16):
    """A fixed (Dirichlet) boundary for set-up figures: a hatched strip of `width` outside the straight
    boundary from p0 to p1, with a heavier line on the boundary. side: "left", "right", "bottom", "top"."""
    from matplotlib.patches import Rectangle

    (x0, y0), (x1, y1) = p0, p1
    r = {
        "left": (x0 - width, y0, width, y1 - y0),
        "right": (x0, y0, width, y1 - y0),
        "bottom": (x0, y0 - width, x1 - x0, width),
        "top": (x0, y0, x1 - x0, width),
    }[side]
    ax.add_patch(Rectangle(r[:2], r[2], r[3], fill=False, hatch="////", edgecolor=INK, lw=0.5))
    ax.plot([x0, x1], [y0, y1], color=INK, lw=1.0)


def load_arrows(ax, x0: float, x1: float, y: float, *, n: int = 7, length: float = 0.28, label=None):
    """A distributed load on a horizontal boundary at height y pointing down onto it, drawn as `n` arrows
    under a connecting line, with an optional label above."""
    for xx in np.linspace(x0, x1, n):
        ax.annotate(
            "",
            xy=(xx, y),
            xytext=(xx, y + length),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, shrinkA=0, shrinkB=0, mutation_scale=7),
        )
    ax.plot([x0, x1], [y + length] * 2, color=INK, lw=0.6)
    if label:
        ax.text(
            0.5 * (x0 + x1),
            y + length + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=mpl.rcParams["legend.fontsize"] + 1,
        )


def dimension(ax, p, q, text, *, gap: float = 0.0):
    """A dimension line with arrow heads at both ends between p and q and `text` centred beside it
    (below a horizontal line, left of a vertical one)."""
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="<->", color=INK, lw=0.5, shrinkA=0, shrinkB=0))
    mx, my = 0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1])
    fs = mpl.rcParams["legend.fontsize"] + 1
    if abs(q[0] - p[0]) >= abs(q[1] - p[1]):
        ax.text(mx, my - 0.06 - gap, text, ha="center", va="top", fontsize=fs)
    else:
        ax.text(mx - 0.06 - gap, my, text, ha="right", va="center", fontsize=fs)


def field_size(ncols: int, aspect: float, medium: str = "paper", *, nrows: int = 1, extra: float = 0.35):
    """Figure size for a grid of fields (heatmaps) at the medium's full width, height fitted to the
    fields so there is no empty band: aspect = field height / width; `extra` = inches per row for
    labels and gaps."""
    w = SIZE[medium][0]
    panel_w = w / ncols * 0.84  # room for each colorbar and its labels
    return (w, nrows * (panel_w * aspect + extra))


def panel_letters(axes, *, start: str = "a"):
    """(a), (b), ... at the top-left of each panel, for captions that refer to panels."""
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(
            0.0,
            1.02,
            f"({chr(ord(start) + i)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=mpl.rcParams["font.size"],
            color=INK,
        )
