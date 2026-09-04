"""Render the jNO overview diagram (assets/overview-{light,dark}.svg).

One image for the README: the workflow spine you actually write a jNO script
along, and the modules that hang off it, split into the two pillars.

Run from the repo root::

    python assets/make_overview.py

Everything named in the diagram is public API -- keep it that way when editing.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

# ---------------------------------------------------------------------------
# palette -- teal accent on charcoal ink (house style); the amber for pillar 2
# is the accent already used by the docs theme (docs/stylesheets).
# ---------------------------------------------------------------------------
THEMES = {
    "light": dict(
        bg="#FFFFFF",
        edge="#E4E9EF",
        ink="#1A202C",
        muted="#5B6773",
        hair="#E2E8F0",
        card="#FFFFFF",
        card_stroke="#DDE3EA",
        panel="#F8FAFC",
        panel_stroke="#E2E8F0",
        teal="#0D9488",
        band_fill="#F0FDFA",
        band_stroke="#99F6E4",
        p1_head="#0F766E",
        p1_head_ink="#FFFFFF",
        p1_name="#0F766E",
        p2_head="#B45309",
        p2_head_ink="#FFFFFF",
        p2_name="#B45309",
        badge_ink="#FFFFFF",
    ),
    "dark": dict(
        bg="#0D1117",
        edge="#222B35",
        ink="#E6EDF3",
        muted="#9BA7B4",
        hair="#242D38",
        card="#141B23",
        card_stroke="#27313C",
        panel="#11171F",
        panel_stroke="#242D38",
        teal="#2DD4BF",
        band_fill="#0B2320",
        band_stroke="#155E56",
        p1_head="#115E59",
        p1_head_ink="#CCFBF1",
        p1_name="#2DD4BF",
        p2_head="#7C3D0A",
        p2_head_ink="#FDE9CC",
        p2_name="#F0A93B",
        badge_ink="#0D1117",
    ),
}

SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

# ---------------------------------------------------------------------------
# content -- the workflow spine
# ---------------------------------------------------------------------------
STAGES = [
    dict(
        label="GEOMETRY",
        api=["jno.Shape · jno.Path"],
        body=["CSG solids, boolean ops,", "multi-material regions,", "gmsh-OCC meshing"],
    ),
    dict(
        label="DOMAIN",
        api=["jno.domain"],
        body=["tri · quad · tet · hex", "tag regions & boundaries", "variable · fem_symbols"],
    ),
    dict(
        label="THE TERM LIST",
        api=["jno.np · fn · noise · nn"],
        body=["a weak form, a stencil,", "or a residual — as a list", "u.x · u.dd() · ∫ · units"],
    ),
    dict(
        label="SOLVE / TRAIN",
        api=["jno.fem · jno.fdm", "jno.rcwa · jno.core"],
        body=["four front doors, one", "term list — sparse,", "matrix-free, GPU-ready"],
    ),
    dict(
        label="RESULT",
        api=["fields · ∂/∂θ", ".plot · save · iree"],
        body=["the gradient flows", "through the whole solve —", "inverse problems & design"],
    ),
]

BAND_TITLE = "ONE DIFFERENTIABLE GRAPH"
BAND_TEXT = (
    "every list above lowers to the same jit-compiled JAX graph — reverse-mode differentiable end to end, on CPU or GPU"
)

# ---------------------------------------------------------------------------
# content -- the two pillars
# ---------------------------------------------------------------------------
PILLAR_1 = (
    "PILLAR 1 — DIFFERENTIABLE NUMERICAL METHODS",
    [
        (
            "jno.fem",
            [
                "Lagrange P1–P3+ · RT H(div) · Nédélec H(curl) ·",
                "C¹ Hermite / Argyris / Morley · quad & hex + hanging nodes",
                "periodic · complex · coupled multifield · 2-D & 3-D",
            ],
        ),
        (
            "jno.fdm",
            [
                "strong-form collocation · structured grids +",
                "geometric multigrid · unstructured · flux & periodic",
            ],
        ),
        ("jno.rcwa", ["vector Maxwell · anisotropic · orders · Jones readout"]),
        (
            "jno.solve",
            [
                "lu · cg bicgstab gmres minres chebyshev · AMG ·",
                "newton picard staggered · eigs · θ / exponential /",
                "adaptive time · remesh · refine · relocate",
            ],
        ),
        ("jno.precond", ["jacobi · gmg · amg · ams · nyström · block · cached"]),
    ],
)

PILLAR_2 = (
    "PILLAR 2 — SCIENTIFIC MACHINE LEARNING",
    [
        (
            "jno.nn",
            [
                "any Equinox module · foundax MLPs, transformers,",
                "DeepONet · FNO · U-Net · PROSE",
            ],
        ),
        ("jno.bayesian", ["NUTS · HMC · MALA · SGLD · SGHMC · VI — per-parameter"]),
        ("jno.optimizers", ["engd · mma · soap · ssbroyden · md · SIMP + any optax"]),
        ("jno.lora", ["LoRA · DoRA · rsLoRA · PiSSA · VeRA · LoKr · OFT · IA3"]),
        ("jno.sampler", ["adaptive resampling RAD · RARD · CR3 · R3 · pinnfluence"]),
        ("jno.schedule", ["LR & loss-weight schedules · callbacks · W&B · Orbax"]),
        ("jno.noise", ["gaussian · uniform · laplace · Gaussian random fields"]),
    ],
)

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
W = 1060
M = 30
CW = W - 2 * M

SPINE_H = 138
SPINE_GAP = 20
SPINE_W = (CW - 4 * SPINE_GAP) // 5

BAND_H = 50
PANEL_GAP = 28
PANEL_W = (CW - PANEL_GAP) // 2
HEAD_H = 32
ROW_PAD = 16
LINE_H = 15

Y_SPINE = 28
Y_BAND = Y_SPINE + SPINE_H + 26
Y_PANEL = Y_BAND + BAND_H + 28


def _rows_height(rows) -> int:
    return sum(ROW_PAD + LINE_H * len(lines) for _, lines in rows)


PANEL_BODY_H = max(_rows_height(PILLAR_1[1]), _rows_height(PILLAR_2[1]))
PANEL_H = HEAD_H + 12 + PANEL_BODY_H + 10
H = Y_PANEL + PANEL_H + 28


def text(x, y, s, *, size, fill, family=SANS, weight="400", anchor="start", spacing=None):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{ls}>{escape(s)}</text>'
    )


def build(theme_name: str) -> str:
    c = THEMES[theme_name]
    o: list[str] = []
    o.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="jNO workflow and module overview">'
    )
    o.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" fill="{c["bg"]}" stroke="{c["edge"]}"/>')

    # ---- the spine ------------------------------------------------------
    for i, st in enumerate(STAGES):
        x = M + i * (SPINE_W + SPINE_GAP)
        y = Y_SPINE
        o.append(
            f'<rect x="{x}" y="{y}" width="{SPINE_W}" height="{SPINE_H}" rx="10" '
            f'fill="{c["card"]}" stroke="{c["card_stroke"]}"/>'
        )
        # accent cap along the top edge
        o.append(
            f'<path d="M{x + 10} {y + 0.5} h{SPINE_W - 20}" stroke="{c["teal"]}" '
            f'stroke-width="3" stroke-linecap="round" fill="none"/>'
        )
        o.append(f'<circle cx="{x + 24}" cy="{y + 26}" r="9" fill="{c["teal"]}"/>')
        o.append(text(x + 24, y + 30, str(i + 1), size=11, fill=c["badge_ink"], weight="700", anchor="middle"))
        o.append(text(x + 40, y + 30, st["label"], size=12, fill=c["ink"], weight="700", spacing="0.5"))
        o.append(f'<path d="M{x + 14} {y + 44} h{SPINE_W - 28}" stroke="{c["hair"]}" stroke-width="1"/>')
        ty = y + 62
        for line in st["api"]:
            o.append(text(x + 14, ty, line, size=11, fill=c["teal"], family=MONO, weight="600"))
            ty += 15
        ty += 4
        for line in st["body"]:
            o.append(text(x + 14, ty, line, size=11, fill=c["muted"]))
            ty += 14

        if i < len(STAGES) - 1:
            ax = x + SPINE_W + SPINE_GAP / 2
            ay = y + SPINE_H / 2
            o.append(
                f'<path d="M{ax - 4} {ay - 5} l5 5 l-5 5" fill="none" stroke="{c["muted"]}" '
                f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
            )

    # ---- the substrate band --------------------------------------------
    o.append(
        f'<rect x="{M}" y="{Y_BAND}" width="{CW}" height="{BAND_H}" rx="10" '
        f'fill="{c["band_fill"]}" stroke="{c["band_stroke"]}"/>'
    )
    o.append(
        f'<path d="M{M + 4} {Y_BAND + 12} v{BAND_H - 24}" stroke="{c["teal"]}" stroke-width="3" stroke-linecap="round"/>'
    )
    o.append(text(M + 20, Y_BAND + 30, BAND_TITLE, size=12.5, fill=c["teal"], weight="700", spacing="0.6"))
    o.append(f'<path d="M{M + 226} {Y_BAND + 14} v{BAND_H - 28}" stroke="{c["band_stroke"]}" stroke-width="1"/>')
    o.append(text(M + 242, Y_BAND + 30, BAND_TEXT, size=11.5, fill=c["muted"]))

    # ---- the two pillars ------------------------------------------------
    for side, (title, rows) in enumerate((PILLAR_1, PILLAR_2)):
        px = M + side * (PANEL_W + PANEL_GAP)
        head = c["p1_head"] if side == 0 else c["p2_head"]
        head_ink = c["p1_head_ink"] if side == 0 else c["p2_head_ink"]
        name_col = c["p1_name"] if side == 0 else c["p2_name"]

        o.append(
            f'<rect x="{px}" y="{Y_PANEL}" width="{PANEL_W}" height="{PANEL_H}" rx="10" '
            f'fill="{c["panel"]}" stroke="{c["panel_stroke"]}"/>'
        )
        o.append(
            f'<path d="M{px} {Y_PANEL + HEAD_H} v-{HEAD_H - 10} a10 10 0 0 1 10 -10 '
            f'h{PANEL_W - 20} a10 10 0 0 1 10 10 v{HEAD_H - 10} z" fill="{head}"/>'
        )
        o.append(text(px + 16, Y_PANEL + 21, title, size=11.5, fill=head_ink, weight="700", spacing="0.7"))

        pad = ROW_PAD + (PANEL_BODY_H - _rows_height(rows)) / len(rows)
        ry = Y_PANEL + HEAD_H + 12
        for k, (name, lines) in enumerate(rows):
            if k:
                o.append(f'<path d="M{px + 14} {ry - 8} h{PANEL_W - 28}" stroke="{c["hair"]}" stroke-width="1"/>')
            o.append(text(px + 14, ry + 12, name, size=12, fill=name_col, family=MONO, weight="700"))
            ly = ry + 12
            for line in lines:
                o.append(text(px + 132, ly, line, size=11.5, fill=c["muted"]))
                ly += LINE_H
            ry += pad + LINE_H * len(lines)

    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    # the README reads assets/, the docs site reads docs/assets/ -- same bytes
    for name in THEMES:
        svg = build(name)
        for d in (root / "assets", root / "docs" / "assets"):
            (d / f"overview-{name}.svg").write_text(svg, encoding="utf-8")
        print(f"wrote {name}  ({W}x{H})")
