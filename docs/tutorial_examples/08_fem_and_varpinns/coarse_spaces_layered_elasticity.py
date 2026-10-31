# --8<-- [start:code]
"""Coarse spaces for two-level Schwarz: rigid-body modes, where they stop being enough, and GenEO.

Plane-strain elasticity on a [0, 4] x [0, 1] block of four horizontal layers, clamped on the left, under
its own weight, P1. The layers alternate between a stiff E = 1e4 and a soft E = 1e4 / contrast (nu = 0.3);
contrast = 1 is a homogeneous block. CG with overlapping Schwarz on METIS subdomains:

  one-level  S = schwarz(parts=p, coarse=False): local solves only; the count grows with the parts
  rigid      Q >> S >> Q with Q = coarse(rigid-body modes, parts=S): three vectors per subdomain, the
             near-kernel of every floating elastic piece (balancing Neumann-Neumann, BNN)
  geneo      the same with Q = coarse(jno.space.geneo(), parts=S): per subdomain, the eigenvectors of a local
             generalized eigenproblem below tau = 0.2 -- the rigid modes plus whatever else floats

Every solve is checked against the sparse-direct answer. Two-level Schwarz with rigid modes: Toselli &
Widlund, *Domain Decomposition Methods*, Springer 2005, ch. 8; the BNN form: Tang, Nabben, Vuik &
Erlangga, J. Sci. Comput. 39, 2009; coarse spaces for high contrast (GenEO): Spillane et al., Numer.
Math. 126, 2014.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # Schwarz keys its entries in int64; the solves run in float64
import jno  # noqa: E402

R = jno.shape.rect
NU, E_STIFF, RHO_G = 0.3, 1e4, 0.1
symgrad, inner, trace = jno.np.symgrad, jno.np.inner, jno.np.trace


def layered_block(contrast, h=0.05):
    """The clamped block; soft layers have E = E_STIFF / contrast."""
    soft = E_STIFF / contrast
    d = (
        (
            R(0, 0.00, 4, 0.25).name("layer0").attach(E=E_STIFF)
            + R(0, 0.25, 4, 0.50).name("layer1").attach(E=soft)
            + R(0, 0.50, 4, 0.75).name("layer2").attach(E=E_STIFF)
            + R(0, 0.75, 4, 1.00).name("layer3").attach(E=soft)
        )
        .sized(h)
        .domain()
    )
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u, v = d.fem_symbols(value_shape=(2,))
    u_, v_ = u.bind(x=x, y=y), v.bind(x=x, y=y)

    ε = lambda w: symgrad(w, [x, y])  # noqa: E731
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731
    lam, mu = d.E * NU / ((1 + NU) * (1 - 2 * NU)), d.E / (2 * (1 + NU))

    fem = jno.fem([2 * mu * ddot(ε(u_), ε(v_)) + lam * trace(ε(u_)) * trace(ε(v_)) + RHO_G * v_[1], u(xl, yl) - (0.0, 0.0)])
    return fem, (x, y)


def cg_iterations(fem, precond, ref):
    sol = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-8, maxiter=5000), precond=precond))
    assert np.abs(sol - ref).max() < 1e-5 * np.abs(ref).max(), "the preconditioned solve differs from the direct one"
    return fem.stats["linear_iterations"]["total"]


PARTS, CONTRASTS = (4, 16, 64), (1, 1e4)
its, solved = {}, {}
for contrast in CONTRASTS:
    fem, (x, y) = layered_block(contrast)
    ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    rigid = [(1, 0), (0, 1), (-y, x)]  # the 2-D rigid-body modes: two translations, one rotation
    for p in PARTS:
        S = jno.precond.schwarz(parts=p, coarse=False)
        Q = jno.precond.coarse(rigid, parts=S)  # the same vectors, cut to S's subdomains
        G = jno.precond.coarse(jno.space.geneo(), parts=S)  # computed from S's subdomains
        its[contrast, "one-level", p] = cg_iterations(fem, S, ref)
        its[contrast, "rigid", p] = cg_iterations(fem, Q >> S >> Q, ref)
        its[contrast, "geneo", p] = cg_iterations(fem, G >> S >> G, ref)
    solved[contrast] = fem, S
    print(f"contrast {contrast:g}  ({fem.dofs} dofs): CG iterations to 1e-8")
    for name in ("one-level", "rigid", "geneo"):
        print(f"  {name:10s}" + "".join(f"  parts={p:<3d} {its[contrast, name, p]:4d}" for p in PARTS))

# Rigid modes keep the count from growing with the parts; with a stiffness jump it stays higher
for c in CONTRASTS:
    assert its[c, "rigid", 64] < 0.6 * its[c, "one-level", 64]
assert its[1, "rigid", 64] < 1.5 * its[1, "rigid", 4]
assert its[1e4, "rigid", 64] > 2 * its[1, "rigid", 64]
# GenEO closes the gap: the contrast costs (almost) nothing
for p in PARTS:
    assert its[1e4, "geneo", p] < 0.6 * its[1e4, "rigid", p]
    assert its[1e4, "geneo", p] < 1.3 * its[1, "geneo", p]
# --8<-- [end:code]

# ---- figure: the 64 subdomains over the layers | CG iterations against the number of subdomains ----
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

fem, S = solved[1e4]
pts = np.asarray(fem.points)
part = np.asarray(S._pattern.part)[0::2]  # DOFs interleave (u_x, u_y) per node
tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
cell_part = part[tri.triangles].min(axis=1)  # each triangle coloured by one of its nodes' subdomains
tri.set_mask(cell_part >= S._pattern.p)  # clamped nodes belong to no subdomain
shuffle = np.random.default_rng(0).permutation(S._pattern.p)  # neighbouring subdomains get distinct colours

fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={"height_ratios": [1.1, 1]})
ax[0].tripcolor(tri, facecolors=shuffle[np.minimum(cell_part, S._pattern.p - 1)], cmap="turbo", alpha=0.6)
ax[0].hlines((0.25, 0.5, 0.75), 0, 4, color="k", lw=1.2)
for yy, label in ((0.125, "stiff"), (0.375, "soft"), (0.625, "stiff"), (0.875, "soft")):
    ax[0].text(4.04, yy, label, va="center", fontsize=9)
ax[0].set_title("64 METIS subdomains (colours) cut across the layers (black lines); clamped on the left")
ax[0].set_aspect("equal")
ax[0].set_axis_off()

style = {"one-level": "--", "rigid": "-", "geneo": ":"}
colour = {1: "#3b6fb6", 1e4: "#c0504d"}
for c in CONTRASTS:
    for name in ("one-level", "rigid", "geneo"):
        label = f"{name}, {'homogeneous' if c == 1 else f'contrast {c:g}'}"
        mk = {"geneo": "s"}.get(name, "o")  # GenEO's two curves lie on top of each other
        ax[1].plot(
            PARTS,
            [its[c, name, p] for p in PARTS],
            style[name],
            marker=mk,
            mfc="none" if c == 1 else None,
            color=colour[c],
            label=label,
        )
ax[1].set_xscale("log", base=2)
ax[1].set_xticks(PARTS, [str(p) for p in PARTS])
ax[1].set_xlabel("subdomains")
ax[1].set_ylabel("CG iterations to 1e-8")
ax[1].set_title("one-level grows with the subdomains; rigid modes hold it but not the contrast; GenEO holds both")
ax[1].legend(fontsize=8, frameon=False)
ax[1].set_ylim(0, None)
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "coarse_spaces_layered_elasticity.png")
