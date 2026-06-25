"""2D axisymmetric quasi-static magnetostatics — single-phase E-I core transformer.

Models a shell-type transformer in the r-z half-plane (rotational symmetry around
the z-axis). The azimuthal component A_φ(r, z) of the magnetic vector potential
satisfies the axisymmetric curl-curl equation

    −∂ᵣ(ν r ∂ᵣA) − ∂_z(ν r ∂_zA) + νA/r = r J_φ,

where ν = 1/μ is the magnetic reluctivity (m/H) and J_φ is the source current density
(A/m²). Weak form (test function v, axisymmetric measure r dr dz):

    ∫∫ ν (∂A/∂r ∂v/∂r + ∂A/∂z ∂v/∂z + Av/r²) r dr dz = ∫∫ J v r dr dz.

The r factor appears explicitly in the integrand — jNO's standard Cartesian 2-D
assembler then gives the correct cylindrical result (see §5.3, Ref. [2]).

Geometry (r-z cross-section of an E-I shell-type core):
  • Iron E-core (μᵣ = 1000): centre limb + top/bottom yokes + outer limb.
  • Primary winding: uniform current density J_P (open-circuit excitation).
  • Secondary winding: no current (open circuit, J_S = 0).
  • Surrounding air.

Per-region material properties are realised with the *additive-correction pattern*:
a global base term (ν_air, whole domain) plus a core correction (Δν = ν_iron − ν_air,
core cells only). Region-specific integration uses ``d.tag(name, shapely_geom)`` and
``d.variable(name, split=True)`` so each weak-form term integrates over its own cells.

Field post-processing at element centroids (piecewise constant for P1):
    B_r = −∂A_φ/∂z,    B_z = ∂A_φ/∂r + A_φ/r.

Verified by the physical gate: the iron core must concentrate magnetic flux, so
⟨|B|⟩_core > ⟨|B|⟩_outer-air.

References
----------
[1] J. Bastos & N. Sadowski, *Electromagnetic Modeling by Finite Element Methods*,
    Marcel Dekker (2003), §4.3 (axisymmetric A-formulation).
[2] P. Dular & C. Geuzaine, *GetDP Reference Manual*, §4.2, 2023 — open-circuit
    axisymmetric magnetostatics in A-formulation.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely import contains_xy as _cxy
from shapely.geometry import box
from shapely.ops import unary_union

import jno

# ── Physical parameters ───────────────────────────────────────────────────────
MU0 = 4 * np.pi * 1e-7  # H/m, vacuum permeability
MU_R = 1000.0  # relative permeability of electrical steel
NU_IRON = 1.0 / (MU0 * MU_R)  # reluctivity of iron [m/H]
NU_AIR = 1.0 / MU0  # reluctivity of air / copper windings
J_PRIM = 2.0e6  # A/m², primary current density (open-circuit excitation)

# ── Geometry (metres) ─────────────────────────────────────────────────────────
R_MAX, Z_MAX = 0.080, 0.100  # domain extent

# E-shaped iron core
r_cl = 0.015  # centre limb outer radius
r_ol1, r_ol2 = 0.045, 0.055  # outer limb inner / outer radius
z_bot, z_b = 0.005, 0.015  # bottom yoke: z_bot … z_b
z_t, z_top = 0.085, 0.095  # top yoke:    z_t  … z_top

core_cent = box(0.0, z_b, r_cl, z_t)  # centre limb (between yokes)
core_bot = box(0.0, z_bot, r_ol2, z_b)  # bottom yoke
core_top = box(0.0, z_t, r_ol2, z_top)  # top yoke
core_outer = box(r_ol1, z_b, r_ol2, z_t)  # outer limb
core_geom = unary_union([core_cent, core_bot, core_top, core_outer])

# Windings in the window [r_cl, r_ol1] × [z_b, z_t], with small clearance
prim_geom = box(0.018, 0.020, 0.030, 0.080)  # primary (inner half of window)
sec_geom = box(0.030, 0.020, 0.042, 0.080)  # secondary (outer half of window)

air_geom = box(0.0, 0.0, R_MAX, Z_MAX).difference(core_geom).difference(prim_geom).difference(sec_geom)

# ── Mesh ──────────────────────────────────────────────────────────────────────
dom = jno.domain({"core": core_geom, "prim": prim_geom, "sec": sec_geom, "air": air_geom})
dom = dom.build_mesh(0.003, sizes={"core": 0.0015, "prim": 0.0015, "sec": 0.0015})

# Register material regions as tags so jno.fem can integrate each term on its cells.
# d.tag(name, shapely_geom) wraps shapely.contains_xy into a per-centroid predicate.
dom.tag("core", core_geom)
dom.tag("prim", prim_geom)
dom.tag("sec", sec_geom)

# ── FEM symbols and per-region coordinates ────────────────────────────────────
A, phi = dom.fem_symbols()

xi, yi, _ = dom.variable("interior", split=True)  # whole-domain coordinates
xb, yb, _ = dom.variable("boundary", split=True)  # boundary (for Dirichlet)
xc, yc, _ = dom.variable("core", split=True)  # iron-core cells
xp, yp, _ = dom.variable("prim", split=True)  # primary-winding cells

Ai, vi = A.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
Ac, vc = A.bind(x=xc, y=yc), phi.bind(x=xc, y=yc)
vp = phi.bind(x=xp, y=yp)  # test function on primary (source term has no trial A)


def axi_stiff(nu, r, Ab, vb):
    """Axisymmetric bilinear form ν(∂A/∂r·∂v/∂r + ∂A/∂z·∂v/∂z + Av/r²)·r."""
    return nu * (Ab.x * vb.x + Ab.y * vb.y + Ab * vb / r**2) * r


# Additive-correction approach:
#   base term  (ν_air, whole domain) + core Δν correction (ν_iron−ν_air, core only).
# Source: −J·v·r integrated over the primary-winding cells only.
# Dirichlet: A = 0 on every boundary (axis r=0 physical; outer edges far-field).
fem = jno.fem(
    [
        axi_stiff(NU_AIR, xi, Ai, vi),  # base stiffness (air everywhere)
        axi_stiff(NU_IRON - NU_AIR, xc, Ac, vc),  # Δν correction in iron
        -J_PRIM * vp * xp,  # primary-winding source ∫ J v r dr dz
        A(xb, yb) - 0.0,  # A_φ = 0 on all boundaries
    ]
)

sol = np.asarray(fem.solve())
print("\n2D axisymmetric transformer (open circuit, E-I core, P1 Lagrange)")
print(f"  DOFs = {fem.dofs}   max A_φ = {sol.max():.4e} Wb/m")

# ── B field at element centroids (piecewise-constant gradient for P1) ─────────
pts = np.asarray(fem.points)[:, :2]  # (n_nodes, 2)
cells = np.asarray(fem.domain.built_mesh.cells_dict["triangle"])  # (n_cells, 3)
p0, p1, p2 = pts[cells[:, 0]], pts[cells[:, 1]], pts[cells[:, 2]]
A0, A1, A2 = sol[cells[:, 0]], sol[cells[:, 1]], sol[cells[:, 2]]

# Shape-function gradient via 2×2 Cramer rule (piecewise constant for P1)
dr01, dz01 = p1[:, 0] - p0[:, 0], p1[:, 1] - p0[:, 1]
dr02, dz02 = p2[:, 0] - p0[:, 0], p2[:, 1] - p0[:, 1]
det = dr01 * dz02 - dr02 * dz01
dA_dr = ((A1 - A0) * dz02 - (A2 - A0) * dz01) / det
dA_dz = ((A2 - A0) * dr01 - (A1 - A0) * dr02) / det

cent = (p0 + p1 + p2) / 3.0  # (n_cells, 2), r = cent[:,0] > 0 for all interior cells
A_cent = (A0 + A1 + A2) / 3.0

B_r = -dA_dz  # B_r = −∂A_φ/∂z
B_z = dA_dr + A_cent / cent[:, 0]  # B_z = ∂A_φ/∂r + A_φ/r
B_mag = np.sqrt(B_r**2 + B_z**2)

# ── Physical gate: iron must concentrate the flux ─────────────────────────────
in_core = np.asarray(_cxy(core_geom, cent[:, 0], cent[:, 1]))
outer_air = cent[:, 0] > r_ol2  # cells to the right of the outer limb

B_core = float(B_mag[in_core].mean())
B_outer = float(B_mag[outer_air].mean())
ratio = B_core / max(B_outer, 1e-20)
print(f"  ⟨|B|⟩ core = {B_core:.3f} T   ⟨|B|⟩ outer air = {B_outer:.4f} T   ratio = {ratio:.0f}×")

# ── Figure ────────────────────────────────────────────────────────────────────
tris = cells
triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))


def _draw(geom, ax, color, lw=0.9, label=None):
    """Draw all rings (exterior + interiors) of a shapely polygon."""
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    for g in parts:
        c = np.array(g.exterior.coords)
        ax.plot(c[:, 0] * 1e3, c[:, 1] * 1e3, color=color, lw=lw, label=label)
        label = None  # only label the first ring
        for ring in g.interiors:
            rc = np.array(ring.coords)
            ax.plot(rc[:, 0] * 1e3, rc[:, 1] * 1e3, color=color, lw=lw)


# Convert pts to mm for display
pts_mm = pts * 1e3
triang_mm = mtri.Triangulation(pts_mm[:, 0], pts_mm[:, 1], tris)
cent_mm = cent * 1e3

# Left panel: A_φ contours = magnetic flux tubes
cf = ax1.tricontourf(triang_mm, sol * 1e3, levels=30, cmap="RdBu_r")  # mWb/m
ax1.tricontour(triang_mm, sol * 1e3, levels=20, colors="k", linewidths=0.35, alpha=0.55)
ax1.triplot(triang_mm, color="k", lw=0.12, alpha=0.30)
plt.colorbar(cf, ax=ax1, label="$A_\\varphi$ [mWb/m]", shrink=0.85)
_draw(core_geom, ax1, "steelblue", label="iron core")
_draw(prim_geom, ax1, "crimson", label="primary")
_draw(sec_geom, ax1, "forestgreen", label="secondary")
ax1.legend(fontsize=8, loc="upper right")
ax1.set_title(f"Magnetic flux lines  $A_\\varphi(r,z)$\n(open circuit, $J_0 = {J_PRIM:.0e}$ A/m²)", fontsize=10)

# Right panel: |B| field at centroids (flat shading, one value per element)
cb = ax2.tripcolor(triang_mm, B_mag, shading="flat", cmap="hot_r", vmin=0)
ax2.triplot(triang_mm, color="w", lw=0.12, alpha=0.25)
plt.colorbar(cb, ax=ax2, label="|B| [T]", shrink=0.85)
_draw(core_geom, ax2, "cyan", label="iron core")
_draw(prim_geom, ax2, "yellow", label="primary")
_draw(sec_geom, ax2, "lime", label="secondary")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_title(f"|B|(r,z) at element centroids\n(core: {B_core:.2f} T vs outer air: {B_outer:.4f} T)", fontsize=10)

for ax in (ax1, ax2):
    ax.set_xlim(0, R_MAX * 1e3)
    ax.set_ylim(0, Z_MAX * 1e3)
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_aspect("equal")

fig.suptitle(
    "2D axisymmetric transformer — jno.fem, P1 Lagrange, E-I core (μᵣ = 1000), open circuit",
    fontsize=11,
)
fig.tight_layout()
_out = Path(__file__).parents[2] / "assets" / "maxwell_axisym_transformer.png"
fig.savefig(_out, dpi=130, bbox_inches="tight")
print(f"  saved → docs/assets/{_out.name}")

assert sol.max() > 0, "A_φ must be positive for a +φ current excitation"
assert B_core > B_outer * 5, f"iron must concentrate flux: ⟨|B|⟩_core / ⟨|B|⟩_air = {ratio:.1f} (expected > 5)"
