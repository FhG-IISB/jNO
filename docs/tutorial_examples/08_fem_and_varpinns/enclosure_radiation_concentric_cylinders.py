"""Conduction + grey-body enclosure radiation (concentric cylinders).

Two solid rings separated by a vacuum gap: heat conducts from the hot inner edge through the inner ring,
**radiates across the gap** to the outer ring, and conducts out to the cold edge. Radiation is nonlinear
(``T⁴``) and nonlocal (every gap-facing element exchanges with every other via the view-factor matrix), so
it is written as math in ``jno.np`` on top of ``domain.enclosure(...)`` and coupled to the conduction FEM
as a consistent surface load — there is no ``jno.radiation()`` helper.

The coupled solution matches the closed-form two-surface series

    Q = 2π k (T_hot - Ts1)/ln(r1/r0) = 2π r1 σ (Ts1⁴ - Ts2⁴)/D = 2π k (Ts2 - T_cold)/ln(r3/r2),
    D = 1/ε1 + (r1/r2)(1/ε2 - 1).

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4-5.
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import fsolve
from shapely.geometry import Point

import jno

SIGMA = 5.670374419e-8  # Stefan-Boltzmann [W/m^2/K^4]
r0, r1, r2, r3 = 0.10, 0.20, 0.25, 0.35  # hot edge | gap inner | gap outer | cold edge
k, eps1, eps2 = 20.0, 0.8, 0.6
T_hot, T_cold = 1000.0, 300.0  # Kelvin

# --- two disjoint solid rings; the gap r1..r2 is NOT meshed (heat crosses only by radiation) ---
ring = lambda a, b: Point(0, 0).buffer(b, 24).difference(Point(0, 0).buffer(a, 24))  # noqa: E731
d = jno.domain(ring(r0, r1).union(ring(r2, r3)), mesh_size=0.35)
rad = lambda x, y: jnp.hypot(x, y)  # noqa: E731  (JAX-traceable: jno.fem traces tag predicates)
d.tag("hot", lambda x, y: jnp.abs(rad(x, y) - r0) < 4e-2)
d.tag("cold", lambda x, y: jnp.abs(rad(x, y) - r3) < 4e-2)
d.tag("inner_gap", lambda x, y: jnp.abs(rad(x, y) - r1) < 4e-2)
d.tag("outer_gap", lambda x, y: jnp.abs(rad(x, y) - r2) < 4e-2)

# --- conduction FEM (whole-domain k, Dirichlet on hot/cold; the gap edges are left natural) ---
u, v = d.fem_symbols()
xi, yi, _ = d.variable("interior", split=True)
ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
xh, yh, _ = d.variable("hot", split=True)
xc, yc, _ = d.variable("cold", split=True)
fem = jno.fem([k * (ui.x * vi.x + ui.y * vi.y), u(xh, yh) - T_hot, u(xc, yc) - T_cold])
A = fem.operator[0]  # BCOO sparse (do not densify)
b = np.asarray(fem.operator[1]).reshape(-1)
n = b.size

# --- enclosure radiation: view matrix + per-element emissivity (jNO supplies F; you write the math) ---
gap = d.enclosure(["inner_gap", "outer_gap"])
gap.check()  # F-quality gate: closure + reciprocity
F = gap.view_factor
eps = gap.emissivity({"inner_gap": eps1, "outer_gap": eps2})
rho = 1.0 - eps
eye = jnp.eye(gap.size)
mi, mo = gap.tag_mask("inner_gap"), gap.tag_mask("outer_gap")
ar = np.asarray(gap.areas)


def q_rad(uu):  # full grey-body radiosity (reflections):  q = (I - F)(I - diag(rho)F)^-1 diag(eps) sigma T^4
    Ts = gap.field(uu)  # nonlocal gather: per-element temperature
    J = jnp.linalg.solve(eye - rho[:, None] * F, eps * SIGMA * Ts**4)
    return J - F @ J


# --- couple: A u = b - gap.load(q_rad(u)). Penalty-Dirichlet A is ill-conditioned -> direct LU + Picard ---
ad, ai = np.asarray(A.data), np.asarray(A.indices)
lu = spla.splu(sp.coo_matrix((ad, (ai[:, 0], ai[:, 1])), shape=(n, n)).tocsc())
T = lu.solve(b)  # warm start: pure conduction
for _ in range(200):
    Tn = lu.solve(b - np.asarray(gap.load(q_rad(jnp.asarray(T)), size=n)))
    if float(np.abs(Tn - T).max()) < 1e-5:
        T = Tn
        break
    T = 0.7 * T + 0.3 * Tn  # under-relax (radiation is stiff)

# --- surface temperatures + heat flow vs the analytic series ---
Tsf = np.asarray(gap.field(jnp.asarray(T)))
Ts1 = float((Tsf[mi] * ar[mi]).sum() / ar[mi].sum())
Ts2 = float((Tsf[mo] * ar[mo]).sum() / ar[mo].sum())
Q_fem = float((np.asarray(q_rad(jnp.asarray(T)))[mi] * ar[mi]).sum())
D = 1 / eps1 + (r1 / r2) * (1 / eps2 - 1)
ts1_a, ts2_a = fsolve(
    lambda s: [
        2 * np.pi * k * (T_hot - s[0]) / np.log(r1 / r0) - 2 * np.pi * r1 * SIGMA * (s[0] ** 4 - s[1] ** 4) / D,
        2 * np.pi * r1 * SIGMA * (s[0] ** 4 - s[1] ** 4) / D - 2 * np.pi * k * (s[1] - T_cold) / np.log(r3 / r2),
    ],
    [800.0, 500.0],
)
Q_a = 2 * np.pi * r1 * SIGMA * (ts1_a**4 - ts2_a**4) / D
print("\nConduction + grey-body enclosure radiation (concentric cylinders)")
print(f"  dofs={n}, enclosure elements={gap.size}, F closure/reciprocity={gap.quality()[0]:.1e}/{gap.quality()[1]:.1e}")
print(f"  Ts1 (inner gap): fem={Ts1:.1f} K  analytic={ts1_a:.1f} K")
print(f"  Ts2 (outer gap): fem={Ts2:.1f} K  analytic={ts2_a:.1f} K")
print(f"  Q (radiated):    fem={Q_fem:.0f} W/m  analytic={Q_a:.0f} W/m")

# --- figure: the solved temperature field + radial profile vs analytic ---
pts = np.asarray(d.mesh.points)[:, :2]
tris = np.asarray(d.mesh.cells_dict["triangle"])
fig, (axf, axr) = plt.subplots(1, 2, figsize=(12, 5.2))
tr = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
tpc = axf.tripcolor(tr, T, cmap="inferno", shading="gouraud")
axf.set_aspect("equal")
axf.set_xticks([])
axf.set_yticks([])
axf.set_title("temperature field (two rings, vacuum gap)", fontsize=11)
fig.colorbar(tpc, ax=axf, shrink=0.8, label="T [K]")

rr = np.hypot(pts[:, 0], pts[:, 1])
axr.scatter(rr, T, s=6, alpha=0.4, color="#0072B2", label="FEM nodes")
ri = np.linspace(r0, r1, 50)
ro = np.linspace(r2, r3, 50)
axr.plot(ri, T_hot - (T_hot - ts1_a) * np.log(ri / r0) / np.log(r1 / r0), color="#D55E00", lw=2, label="analytic")
axr.plot(ro, ts2_a + (T_cold - ts2_a) * np.log(ro / r2) / np.log(r3 / r2), color="#D55E00", lw=2)
axr.axvspan(r1, r2, color="0.9", label="vacuum gap (radiation)")
axr.set_xlabel("radius r [m]")
axr.set_ylabel("T [K]")
axr.set_title("radial profile: conduction + radiative gap jump", fontsize=11)
axr.legend(frameon=False, fontsize=9)
fig.suptitle("Conduction + grey-body radiation across a vacuum gap", fontsize=12)
fig.savefig(
    Path(__file__).parents[2] / "assets" / "enclosure_radiation_concentric_cylinders.png", dpi=130, bbox_inches="tight"
)

assert T_hot > Ts1 > Ts2 > T_cold, "temperatures must be monotone hot > Ts1 > Ts2 > cold"
assert abs(Ts1 - ts1_a) / ts1_a < 1e-2 and abs(Ts2 - ts2_a) / ts2_a < 1e-2, "surface temps must match the analytic series"
assert abs(Q_fem - Q_a) / abs(Q_a) < 2e-2, "radiated heat must match the analytic series"
