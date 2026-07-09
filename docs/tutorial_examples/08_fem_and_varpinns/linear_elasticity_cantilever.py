# --8<-- [start:code]
"""06 - Linear-elastic cantilever beam under an end load (vector FEM) via ``jno.fem``.

Isotropic plane-stress elasticity on a slender beam (length L = 10, height H = 1) clamped at
the root and loaded by a downward shear traction on the tip. The unknown is the vector
displacement u = (u_x, u_y) -- ``fem_symbols(value_shape=(2,))`` -- with P2 elements
(``order=2``; constant-strain TRI3 is too stiff in bending). The weak form is the isotropic
elasticity bilinear form  lambda (div u)(div phi) + 2 mu eps(u):eps(phi).

The tip deflection is checked against Euler-Bernoulli beam theory  delta = P L^3 / (3 E I).
EB is a slender-beam approximation (it ignores shear deformation and the clamped-end boundary
layer), yet a P2 solve on a 10:1 beam matches it to ~1% (Timoshenko & Goodier, *Theory of
Elasticity*, 1970). Linearity in the load is inherent to the linear operator.
"""

import jax.numpy as jnp
import numpy as np

import jno

E, nu, L, H = 1000.0, 0.3, 10.0, 1.0
lam, mu = E * nu / (1.0 - nu**2), E / (2.0 * (1.0 + nu))  # plane-stress Lame parameters
Iz = H**3 / 12.0  # second moment of area (unit thickness)
inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

d = jno.Shape.rect(0.0, 0.0, L, H, size=0.5).domain()
u, phi = d.fem_symbols(value_shape=(2,), order=2)  # P2 vector displacement
xi, yi, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)  # clamped root
xr, yr, _ = d.variable("right", split=True)  # loaded tip
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak = lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2)


q = 0.1  # downward shear traction (0, -q) on the tip; total load P = q * H
traction = -1.0 * inner(jnp.array([0.0, -q]), phi.bind(x=xr, y=yr), n_contract=1)
fem = jno.fem([weak, u(xl, yl) - (0.0, 0.0), traction])
# The slender-beam P2 stiffness matrix is too ill-conditioned for Jacobi-CG in float32 -> sparse-direct slot.
sol = jnp.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1, 2)  # (n_nodes, 2)
tip = np.asarray(fem.points)[:, 0] > L - 1e-6
delta, eb = float(-jnp.mean(sol[tip, 1])), (q * H) * L**3 / (3.0 * E * Iz)
print(
    f"\nCantilever (P2 elasticity): dofs={fem.dofs}  FEM tip={delta:.4f}  Euler-Bernoulli={eb:.4f}  ratio={delta / eb:.3f}"
)
assert fem.is_linear and abs(delta - eb) / eb < 0.05  # matches beam theory to ~1%
# --8<-- [end:code]

# ---- solution figure: deformed shape coloured by |u| | FEM vs Euler-Bernoulli tip deflection ----
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

pts = np.asarray(fem.points)  # P2 node coords, aligned with sol
umag = np.linalg.norm(sol, axis=1)  # displacement magnitude per node
SCALE = 2.0  # deformation exaggeration (labelled) -- tip deflection ~0.4 on a height-1 beam
tri_ref = mtri.Triangulation(pts[:, 0], pts[:, 1])  # connectivity from the reference config
tri_def = mtri.Triangulation(pts[:, 0] + SCALE * sol[:, 0], pts[:, 1] + SCALE * sol[:, 1], tri_ref.triangles)

fig, ax = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={"height_ratios": [2, 1]})
ax[0].triplot(tri_ref, color="0.75", lw=0.4)  # undeformed outline (reference mesh)
tp = ax[0].tripcolor(tri_def, umag, cmap="cividis", shading="gouraud")
fig.colorbar(tp, ax=ax[0], shrink=0.85, label=r"$|u|$")
ax[0].set_title(rf"deformed cantilever (displacement $\times{SCALE:g}$) coloured by $|u|$;  dofs={fem.dofs}")
ax[0].set_aspect("equal")
ax[0].set_axis_off()

ax[1].bar(["FEM (P2)", "Euler-Bernoulli"], [delta, eb], color=["#3b6fb6", "#b6b6b6"], width=0.5)
ax[1].set_ylabel("tip deflection")
ax[1].set_title(rf"tip deflection: FEM {delta:.4f} vs EB {eb:.4f}  (ratio {delta / eb:.3f})")
for i, v in enumerate([delta, eb]):
    ax[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
ax[1].margins(y=0.2)
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "linear_elasticity_cantilever.png")
