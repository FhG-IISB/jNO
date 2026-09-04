# --8<-- [start:code]
"""**DFG benchmark 2D-1** — steady Navier–Stokes past a cylinder at Re = 20, checked against the
published reference values rather than against a manufactured solution.

    (u.grad)u - nu lap u + grad p = 0,   div u = 0,   Re = U_mean D / nu = 20

Configuration and reference values are Schäfer & Turek (1996), the DFG/Featflow benchmark:

    domain     [0, 2.2] x [0, 0.41] minus a disk of radius 0.05 at (0.2, 0.2)
    inflow     u = (4 U y (0.41 - y) / 0.41^2, 0),  U = 0.3  ->  U_mean = 2/3 U = 0.2
    cD, cL     2 F / (U_mean^2 D)          dP  p(0.15, 0.2) - p(0.25, 0.2)
    reference  cD = 5.57953523384   cL = 0.010618948146   dP = 0.11752016697

Two jNO-specific things carry this example. The **outflow is written by not writing it** — an
untagged boundary is do-nothing, which is the traction-free condition an outlet wants, and it fixes
the pressure level so no `p.pin()` appears anywhere. And the **forces are reactions**: `fem.eval`
assembles the momentum residual with no essential elimination, so the sum over the cylinder's
constrained DOFs is the force holding it in place. That is the accurate route to a force from a
finite element solution (John, *Int. J. Numer. Meth. Fluids* 44, 2004) — better than integrating
stress over the surface.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace

L, H = 2.2, 0.41
CX, CY, RR = 0.2, 0.2, 0.05
UMAX, NU = 0.3, 1e-3
UMEAN, DIA = 2.0 / 3.0 * UMAX, 2.0 * RR
EPS = 1e-9
CD_REF, CL_REF, DP_REF = 5.57953523384, 0.010618948146, 0.11752016697

# CSG, with a finer size on the disk: drag accuracy lives on the boundary layer around the cylinder.
shape = jno.Shape.rect(0, 0, L, H, size=0.035) - jno.Shape.disk(CX, CY, RR, size=0.006)
d = shape.domain()
d.tag("inlet", lambda x, y: x < EPS)
d.tag("walls", lambda x, y: (y < EPS) | (y > H - EPS))
d.tag("cyl", lambda x, y: (x - CX) ** 2 + (y - CY) ** 2 < (RR + 1e-4) ** 2)

u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)  # P2 velocity
p, q = d.fem_symbols(names=("p", "q"), order=1)  # P1 pressure  -> Taylor-Hood
xi, yi = d.variable("interior", split=True)[:2]
xin, yin = d.variable("inlet", split=True)[:2]
xw, yw = d.variable("walls", split=True)[:2]
xc, yc = d.variable("cyl", split=True)[:2]

gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
conv = inner(gu, ub, n_contract=1)  # (u.grad)u
momentum = inner(conv, vv, n_contract=1) + NU * inner(gu, gv, n_contract=2) - pp * trace(gv)
profile = 4.0 * UMAX * yin * (H - yin) / H**2

fem = jno.fem(
    [
        momentum,
        -qq * trace(gu),
        u(xin, yin)[0] - profile,  # parabolic inflow
        u(xin, yin)[1] - 0.0,
        u(xw, yw)[0] - 0.0,  # no-slip channel walls
        u(xw, yw)[1] - 0.0,
        u(xc, yc)[0] - 0.0,  # no-slip cylinder
        u(xc, yc)[1] - 0.0,
    ]  # the outlet gets nothing: natural (do-nothing), and it sets the pressure level
)
assert not fem.is_linear, "Re=20 keeps the convective term, so the system is nonlinear"
print(f"\nDFG 2D-1, steady Navier-Stokes past a cylinder (Re={UMEAN * DIA / NU:.0f}): dofs={fem.dofs}")

sol = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))  # saddle system -> direct

# ---- drag and lift as the reaction conjugate to the cylinder's no-slip constraint ----
free = np.asarray(fem.eval(momentum, sol))
scale = 2.0 / (UMEAN**2 * DIA)
cD = -scale * float(free[fem.region_dofs("cyl", field=u, component=0)].sum())
cL = -scale * float(free[fem.region_dofs("cyl", field=u, component=1)].sum())

off = fem.offsets
pre, ppts = sol[off[1] :], np.asarray(fem.field_points[1])
probe = lambda pt: float(pre[int(np.argmin(np.sum((ppts - np.asarray(pt)) ** 2, axis=1)))])  # noqa: E731
dP = probe((CX - RR, CY)) - probe((CX + RR, CY))

print(f"  cD = {cD:9.5f}   reference {CD_REF:9.5f}   ({100 * abs(cD - CD_REF) / CD_REF:.2f}%)")
print(f"  cL = {cL:9.5f}   reference {CL_REF:9.5f}   ({100 * abs(cL - CL_REF) / CL_REF:.2f}%)")
print(f"  dP = {dP:9.5f}   reference {DP_REF:9.5f}   ({100 * abs(dP - DP_REF) / DP_REF:.2f}%)")

assert abs(cD - CD_REF) / CD_REF < 5e-3
assert abs(dP - DP_REF) / DP_REF < 1.5e-2
# Lift is NOT held tightly: the cylinder sits 0.005 off the channel axis, so cL is a small residue
# of nearly-cancelling forces and is dominated by the random asymmetry of an unstructured mesh.
# Across four refinements the drag error goes 0.39 / 0.10 / 0.04 / 0.02 % while the lift error
# wanders 6.9 / 1.9 / 3.8 / 6.7 %. Sign and magnitude are what reproduce here.
assert 0.0 < cL < 5.0 * CL_REF
# --8<-- [end:code]

# ---- figure: speed with streamlines, and the cylinder cut out ----
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402

vel = sol[off[0] : off[1]].reshape(-1, 2)
pts = np.asarray(fem.field_points[0])
gx, gy = np.meshgrid(np.linspace(0, 0.9, 520), np.linspace(0, H, 210))  # near field only
UX = griddata(pts, vel[:, 0], (gx, gy), method="linear")
UY = griddata(pts, vel[:, 1], (gx, gy), method="linear")
hole = (gx - CX) ** 2 + (gy - CY) ** 2 < RR**2
UX[hole] = np.nan
UY[hole] = np.nan

fig, ax = plt.subplots(figsize=(9.2, 2.6))
im = ax.pcolormesh(gx, gy, np.hypot(UX, UY), cmap="viridis", shading="auto")
ax.streamplot(gx, gy, UX, UY, color="white", density=1.5, linewidth=0.5, arrowsize=0.6)
ax.add_patch(plt.Circle((CX, CY), RR, facecolor="#EDF2F4", edgecolor="#1A202C", lw=1.0, zorder=3))
ax.set_xlim(0, 0.9)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.012, 0.9, f"Re = {UMEAN * DIA / NU:.0f}", transform=ax.transAxes, fontsize=9, color="white", va="top")
fig.colorbar(im, ax=ax, label="|u|", fraction=0.02, pad=0.01)
fig.tight_layout()
fig.savefig(Path(__file__).parents[2] / "assets" / "navier_stokes_cylinder_dfg.png", dpi=135)
