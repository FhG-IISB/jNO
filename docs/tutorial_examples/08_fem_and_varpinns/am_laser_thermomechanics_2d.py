"""14 - Additive manufacturing: a scanning-laser thermo-mechanical model (heat + a moving source + stress).

A laser sweeps across a clamped metal plate (think one pass of a powder-bed or directed-energy build).
The moving hot spot conducts heat into the plate and loses some to the surroundings; the locally hot
material wants to expand but is held back by the cooler material around it, so a **thermal stress** field
travels with the beam. Two physics, one coupled ``jno.fem``:

* **heat** (transient): ``dT/dt = lap T - Bi*T``  + a **moving Gaussian source** (the laser);
* **thermo-elasticity** (quasi-static): ``div sigma = 0``, ``sigma = lam tr(e) I + 2 mu e - beta (T-T_ref) I``
  — temperature drives strain through the ``-beta (T-T_ref)`` term, a *linear cross-coupling* structurally
  identical to Boussinesq buoyancy (``rayleigh_benard_2d.py``).

Two fields — temperature ``T`` (P1) and displacement ``u`` (P2 vector) — assembled into a single
``jno.fem([...])``. The whole system is **linear** with a **constant** operator, so we factor ``M/dt+A``
**once** (sparse LU) and back-substitute every step — the moving laser is just a different right-hand
side each step (jNO can't put a *time-dependent* source in the weak form, so we add it as a per-step
load in our own backward-Euler stepper — the same trick the oven used for its radiation load).

The thermal field is checked against the classic **Rosenthal** moving-point-source solution. Note: pure
thermo-elasticity shows the *transient* stress that travels with the beam and relaxes once the plate
re-cools uniformly; **permanent residual stress / warping needs plasticity** (a documented extension),
so we do not call this "residual stress." The animation is the *computed* temperature and von Mises
stress; nothing is painted in.

References: D. Rosenthal, "The theory of moving sources of heat and its application to metal treatments",
Trans. ASME 68:849-866, 1946 (the moving-source temperature field).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"  # linear factorize-once sparse LU: fast on CPU, no GPU contention/OOM
os.environ["FEAX_X64"] = "1"  # float64 feax assembly (the test session defaults FEAX_X64=0; this subprocess opts in)
os.environ["MPLBACKEND"] = "Agg"

import jax

jax.config.update("jax_enable_x64", True)  # feax assembly is float64

from pathlib import Path  # noqa: E402

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as spsp  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402
from scipy.special import kn  # noqa: E402  (modified Bessel K0, for the Rosenthal reference)
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

symgrad, inner, trace = jno.np.symgrad, jno.np.inner, jno.np.trace

# --- non-dimensional parameters (length in plate units, thermal diffusivity alpha = 1) ---
Lx, Ly = 2.0, 1.0
lam, mu, beta = 1.0, 1.0, 0.6  # Lame parameters + thermal-expansion coupling strength
v_scan, P_laser, r0 = 4.0, 24.0, 0.09  # scan speed (Peclet = v*Lx = 8), laser power, spot radius
Bi = 3.0  # surface heat loss to the surroundings (keeps the far field near ambient)
T_ref = 0.0  # stress-free reference temperature
x0, y0 = 0.30, Ly / 2  # laser start (scans along the centre line)

# --- coupled fields: temperature (P1) + displacement (P2 vector) ---
d = jno.domain(box(0, 0, Lx, Ly), mesh_size=0.05, time=(0.0, 0.35, 2))
T, sT = d.fem_symbols(names=("T", "sT"), order=1)
u, phi = d.fem_symbols(value_shape=(2,), names=("u", "phi"), order=2)
xi, yi, ti = d.variable("interior", split=True)
xb, yb, _ = d.variable("boundary", split=True)
ci = d.variable("initial", split=True)
Tb, sb = T.bind(x=xi, y=yi, t=ti), sT.bind(x=xi, y=yi, t=ti)
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])

thermal = Tb.t * sb + (Tb.x * sb.x + Tb.y * sb.y) + Bi * Tb * sb  # dT/dt - lap T + surface loss (laser added per-step)
mech = lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2) - beta * (Tb - T_ref) * trace(ep)
fem = jno.fem(
    [
        thermal,
        mech,
        u(xb, yb) - 0.0,  # clamp the plate edges -> the constrained hot zone builds stress
        T(ci[0], ci[1]) - 0.0,  # start at ambient
        u(ci[0], ci[1]) - 0.0,  # at rest
    ]
)
assert fem.is_transient and fem.is_linear, "linear transient thermo-elasticity"
off = fem.problem.offset
nT = int(off[1])  # temperature is field 0: DOFs w[:nT]; displacement is field 1: w[nT:]
pts_T = np.asarray(fem.problem.mesh[0].points)[:, :2]  # P1 temperature nodes
pts_u = np.asarray(fem.problem.mesh[1].points)[:, :2]  # P2 displacement nodes
tris = np.asarray(d.built_mesh.cells_dict["triangle"])
triT = mtri.Triangulation(pts_T[:, 0], pts_T[:, 1], tris)


def _csc(B):  # jax operator (BCOO or dense) -> scipy CSC
    if hasattr(B, "sum_duplicates"):
        B = B.sum_duplicates()
        ij = np.asarray(B.indices)
        return spsp.csc_matrix((np.asarray(B.data), (ij[:, 0], ij[:, 1])), shape=tuple(B.shape))
    return spsp.csc_matrix(np.asarray(B))


# --- linear + constant operator: factor M + dt*A ONCE, then back-substitute each step ---
M, A = _csc(fem.M), _csc(fem.operator.A)
dt, nsteps, nframes = 0.007, 50, 20
nodal_area = np.asarray(M[:nT, :nT].sum(axis=1)).ravel()  # row-sums of the T mass = lumped nodal areas
lu = splu((M + dt * A).tocsc())
print(
    f"\nAM laser thermo-mechanics (Pe={v_scan * Lx:g}): dofs={fem.dofs} (T {nT} + u {fem.dofs - nT}), factor-once + back-sub"
)


def laser_load(t):  # moving Gaussian: a per-step heat load on the temperature block
    cx, cy = x0 + v_scan * t, y0
    g = (2 * P_laser / (np.pi * r0**2)) * np.exp(-2 * ((pts_T[:, 0] - cx) ** 2 + (pts_T[:, 1] - cy) ** 2) / r0**2)
    load = np.zeros(int(fem.dofs))
    load[:nT] = g * nodal_area
    return load, cx


w = np.asarray(fem.state0)
frames, centres, e_in = [w.copy()], [x0], 0.0
for step in range(nsteps):
    t_next = (step + 1) * dt
    load, cx = laser_load(t_next)
    w = lu.solve(M.dot(w) + dt * load)
    e_in += dt * float(load[:nT].sum())  # deposited laser energy so far
    if (step + 1) % max(1, nsteps // nframes) == 0:
        frames.append(w.copy())
        centres.append(cx)
frames = np.stack(frames)


def von_mises(w):
    """von Mises stress on a regular grid (audit the computed displacement, not a hand-built field)."""
    ux = griddata(pts_u, w[nT:].reshape(-1, 2)[:, 0], (GX, GY), method="cubic", fill_value=0.0)
    uy = griddata(pts_u, w[nT:].reshape(-1, 2)[:, 1], (GX, GY), method="cubic", fill_value=0.0)
    Tg = griddata(pts_T, w[:nT], (GX, GY), method="cubic", fill_value=0.0)
    exx, exy = np.gradient(ux, gx, axis=1), 0.5 * (np.gradient(ux, gy, axis=0) + np.gradient(uy, gx, axis=1))
    eyy = np.gradient(uy, gy, axis=0)
    tr = exx + eyy
    sxx = lam * tr + 2 * mu * exx - beta * (Tg - T_ref)
    syy = lam * tr + 2 * mu * eyy - beta * (Tg - T_ref)
    sxy = 2 * mu * exy
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2), sxx + syy  # von Mises, trace(sigma)


gx, gy = np.linspace(0, Lx, 160), np.linspace(0, Ly, 80)
GX, GY = np.meshgrid(gx, gy)
vm_last, trace_last = von_mises(frames[-1])

# --- diagnostics + validation ---
Tf = frames[-1, :nT]
e_stored = float((Tf * nodal_area).sum())  # ~ integral of T over the plate (rho*c = 1)
e_lost = float((frames[:, :nT].mean(axis=0) * nodal_area).sum() * Bi * (nsteps * dt))  # ~ integral of Bi*T dt
cx_last = centres[-1]
# Rosenthal moving-point-source (2D thin plate), in the moving frame along the scan line behind the beam:
xi_w = np.linspace(-0.5, -0.05, 40)  # wake (xi = x - x_laser < 0)
r_w = np.abs(xi_w)
ros = (P_laser / (2 * np.pi)) * np.exp(-v_scan * xi_w / 2.0) * kn(0, v_scan * r_w / 2.0)
fem_line = mtri.LinearTriInterpolator(triT, Tf)(cx_last + xi_w, np.full_like(xi_w, y0))
fem_line = np.asarray(fem_line)
print(
    f"  T range [{Tf.min():.2f}, {Tf.max():.2f}]  max|u|={np.abs(frames[-1, nT:]).max():.3f}  von Mises max={vm_last.max():.2f}"
)
print(
    f"  energy: deposited={e_in:.1f}  stored~{e_stored:.1f}  lost~{e_lost:.1f}  (in ~ stored+lost: {e_stored + e_lost:.1f})"
)
print(
    f"  Rosenthal wake T (fem vs analytic) at xi=-0.2: {np.interp(-0.2, xi_w[::-1], fem_line[::-1]):.2f} vs {np.interp(-0.2, xi_w[::-1], ros[::-1]):.2f}"
)

# stress is compressive in the (constrained) hot zone -> trace(sigma) < 0 near the beam
hot = (np.abs(GX - cx_last) < 0.15) & (np.abs(GY - y0) < 0.15)
trace_hot = float(np.nanmean(trace_last[hot]))
print(f"  mean trace(sigma) in the hot zone = {trace_hot:.2f}  (compressive < 0)")

assert Tf.min() > -0.05 and Tf.max() < 1e3, "temperature must stay physical (>= ambient, finite)"
assert Tf.max() > 3 * np.median(Tf), "the laser must create a localised hot spot"
assert np.abs(frames[-1, nT:]).max() > 1e-3, "thermal expansion must produce displacement"
assert vm_last.max() > 0.5, "a thermal stress field must develop"
assert trace_hot < 0, "the constrained hot zone must be in compression"
assert abs(e_stored + e_lost - e_in) / e_in < 0.25, "energy balance: deposited ~ stored + lost"

# --- animate: travelling temperature (top) + travelling von Mises stress (bottom) ---
vmax_T = float(frames[:, :nT].max())
vmf = [von_mises(frames[j])[0] for j in range(len(frames))]
vmax_S = float(np.percentile(np.stack(vmf), 99))
fig, (axT, axS) = plt.subplots(2, 1, figsize=(7.4, 6.2))
tpcT = axT.tripcolor(triT, frames[0, :nT], cmap="inferno", shading="gouraud", vmin=0.0, vmax=vmax_T)
imS = axS.imshow(vmf[0], origin="lower", extent=(0, Lx, 0, Ly), cmap="viridis", vmin=0.0, vmax=vmax_S)
mkT = axT.plot([centres[0]], [y0], "co", ms=7, mec="k")[0]
mkS = axS.plot([centres[0]], [y0], "ro", ms=6, mec="k")[0]
for ax in (axT, axS):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(tpcT, ax=axT, shrink=0.9, label="temperature $T$")
fig.colorbar(imS, ax=axS, shrink=0.9, label="von Mises stress")


def _frame(j):
    tpcT.set_array(frames[j, :nT])
    imS.set_data(vmf[j])
    mkT.set_data([centres[j]], [y0])
    mkS.set_data([centres[j]], [y0])
    axT.set_title(f"Scanning-laser AM — temperature (Pe={v_scan * Lx:g}), frame {j}/{len(frames) - 1}", fontsize=10)
    return tpcT, imS, mkT, mkS


ani = animation.FuncAnimation(fig, _frame, frames=len(frames), interval=120, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "am_laser_thermomechanics_2d.gif", writer="pillow", fps=8, dpi=84)

# --- static figure: quasi-steady T + von Mises, and the Rosenthal wake comparison ---
fig2, (axf, axr) = plt.subplots(1, 2, figsize=(12, 4.4))
tp2 = axf.tripcolor(triT, Tf, cmap="inferno", shading="gouraud", vmin=0.0, vmax=vmax_T)
axf.contour(GX, GY, vm_last, levels=6, colors="w", linewidths=0.5, alpha=0.7)
axf.plot([cx_last], [y0], "co", ms=8, mec="k")
axf.set_aspect("equal")
axf.set_xticks([])
axf.set_yticks([])
axf.set_title("temperature + von Mises contours (the travelling stress)", fontsize=11)
fig2.colorbar(tp2, ax=axf, shrink=0.85, label="temperature $T$")
axr.plot(xi_w, fem_line, "o-", color="#0072B2", ms=4, label="FEM (along scan line)")
axr.plot(xi_w, ros, "-", color="#D55E00", lw=2, label="Rosenthal point source")
axr.set_xlabel(r"moving-frame coordinate $\xi = x - x_{laser}$  (wake)")
axr.set_ylabel("temperature rise $T$")
axr.set_title("trailing thermal wake vs the Rosenthal solution", fontsize=11)
axr.legend(frameon=False, fontsize=9)
fig2.suptitle("Additive manufacturing: scanning-laser thermo-mechanics", fontsize=12)
fig2.tight_layout(rect=(0, 0, 1, 0.95))
fig2.savefig(Path(__file__).parents[2] / "assets" / "am_laser_thermomechanics_2d.png", dpi=130, bbox_inches="tight")
print("\nsaved assets/am_laser_thermomechanics_2d.gif and .png")
