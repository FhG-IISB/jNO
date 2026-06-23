"""15 - Poroelastic consolidation (Biot): pore fluid pressure coupled to a deforming porous solid.

Load a water-saturated soil column on top and it does not settle all at once: the pore water carries the
load instantly (an excess pressure spike), then slowly drains out through the top while the grains take
over and the column consolidates. This is **Biot poroelasticity** — solid displacement ``u`` and pore
pressure ``p`` two-way coupled, one ``jno.fem``:

    div sigma' - alpha grad p = 0,    sigma' = lam tr(e) I + 2 mu e        (solid: effective stress + Biot)
    S dp/dt + alpha d(div u)/dt = kappa lap p                              (fluid: storage + Darcy drainage)

Two fields — displacement ``u`` (P2 vector) and pressure ``p`` (P1) — in a single ``jno.fem([...])``. The
coupling is genuinely two-way: ``-alpha p (div v)`` feeds pressure into the solid balance, and the
**rate of volume change** feeds the solid back into the fluid balance. That second term, ``alpha (div u)_t
q``, is integrated by parts in space to ``-alpha u_t . grad q`` so it uses only first-order derivatives —
and it lands in the **mass matrix as a cross-field (p-u) block**, which is exactly the Biot coupling.

The system is **linear** with a **constant** operator, so we factor ``M + dt A`` once (sparse LU) and
back-substitute each step; the top load enters through ``fem.operator.forcing_vector_fn``. The dissipating
pressure and the settlement curve are checked against the classic **Terzaghi** 1-D consolidation theory.
The animation is the *computed* pressure on the *computed* (settling) mesh; nothing is painted in.

References: K. von Terzaghi, *Erdbaumechanik auf bodenphysikalischer Grundlage*, 1925 (1-D consolidation);
M. A. Biot, "General theory of three-dimensional consolidation", J. Appl. Phys. 12:155-164, 1941.
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
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

symgrad, inner, trace = jno.np.symgrad, jno.np.inner, jno.np.trace

# --- non-dimensional parameters ---
W, H = 0.3, 1.0  # a slender soil column
lam, mu, alpha = 1.0, 1.0, 1.0  # Lame parameters + Biot coefficient
S, kappa = 0.05, 1.0  # storativity + permeability/viscosity (Darcy)
load = 0.1  # surface load on the top (small -> small-strain linear elasticity stays valid)
Kv = lam + 2.0 * mu  # 1-D constrained modulus
cv = kappa / (S + alpha**2 / Kv)  # Terzaghi consolidation coefficient -> dimensionless time Tv = cv t / H^2

# --- coupled fields: displacement (P2 vector) + pore pressure (P1) ---
d = jno.domain(box(0, 0, W, H), mesh_size=0.04, time=(0.0, 0.30, 2))
u, phi = d.fem_symbols(value_shape=(2,), names=("u", "phi"), order=2)
p, q = d.fem_symbols(names=("p", "q"), order=1)
xi, yi, ti = d.variable("interior", split=True)
d.tag("top", lambda x, y: y > H - 1e-6)
d.tag("bottom", lambda x, y: y < 1e-6)
d.tag("left", lambda x, y: x < 1e-6)
d.tag("right", lambda x, y: x > W - 1e-6)
xt, yt, _ = d.variable("top", split=True)
xbo, ybo, _ = d.variable("bottom", split=True)
xl, yl, _ = d.variable("left", split=True)
xr, yr, _ = d.variable("right", split=True)
ci = d.variable("initial", split=True)
ub, vb = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
pb, qb = p.bind(x=xi, y=yi, t=ti), q.bind(x=xi, y=yi, t=ti)
eu, ev = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])

solid = lam * trace(eu) * trace(ev) + 2.0 * mu * inner(eu, ev, n_contract=2) - alpha * pb * trace(ev)
traction = -inner(jno.np.array([0.0, -load]), phi.bind(x=xt, y=yt), n_contract=1)  # downward load on top
# fluid: S p_t + alpha (div u)_t - kappa lap p = 0 ; (div u)_t q --IBP--> -u_t . grad q (boundary terms vanish)
fluid = S * pb.t * qb - alpha * (ub.t[0] * qb.x + ub.t[1] * qb.y) + kappa * (pb.x * qb.x + pb.y * qb.y)
fem = jno.fem(
    [
        solid,
        fluid,
        traction,
        p(xt, yt) - 0.0,  # drained top (free-draining surface)
        u(xbo, ybo)[1] - 0.0,  # impermeable + roller base (u_y = 0)
        u(xl, yl)[0] - 0.0,  # confined sides (u_x = 0) -> 1-D consolidation
        u(xr, yr)[0] - 0.0,
        p(ci[0], ci[1]) - 0.0,  # initial: no excess pressure
        u(ci[0], ci[1]) - 0.0,
    ]
)
assert fem.is_transient and fem.is_linear, "linear transient Biot poroelasticity"
off = fem.problem.offset
nU = int(off[1])  # displacement is field 0: DOFs w[:nU]; pressure is field 1: w[nU:]
pts_u = np.asarray(fem.problem.mesh[0].points)[:, :2]  # P2 displacement nodes
pts_p = np.asarray(fem.problem.mesh[1].points)[:, :2]  # P1 pressure nodes
tris = np.asarray(d.built_mesh.cells_dict["triangle"])
triP = mtri.Triangulation(pts_p[:, 0], pts_p[:, 1], tris)


def _csc(B):  # jax operator (BCOO or dense) -> scipy CSC
    if hasattr(B, "sum_duplicates"):
        B = B.sum_duplicates()
        ij = np.asarray(B.indices)
        return spsp.csc_matrix((np.asarray(B.data), (ij[:, 0], ij[:, 1])), shape=tuple(B.shape))
    return spsp.csc_matrix(np.asarray(B))


# --- the Biot coupling lives in the mass matrix (cross-field p-u block); factor M + dt*A ONCE ---
M, A = _csc(fem.M), _csc(fem.operator.A)
b = np.asarray(fem.operator.forcing_vector_fn(0.0, {}))  # constant top-load vector
dt, nsteps, nframes = 0.005, 60, 20
lu = splu((M + dt * A).tocsc())
coupling = abs(M[nU:, :nU]).sum()
print(
    f"\nBiot consolidation: dofs={fem.dofs} (u {nU} + p {fem.dofs - nU}), cv={cv:.3f}, ||M[p,u] coupling||={coupling:.2f}"
)
assert coupling > 1e-6, "the Biot (div u)_t coupling must populate the cross-field mass block"

w = np.asarray(fem.state0)
frames, times = [w.copy()], [0.0]
for step in range(nsteps):
    w = lu.solve(M.dot(w) + dt * b)
    if (step + 1) % max(1, nsteps // nframes) == 0:
        frames.append(w.copy())
        times.append((step + 1) * dt)
frames = np.stack(frames)

# --- diagnostics + Terzaghi validation ---
P = frames[:, nU:]  # pressure history (P1 nodes)
y_p = pts_p[:, 1]
p0a = load * alpha / (Kv * S + alpha**2)  # undrained (Skempton) uniform excess pressure at t=0+
p_mid = np.array([np.interp(H / 2, *zip(*sorted(zip(y_p, Pk)))) for Pk in P])  # pressure at mid-depth vs time


def terzaghi_U(Tv, nterms=60):  # degree of consolidation (average pressure dissipation)
    return 1.0 - sum(
        (2.0 / ((2 * m + 1) * np.pi / 2) ** 2) * np.exp(-(((2 * m + 1) * np.pi / 2) ** 2) * Tv) for m in range(nterms)
    )


def terzaghi_iso(z_over_H, Tv, nterms=60):  # pressure isochrone p/p0 vs depth from the drained top
    return sum(
        (2.0 / ((2 * m + 1) * np.pi / 2))
        * np.sin((2 * m + 1) * np.pi / 2 * z_over_H)
        * np.exp(-(((2 * m + 1) * np.pi / 2) ** 2) * Tv)
        for m in range(nterms)
    )


# degree of consolidation from average excess pressure: U = 1 - <p(t)>/p0  (p0 = uniform undrained value)
avg_p = P.mean(axis=1)
U_fem = 1.0 - avg_p / p0a
Tv = cv * np.array(times) / H**2
U_terz = np.array([terzaghi_U(t) if t > 0 else 0.0 for t in Tv])
err_U = np.max(np.abs(U_fem[2:] - U_terz[2:]))  # skip t=0 and the first (undrained) step
settle = -np.array([np.interp(H, *zip(*sorted(zip(pts_u[:, 1], wk[:nU].reshape(-1, 2)[:, 1])))) for wk in frames])
print(f"  p_mid: {p_mid[1]:.3f} -> {p_mid[-1]:.3f} (dissipating); top settlement: {settle[1]:.4f} -> {settle[-1]:.4f}")
print(f"  degree of consolidation U vs Terzaghi: max abs error = {err_U:.3f}")

assert np.all(np.diff(p_mid[1:]) < 1e-9), "excess pore pressure must dissipate monotonically"
assert p_mid[1] > 0.5 * p0a, "loading must create a positive excess pore-pressure spike (~the undrained value)"
assert settle[-1] > settle[1] > 0, "the column must settle (and keep settling as it drains)"
assert err_U < 0.06, "degree of consolidation must match Terzaghi 1-D theory"

# --- animate: pore pressure on the settling column (displacement exaggerated) ---
EXAG = 6.0  # exaggerate the (small) settlement so it is visible
pmax = float(P.max())


def deformed(wk):
    ux = griddata(pts_u, wk[:nU].reshape(-1, 2)[:, 0], pts_p, method="linear", fill_value=0.0)
    uy = griddata(pts_u, wk[:nU].reshape(-1, 2)[:, 1], pts_p, method="linear", fill_value=0.0)
    return mtri.Triangulation(pts_p[:, 0] + EXAG * ux, pts_p[:, 1] + EXAG * uy, tris)


fig, ax = plt.subplots(figsize=(3.6, 5.6))
tpc = ax.tripcolor(deformed(frames[0]), P[0], cmap="viridis", shading="gouraud", vmin=0.0, vmax=pmax)
ax.set_xlim(-0.15, W + 0.15)
ax.set_ylim(-0.02, H + 0.05)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(tpc, ax=ax, shrink=0.8, label="excess pore pressure $p$")


def _frame(j):
    ax.clear()
    ax.tripcolor(deformed(frames[j]), P[j], cmap="viridis", shading="gouraud", vmin=0.0, vmax=pmax)
    ax.set_xlim(-0.15, W + 0.15)
    ax.set_ylim(-0.02, H + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Consolidation: pore pressure\ndrains + column settles, t={times[j]:.2f}", fontsize=10)


ani = animation.FuncAnimation(fig, _frame, frames=len(frames), interval=130, blit=False)
ani.save(Path(__file__).parents[2] / "assets" / "poroelastic_consolidation_2d.gif", writer="pillow", fps=8, dpi=86)

# --- static figure: pressure isochrones vs Terzaghi + the consolidation curve ---
fig2, (axi, axu) = plt.subplots(1, 2, figsize=(11, 4.8))
zc = np.linspace(0, 1, 100)  # depth from the drained top, z/H
order = np.argsort(y_p)
sel = [1, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1]
colors = plt.cm.viridis(np.linspace(0, 0.85, len(sel)))
for c, j in zip(colors, sel):
    axi.plot(P[j][order] / p0a, H - y_p[order], "o", color=c, ms=2.5, alpha=0.4)  # FEM (depth from drained top)
    axi.plot(terzaghi_iso(zc, Tv[j]), zc * H, "-", color=c, lw=1.8, label=f"Tv={Tv[j]:.2f}")
axi.set_xlabel(r"normalised excess pressure $p/p_0$")
axi.set_ylabel(r"depth from drained top $z$")
axi.invert_yaxis()
axi.set_title("pressure isochrones: FEM (dots) vs Terzaghi (lines)", fontsize=11)
axi.legend(frameon=False, fontsize=8)
axu.plot(Tv[1:], U_fem[1:], "o", color="#0072B2", ms=5, label="FEM")
axu.plot(
    np.linspace(Tv[1], Tv[-1], 80),
    [terzaghi_U(t) for t in np.linspace(Tv[1], Tv[-1], 80)],
    "-",
    color="#D55E00",
    lw=2,
    label="Terzaghi",
)
axu.set_xlabel(r"dimensionless time $T_v = c_v t / H^2$")
axu.set_ylabel("degree of consolidation $U$")
axu.set_title("consolidation curve", fontsize=11)
axu.legend(frameon=False, fontsize=9)
fig2.suptitle("Poroelastic (Biot) consolidation of a loaded soil column", fontsize=12)
fig2.tight_layout(rect=(0, 0, 1, 0.95))
fig2.savefig(Path(__file__).parents[2] / "assets" / "poroelastic_consolidation_2d.png", dpi=130, bbox_inches="tight")
print("\nsaved assets/poroelastic_consolidation_2d.gif and .png")
