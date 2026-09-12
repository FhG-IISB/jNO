"""The melt pool in 3-D: conduction + Marangoni convection + thermo-elasticity, tack welds.

A faithful port of `melt_pool.py`. Same material, same enthalpy-porosity phase change, same
Tezduyar-Osawa SUPG/PSPG, same Carman-Kozeny drag, same liquid-fraction-gated thermocapillary
traction, same solid-fraction-degraded thermo-elastic block -- with `z` added and the laser a
surface Gaussian in (x, z) rather than a line source in x.

WHY 3-D AT ALL, given the 2-D model is mesh-converged. Two reasons, and only the second is about
the physics. (1) A 2-D melt pool conducts heat in two directions, not three, so it over-predicts
the pool and under-predicts the cooling rate; the residual stress that follows is the quantity we
actually want and it inherits that error. (2) A line source is not a spot: the 2-D "tack weld" is
really an infinite groove in z.

WHAT THIS COSTS, measured before writing it. The assembly build is ~0.62 MB per node after the
int32/collapse fix (it was 1.37), and ~80% of that is transient scratch freed when `jno.fem`
returns. So a 62 GB machine reaches its build ceiling near 100k nodes. The direct LU is the tighter
limit: at 57k dofs its factors already hold 226M nonzeros (3.6 GB, 105x fill), against 2-D's
near-optimal fill. That is the whole reason the PRECOND knob exists here -- in 2-D the monolithic LU
wins by 28x and preconditioning is pointless, and 3-D is where that is expected to invert. This
script is how that gets tested rather than assumed.

    SIZE=tiny|small|med   preset domain+mesh (or set LX_UM/LZ_UM/LY_UM/H_FINE_UM directly)
    PRECOND=lu|blocklu|jacobi   how the Newton tangent is solved
    FLOW, MARANGONI, MECH, MAR_GATE, C_CK, POWER_W, TEND_MS, SPOT_*   as in melt_pool.py
    BUILD_ONLY=1          build the fem, report dofs and RSS, and stop (a cost probe)

NOT ESTABLISHED: any comparison against experiment, and any mesh convergence in 3-D. The 2-D model
is converged (conduction 1.555/1.555/1.556 um at h=8/4/2um); nothing here has been refined twice.
"""

import os
import resource
import sys
import time
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
symgrad, ident = jno.np.symgrad, jno.np.identity
exp, tanh = jno.np.exp, jno.np.tanh

flag = lambda k, d: os.environ.get(k, d) == "1"  # noqa: E731
num = lambda k, d: float(os.environ.get(k, d))  # noqa: E731
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB  # noqa: E731

FLOW, MARANGONI, MECH = flag("FLOW", "1"), flag("MARANGONI", "1"), flag("MECH", "1")
MAR_GATE = flag("MAR_GATE", "1")
BUILD_ONLY = flag("BUILD_ONLY", "0")

# ---- 316L-ish stainless steel, SI (identical to the 2-D script) ---------------------------------
RHO, CP, K_TH = 7000.0, 750.0, 30.0
T_SOL, T_LIQ, T0 = 1690.0, 1730.0, 300.0
L_FUS, MU = num("L_FUS", 2.7e5), 6e-3
E_MOD, NU_P, ALPHA = 200e9, 0.3, 1.7e-5
G_EL = E_MOD / (2.0 * (1.0 + NU_P))
LAM = E_MOD * NU_P / ((1.0 + NU_P) * (1.0 - 2.0 * NU_P))
E_MIN = 1e-6
DGDT = -4.0e-4 * num("MAR_SCALE", 1.0)
NU_F, C_I = MU / RHO, 36.0
C_CK = num("C_CK", 1e10)  # see melt_pool.py: 1e6 does NOT lock the solid, and that is mesh-dependent
EMIS, SIGMA_SB, H_CONV = 0.4, 5.67e-8, 20.0

# ---- geometry and mesh --------------------------------------------------------------------------
# 3-D cost is cubic in the refinement, so the presets exist to make the ladder explicit rather than
# leaving someone to discover the ceiling by being OOM-killed.
_PRESETS = {  # LX, LZ, LY, H_FINE (um)
    "tiny": (400.0, 300.0, 150.0, 25.0),
    "small": (600.0, 400.0, 200.0, 16.0),
    "med": (900.0, 500.0, 250.0, 12.0),
}
_p = _PRESETS[os.environ.get("SIZE", "tiny")]
LX = num("LX_UM", _p[0]) * 1e-6
LZ = num("LZ_UM", _p[1]) * 1e-6
LY = num("LY_UM", _p[2]) * 1e-6
H_FINE = num("H_FINE_UM", _p[3]) * 1e-6
H_COARSE, BAND = num("H_COARSE_UM", 60.0) * 1e-6, num("BAND_UM", 120.0) * 1e-6

POWER, R0, ETA = num("POWER_W", 80.0), 150e-6, 0.35
T_END = num("TEND_MS", 4.0) * 1e-3
NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
DT = T_END / NSTEPS
_csv = lambda k, d: [float(v) for v in os.environ.get(k, d).split(",")]  # noqa: E731
SPOT_X = [v * 1e-6 for v in _csv("SPOT_X_UM", "150,300")]
SPOT_ON = [v * 1e-3 for v in _csv("SPOT_ON_MS", "0.2,2.0")]
SPOT_DWELL = num("SPOT_DWELL_MS", 1.2) * 1e-3

# fine near the top surface where the pool lives, coarse in the cold substrate below
h_of = lambda x, y, z: H_FINE + (H_COARSE - H_FINE) * min(1.0, max(0.0, (LY - y) / BAND))  # noqa: E731
d = jno.Shape.box(0.0, 0.0, 0.0, LX, LY, LZ, size=h_of).domain(time=(0.0, T_END, NSTEPS + 1))
# NOTE the names. `jno.Shape.box` auto-tags back/bottom/boundary/front/left/right/top, and
# re-tagging one of those SILENTLY keeps the auto region -- the user predicate is discarded with no
# error. Naming this "top" put the laser on the box's z=max face instead of y=max, and when the outer
# Dirichlet also pinned that face the whole solve collapsed to T0 (residual 2e-15, caught only by the
# march-did-not-move guard). Use names that cannot collide.
d.tag("surface", lambda x, y, z: y > LY - 1e-9)
d.tag("outer", lambda x, y, z: (y < 1e-9) | (x < 1e-9) | (x > LX - 1e-9) | (z < 1e-9) | (z > LZ - 1e-9))
d.tag("clamp", lambda x, y, z: y < 1e-9)
d.point_region("ppin", (0.5 * LX, 0.0, 0.5 * LZ))

T, S = d.fem_symbols(names=("T", "S"), order=1)
u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), order=1)
p, q = d.fem_symbols(names=("p", "q"), order=1)
xi, yi, zi, ti = d.variable("interior", split=True)
xt, yt, zt, tt = d.variable("surface", split=True)
xf, yf, zf, _ = d.variable("outer", split=True)
xb, yb, zb, _ = d.variable("clamp", split=True)
pn = d.variable("ppin", split=True)[:3]
ci = d.variable("initial", split=True)
ax = [xi, yi, zi]

Ti, Si = T.bind(x=xi, y=yi, z=zi, t=ti), S.bind(x=xi, y=yi, z=zi, t=ti)
Tt_, St = T.bind(x=xt, y=yt, z=zt, t=tt), S.bind(x=xt, y=yt, z=zt, t=tt)
ui, vi = u.bind(x=xi, y=yi, z=zi, t=ti), v.bind(x=xi, y=yi, z=zi, t=ti)
pp, qq = p.bind(x=xi, y=yi, z=zi, t=ti), q.bind(x=xi, y=yi, z=zi, t=ti)
gu, gv, gp = grad(u, ax), grad(v, ax), grad(p, ax)
gq = grad(q, ax)

div = lambda g: trace(g)  # noqa: E731
adv = lambda g, w_: inner(g, w_, n_contract=1)  # noqa: E731
G = d.cell_metric
gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731

# ---- phase change, as formulas ------------------------------------------------------------------
T_MID, T_HALF = 0.5 * (T_SOL + T_LIQ), 0.5 * (T_LIQ - T_SOL)
fl = 0.5 * (1.0 + tanh((Ti - T_MID) / T_HALF))
dfl = 0.5 * (1.0 - tanh((Ti - T_MID) / T_HALF) ** 2) / T_HALF
c_eff = CP + L_FUS * dfl
A_ck = jno.lag(C_CK * (1.0 - fl) ** 2 / (fl**3 + 1e-3))

# ---- energy: storage and advection as SEPARATE terms (the nested spelling is refused) -----------
u_adv = adv(grad(T, ax), ui) if FLOW else 0.0
energy = (
    RHO * c_eff * Ti.t * Si
    + (RHO * c_eff * u_adv * Si if FLOW else 0.0 * Si)
    + K_TH * inner(grad(T, ax), grad(S, ax), n_contract=1)
)
if FLOW:
    alpha_th = K_TH / (RHO * c_eff)
    tau_T = jno.lag((4.0 / DT**2 + gG(ui) + C_I * alpha_th**2 * inner(G, G, n_contract=2)) ** -0.5)
    energy = energy + RHO * c_eff * tau_T * adv(grad(S, ax), ui) * u_adv

# ---- laser: two stationary tack welds, a surface Gaussian in (x, z) -----------------------------
h_rad = jno.lag(EMIS * SIGMA_SB * (Tt_**2 + T0**2) * (Tt_ + T0))
_q0 = 2.0 * ETA * POWER / (np.pi * R0**2)
_gw = 0.15e-3
_gate = lambda t, t0, t1: 0.5 * (tanh((t - t0) / _gw) - tanh((t - t1) / _gw))  # noqa: E731
q_laser = sum(
    _q0 * exp(-2.0 * (((xt - xc) ** 2 + (zt - 0.5 * LZ) ** 2) / R0**2)) * _gate(tt, t0, t0 + SPOT_DWELL)
    for xc, t0 in zip(SPOT_X, SPOT_ON)
)

terms = [
    energy,
    -(q_laser - (h_rad + H_CONV) * (Tt_ - T0)) * St,
    T(xf, yf, zf) - T0,
    T(*ci) - T0,
]

if FLOW:
    mom = (
        RHO * inner(ui.t, vi, n_contract=1)
        + RHO * inner(adv(gu, ui), vi, n_contract=1)
        + MU * inner(gu, gv, n_contract=2)
        - pp * div(gv)
        + A_ck * inner(ui, vi, n_contract=1)
    )
    cont = -qq * div(gu)
    tau = jno.lag(
        (4.0 / DT**2 + gG(ui) + C_I * NU_F**2 * inner(G, G, n_contract=2) + (A_ck / RHO) ** 2) ** -0.5
    )
    r_m = adv(gu, ui) + gp / RHO + (A_ck / RHO) * ui
    mom = mom + RHO * tau * inner(adv(gv, ui), r_m, n_contract=1)
    cont = cont - tau * inner(gq, r_m, n_contract=1)
    terms += [mom, cont]
    terms += [u(xf, yf, zf)[k] - 0.0 for k in range(3)]
    terms += [u(xt, yt, zt)[1] - 0.0, p(*pn) - 0.0]           # flat top: no through-surface flow
    terms += [u(*ci)[k] - 0.0 for k in range(3)]
    if MARANGONI:
        # gated by the liquid fraction: thermocapillary stress is a property of a free LIQUID
        # surface. Ungated it drives the solid -- see melt_pool.py, where the global peak sat at a
        # node BELOW the solidus. Lagged, because the tanh gate's Newton tangent NaNs the march.
        fl_t = jno.lag(0.5 * (1.0 + tanh((Tt_ - T_MID) / T_HALF))) if MAR_GATE else 1.0
        vt = v.bind(x=xt, y=yt, z=zt, t=tt)
        # the surface is flat, so the tangent plane is (e_x, e_z) and the traction has both
        terms.append(-DGDT * fl_t * Tt_.x * vt[0])
        terms.append(-DGDT * fl_t * Tt_.z * vt[2])

if MECH:
    w, phi = d.fem_symbols(value_shape=(3,), names=("w", "phi"), order=1)
    s_deg = jno.lag((1.0 - fl) ** 2 + E_MIN)  # a liquid carries no shear
    eps = lambda zz: symgrad(zz, ax)  # noqa: E731
    I3 = ident(3)
    eps_th = ALPHA * (Ti - T0) * I3
    sigma = (
        2.0 * G_EL * s_deg * eps(w)
        + LAM * s_deg * trace(eps(w)) * I3
        - (3.0 * LAM + 2.0 * G_EL) * s_deg * eps_th
    )
    terms.append(inner(sigma, eps(phi), n_contract=2))
    terms += [w(xb, yb, zb)[k] - 0.0 for k in range(3)]        # clamped base, free top and sides

t_build = time.time()
fem = jno.fem(terms)
pts = np.asarray(fem.points)
print(f"mesh: {len(pts)} nodes, {fem.dofs} dofs, h_fine={H_FINE * 1e6:g}um")
print(f"domain: {LX * 1e6:g} x {LZ * 1e6:g} x {LY * 1e6:g} um   time: {NSTEPS} steps x {DT * 1e6:.2f}us")
print(f"build:  {time.time() - t_build:.1f}s, peak RSS {rss():.2f} GB", flush=True)
if BUILD_ONLY:
    sys.exit(0)

# ---- solver slot ---------------------------------------------------------------------------------
PRECOND = os.environ.get("PRECOND", "lu")
_fields = [T] + ([u, p] if FLOW else []) + ([w] if MECH else [])
if PRECOND == "lu":
    _LINEAR, _PKW = jno.solve.lu(backend="host", reuse=False), {}
elif PRECOND in ("blocklu", "jacobi"):
    _child = (lambda: jno.precond.inner(jno.solve.lu(backend="host"))) if PRECOND == "blocklu" \
        else (lambda: jno.precond.jacobi())
    _LINEAR = jno.solve.fgmres(tol=num("KTOL", 1e-8), restart=int(num("KRESTART", 120)),
                               maxiter=int(num("KMAXIT", 600)))
    _PKW = {"precond": jno.precond.triangular(*[(f, _child()) for f in _fields])}
else:
    raise SystemExit(f"PRECOND={PRECOND!r} is not one of lu/blocklu/jacobi")
print(f"solver: PRECOND={PRECOND} over blocks {[f.name for f in _fields]}", flush=True)

t_start = time.time()
try:
    traj = np.asarray(
        fem.solve(
            nonlinear=jno.solve.newton(
                direct=True, rtol=num("RTOL", 1e-6), atol=num("ATOL", 1e-6),
                line_search=flag("LS", "0"), damping=num("DAMP", 1.0),
                max_steps=int(num("MAXIT", 100)),
            ),
            linear=_LINEAR,
            **_PKW,
        ).fn()
    )
except Exception:
    print(f"\nSOLVE FAILED after {time.time() - t_start:.1f}s:\n", flush=True)
    traceback.print_exc()
    sys.exit(1)
print(f"marched in {time.time() - t_start:.1f}s, peak RSS {rss():.2f} GB", flush=True)

ts = np.linspace(0.0, T_END, NSTEPS + 1)
bT = fem.blocks[fem.block_index(T)]
Tf = traj[:, bT.start : bT.stop]
k = int(np.argmax(Tf.max(axis=1)))
print(f"  peak T          {Tf.max():.0f} K at t={ts[k] * 1e3:.3f} ms   (solidus {T_SOL:.0f})")
print(f"  molten nodes    {int((Tf[k] > T_SOL).sum())} at that step")

if FLOW:
    bu = fem.blocks[fem.block_index(u)]
    U = traj[:, bu.start : bu.stop].reshape(len(traj), -1, 3)
    sp = np.linalg.norm(U, axis=-1)
    liq = 0.5 * (1.0 + np.tanh((Tf[k] - T_MID) / T_HALF)) > 0.99
    print(f"  peak |u|        {sp.max():.4e} m/s  (final step {sp[-1].max():.4e})")
    print(
        f"  |u| in liquid   {np.linalg.norm(U[k][liq], axis=-1).max() if liq.any() else 0.0:.4e} m/s"
        f"   ({int(liq.sum())} fully-liquid nodes)"
    )

if MECH:
    bw = fem.blocks[fem.block_index(w)]
    W = traj[:, bw.start : bw.stop].reshape(len(traj), -1, 3)
    mag = np.linalg.norm(W, axis=-1)
    kk = int(np.argmax(mag.max(axis=1)))
    print(f"  peak |w|        {mag.max() * 1e6:.3f} um at t={ts[kk] * 1e3:.3f} ms")
    print(f"  |w| at the end  {mag[-1].max() * 1e6:.3f} um")
    molten_end = int((Tf[-1] > T_SOL).sum())
    print(
        "  the pool has FROZEN -- the end-state distortion above is RESIDUAL"
        if molten_end == 0
        else f"  the pool has NOT frozen ({molten_end} molten nodes) -- run longer"
    )

np.savez_compressed(
    os.environ.get("OUT", "scratch/melt_pool_3d_out.npz"),
    pts=pts, T=Tf.astype(np.float32), ts=ts,
    **({"U": U.astype(np.float32)} if FLOW else {}),
    **({"W": W.astype(np.float32)} if MECH else {}),
)
print(f"  wrote {os.environ.get('OUT', 'scratch/melt_pool_3d_out.npz')}", flush=True)
