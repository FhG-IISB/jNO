"""A scanning laser on a steel plate: melting, thermal stress, and the distortion left behind.

The chain, as one ``jno.fem`` term list marched once:

    laser -> conduction with latent heat -> a melt pool -> thermo-elastic stress -> cooling ->
    residual distortion

Two physics, two blocks:

* **energy** ``rho c_eff(T) T_t = div(k grad T)``, the laser entering as a surface flux on the top.
  Latent heat rides in an APPARENT HEAT CAPACITY, ``c_eff = c_p + L_f dfl/dT``, with the liquid
  fraction ``fl`` a smooth ramp through the mushy range -- a formula, so it goes straight in the term
  list with no phase-change machinery anywhere;
* **equilibrium** ``div sigma = 0``, quasi-static, with the thermo-elastic stress
  ``sigma = 2G eps(w) + lam tr(eps(w)) I - (3 lam + 2G) alpha (T - T0) I``.

What makes this a melt-pool model rather than a heated plate is the **stiffness degradation**: a
liquid carries no shear, so the moduli are scaled by the solid fraction squared. The melt goes soft,
cannot support the thermal strain it is under, and the stress that survives once the pool freezes is
the residual stress -- which is the thing anyone actually wants out of this model.

**The mesh is graded**, and it has to be. The pool is ~50 um deep in a 400 um domain, so a uniform
mesh fine enough to resolve it wastes almost every element: measured, 4 um uniform is 35,226 nodes
against 2,035 graded, at the same near-surface resolution. ``size=`` takes a callable ``f(x, y, z)``
-- three coordinates in 2-D as well as 3-D -- which becomes a gmsh mesh-size callback.

Quasi-static is safe here by four orders of magnitude: an elastic wave crosses the 1.2 mm domain in
~0.2 us against a 4 ms process, so equilibrium is reached well within a time step.

SCOPE, up front. This is CONDUCTION MODE: the top surface stays flat, so there is no depression, no
humping and no keyhole, and there is no evaporation or recoil pressure. There is also no melt
CONVECTION here -- see the note at the bottom of the tutorial page, which reports what was measured
about Marangoni flow and why it is not switched on.
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, grad, trace, symgrad, ident = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.symgrad, jno.np.identity
exp, tanh = jno.np.exp, jno.np.tanh

# ---- 316L-ish stainless steel, SI --------------------------------------------------------------
RHO, CP, K_TH = 7000.0, 750.0, 30.0
T_SOL, T_LIQ, L_FUS, T0 = 1690.0, 1730.0, 2.7e5, 300.0
E_MOD, NU, ALPHA = 200e9, 0.3, 1.7e-5
G_EL = E_MOD / (2.0 * (1.0 + NU))
LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))  # plane strain
E_MIN = 1e-6  # residual stiffness of the melt: keeps the block invertible where the solid fraction is 0
# ---- laser and geometry ------------------------------------------------------------------------
POWER, R0, ETA, VSCAN = 105.0, 150e-6, 0.35, 0.15  # W, m, -, m/s
X_START, LX, LY = 0.3e-3, 1.2e-3, 0.4e-3
EMIS, SIGMA_SB, H_CONV = 0.4, 5.67e-8, 20.0
# ---- mesh grading and time ---------------------------------------------------------------------
H_FINE, H_COARSE, BAND = 4e-6, 30e-6, 100e-6
T_END = float(os.environ.get("TEND_MS", 20.0)) * 1e-3  # long enough for the pool to FREEZE
NSTEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

# Three arguments, not two: gmsh calls a mesh-size function as f(x, y, z) whatever the dimension.
h_of = lambda x, y, z: H_FINE + (H_COARSE - H_FINE) * min(1.0, max(0.0, (LY - y) / BAND))  # noqa: E731

d = jno.shape.rect(0.0, 0.0, LX, LY, size=h_of).domain(time=(0.0, T_END, NSTEPS + 1))
d.tag("top", lambda x, y: y > LY - 1e-9)
d.tag("far", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > LX - 1e-9))
d.tag("base", lambda x, y: y < 1e-9)

T, S = d.fem_symbols(names=("T", "S"), order=1)
w, phi = d.fem_symbols(value_shape=(2,), names=("w", "phi"), order=1)
xi, yi, ti = d.variable("interior", split=True)
xt, yt, tt = d.variable("top", split=True)
xf, yf, _ = d.variable("far", split=True)
xb, yb, _ = d.variable("base", split=True)
ci = d.variable("initial", split=True)
ax = [xi, yi]

Ti, Si = T.bind(x=xi, y=yi, t=ti), S.bind(x=xi, y=yi, t=ti)
St, Tt_ = S.bind(x=xt, y=yt, t=tt), T.bind(x=xt, y=yt, t=tt)

# ---- phase change, written as formulas ---------------------------------------------------------
T_MID, T_HALF = 0.5 * (T_SOL + T_LIQ), 0.5 * (T_LIQ - T_SOL)
fl = 0.5 * (1.0 + tanh((Ti - T_MID) / T_HALF))  # liquid fraction
dfl = 0.5 * (1.0 - tanh((Ti - T_MID) / T_HALF) ** 2) / T_HALF  # its derivative: the latent-heat spike
c_eff = CP + L_FUS * dfl

# ---- constitutive law --------------------------------------------------------------------------
# Solid fraction squared, LAGGED. It spans six decades across a 40 K mushy range, and its Newton
# tangent is what makes the coupled solve stiff; freezing it within each linearisation is the
# standard Picard treatment (the same one the radiation coefficient below needs).
s_deg = jno.lag((1.0 - fl) ** 2 + E_MIN)
eps = lambda u_: symgrad(u_, ax)  # noqa: E731
I2 = ident(2)
eps_th = ALPHA * (Ti - T0) * I2
sigma = 2.0 * G_EL * s_deg * eps(w) + LAM * s_deg * trace(eps(w)) * I2 - (3.0 * LAM + 2.0 * G_EL) * s_deg * eps_th

# ---- surface flux ------------------------------------------------------------------------------
# Radiation as a LAGGED heat-transfer coefficient h_rad(T)(T - T0), an exact rewrite of the quartic.
# Newton on the raw sigma(T^4 - T0^4) is unstable here: one overshooting iterate sends T^4 to infinity
# and the solve diverges -- measured, 2.8e8 K. Lagging the coefficient is stable and 2x faster.
h_rad = jno.lag(EMIS * SIGMA_SB * (Tt_**2 + T0**2) * (Tt_ + T0))
q_laser = (2.0 * ETA * POWER / (np.pi * R0**2)) * exp(-2.0 * ((xt - (X_START + VSCAN * tt)) / R0) ** 2)

fem = jno.fem(
    [
        RHO * c_eff * Ti.t * Si + K_TH * inner(grad(T, ax), grad(S, ax), n_contract=1),
        -(q_laser - (h_rad + H_CONV) * (Tt_ - T0)) * St,  # laser in, radiation + convection out
        inner(sigma, eps(phi), n_contract=2),  # div sigma = 0, weakly; top and sides traction-free
        T(xf, yf) - T0,  # far field at ambient
        w(xb, yb)[0] - 0.0,  # the plate is clamped to its base
        w(xb, yb)[1] - 0.0,
        T(*ci) - T0,
    ]
)
print(
    f"dofs={fem.dofs}  fields={len(fem.blocks)}  nodes={len(np.asarray(fem.points))}  "
    f"steps={NSTEPS}  dt={T_END / NSTEPS * 1e6:.1f}us",
    flush=True,
)

t0 = time.time()
traj = np.asarray(
    fem.solve(
        # A sparse-direct tangent: the phase-change stiffness contrast is exactly the case
        # jno.solve.newton(direct=True) documents, and a matrix-free Krylov inner solve stalls on it.
        nonlinear=jno.solve.newton(direct=True, rtol=1e-6, atol=1e-6),
        linear=jno.solve.lu(backend="host"),
    ).fn()
)
print(f"  marched in {time.time() - t0:.1f}s", flush=True)

# ---- read the answer ---------------------------------------------------------------------------
pts = np.asarray(fem.points)
Tf = traj[:, fem.blocks[fem.block_index(T)].start : fem.blocks[fem.block_index(T)].stop]
Wf = traj[:, fem.blocks[fem.block_index(w)].start : fem.blocks[fem.block_index(w)].stop].reshape(len(traj), -1, 2)
mag = np.linalg.norm(Wf, axis=-1)
top = pts[:, 1] > LY - 1e-9
ts = np.linspace(0.0, T_END, NSTEPS + 1)
t_exit = (LX - X_START) / VSCAN
k_hot, k_def = int(np.argmax(Tf.max(axis=1))), int(np.argmax(mag.max(axis=1)))

molten = Tf > T_SOL
print(f"  peak T {Tf.max():.0f} K at t={ts[k_hot] * 1e3:.2f} ms  (solidus {T_SOL:.0f}, boiling ~3100)")
print(f"  molten nodes, max {molten.sum(axis=1).max()}   (the graded mesh is what resolves the pool)")
if molten.any():
    hot = pts[molten[k_hot]]
    print(
        f"  pool at that step: {(hot[:, 0].max() - hot[:, 0].min()) * 1e6:.0f} um long, "
        f"{(LY - hot[:, 1].min()) * 1e6:.0f} um deep  ({(LY - hot[:, 1].min()) / H_FINE:.0f} cells through the depth)"
    )
print(f"  beam leaves the domain at {t_exit * 1e3:.1f} ms")
print(
    f"  final T max {Tf[-1].max():.0f} K, molten nodes {(Tf[-1] > T_SOL).sum()}  -- the pool has "
    f"{'FROZEN' if (Tf[-1] > T_SOL).sum() == 0 else 'NOT frozen'}"
)
print(
    f"  peak |displacement| {mag.max() * 1e6:.3f} um at t={ts[k_def] * 1e3:.2f} ms "
    f"(surface {Wf[k_def][top, 1].max() * 1e6:+.3f} um)"
)
print(
    f"  RESIDUAL |displacement| once cooled: {mag[-1].max() * 1e6:.3f} um   "
    f"surface {Wf[-1][top, 1].min() * 1e6:+.3f} to {Wf[-1][top, 1].max() * 1e6:+.3f} um"
)
# The hot bulge is a check, not just a number. Free thermal expansion over the heated depth is
# alpha * dT * L, and the heated depth at time t is the thermal diffusion length sqrt(alpha t) -- NOT
# the mesh's grading band, which is an arbitrary numerical choice. dT is bounded by the peak rise, so
# this is an upper estimate and the computed rise should sit below it, same order.
alpha_th = K_TH / (RHO * CP)
L_therm = np.sqrt(alpha_th * ts[k_def])
print(
    f"  check -- free expansion alpha*dT*L over the thermal depth sqrt(alpha t) = {L_therm * 1e6:.0f} um: "
    f"<= {ALPHA * (Tf[k_def].max() - T0) * L_therm * 1e6:.2f} um against a computed surface rise of "
    f"{Wf[k_def][top, 1].max() * 1e6:.2f} um"
)
if len(sys.argv) > 2:
    np.savez_compressed(
        sys.argv[2],
        pts=pts,
        cells=np.asarray(d._cells_p1()),
        T=Tf.astype(np.float32),
        W=Wf.astype(np.float32),
        ts=ts,
        LX=LX,
        LY=LY,
        T_SOL=T_SOL,
        T_LIQ=T_LIQ,
    )
