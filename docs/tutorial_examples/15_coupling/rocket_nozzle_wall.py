# --8<-- [start:code]
"""01 - A rocket nozzle's wall heated and pressed by its gas: jno.fvm and jno.fem on their own meshes.

The gas: the axisymmetric Euler equations in the (x, r) meridian of a converging-diverging nozzle, a hot chamber
feeding a supersonic exit (jno.fvm, explicit, its own CFL substeps). The wall: a shell around it, with transient
conduction and quasi-static thermo-elasticity, hoop strain included (jno.fem), cooled on its outside.

They meet on the liner through two conditions, both loads on the wall (the gas hears nothing back):

    k_w dT_w/dn = h(q_g) (T_aw(q_g) - T_w)      a heat-transfer correlation (Bartz-like: h ~ (rho u)^0.8),
    sigma(u, T) . n + p_g n = 0                  the gas pressure.

Nondimensional units (chamber pressure 1, gas constant 1); the burn is short so the example runs in minutes.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

GAMMA, CP = 1.4, 1.4 / 0.4  # R = 1
T_BURN, N_TIME = 4.0, 9  # coupling windows of 0.5
P_CH, T_CH = 1.0, 3.0  # chamber pressure and temperature
H0, RECOVERY = 1.0, 0.9  # the correlation's scale and recovery factor
K_W, RHOC_W, H_COOL, T_COOL = 1.0, 0.5, 2.0, 1.0  # wall conductivity, heat capacity, coolant
E_W, NU_W, ALPHA_W = 100.0, 0.3, 1e-2  # wall stiffness, Poisson ratio, expansion
LAM, MU = E_W * NU_W / ((1 + NU_W) * (1 - 2 * NU_W)), E_W / (2 * (1 + NU_W))

# the contour R(x): chamber, converging cone, throat, diverging cone
CONTOUR = [(0.0, 1.0), (0.6, 1.0), (1.2, 0.5), (2.4, 0.8)]
THICK = 0.2

# ---- the gas ------------------------------------------------------------------------------------------------
gas_shape = jno.shape.polygon([(0.0, 0.0), (2.4, 0.0), *CONTOUR[::-1][:-1], (0.0, 1.0)], size=0.04)
dg = gas_shape.domain(time=(0.0, T_BURN, N_TIME))
dg.tag("liner", lambda x, n, name: np.isin(np.asarray(name), ["e2", "e3", "e4"]))  # the three contour edges
x, r, t = dg.variable("interior", split=True)
xi, ri, _ = dg.variable("initial", split=True)
q = dg.unknown.vector(4)  # (rho, m_x, m_r, E)
qi = q.bind(x=x, y=r, t=t)
rho, mx, mr, En = qi[0], qi[1], qi[2], qi[3]
ux, ur = mx / rho, mr / rho
p = (GAMMA - 1.0) * (En - 0.5 * (mx * ux + mr * ur))
fv = jno.fv(flux="hll", limiter="minmod")
Fx = jnn.stack([mx, mx * ux + p, mr * ux, ux * (En + p)], axis=-1)
Fr = jnn.stack([mr, mx * ur, mr * ur + p, ur * (En + p)], axis=-1)
hoop = jnn.stack([0.0 * p, 0.0 * p, p, 0.0 * p], axis=-1)  # the pressure on the meridian's ring element
rho_ch = P_CH / T_CH
U_CH = 0.45  # fast enough that the throat chokes
chamber = [rho_ch, rho_ch * U_CH, 0.0, P_CH / (GAMMA - 1.0) + 0.5 * rho_ch * U_CH**2]


def state(c, like):
    return jnn.stack([c[k] + 0.0 * like for k in range(4)], axis=-1)


gas_cons = [r * qi.t + (r * Fx).d(x, scheme=fv) + (r * Fr).d(r, scheme=fv) - hoop]
xin, rin, _ = dg.variable("e5", split=True)  # the chamber inflow, x = 0
gas_cons.append(q(xin, rin) - state(chamber, xin))
for tag in ("e0", "liner"):  # the axis and the liner: slip
    xe, re, _ = dg.variable(tag, split=True)
    ne = dg.variable(tag, normals=True)
    qe = q.bind(x=xe, y=re)
    gas_cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
gas_cons.append(q(xi, ri) - state(chamber, xi))  # e1, the exit: no condition (supersonic outflow)
gas = jno.fvm(gas_cons)

# what the wall reads of the gas, on the liner
xl, rl, _ = dg.variable("liner", split=True)
ql = q.bind(x=xl, y=rl)
pl = (GAMMA - 1.0) * (ql[3] - 0.5 * (ql[1] * ql[1] + ql[2] * ql[2]) / ql[0])
speed2 = (ql[1] * ql[1] + ql[2] * ql[2]) / (ql[0] * ql[0])
T_aw = pl / ql[0] + RECOVERY * speed2 / (2.0 * CP)  # the adiabatic-wall (recovery) temperature
h_gas = H0 * (jnn.sqrt(ql[1] * ql[1] + ql[2] * ql[2]) + 1e-6) ** 0.8  # Bartz-like scaling with the mass flux

# ---- the wall -----------------------------------------------------------------------------------------------
outer = [(xx, rr + THICK) for xx, rr in CONTOUR]
dw = jno.shape.polygon([*CONTOUR, *outer[::-1]], size=0.03).domain(time=(0.0, T_BURN, N_TIME))
dw.tag("bore", lambda x, n, name: np.isin(np.asarray(name), ["e0", "e1", "e2"]))
dw.tag("outside", lambda x, n, name: np.isin(np.asarray(name), ["e4", "e5", "e6"]))
T, S = dw.fem_symbols(names=("T", "S"), order=1)
U, V = dw.fem_symbols(value_shape=(2,), names=("U", "V"), order=1)  # (u_x, u_r)
xw, rw, tw = dw.variable("interior", split=True)
cw = dw.variable("initial", split=True)
Tb, Sb = T.bind(x=xw, y=rw, t=tw), S.bind(x=xw, y=rw, t=tw)
Ub, Vb = U.bind(x=xw, y=rw, t=tw), V.bind(x=xw, y=rw, t=tw)


def strains(w, rr):
    """(e_xx, e_rr, e_qq, e_xr) of an axisymmetric displacement (u_x, u_r): the hoop strain u_r/r has no 2-D part."""
    return w.x[0], w.y[1], w[1] / rr, 0.5 * (w.y[0] + w.x[1])


eu, ev = strains(Ub, rw), strains(Vb, rw)
tr_u, tr_v = eu[0] + eu[1] + eu[2], ev[0] + ev[1] + ev[2]
thermal = (3 * LAM + 2 * MU) * ALPHA_W * (Tb - T_COOL)
energy = LAM * tr_u * tr_v + 2 * MU * (eu[0] * ev[0] + eu[1] * ev[1] + eu[2] * ev[2] + 2 * eu[3] * ev[3]) - thermal * tr_v
ring = 2.0 * np.pi * rw
xo, ro, _ = dw.variable("outside", split=True)
xh, rh, _ = dw.variable("e7", split=True)  # the head end, x = 0: held axially
wall = jno.fem(
    [
        (RHOC_W * Tb.t * Sb + K_W * (Tb.x * Sb.x + Tb.y * Sb.y)) * ring,
        energy * ring,
        H_COOL * (T.bind(x=xo, y=ro) - T_COOL) * S.bind(x=xo, y=ro) * (2.0 * np.pi * ro),  # the coolant
        U(xh, rh)[0] - 0.0,
        T(*cw) - T_COOL,
    ]
)

# what the gas reads of the wall, on the bore
xs, rs, _ = dw.variable("bore", split=True)
ns = dw.variable("bore", normals=True)
Ts, Us = T.bind(x=xs, y=rs), U.bind(x=xs, y=rs)
es = strains(Us, rs)
tr_s = es[0] + es[1] + es[2]
sig_xx = LAM * tr_s + 2 * MU * es[0] - (3 * LAM + 2 * MU) * ALPHA_W * (Ts - T_COOL)
sig_rr = LAM * tr_s + 2 * MU * es[1] - (3 * LAM + 2 * MU) * ALPHA_W * (Ts - T_COOL)
sig_xr = 2 * MU * es[3]
traction = jnn.stack([sig_xx * ns[..., 0] + sig_xr * ns[..., 1], sig_xr * ns[..., 0] + sig_rr * ns[..., 1]], axis=-1)

# ---- the coupling -------------------------------------------------------------------------------------------
core = jno.core(
    [
        gas,
        wall,
        K_W * Ts.d(ns) - h_gas * (T_aw - Ts),  # the heat the gas puts into the wall
        traction + pl * ns,  # the pressure it puts on it
    ]
)
spec = jno.solve.coupling(
    measure="axisymmetric",  # r is the second coordinate: interface weights and the wall's loads carry 2 pi r
    time={gas: jno.solve.erk("ssprk3", cfl=0.5), wall: None},
)
traj_gas, traj_wall = core.solve(coupling=spec)
# --8<-- [end:code]

if __name__ == "__main__":
    c = core.coupled
    print(f"windows: {len(c.window_iterations)}, sweeps per window: {set(c.window_iterations)} (one-way loads)")
    g = np.asarray(traj_gas[-1]).reshape(4, -1)
    cen = np.asarray(gas.centroids)
    exit_cells = cen[:, 0] > 2.3
    mach = np.sqrt((g[1] ** 2 + g[2] ** 2) / g[0] ** 2) / np.sqrt(
        GAMMA * (GAMMA - 1) * (g[3] - 0.5 * (g[1] ** 2 + g[2] ** 2) / g[0]) / g[0]
    )
    print(f"exit Mach {mach[exit_cells].mean():.2f}")
    sl_T = wall.blocks[wall.block_index(T)]
    sl_U = wall.blocks[wall.block_index(U)]
    w = np.asarray(traj_wall)
    print(f"wall temperature: {w[0, sl_T.start : sl_T.stop].max():.3f} -> {w[-1, sl_T.start : sl_T.stop].max():.3f}")
    disp = w[-1, sl_U.start : sl_U.stop].reshape(-1, 2)
    print(f"largest radial displacement {np.abs(disp[:, 1]).max():.2e}")
