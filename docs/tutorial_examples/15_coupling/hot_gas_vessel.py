# --8<-- [start:code]
"""02 - A hot gas cooling in a thick-walled vessel: a temperature match, a heat-flux balance and a pressure load.

The gas (jno.fvm): the Euler equations with heat conduction, at rest in a long cylinder r < a; its temperature
T = p/rho (R = 1) is a fifth, algebraic unknown, so it can take a wall temperature. The wall (jno.fem), a < r < b:
transient conduction and plane-strain thermo-elasticity, insulated outside. A quarter of the section, with symmetry
on both cuts. Hot gas over a thermal layer that already meets the wall's temperature cools into the wall.

On the bore the two meet through three conditions, written as for one problem:

    Tg - Ts = 0                              the temperatures agree,
    k_g dTg/dn_g + k_w dTs/dn_s = 0          what leaves the gas enters the wall,
    sigma(u, Ts) . n_s + p_g n_s = 0         the gas pressure loads the wall.

Nondimensional units; the gas's conductivity stands for the heat transfer of a resolved layer.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

GAMMA = 1.4
A, B = 0.5, 1.0  # bore and outer radius
K_G, K_W, RHOC_W = 0.1, 1.0, 1.0  # gas and wall conductivity, wall heat capacity
E_W, NU_W, ALPHA_W = 100.0, 0.3, 1e-2  # wall stiffness, Poisson ratio, expansion
LAM, MU = E_W * NU_W / ((1 + NU_W) * (1 - 2 * NU_W)), E_W / (2 * (1 + NU_W))
P0, T_CORE, T_WALL = 3.0, 3.0, 1.0  # the gas pressure, core and wall temperature
T_END, WINDOWS, GAS_STEPS = 1.0, 10, 20  # the wall steps once per window, the gas GAS_STEPS times
H = 0.04

# ---- the gas ------------------------------------------------------------------------------------------------
quarter = jno.shape.disk(0.0, 0.0, A) & jno.shape.rect(0.0, 0.0, 2.0, 2.0)
dg = quarter.sized(H).domain(time=(0.0, T_END, WINDOWS * GAS_STEPS + 1))
x, y, t = dg.variable("interior", split=True)
xi, yi, _ = dg.variable("initial", split=True)
q, T = dg.unknown.vector(4), dg.unknown()  # (rho, m_x, m_y, E) and the temperature
qi, Ti = q.bind(x=x, y=y, t=t), T.bind(x=x, y=y, t=t)
rho, mx, my, En = qi[0], qi[1], qi[2], qi[3]
u, v = mx / rho, my / rho
p = (GAMMA - 1.0) * (En - 0.5 * (mx * u + my * v))
fv = jno.fv(flux="hll", limiter="minmod")
Fx = jnn.stack([mx, mx * u + p, my * u, u * (En + p)], axis=-1)
Fy = jnn.stack([my, mx * v, my * v + p, v * (En + p)], axis=-1)
# the conduction, in the energy row and in divergence form: a face flux, so the gas conserves its energy
zx, zy = 0.0 * Ti.x, 0.0 * Ti.y
Qx = jnn.stack([zx, zx, zx, -K_G * Ti.x], axis=-1)
Qy = jnn.stack([zy, zy, zy, -K_G * Ti.y], axis=-1)
fd = "finite_difference"
gas_cons = [
    qi.t + Fx.d(x, scheme=fv) + Fy.d(y, scheme=fv) + Qx.d(x, scheme=fd) + Qy.d(y, scheme=fd),
    Ti - p / rho,  # T = p/rho: no time derivative, solved in every cell with the march
]
for tag in ("left", "bottom", "arc"):  # slip: the symmetry cuts and the bore
    xe, ye, _ = dg.variable(tag, split=True)
    ne = dg.variable(tag, normals=True)
    qe = q.bind(x=xe, y=ye)
    gas_cons.append(qe[1] * ne[..., 0] + qe[2] * ne[..., 1])
# at rest at one pressure: a hot core over a thermal layer that meets the wall at its temperature
layer = 0.5 * (1.0 - jnn.tanh((jnn.sqrt(xi * xi + yi * yi) - (A - 0.15)) / 0.04))
T0 = T_WALL + (T_CORE - T_WALL) * layer
gas_cons += [q(xi, yi) - jnn.stack([P0 / T0, 0.0 * xi, 0.0 * xi, P0 / (GAMMA - 1.0) + 0.0 * xi], axis=-1)]
gas_cons += [T(xi, yi) - T0]
gas = jno.fvm(gas_cons)

# the gas on the bore
xg, yg, _ = dg.variable("arc", split=True)
ng = dg.variable("arc", normals=True)
Tg, qg = T.bind(x=xg, y=yg), q.bind(x=xg, y=yg)
pg = (GAMMA - 1.0) * (qg[3] - 0.5 * (qg[1] * qg[1] + qg[2] * qg[2]) / qg[0])

# ---- the wall -----------------------------------------------------------------------------------------------
inner, trace, symgrad, ident, grad = jno.np.inner, jno.np.trace, jno.np.symgrad, jno.np.identity, jno.np.grad
ring = (jno.shape.disk(0.0, 0.0, B) - jno.shape.disk(0.0, 0.0, A)) & jno.shape.rect(0.0, 0.0, 2.0, 2.0)
dw = ring.sized(H).domain(time=(0.0, T_END, WINDOWS + 1))
dw.tag("bore", lambda x, n, name: (np.asarray(name) == "arc") & (np.hypot(x[..., 0], x[..., 1]) < 0.5 * (A + B)))
Tw, S = dw.fem_symbols(names=("T", "S"), order=1)
w, phi = dw.fem_symbols(value_shape=(2,), names=("w", "phi"), order=1)
xw, yw, tw = dw.variable("interior", split=True)
ax, I2 = [xw, yw], ident(2)


def sigma(wv, Tv, axes):
    """The plane-strain thermo-elastic stress."""
    e = symgrad(wv, axes)
    return 2.0 * MU * e + LAM * trace(e) * I2 - (3.0 * LAM + 2.0 * MU) * ALPHA_W * (Tv - T_WALL) * I2


Tb, Sb = Tw.bind(x=xw, y=yw, t=tw), S.bind(x=xw, y=yw, t=tw)
xl, yl, _ = dw.variable("left", split=True)
xb, yb, _ = dw.variable("bottom", split=True)
wall = jno.fem(
    [
        RHOC_W * Tb.t * Sb + K_W * inner(grad(Tw, ax), grad(S, ax), n_contract=1),
        inner(sigma(w, Tw.bind(x=xw, y=yw), ax), symgrad(phi, ax), n_contract=2),
        w(xl, yl)[0] - 0.0,  # symmetry rollers
        w(xb, yb)[1] - 0.0,
        Tw(*dw.variable("initial", split=True)) - T_WALL,
    ]
)

# the wall on the bore
xs, ys, _ = dw.variable("bore", split=True)
ns = dw.variable("bore", normals=True)
Ts = Tw.bind(x=xs, y=ys)
traction = inner(sigma(w.bind(x=xs, y=ys), Ts, [xs, ys]), ns, n_contract=1)

# ---- the coupling -------------------------------------------------------------------------------------------
core = jno.core(
    [
        gas,
        wall,
        Tg - Ts,  # the temperatures agree
        K_G * Tg.d(ng) + K_W * Ts.d(ns),  # what leaves the gas enters the wall
        traction + pg * ns,  # the gas pressure on the wall
    ]
)
traj_gas, traj_wall = core.solve(
    coupling=jno.solve.coupling(time={gas: jno.solve.imex("ars222", cfl=0.5), wall: None}),
)
# --8<-- [end:code]

if __name__ == "__main__":
    c = core.coupled
    names = {id(gas): "gas", id(wall): "wall"}
    print("takes:", {k: names[id(c.parts[i].problem)] for k, i in c.assign.items()}, "(conditions 0, 1, 2)")
    print(f"{len(c.window_iterations)} windows, sweeps per window {c.window_iterations}")
    g = np.asarray(traj_gas).reshape(len(c.times), 5, -1)
    V = np.asarray(gas.volumes)
    pts = np.asarray(dw.mesh_connectivity["points"])[:, :2]
    tri = np.asarray(dw.mesh_connectivity["triangles"])
    e1, e2 = pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    wl = np.asarray(traj_wall)
    sl_T, sl_w = wall.blocks[wall.block_index(Tw)], wall.blocks[wall.block_index(w)]
    TW = wl[:, sl_T.start : sl_T.stop]
    heat_gas, heat_wall = g[:, 3] @ V, RHOC_W * (TW[:, tri].mean(axis=2) @ area)
    total = heat_gas + heat_wall
    moved = heat_wall[-1] - heat_wall[0]
    print(f"heat into the wall {moved:.4f}; energy drift {np.abs(total - total[0]).max() / moved:.2%} of it")
    p_mean = (GAMMA - 1.0) * (g[:, 3] - 0.5 * (g[:, 1] ** 2 + g[:, 2] ** 2) / g[:, 0]) @ V / V.sum()
    print(f"gas pressure {p_mean[0]:.3f} -> {p_mean[-1]:.3f}; wall temperature up to {TW[-1].max():.3f}")
    W = wl[-1, sl_w.start : sl_w.stop].reshape(-1, 2)
    r = np.hypot(pts[:, 0], pts[:, 1])
    print(f"bore radial displacement {((W[:, 0] * pts[:, 0] + W[:, 1] * pts[:, 1]) / r)[r < A + 1e-9].mean():.2e}")
