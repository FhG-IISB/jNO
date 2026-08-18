# --8<-- [start:code]
"""Topology optimisation on a deformable mesh: density AND nodal positions as design variables.

The classic SIMP cantilever, but the mesh is a design variable too. The optimiser drives one
vector [rho, d_x, d_y] -- an element density per triangle, plus the position of every interior
node -- so the boundary is resolved by *moving nodes onto it* rather than by refining until the
staircase is small. This is the method of

  K. Jung and D.-N. Kim, "Density-based topology optimization using a deformable mesh",
  Computers & Structures (2025), doi:10.1016/j.compstruc.2025.107879.
  Perimeter control follows R.B. Haber, C.S. Jog and M.P. Bendsoe, "A new approach to
  variable-topology shape design using a constraint on perimeter", Struct. Optim. 11 (1996) 1-12.

Three pieces make it work, and each is one line here:

  * `space="P0"`         one design value per ELEMENT (their eq. 12), not per node
  * `d.patch_filter()`   their eq. (17)-(19): a physical density that drives one-node
                         connections and lone dense elements to zero. It replaces the usual
                         density filter, and being NON-LOCAL it enters as a reparameterisation
                         (`rho.constrain`), not as a term in the weak form
  * `jno.le(...)`        geometric constraints (min angle, max/min element volume) that keep the
                         mesh valid while its nodes move -- their eq. (24)/(26)/(28)

**Every sensitivity is automatic.** The paper hand-derives dC/drho and dC/dX -- the latter needs
the derivative of each element stiffness with respect to its three nodes' coordinates -- across
two pages. Here `fem.solve()` is a differentiable trace node and trainable coordinates are
ordinary parameters, so MMA gets its gradients from AD and none of that is written down.

The run ends with a **reanalysis**, which is not optional: an optimiser that moves nodes can
lower compliance either by improving the structure or by distorting elements until they
under-integrate strain energy, and it cannot tell those apart from the inside. Transferring the
converged density to a clean, undistorted mesh and re-solving is the check that separates them.
"""

import jax

jax.config.update("jax_enable_x64", True)  # geometric constraints at p=50 need the exponent range

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import jno  # noqa: E402

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

L, H, h = 60.0, 30.0, 1.0  # domain and target edge length (the paper's constants below assume h=1)
E0, EMIN, NU, PENAL, VOLFRAC = 1.0, 1e-9, 0.3, 3.0, 0.4
LAM, MU = E0 * NU / (1 - NU**2), E0 / (2 * (1 + NU))  # plane stress
THETA_MIN, PNORM = np.deg2rad(20.0), 50.0  # min interior angle; p-norm aggregation exponent
V0MAX, V0MIN = 0.75 * h**2, 0.05 * h**2  # element-volume bounds, scaled from edge length 1
MOVE_BOUND, SPAN, TOL, ITERS = 2.0 * h, 2.0 * h, 1e-6 * L, 400
PSTAR, BETA, GAMMA, BETA_MIN = 650.0, 2e-4, 0.997, 1e-4  # perimeter target and barrier schedule

d = jno.Shape.rect(0, 0, L, H, size=h).domain()
pts0 = np.asarray(d.mesh.points)[:, :2]
cells = np.asarray(d._cells_p1())

# --- design variables 2 and 3: the interior nodal coordinates --------------------------------
# Nodes on the outer boundary stay put (they define the domain); everything inside may move,
# bounded by +/-MOVE_BOUND from where it started. `.trainable()` must precede `jno.fem(...)`,
# which reads the coordinate registry when it builds.
interior = lambda x, y: (x > TOL) & (x < L - TOL) & (y > TOL) & (y < H - TOL)  # noqa: E731
xm, ym, _ = d.variable("mv", where=interior, split=True)
xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y")
ids = np.asarray(d._trainable_coords[0]["ids"], dtype=int)
for spec in d._trainable_coords:
    x0 = pts0[ids, int(spec["axis"])]
    spec["expr"].optimizer(
        jno.optimizers.mma(move=0.02, lower=x0 - MOVE_BOUND, upper=x0 + MOVE_BOUND, move_gamma=0.9985, move_min=0.001)
    )

# --- design variable 1: one density per element ----------------------------------------------
u, phi = d.fem_symbols(value_shape=(2,))  # P1 vector displacement
_r, s = d.fem_symbols(space="P0", names=("r", "s"))  # P0 scalar -> one dof per triangle
xi, yi, _ = d.variable("interior", split=True)
xl, yl, _ = d.variable("left", split=True)  # clamped root
xt, yt, _ = d.variable("tip", where=lambda x, y: (x > L - TOL) & (y < SPAN + TOL), split=True)

rho = jno.np.parameter(s, name="rho")
rho.dtype(jnp.float64)
rho.initialize(jax.nn.initializers.constant(VOLFRAC))
rho.optimizer(jno.optimizers.mma(move=0.15, lower=1e-3, upper=1.0, move_gamma=0.9985, move_min=0.0075))

# `penal` is a runtime parameter so the continuation can raise it without triggering a recompile.
penal_p = jno.np.parameter((1,), name="penal")
penal_p.dtype(jnp.float64)
penal_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, PENAL))
penal_p.optimizer(optax.sgd(1.0))

# Solid, non-designable material where the load is applied. Without it the optimiser empties the
# loaded elements and the compliance then measures that artefact rather than the structure.
cen0 = pts0[cells].mean(axis=1)
pmask = jnp.asarray((cen0[:, 0] > L - SPAN) & (cen0[:, 1] < SPAN))
patch = d.patch_filter()  # eq. (17)-(19)
rho.constrain(lambda r: jnp.where(pmask, 1.0, patch(r)))  # physics sees the PHYSICAL density

# --- the elasticity problem -------------------------------------------------------------------
# Bind the notation once so the weak form and the objective below read like the equations, and so
# they demonstrably share one bilinear form rather than being two hand-typed copies of it.
eps = lambda w: symgrad(w, [xi, yi])  # noqa: E731
a = lambda p, q: LAM * trace(p) * trace(q) + 2 * MU * inner(p, q, n_contract=2)  # noqa: E731  sigma(p):q
E = lambda r: EMIN + r**penal_p * (E0 - EMIN)  # noqa: E731  -- SIMP: E(rho) = Emin + rho^p (E0 - Emin)

fem = jno.fem(
    [
        E(rho) * a(eps(u), eps(phi)),
        u(xl, yl) - (0.0, 0.0),  # clamped root
        -1.0 * inner(jnp.array([0.0, -1.0 / SPAN]), phi.bind(x=xt, y=yt), n_contract=1),
    ],
    quad_degree=2,
)

# --- the objective and the constraints, as the integrals they are ------------------------------
# Compliance is the strain energy C = a(u,u) = ∫ sigma(u):eps(u) dOmega -- the SAME form the weak
# statement above is built from, integrated at the solution. `.integrate(fem)` inherits the
# quadrature the operator was assembled with, so this equals f.u exactly rather than to within a
# quadrature error; naming the `fem` supplies what the expression cannot (which solution, which
# system to differentiate through), and every functional over one `fem` shares a single solve.
cellv, angles = d.cell_volume(), d.cell_angles()  # differentiable in the nodal coordinates
# A P0 parameter evaluates to (n_cells, 1) while `cell_volume()` is (n_cells,), so pairing them
# without this flatten broadcasts to an (n_cells, n_cells) OUTER PRODUCT whose sum is n_cells times
# too large -- silently, since it is a valid shape. Flatten once, here.
rho_e = rho.reshape(-1)
compliance = (E(rho) * a(eps(u), eps(u))).integrate(fem).name("C")
# The volume needs no quadrature: rho is piecewise constant, so ∫rho dOmega is exactly sum(rho_k |K|)
# and `cell_volume()` is already a node differentiable in the moving mesh. Routing it through the FEM
# functional would run a full element map to compute a weighted sum -- same answer, far more work.
volume = ((rho_e * cellv).sum / (VOLFRAC * cellv.sum)).name("V")
g_ang = (((2 * jnp.pi - angles) / (2 * jnp.pi - THETA_MIN)).reshape(-1).name("g1")).pnorm(
    PNORM, normalize=True
)  # eq. (24): no interior angle below theta_min
g_vmx = ((cellv / ((2.0 - rho_e) * V0MAX)).name("g2")).pnorm(PNORM, normalize=True)  # eq. (26): no element grows past V0max
g_vmn = (((2 * V0MAX - cellv) / (2 * V0MAX - V0MIN)).name("g3")).pnorm(
    PNORM, normalize=True
)  # eq. (28): none collapses below V0min

# --- perimeter control: the feature-scale lever, eq. (38)-(41) ---------------------------------
# P sums the density jump across every interior edge (Haber, Jog & Bendsoe 1996), smoothed by
# zeta. Holding it under a target P* forbids the optimiser from buying stiffness with ever-finer
# members, so it is the manufacturability knob: lower P* -> fewer, thicker bars.
#
# It enters as an INTERIOR PENALTY, R = -beta * log(P* - P), not as a `jno.le` constraint --
# that is the paper's eq. (39)-(40). `log_barrier` extends the log quadratically once P gets
# within tau of the bound: a plain log(max(P*-P, eps)) goes CONSTANT above the bound, so its
# gradient is exactly zero there and the barrier silently stops doing anything.
# `beta` decays geometrically (eq. 41) so the barrier stops distorting the converged optimum.
perim = rho.perimeter(zeta=0.1)
beta_p = jno.np.parameter((1,), name="beta")
beta_p.dtype(jnp.float64)
beta_p.initialize(lambda k, sh, dtype=None: jnp.full(sh, BETA))
beta_p.optimizer(optax.sgd(1.0))

# `jno.le` marks a constraint the optimiser handles but does NOT add to the loss -- without it
# every constraint doubles as a soft penalty and fights MMA's own dual handling.
terms = [compliance]
# `watch=0` points the continuation at the COMPLIANCE: under a decaying barrier the total loss
# keeps drifting, so a convergence test on the total would never fire.
callbacks = [jno.optimizers.simp_continuation(penal_p, rho, physical=patch, every=25, watch=0)]
if PSTAR > 0:
    terms.append((beta_p[0] * perim.log_barrier(PSTAR)).name("R"))
    callbacks.append(jno.optimizers.geometric_decay(beta_p, GAMMA, start=BETA, minimum=BETA_MIN))
terms += [jno.le(volume, 1.0), jno.le(g_ang, 1.0), jno.le(g_vmx, 1.0), jno.le(g_vmn, 1.0)]

crux = jno.core(terms, domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
crux.solve(ITERS, callbacks=callbacks)

rho_f = np.asarray(crux.eval([rho])).reshape(-1)  # PHYSICAL density (constrain applies first)
pts_f = pts0.copy()
for spec in d._trainable_coords:
    pts_f[ids, int(spec["axis"])] = np.asarray(crux.eval([spec["expr"]])).reshape(-1)
C = float(np.asarray(crux.eval([compliance])).reshape(-1)[0])
P = float(np.asarray(crux.eval([perim])).reshape(-1)[0])  # the smoothed perimeter, eq. (38)

# --- the honesty check: re-solve the SAME design on a clean, undistorted mesh -------------------
d_ref = jno.Shape.rect(0, 0, L, H, size=h / 2).domain()  # fresh, twice as fine, undeformed
rho_ref = d.transfer_cell_field(rho_f, d_ref, points=pts_f, outside=1e-3)  # `points=` -> deformed source
u2, phi2 = d_ref.fem_symbols(value_shape=(2,))
_r2, s2 = d_ref.fem_symbols(space="P0", names=("r2", "s2"))
xi2, yi2, _ = d_ref.variable("interior", split=True)
xl2, yl2, _ = d_ref.variable("left", split=True)
xt2, yt2, _ = d_ref.variable("tip", where=lambda x, y: (x > L - TOL) & (y < SPAN + TOL), split=True)
rho2 = jno.np.parameter(s2, name="rho2")
rho2.dtype(jnp.float64)
eu2, ep2 = symgrad(u2, [xi2, yi2]), symgrad(phi2, [xi2, yi2])
fem_ref = jno.fem(
    [
        (EMIN + rho2**PENAL * (E0 - EMIN)) * (LAM * trace(eu2) * trace(ep2) + 2 * MU * inner(eu2, ep2, n_contract=2)),
        u2(xl2, yl2) - (0.0, 0.0),
        -1.0 * inner(jnp.array([0.0, -1.0 / SPAN]), phi2.bind(x=xt2, y=yt2), n_contract=1),
    ],
    quad_degree=2,
)
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402


def solve_C(operator, args):  # compliance from an assembled system, by sparse-direct factorisation
    A, b = operator.evaluate(args)
    f = np.asarray(jnp.asarray(b).reshape(-1), dtype=np.float64)
    i = np.asarray(A.indices)
    K = sp.csr_matrix((np.asarray(A.data, dtype=np.float64), (i[:, 0], i[:, 1])), shape=(f.size, f.size))
    return float(f @ spla.spsolve(K.tocsc(), f))


C_ref = solve_C(fem_ref.operator, {"rho2": jnp.asarray(np.asarray(rho_ref).reshape(-1))})

# A raw reanalysis gap is NOT a distortion measurement: the reference mesh is finer, and a coarse
# mesh is over-stiff on its own. Calibrate that away with a control carrying no design and no
# distortion at all -- a uniform density on the undeformed coarse mesh and on the clean fine one.
# Whatever gap THAT shows is pure discretisation, and only the excess above it is attributable to
# the moved nodes. (A first-order correction: compliance error does not factorise exactly.)
n_c, n_f = cells.shape[0], d_ref._cells_p1().shape[0]
coord0 = {sp["name"]: jnp.asarray(pts0[ids, int(sp["axis"])]) for sp in d._trainable_coords}  # UNdeformed
C_u_coarse = solve_C(fem.operator, {"rho": jnp.full(n_c, VOLFRAC), "penal": jnp.asarray([PENAL]), **coord0})
C_u_fine = solve_C(fem_ref.operator, {"rho2": jnp.full(n_f, VOLFRAC)})
discretisation = C_u_fine / C_u_coarse  # the coarse mesh's intrinsic over-stiffness
excess = (C_ref / C) / discretisation - 1.0  # what the node movement bought that was not real

vol = (
    np.abs(np.linalg.det(np.stack([pts_f[cells][:, 1] - pts_f[cells][:, 0], pts_f[cells][:, 2] - pts_f[cells][:, 0]], -1)))
    / 2
)
print(
    f"\nTopology optimisation: {n_c} elements, {fem.dofs} dofs, {ITERS} iterations"
    f"\n  compliance (own deformed mesh)          = {C:9.4f}"
    f"\n  compliance (clean mesh, {n_f} elements) = {C_ref:9.4f}   raw gap {100 * (C_ref - C) / C:+.1f} %"
    f"\n  control, uniform density, no distortion = {C_u_coarse:9.4f} -> {C_u_fine:9.4f}"
    f"   ({100 * (discretisation - 1):+.1f} % is pure discretisation)"
    f"\n  => stiffness over-report attributable to the moved nodes: {100 * excess:+.1f} %"
    f"\n  perimeter P = {P:.3f}"
    + (f" against target P* = {PSTAR:.1f}" if PSTAR > 0 else " (uncontrolled)")
    + f"\n  volume fraction = {np.sum(rho_f * vol) / np.sum(vol):.4f}   "
    f"M_nd = {4 * np.mean(rho_f * (1 - rho_f)):.4f}   inverted elements = {int((vol <= 0).sum())}"
)
assert PSTAR <= 0 or P < PSTAR  # the barrier kept the perimeter under its bound
assert np.sum(rho_f * vol) / np.sum(vol) < VOLFRAC * 1.05  # volume constraint respected
assert int((vol <= 0).sum()) == 0  # the geometric constraints kept every element valid
# --8<-- [end:code]

# ---- solution figure: the design on its own mesh | the mesh it earned | the reanalysis gap ----
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

plt.rcParams.update(
    {"savefig.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10, "figure.dpi": 120}
)

fig = plt.figure(figsize=(11, 7))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.1], hspace=0.45, wspace=0.25)

ax = fig.add_subplot(gs[0, :])
ax.tripcolor(Triangulation(pts_f[:, 0], pts_f[:, 1], cells), facecolors=rho_f, cmap="gray_r", vmin=0, vmax=1)
ax.set_aspect("equal"), ax.set_axis_off()
ax.set_title(rf"physical density $\bar\rho$ on the deformed mesh — C = {C:.2f}, P = {P:.1f} (target {PSTAR:g})")

ax = fig.add_subplot(gs[1, :])
ax.triplot(Triangulation(pts_f[:, 0], pts_f[:, 1], cells), color="0.8", lw=0.25)
moved = np.linalg.norm(pts_f - pts0, axis=1)  # how far each node actually travelled
sc = ax.scatter(pts_f[ids, 0], pts_f[ids, 1], c=moved[ids], s=1.5, cmap="magma_r", vmin=0, vmax=MOVE_BOUND)
fig.colorbar(sc, ax=ax, shrink=0.8, label=r"$\|x-x_0\|$")
ax.set_aspect("equal"), ax.set_axis_off()
ax.set_title(rf"the optimised mesh, interior nodes coloured by distance moved (bound $\pm{MOVE_BOUND:g}$)")

ax = fig.add_subplot(gs[2, 0])
# Plot the reanalysis against the discretisation control, so the bar the eye compares is the one
# that means something. C * discretisation is what a NON-distorting design would have read.
bars = [C, C * discretisation, C_ref]
ax.bar(
    ["own\nmesh", "clean mesh,\nexpected", "clean mesh,\nactual"], bars, color=["#3b6fb6", "#b6b6b6", "#c1543a"], width=0.55
)
for i, v in enumerate(bars):
    ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_ylabel("compliance"), ax.margins(y=0.25)
ax.tick_params(labelsize=8)
ax.set_title(f"reanalysis: {100 * excess:+.1f} % beyond discretisation")

ax = fig.add_subplot(gs[2, 1])
ax.hist(rho_f, bins=40, color="#3b6fb6")
ax.set_xlabel(r"$\bar\rho$"), ax.set_ylabel("elements"), ax.set_yscale("log")
ax.set_title(rf"density histogram — $M_{{nd}}$ = {4 * np.mean(rho_f * (1 - rho_f)):.3f}")

fig.savefig(Path(__file__).parents[2] / "assets" / "topology_optimisation_cantilever.png")
