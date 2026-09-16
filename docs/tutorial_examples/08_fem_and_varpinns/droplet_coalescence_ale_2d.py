# --8<-- [start:code]
"""Two droplets, meshed as BODIES IN A VOID, touch and become one -- the sharp-interface route.

The liquid is the mesh. Its surface is a real boundary carrying the capillary traction, and it moves
with the fluid:

    rho (du/dt|_X + ((u - w).grad)u) = -grad p + div(2 eta D(u)),   div u = 0        (ALE, in the liquid)
    T.n = -sigma H n   on the whole surface                                          (a traction, no BC)
    dX/dt = u          on the whole surface                                          (it is Lagrangian)

with w the MESH velocity and D(u) the symmetric gradient. Two things make the topology change possible.
The surface rides with the fluid, written as geometry terms `xs.d(ts) - u`, so `jno.fem` marches the mesh
and the flow together. And `remesh(alpha=...)` re-decides which nodes form elements each step -- the alpha
shape of the Particle Finite Element Method -- so when the gap between two bodies closes below ~2 alpha h,
the bridging triangles survive the filter and two meshes become one. Every node stays where it is, so the
P1 state carries across by identity.

Three details are not optional, and each is measured in the tests:

* the viscous term must be `2 eta D(u):D(v)`; `eta grad u : grad v` is a pseudo-traction that is correct
  only behind Dirichlet walls, and at a free surface it silently destroys the motion;
* the SUPG/PSPG `tau` must be scaled for this regime. A capillary drop is nearly inviscid and nearly
  stagnant -- the opposite of the advection-dominated flow that `tau` is built for -- and the unscaled
  recipe contributes about ten times the physical damping;
* reconnection must run EVERY step (`every=1`). The neck opens fast, and at `every=2` the mesh tangles.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jno  # noqa: E402

inner, symgrad = jno.np.inner, jno.np.symgrad
ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731  A:B

RHO, ETA, SIGMA = 1.0, 0.01, 10.0  # density, viscosity, surface tension
NU = ETA / RHO
R, GAP, H = 0.20, 0.02, 0.04  # two drops of radius R, rims GAP apart, element size H
DT, N_STEPS = 1e-4, 300  # the capillary limit is tighter than sqrt(rho h^3 / 2 pi sigma) suggests
C_I, TAU_SCALE = 36.0, 1e-4  # ... and tau must be scaled DOWN for a capillary flow (see above)
T_CAP = np.sqrt(RHO * R**3 / SIGMA)  # capillary time: 0.0283

cx = R + GAP / 2.0
d = (jno.shape.disk(-cx, 0.0, R, size=H) | jno.shape.disk(cx, 0.0, R, size=H)).domain(time=(0.0, N_STEPS * DT, N_STEPS + 1))
n_node = int(np.asarray(d.mesh.points).shape[0])
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)  # P1/P1, stabilised
p, q = d.fem_symbols(names=("p", "q"), order=1)
xi, yi, ti = d.variable("interior", split=True)
xs, ys, ts, nx, ny = d.variable("boundary", normals=True, split=True)  # the free surface, with normals
x0, y0, _t0 = d.variable("initial", split=True)

B = dict(x=xi, y=yi, t=ti)
ub, vv, pp, qq = u.bind(**B), v.bind(**B), p.bind(**B), q.bind(**B)
vs = v.bind(x=xs, y=ys)
D = lambda w: symgrad(w, [xi, yi])  # noqa: E731
ndv = lambda f, i: nx * f.x[i] + ny * f.y[i]  # noqa: E731  (grad f_i).n
div_G = lambda f: f.x[0] + f.y[1] - (nx * ndv(f, 0) + ny * ndv(f, 1))  # noqa: E731  surface divergence

# The MESH velocity w is `coord.d(t)` read inside the weak form, so u - w is the ALE convective velocity.
# Componentwise: a vector cannot be assembled from two scalar expressions.
c0, c1 = ub[0] - xi.d(ti), ub[1] - yi.d(ti)
conv = lambda i: c0 * ub.x[i] + c1 * ub.y[i]  # noqa: E731
G = d.cell_metric
gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
tau = jno.lag(TAU_SCALE * ((2.0 / DT) ** 2 + gG(ub) + C_I * NU**2 * inner(G, G, n_contract=2)) ** -0.5)
r0, r1 = ub.t[0] + conv(0) + pp.x / RHO, ub.t[1] + conv(1) + pp.y / RHO  # lap(u) vanishes on P1

momentum = (
    RHO * (ub.t[0] * vv[0] + ub.t[1] * vv[1])
    + RHO * (conv(0) * vv[0] + conv(1) * vv[1])
    + 2.0 * ETA * ddot(D(ub), D(vv))  # the TRUE stress: grad:grad would be a pseudo-traction
    - pp * (vv.x[0] + vv.y[1])
    + tau * ((c0 * vv.x[0] + c1 * vv.y[0]) * r0 + (c0 * vv.x[1] + c1 * vv.y[1]) * r1)  # SUPG
)
continuity = -qq * (ub.x[0] + ub.y[1]) - tau * (qq.x * r0 + qq.y * r1)  # PSPG: no pressure pin needed
capillary = SIGMA * div_G(vs)  # T n = -sigma H n, integrated by parts onto the test function
uf = u.bind(x=xs, y=ys).freeze(np.zeros((n_node, 2)))  # the solved velocity, delivered each step

fem = jno.fem(
    [
        momentum,
        continuity,
        capillary,  # the traction also fixes the pressure LEVEL: there is no Dirichlet anywhere
        u(x0, y0)[0] - 0.0,  # both drops start at rest
        u(x0, y0)[1] - 0.0,
        xs.d(ts) - uf[0],  # the surface is Lagrangian: dX/dt = u
        ys.d(ts) - uf[1],
    ]
)
traj = fem.solve(
    nonlinear=jno.solve.newton(direct=True),  # a saddle step wants a sparse-direct Newton
    adapt=jno.solve.remesh(alpha=1.2, every=1),  # re-triangulate the moved nodes EVERY step
)


def bodies(k):
    """How many separate liquid bodies frame ``k`` has (connected components of the triangulation)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pts, cells = np.asarray(traj.meshes[k][0]), np.asarray(traj.meshes[k][1])
    r = np.concatenate([cells[:, 0], cells[:, 1], cells[:, 2]])
    c = np.concatenate([cells[:, 1], cells[:, 2], cells[:, 0]])
    used = np.zeros(len(pts), bool)
    used[cells.reshape(-1)] = True
    _n, lab = connected_components(coo_matrix((np.ones(r.size), (r, c)), shape=(len(pts),) * 2), directed=False)
    return int(np.unique(lab[used]).size)


def boundary_loops(k):
    """How many closed boundary loops frame ``k`` has. A simply-connected drop has exactly one per body;
    an interior hole adds another -- which is what the alpha filter leaves behind if it deletes stretched
    interior triangles, and what a connected-component count alone will NOT notice."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from jno.utils.solver.fem_adapt import _boundary_edges_from_triangles

    pts = np.asarray(traj.meshes[k][0])
    e = np.asarray(_boundary_edges_from_triangles(np.asarray(traj.meshes[k][1])))
    used = np.zeros(len(pts), bool)
    used[e.reshape(-1)] = True
    _n, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(pts),) * 2), directed=False)
    return int(np.unique(lab[used]).size)


def perimeter(k):
    """Total length of the free surface at frame ``k``."""
    from jno.utils.solver.fem_adapt import _boundary_edges_from_triangles

    pts = np.asarray(traj.meshes[k][0])
    e = np.asarray(_boundary_edges_from_triangles(np.asarray(traj.meshes[k][1])))
    return float(np.sum(np.linalg.norm(pts[e[:, 0]] - pts[e[:, 1]], axis=1)))


def shape_of(k):
    """``(area, x-extent, y-extent, neck width)`` of frame ``k``."""
    pts, cells = np.asarray(traj.meshes[k][0]), np.asarray(traj.meshes[k][1])
    t = pts[cells]
    a, b = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    area = float(np.abs(0.5 * np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])))
    mid = np.abs(pts[:, 0]) < 0.02
    return area, np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), (np.ptp(pts[mid, 1]) if mid.any() else 0.0)


last = len(traj.states) - 1
print(f"{n_node} nodes, {fem.dofs} dofs, {N_STEPS} steps = {N_STEPS * DT / T_CAP:.2f} capillary times")
for k in (0, last // 4, last // 2, last):
    ar, ex, ey, neck = shape_of(k)
    print(
        f"  t = {float(traj.times[k]) / T_CAP:4.2f} t_cap: {bodies(k)} body(ies), area {ar:.5f}, "
        f"width {ex:.4f}, height {ey:.4f}, neck {neck:.4f}"
    )
reconnects = sum(h["remeshed"] for h in fem.adapt_history)
area0, area1 = shape_of(0)[0], shape_of(last)[0]
print(f"{reconnects} reconnections; area {area0:.5f} -> {area1:.5f} ({100 * (area1 / area0 - 1):+.2f} %)")

assert bodies(0) == 2, "the drops did not start as two separate meshes"
assert bodies(last) == 1, "the drops never merged"
assert shape_of(last)[3] > 0.5 * shape_of(last)[2], "the neck never opened"
# Area is two statements, not one. MERGING ADDS LIQUID: the bridge fills the gap the drops left between
# them, which is what a mesh-length contact model does (+3.3 % here, most of it in the first few steps) --
# it is not a conservation error, and quoting it as one would flatter or damn the scheme by accident.
# What the scheme owes is conservation AFTER the topology settles, and that is the tight bound.
_a_merged = shape_of(last // 4)[0]
assert abs(area1 / area0 - 1.0) < 0.05, f"the merge added {100 * (area1 / area0 - 1):.1f} % of liquid"
assert abs(area1 / _a_merged - 1.0) < 0.01, f"area drifted {100 * (area1 / _a_merged - 1):.2f} % after the merge"
for _k in range(last + 1):  # the liquid must stay SOLID: no voids punched by the alpha filter
    assert boundary_loops(_k) == bodies(_k), f"frame {_k} has interior holes ({boundary_loops(_k)} loops)"
    assert np.isfinite(np.asarray(traj.states[_k])).all(), f"frame {_k} carries non-finite values"
# Surface tension SHRINKS the surface, so a sudden LENGTHENING means the alpha filter bit a wedge out of
# it -- and a notch that reaches the surface adds no boundary loop, so the loop count above cannot see it.
# The filter is held steady by hysteresis (an existing cell survives to a wider threshold), which removes
# most of that: measured here, 2 steps of 300 still move the perimeter by more than 5 % (worst +13 %),
# because a Delaunay edge flip creates triangles that are NEW and so face `alpha` alone. Hence a loose
# catastrophe bound, plus the check that matters for the figure: the last frame must be representative of
# its neighbours rather than a bitten outlier.
_per = np.array([perimeter(k) for k in range(last + 1)])
_jump = float(np.max(_per[1:] / _per[:-1] - 1.0))
_median_tail = float(np.median(_per[-20:]))
assert _per[-1] < _per[0], "the free surface grew: surface tension does not do that"
assert abs(_per[-1] / _median_tail - 1.0) < 0.03, f"the last frame is an outlier ({_per[-1]:.4f} vs {_median_tail:.4f})"
assert _jump < 0.20, f"the alpha filter bit deep into the free surface (perimeter jumped {100 * _jump:.1f} % in a step)"
print(
    f"perimeter {_per[0]:.4f} -> {_per[-1]:.4f}; worst one-step change {100 * _jump:+.2f} %; "
    f"last frame {100 * (_per[-1] / _median_tail - 1):+.2f} % from its neighbours' median"
)
# --8<-- [end:code]

# ---- figure (hidden from the docs): the liquid at three instants, each on its own mesh -> a PNG ----
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Frutiger 45 Light", "Frutiger", "FreeSans", "DejaVu Sans"],
        "font.weight": "light",
        "font.size": 11,
        "axes.titleweight": "light",
    }
)
off = [int(o) for o in fem.offsets]
frames = (0, last // 2, last)
speeds = [np.linalg.norm(np.asarray(traj.states[k])[off[0] : off[1]].reshape(-1, 2), axis=1) for k in frames]
v_max = float(max(s.max() for s in speeds))  # one scale for all three panels, so they are comparable
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
for ax, k, speed in zip(axes, frames, speeds):
    pts, cells = np.asarray(traj.meshes[k][0]), np.asarray(traj.meshes[k][1])
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], cells[:, :3])
    im = ax.tripcolor(tri, speed, cmap="magma", vmin=0.0, vmax=v_max, shading="gouraud")
    ax.triplot(tri, lw=0.2, color="#1A202C", alpha=0.35)  # the mesh this frame was computed on
    ax.set_title(f"$t = {float(traj.times[k]) / T_CAP:.2f}\\,t_\\sigma$")
    ax.set_axis_off()
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.32, 0.32)
    ax.set_box_aspect(0.64 / 1.10)
# A colorbar, because the top of `magma` is nearly white: without a scale the FASTEST region reads as a
# hole in the liquid rather than as a value.
fig.canvas.draw()
pos = axes[-1].get_position()
cax = fig.add_axes([pos.x1 + 0.012, pos.y0, 0.035 * pos.width, pos.height])
cbar = fig.colorbar(im, cax=cax, label="$|\\mathbf{u}|$")
cbar.outline.set_visible(False)
cbar.minorticks_off()
cbar.ax.tick_params(length=0)
fig.savefig(Path(__file__).parents[2] / "assets" / "droplet_coalescence_ale_2d.png")
