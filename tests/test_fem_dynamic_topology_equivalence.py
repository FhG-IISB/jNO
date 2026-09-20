"""A runtime connectivity bundle must give the SAME answer as rebuilding on that connectivity.

``assemble_fem_native(dynamic_topology=True)`` lets a reconnecting march hand a new triangulation to an
already-compiled operator instead of rebuilding it (measured: one XLA compilation per Delaunay flip, 31 s
each, for a flip that changes 2-6 cells out of 568 and no shape at all).

The hazard is not that it crashes -- it is that some array derived from the cells stays BAKED and is
silently read at its old value while every shape still matches. ``parent_cell`` (face -> element) and
``lface`` (which local edge of that element) both renumber under a flip, so a half-threaded operator
returns plausible, wrong numbers. This oracle is what makes that impossible to miss: assemble on mesh A,
hand it mesh B's connectivity, and require agreement with an operator assembled on B from scratch.

The flip is constructed exactly, not sampled: two triangles sharing an edge are replaced by the other
diagonal of their quadrilateral, so the node POSITIONS are untouched and every shape is preserved -- the
same situation an alpha reconnection produces.
"""

import jax
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def flip_one_interior_edge(cells):
    """Return ``(cells_flipped, (i, j))``: one interior edge swapped for the quad's other diagonal.

    Picks the first edge shared by exactly two triangles whose opposite vertices are distinct, and
    rewrites both triangles. Node positions are untouched, so the mesh stays valid for the same points
    and every array shape is preserved."""
    cells = np.asarray(cells)
    edges = {}
    for ti, t in enumerate(cells):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges.setdefault((min(a, b), max(a, b)), []).append(ti)
    for (a, b), ts in edges.items():
        if len(ts) != 2:
            continue
        i, j = ts
        oi = [v for v in cells[i] if v not in (a, b)]
        oj = [v for v in cells[j] if v not in (a, b)]
        if len(oi) != 1 or len(oj) != 1 or oi[0] == oj[0]:
            continue
        out = cells.copy()
        out[i] = [oi[0], oj[0], b]
        out[j] = [oj[0], oi[0], a]
        return out, (i, j)
    raise AssertionError("no flippable interior edge found")


def test_a_flipped_edge_is_only_a_relabelling():
    """Sanity check on the oracle's own instrument before it is trusted to judge the feature."""
    cells = np.array([[0, 1, 2], [1, 3, 2], [0, 2, 4]])
    out, (i, j) = flip_one_interior_edge(cells)
    assert out.shape == cells.shape
    assert set(np.unique(out)) <= set(np.unique(cells))
    assert not np.array_equal(out, cells)


def _steady_pair(P, CA, CB):
    """``(opA_with_bundle_for_CB, opB_built_on_CB, u)`` -- the operator-level arm."""
    import jno
    from jno.utils.solver.fem_adapt import _boundary_edges_from_triangles, _domain_from_arrays
    from jno.utils.solver.fem_facets import build_facet_connectivity

    n = jno.np
    inner, symgrad = n.inner, n.symgrad
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731

    def build(cells, dynamic):
        tmpl = jno.shape.disk(0.0, 0.0, 1.0, size=0.5).domain()
        d = _domain_from_arrays(
            tmpl, P, cells, np.asarray(_boundary_edges_from_triangles(cells), dtype=np.int64), copy=True
        )
        if dynamic:
            d._fem_want_dynamic_topology = True
        u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
        p_, q = d.fem_symbols(names=("p", "q"), order=1)
        iv = d.variable("interior", split=True)
        bv = d.variable("boundary", normals=True, split=True)
        xi, yi = iv[0], iv[1]
        xs, ys, nx, ny = bv[0], bv[1], bv[-2], bv[-1]
        B = dict(x=xi, y=yi)
        ub, vv, pp, qq = u.bind(**B), v.bind(**B), p_.bind(**B), q.bind(**B)
        vs = v.bind(x=xs, y=ys)
        D = lambda w: symgrad(w, [xi, yi])  # noqa: E731
        ndv = lambda f, i: nx * f.x[i] + ny * f.y[i]  # noqa: E731
        div_G = lambda f: f.x[0] + f.y[1] - (nx * ndv(f, 0) + ny * ndv(f, 1))  # noqa: E731
        conv = lambda i: ub[0] * ub.x[i] + ub[1] * ub.y[i]  # noqa: E731
        mom = (
            1.0 * (conv(0) * vv[0] + conv(1) * vv[1])
            + 2.0 * 0.01 * ddot(D(ub), D(vv))
            - pp * (vv.x[0] + vv.y[1])
        )
        return d, jno.fem([mom, -qq * (ub.x[0] + ub.y[1]), 10.0 * div_G(vs)])

    dA, femA = build(CA, True)
    _dB, femB = build(CB, False)
    cn = build_facet_connectivity(CB, "triangle")
    bundle = dA._fem_native_topology_bundle(CB, [CB, CB], cn.parent_cell, cn.face_nodes, cn.local_face)
    return femA._op, femB._op, bundle


def test_a_runtime_bundle_matches_a_rebuilt_operator():
    """Operator level: hand an operator a DIFFERENT triangulation and it must assemble that one.

    This is the fast arm -- it localises a stale array to the assembler. It is not sufficient on its own:
    it cannot see anything delivered on ``args`` that it does not itself populate, which is exactly how a
    baked load-path connectivity survived it (see the march test below)."""
    from jno.utils.solver.fem_native import TOPOLOGY_ARG

    rng = np.random.default_rng(0)
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    P = np.c_[np.cos(th), np.sin(th)]
    P = np.vstack([P, np.zeros((1, 2)), 0.5 * P[:8]])
    from scipy.spatial import Delaunay

    CA = np.asarray(Delaunay(P).simplices, dtype=np.int64)
    CB, _ = flip_one_interior_edge(CA)
    opA, opB, bundle = _steady_pair(P, CA, CB)
    u = jax.numpy.asarray(rng.standard_normal(opB.size) * 1e-2)
    rA = np.asarray(opA.residual(u, {TOPOLOGY_ARG: bundle}))
    rB = np.asarray(opB.residual(u, None))
    scale = max(float(np.abs(rB).max()), 1e-30)
    assert np.abs(rA - rB).max() / scale < 1e-12, "the bundle did not reproduce the rebuilt residual"
    JA, JB = opA.jacobian(u, {TOPOLOGY_ARG: bundle}), opB.jacobian(u, None)
    dA_ = np.asarray(JA.todense() if hasattr(JA, "todense") else JA)
    dB_ = np.asarray(JB.todense() if hasattr(JB, "todense") else JB)
    assert np.abs(dA_ - dB_).max() / max(float(np.abs(dB_).max()), 1e-30) < 1e-12, "Jacobian differs"


def _march(dynamic, nsteps):
    """A short two-drop coalescence: merges, then flips. Returns ``(times, meshes)``."""
    import jno

    n = jno.np
    inner, symgrad = n.inner, n.symgrad
    ddot = lambda a, b: inner(a, b, n_contract=2)  # noqa: E731
    RHO, ETA, SIGMA, NU = 1.0, 0.01, 10.0, 0.01
    RA, RB, GAP, H, DT, C_I = 0.20, 0.15, 0.02, 0.03, 1e-4, 36.0
    xa, xb = -(RA + GAP / 2), (RB + GAP / 2)
    d = (jno.shape.disk(xa, 0.0, RA, size=H) | jno.shape.disk(xb, 0.0, RB, size=H)).domain(
        time=(0.0, nsteps * DT, nsteps + 1)
    )
    if dynamic:
        d._fem_want_dynamic_topology = True
    nnode = len(np.asarray(d.mesh.points))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, ti = d.variable("interior", split=True)
    xs, ys, ts, nx, ny = d.variable("boundary", normals=True, split=True)
    x0, y0, _t0 = d.variable("initial", split=True)
    B = dict(x=xi, y=yi, t=ti)
    ub, vv, pp, qq = u.bind(**B), v.bind(**B), p.bind(**B), q.bind(**B)
    vs = v.bind(x=xs, y=ys)
    D = lambda w: symgrad(w, [xi, yi])  # noqa: E731
    ndv = lambda f, i: nx * f.x[i] + ny * f.y[i]  # noqa: E731
    div_G = lambda f: f.x[0] + f.y[1] - (nx * ndv(f, 0) + ny * ndv(f, 1))  # noqa: E731
    # `xi.d(ti)` is the ALE mesh velocity: it is delivered per step on `args["__loadpath__"]` and gathered
    # per cell through the connectivity. That channel is the one the operator-level test above cannot see.
    c0, c1 = ub[0] - xi.d(ti), ub[1] - yi.d(ti)
    conv = lambda i: c0 * ub.x[i] + c1 * ub.y[i]  # noqa: E731
    G = d.cell_metric
    gG = lambda a: inner(a, inner(G, a, n_contract=1), n_contract=1)  # noqa: E731
    tau = jno.lag(((2.0 / DT) ** 2 + gG(ub) + C_I * NU**2 * ddot(G, G)) ** -0.5)
    r0, r1 = ub.t[0] + conv(0) + pp.x / RHO, ub.t[1] + conv(1) + pp.y / RHO
    momentum = (
        RHO * (ub.t[0] * vv[0] + ub.t[1] * vv[1])
        + RHO * (conv(0) * vv[0] + conv(1) * vv[1])
        + 2.0 * ETA * ddot(D(ub), D(vv))
        - pp * (vv.x[0] + vv.y[1])
        + tau * ((c0 * vv.x[0] + c1 * vv.y[0]) * r0 + (c0 * vv.x[1] + c1 * vv.y[1]) * r1)
    )
    continuity = -qq * (ub.x[0] + ub.y[1]) - tau * (qq.x * r0 + qq.y * r1)
    uf = u.bind(x=xs, y=ys).freeze(np.zeros((nnode, 2)))
    fem = jno.fem(
        [
            momentum,
            continuity,
            SIGMA * div_G(vs),
            u(x0, y0)[0] - 0.0,
            u(x0, y0)[1] - 0.0,
            xs.d(ts) - uf[0],
            ys.d(ts) - uf[1],
        ]
    )
    traj = fem.solve(nonlinear=jno.solve.newton(direct=True), adapt=jno.solve.remesh(alpha=1.2, every=1))
    return [(np.asarray(m[0]), np.asarray(m[1])) for m in traj.meshes]


def test_a_reconnecting_march_is_unchanged_by_the_runtime_path():
    """End to end: the runtime-connectivity march must reproduce the rebuilding one EXACTLY.

    The operator-level test above passes while a per-cell array delivered on ``args`` is stale, because it
    never populates that channel. This one marches, so every channel is live -- and a reconnection is a
    THRESHOLD on the solved geometry, so a wrong value does not stay small: it changes which edges flip and
    the two runs separate. Requiring the flip STEPS to match is therefore a sharper check than any norm.
    """
    ref = _march(False, 14)
    got = _march(True, 14)
    assert len(ref) == len(got)
    key = lambda c: set(map(tuple, np.sort(c, axis=1).tolist()))  # noqa: E731
    worst = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(ref, got))
    scale = float(np.abs(ref[0][0]).max())
    assert worst / scale < 1e-12, f"trajectories differ by {worst:.3e} (domain scale {scale:.3f})"
    flips_ref = [k for k in range(1, len(ref)) if key(ref[k][1]) != key(ref[k - 1][1])]
    flips_got = [k for k in range(1, len(got)) if key(got[k][1]) != key(got[k - 1][1])]
    assert flips_ref == flips_got, f"reconnections happen at different steps: {flips_ref} vs {flips_got}"
    assert flips_ref, "the window saw no reconnection at all -- the test proves nothing"
