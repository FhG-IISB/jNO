"""``relocate()`` on a MOVING mesh: slide the vertices, keep the connectivity, rebuild nothing.

r-adaptivity is the one adaptivity kind that composes cheaply with a geometry-term march. Vertex
positions ride the CARRY, not the closure, so a relocation hands a new ``X`` to the SAME compiled
program: no new nodes, no new connectivity, no rebuild, no XLA compilation. h-adaptivity is the
opposite -- it changes array shapes, which is the 8-15 s the march pays per node-set change.

Oracles here:
  * relocation MOVES nodes and does not rebuild -- node count and connectivity are untouched;
  * the enclosed AREA is unchanged to machine precision, because a boundary vertex may only slide
    along the chord joining its neighbours (polygon area is linear in each vertex, so that kills the
    first-order change) and a normal offset then removes the quadratic cross terms;
  * it equidistributes -- the monitor-mass spread across cells falls against the same march without it;
  * ``.remesh(alpha=...)`` chains a reconnection in front of it, which is what repairs a SLIVER --
    a connectivity defect that moving nodes cannot fix, because the same three nodes still form the
    same bad triangle;
  * 3-D is refused by name rather than silently doing something else.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _disk(adapt, *, vel=0.0, n=13, T=0.32, size=0.08):
    """A disk carrying a sharp front. ``vel=0`` freezes the domain, so any vertex motion is relocation."""
    d = jno.shape.disk(0.0, 0.0, 0.5, size=size).domain(time=(0.0, T, n))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.05 * (ui.x * vi.x + ui.y * vi.y),
            u(ci[0], ci[1]) - jno.np.tanh(10.0 * ci[0]),
            xi.d(ti) - vel,
        ]
    )
    return fem, fem.solve(adapt=adapt)


def _area(P, C):
    a, b, c = P[C[:, 0]], P[C[:, 1]], P[C[:, 2]]
    return float(0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).sum())


def _defect(P, C, u):
    """Spread of monitor mass per cell; 0 is perfect equidistribution."""
    a, b, c = P[C[:, 0]], P[C[:, 1]], P[C[:, 2]]
    A = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    g = np.zeros(len(C))
    for k, (i, j, r) in enumerate(C):
        J = np.array([P[j] - P[i], P[r] - P[i]]).T
        if abs(np.linalg.det(J)) < 1e-300:
            continue
        g[k] = float(np.hypot(*np.linalg.solve(J.T, np.array([u[j] - u[i], u[r] - u[i]]))))
    m = np.sqrt(1.0 + (g / max(g.max(), 1e-300) * 8.0) ** 2) * A
    return float(m.std() / m.mean())


def test_relocation_moves_the_nodes_without_rebuilding():
    """The whole point: a new ``X`` for the same compiled program."""
    fem, tr = _disk(jno.solve.relocate())
    P0, C0 = (np.asarray(x) for x in tr.meshes[0])
    Pn, Cn = (np.asarray(x) for x in tr.meshes[-1])
    hist = getattr(fem, "adapt_history", []) or []

    assert sum(1 for h in hist if h.get("relocated")) > 0, "no relocation ran"
    assert sum(1 for h in hist if h.get("rebuilt")) == 0, "a relocation must not rebuild"
    assert len(P0) == len(Pn), f"node count changed {len(P0)} -> {len(Pn)}"
    assert np.array_equal(C0, Cn), "connectivity changed: relocation must keep it"
    assert np.abs(Pn - P0).max() > 1e-6, "nodes did not move at all"


def test_relocation_conserves_the_enclosed_area():
    """Boundary vertices slide along the neighbour chord, then a normal offset removes the residual."""
    _, tr = _disk(jno.solve.relocate())
    P0, C0 = (np.asarray(x) for x in tr.meshes[0])
    Pn, Cn = (np.asarray(x) for x in tr.meshes[-1])
    a0, an = _area(P0, C0), _area(Pn, Cn)
    rel = abs(an - a0) / a0
    # chord projection alone leaves the quadratic cross terms (measured 4.7e-3 on a 0.05 move); the
    # iterated normal correction takes it to ~1e-11, so anything above 1e-8 means that regressed.
    assert rel < 1e-8, f"area moved by {rel:.3e} (chord projection + normal correction should hold it)"


def test_relocation_equidistributes_better_than_not_relocating():
    """The point of moving them: monitor mass spread across cells goes DOWN at the same node count."""
    _, ref = _disk(jno.solve.remesh(alpha=1.2, every=5))
    _, rel = _disk(jno.solve.relocate())
    Pr, Cr = (np.asarray(x) for x in ref.meshes[-1])
    Pl, Cl = (np.asarray(x) for x in rel.meshes[-1])
    d_ref = _defect(Pr, Cr, np.asarray(ref.states[-1])[: len(Pr)])
    d_rel = _defect(Pl, Cl, np.asarray(rel.states[-1])[: len(Pl)])
    assert len(Pl) == len(Pr), f"different node counts ({len(Pl)} vs {len(Pr)}) would make this meaningless"
    assert d_rel < d_ref, f"relocation did not equidistribute: {d_rel:.4f} vs {d_ref:.4f}"


def test_a_chained_remesh_reconnects_in_front_of_the_relocation():
    """A sliver is a CONNECTIVITY defect; only retriangulation repairs it, so the chain runs first."""
    fem, tr = _disk(jno.solve.relocate().remesh(alpha=1.2, every=1))
    P0, _ = (np.asarray(x) for x in tr.meshes[0])
    Pn, _ = (np.asarray(x) for x in tr.meshes[-1])
    hist = getattr(fem, "adapt_history", []) or []
    assert sum(1 for h in hist if h.get("relocated")) > 0, "the chained form must still relocate"
    assert len(P0) == len(Pn), "the reconnection runs at a FIXED node set (manage=False)"


def test_relocate_on_a_moving_mesh_is_2d_only():
    d = jno.shape.box(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, size=0.5).domain(time=(0.0, 0.1, 3))
    u, v = d.fem_symbols()
    xi, yi, zi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi, t=ti), v.bind(x=xi, y=yi, z=zi, t=ti)
    fem = jno.fem([ui.t * vi + 0.05 * (ui.x * vi.x), u(ci[0], ci[1], ci[2]) - 1.0, xi.d(ti) - 0.0])
    with pytest.raises(NotImplementedError, match="2-D only"):
        fem.solve(adapt=jno.solve.relocate())


def test_a_relocation_may_not_invert_a_cell():
    """Relocation keeps the CELLS, so an orientation change is a tangle -- and must be rejected.

    The march's tangle test compares each cell's signed measure against a reference sign carried in
    the topology. ``_make_topo`` re-reads that reference from whatever mesh it is handed, which is
    right after a RECONNECTION (a flip legitimately reverses a cell) and wrong after a RELOCATION,
    where it would record the inversion as the new reference and disarm the test permanently.

    Measured cost of not checking: a 600 W melt-ball run lost 55 % of its domain area across ~1 ms
    and ~4000 frames with nothing raised, and its temperatures stayed in a plausible band the whole
    way -- so a physical-bounds check would not have caught it either. Only area and orientation do.
    """
    fem, tr = _disk(jno.solve.relocate(method="monge_ampere", every=2, relax=10), vel=0.0, n=17, T=0.4)
    a0 = None
    for k in range(len(tr)):
        P, C = (np.asarray(x) for x in tr.meshes[k])
        a, b, c = P[C[:, 0]], P[C[:, 1]], P[C[:, 2]]
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        signs = np.sign(cross)
        assert np.all(signs == signs[0]), f"frame {k}: {int((signs != signs[0]).sum())} cells inverted"
        area = float(np.abs(cross).sum() * 0.5)
        a0 = area if a0 is None else a0
        assert abs(area / a0 - 1.0) < 1e-6, f"frame {k}: area drifted to {area / a0:.6f} of the start"


def test_the_monge_ampere_operator_is_nonsingular_on_disconnected_bodies():
    """``K``'s null space is one constant PER CONNECTED COMPONENT, not one overall.

    The P1 stiffness satisfies ``K·1_c = 0`` separately on each component, so a mesh of ``m`` disjoint
    bodies has an ``m``-dimensional constant null space. A single rank-one ``(1/n)·11ᵀ`` shift leaves
    ``m-1`` of it, and a singular operator means the Monge-Ampere potential is undetermined by a
    constant per body -- an arbitrary RELATIVE DISPLACEMENT between them. Two disjoint disks measured
    a surviving singular value of 8.1e-17 before the per-component shift.

    This matters because a weld bridges two bodies: the mesh is disconnected until they merge.
    """
    import numpy as _np

    from jno.utils.solver.fem_adapt import _p1_operators

    def _disk(cx, r=1.0, n=12):
        th = _np.linspace(0.0, 2 * _np.pi, n, endpoint=False)
        pts = _np.vstack([[cx, 0.0], _np.column_stack([cx + r * _np.cos(th), r * _np.sin(th)])])
        cells = _np.array([[0, i + 1, (i + 1) % n + 1] for i in range(n)], dtype=_np.int64)
        return pts, cells

    p1, c1 = _disk(0.0)
    p2, c2 = _disk(3.0)
    p3, c3 = _disk(6.0)
    cases = {
        1: (p1, c1),
        2: (_np.vstack([p1, p2]), _np.vstack([c1, c2 + len(p1)])),
        3: (_np.vstack([p1, p2, p3]), _np.vstack([c1, c2 + len(p1), c3 + 2 * len(p1)])),
    }
    for m, (pts, cells) in cases.items():
        k = _np.asarray(_p1_operators(pts, cells, 2)[3])
        sv = _np.linalg.svd(k, compute_uv=False)
        n_null = int((sv < sv.max() * 1e-12).sum())
        assert n_null == 0, f"{m} bodies: operator still has a {n_null}-dimensional null space"
