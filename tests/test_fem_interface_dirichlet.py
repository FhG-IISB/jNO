"""A Dirichlet condition on an internal INTERFACE, or on a VOLUME sub-region, was silently dropped.

``_boundary_node_ids`` took its node set from the assembly mesh's **boundary facets** and then filtered a
named region among those. Neither of these regions has boundary facets of its own:

* an internal interface (``"fluid|solid"``) lies inside the domain, so only the two endpoints that happen
  to touch the outer boundary survived the filter -- **4 DOFs of 21 nodes**;
* a volume sub-region fell through to ``domain.tag_node_mask``, which for a volume region is a proximity
  test against the region's *sampled* points rather than a containment test -- **32 of 33**.

Nothing raised in either case. The condition simply stopped being imposed, which is the same failure mode
as the tie-reduction defect in ``test_fem_tie_dirichlet_conflict.py``: a boundary condition that is
written, accepted, and then not applied.

Both now resolve from **mesh topology** -- ``_region_node_ids_from_cells``, built on the same
``_cell_region_mask`` (a cell is in a region iff its centroid is) that the assembler's own region masking
uses, so the Dirichlet node set cannot disagree with the cells the equations were integrated over. A P2
edge midpoint on an interface is included for free: it belongs to a cell on each side, and the interface
is the intersection of the two sides' node sets.

The physics oracle is at the bottom: channel flow over a two-region mesh, where the wetted wall is an
internal interface. Before the fix the no-slip condition there was absent and the "flow" was wrong by
O(1); after it, the discrete solution reproduces Poiseuille.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

L, H, X0, X1, T = 4.0, 1.0, 1.0, 3.0, 0.4
MU, UMAX = 1.0, 1.0


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _two_region(size=0.22):
    return (
        jno.Shape.rect(0.0, 0.0, L, H).name("fluid").sized(size) + jno.Shape.rect(X0, -T, X1, 0.0).name("solid").sized(size)
    ).domain()


def _pinned_nodes(d, terms, vec=2):
    jno.fem(terms)
    pairs = list(getattr(d, "_fem_native_dirichlet_pairs", None) or [])
    return np.unique(np.array([int(a) for a, _g in pairs], dtype=np.int64) // vec)


def _lap(u, phi, r):
    a, b = u.bind(x=r[0], y=r[1]), phi.bind(x=r[0], y=r[1])
    return a.x[0] * b.x[0] + a.y[0] * b.y[0] + a.x[1] * b.x[1] + a.y[1] * b.y[1]


def test_an_interface_dirichlet_reaches_every_interface_node():
    """The defect, as a count. The interface is internal, so the boundary-facet path saw almost none of it."""
    d = _two_region()
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    d.tag("noslip", lambda x, y: (y > H - 1e-9) | (abs(y) < 1e-9))
    ci, iv = d.variable("interior", split=True), d.variable("fluid|solid", split=True)
    xw, yw, _ = d.variable("noslip", split=True)
    base = _pinned_nodes(d, [_lap(v, psi, ci), v(xw, yw) - 0.0])
    both = _pinned_nodes(d, [_lap(v, psi, ci), v(xw, yw) - 0.0, v(iv[0], iv[1]) - 0.0])

    pts = np.asarray(d._fem_native_dof_points_all[0])
    iface = np.flatnonzero((np.abs(pts[:, 1]) < 1e-9) & (pts[:, 0] > X0 - 1e-9) & (pts[:, 0] < X1 + 1e-9))
    assert iface.size == 21, iface.size
    assert set(iface.tolist()) <= set(both.tolist()), "every interface node must be pinned"
    # and the interface is genuinely INTERNAL: almost none of it is reachable from the boundary facets
    assert len(set(iface.tolist()) & set(base.tolist())) == 2, "only the two outer-boundary endpoints"


def test_a_volume_region_dirichlet_reaches_every_node_of_that_region():
    """The second half: a pin on a volume sub-region used to resolve by proximity to sampled points."""
    d = _two_region()
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    ci, sv = d.variable("interior", split=True), d.variable("solid", split=True)
    got = _pinned_nodes(d, [_lap(v, psi, ci), v(sv[0], sv[1]) - 0.0])

    from jno.utils.solver.fem_utils import _cell_region_mask

    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    truth = np.unique(cells[np.asarray(_cell_region_mask(d, "solid")).reshape(-1) > 0])
    assert set(truth.tolist()) == set(got.tolist()), (len(truth), len(got))
    assert (np.asarray(d._fem_native_dof_points_all[0])[truth, 1] < 1e-9).all(), "all at or below y=0"


def test_a_volume_region_dirichlet_works_on_independently_meshed_bodies():
    """The same pin, on the OTHER kind of multi-region domain -- and it was refused outright.

    ``_two_region`` above composes with ``+``, which builds through the polygon domain and populates
    ``domain._source_regions``. The classifier gated the volumetric pin on exactly that dict. But
    ``Shape.regions(..., conforming=False)`` is built by the gmsh emitter, which never writes it, so
    the pin raised "did you forget the test function?" on precisely the domains the tie machinery
    exists for -- while ``_region_node_ids_from_cells`` was already able to resolve the region's nodes
    from cell topology. The resolution existed; the gate would not let it be reached.

    A cell set stores ``[volume_cells, facets]``, so a body has entries in the first and a boundary tag
    in the second. Gating on that keeps the guard that matters -- a whole-domain trial-only term really
    is a forgotten test function -- which the last assertion pins.
    """
    d = jno.Shape.regions(
        fluid=jno.Shape.rect(0.0, 0.0, L, H).sized(0.22),
        solid=jno.Shape.rect(X0, -T, X1, 0.0).sized(0.22),
        conforming=False,
    ).domain()
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    ci, sv = d.variable("interior", split=True), d.variable("solid", split=True)
    got = _pinned_nodes(d, [_lap(v, psi, ci), v(sv[0], sv[1]) - 0.0])

    from jno.utils.solver.fem_utils import _cell_region_mask

    cells = np.asarray(d._fem_native_assembly_cells_all[0])
    truth = np.unique(cells[np.asarray(_cell_region_mask(d, "solid")).reshape(-1) > 0])
    assert len(truth) > 10, "sanity: the body must own a node set worth pinning"
    assert set(truth.tolist()) == set(got.tolist()), (len(truth), len(got))

    # ...and the guard it must NOT weaken: no region name at all is still a forgotten test function
    ci2 = d.variable("interior", split=True)
    with pytest.raises(ValueError, match="forget the test function"):
        jno.fem([_lap(v, psi, ci), v(ci2[0], ci2[1]) - 0.0])


def test_a_boundary_tag_is_unchanged():
    """The no-regression side: an ordinary boundary tag must resolve exactly as it always did."""
    d = _two_region()
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    d.tag("inlet", lambda x, y: x < 1e-9)
    ci = d.variable("interior", split=True)
    xin, yin, _ = d.variable("inlet", split=True)
    got = _pinned_nodes(d, [_lap(v, psi, ci), v(xin, yin) - 0.0])
    pts = np.asarray(d._fem_native_dof_points_all[0])
    assert set(got.tolist()) == set(np.flatnonzero(np.abs(pts[:, 0]) < 1e-9).tolist())


def test_the_physics_oracle_channel_flow_over_an_internal_wall():
    """What the missing condition cost. The wetted wall is an internal interface; without no-slip there
    the channel is not bounded and the answer is wrong by O(1). P2 velocity represents the Poiseuille
    parabola exactly and P1 the linear pressure drop exactly, so the discrete flow is exact -- up to the
    ballast that keeps the unused DOFs in the solid from being singular, which enters at O(EPS)."""
    EPS = 1e-10
    d = _two_region()
    v, psi = d.fem_symbols(value_shape=(2,), names=("v", "psi"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    d.tag("inlet", lambda x, y: x < 1e-9)
    d.tag("noslip", lambda x, y: (y > H - 1e-9) | (abs(y) < 1e-9))
    fr, sr = d.variable("fluid", split=True), d.variable("solid", split=True)
    iv = d.variable("fluid|solid", split=True)
    vb, wb = v.bind(x=fr[0], y=fr[1]), psi.bind(x=fr[0], y=fr[1])
    pb, qb = p.bind(x=fr[0], y=fr[1]), q.bind(x=fr[0], y=fr[1])
    mom = (
        MU * (vb.x[0] * wb.x[0] + vb.y[0] * wb.y[0] + vb.x[1] * wb.x[1] + vb.y[1] * wb.y[1])
        + (vb[0] * vb.x[0] + vb[1] * vb.y[0]) * wb[0]
        + (vb[0] * vb.x[1] + vb[1] * vb.y[1]) * wb[1]
        - pb * (wb.x[0] + wb.y[1])
    )
    vs, ws = v.bind(x=sr[0], y=sr[1]), psi.bind(x=sr[0], y=sr[1])
    ps, qs = p.bind(x=sr[0], y=sr[1]), q.bind(x=sr[0], y=sr[1])
    xin, yin, _ = d.variable("inlet", split=True)
    xw, yw, _ = d.variable("noslip", split=True)
    fem = jno.fem(
        [
            mom,
            qb * (vb.x[0] + vb.y[1]),
            EPS * (vs[0] * ws[0] + vs[1] * ws[1]),
            EPS * ps * qs,
            v(xin, yin)[0] - 4.0 * UMAX * yin * (H - yin) / H**2,
            v(xin, yin)[1] - 0.0,
            v(xw, yw) - 0.0,
            v(iv[0], iv[1]) - 0.0,  # the condition that used to vanish
        ]
    )
    sol = np.asarray(
        fem.solve(
            nonlinear=jno.solve.newton(direct=True, rtol=1e-11, atol=1e-13),
            linear=jno.solve.lu(backend="host"),
        )
    )
    off, ivx, ipx = list(fem.offsets), fem.block_index(v), fem.block_index(p)
    pv, pp = np.asarray(fem.field_points[ivx]), np.asarray(fem.field_points[ipx])
    vel = sol[off[ivx] : off[ivx + 1]].reshape(-1, 2)
    pre = sol[off[ipx] : off[ipx + 1]]
    F, FP = pv[:, 1] > -1e-12, pp[:, 1] > -1e-12
    e_v = np.abs(vel[F, 0] - 4.0 * UMAX * pv[F, 1] * (H - pv[F, 1]) / H**2).max()
    e_p = np.abs(pre[FP] - 8.0 * MU * UMAX / H**2 * (L - pp[FP, 0])).max()
    # measured: 1.9 * EPS and 7.6 * EPS. Before the fix these were 9.5e-01 and 2.4e+01.
    assert e_v < 100 * EPS, f"velocity {e_v:.3e}"
    assert e_p < 1e4 * EPS, f"pressure {e_p:.3e}"
    assert np.abs(vel[F, 1]).max() < 100 * EPS, "the transverse velocity must vanish"
