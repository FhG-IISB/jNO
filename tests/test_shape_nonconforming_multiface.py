"""A non-conforming interface that spans SEVERAL geometric faces.

Every other non-conforming test in this suite glues two blocks across **one** shared face, and that
hid a defect that only an *embedded* body shows. A beam standing in a channel touches the fluid on
three faces -- its two sides and its top -- and the emitter matches the two coincident surfaces face
by face, grouping the mesher's outer faces by bounding box. It then wrote the result with

    cell_sets[f"{pair}.{region}"] = [empty, idx]

once per matched box, so each face **overwrote** the last and the tag kept only whichever came out of
the dict last. Measured on the beam below: the fluid side of the interface held 17 of its 37 facets.

Nothing failed. The dropped faces are still withheld from the catch-all ``"boundary"`` (they go into
``nonconf_iface`` before the assignment), so they end up in no region at all -- a tie transmits
nothing across them and a Dirichlet does not reach them. In a Navier-Stokes solve over that channel
the fluid simply flowed *through* the top two thirds of the beam, and Newton diverged to 6.1e+06
chasing a solution that did not exist. The one-body control -- the same channel with the beam cut out
as a hole, where the wall is ordinary outer boundary -- converged to 1.8544.

So the assertions are about coverage, and the oracle is geometric: the tag must contain every facet
of its own body that lies on the shared surface, and nothing else.
"""

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    """Coordinates are compared at 1e-9; in float32 a P2 midpoint lands ~1e-7 off and the geometric
    oracle misclassifies a node or two. FEM assembly requires x64 anyway."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)

#: Channel, and a beam standing on its floor. The beam's own bottom (y = 0) is part of the channel
#: floor -- an OUTER face, not an interface -- which is what stops this from being a symmetric case
#: where a bug in the matching could still look right.
_L, _H = 4.0, 2.0
_X0, _X1, _TOP = 1.5, 1.8, 1.2


def _channel():
    """The beam as its own body, meshed independently of the fluid around it."""
    notch = jno.Shape.rect(_X0, -0.1, _X1, _TOP)  # cut BELOW y=0 so the notch opens onto the floor
    return jno.Shape.regions(
        fluid=(jno.Shape.rect(0.0, 0.0, _L, _H) - notch).sized(0.12),
        beam=jno.Shape.rect(_X0, 0.0, _X1, _TOP).sized(0.09),
        conforming=False,
    ).domain()


def _on_wetted_surface(p):
    """The beam's three wetted faces, from the geometry alone. Its bottom is deliberately excluded."""
    x, y = p[..., 0], p[..., 1]  # `...` so this reads both a node array (N,2) and a facet one (M,2,2)
    sides = (np.abs(x - _X0) < 1e-9) | (np.abs(x - _X1) < 1e-9)
    return ((sides & (y > -1e-9) & (y < _TOP + 1e-9)) |
            ((np.abs(y - _TOP) < 1e-9) & (x > _X0 - 1e-9) & (x < _X1 + 1e-9)))


def _tag_facets(d, tag):
    """The line facets a tag owns, as vertex-id rows."""
    lines = np.asarray(d.built_mesh.cells_dict["line"])
    return lines[np.asarray(d.built_mesh.cell_sets[tag][1]).reshape(-1).astype(int)]


def _body_facets_on_surface(d, region):
    """The oracle: every facet of ``region``'s own cells that lies wholly on the wetted surface.

    Computed from the mesh and the geometry, never from the tag -- otherwise this would just restate
    whatever the emitter produced."""
    from jno.utils.solver.fem_utils import _cell_region_mask

    tri = np.asarray(d.built_mesh.cells_dict["triangle"])
    pts = np.asarray(d.built_mesh.points)[:, :2]
    mine = tri[np.asarray(_cell_region_mask(d, region)).reshape(-1) > 0]
    edges = np.sort(mine[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = uniq[counts == 1]  # a facet of this body's surface belongs to exactly one of its cells
    return boundary[_on_wetted_surface(pts[boundary]).all(axis=1)]


@pytest.mark.parametrize("region", ["fluid", "beam"])
def test_the_tag_covers_every_face_of_a_multi_face_interface(region):
    """The defect, stated directly: all three faces, not just the one that happened to be written last."""
    d = _channel()
    got = {frozenset(map(int, r)) for r in _tag_facets(d, f"beam|fluid.{region}")}
    want = {frozenset(map(int, r)) for r in _body_facets_on_surface(d, region)}
    assert want, "the oracle found no wetted facets -- the geometry is wrong, not the tag"
    assert got == want, f"{region}: tag has {len(got)} facets, the surface has {len(want)}"


def test_the_tag_spans_all_three_faces_and_not_the_beams_base():
    """Coverage per face, so a partial fix that restores two of three still fails -- and the beam's
    base is the control: it is an outer boundary that must NOT be swept in."""
    d = _channel()
    pts = np.asarray(d.built_mesh.points)[:, :2]
    mid = pts[_tag_facets(d, "beam|fluid.beam")].mean(axis=1)
    assert (np.abs(mid[:, 0] - _X0) < 1e-9).sum() > 3, "the beam's x=X0 face is missing"
    assert (np.abs(mid[:, 0] - _X1) < 1e-9).sum() > 3, "the beam's x=X1 face is missing"
    assert (np.abs(mid[:, 1] - _TOP) < 1e-9).sum() > 0, "the beam's top face is missing"
    assert (np.abs(mid[:, 1]) < 1e-9).sum() == 0, "the beam's base is outer boundary, not interface"


def test_a_dirichlet_on_the_interface_reaches_the_whole_wetted_surface():
    """How the gap actually bit. A no-slip written on the fluid side of the interface reached 34 of
    its 73 P2 wall nodes, so the condition was silently imposed on part of the wall only."""
    d = _channel()
    u, v = d.fem_symbols(order=2)
    c = d.variable("interior", split=True)
    ui, vi = u.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])
    w = d.variable("beam|fluid.fluid", split=True)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(w[0], w[1]) - 0.0])

    pinned = {int(a) for a, _g in (getattr(d, "_fem_native_dirichlet_pairs", None) or [])}
    pts = np.asarray(fem.field_points[0])
    wetted = set(np.flatnonzero(_on_wetted_surface(np.asarray(pts))).tolist())
    assert len(wetted) > 40, "expected a P2 wall node set worth measuring"
    assert wetted <= pinned, f"{len(wetted - pinned)} of {len(wetted)} wall nodes were left free"


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="The mortar parametrises an interface by ONE tangent plane (see `_interface_frame`), so a "
    "cornered interface folds its parallel faces onto the same interval and every secondary edge is "
    "covered twice. Reaching this is progress: while the tag held only one face the projection was "
    "flat and the tie 'worked' -- on a third of the interface. Remove the marker when the interface "
    "is parametrised by arc length along its facet chain instead.",
)
def test_the_tie_glues_across_every_face():
    """The physics oracle, and the one that would have caught the Navier-Stokes divergence: a tied
    two-body solve must reproduce the conforming single-mesh one. With two of three faces missing the
    bodies are joined along a single line and the peak is wrong, not merely less accurate."""

    def solve(conforming):
        notch = jno.Shape.rect(_X0, -0.1, _X1, _TOP)
        d = jno.Shape.regions(
            fluid=(jno.Shape.rect(0.0, 0.0, _L, _H) - notch).sized(0.12),
            beam=jno.Shape.rect(_X0, 0.0, _X1, _TOP).sized(0.09),
            conforming=conforming,
        ).domain()
        u, v = d.fem_symbols()
        c = d.variable("interior", split=True)
        ui, vi = u.bind(x=c[0], y=c[1]), v.bind(x=c[0], y=c[1])
        terms = [ui.x * vi.x + ui.y * vi.y - 1.0 * vi]
        if not conforming:
            a, b = (d.variable(t, split=True) for t in ("beam|fluid.fluid", "beam|fluid.beam"))
            terms.append(u(a[0], a[1]) - u(b[0], b[1]))
        zb = d.variable("boundary", split=True)
        terms.append(u(zb[0], zb[1]) - 0.0)
        return float(np.asarray(jno.fem(terms).solve()).reshape(-1).max())

    ref, got = solve(True), solve(False)
    assert abs(got - ref) / ref < 0.05, f"tied {got:.6f} vs conforming {ref:.6f}"
