"""Floquet/Bloch periodic ties on a Nédélec (N1E) edge field — DOF-level edge prolongation.

An N1E DOF is one edge's tangential moment, so a periodic tie matches boundary edges across the tied
faces (by transverse midpoint) with an orientation sign and a Bloch phase. Because edge DOFs must line
up one-to-one, jno RE-MESHES the box conforming (gmsh setPeriodic) automatically when it sees periodic
ties on an N1E field — inferred from the constraint list, no explicit request.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402

inner = jno.np.inner
_dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _periodic_box(size=0.3):
    d = jno.domain(jno.Shape.box(0, 0, 0, 0.6, 0.6, 1.0, size=size))
    e = 1e-6
    d.tag("left", lambda x, y, z: x < e)
    d.tag("right", lambda x, y, z: x > 0.6 - e)
    d.tag("front", lambda x, y, z: y < e)
    d.tag("back", lambda x, y, z: y > 0.6 - e)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)

    def face(nm):
        cc = d.variable(nm, split=True)
        return u.bind(x=cc[0], y=cc[1], z=cc[2])

    ties = [face("left") - face("right"), face("front") - face("back")]
    return d, (inner(cu, cv) + inner(ui, vi)), ties


def test_periodic_n1e_reduces_and_is_symmetric_pd():
    """Periodic ties on an N1E field eliminate the slave-face edge DOFs (``n_red < n_full``) and the reduced
    curl-curl + mass operator PᵀAP stays symmetric positive-definite. The box is auto-re-meshed conforming."""
    from jno.utils.solver.fem_utils import reduce_matrix_periodic

    d, vol, ties = _periodic_box(0.3)
    fem = jno.fem([vol, *ties])  # triggers the conforming re-mesh + edge reduction
    per = fem._periodic
    assert per is not None and per["n_red"] < per["n_full"], "periodic reduction eliminated no slave-face edges"

    A_red = _dense(reduce_matrix_periodic(per, fem.A))
    assert A_red.shape == (per["n_red"], per["n_red"])
    np.testing.assert_allclose(A_red, A_red.T, atol=1e-9)
    assert float(np.linalg.eigvalsh(A_red).min()) > 0.0  # +mass makes the reduced operator PD


def test_periodic_n1e_infers_remesh_from_constraints_only():
    """The conforming re-mesh is inferred purely from the periodic constraints — no explicit ``periodic=``
    argument. After a periodic solve the domain is marked, and a non-periodic assembly on it is a no-op."""
    d, vol, ties = _periodic_box(0.25)
    assert not getattr(d, "_periodic_meshed", frozenset())  # nothing yet
    jno.fem([vol, *ties])  # authoring periodic ties on an N1E field re-meshes the box
    marked = getattr(d, "_periodic_meshed", frozenset())
    assert frozenset({"left", "right"}) in marked and frozenset({"front", "back"}) in marked


def test_periodic_n1e_bloch_phase_is_complex():
    """A Bloch (quasi-periodic) phase makes the prolongation — and the reduced operator — complex, while a
    plain periodic tie stays real. Validates the phase threads through the edge prolongation."""
    d, vol, _ = _periodic_box(0.25)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)

    def face(nm):
        cc = d.variable(nm, split=True)
        return u.bind(x=cc[0], y=cc[1], z=cc[2])

    phase = np.exp(1j * 0.7)  # Bloch phase e^{iφ}
    fem = jno.fem([inner(cu, cv) + inner(ui, vi), face("left") - phase * face("right"), face("front") - face("back")])
    assert fem._periodic is not None and fem._periodic.get("is_bloch")  # quasi-periodic reduction


# --------------------------------------------------------------------------------------------
# which global edges sit on a periodic face: a vertex mask, not a per-edge Python scan
# --------------------------------------------------------------------------------------------
def _edges_on_tag_reference(edge_vertices, tag_vertex_ids):
    """The per-edge comprehension ``_edges_on_tag`` replaced, kept as the oracle."""
    vset = set(int(v) for v in np.asarray(tag_vertex_ids).reshape(-1))
    ev = np.asarray(edge_vertices).reshape(-1, 2)
    return np.asarray([e for e in range(len(ev)) if int(ev[e, 0]) in vset and int(ev[e, 1]) in vset], dtype=int)


@pytest.mark.parametrize(
    "ev,ids",
    [
        (np.array([[0, 1], [1, 2], [2, 3], [0, 3]]), np.array([0, 1, 2])),
        (np.array([[0, 1]]), np.array([5])),  # nothing on the tag
        (np.zeros((0, 2), dtype=int), np.array([1, 2])),  # no edges at all
        (np.array([[0, 1], [1, 2]]), np.zeros(0, dtype=int)),  # empty tag
        (np.array([[3, 7], [7, 9]]), np.array([9, 7, 3, 3])),  # unsorted ids, with a duplicate
    ],
)
def test_edges_on_tag_matches_the_per_edge_scan(ev, ids):
    from jno._fem import _edges_on_tag

    got, ref = _edges_on_tag(ev, ids), _edges_on_tag_reference(ev, ids)
    assert got.shape == ref.shape and np.array_equal(got, ref)


def test_edges_on_tag_matches_on_a_real_mesh():
    """Both faces of a real periodic pair, against the scan."""
    from jno._fem import _edges_on_tag
    from jno.utils.solver.fem_topology import BASIX_TET_EDGES, build_edge_topology

    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain()
    cells = np.asarray(d.built_mesh.cells_dict["tetra"])
    ev = build_edge_topology(cells, BASIX_TET_EDGES).edge_vertices
    pts = np.asarray(d.built_mesh.points)
    for lo_hi in (0.0, 1.0):
        ids = np.flatnonzero(np.abs(pts[:, 0] - lo_hi) < 1e-9)
        assert ids.size > 0
        assert np.array_equal(_edges_on_tag(ev, ids), _edges_on_tag_reference(ev, ids))
