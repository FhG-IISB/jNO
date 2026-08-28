"""Mixed NON-NODAL x NODAL fields on the non-nodal assembler: the N1E x Lagrange (A-V) pair.

``jno.fem`` routes a form to the non-nodal assembler as soon as any field is non-nodal, and that
path used to admit only RT / N1E / P0 / Hermite / Argyris / Morley. A nodal ``Lagrange`` scalar
alongside an N1E field was therefore rejected outright -- which rules out the A-V pair, where the
magnetic vector potential A is H(curl) (edge DOFs) and the electric scalar potential V is nodal.
V is what carries a terminal condition on a cut conductor; A alone cannot express one.

Two behaviours are pinned here beyond "it assembles":

* the Dirichlet judgement is **per field**, not global. The old check rejected every nodal
  Dirichlet the moment an edge field existed anywhere in the form, so ``V = g`` was unreachable.
  A nodal Dirichlet on the N1E field must still raise -- its essential BC is the edge trace.
* the Lagrange pins are **per vertex**, so a position-dependent ``g`` is enforced pointwise rather
  than collapsed to one constant. ``test_..._per_vertex`` uses ``g = x`` and asserts free interior
  vertices exist, so it cannot pass on a mesh whose every vertex is pinned.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")
import jax  # noqa: E402

import jno  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _grad(s):  # scalar gradient, TRIAL side -> (n_quad, 3)
    return vec(s.x, s.y, s.z)


def _tgrad(s):  # scalar gradient, TEST side -> (n_quad, n_dof, 3), the rank an N1E test basis has
    return jno.np.stack([s.x, s.y, s.z], axis=2)


def _av(size=0.3, g=None, dirichlet_on_edge_field=False):
    """The A-V operator with jw -> 1: a curl-curl + mass block on N1E, a Laplacian on Lagrange, and
    the grad-V / A coupling that makes it genuinely mixed rather than two independent problems."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    p, q = d.fem_symbols(names=("p", "q"), space="Lagrange")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    b = d.variable("boundary", split=True)

    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    Vs, Vt = p.bind(x=x, y=y, z=z), q.bind(x=x, y=y, z=z)
    cA, cV = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    f = vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x)

    terms = [
        inner(cA, cV) + inner(A_, V_) + inner(_grad(Vs), V_) - inner(f, V_),
        inner(A_, _tgrad(Vt)) + inner(_grad(Vs), _tgrad(Vt)),
        u.vector.cross(d.variable("boundary", normals=True)),
    ]
    if dirichlet_on_edge_field:
        terms.append(u.bind(x=b[0], y=b[1], z=b[2]) - 0.0)
    else:
        terms.append(p.bind(x=b[0], y=b[1], z=b[2]) - (b[0] if g is None else g(b)))
    return d, jno.fem(terms)


def _blocks(d, fem):
    offs = [int(o) for o in fem.offsets]
    return offs, [offs[i + 1] - offs[i] for i in range(len(offs) - 1)]


def test_mixed_n1e_lagrange_assembles_with_one_block_per_space():
    """The headline: on main this raised NotImplementedError before assembling anything."""
    d, fem = _av(size=0.6)
    offs, blk = _blocks(d, fem)
    n_verts = len(d.mesh.points)
    assert len(blk) == 2, f"expected an N1E block and a Lagrange block, got {blk}"
    assert n_verts in blk, f"no block sized one DOF per mesh vertex (n_verts={n_verts}); blocks {blk}"
    n_edges = [n for n in blk if n != n_verts][0]
    assert n_edges > n_verts, "the N1E block should carry one DOF per EDGE, which outnumbers vertices"


def test_lagrange_dirichlet_is_enforced_per_vertex():
    """A position-dependent ``g`` must be enforced vertex-by-vertex, not broadcast from one value."""
    from jno.utils.solver.fem_1d import _region_node_ids

    d, fem = _av(size=0.3)  # fine enough that interior vertices exist
    pts = np.asarray(d.mesh.points)
    n_verts = len(pts)
    bn = np.asarray(_region_node_ids(d, "boundary"), dtype=np.int64)
    free = np.setdiff1d(np.arange(n_verts), bn)
    assert free.size > 0, "every vertex is pinned -- the test would be vacuous; refine the mesh"

    offs, blk = _blocks(d, fem)
    iv = blk.index(n_verts)
    V = np.asarray(jno.np.asarray(fem.solve())).reshape(-1)[offs[iv] : offs[iv + 1]]

    assert np.abs(V[bn] - pts[bn, 0]).max() < 1e-8, "g = x is not enforced pointwise on the region"
    assert np.ptp(V[free]) > 1e-3, "the unpinned interior did not vary -- it was pinned too"


def test_nodal_dirichlet_on_the_edge_field_still_raises():
    """Per-field judgement must not become 'anything goes': N1E takes the edge trace, not a value."""
    with pytest.raises(NotImplementedError, match="edge trace"):
        _av(size=0.6, dirichlet_on_edge_field=True)


def test_the_space_message_lists_every_space_it_admits():
    """The message and the tuple it guards drifted apart: Lagrange was admitted but still denied in
    the text, so the next person to hit a genuinely unsupported mix got advice contradicting the
    code. Pin them together."""
    import inspect

    from jno.utils.solver import fem_nonnodal

    src = inspect.getsource(fem_nonnodal)
    i = src.index("supported element spaces")
    msg = src[i : i + 400]
    for space in ("RT", "N1E", "P0", "Hermite", "Argyris", "Morley", "Lagrange"):
        assert space in msg, f"{space} is admitted by the guard but missing from its message"


def test_block_composition_inherits_complex_native():
    """``triangular((A, ams()), (V, amg()))`` on a complex mixed system must declare itself
    complex-native, or it falls through to the fused real-equivalent 2n operator: the block slices
    then describe the n-sized complex layout and cover the wrong half, and AMS is handed the
    skew-dominated block its own docs say it diverges on."""
    tri = jno.precond.triangular(("u", jno.precond.ams()), ("p", jno.precond.amg()))
    bd = jno.precond.block_diag(("u", jno.precond.ams()), ("p", jno.precond.amg()))
    assert tri.complex_native, "triangular(ams, amg) must be complex-native"
    assert bd.complex_native, "block_diag(ams, amg) must be complex-native"
    plain = jno.precond.block_diag(("u", jno.precond.amg()), ("p", jno.precond.amg()))
    assert not plain.complex_native, "without a complex-native child there is nothing to inherit"


def _curl_curl(pins=None, size=0.6):
    """Curl-curl + mass on N1E, optionally with caller-supplied ``(dof, value)`` pins."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    A_, V_ = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    cA, cV = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    if pins is not None:
        d._extra_dof_pins = pins
    return d, jno.fem(
        [
            inner(cA, cV) + inner(A_, V_) - inner(vec(1.0 + 0.0 * x, 0.0 * x, 0.0 * x), V_),
            u.vector.cross(d.variable("boundary", normals=True)),
        ]
    )


def test_extra_dof_pins_are_applied():
    """``domain._extra_dof_pins`` is how a caller imposes a gauge the DSL cannot express -- a
    tree-cotree spanning tree, an air-region restriction. The unpinned values are asserted nonzero
    first, so this cannot pass by pinning DOFs that were already at the target."""
    _d, free = _curl_curl()
    s0 = np.asarray(jno.np.asarray(free.solve())).reshape(-1)
    assert abs(s0[3]) > 1e-6 and abs(s0[7]) > 1e-6, "unpinned DOFs are already ~0; pick different ones"

    _d, pinned = _curl_curl(pins=[(3, 0.0), (7, 0.5)])
    s1 = np.asarray(jno.np.asarray(pinned.solve())).reshape(-1)
    assert abs(s1[3] - 0.0) < 1e-12
    assert abs(s1[7] - 0.5) < 1e-12


def test_out_of_range_extra_dof_pins_raise():
    """A pin list built against a different mesh or DOF layout would otherwise pin arbitrary DOFs
    and return a plausible field, so the range check must fail loudly instead."""
    _d, fem = _curl_curl()
    ndof = int(np.asarray(jno.np.asarray(fem.b)).size)
    with pytest.raises(ValueError, match="outside"):
        _curl_curl(pins=[(ndof + 50, 0.0)])
