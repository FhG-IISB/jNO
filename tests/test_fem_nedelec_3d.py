"""3D Nédélec (N1E) edge elements — H(curl) mass + curl-curl on a tetrahedral mesh.

The first 3D non-nodal element in ``jno.fem``: lowest-order first-kind Nédélec on tets (6 edge DOFs,
``value_size=3``), the correct H(curl)-conforming discretisation for Maxwell / curl-curl problems. It
reuses the 2D edge-DOF machinery — the covariant Piola ``Φ_phys = J^{-T} Φ_ref`` is dimension-agnostic,
and the (vector) curl is recovered from the covariant physical gradient (its antisymmetric parts), so no
separate 3D curl push-forward is needed. The ``.curl(x, y, z)`` DSL view assembles it directly.

Reference: J.-C. Nédélec, *Mixed finite elements in* R³, Numer. Math. 35 (1980) 315–341 (first-kind
H(curl) edge elements).
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")

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


def test_nedelec_tet_element_pushforward():
    """Element level (no assembler): the covariant push-forward gives a symmetric-PD H(curl) mass, and the
    physical curl taken from the covariant gradient (antisymmetric parts) equals the contravariant push of
    the reference curl ``(1/detJ) J·curl_ref`` — the identity that lets ``.curl()`` skip a dedicated 3D curl
    map."""
    from jno.utils.solver.fem_elements import nedelec_tet, piola_covariant, piola_covariant_grad

    spec = nedelec_tet(degree=1, quad_degree=3)
    assert spec.n_dof == 6 and spec.value_size == 3
    rv, rg, qw = jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_grads), jnp.asarray(spec.quad_weights)

    verts = np.array([[0.0, 0, 0], [2.0, 0, 0], [0.3, 1.5, 0], [0.1, 0.2, 1.7]])  # a generic tet
    J = jnp.asarray(np.stack([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]], axis=1))
    detJ = jnp.linalg.det(J)
    signs = jnp.ones(6)
    vals, curl2 = piola_covariant(rv, None, J, detJ, signs)  # 3D N1E: no scalar curl
    assert curl2 is None
    grad = piola_covariant_grad(rg, J, detJ, signs)  # (nq, 6, 3, 3)

    M = jnp.einsum("q,qai,qbi->ab", qw * jnp.abs(detJ), vals, vals)  # H(curl) mass
    assert float(jnp.abs(M - M.T).max()) < 1e-12
    assert float(jnp.linalg.eigvalsh(M).min()) > 0.0

    def _curl(g):  # antisymmetric parts of a (..., 3, 3) gradient tensor -> (..., 3)
        return jnp.stack([g[..., 2, 1] - g[..., 1, 2], g[..., 0, 2] - g[..., 2, 0], g[..., 1, 0] - g[..., 0, 1]], -1)

    curl_from_grad = _curl(grad)
    ref_curl = _curl(rg)  # reference curl vector
    curl_piola = jnp.einsum("ij,qnj->qni", J, ref_curl) / detJ  # contravariant push
    assert float(jnp.abs(curl_from_grad - curl_piola).max()) < 1e-12


def _n1e_cube(mesh_size=0.5):
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    coords = d.variable("interior", split=True)
    xi, yi, zi = coords[0], coords[1], coords[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    return d, (xi, yi, zi), (ui, vi), (cu, cv)


def test_nedelec_tet_mass_and_curlcurl_symmetric_pd():
    """On a tet cube the coercive H(curl) operator ``∫ u·v + ∫ curl u · curl v`` assembles to a symmetric,
    positive-definite matrix of size ``n_edges`` — the whole ``value_size=3`` covariant edge path (mass +
    vector curl-curl through ``.curl(x,y,z)``)."""
    d, _, (ui, vi), (cu, cv) = _n1e_cube(0.5)
    A = _dense(jno.fem([inner(ui, vi) + inner(cu, cv)]).A)
    n_edges = A.shape[0]
    assert A.shape == (n_edges, n_edges)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(A).min()) > 0.0  # PD (the +mass block; pure curl-curl is singular)


def test_nedelec_tet_curl_curl_exact_bilinear():
    """The definitive edge-element check on a **multi-tet** mesh (shared edges of differing local
    orientation — a single tet would be a false green): a field in the N1E0 space, ``u* = ½(−y, x, 0)`` with
    constant ``curl u* = (0,0,1)``, has ``∫|curl u*|² = vol(unit cube) = 1``. Projecting ``u*`` for its edge
    DOFs and evaluating the pure curl-curl form ``uᵀ K u`` must reproduce ``1`` exactly — this validates the
    curl-curl assembly *and* the global tet-edge orientation together."""
    d, (xi, yi, zi), (ui, vi), (cu, cv) = _n1e_cube(0.5)
    _solve = lambda A, b: np.linalg.solve(_dense(A), np.asarray(jnp.asarray(b)).reshape(-1))  # noqa: E731
    M = _dense(jno.fem([inner(ui, vi)]).A)
    b = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - (-0.5 * yi * vi[0] + 0.5 * xi * vi[1] + 0.0 * vi[2])]).b))
    u_dof = np.linalg.solve(M, b.reshape(-1))
    K = _dense(jno.fem([inner(cu, cv)]).A)  # pure curl-curl (no mass)
    np.testing.assert_allclose(float(u_dof @ K @ u_dof), 1.0, atol=1e-9)


def test_3d_nonnodal_only_nedelec_supported():
    """On a 3D tet mesh only N1E is wired; the 2D-only families (here Morley) must raise a clear
    NotImplementedError rather than silently mis-map their 2D edge/vertex machinery onto a tet."""
    d = jno.domain(constructor=jno.domain.cube(mesh_size=0.6))
    u, phi = d.fem_symbols(space="Morley")
    coords = d.variable("interior", split=True)
    xi, yi, zi = coords[0], coords[1], coords[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    with pytest.raises(NotImplementedError, match="3D"):
        jno.fem([inner(jno.np.hessian(ui, [xi, yi, zi]), jno.np.hessian(vi, [xi, yi, zi]), n_contract=2) - vi]).A


def test_pec_tangential_pins_are_boundary_face_edges():
    """Decisive check for the 3D boundary-edge detection: the PEC BC ``n×E=0`` must pin **exactly** the edges
    of the boundary FACES (facet connectivity), value 0, with no interior edge touched. This is NOT the 2D
    "edge used by one cell" set — on a tet mesh most boundary edges are shared by several tets, so that rule
    would miss them (a wrong-DOF bug that still solves plausibly)."""
    from jno.utils.solver.fem_facets import build_facet_connectivity
    from jno.utils.solver.fem_nonnodal import _n1e_tangential_pins_3d
    from jno.utils.solver.fem_topology import BASIX_TET_EDGES, build_edge_topology

    d = jno.domain(constructor=jno.domain.cube(mesh_size=0.4))
    cells = np.asarray(d.mesh.cells_dict["tetra"])
    top = build_edge_topology(cells, BASIX_TET_EDGES)
    pins = _n1e_tangential_pins_3d([("u", "boundary", 0.0)], d, {"u": 0}, ["N1E"], top, [0])
    pinned = sorted(int(dof) for dof, _ in pins)
    assert {float(v) for _, v in pins} == {0.0}  # homogeneous PEC

    fc = build_facet_connectivity(cells, "tetrahedron")
    eid = {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(top.edge_vertices))}
    truth = set()
    for f in range(fc.n_bfaces):
        fn = [int(x) for x in fc.face_nodes[f]]
        for a, b in ((fn[0], fn[1]), (fn[1], fn[2]), (fn[0], fn[2])):
            truth.add(eid[(min(a, b), max(a, b))])
    assert pinned == sorted(truth)  # exactly the boundary-face edges

    counts = np.bincount(np.asarray(top.cell_edges).reshape(-1), minlength=top.n_edges)
    assert sum(1 for e in truth if counts[e] > 1) > 0  # multi-use boundary edges exist (2D rule would miss them)


def _driven_pec_l2_error(mesh_size):
    """Driven H(curl) problem ``curl-curl + mass`` with a manufactured field ``E* = (sin πy sin πz,
    sin πx sin πz, sin πx sin πy)`` (zero tangential trace on every cube face), PEC ``n×E=0`` on the whole
    boundary. Returns ``(n_dof, ‖E_h − E*‖_L²)`` via ``‖E-E*‖² = EᵀME − 2Eᵀb* + ∫|E*|²`` (``∫|E*|²=¾``)."""
    sin, cos, vec = jno.np.sin, jno.np.cos, jno.np.vector
    pi = np.pi
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    sx, sy, sz, cx, cy, cz = sin(pi * xi), sin(pi * yi), sin(pi * zi), cos(pi * xi), cos(pi * yi), cos(pi * zi)
    Estar = vec(sy * sz, sx * sz, sx * sy)
    curlE = vec(pi * sx * (cy - cz), pi * sy * (cz - cx), pi * sz * (cx - cy))  # curl E*
    nb = d.variable("boundary", normals=True, split=True)
    pec = u.vector.cross(vec(nb[4], nb[5], nb[6]))  # n×E = 0 on the whole boundary
    driven = [inner(cu, cv) + inner(ui, vi) - inner(curlE, cv) - inner(Estar, vi), pec]
    solve = lambda A, b: np.linalg.solve(_dense(A), np.asarray(jnp.asarray(b)).reshape(-1))  # noqa: E731
    M = _dense(jno.fem([inner(ui, vi)]).A)
    bstar = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - inner(Estar, vi)]).b)).reshape(-1)
    E = np.asarray(jno.fem(driven).solve(solve)).reshape(-1)
    e2 = float(E @ M @ E - 2.0 * (E @ bstar) + 0.75)
    return E.size, float(np.sqrt(max(e2, 0.0)))


def test_pec_driven_problem_converges():
    """End-to-end (gesture → peel → facet pin → solve): the driven manufactured problem with PEC on the whole
    boundary recovers ``E*``. ``E*`` is trigonometric (not in N1E₀) so recovery is not exact — but the L²
    error DECREASES under refinement, validating the PEC BC works in a real coupled solve."""
    n0, e0 = _driven_pec_l2_error(0.5)
    n1, e1 = _driven_pec_l2_error(0.25)
    assert n1 > n0
    assert e1 < 0.85 * e0, f"PEC driven problem must converge: {e0:.3e} (ndof {n0}) -> {e1:.3e} (ndof {n1})"
