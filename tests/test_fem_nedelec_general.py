"""Higher-order H(curl) / H(div) elements (N1E_k, N2E_k, RT_k) and Lagrange P_k on the non-nodal path.

Every DOF beyond the lowest order lives on an edge, face or cell interior and must be ORIENTED: an edge
reversal mixes the edge's ``k`` DOFs, a face rotation/reflection mixes the face's. jNO takes those maps
from basix's base transformations (``jno/utils/solver/fem_dofmap.py``) -- these tests pin that they are
applied the right way round (conformity), that degree 1 is untouched, and that the resulting spaces
deliver what the theory promises: polynomial reproduction, rate-k convergence, a spurious-free Maxwell
spectrum, exact nonzero traces, and the discrete de Rham property ``curl(G φ) = 0``.

References: J.-C. Nédélec, Numer. Math. 35 (1980) 315-341 and 50 (1986) 57-81 (first/second kind);
P.-A. Raviart & J.-M. Thomas (1977); M. W. Scroggs, J. S. Dokken, C. N. Richardson, G. N. Wells,
ACM Trans. Math. Softw. 48(2) (2022), "Construction of arbitrary order finite element degree-of-freedom
maps on polygonal and polyhedral cell meshes" (the base-transformation formalism used here);
P. Monk, *Finite Element Methods for Maxwell's Equations* (2003), Thm 6.10 / 8.15 (convergence rates).
"""

from __future__ import annotations

import functools
import operator
from collections import defaultdict

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for meshing")
pytest.importorskip("basix")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_dofmap import build_dofmap, cell_jacobians  # noqa: E402
from jno.utils.solver.fem_nonnodal import nonnodal_field_at  # noqa: E402

inner = jno.np.inner
PI = float(np.pi)


def _dense(A):
    return np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)


def _sum(terms):
    return functools.reduce(operator.add, terms)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ------------------------------------------------------------------------------------------------
# the DOF map itself
# ------------------------------------------------------------------------------------------------


def _random_mesh(tdim, n, seed):
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    P = rng.random((n, tdim))
    cells = Delaunay(P).simplices.astype(np.int64)
    perm = rng.permutation(n)  # scramble the global vertex labels: orientations of every kind occur
    Pp = np.empty_like(P)
    Pp[perm] = P
    return Pp, perm[cells]


def _trace_jump(dm, P, cells, u, rng):
    """Largest jump of the conforming trace (tangential / normal / full) across interior facets."""
    tdim = cells.shape[1] - 1
    fd = tdim - 1
    owners = defaultdict(list)
    ce = dm.cell_entities[fd]
    for c in range(cells.shape[0]):
        for j in range(ce.shape[1]):
            owners[int(ce[c, j])].append(c)
    J, x0 = cell_jacobians(P, cells)
    worst = 0.0
    for f, cs in owners.items():
        if len(cs) != 2:
            continue
        fv = P[dm.entity_vertices[fd][f]]
        X = rng.dirichlet(np.ones(tdim), 3) @ fv
        vals = []
        for c in cs:
            xi = np.linalg.solve(J[c], (X - x0[c]).T).T
            vals.append(np.asarray(nonnodal_field_at(P, cells, dm, jnp.asarray(u), ref_points=xi, cell_ids=[c]))[0])
        if tdim == 3:
            n = np.cross(fv[1] - fv[0], fv[2] - fv[0])
        else:
            t = fv[1] - fv[0]
            n = np.array([-t[1], t[0]])
        if dm.family in ("N1E", "N2E"):
            tr = [np.cross(v, n) if tdim == 3 else v @ np.array([-n[1], n[0]]) for v in vals]
        elif dm.family == "RT":
            tr = [v @ n for v in vals]
        else:
            tr = vals
        worst = max(worst, float(np.abs(tr[0] - tr[1]).max()))
    return worst


@pytest.mark.parametrize("tdim", [2, 3])
@pytest.mark.parametrize("family", ["N1E", "N2E", "RT", "Lagrange"])
def test_global_space_is_conforming_for_every_orientation(tdim, family):
    """The measurement that fixed ``B = T(cell_info)⁻¹``: on a Delaunay mesh with SCRAMBLED vertex labels
    (so every edge reversal and face rotation/reflection occurs), a random global DOF vector has a
    continuous tangential (H(curl)) / normal (H(div)) / full (H¹) trace across every interior facet.
    With ``T`` or ``Tᵀ`` in place of ``T⁻¹`` the tet traces jump by O(1)-O(100)."""
    rng = np.random.default_rng(3)
    P, cells = _random_mesh(tdim, 14 if tdim == 3 else 16, seed=tdim)
    for k in (1, 2, 3):
        if family == "Lagrange" and k == 1:
            continue
        dm = build_dofmap(cells, family, k, n_verts=P.shape[0])
        u = rng.standard_normal(dm.n_dofs)
        assert _trace_jump(dm, P, cells, u, rng) < 1e-10, (family, k)


def test_degree_one_map_is_the_edge_topology():
    """Degree 1 must be the historic edge map exactly: same first-encounter edge numbering, same ±1
    signs (exactly, not to rounding -- basix's RT reflection is -1 - 2e-16 and is snapped)."""
    from jno.utils.solver.fem_topology import BASIX_TET_EDGES, BASIX_TRIANGLE_EDGES, build_edge_topology

    for tdim in (2, 3):
        P, cells = _random_mesh(tdim, 20, seed=7)
        top = build_edge_topology(cells, BASIX_TET_EDGES if tdim == 3 else BASIX_TRIANGLE_EDGES)
        fams = ("N1E", "RT") if tdim == 2 else ("N1E",)
        for fam in fams:
            dm = build_dofmap(cells, fam, 1, n_verts=P.shape[0])
            assert dm.is_diagonal
            np.testing.assert_array_equal(dm.cell_dofs, top.cell_edges)
            np.testing.assert_array_equal(dm.signs, top.cell_edge_signs.astype(np.float64))
            assert set(np.unique(dm.signs)) <= {-1.0, 1.0}


# ------------------------------------------------------------------------------------------------
# helpers to set up problems through jno.fem
# ------------------------------------------------------------------------------------------------


def _domain(tdim, h):
    if tdim == 2:
        return jno.domain(box(0, 0, 1, 1), mesh_size=h)
    return jno.Shape.box(0, 0, 0, 1, 1, 1, size=h).domain()


def _bound(d, tdim, space, k, names=("u", "v")):
    u, v = d.fem_symbols(value_shape=(tdim,), names=names, space=space, order=k)
    c = d.variable("interior", split=True)
    X = dict(zip("xyz", c[:tdim]))
    return u, v, u.bind(**X), v.bind(**X), c[:tdim]


def _readback(d, sol, ref_points=None, derivative=None, block=None):
    topo = d._fem_nonnodal_topology
    dm = topo["dofmap"]
    pts = np.asarray(d.mesh.points)[:, : dm.tdim]
    cells = topo["cells"]
    u = np.asarray(sol).reshape(-1)
    u = u[: dm.n_dofs] if block is None else u[block[0] : block[1]]
    rp = ref_points if ref_points is not None else np.random.default_rng(0).dirichlet(np.ones(dm.tdim + 1), 6)[:, : dm.tdim]
    val = np.asarray(nonnodal_field_at(pts, cells, dm, jnp.asarray(u), ref_points=rp, derivative=derivative))
    J, x0 = cell_jacobians(pts, cells)
    X = x0[:, None, :] + np.einsum("qa,cda->cqd", rp, J)
    return val, X, cells


def _sparse_solve(fem):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    A, b = fem.operator
    idx = np.asarray(A.indices)
    As = sp.csr_matrix((np.asarray(A.data), (idx[:, 0], idx[:, 1])), shape=A.shape)
    return spla.spsolve(As.tocsc(), np.asarray(b).reshape(-1))


def _vec(fs):
    return jno.np.vector(*fs)


# ------------------------------------------------------------------------------------------------
# reproduction, convergence, spectrum, traces
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tdim", [2, 3])
def test_projection_reproduces_polynomials_of_the_space(tdim):
    """N1E_k contains every vector polynomial of degree k-1: the L² projection of one is exact."""
    for k in (1, 2, 3):
        if tdim == 3 and k == 3:
            h = 0.9
        else:
            h = 0.45 if tdim == 2 else 0.7
        d = _domain(tdim, h)
        u, v, ui, vi, c = _bound(d, tdim, "N1E", k)
        x, y = c[0], c[1]
        if tdim == 2:
            g = [1.0 + 0 * x, 2.0 + 0 * x] if k == 1 else ([1 + y, 2 - x] if k == 2 else [x * y + 1, x * x - y])
        else:
            z = c[2]
            g = (
                [1.0 + 0 * x, 2.0 + 0 * x, -1.0 + 0 * x]
                if k == 1
                else ([1 + y, 2 - x, z] if k == 2 else [x * y, z * z, x - y * z])
            )
        fem = jno.fem([inner(ui, vi) - _sum([gi * vi[i] for i, gi in enumerate(g)])])
        sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
        val, X, _ = _readback(d, sol)
        xx, yy = X[..., 0], X[..., 1]
        if tdim == 2:
            ex = (
                [np.ones_like(xx), 2 * np.ones_like(xx)]
                if k == 1
                else ([1 + yy, 2 - xx] if k == 2 else [xx * yy + 1, xx * xx - yy])
            )
        else:
            zz = X[..., 2]
            ex = (
                [np.ones_like(xx), 2 * np.ones_like(xx), -np.ones_like(xx)]
                if k == 1
                else ([1 + yy, 2 - xx, zz] if k == 2 else [xx * yy, zz * zz, xx - yy * zz])
            )
        np.testing.assert_allclose(val, np.stack(ex, -1), atol=1e-10, err_msg=f"k={k}")


def _curlcurl_2d_error(k, h):
    """curl curl u + u = f, u×n = 0 on the unit square, u = (sin πy, sin πx): L² error of u_h."""
    d = _domain(2, h)
    u, v, ui, vi, (x, y) = _bound(d, 2, "N1E", k)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ub = u.bind(x=xb, y=yb)
    fac = PI**2 + 1.0
    f = [fac * jno.np.sin(PI * y), fac * jno.np.sin(PI * x)]
    fem = jno.fem([inner(ui, vi) + ui.curl() * vi.curl() - (f[0] * vi[0] + f[1] * vi[1]), ub[0] * ny - ub[1] * nx - 0.0])
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    gp, gw = _tri_rule(2 * k + 4)
    val, X, cells = _readback(d, sol, ref_points=gp)
    pts = np.asarray(d.mesh.points)[:, :2]
    J, _ = cell_jacobians(pts, cells)
    ex = np.stack([np.sin(PI * X[..., 1]), np.sin(PI * X[..., 0])], -1)
    err2 = np.einsum("q,c,cq->", gw, np.abs(np.linalg.det(J)), np.sum((val - ex) ** 2, -1))
    return float(np.sqrt(err2)), int(sol.size)


def _tri_rule(deg):
    import basix

    qp, qw = basix.make_quadrature(basix.CellType.triangle, deg)
    return np.asarray(qp), np.asarray(qw)


def _tet_rule(deg):
    import basix

    qp, qw = basix.make_quadrature(basix.CellType.tetrahedron, deg)
    return np.asarray(qp), np.asarray(qw)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_curl_curl_2d_converges_at_rate_k(k):
    """N1E_k is O(h^k) in L² for a smooth solution (Monk 2003, Thm 6.10 with the curl term)."""
    hs = (0.5, 0.25, 0.125)
    errs = [_curlcurl_2d_error(k, h)[0] for h in hs]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
    assert rates[-1] > k - 0.3, f"k={k}: errors {errs}, rates {rates}"


def _curlcurl_3d_error(k, h):
    """curl curl u + u = f, n×u = 0 on the unit cube, u = (sin πy sin πz, sin πx sin πz, sin πx sin πy)."""
    d = _domain(3, h)
    u, v, ui, vi, (x, y, z) = _bound(d, 3, "N1E", k)
    cu, cv = ui.vector.curl(x, y, z), vi.vector.curl(x, y, z)
    nvec = d.variable("boundary", normals=True)
    s = jno.np.sin
    # curl curl u = 2π² u for this field (each component is a product of two sines, ∇·u = 0)
    fac = 2 * PI**2 + 1.0
    f = _vec([fac * s(PI * y) * s(PI * z), fac * s(PI * x) * s(PI * z), fac * s(PI * x) * s(PI * y)])
    fem = jno.fem([inner(cu, cv) + inner(ui, vi) - inner(f, vi), u.vector.cross(nvec)])
    sol = _sparse_solve(fem)
    gp, gw = _tet_rule(2 * k + 2)
    val, X, cells = _readback(d, sol, ref_points=gp)
    pts = np.asarray(d.mesh.points)[:, :3]
    J, _ = cell_jacobians(pts, cells)
    sx, sy, sz = (np.sin(PI * X[..., i]) for i in range(3))
    ex = np.stack([sy * sz, sx * sz, sx * sy], -1)
    err2 = np.einsum("q,c,cq->", gw, np.abs(np.linalg.det(J)), np.sum((val - ex) ** 2, -1))
    return float(np.sqrt(err2)), int(sol.size)


def test_curl_curl_3d_converges_at_rate_k():
    """3-D, k = 2: second-order L² convergence on tets (face DOFs with every rotation/reflection)."""
    e0, n0 = _curlcurl_3d_error(2, 0.5)
    e1, n1 = _curlcurl_3d_error(2, 0.25)
    rate = np.log(e0 / e1) / np.log((n1 / n0) ** (1 / 3))
    assert rate > 1.6, f"N1E_2 3-D: errors {e0:.3e} ({n0} dofs) -> {e1:.3e} ({n1} dofs), rate {rate:.2f}"


def _gen_eigs(K, M, keep):
    Ki, Mi = K[np.ix_(keep, keep)], M[np.ix_(keep, keep)]
    L = np.linalg.cholesky(Mi)
    Asym = np.linalg.solve(L, np.linalg.solve(L, Ki).T).T
    return np.sort(np.linalg.eigvalsh(0.5 * (Asym + Asym.T)))


def _pec_pins(d, field_space, k):
    from jno.utils.solver.fem_nonnodal import _general_trace_pins

    topo = d._fem_nonnodal_topology
    dm = topo["dofmap"]
    pts = np.asarray(d.mesh.points)[:, : dm.tdim]
    return {dof for dof, _ in _general_trace_pins(dm, "tangential", "boundary", 0.0, d, pts, topo["cells"], 0)}


@pytest.mark.parametrize("k", [2, 3])
def test_square_cavity_spectrum_is_exact_and_spurious_free(k):
    """PEC unit square, curl curl E = λ E: λ = π²(m² + n²) with the right multiplicities
    (1,1,2,4,4,5,5,8 in units of π²), a kernel of exactly dim(grad P_k,0) zero eigenvalues and
    NOTHING spurious in between (the property nodal elements lack)."""
    d = _domain(2, 0.25)
    u, v, ui, vi, _ = _bound(d, 2, "N1E", k)
    K = _dense(jno.fem([ui.curl() * vi.curl()]).A)
    M = _dense(jno.fem([inner(ui, vi)]).A)
    pinned = _pec_pins(d, "N1E", k)
    keep = np.array([i for i in range(K.shape[0]) if i not in pinned])
    w = _gen_eigs(K, M, keep)
    nonzero = w[w > 1.0]
    kernel = int(np.sum(w < 1e-6))
    assert np.all(w[(w >= 1e-6)] > 1.0), "an eigenvalue sits between the kernel and the first mode (spurious)"
    exact = PI**2 * np.array([1, 1, 2, 4, 4, 5, 5, 8])
    rel = np.abs(nonzero[:8] - exact) / exact
    assert rel.max() < (5e-3 if k == 2 else 5e-4), f"k={k}: {nonzero[:8] / PI**2}"  # measured 2.7e-3 / 2.3e-4
    # kernel = gradients of the interior P_k space: interior vertices + (k-1)*interior edges + interior cell dofs
    dmL = build_dofmap(d._fem_nonnodal_topology["cells"], "Lagrange", k, n_verts=np.asarray(d.mesh.points).shape[0])
    from jno.utils.solver.fem_dofmap import entity_dofs_of_tasks, region_trace_entities

    mask = np.asarray(d.tag_node_mask("boundary", np.asarray(d.mesh.points)), dtype=bool).reshape(-1)
    bnd = set(entity_dofs_of_tasks(dmL, region_trace_entities(dmL, mask, boundary_only=True)).tolist())
    used = np.unique(dmL.cell_dofs)
    assert kernel == len(set(used.tolist()) - bnd), f"kernel {kernel} != interior P_{k} dofs"


def test_cube_cavity_spectrum_at_degree_two():
    """PEC unit cube, N1E_2 on a coarse mesh: 2π² (×3) then 3π² (×2), spurious-free."""
    d = _domain(3, 0.5)
    u, v, ui, vi, (x, y, z) = _bound(d, 3, "N1E", 2)
    cu, cv = ui.vector.curl(x, y, z), vi.vector.curl(x, y, z)
    K = _dense(jno.fem([inner(cu, cv)]).A)
    M = _dense(jno.fem([inner(ui, vi)]).A)
    pinned = _pec_pins(d, "N1E", 2)
    keep = np.array([i for i in range(K.shape[0]) if i not in pinned])
    w = _gen_eigs(K, M, keep)
    assert np.all(w[w >= 1e-6] > 0.8 * 2 * PI**2), "spurious mode below the first cavity resonance"
    nz = w[w > 1.0]
    # h = 0.5 is two cells across: measured 2π² within 2%, 3π² within 4.5% (N1E_1 is 12% off at h = 0.4)
    np.testing.assert_allclose(nz[:3] / PI**2, 2.0, rtol=0.03)
    np.testing.assert_allclose(nz[3:5] / PI**2, 3.0, rtol=0.06)
    assert nz[5] / PI**2 > 4.5  # next is 5π² (×6)


def _tag_faces(d, tdim):
    faces = {}
    for ax in range(tdim):
        for side, val in ((0, 0.0), (1, 1.0)):
            name = f"{'xyz'[ax]}{side}"
            n = [0.0] * tdim
            n[ax] = 1.0 if side else -1.0
            d.tag(name, (lambda a, v: lambda *X: np.abs(X[a] - v) < 1e-9)(ax, val))
            faces[name] = tuple(n)
    return faces


@pytest.mark.parametrize("tdim", [2, 3])
def test_nonzero_tangential_trace_is_exact(tdim):
    """curl curl u + u = f with the INHOMOGENEOUS essential trace u×n = g taken from a polynomial u in
    the space: the discrete solution is u itself (to round-off). k = 3 in 2-D (u quadratic), k = 2 in
    3-D (u linear) -- every boundary edge AND face DOF is set by interpolating g. One BC per side, each
    with its own data (the value node is a function of position only)."""
    k = 3 if tdim == 2 else 2
    d = _domain(tdim, 0.4 if tdim == 2 else 0.6)
    faces = _tag_faces(d, tdim)
    u, v, ui, vi, c = _bound(d, tdim, "N1E", k)
    bcs = []
    if tdim == 2:
        x, y = c
        uex = lambda X: [X[1] * X[1], X[0] * X[0] + X[1]]  # noqa: E731  curl u = 2x - 2y; curl curl u = (-2, -2)
        f = [y * y - 2.0, x * x + y - 2.0]
        for name, n in faces.items():
            xb, yb, _, nx, ny = d.variable(name, normals=True, split=True)
            ub = u.bind(x=xb, y=yb)
            ue = uex([xb, yb])
            bcs.append(ub[0] * ny - ub[1] * nx - (ue[0] * n[1] - ue[1] * n[0]))
        fem = jno.fem([inner(ui, vi) + ui.curl() * vi.curl() - (f[0] * vi[0] + f[1] * vi[1])] + bcs)
        exact = lambda X: np.stack([X[..., 1] ** 2, X[..., 0] ** 2 + X[..., 1]], -1)  # noqa: E731
    else:
        x, y, z = c
        cu, cv = ui.vector.curl(x, y, z), vi.vector.curl(x, y, z)
        uex = lambda X: [X[1] + X[2], 2 * X[0] - X[2], X[0] + 3 * X[1]]  # noqa: E731  linear: curl curl u = 0
        f = _vec(uex([x, y, z]))
        for name, n in faces.items():
            nvec = d.variable(name, normals=True)
            Xb = d.variable(name, split=True)[:3]
            ue = uex(Xb)
            g = [ue[1] * n[2] - ue[2] * n[1], ue[2] * n[0] - ue[0] * n[2], ue[0] * n[1] - ue[1] * n[0]]  # u×n
            bcs.append(u.vector.cross(nvec) - _vec(g))
        fem = jno.fem([inner(cu, cv) + inner(ui, vi) - inner(f, vi)] + bcs)
        exact = lambda X: np.stack([X[..., 1] + X[..., 2], 2 * X[..., 0] - X[..., 2], X[..., 0] + 3 * X[..., 1]], -1)  # noqa: E731
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))
    val, X, _ = _readback(d, sol)
    np.testing.assert_allclose(val, exact(X), atol=1e-9)


# ------------------------------------------------------------------------------------------------
# surface terms at degree k
# ------------------------------------------------------------------------------------------------


def _cube_face_integral(fn, n_gauss=5):
    """∮ fn(X, n) dS over the unit cube, tensor Gauss on each face (independent of jNO)."""
    g, w = np.polynomial.legendre.leggauss(n_gauss)
    g, w = 0.5 * (g + 1), 0.5 * w
    S, T = np.meshgrid(g, g, indexing="ij")
    W = np.outer(w, w).reshape(-1)
    tot = 0.0
    for ax in range(3):
        for val, sgn in ((0.0, -1.0), (1.0, 1.0)):
            o = [a for a in range(3) if a != ax]
            X = np.zeros((S.size, 3))
            X[:, ax] = val
            X[:, o[0]], X[:, o[1]] = S.reshape(-1), T.reshape(-1)
            n = np.zeros(3)
            n[ax] = sgn
            tot += float(np.sum(W * fn(X, n)))
    return tot


def test_impedance_mass_and_incident_load_are_exact_at_degree_two():
    """``∮ c (u×n)·(v×n)`` and ``∮ g·(v×n)`` assembled at degree 2 reproduce the boundary integrals of
    polynomial fields of the space exactly -- face DOFs included, with their orientation."""
    k = 2
    d = _domain(3, 0.6)
    u, v, ui, vi, (x, y, z) = _bound(d, 3, "N1E", k)
    cu, cv = ui.vector.curl(x, y, z), vi.vector.curl(x, y, z)
    nvec = d.variable("boundary", normals=True)
    tu, tv = u.vector.cross(nvec), v.vector.cross(nvec)
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    vol = inner(cu, cv) + inner(ui, vi)
    S = _dense(jno.fem([vol, 2.0 * inner(tu, tv)]).A) - _dense(jno.fem([vol]).A)
    g = _vec([1.0 + 0.0 * xb, zb, xb * yb])
    L = np.asarray(jno.fem([vol, inner(g, tv)]).b).reshape(-1) - np.asarray(jno.fem([vol]).b).reshape(-1)
    M = _dense(jno.fem([inner(ui, vi)]).A)

    def dofs(f_sym):
        return np.linalg.solve(M, np.asarray(jno.fem([inner(ui, vi) - inner(_vec(f_sym), vi)]).b).reshape(-1))

    a = dofs([y + 0.0 * x, z + 0.0 * x, x + 0.0 * x])
    b = dofs([1.0 + z, x + 0.0 * x, y + 0.0 * x])
    ua = lambda X: np.stack([X[:, 1], X[:, 2], X[:, 0]], 1)  # noqa: E731
    ub = lambda X: np.stack([1 + X[:, 2], X[:, 0], X[:, 1]], 1)  # noqa: E731
    gx = lambda X: np.stack([np.ones(len(X)), X[:, 2], X[:, 0] * X[:, 1]], 1)  # noqa: E731
    exact_S = _cube_face_integral(lambda X, n: 2.0 * np.einsum("qi,qi->q", np.cross(ua(X), n), np.cross(ub(X), n)))
    # the incident term is a SOURCE: it enters b with + (the convention of `_assemble_n1e_surface_load`,
    # unchanged from degree 1), b ∋ +∮ g·(v×n)
    exact_L = _cube_face_integral(lambda X, n: np.einsum("qi,qi->q", gx(X), np.cross(ub(X), n)))
    np.testing.assert_allclose(a @ S @ b, exact_S, rtol=1e-10)
    np.testing.assert_allclose(b @ L, exact_L, rtol=1e-10)


# ------------------------------------------------------------------------------------------------
# H(div): the natural pressure load at degree k
# ------------------------------------------------------------------------------------------------


def test_general_pressure_load_reproduces_the_rt0_shortcut():
    """At degree 1 the facet-quadrature pressure load ∮ p_D (φ·n) equals the legacy RT0 edge-average
    shortcut exactly (to round-off) -- which pins its SIGN and normal convention to the validated one."""
    from jno.trace import Variable  # noqa: F401
    from jno.utils.solver.fem_nonnodal import _rt_pressure_load_general

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.3)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    vb = v.bind(x=xb, y=yb)
    pD = xb * xb + 0.5 * yb
    fem = jno.fem([inner(ui, vi) - pp * vi.div(), qq * ui.div(), pD * (vb[0] * nx + vb[1] * ny)])
    b_legacy = np.asarray(fem.b).reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["triangle"], dtype=np.int64)
    dm = build_dofmap(cells, "RT", 1, n_verts=pts.shape[0])
    b_gen = _rt_pressure_load_general(np.zeros_like(b_legacy), pD, 0, "boundary", dm, d, pts, cells, fem.offsets, 6)
    ne = fem.offsets[1]
    np.testing.assert_allclose(b_gen[:ne], b_legacy[:ne], atol=1e-13)


# ------------------------------------------------------------------------------------------------
# second-kind Nédélec
# ------------------------------------------------------------------------------------------------


def _n2e_2d_error(k, h):
    d = _domain(2, h)
    u, v, ui, vi, (x, y) = _bound(d, 2, "N2E", k)
    xb, yb, _, nx, ny = d.variable("boundary", normals=True, split=True)
    ub = u.bind(x=xb, y=yb)
    fac = PI**2 + 1.0
    f = [fac * jno.np.sin(PI * y), fac * jno.np.sin(PI * x)]
    fem = jno.fem([inner(ui, vi) + ui.curl() * vi.curl() - (f[0] * vi[0] + f[1] * vi[1]), ub[0] * ny - ub[1] * nx - 0.0])
    sol = _sparse_solve(fem)
    gp, gw = _tri_rule(2 * k + 4)
    val, X, cells = _readback(d, sol, ref_points=gp)
    cur, _, _ = _readback(d, sol, ref_points=gp, derivative="curl")
    pts = np.asarray(d.mesh.points)[:, :2]
    J, _ = cell_jacobians(pts, cells)
    dJ = np.abs(np.linalg.det(J))
    ex = np.stack([np.sin(PI * X[..., 1]), np.sin(PI * X[..., 0])], -1)
    exc = PI * np.cos(PI * X[..., 0]) - PI * np.cos(PI * X[..., 1])
    e0 = np.sqrt(np.einsum("q,c,cq->", gw, dJ, np.sum((val - ex) ** 2, -1)))
    e1 = np.sqrt(np.einsum("q,c,cq->", gw, dJ, (cur - exc) ** 2))
    return float(e0), float(e1)


@pytest.mark.parametrize("k", [1, 2])
def test_n2e_curl_curl_rates(k):
    """N2E_k (full P_k): L² error O(h^{k+1}), curl error O(h^k) (Nédélec 1986; Monk 2003, Thm 8.15)."""
    e = [_n2e_2d_error(k, h) for h in (0.5, 0.25, 0.125)]
    r0 = np.log2(e[1][0] / e[2][0])
    r1 = np.log2(e[1][1] / e[2][1])
    assert r0 > k + 1 - 0.35 and r1 > k - 0.3, f"N2E_{k}: {e}, rates L2 {r0:.2f} curl {r1:.2f}"


# ------------------------------------------------------------------------------------------------
# the A-V pair at degree k: N1E_k x Lagrange P_k
# ------------------------------------------------------------------------------------------------


def test_lagrange_pk_mixed_with_n1e_k_reproduces_exactly():
    """A Lagrange P_2 field on the non-nodal path (mixed with N1E_2) with an inhomogeneous Dirichlet
    BC: a quadratic V and a linear A are reproduced exactly -- P_k tabulation, the P_k DOF map and the
    P_k boundary pins (vertex, edge and face DOFs) all have to be right."""
    d = _domain(3, 0.6)
    u, v, ui, vi, (x, y, z) = _bound(d, 3, "N1E", 2)
    p, q = d.fem_symbols(names=("p", "q"), space="Lagrange", order=2)
    X = dict(x=x, y=y, z=z)
    pp, qq = p.bind(**X), q.bind(**X)
    b = d.variable("boundary", split=True)
    Vex = lambda a, b_, c: a * a + 2 * b_ * c - c + 1.0  # noqa: E731
    f = -(2.0 + 0.0 * x)  # -ΔV
    gp, gq = jno.np.grad(pp, [x, y, z]), jno.np.grad(qq, [x, y, z])
    w = _vec([y + 0.0 * x, z + 0.0 * x, x + 0.0 * x])
    fem = jno.fem(
        [
            inner(ui, vi) - inner(w, vi),
            inner(gp, gq) - f * qq,
            p.bind(x=b[0], y=b[1], z=b[2]) - Vex(b[0], b[1], b[2]),
        ]
    )
    sol = _sparse_solve(fem)
    off = fem.offsets
    topo = d._fem_nonnodal_topology
    pts, cells = np.asarray(d.mesh.points)[:, :3], topo["cells"]
    rp = np.random.default_rng(1).dirichlet(np.ones(4), 5)[:, :3]
    dmL = build_dofmap(cells, "Lagrange", 2, n_verts=pts.shape[0])  # the map the assembler builds (deterministic)
    assert dmL.n_dofs == off[2] - off[1]
    Vh = np.asarray(nonnodal_field_at(pts, cells, dmL, jnp.asarray(sol[off[1] : off[2]]), ref_points=rp))
    val, Xq, _ = _readback(d, sol, ref_points=rp, block=(off[0], off[1]))
    np.testing.assert_allclose(Vh, Vex(Xq[..., 0], Xq[..., 1], Xq[..., 2]), atol=1e-10)
    np.testing.assert_allclose(val, np.stack([Xq[..., 1], Xq[..., 2], Xq[..., 0]], -1), atol=1e-10)
