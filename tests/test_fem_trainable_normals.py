"""Differentiable boundary-facet normals (Feature 3) — ``fem_native._face_normals_jax`` and its use in
the surface-assembly path, so a moving boundary node re-evaluates its normal *and* stays differentiable.

The host-numpy ``compute_face_normals`` freezes the mesh out of the autodiff trace; the JAX companion
recomputes the outward unit normal from the (traced) facet vertices with a frozen orientation sign (locally
constant away from tangling). Oracles:
  * **numpy match** — the JAX normal reproduces ``compute_face_normals`` at the initial mesh (2D + 3D) and is
    unit-norm;
  * **FD vs autodiff** — a *normal-dependent* surface functional is differentiable in a moving boundary
    coordinate (the end-to-end proof that surface terms flow gradients under coordinate motion).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.fem_facets import build_facet_connectivity, compute_face_normals
from jno.utils.solver.fem_native import _face_normals_jax


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _frozen_sign(pts, conn, normals_np, dim):
    fv = np.asarray(pts)[conn.face_nodes]
    if dim == 2:
        t = fv[:, 1] - fv[:, 0]
        nraw = np.stack([t[:, 1], -t[:, 0]], axis=1)
    else:
        nraw = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    return np.where(np.sum(nraw * np.asarray(normals_np), axis=1) >= 0, 1.0, -1.0)


def test_face_normals_jax_matches_numpy_2d():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    pts = np.asarray(d.mesh.points)[:, :2]  # the assembler builds geometry from dim-sliced points
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    conn = build_facet_connectivity(cells, "triangle")
    nnp = compute_face_normals(pts, conn, cells, "triangle")
    sign = _frozen_sign(pts, conn, nnp, 2)
    njax = np.asarray(_face_normals_jax(jnp.asarray(pts), jnp.asarray(conn.face_nodes), jnp.asarray(sign)))
    assert np.allclose(njax, nnp, atol=1e-12), "JAX normals disagree with compute_face_normals (2D)"
    assert np.allclose(np.linalg.norm(njax, axis=1), 1.0, atol=1e-12), "JAX normals not unit-norm (2D)"


def test_face_normals_jax_matches_numpy_3d():
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.5).domain()
    pts = np.asarray(d.mesh.points)[:, :3]
    cells = np.asarray(d.mesh.cells_dict["tetra"])
    conn = build_facet_connectivity(cells, "tetrahedron")
    nnp = compute_face_normals(pts, conn, cells, "tetrahedron")
    sign = _frozen_sign(pts, conn, nnp, 3)
    njax = np.asarray(_face_normals_jax(jnp.asarray(pts), jnp.asarray(conn.face_nodes), jnp.asarray(sign)))
    assert np.allclose(njax, nnp, atol=1e-12), "JAX normals disagree with compute_face_normals (3D)"
    assert np.allclose(np.linalg.norm(njax, axis=1), 1.0, atol=1e-12), "JAX normals not unit-norm (3D)"


def test_normal_is_differentiable_in_vertices():
    """The JAX normal has a finite, nonzero derivative w.r.t. the facet vertices (numpy normals do not)."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    pts = jnp.asarray(np.asarray(d.mesh.points)[:, :2])
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    conn = build_facet_connectivity(cells, "triangle")
    nnp = compute_face_normals(np.asarray(pts), conn, cells, "triangle")
    sign = jnp.asarray(_frozen_sign(np.asarray(pts), conn, nnp, 2))
    fn = jnp.asarray(conn.face_nodes)
    jac = jax.jacfwd(lambda P: _face_normals_jax(P, fn, sign))(pts)
    assert np.all(np.isfinite(np.asarray(jac))), "normal Jacobian w.r.t. vertices not finite"
    assert np.linalg.norm(np.asarray(jac)) > 1e-6, "normal does not depend on vertex positions"


def test_surface_normal_functional_gradient_matches_fd():
    """End-to-end: a normal-dependent Neumann flux makes a surface functional differentiable in a moving
    boundary coordinate, matching finite differences (the JAX normals flow gradients through the solve)."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    u, phi = d.fem_symbols()
    xin, yin, _ = d.variable("interior", split=True)
    xb, yb, _tb, nxb, nyb = d.variable("boundary", split=True, normals=True)
    Xp = d.variable("boundary", split=True)[0].trainable(name="cb")  # trainable boundary x-coordinates
    del Xp
    spec = d._trainable_coords[0]
    ids, axis, name = spec["ids"], spec["axis"], spec["name"]
    ui, vi = u.bind(x=xin, y=yin), phi.bind(x=xin, y=yin)
    vb = phi.bind(x=xb, y=yb)
    # Helmholtz-like (well-posed under pure Neumann via +u·v) with a NORMAL-dependent flux (nx+ny)·v.
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui * vi - 1.0 * vi, -(nxb + nyb) * vb], quad_degree=3)
    op = fem.operator
    X0 = jnp.asarray(np.asarray(d.mesh.points)[ids, axis])

    def Jf(X):
        A, b = op.evaluate({name: X})
        uu = jnp.linalg.solve(jnp.asarray(A.todense()), jnp.asarray(b).reshape(-1))
        return 0.5 * jnp.sum(uu * uu)

    g_ad = np.asarray(jax.grad(Jf)(X0))
    eps = 1e-6
    g_fd = np.array([(float(Jf(X0.at[i].add(eps))) - float(Jf(X0.at[i].add(-eps)))) / (2 * eps) for i in range(len(X0))])
    assert np.linalg.norm(g_ad) > 1e-8, "surface functional gradient is zero — normals not differentiable"
    rel = np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd)
    assert rel < 1e-6, f"surface ∂J/∂X autodiff vs FD rel err {rel:.2e}"
