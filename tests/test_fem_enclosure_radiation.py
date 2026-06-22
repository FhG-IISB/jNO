"""Enclosure-radiation view factors — analytic validation (F-correctness gate).

The radiosity in an enclosure-radiation boundary condition is only as good as the geometric view
factor matrix ``F``, so these tests pin the view-factor *kernels* against closed-form results before
any radiosity / FEM coupling is built on top. The canonical analytic case is two **concentric
cylinders** (radii ``r1 < r2``): for infinitely long cylinders the surface-averaged view factors are

    F12 = 1 ,   F21 = r1/r2 ,   F22 = 1 - r1/r2     (the outer cylinder's *concave self-view*)

and the per-point kernel must satisfy **closure** (row sums -> 1 for a closed enclosure) and
**reciprocity** (A_i F_ij = A_j F_ji). The concave self-view ``F22`` is carried entirely by the
visibility matrix (the inner cylinder occludes outer-to-outer rays); the kernel only removes the
self-pair (diagonal). The axisymmetric kernel integrates the diffuse point-to-ring kernel over the
azimuthal angle and must reproduce the same factors for a tall cylinder.

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4 (view factors; concentric
cylinders; bodies of revolution).
"""

from __future__ import annotations

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np
import pytest

from jno.domain.mesh_utils import MeshUtils


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _seg_hits_disk(p, q, radius):
    """True if segment p-q passes strictly inside the disk of the given radius (occlusion test)."""
    d = q - p
    length2 = float(np.dot(d, d))
    if length2 == 0.0:
        return False
    t = float(np.clip(-np.dot(p, d) / length2, 0.0, 1.0))
    closest = p + t * d
    return float(np.hypot(*closest)) < radius - 1e-9


def _concentric_2d(r1, r2, n1, n2):
    """Concentric-circle cross-section: points, visibility (inner cylinder occludes), normals, ds."""
    th1 = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
    th2 = np.linspace(0.0, 2 * np.pi, n2, endpoint=False)
    p_in = np.c_[r1 * np.cos(th1), r1 * np.sin(th1)]
    p_out = np.c_[r2 * np.cos(th2), r2 * np.sin(th2)]
    points = np.vstack([p_in, p_out])
    normals = np.vstack([np.c_[np.cos(th1), np.sin(th1)], -np.c_[np.cos(th2), np.sin(th2)]])  # into the gap
    ds = np.concatenate([np.full(n1, 2 * np.pi * r1 / n1), np.full(n2, 2 * np.pi * r2 / n2)])
    n = n1 + n2
    vm = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j or _seg_hits_disk(points[i], points[j], r1):
                vm[i, j] = 0.0
    return points, vm, normals, ds


def test_view_factor_2d_concentric_cylinders():
    r1, r2, n1, n2 = 0.20, 0.35, 240, 360
    points, vm, normals, ds = _concentric_2d(r1, r2, n1, n2)
    F = np.asarray(
        MeshUtils.get_view_factor_2d(jnp.asarray(points), jnp.asarray(vm), jnp.asarray(normals), jnp.asarray(ds))
    )
    A = ds  # 2-D "area" per node is the arc length
    rows = F.sum(axis=1)

    assert np.allclose(rows, 1.0, atol=5e-3), f"closure violated: row sums {rows.min():.4f}..{rows.max():.4f}"
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "reciprocity A_i F_ij = A_j F_ji violated"

    F12 = (A[:n1, None] * F[:n1, n1:]).sum() / A[:n1].sum()
    F21 = (A[n1:, None] * F[n1:, :n1]).sum() / A[n1:].sum()
    F22 = (A[n1:, None] * F[n1:, n1:]).sum() / A[n1:].sum()
    assert abs(F12 - 1.0) < 5e-3, f"F12 should be 1, got {F12:.4f}"
    assert abs(F21 - r1 / r2) < 5e-3, f"F21 should be r1/r2={r1 / r2:.4f}, got {F21:.4f}"
    assert abs(F22 - (1 - r1 / r2)) < 5e-3, f"concave self-view F22 should be {1 - r1 / r2:.4f}, got {F22:.4f}"


def test_view_factor_axisymmetric_tall_cylinder():
    """Axisymmetric kernel on a tall concentric cylinder reproduces F12->1, F21->r1/r2, with exact
    reciprocity. Locks the azimuthal-integral normalization (a plain mean under-counts by 2*pi)."""
    r1, r2, height, nz = 0.20, 0.35, 6.0, 60
    z = np.linspace(0.0, height, nz)
    points = np.vstack([np.c_[np.full(nz, r1), z], np.c_[np.full(nz, r2), z]])
    normals = np.vstack([np.tile([1.0, 0.0], (nz, 1)), np.tile([-1.0, 0.0], (nz, 1))])
    ds = np.concatenate([np.full(nz, height / nz), np.full(nz, height / nz)])
    vm = np.ones((2 * nz, 2 * nz))  # tall thin gap, mid-ring check ignores end occlusion

    F = np.asarray(
        MeshUtils.get_view_factor_axisymmetric(
            jnp.asarray(points), jnp.asarray(vm), jnp.asarray(normals), jnp.asarray(ds), n_phi=64
        )
    )
    A = 2 * np.pi * points[:, 0] * ds  # ring areas
    assert np.all(np.isfinite(F)) and np.allclose(np.diag(F), 0.0)
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "axisymmetric reciprocity violated"

    mid = nz // 2
    f12_mid = F[mid, nz:].sum()  # a mid inner ring sees essentially the whole outer wall
    F21 = (A[nz:, None] * F[nz:, :nz]).sum() / A[nz:].sum()
    assert abs(f12_mid - 1.0) < 3e-2, f"axisymmetric F12 (mid ring) should be ~1, got {f12_mid:.3f}"
    assert abs(F21 - r1 / r2) < 3e-2, f"axisymmetric F21 should be r1/r2={r1 / r2:.3f}, got {F21:.3f}"


def _ray_crosses_disk(p, q, radius):
    """True if the OPEN segment p-q dips inside the disk (closest approach strictly interior)."""
    d = q - p
    length2 = float(np.dot(d, d))
    if length2 == 0.0:
        return False
    t = -float(np.dot(p, d)) / length2  # unclamped: a genuine crossing needs 0 < t < 1
    if t <= 1e-9 or t >= 1.0 - 1e-9:
        return False
    closest = p + t * d
    return float(np.hypot(*closest)) < radius - 1e-9


def _concentric_2d_elements(r1, r2, n1, n2):
    """Concentric circles as boundary *elements* (edges): endpoints, element normals, and element-level
    visibility (midpoint occlusion by the inner cylinder)."""
    th1 = np.linspace(0.0, 2 * np.pi, n1, endpoint=False)
    th2 = np.linspace(0.0, 2 * np.pi, n2, endpoint=False)
    v_in = np.c_[r1 * np.cos(th1), r1 * np.sin(th1)]
    v_out = np.c_[r2 * np.cos(th2), r2 * np.sin(th2)]
    e0 = np.vstack([v_in, v_out])
    e1 = np.vstack([np.roll(v_in, -1, axis=0), np.roll(v_out, -1, axis=0)])
    mids = 0.5 * (e0 + e1)
    nrm = mids / np.linalg.norm(mids, axis=1, keepdims=True)
    nrm[n1:] *= -1.0  # inner outward, outer inward -> both point into the gap
    m = n1 + n2
    vm = np.ones((m, m))
    for i in range(m):
        for j in range(m):
            if i == j or _ray_crosses_disk(mids[i], mids[j], r1):
                vm[i, j] = 0.0
    return e0, e1, nrm, vm


def test_view_factor_2d_element_concentric_cylinders():
    """Element-based double-area kernel reproduces the analytic concentric-cylinder factors (closure,
    reciprocity, F12=1, F21=r1/r2, concave self-view F22=1-r1/r2). Occlusion is element-level
    (midpoint) and converges with resolution; the double-area quadrature sharpens the near field."""
    r1, r2, n1, n2 = 0.20, 0.35, 240, 360
    e0, e1, nrm, vm = _concentric_2d_elements(r1, r2, n1, n2)
    A = np.linalg.norm(e1 - e0, axis=1)  # element lengths
    F = np.asarray(
        MeshUtils.get_view_factor_2d_element(jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(nrm), jnp.asarray(vm), n_quad=3)
    )
    rows = F.sum(axis=1)
    F12 = (A[:n1, None] * F[:n1, n1:]).sum() / A[:n1].sum()
    F21 = (A[n1:, None] * F[n1:, :n1]).sum() / A[n1:].sum()
    F22 = (A[n1:, None] * F[n1:, n1:]).sum() / A[n1:].sum()

    assert np.allclose(rows, 1.0, atol=5e-3), f"closure: row sums {rows.min():.4f}..{rows.max():.4f}"
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "element reciprocity violated"
    assert abs(F12 - 1.0) < 5e-3, f"F12 should be 1, got {F12:.4f}"
    assert abs(F21 - r1 / r2) < 5e-3, f"F21 should be r1/r2={r1 / r2:.4f}, got {F21:.4f}"
    assert abs(F22 - (1 - r1 / r2)) < 5e-3, f"concave self-view F22 should be {1 - r1 / r2:.4f}, got {F22:.4f}"


def test_view_factor_axisymmetric_element_tall_cylinder():
    """Element-based axisymmetric kernel (meridional element quadrature x azimuthal integration) on a
    tall concentric cylinder: a mid inner element sees ~the whole outer wall (F12->1), F21->r1/r2, with
    exact reciprocity (A_i F_ij = A_j F_ji, A = ring area)."""
    r1, r2, height, nz = 0.20, 0.35, 6.0, 60
    z = np.linspace(0.0, height, nz + 1)
    z0, z1 = z[:-1], z[1:]
    e0 = np.vstack([np.c_[np.full(nz, r1), z0], np.c_[np.full(nz, r2), z0]])
    e1 = np.vstack([np.c_[np.full(nz, r1), z1], np.c_[np.full(nz, r2), z1]])
    nrm = np.vstack([np.tile([1.0, 0.0], (nz, 1)), np.tile([-1.0, 0.0], (nz, 1))])
    vm = np.ones((2 * nz, 2 * nz))  # tall thin gap; mid-element check ignores end occlusion

    F = np.asarray(
        MeshUtils.get_view_factor_axisymmetric_element(
            jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(nrm), jnp.asarray(vm), n_quad=3, n_phi=64
        )
    )
    rmid = 0.5 * (e0[:, 0] + e1[:, 0])
    length = np.linalg.norm(e1 - e0, axis=1)
    A = 2 * np.pi * rmid * length  # ring areas
    assert np.all(np.isfinite(F)) and np.allclose(np.diag(F), 0.0)
    assert np.abs(A[:, None] * F - (A[:, None] * F).T).max() < 1e-12, "axisymmetric element reciprocity violated"

    mid = nz // 2
    f12_mid = F[mid, nz:].sum()
    F21 = (A[nz:, None] * F[nz:, :nz]).sum() / A[nz:].sum()
    assert abs(f12_mid - 1.0) < 3e-2, f"axisymmetric element F12 (mid) should be ~1, got {f12_mid:.3f}"
    assert abs(F21 - r1 / r2) < 3e-2, f"axisymmetric element F21 should be r1/r2={r1 / r2:.3f}, got {F21:.3f}"
