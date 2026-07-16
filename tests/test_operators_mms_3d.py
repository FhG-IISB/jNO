"""3-D Method-of-Manufactured-Solutions for FD differential operators.

The 2-D MMS suite in ``tests/test_operators_mms.py`` only exercises the 2-D
FD paths (``compute_fd_gradient_2d_simple``, etc.). The 3-D code paths
(``compute_fd_gradient_3d_simple``, ``compute_fd_laplacian_3d_simple``,
``compute_fd_hessian_3d_simple`` in ``jno/differential_operators.py``) were
unverified — a stencil regression in any of them would not have been
caught by the existing suite.

This file fixes that gap. We use a unit cube tetrahedral mesh
(``jno.domain.cube`` is the only 3-D geometry jNO currently exposes; the
CSG path is 2-D-only per ``polygon_domain.py``) and a smooth analytic
field whose derivatives are computable in closed form:

    u(x, y, z) = sin(πx) cos(πy) (1 + z²)

    ∇u  = ( π cos(πx) cos(πy) (1 + z²),
           −π sin(πx) sin(πy) (1 + z²),
            2z sin(πx) cos(πy) )

    Δu  = −2π² sin(πx) cos(πy) (1 + z²) + 2 sin(πx) cos(πy)

L² (RMS-relative) norms are used — see ``tests/test_operators_mms.py`` for
the rationale. 3-D tetrahedral meshes have fewer neighbours per node than
2-D triangulations so stencil error is naturally larger; tolerances are
calibrated to ~1.3× the observed L² error at h=0.10.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.differential_operators import DifferentialOperators

# ────────────────────────────────────────────────────────────────────────
# Geometry + helpers
# ────────────────────────────────────────────────────────────────────────


def _build_cube(mesh_size: float = 0.10):
    return jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain(
        compute_mesh_connectivity=True,
    )


def _interior_indices(mc) -> np.ndarray:
    all_idx = np.arange(int(mc["n_points"]))
    bnd = np.asarray(mc["boundary_indices"], dtype=np.int64)
    return np.setdiff1d(all_idx, bnd)


def _rel_l2(computed: jnp.ndarray, analytic: jnp.ndarray) -> float:
    diff = computed - analytic
    return float(jnp.sqrt(jnp.mean(diff**2)) / jnp.sqrt(jnp.mean(analytic**2)))


# ────────────────────────────────────────────────────────────────────────
# Analytic field
# ────────────────────────────────────────────────────────────────────────


def _u(x, y, z):
    return jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * (1.0 + z * z)


def _ux(x, y, z):
    return jnp.pi * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y) * (1.0 + z * z)


def _uy(x, y, z):
    return -jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * (1.0 + z * z)


def _uz(x, y, z):
    return 2.0 * z * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)


def _uxx(x, y, z):
    return -(jnp.pi**2) * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * (1.0 + z * z)


def _uyy(x, y, z):
    return _uxx(x, y, z)


def _uzz(x, y, z):
    return 2.0 * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)


def _uxy(x, y, z):
    return -(jnp.pi**2) * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * (1.0 + z * z)


def _uxz(x, y, z):
    return 2.0 * z * jnp.pi * jnp.cos(jnp.pi * x) * jnp.cos(jnp.pi * y)


def _uyz(x, y, z):
    return -2.0 * z * jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def _laplacian(x, y, z):
    return _uxx(x, y, z) + _uyy(x, y, z) + _uzz(x, y, z)


# ────────────────────────────────────────────────────────────────────────
# Shared mesh + field evaluation — used by all tests below.
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cube_mesh():
    dom = _build_cube(mesh_size=0.10)
    mc = dom.mesh_connectivity
    points = jnp.asarray(mc["points"])
    tetrahedra = jnp.asarray(mc["tetrahedra"])
    interior = _interior_indices(mc)
    assert interior.size > 0, "3-D cube mesh has no interior nodes"
    return points, tetrahedra, interior


# ────────────────────────────────────────────────────────────────────────
# 1. Gradient — area-weighted on tetrahedral cube
# ────────────────────────────────────────────────────────────────────────


class TestGradientMMS3D:
    """Observed L² rel errors at h=0.10: ∂x 5.7%, ∂y 5.2%, ∂z 7.8%.
    Threshold 10% gives ~1.3× headroom on the worst component (∂z, which
    has fewer aligned-with-edge neighbours on the unit-cube mesh)."""

    def test_du_dx(self, cube_mesh):
        points, tetrahedra, interior = cube_mesh
        u = _u(points[:, 0], points[:, 1], points[:, 2])
        d = DifferentialOperators.compute_fd_gradient_3d_simple(u, points, tetrahedra, dim=0, method="area_weighted")
        analytic = _ux(points[:, 0], points[:, 1], points[:, 2])
        rel = _rel_l2(d[interior], analytic[interior])
        assert rel < 0.10, f"∂u/∂x L² rel err {rel * 100:.2f}% > 10%"

    def test_du_dy(self, cube_mesh):
        points, tetrahedra, interior = cube_mesh
        u = _u(points[:, 0], points[:, 1], points[:, 2])
        d = DifferentialOperators.compute_fd_gradient_3d_simple(u, points, tetrahedra, dim=1, method="area_weighted")
        analytic = _uy(points[:, 0], points[:, 1], points[:, 2])
        rel = _rel_l2(d[interior], analytic[interior])
        assert rel < 0.10, f"∂u/∂y L² rel err {rel * 100:.2f}% > 10%"

    def test_du_dz(self, cube_mesh):
        points, tetrahedra, interior = cube_mesh
        u = _u(points[:, 0], points[:, 1], points[:, 2])
        d = DifferentialOperators.compute_fd_gradient_3d_simple(u, points, tetrahedra, dim=2, method="area_weighted")
        analytic = _uz(points[:, 0], points[:, 1], points[:, 2])
        rel = _rel_l2(d[interior], analytic[interior])
        assert rel < 0.10, f"∂u/∂z L² rel err {rel * 100:.2f}% > 10%"


class TestGradientSubSchemesMMS3D:
    """3-D gradient via uniform / inverse_distance / least_squares.

    Observed L² rel errs at h=0.10 (worst dim, ∂z):
      - uniform           ≈ 8.0%
      - inverse_distance  ≈ 7.9%
      - least_squares     ≈ 19.7%

    Threshold per-method gives ~1.3× headroom. The 2-D analogues are
    tested in ``tests/test_operators_mms.py::TestGradientSubSchemesMMS2D``;
    this is the symmetric 3-D coverage.
    """

    @pytest.mark.parametrize(
        "method,tol",
        [
            ("uniform", 0.12),
            ("inverse_distance", 0.12),
            ("least_squares", 0.25),
        ],
    )
    @pytest.mark.parametrize("dim,name,fn", [(0, "x", _ux), (1, "y", _uy), (2, "z", _uz)])
    def test_3d_gradient_sub_scheme(self, cube_mesh, method, tol, dim, name, fn):
        points, tetrahedra, interior = cube_mesh
        u = _u(points[:, 0], points[:, 1], points[:, 2])
        d = DifferentialOperators.compute_fd_gradient_3d_simple(u, points, tetrahedra, dim=dim, method=method)
        analytic = fn(points[:, 0], points[:, 1], points[:, 2])
        rel = _rel_l2(d[interior], analytic[interior])
        assert rel < tol, f"3-D {method} ∂u/∂{name}: L² rel err {rel * 100:.2f}% > {tol * 100:.1f}%"


class TestLaplacianSubSchemesMMS3D:
    """3-D Laplacian via ``lsq_of_gradient``.

    Observed L² rel err at h=0.10 ≈ 34%; threshold 45% gives ~1.3× headroom.
    Considerably noisier than the gradient-of-gradient 3-D Laplacian
    (16.9% on the same mesh) — the LSQ stencil weights are less regular
    on tetrahedral meshes near boundary faces.
    """

    def test_lsq_of_gradient_3d(self, cube_mesh):
        points, tetrahedra, interior = cube_mesh
        u = _u(points[:, 0], points[:, 1], points[:, 2])
        lap = DifferentialOperators.compute_fd_laplacian_3d_simple(
            u, points, tetrahedra, dims=(0, 1, 2), method="lsq_of_gradient"
        )
        analytic = _laplacian(points[:, 0], points[:, 1], points[:, 2])
        rel = _rel_l2(lap[interior], analytic[interior])
        assert rel < 0.45, f"3-D lsq_of_gradient Δu L² rel err {rel * 100:.2f}% > 45%"


# ────────────────────────────────────────────────────────────────────────
# 2. Laplacian — gradient-of-gradient
# ────────────────────────────────────────────────────────────────────────


class TestLaplacianMMS3D:
    """Observed L² rel err at h=0.10: 16.9%. Threshold 22% gives ~1.3×
    headroom. Double-FD on tetrahedral meshes is genuinely lossier than
    2-D triangulation due to fewer support neighbours per stencil."""

    def test_laplacian_gradient_of_gradient(self, cube_mesh):
        points, tetrahedra, interior = cube_mesh
        u = _u(points[:, 0], points[:, 1], points[:, 2])
        lap = DifferentialOperators.compute_fd_laplacian_3d_simple(
            u, points, tetrahedra, dims=(0, 1, 2), method="gradient_of_gradient"
        )
        analytic = _laplacian(points[:, 0], points[:, 1], points[:, 2])
        rel = _rel_l2(lap[interior], analytic[interior])
        assert rel < 0.22, f"Δu L² rel err {rel * 100:.2f}% > 22%"


# ────────────────────────────────────────────────────────────────────────
# 3. Hessian — full 3x3 matrix
# ────────────────────────────────────────────────────────────────────────


class TestHessianMMS3D:
    """Observed L² rel errs at h=0.10: xx 9%, yy 25%, zz 21%, xy 9%,
    xz 22%, yz 20%. Threshold 30% gives ~1.2× headroom on yy (worst
    component). FD Hessian symmetry is exact (machine zero) — we assert
    that as a regression guard."""

    @pytest.fixture(scope="class")
    def hessian(self, cube_mesh):
        points, tetrahedra, _ = cube_mesh
        u = _u(points[:, 0], points[:, 1], points[:, 2])
        var_dims = [(i, i, j, j) for i in range(3) for j in range(3)]
        H = DifferentialOperators.compute_fd_hessian_3d_simple(u, points, tetrahedra, var_dims)
        return H

    @pytest.mark.parametrize(
        "i,j,fn,name",
        [
            (0, 0, _uxx, "xx"),
            (1, 1, _uyy, "yy"),
            (2, 2, _uzz, "zz"),
            (0, 1, _uxy, "xy"),
            (0, 2, _uxz, "xz"),
            (1, 2, _uyz, "yz"),
        ],
    )
    def test_hessian_component(self, hessian, cube_mesh, i, j, fn, name):
        points, _, interior = cube_mesh
        analytic = fn(points[:, 0], points[:, 1], points[:, 2])
        rel = _rel_l2(hessian[interior, i, j], analytic[interior])
        assert rel < 0.30, f"Hessian {name}: L² rel err {rel * 100:.2f}% > 30%"

    def test_hessian_is_symmetric(self, hessian, cube_mesh):
        _, _, interior = cube_mesh
        # FD Hessian assembles symmetric stencils → exact equality expected.
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            err = float(jnp.max(jnp.abs(hessian[interior, i, j] - hessian[interior, j, i])))
            assert err < 1e-5, f"Hessian symmetry {i}{j}: max |H[{i},{j}] - H[{j},{i}]| = {err:.3e}"


# ────────────────────────────────────────────────────────────────────────
# 4. Mesh sanity — fixture provides a viable mesh
# ────────────────────────────────────────────────────────────────────────


def test_cube_mesh_has_tetrahedra(cube_mesh):
    _, tetrahedra, interior = cube_mesh
    assert tetrahedra.shape[1] == 4, "Expected (n_tets, 4) tetrahedral connectivity"
    assert tetrahedra.shape[0] > 0, "Cube mesh produced no tetrahedra"
    assert interior.size >= 100, f"Cube mesh interior size {interior.size} too small for MMS"
