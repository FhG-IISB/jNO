"""Sparse per-element assembly on the non-nodal (RT/N1E/P0) path.

The steady-linear matrix for edge/cell-DOF families is assembled one element at a time — ``jacfwd``
of each cell's element residual w.r.t. its LOCAL dofs, scattered into a BCOO — instead of a single
global ``jacfwd(full_residual)`` that materialises an ``O(n_edges × n_cells)`` tangent and overflows
the 2³¹ XLA element limit past ~10⁴ edges. This mirrors the native (Lagrange) assembler
``fem_native._make_jacobian``.

These tests pin both properties the refactor must guarantee:
  1. the assembled operator is genuinely sparse (BCOO), and entry-for-entry correct (symmetric-PD
     coercive H(curl) operator, and the exact curl-curl bilinear value);
  2. a problem far past the old dense-``jacfwd`` ceiling assembles and solves (scale regression).
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


def _n1e_cube(mesh_size):
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    return d, (xi, yi, zi), (ui, vi), (cu, cv)


def test_steady_n1e_operator_is_bcoo_sparse():
    """The steady-linear N1E matrix must be a BCOO — proof the per-element sparse path (not the dense
    global ``jacfwd``) is taken. A dense ``ndarray`` here would mean the O(n²) path is still live."""
    _, _, (ui, vi), (cu, cv) = _n1e_cube(0.5)
    fem = jno.fem([inner(cu, cv) + inner(ui, vi)])
    A_raw = fem.operator[0]  # raw assembled operator (fem.A densifies for convenience; .operator does not)
    assert hasattr(A_raw, "indices") and hasattr(A_raw, "todense"), f"expected a BCOO operator, got {type(A_raw).__name__}"


def test_sparse_assembly_is_symmetric_pd():
    """Entry-for-entry correctness: the coercive H(curl) operator ``∫ u·v + ∫ curl u·curl v`` assembled
    per-element is symmetric positive-definite, exactly as the (previously dense-assembled) form was."""
    _, _, (ui, vi), (cu, cv) = _n1e_cube(0.5)
    A = _dense(jno.fem([inner(ui, vi) + inner(cu, cv)]).A)
    np.testing.assert_allclose(A, A.T, atol=1e-12)
    assert float(np.linalg.eigvalsh(A).min()) > 0.0


def test_sparse_curl_curl_exact_bilinear():
    """Orientation + assembly together: projecting a field with constant ``curl u* = (0,0,1)`` and
    evaluating the pure curl-curl form ``uᵀ K u`` reproduces ``∫|curl u*|² = vol(unit cube) = 1`` — now
    through the sparse scatter (global tet-edge signs must survive the per-element assembly)."""
    _, (xi, yi, zi), (ui, vi), (cu, cv) = _n1e_cube(0.5)
    M = _dense(jno.fem([inner(ui, vi)]).A)
    b = np.asarray(jnp.asarray(jno.fem([inner(ui, vi) - (-0.5 * yi * vi[0] + 0.5 * xi * vi[1] + 0.0 * vi[2])]).b))
    u_dof = np.linalg.solve(M, b.reshape(-1))
    K = _dense(jno.fem([inner(cu, cv)]).A)
    np.testing.assert_allclose(float(u_dof @ K @ u_dof), 1.0, atol=1e-9)


@pytest.mark.slow
def test_sparse_assembly_scales_past_dense_ceiling():
    """Scale regression: assemble+solve a complex N1E system with ~2×10⁴ edges — well past the old dense
    ``jacfwd`` ceiling (its ``O(n_edges × n_cells)`` tangent overflows the 2³¹ element limit near ~10⁴
    edges). Must return a finite, genuinely complex field. This is the acceptance test for the refactor."""
    d, (xi, yi, zi), (ui, vi), (cu, cv) = _n1e_cube(0.08)
    # complex coercive eddy-like operator: curl-curl + i·mass, forced by a real source (so the solution is
    # genuinely complex). Coercive → no BC needed to be nonsingular. (.A raises for a complex form — it is
    # stored as two real legs — so read the size from the solution.)
    fvec = jno.np.vector(1.0 + 0.0 * xi, 0.0 * yi, 0.0 * zi)
    fem = jno.fem([inner(cu, cv) + 1j * inner(ui, vi) - inner(fvec, vi)])
    sol = np.asarray(jnp.asarray(fem.solve())).reshape(-1)
    # ~1.2×10⁴ edges over ~10⁴ cells: the old dense-jacfwd tangent (n_edges × n_cells × 24 ≈ 3×10⁹ elements)
    # exceeds the 2³¹ XLA limit, so this problem was unassemblable before the per-element refactor.
    assert sol.size > 10000, f"mesh too coarse to exercise the scale regime: {sol.size} edges"
    assert np.all(np.isfinite(sol))
    assert np.iscomplexobj(sol) and np.max(np.abs(sol.imag)) > 0.0
