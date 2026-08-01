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
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
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


# ------------------------------------------------------------------------------------
# The PARAMETRIC (inverse) branch assembles sparsely too.
#
# It used to take the dense global ``jacfwd`` even for N1E, which put a ~10⁴-edge ceiling on 3-D
# vector INVERSE design -- the forward solve was sparse but every optimizer step re-assembled dense.
# These tests pin the three things the port must guarantee: sparse, correct, differentiable.
# ------------------------------------------------------------------------------------
def _n1e_cube_param(mesh_size, name="k"):
    """An N1E curl-curl form whose mass coefficient is a P1 NODAL FIELD parameter -- i.e. ε(x)."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain()
    xi, yi, zi, _ = d.variable("interior", split=True)
    kf, _ = d.fem_symbols()  # a P1 coefficient field, independent of the N1E trial
    k = jno.np.parameter(kf, name=name)
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    fem = jno.fem([inner(cu, cv) - k * inner(ui, vi)])
    return d, fem, (xi, yi, zi), (ui, vi), (cu, cv)


def test_parametric_n1e_operator_is_bcoo_sparse():
    """The PARAMETRIC N1E operator must be a BCOO. A dense ``ndarray`` here means the inverse path is
    still re-assembling O(n²) on every optimizer step -- the real ceiling on 3-D vector inverse design."""
    d, fem, *_ = _n1e_cube_param(0.5)
    n_verts = np.asarray(d.built_mesh.points).shape[0]
    A, _b = fem.operator.evaluate({"k": jnp.ones((n_verts,))})
    assert hasattr(A, "indices") and hasattr(A, "todense"), f"expected a BCOO operator, got {type(A).__name__}"


def test_parametric_sparse_matches_the_non_parametric_assembly():
    """Equivalence: the parametric operator at a CONSTANT field must equal the same form with that
    constant written analytically -- the per-element scatter must reproduce the dense assembly exactly."""
    d, fem, (xi, yi, zi), *_ = _n1e_cube_param(0.4)
    n_verts = np.asarray(d.built_mesh.points).shape[0]
    A_param, _ = fem.operator.evaluate({"k": jnp.full((n_verts,), 2.5)})

    u2, v2 = d.fem_symbols(value_shape=(3,), names=("u2", "v2"), space="N1E")
    ux, vx = u2.bind(x=xi, y=yi, z=zi), v2.bind(x=xi, y=yi, z=zi)
    cux, cvx = u2.vector.curl(xi, yi, zi), v2.vector.curl(xi, yi, zi)
    A_ref = jno.fem([inner(cux, cvx) - 2.5 * inner(ux, vx)]).A
    assert np.max(np.abs(_dense(A_param) - _dense(A_ref))) < 1e-10


def test_parametric_solve_gradient_matches_finite_differences():
    """The inverse-design payoff, end to end: differentiate a complex driven N1E SOLVE (not just the
    operator) w.r.t. a nodal ε(x) field. Central differences must agree -- the per-element scatter has to
    carry the tangent correctly, not merely produce a plausible matrix."""
    d = jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.4).domain()
    xi, yi, zi, _ = d.variable("interior", split=True)
    kf, _ = d.fem_symbols()
    eps = jno.np.parameter(kf, name="eps")
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    fvec = jno.np.vector(1.0 + 0.0 * xi, 0.0 * yi, 0.0 * zi)
    # coercive complex operator (curl-curl + i·ε·mass), real forcing -> a genuinely complex solution
    fem = jno.fem([inner(cu, cv) + 1j * eps * inner(ui, vi) - inner(fvec, vi)])
    n_verts = np.asarray(d.built_mesh.points).shape[0]
    e0 = jnp.full((n_verts,), 1.5)

    # A complex form is FUSED into one real 2n system at assembly (``.operator``); the unfused REAL
    # legs are retained on ``_complex_legs``, and it is those that must each assemble sparsely.
    opr, opi = fem._complex_legs
    assert hasattr(opr.evaluate({"eps": e0})[0], "indices"), "the real leg densified"
    assert hasattr(opi.evaluate({"eps": e0})[0], "indices"), "the imaginary leg densified"

    def loss(ev):
        Ar, br = opr.evaluate({"eps": ev})
        Ai, bi = opi.evaluate({"eps": ev})
        A = jnp.asarray(Ar.todense()) + 1j * jnp.asarray(Ai.todense())
        b = jnp.asarray(br).reshape(-1) + 1j * jnp.asarray(bi).reshape(-1)
        return jnp.sum(jnp.abs(jnp.linalg.solve(A, b)) ** 2)

    j = 3  # perturb one interior node
    g = np.asarray(jax.grad(loss)(e0))
    h = 1e-5
    ep = e0.at[j].add(h)
    em = e0.at[j].add(-h)
    fd = (float(loss(ep)) - float(loss(em))) / (2 * h)
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0.0
    assert abs(g[j] - fd) <= 1e-4 * max(1.0, abs(fd)), f"autodiff {g[j]} vs finite-diff {fd}"


@pytest.mark.slow
def test_parametric_assembly_scales_past_dense_ceiling():
    """Scale regression for the INVERSE path: assemble a parametric N1E operator with ~10⁴ edges. The old
    dense global ``jacfwd`` materialised an ``O(n_edges × n_cells)`` tangent that overflows the 2³¹ XLA
    element limit here, so this was unassemblable -- and it ran once per optimizer step."""
    d, fem, *_ = _n1e_cube_param(0.08)
    n_verts = np.asarray(d.built_mesh.points).shape[0]
    A, _b = fem.operator.evaluate({"k": jnp.ones((n_verts,))})
    assert hasattr(A, "indices"), "parametric assembly densified at scale"
    assert A.shape[0] > 10000, f"mesh too coarse to exercise the scale regime: {A.shape[0]} edges"
    assert bool(jnp.all(jnp.isfinite(A.data)))
