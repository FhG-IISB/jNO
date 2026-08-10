"""Moving-mesh primitives — `move_mesh` (ALE vertex motion) + `harmonic_extension` (Laplacian mesh
smoothing). These are the free-boundary companions of `transfer_solution`: when the *boundary* of the
domain moves, the mesh deforms to follow it (keeping connectivity, so the field rides along — no
re-interpolation), and the interior is carried by harmonically extending the boundary motion.

Oracles used here:
  * **affine reproduction** — an affine boundary displacement (uniform expansion/translation/shear) is
    harmonic, so `harmonic_extension` reproduces it in the interior to machine precision (2-D & 3-D);
  * **maximum principle** — a non-affine boundary motion extends to interior values bounded by the
    boundary range (a genuine property of the harmonic solve), and must not tangle the mesh;
  * **exact scaling law** — uniformly scaling a square by L (via a move) scales the `-Δu=1, u=0`
    solution by exactly L² (the 2-D stiffness is scale-invariant; the load scales with area), so the
    moved domain is a *correct* FEM domain;
and the extremes: tangling detection, shape guards, in-place vs copy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.fem_adapt import harmonic_extension, move_mesh


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # float64 for the machine-precision exactness asserts
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _rect(size, x0=0.0, y0=0.0, x1=1.0, y1=1.0):
    return jno.Shape.rect(x0, y0, x1, y1, size=size).domain()


def _box(size):
    return jno.Shape.box(0, 0, 0, 1, 1, 1, size=size).domain()


def _verts(d):
    dim = int(d.dimension)
    return np.asarray(d.mesh.points)[:, :dim]


# ── affine reproduction: harmonic_extension is exact for a linear boundary field ──────────────
def test_harmonic_extension_reproduces_affine_2d():
    """d = A x + b on the boundary ⇒ interior is exactly A x + b (an affine field is harmonic)."""
    d = _rect(0.09)
    x = _verts(d)
    rng = np.random.default_rng(1)
    A, b = rng.standard_normal((2, 2)), rng.standard_normal(2)
    field = x @ A.T + b  # (n, 2) affine at every vertex; only boundary rows are read
    disp = harmonic_extension(d, field)
    assert np.max(np.abs(disp - field)) < 1e-9, "harmonic extension of an affine field is not exact"


def test_harmonic_extension_reproduces_affine_3d():
    d = _box(0.28)
    x = _verts(d)
    rng = np.random.default_rng(2)
    A, b = rng.standard_normal((3, 3)), rng.standard_normal(3)
    field = x @ A.T + b
    disp = harmonic_extension(d, field)
    assert np.max(np.abs(disp - field)) < 1e-7


def test_harmonic_extension_obeys_maximum_principle_nonaffine():
    """Push only the top edge up by δ, hold the rest fixed (a non-affine boundary motion). The harmonic
    interior displacement must stay within the boundary range [0, δ] (max principle) — a real oracle for
    the smoothing — and must not invert any element."""
    d = _rect(0.07)
    x = _verts(d)
    delta = 0.2
    bd = np.zeros_like(x)
    bd[x[:, 1] > 1.0 - 1e-6, 1] = delta  # top-edge vertices move up; everything else pinned
    disp = harmonic_extension(d, bd)
    assert disp[:, 1].min() > -1e-9 and disp[:, 1].max() < delta + 1e-9, "interior escaped the boundary range"
    assert np.max(np.abs(disp[:, 0])) < 1e-9, "no horizontal boundary motion ⇒ none in the interior"
    moved = move_mesh(d, disp)  # must not raise (no tangling for a modest push)
    assert np.asarray(moved.mesh.points)[:, 1].max() > 1.0 + 0.5 * delta  # the domain actually grew upward


# ── move_mesh: uniform scaling + validity ─────────────────────────────────────────────────────
def test_move_mesh_uniform_scaling_2d():
    d = _rect(0.1)
    x = _verts(d)
    alpha = 0.4
    moved = move_mesh(d, alpha * x)  # d(x)=αx ⇒ new = (1+α)x, a pure expansion
    new = np.asarray(moved.mesh.points)[:, :2]
    assert np.max(np.abs(new - (1.0 + alpha) * x)) < 1e-12
    assert moved is not d  # copy=True by default


def test_move_mesh_in_place():
    d = _rect(0.12)
    x = _verts(d).copy()
    out = move_mesh(d, 0.1 * x, copy=False)
    assert out is d
    assert np.max(np.abs(np.asarray(d.mesh.points)[:, :2] - 1.1 * x)) < 1e-12


def test_move_mesh_detects_tangling():
    """Mirroring every vertex across the vertical centreline reverses orientation ⇒ every cell inverts —
    must fail loud, not solve garbage. (A *point* reflection is a 180° rotation in 2-D and preserves
    orientation, so a genuine axis mirror is needed to invert.)"""
    d = _rect(0.15)
    x = _verts(d)
    cx = x[:, 0].mean()
    disp = np.zeros_like(x)
    disp[:, 0] = 2.0 * (cx - x[:, 0])  # new_x = 2·cx - x, new_y = y : mirror across x=cx ⇒ orientation flips
    with pytest.raises(ValueError, match="invert|collapse|tangle"):
        move_mesh(d, disp)


def test_move_mesh_shape_guard():
    d = _rect(0.15)
    with pytest.raises(ValueError, match=r"\(n_vert, dim\)"):
        move_mesh(d, np.zeros((len(_verts(d)) + 2, 2)))


def test_harmonic_extension_shape_guard():
    d = _rect(0.15)
    with pytest.raises(ValueError, match=r"\(n_vert, dim\)"):
        harmonic_extension(d, np.zeros((len(_verts(d)), 3)))


# ── the moved domain is a CORRECT FEM domain (exact scaling-law oracle) ───────────────────────
def _poisson_unit_source(d):
    """Solve -Δu = 1 with u = 0 on the whole boundary; return the nodal solution."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    weak = ui.x * vi.x + ui.y * vi.y - 1.0 * vi  # ∫∇u·∇v - ∫1·v = 0
    return np.asarray(jno.fem([weak, u(xb, yb) - 0.0]).solve())


def test_moved_domain_solves_correctly_scaling_law():
    """Move the unit square to a side-2 square by an affine (harmonic) expansion, solve -Δu=1 on both.
    In 2-D the stiffness is scale-invariant and the load scales with area, so the discrete solution must
    scale by *exactly* L²=4 at the (topologically identical) moved vertices — proving the moved mesh is a
    valid, correctly-tagged FEM domain."""
    d1 = _rect(0.1)
    x1 = _verts(d1)
    u1 = _poisson_unit_source(d1)

    disp = harmonic_extension(d1, x1)  # d(x)=x is affine ⇒ interior = x ⇒ new = 2x (uniform ×2 scale)
    d2 = move_mesh(d1, disp)
    assert np.max(np.abs(np.asarray(d2.mesh.points)[:, :2] - 2.0 * x1)) < 1e-10
    u2 = _poisson_unit_source(d2)  # same topology, side doubled ⇒ u2 == 4·u1 (exact)

    denom = max(np.max(np.abs(u1)), 1e-30)
    assert np.max(np.abs(u2 - 4.0 * u1)) / denom < 1e-8, "moved-domain solve broke the L² scaling law"


def test_move_mesh_field_rides_along_no_transfer_needed():
    """Connectivity-preserving move ⇒ a nodal field stays valid on the same (moved) vertices with no
    re-interpolation. Sanity: the field array is unchanged in length and the vertices are 1:1."""
    d = _rect(0.12)
    x = _verts(d)
    field = jnp.asarray(np.sin(np.pi * x[:, 0]) * x[:, 1])
    moved = move_mesh(d, 0.05 * x)
    assert np.asarray(moved.mesh.points).shape[0] == field.shape[0]  # same vertex count ⇒ field still aligned
    assert np.max(np.abs(np.asarray(moved.mesh.points)[:, :2] - 1.05 * x)) < 1e-12


def test_p1_stiffness_matvec_matches_the_energy_gradient_oracle():
    """``_p1_stiffness_jax`` is checked against a route that shares no code with it.

    ``_dirichlet_energy_jax`` computes ``Σ|∇u|²·vol = uᵀKu`` from the element gradients directly, so
    ``½·∂E/∂u`` is the same matvec by a completely different path (reverse-mode over the energy, rather
    than an assembled element block). Also asserts the two structural properties a Laplacian must have:
    symmetry, and a constant null space (rows sum to zero).
    """
    import jax
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _dirichlet_energy_jax, _mesh_cells, _p1_stiffness_jax

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    cells, dim = _mesh_cells(d)
    pts = jnp.asarray(np.asarray(d.mesh.points)[:, :dim])
    matvec, diag = _p1_stiffness_jax(pts, cells, dim)

    u = jnp.asarray(np.random.default_rng(0).standard_normal(pts.shape[0]))
    oracle = 0.5 * jax.grad(lambda uu: _dirichlet_energy_jax(pts, uu, cells, dim))(u)
    assert float(jnp.abs(matvec(u) - oracle).max()) < 1e-12, "matvec disagrees with the energy-gradient oracle"

    K = jnp.stack([matvec(jnp.eye(pts.shape[0])[i]) for i in range(pts.shape[0])])
    assert jnp.allclose(K, K.T, atol=1e-12), "the stiffness must be symmetric"
    assert float(jnp.abs(K.sum(axis=1)).max()) < 1e-12, "a constant field must be in the null space"
    assert float(jnp.abs(jnp.diag(K) - diag).max()) < 1e-14, "the returned diagonal is not K's diagonal"


def test_harmonic_extension_is_differentiable_in_the_prescribed_motion():
    """The reason it is JAX at all: ``scipy.splu`` was a hard break in the gradient chain, so no gradient
    could flow through a moving mesh. Checked against central differences, and asserted **non-zero** —
    a vanishing gradient is the failure mode that a relative-error check alone would pass."""
    import jax
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _harmonic_extension_jax, _mesh_cells

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    cells, dim = _mesh_cells(d)
    pts = jnp.asarray(np.asarray(d.mesh.points)[:, :dim])
    x = np.asarray(pts)
    is_b = (x[:, 0] < 1e-9) | (x[:, 0] > 1 - 1e-9) | (x[:, 1] < 1e-9) | (x[:, 1] > 1 - 1e-9)
    bmask = jnp.asarray(is_b)

    def total(amp):
        given = jnp.zeros_like(pts).at[:, 1].set(amp * bmask.astype(pts.dtype) * pts[:, 0])
        return jnp.sum(_harmonic_extension_jax(pts, cells, dim, given, bmask) ** 2)

    g = float(jax.grad(total)(0.3))
    h = 1e-5
    fd = (total(0.3 + h) - total(0.3 - h)) / (2 * h)
    assert abs(g) > 1e-8, "gradient vanished — the extension is not carrying the prescribed motion"
    assert g == pytest.approx(float(fd), rel=1e-5), f"AD {g:.6e} vs FD {float(fd):.6e}"


def test_harmonic_extension_is_differentiable_in_the_vertex_positions():
    """``d(displacement)/d(mesh)`` — needed for the initial-mesh gradient, and the reason the stiffness is
    rebuilt from ``pts`` in JAX rather than assembled once on the host."""
    import jax
    import jax.numpy as jnp

    from jno.utils.solver.fem_adapt import _harmonic_extension_jax, _mesh_cells

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    cells, dim = _mesh_cells(d)
    x = np.asarray(d.mesh.points)[:, :dim]
    is_b = (x[:, 0] < 1e-9) | (x[:, 0] > 1 - 1e-9) | (x[:, 1] < 1e-9) | (x[:, 1] > 1 - 1e-9)
    bmask = jnp.asarray(is_b)
    given = jnp.zeros((x.shape[0], dim)).at[:, 1].set(0.2 * bmask.astype(jnp.float64) * jnp.asarray(x[:, 0]))

    def total(p):
        return jnp.sum(_harmonic_extension_jax(p, cells, dim, given, bmask) ** 2)

    i = int(np.where(~is_b)[0][0])  # an interior vertex
    g = np.asarray(jax.grad(total)(jnp.asarray(x)))
    h = 1e-6
    up = jnp.asarray(x).at[i, 0].add(h)
    dn = jnp.asarray(x).at[i, 0].add(-h)
    fd = (total(up) - total(dn)) / (2 * h)
    assert abs(g[i, 0]) > 1e-10, "no sensitivity to an interior vertex position"
    assert g[i, 0] == pytest.approx(float(fd), rel=1e-4), f"AD {g[i, 0]:.6e} vs FD {float(fd):.6e}"
