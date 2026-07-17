"""Extensive 3D and 3D+time FEM coverage through ``jno.fem``.

These mirror the 2D workflow/vector suites on a tetrahedral unit cube
(``jno.domain.cube`` -> pygmsh -> ``TET4``) and exercise every axis of the
driver in three dimensions:

* steady + transient,
* scalar + vector (``vec=3``) unknowns,
* linear + nonlinear,
* the full BC set: Dirichlet (all-component *and* per-component roller), Neumann
  traction, and Robin, on the six auto-tagged cube faces.

Sharp checks use ``TET4``-exact linear manufactured solutions (recovered to
machine precision, mesh-independent). The genuinely higher-order cases (a bubble
source, a sin-mode heat decay) use mesh-appropriate tolerances.

Cube face -> axis convention (from ``Geometries.cube``):
``left``/``right`` = x=0/1, ``front``/``back`` = y=0/1, ``bottom``/``top`` = z=0/1.

Note: 3D assembly is more memory-hungry than 2D; on a small (8 GB) GPU these may
need ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` (or a CPU run) when the device is
already loaded. The meshes here are deliberately coarse to keep that headroom.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("pygmsh", reason="pygmsh required for cube meshing")

import jax  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """FEM assembly is float64, so these tests opt into x64 per-test. The session default is
    x64-off (see tests/conftest.py); save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def _solve(fem):
    return np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))


def _cube(mesh_size=0.4, **kwargs):
    return jno.Shape.box(0, 0, 0, 1, 1, 1, size=mesh_size).domain(**kwargs)


def _xyz(d):
    co = d.variable("interior", split=True)
    return co[0], co[1], co[2]


def _coords(d):
    return np.asarray(d.mesh.points)[:, :3]


# ==========================================================================
# structure
# ==========================================================================
def test_cube_tags_all_six_faces():
    d = _cube(0.5)
    assert d.dimension == 3
    regions = set(getattr(d, "_boundary_regions", {}))
    assert {"left", "right", "front", "back", "bottom", "top"} <= regions


# ==========================================================================
# steady scalar — linear, all BC kinds, recovered exactly
# ==========================================================================
def test_poisson_dirichlet_recovers_linear_field():
    # Harmonic linear field u = 1 + 2x + 3y + 4z (lap u = 0), prescribed on the
    # whole boundary. A linear field lives in the TET4 space -> recovered exactly.
    d = _cube(0.45)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    g = 1.0 + 2.0 * cb[0] + 3.0 * cb[1] + 4.0 * cb[2]
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, u(cb[0], cb[1], cb[2]) - g], element_type="TET4")
    sol = _solve(fem)
    c = _coords(d)
    exact = 1.0 + 2.0 * c[:, 0] + 3.0 * c[:, 1] + 4.0 * c[:, 2]
    assert np.linalg.norm(exact - sol) / np.linalg.norm(exact) < 1e-9


def test_poisson_neumann_recovers_linear_solution():
    # -lap u = 0 ; u=0 on the left face (x=0) ; du/dn = 1 on the right (x=1) ;
    # natural zero-flux on the other four faces => u = x. Boundary term is -g_N*phi.
    d = _cube(0.4)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols()
    fl = d.variable("left", split=True)
    fr = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    fem = jno.fem(
        [ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, -1.0 * phi.bind(x=fr[0], y=fr[1], z=fr[2]), u(fl[0], fl[1], fl[2]) - 0.0],
        element_type="TET4",
    )
    assert "surface@right" in fem.classification
    sol = _solve(fem)
    c = _coords(d)
    assert np.linalg.norm(c[:, 0] - sol) / np.linalg.norm(c[:, 0]) < 1e-8


def test_poisson_robin_recovers_linear_solution():
    # du/dn + a u = g on the right face, with g = 1 + a so that u = x is exact;
    # u = 0 on the left face.
    a = 2.0
    d = _cube(0.4)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols()
    fl = d.variable("left", split=True)
    fr = d.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    ur, vr = u.bind(x=fr[0], y=fr[1], z=fr[2]), phi.bind(x=fr[0], y=fr[1], z=fr[2])
    robin = (a * ur - (1.0 + a)) * vr
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z, robin, u(fl[0], fl[1], fl[2]) - 0.0], element_type="TET4")
    assert "surface@right" in fem.classification
    sol = _solve(fem)
    c = _coords(d)
    assert np.linalg.norm(c[:, 0] - sol) / np.linalg.norm(c[:, 0]) < 1e-8


# ==========================================================================
# steady scalar — nonlinear (Newton on a manufactured bubble)
# ==========================================================================
def test_nonlinear_reaction_newton_recovers_manufactured():
    spo = pytest.importorskip("scipy.optimize")
    # -lap u + u^3 = f, u_exact = x(1-x)y(1-y)z(1-z), u=0 on boundary,
    # f = -lap(u_exact) + u_exact^3 = 2(gy*gz + gx*gz + gx*gy) + u^3.
    d = _cube(0.34)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    gx, gy, gz = xi * (1 - xi), yi * (1 - yi), zi * (1 - zi)
    g = gx * gy * gz
    f = 2.0 * (gy * gz + gx * gz + gx * gy) + g**3
    fem = jno.fem(
        [ui.x * vi.x + ui.y * vi.y + ui.z * vi.z + (u * u * u) * vi - f * vi, u(cb[0], cb[1], cb[2]) - 0.0],
        element_type="TET4",
    )
    assert not fem.is_linear
    sol = spo.root(
        lambda v: np.asarray(fem.residual(v)),
        np.zeros(fem.dofs),
        jac=lambda v: _dense(fem.jacobian(v)),
        method="hybr",
    )
    assert sol.success
    c = _coords(d)
    exact = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1]) * c[:, 2] * (1 - c[:, 2])
    assert np.linalg.norm(exact - sol.x) / np.linalg.norm(exact) < 1e-1


# ==========================================================================
# transient (3D + time)
# ==========================================================================
def test_transient_heat_decays_to_analytic():
    nu = 1.0
    d = _cube(0.3, time=(0.0, 0.02, 6))
    co = d.variable("interior", split=True)
    xi, yi, zi, ti = co[0], co[1], co[2], co[3]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi, t=ti), phi.bind(x=xi, y=yi, z=zi, t=ti)
    ic = u(ci[0], ci[1], ci[2]) - jno.fn(
        lambda x, y, z: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z), [ci[0], ci[1], ci[2]]
    )
    fem = jno.fem(
        [ui.t * vi + nu * (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z), u(cb[0], cb[1], cb[2]) - 0.0, ic],
        element_type="TET4",
    )
    assert fem.is_transient

    M, A = _dense(fem.M), _dense(fem.operator.A)
    w, dt = np.asarray(fem.state0), float(fem.dt)
    nsteps = round((fem.t1 - fem.t0) / dt)
    for _ in range(nsteps):  # backward Euler: (M + dt A) w_next = M w
        w = np.linalg.solve(M + dt * A, M @ w)

    c = _coords(d)
    mode = np.sin(np.pi * c[:, 0]) * np.sin(np.pi * c[:, 1]) * np.sin(np.pi * c[:, 2])
    analytic = np.exp(-3 * nu * np.pi**2 * fem.t1) * mode
    # coarse TET4 + first-order backward Euler in 3D -> a generous tolerance
    assert np.linalg.norm(analytic - w) / np.linalg.norm(analytic) < 0.2
    # the field must actually decay (and not blow up or flip sign)
    assert 0.0 < np.linalg.norm(w) < np.linalg.norm(np.asarray(fem.state0))


def test_transient_nonlinear_assembles_residual_block():
    # 3D Allen-Cahn-style reaction: u_t*phi + grad.grad + (u^3 - u)*phi.
    d = _cube(0.4, time=(0.0, 0.1, 6))
    xi, yi, zi = _xyz(d)
    co = d.variable("interior", split=True)
    ti = co[3]
    u, phi = d.fem_symbols()
    cb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi, t=ti), phi.bind(x=xi, y=yi, z=zi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z) + (u * u * u - u) * vi,
            u(cb[0], cb[1], cb[2]) - 0.0,
            u(ci[0], ci[1], ci[2]) - 0.0,
        ],
        element_type="TET4",
    )
    assert fem.is_transient and not fem.is_linear
    block = fem.operator
    assert block.residual is not None and block.jacobian is not None and block.mass is not None
    R0 = np.asarray(block.residual(np.asarray(fem.state0), float(fem.t0), None))
    assert R0.shape == (fem.dofs,)


# ==========================================================================
# steady vector (vec=3): elasticity, decoupled Poisson, roller, traction
# ==========================================================================
def test_vec3_inferred_and_symmetric():
    d = _cube(0.45)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols(value_shape=(3,))
    cb = d.variable("boundary", split=True)
    weak = jno.np.inner(jno.np.grad(u, [xi, yi, zi]), jno.np.grad(phi, [xi, yi, zi]), n_contract=2)
    fem = jno.fem([weak, u(cb[0], cb[1], cb[2]) - (0.0, 0.0, 0.0)], element_type="TET4")
    n_nodes = int(np.asarray(d.mesh.points).shape[0])
    assert fem.dofs == 3 * n_nodes  # vec=3 inferred from value_shape
    A = _dense(fem.A)
    assert A.shape == (3 * n_nodes, 3 * n_nodes)
    assert np.allclose(A, A.T, atol=1e-7)


def test_full_elasticity_assembles_symmetric():
    # Full lambda-mu elasticity incl. the volumetric div(u)*div(phi) term, in 3D.
    d = _cube(0.45)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols(value_shape=(3,))
    cb = d.variable("boundary", split=True)
    eps_u, eps_phi = jno.np.symgrad(u, [xi, yi, zi]), jno.np.symgrad(phi, [xi, yi, zi])
    weak = 1.0 * jno.np.trace(eps_u) * jno.np.trace(eps_phi) + 2.0 * jno.np.inner(eps_u, eps_phi, n_contract=2)
    fem = jno.fem([weak, u(cb[0], cb[1], cb[2]) - (0.0, 0.0, 0.0)], element_type="TET4")
    A = _dense(fem.A)
    assert A.shape[0] == A.shape[1] == fem.dofs
    assert np.allclose(A, A.T, atol=1e-6)


def test_roller_per_component_dirichlet_recovers_manufactured():
    # Per-component (roller / symmetry) Dirichlet via u(region)[i] - g, on all six
    # faces. Decoupled vector Laplacian (f=0): pin u_x on left/right, u_y on
    # front/back, u_z on bottom/top; the two orthogonal components on each face are
    # free (natural). Unique solution u = (x, y, z); linear -> TET4 exact. This also
    # confirms the face->axis mapping and that shared edges/corners get every pin.
    d = _cube(0.4)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols(value_shape=(3,))
    fl = d.variable("left", split=True)
    fr = d.variable("right", split=True)
    ff = d.variable("front", split=True)
    fb = d.variable("back", split=True)
    fbo = d.variable("bottom", split=True)
    ft = d.variable("top", split=True)
    weak = jno.np.inner(jno.np.grad(u, [xi, yi, zi]), jno.np.grad(phi, [xi, yi, zi]), n_contract=2)
    fem = jno.fem(
        [
            weak,
            u(fl[0], fl[1], fl[2])[0] - 0.0,
            u(fr[0], fr[1], fr[2])[0] - 1.0,
            u(ff[0], ff[1], ff[2])[1] - 0.0,
            u(fb[0], fb[1], fb[2])[1] - 1.0,
            u(fbo[0], fbo[1], fbo[2])[2] - 0.0,
            u(ft[0], ft[1], ft[2])[2] - 1.0,
        ],
        element_type="TET4",
    )
    assert "dirichlet@left[x]" in fem.classification
    assert "dirichlet@front[y]" in fem.classification
    assert "dirichlet@bottom[z]" in fem.classification
    sol = _solve(fem).reshape(-1, 3)
    c = _coords(d)
    exact = np.stack([c[:, 0], c[:, 1], c[:, 2]], axis=-1)  # u = (x, y, z)
    assert np.linalg.norm(exact - sol) / np.linalg.norm(exact) < 1e-9
    # a freed component genuinely varies (u_y on the left face follows y, not pinned)
    left = np.isclose(c[:, 0], 0.0)
    assert np.linalg.norm(sol[left, 1] - c[left, 1]) < 1e-9
    assert np.ptp(sol[left, 1]) > 0.5


def test_vector_traction_inner_form_recovers_manufactured():
    # 3D vector Neumann traction via the natural form inner(t, phi(region)).
    # Clamp u=(0,0,0) on the left, traction t=(1,0,0) on the right (residual term
    # -t·phi) => decoupled x-component solves to u_x = x while u_y, u_z stay 0.
    d = _cube(0.4)
    xi, yi, zi = _xyz(d)
    u, phi = d.fem_symbols(value_shape=(3,))
    fl = d.variable("left", split=True)
    fr = d.variable("right", split=True)
    t = jnp.array([1.0, 0.0, 0.0])
    weak = jno.np.inner(jno.np.grad(u, [xi, yi, zi]), jno.np.grad(phi, [xi, yi, zi]), n_contract=2)
    traction = -1.0 * jno.np.inner(t, phi.bind(x=fr[0], y=fr[1], z=fr[2]), n_contract=1)
    fem = jno.fem([weak, u(fl[0], fl[1], fl[2]) - (0.0, 0.0, 0.0), traction], element_type="TET4")
    assert "surface@right" in fem.classification
    assert np.linalg.norm(np.asarray(fem.b)) > 0.0
    sol = _solve(fem).reshape(-1, 3)
    c = _coords(d)
    exact = np.stack([c[:, 0], np.zeros_like(c[:, 1]), np.zeros_like(c[:, 2])], axis=-1)  # u = (x, 0, 0)
    assert np.linalg.norm(exact - sol) / np.linalg.norm(exact) < 1e-8
