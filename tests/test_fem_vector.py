"""Vector-valued FEM through ``jno.fem``.

Covers: ``vec`` inferred from the trial's ``value_shape``; a vector elliptic
solve recovers a manufactured solution; full lambda-mu linear elasticity (with
the volumetric ``div(u)*div(phi)`` term enabled by the kernel prefix-alignment
fix) assembles symmetrically. All-component Dirichlet is via the scalar-broadcast
form ``u(region) - 0.0``; per-component (roller/symmetry) Dirichlet is via the
component-indexed form ``u(region)[i] - g``, which pins only component ``i`` and
leaves the others free.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """feax assembly is float64, so these tests opt into x64 per-test. The session default is
    x64-off (see tests/conftest.py); save/restore keeps the flag from leaking to other modules."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def test_vec_inferred_from_value_shape():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2)
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    n_nodes = int(np.asarray(d.mesh.points).shape[0])
    assert fem.dofs == 2 * n_nodes


def test_vector_poisson_recovers_manufactured():
    # Decoupled vector Poisson: -lap(u_k) = f_k, u = 0 on the boundary.
    # u_exact = (g, 0.5 g), g = x(1-x)y(1-y).
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    vi = phi.bind(x=xi, y=yi)
    f1 = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    f2 = 0.5 * f1
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2) - (
        f1 * vi.component(0) + f2 * vi.component(1)
    )
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    assert fem.is_linear and fem.dofs == 2 * int(np.asarray(d.mesh.points).shape[0])
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1)).reshape(-1, 2)
    c = np.asarray(d.mesh.points)[:, :2]
    g = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    exact = np.stack([g, 0.5 * g], axis=-1)
    rel = np.linalg.norm(exact - sol) / np.linalg.norm(exact)
    assert rel < 1e-2


def test_full_elasticity_assembles_symmetric():
    # Full lambda-mu elasticity incl. the volumetric div(u)*div(phi) term.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    eps_u, eps_phi = jno.np.symgrad(u, [xi, yi]), jno.np.symgrad(phi, [xi, yi])
    weak = 1.0 * jno.np.trace(eps_u) * jno.np.trace(eps_phi) + 2.0 * jno.np.inner(eps_u, eps_phi, n_contract=2)
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    A = _dense(fem.A)
    assert A.shape[0] == A.shape[1] == fem.dofs
    assert np.allclose(A, A.T, atol=1e-7)


def test_roller_per_component_dirichlet_recovers_manufactured():
    # Per-component (roller / symmetry) Dirichlet via `u(region)[i] - g`.
    # Decoupled vector Laplacian (f = 0): pin u_x on left/right and u_y on
    # bottom/top; the orthogonal component on each edge is left FREE (natural
    # zero-flux). Unique solution u = (x, y); linear field -> TRI3 exact.
    # This exercises the part the all-component clamp can't: only the named
    # component is constrained, and shared corner nodes get *both* edges' pins.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2)
    fem = jno.fem([weak, u(xl, yl)[0] - 0.0, u(xr, yr)[0] - 1.0, u(xb, yb)[1] - 0.0, u(xt, yt)[1] - 1.0])

    # component-indexed constraints are classified per component (not all-component)
    assert "dirichlet@left[x]" in fem.classification
    assert "dirichlet@bottom[y]" in fem.classification

    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1)).reshape(-1, 2)
    c = np.asarray(d.mesh.points)[:, :2]
    exact = np.stack([c[:, 0], c[:, 1]], axis=-1)
    assert np.linalg.norm(exact - sol) / np.linalg.norm(exact) < 1e-9

    # "u_x = 0, u_y free" on the left edge: the pinned component is exact, while
    # the freed component follows the solution (u_y = y) rather than being clamped.
    left = np.isclose(c[:, 0], 0.0)
    assert np.allclose(sol[left, 0], 0.0, atol=1e-12)  # u_x pinned to 0
    assert np.linalg.norm(sol[left, 1] - c[left, 1]) < 1e-9  # u_y free -> equals y
    assert np.ptp(sol[left, 1]) > 0.5  # and genuinely varies (not silently pinned)


def test_roller_mixed_components_across_boundaries():
    # User scenario: u_x = 0 on one boundary (u_y free) and u_y = 0 on another
    # (u_x free). A diagonal body load f = (1, 1) drives both components; with a
    # roller on left (u_x) and bottom (u_y) the problem is well posed (each
    # decoupled component has a Dirichlet patch). Check the pinned components are
    # exact and the freed components are non-trivial.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    vi = phi.bind(x=xi, y=yi)
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2) - (
        1.0 * vi.component(0) + 1.0 * vi.component(1)
    )
    fem = jno.fem([weak, u(xl, yl)[0] - 0.0, u(xb, yb)[1] - 0.0])
    assert "dirichlet@left[x]" in fem.classification
    assert "dirichlet@bottom[y]" in fem.classification

    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1)).reshape(-1, 2)
    c = np.asarray(d.mesh.points)[:, :2]
    left = np.isclose(c[:, 0], 0.0)
    bottom = np.isclose(c[:, 1], 0.0)
    assert np.allclose(sol[left, 0], 0.0, atol=1e-10)  # u_x pinned on left
    assert np.allclose(sol[bottom, 1], 0.0, atol=1e-10)  # u_y pinned on bottom
    # freed components are not clamped: u_y on left and u_x on bottom are nonzero
    assert np.max(np.abs(sol[left, 1])) > 1e-3
    assert np.max(np.abs(sol[bottom, 0])) > 1e-3


def test_vector_neumann_traction_inner_form_recovers_manufactured():
    # Vector Neumann traction written the natural way: inner(t, phi(region)).
    # Vector Laplacian, clamp u=(0,0) on the left, traction t=(1,0) on the right.
    # Residual convention matches scalar Neumann (boundary term is -t·phi), so
    # du_x/dn = 1 and the decoupled x-component solves to u_x = x while u_y stays
    # 0. Linear field -> TRI3 exact. This is the form that previously misclassified
    # as a volume term (inner stripped the bound view's region); it must now ride
    # the surface path.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    t = jnp.array([1.0, 0.0])
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2)
    traction = -1.0 * jno.np.inner(t, phi.bind(x=xr, y=yr), n_contract=1)
    fem = jno.fem([weak, u(xl, yl) - (0.0, 0.0), traction])

    assert "surface@right" in fem.classification  # NOT silently classified as volume
    b = np.asarray(fem.b).reshape(-1)
    assert np.linalg.norm(b) > 0.0  # the traction contributes to the load vector

    sol = np.linalg.solve(_dense(fem.A), b).reshape(-1, 2)
    c = np.asarray(d.mesh.points)[:, :2]
    exact = np.stack([c[:, 0], np.zeros_like(c[:, 1])], axis=-1)  # u = (x, 0)
    assert np.linalg.norm(exact - sol) / np.linalg.norm(exact) < 1e-9


def test_vector_traction_inner_and_component_forms_agree():
    # The natural vector form inner(t, phi(region)) and the component form
    # t_x * phi(region)[0] describe the same traction; both must classify onto the
    # boundary (coords survive the reduction / the component index) and assemble to
    # the same system. Fresh domain/vars per solve so the in-place quadrature
    # retagging of one solve can't leak into the other.
    def solve(make_traction):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.25)
        u, phi = d.fem_symbols(value_shape=(2,))
        xi, yi, _ = d.variable("interior", split=True)
        xl, yl, _ = d.variable("left", split=True)
        xr, yr, _ = d.variable("right", split=True)
        weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2)
        fem = jno.fem([weak, u(xl, yl) - (0.0, 0.0), make_traction(phi, xr, yr)])
        assert "surface@right" in fem.classification
        return np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1))

    t = jnp.array([1.0, 0.0])
    sol_inner = solve(lambda phi, xr, yr: -1.0 * jno.np.inner(t, phi.bind(x=xr, y=yr), n_contract=1))
    sol_comp = solve(lambda phi, xr, yr: -1.0 * phi.bind(x=xr, y=yr)[0])
    assert np.linalg.norm(sol_inner - sol_comp) / np.linalg.norm(sol_comp) < 1e-12


def test_vec3_field_assembles():
    # 3-component vector unknown: vec inferred as 3, dofs == 3 * n_nodes.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, phi = d.fem_symbols(value_shape=(3,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2)
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    A = _dense(fem.A)
    n_nodes = int(np.asarray(d.mesh.points).shape[0])
    assert fem.dofs == 3 * n_nodes
    assert A.shape[0] == A.shape[1] == 3 * n_nodes
    assert np.allclose(A, A.T, atol=1e-7)
