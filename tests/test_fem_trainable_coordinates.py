"""Differentiable mesh geometry — ``Variable.trainable()`` on a spatial coordinate (Feature 2).

Promoting a coordinate component of a region (``x.trainable()``) makes that region's **vertex x-positions**
a design variable that the assembler scatters into the P1 geometry, so ``∂(fem.solve())/∂X`` is a genuine
JAX autodiff quantity — the keystone for differentiable r-adaptivity (topology-preserving mesh relocation).

Oracles:
  * **FD vs autodiff** of a functional ``J(u)=½∫u²`` on a linear Poisson solve, w.r.t. the moved vertices
    (2D triangle + 3D tet) — the direct correctness proof that the assembly Jacobian is differentiable in X;
  * **region restriction** — only the promoted region's vertices are design variables;
  * **relocation descent** — the gradient is usable: a few steps down ``∂J/∂X`` reduce ``J`` without tangling;
  * **crux integration** — the coordinate parameter is discovered and moved by ``jno.core`` (the training path);
and the fail-loud scope (surface/Neumann terms need differentiable normals, Feature 3).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)  # float64 for the FD-vs-autodiff exactness
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson_2d(d):
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def _poisson_3d(d):
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    return jno.fem([ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - 1.0 * vi, u(xb, yb, zb) - 0.0], quad_degree=2)


def _fd_grad(Jf, X0, eps=1e-6):
    g = np.zeros(X0.shape[0])
    for i in range(X0.shape[0]):
        g[i] = (float(Jf(X0.at[i].add(eps))) - float(Jf(X0.at[i].add(-eps)))) / (2 * eps)
    return g


def test_coordinate_gradient_matches_fd_2d():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.15).domain()
    xi, _, _ = d.variable("mv", where=lambda x, y: (x > 0.25) & (x < 0.75) & (y > 0.25) & (y < 0.75), split=True)
    xi.trainable(name="cx")
    op = d._trainable_coords[0]
    ids, axis, name = op["ids"], op["axis"], op["name"]
    fem = _poisson_2d(d)
    X0 = jnp.asarray(np.asarray(d.mesh.points)[ids, axis])

    def Jf(X):
        A, b = fem.operator.evaluate({name: X})
        u = jnp.linalg.solve(jnp.asarray(A.todense()), jnp.asarray(b).reshape(-1))
        return 0.5 * jnp.sum(u * u)

    g_ad = np.asarray(jax.grad(Jf)(X0))
    g_fd = _fd_grad(Jf, X0)
    assert np.linalg.norm(g_ad) > 1e-8, "coordinate gradient is zero — geometry not differentiable in X"
    rel = np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd)
    assert rel < 1e-6, f"∂J/∂X autodiff vs FD rel err {rel:.2e}"


def test_coordinate_gradient_matches_fd_3d():
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.34).domain()
    xi, _, _, _ = d.variable(
        "mv", where=lambda x, y, z: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8) & (z > 0.2) & (z < 0.8), split=True
    )
    xi.trainable(name="cx")
    op = d._trainable_coords[0]
    ids, axis, name = op["ids"], op["axis"], op["name"]
    assert len(ids) > 0, "no interior vertices selected in 3D box"
    fem = _poisson_3d(d)
    X0 = jnp.asarray(np.asarray(d.mesh.points)[ids, axis])

    def Jf(X):
        A, b = fem.operator.evaluate({name: X})
        u = jnp.linalg.solve(jnp.asarray(A.todense()), jnp.asarray(b).reshape(-1))
        return 0.5 * jnp.sum(u * u)

    g_ad = np.asarray(jax.grad(Jf)(X0))
    g_fd = _fd_grad(Jf, X0, eps=1e-6)
    assert np.linalg.norm(g_ad) > 1e-10, "3D coordinate gradient is zero"
    rel = np.linalg.norm(g_ad - g_fd) / np.linalg.norm(g_fd)
    assert rel < 1e-5, f"3D ∂J/∂X autodiff vs FD rel err {rel:.2e}"


def test_only_promoted_region_is_trainable():
    """The design variable is exactly the promoted region's vertices — literal, per-component."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.15).domain()
    pts = np.asarray(d.mesh.points)
    in_box = (pts[:, 0] > 0.25) & (pts[:, 0] < 0.75) & (pts[:, 1] > 0.25) & (pts[:, 1] < 0.75)
    xi, _, _ = d.variable("mv", where=lambda x, y: (x > 0.25) & (x < 0.75) & (y > 0.25) & (y < 0.75), split=True)
    xi.trainable(name="cx")
    spec = d._trainable_coords[0]
    assert set(spec["ids"].tolist()) == set(np.where(in_box)[0].tolist()), "promoted ids != region vertices"
    assert spec["axis"] == 0, "x.trainable() must promote only the x component"


def test_relocation_descent_reduces_objective():
    """The coordinate gradient is usable: descending ∂J/∂X lowers J without tangling the mesh."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    xi, yi, _ = d.variable("mv", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
    xi.trainable(name="cx")
    spec = d._trainable_coords[0]
    ids, axis, name = spec["ids"], spec["axis"], spec["name"]
    fem = _poisson_2d(d)
    X = jnp.asarray(np.asarray(d.mesh.points)[ids, axis])

    def Jf(Xv):
        A, b = fem.operator.evaluate({name: Xv})
        u = jnp.linalg.solve(jnp.asarray(A.todense()), jnp.asarray(b).reshape(-1))
        return 0.5 * jnp.sum(u * u)

    J0 = float(Jf(X))
    for _ in range(5):
        X = X - 0.02 * jax.grad(Jf)(X)  # small steps -> stays in the valid (non-tangling) region
    J1 = float(Jf(X))
    assert J1 < J0, f"relocation did not reduce the objective: {J0:.6e} -> {J1:.6e}"


def test_crux_trains_the_coordinate_parameter():
    """Integration: jno.core discovers the coordinate parameter and moves it (the training path)."""
    import optax

    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    xi, _, _ = d.variable("mv", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
    cx = xi.trainable(name="cx")
    cx.optimizer(optax.adam(1e-2))
    spec = d._trainable_coords[0]
    X_seed = np.asarray(d.mesh.points)[spec["ids"], spec["axis"]].copy()
    fem = _poisson_2d(d)
    # target = half the seed-geometry solution -> minimizing (u(X) - target)² requires moving the vertices
    A0, b0 = fem.operator.evaluate({spec["name"]: jnp.asarray(X_seed)})
    u_target = 0.5 * jnp.linalg.solve(jnp.asarray(A0.todense()), jnp.asarray(b0).reshape(-1))
    crux = jno.core([(fem.solve() - u_target).mse], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
    crux.solve(15)
    X_trained = np.asarray(crux.eval([cx])[0]).reshape(-1)
    assert np.linalg.norm(X_trained - X_seed) > 1e-4, "crux did not move the coordinate parameter"


def test_coordinate_plus_surface_term_raises():
    """Fail-loud scope: trainable coordinates + surface (Neumann) terms need differentiable normals (Feature 3)."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    xi, _, _ = d.variable("mv", where=lambda x, y: (x > 0.25) & (x < 0.75) & (y > 0.25) & (y < 0.75), split=True)
    xi.trainable(name="cx")
    u, phi = d.fem_symbols()
    xin, yin, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xin, y=yin), phi.bind(x=xin, y=yin)
    vb = phi.bind(x=xb, y=yb)  # a Neumann flux term (boundary integral) -> surface_work
    with pytest.raises(NotImplementedError, match="facet normals"):
        jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, 1.0 * vb], quad_degree=2)
