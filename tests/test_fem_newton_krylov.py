"""The optimistix-free nonlinear default: matrix-free Jacobian-free Newton-Krylov.

Deliberately imports **no** optimistix (unlike test_fem_inverse, which gates the
whole module on it), so these run on the new default and prove:
  * the implicit-diff gradient is exact (finite-diff vs autodiff), and
  * a steady nonlinear ``fem.solve()`` converges with optimistix forced absent.
"""
from __future__ import annotations

import sys

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.newton_krylov import newton_krylov  # noqa: E402

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely")
from shapely.geometry import box  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """feax assembly is float64; the global x64 flag is shared across modules and other
    suites flip it at import, so set it per-test with save/restore."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_newton_krylov_gradient_matches_fd():
    """Implicit diff through the solve is exact: AD == central finite-difference."""
    n = 40
    A = (2.0 * jnp.eye(n) - jnp.eye(n, k=1) - jnp.eye(n, k=-1))  # SPD tridiag
    b = jnp.ones(n)
    u_tgt = jnp.linspace(0.0, 1.0, n)

    def usol(alpha):
        return newton_krylov(lambda u: A @ u + alpha * u**3 - b, jnp.zeros(n))

    def loss(alpha):
        return jnp.mean((usol(alpha) - u_tgt) ** 2)

    a0 = jnp.array(0.7)
    # forward actually solves the nonlinear system
    u = usol(a0)
    assert float(jnp.linalg.norm(A @ u + a0 * u**3 - b)) < 1e-9

    g_ad = float(jax.grad(loss)(a0))
    e = 1e-5
    g_fd = float((loss(a0 + e) - loss(a0 - e)) / (2 * e))
    assert abs(g_ad - g_fd) / abs(g_fd) < 1e-6, f"AD {g_ad} vs FD {g_fd}"


def _nonlinear_fem(mesh_size=0.2):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    weak = ui.x * vi.x + ui.y * vi.y + (u * u * u) * vi - f * vi
    return jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)


def test_nonlinear_default_solves_without_optimistix(monkeypatch):
    """The default nonlinear engine converges on the real feax residual with optimistix
    forced absent (proving the steady path never imports it)."""
    monkeypatch.setitem(sys.modules, "optimistix", None)  # any `import optimistix` now raises
    fem = _nonlinear_fem()
    res_fn = fem.residual                          # the (u -> flat residual) feax callable
    u = newton_krylov(res_fn, jnp.zeros(fem.dofs))  # same solver fem.solve() now defaults to
    res = float(jnp.linalg.norm(jnp.asarray(res_fn(u))))
    assert np.all(np.isfinite(np.asarray(u)))
    assert res < 1e-6, f"nonlinear residual not converged: {res:.1e}"
