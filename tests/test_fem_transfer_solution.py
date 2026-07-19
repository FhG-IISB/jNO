"""Mesh-to-mesh solution transfer (`transfer_solution`) — the AFEM / moving-mesh keystone.

`transfer_solution` carries a nodal field across a remesh by piecewise-linear (barycentric)
interpolation over the source simplices. Strong oracles used here:
  * **exact for affine fields** — P1 interpolation reproduces any linear field to machine precision
    (2-D, 3-D, vector, complex);
  * **identity** — transferring a mesh onto *itself* returns the field unchanged;
  * **linearity** — the transfer is linear in `values`, so its JVP equals the transfer of the tangent
    (an exact differentiability oracle);
and the extremes: smooth-field interpolation error, outside-domain fill policies, and fail-loud guards.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.fem_adapt import transfer_solution


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


# ── exactness oracles ────────────────────────────────────────────────────────
def test_affine_field_transferred_exactly_2d():
    """P1 interpolation reproduces a linear field exactly — the transfer's headline guarantee."""
    src, tgt = _rect(0.1), _rect(0.07)
    aff = lambda X: 0.3 + 1.7 * X[:, 0] - 0.9 * X[:, 1]  # noqa: E731
    out = np.asarray(transfer_solution(src, jnp.asarray(aff(_verts(src))), tgt))
    assert np.max(np.abs(out - aff(_verts(tgt)))) < 1e-9


def test_identity_transfer_is_exact():
    """Mesh → itself: every target vertex coincides with a source vertex ⇒ field returned unchanged."""
    d = _rect(0.12)
    x = _verts(d)
    u = jnp.asarray(np.sin(np.pi * x[:, 0]) * x[:, 1])
    out = np.asarray(transfer_solution(d, u, d))
    assert np.max(np.abs(out - np.asarray(u))) < 1e-10


def test_3d_affine_exact():
    src, tgt = _box(0.34), _box(0.29)
    aff = lambda X: 1.0 - X[:, 0] + 2.0 * X[:, 1] - 0.5 * X[:, 2]  # noqa: E731
    out = np.asarray(transfer_solution(src, jnp.asarray(aff(_verts(src))), tgt))
    assert np.max(np.abs(out - aff(_verts(tgt)))) < 1e-8


def test_vector_field_transfer():
    src, tgt = _rect(0.1), _rect(0.08)
    V = lambda X: np.stack([X[:, 0] - 2.0 * X[:, 1], 3.0 * X[:, 0] + X[:, 1]], axis=1)  # affine vector  # noqa: E731
    out = np.asarray(transfer_solution(src, jnp.asarray(V(_verts(src))), tgt))
    assert out.shape == (len(_verts(tgt)), 2)
    assert np.max(np.abs(out - V(_verts(tgt)))) < 1e-9


def test_complex_field_preserved():
    """The eddy path is complex — the transfer must keep complex dtype and interpolate both parts."""
    src, tgt = _rect(0.1), _rect(0.08)
    aff = lambda X: (0.5 + X[:, 0]) + 1j * (1.0 - X[:, 1])  # noqa: E731
    out = np.asarray(transfer_solution(src, jnp.asarray(aff(_verts(src))), tgt))
    assert np.iscomplexobj(out)
    assert np.max(np.abs(out - aff(_verts(tgt)))) < 1e-9


# ── accuracy on a smooth (non-affine) field ──────────────────────────────────
def test_smooth_field_interpolation_error_is_small():
    src, tgt = _rect(0.05), _rect(0.045)
    f = lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])  # noqa: E731
    out = np.asarray(transfer_solution(src, jnp.asarray(f(_verts(src))), tgt))
    err = np.max(np.abs(out - f(_verts(tgt))))
    assert err < 1.5e-2, f"linear-interp error too large on h≈0.05: {err:.3e}"  # ~O(h²)


# ── differentiability (linear map ⇒ exact JVP oracle) ─────────────────────────
def test_transfer_is_a_differentiable_linear_map():
    src, tgt = _rect(0.12), _rect(0.1)
    n = len(_verts(src))
    rng = np.random.default_rng(0)
    v0 = jnp.asarray(rng.standard_normal(n))
    dv = jnp.asarray(rng.standard_normal(n))
    f = lambda v: transfer_solution(src, v, tgt)  # noqa: E731
    _, jvp = jax.jvp(f, (v0,), (dv,))
    assert np.max(np.abs(np.asarray(jvp) - np.asarray(f(dv)))) < 1e-8  # linear ⇒ JVP == f(tangent)
    g = jax.grad(lambda v: jnp.sum(f(v) ** 2))(v0)  # a real inverse-loop loss differentiates cleanly
    assert np.all(np.isfinite(np.asarray(g)))


# ── extremes: outside-domain handling + fail-loud guards ──────────────────────
def test_outside_domain_fill_policies():
    src = _rect(0.1)
    tgt = _rect(0.12, x0=-0.3, y0=-0.3, x1=1.3, y1=1.3)  # larger ⇒ corners fall outside src
    x = _verts(src)
    u = jnp.asarray(0.3 + x[:, 0] - x[:, 1])
    with pytest.raises(ValueError, match="outside the source mesh"):
        transfer_solution(src, u, tgt, fill="error")
    near = np.asarray(transfer_solution(src, u, tgt, fill="nearest"))
    assert np.all(np.isfinite(near))  # projected onto the nearest simplex ⇒ bounded
    const = np.asarray(transfer_solution(src, u, tgt, fill=-99.0))
    assert np.any(const == -99.0)  # outside vertices took the constant


def test_mismatched_values_raises():
    src, tgt = _rect(0.15), _rect(0.12)
    with pytest.raises(ValueError, match="vertices"):
        transfer_solution(src, jnp.zeros(len(_verts(src)) + 3), tgt)


def test_dimension_mismatch_raises():
    with pytest.raises(ValueError, match="dimensions differ"):
        transfer_solution(_box(0.4), jnp.zeros(len(_verts(_box(0.4)))), _rect(0.1))
