"""``jno.solve.{logdet, trace, applyfun}`` — matrix-free, differentiable matrix functions (via matfree).

``logdet`` / ``trace`` are stochastic (Hutchinson + Lanczos quadrature) so gates are loose; ``applyfun``
(``f(A)·v`` by Lanczos) is deterministic and essentially exact for the exponential. The last test trains
a parameter *through* ``logdet`` — the Bayesian-evidence workflow the layer exists for.
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import box

import jno
from jno.utils.solver.solver_api import LinearOperator

_HAS_MATFREE = importlib.util.find_spec("matfree") is not None
pytestmark = pytest.mark.skipif(not _HAS_MATFREE, reason="requires the optional 'matfree' package")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _spd_fem():
    """An SPD FEM operator (mass + stiffness) and its dense form."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.09)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A = jno.fem([ui * vi + ui.x * vi.x + ui.y * vi.y]).operator[0]
    dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A)
    return LinearOperator(A), dense


def test_logdet_matches_dense():
    op, dense = _spd_fem()
    est = float(jno.solve.logdet(op, samples=500, order=30, key=jax.random.PRNGKey(0)))
    true = float(jnp.linalg.slogdet(dense)[1])
    assert abs(est - true) / abs(true) < 0.02  # stochastic estimate


def test_trace_and_trace_of_inverse():
    op, dense = _spd_fem()
    tr = float(jno.solve.trace(op, samples=800, key=jax.random.PRNGKey(0)))
    assert abs(tr - float(jnp.trace(dense))) / float(jnp.trace(dense)) < 0.03
    tri = float(jno.solve.trace(op, fun=lambda z: 1.0 / z, samples=800, order=30, key=jax.random.PRNGKey(1)))
    assert abs(tri - float(jnp.trace(jnp.linalg.inv(dense)))) / float(jnp.trace(jnp.linalg.inv(dense))) < 0.05


def test_applyfun_matrix_exponential_is_exact():
    op, dense = _spd_fem()
    w = jnp.asarray(np.random.default_rng(0).standard_normal(dense.shape[0]))
    fv = jno.solve.applyfun(op, w, fun=lambda z: jnp.exp(-0.1 * z), order=35)
    true = jax.scipy.linalg.expm(-0.1 * dense) @ w
    assert float(jnp.linalg.norm(fv - true) / jnp.linalg.norm(true)) < 1e-8


def test_logdet_gradient_matches_analytic():
    """∂/∂s log det(sA) = n/s (=n at s=1) — the eigenvalue-scaling identity; autodiff flows through."""
    _op, dense = _spd_fem()
    n = dense.shape[0]
    key = jax.random.PRNGKey(0)

    def ld(s):
        return jno.solve.logdet(s * dense, samples=400, order=30, key=key)

    g = float(jax.grad(ld)(1.0))
    assert abs(g - n) / n < 0.02


@pytest.mark.slow
def test_logdet_trains_a_parameter():
    """Train a scale ``θ`` by matching ``logdet(θ·A)`` to a target — the differentiable-evidence loop.
    With a fixed key the estimator satisfies ``logdet(θA) = logdet(A) + n·log θ`` *exactly* (same probes),
    so the recovered optimum is θ*=2, and reaching it proves the gradient of ``logdet`` drives training."""
    _op, dense = _spd_fem()
    n = dense.shape[0]
    key = jax.random.PRNGKey(0)
    cfg = dict(samples=300, order=30, key=key)
    target = jno.solve.logdet(dense, **cfg) + n * jnp.log(2.0)  # ⇒ θ* = 2

    import optax

    def loss(theta):
        return (jno.solve.logdet(theta * dense, **cfg) - target) ** 2

    theta = jnp.array(1.0)
    opt = optax.adam(0.05)
    state = opt.init(theta)
    grad = jax.jit(jax.grad(loss))
    for _ in range(200):
        updates, state = opt.update(grad(theta), state)
        theta = optax.apply_updates(theta, updates)
    assert abs(float(theta) - 2.0) < 0.05  # recovered the scale by training through logdet


def _advection_fem():
    """A **non-symmetric** advection–diffusion FEM operator ``b·∇u + (∇u,∇v)`` and its dense form."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.16)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    A = jno.fem([5.0 * ui.x * vi + ui.x * vi.x + ui.y * vi.y]).operator[0]  # convection ⇒ A ≠ Aᵀ
    dense = jnp.asarray(A.todense() if hasattr(A, "todense") else A)
    return LinearOperator(A), dense


_CPU = jax.devices("cpu")[0]  # jax.scipy.linalg.schur (the non-symmetric path) is CPU-only


def test_applyfun_nonsymmetric_forward_is_exact():
    """``symmetric=False`` (Arnoldi) computes ``exp(A)·v`` for a non-symmetric advection–diffusion ``A``,
    forward-exact — matching the dense matrix exponential. CPU-only (Schur)."""
    _op, dense = _advection_fem()
    w = jnp.asarray(np.random.default_rng(0).standard_normal(dense.shape[0]))
    with jax.default_device(_CPU):
        fv = jno.solve.applyfun(dense, w, fun=lambda z: jnp.exp(-0.05 * z), order=40, symmetric=False)
        true = jax.scipy.linalg.expm(-0.05 * dense) @ w
    assert float(jnp.linalg.norm(fv - true) / jnp.linalg.norm(true)) < 1e-8


def test_applyfun_nonsymmetric_gradient_is_blocked_loudly():
    """The non-symmetric path is forward-only: differentiating it raises (JAX has no Schur derivative),
    per the house rule against a silently-non-differentiable path dressed up as differentiable."""
    _op, dense = _advection_fem()
    w = jnp.asarray(np.random.default_rng(0).standard_normal(dense.shape[0]))

    def loss(s):
        return jnp.sum(jno.solve.applyfun(s * dense, w, fun=lambda z: jnp.exp(-0.05 * z), symmetric=False) ** 2)

    with jax.default_device(_CPU), pytest.raises(NotImplementedError):
        jax.grad(loss)(1.0)


def test_diagonal_matches_dense():
    """``diagonal`` estimates the per-DOF diagonal field — of ``A`` and (the key use) of ``A⁻¹``, the
    pointwise variance map. Stochastic, so the whole-field error is loose."""
    op, dense = _spd_fem()
    d0 = jno.solve.diagonal(op, samples=4000, key=jax.random.PRNGKey(0))
    true0 = jnp.diagonal(dense)
    assert float(jnp.linalg.norm(d0 - true0) / jnp.linalg.norm(true0)) < 0.1
    # diag(A⁻¹) (the variance map) is a much higher-variance estimand — loose gate, honestly stochastic
    di = jno.solve.diagonal(op, fun=lambda z: 1.0 / z, samples=6000, order=30, key=jax.random.PRNGKey(1))
    truei = jnp.diagonal(jnp.linalg.inv(dense))
    assert float(jnp.linalg.norm(di - truei) / jnp.linalg.norm(truei)) < 0.2


def test_diagonal_gradient_matches_trace():
    """``∂/∂s Σ diag(sA) = Σ diag(A) = tr A`` — autodiff flows through the diagonal estimator."""
    _op, dense = _spd_fem()
    key = jax.random.PRNGKey(0)
    g = float(jax.grad(lambda s: jnp.sum(jno.solve.diagonal(s * dense, samples=2000, key=key)))(1.0))
    assert abs(g - float(jnp.trace(dense))) / float(jnp.trace(dense)) < 0.05


@pytest.mark.skipif(_HAS_MATFREE, reason="matfree installed; this checks the missing-dependency message")
def test_missing_matfree_message():
    _op, dense = _spd_fem()
    with pytest.raises(ImportError, match="matfree"):
        jno.solve.logdet(dense)
