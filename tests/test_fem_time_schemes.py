"""Time-integration schemes via ``fem.solve(time=...)`` — ``jno.solve.theta`` and ``jno.solve.exponential``.

Oracle is transient heat with a fundamental-mode IC (``sin πx sin πy``), which decays as ``e^{-2π²t}``.
The defining property of the exponential integrator is that it is **exact in time**: its answer is
independent of the number of steps, where backward-Euler's is not — and it beats backward-Euler at a
fixed (coarse) step count. The θ-scheme test checks the override (θ=1 ≡ default, θ=½ differs, both valid).
"""

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import box

import jno

_HAS_MATFREE = importlib.util.find_spec("matfree") is not None
PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(nsteps, T=0.03, h=0.11):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h, time=(0.0, T, nsteps))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    ic = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    return jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - ic])


def _final(fem, **kw):
    return np.asarray(fem.solve(**kw).fn())[-1]


def test_theta_override():
    """θ=1 reproduces the default (backward-Euler); θ=½ (Crank–Nicolson) gives a different, valid answer."""
    default = _final(_heat(6))
    be = _final(_heat(6), time=jno.solve.theta(1.0))
    cn = _final(_heat(6), time=jno.solve.theta(0.5))
    assert np.allclose(be, default, atol=1e-10)  # θ=1 ≡ the assembly default
    assert not np.allclose(cn, default, atol=1e-4)  # θ=½ is a genuinely different scheme
    assert np.isfinite(cn).all()


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_is_exact_in_time():
    """The exponential integrator is *exact in time*: its answer does not depend on the step count, where
    backward-Euler's does. exp(2 steps) ≈ exp(8 steps); BE(2) is far from BE(8)."""
    e2 = _final(_heat(2), time=jno.solve.exponential(order=40))
    e8 = _final(_heat(8), time=jno.solve.exponential(order=40))
    assert np.linalg.norm(e2 - e8) / np.linalg.norm(e8) < 1e-5  # step-independent ⇒ exact in time

    b2, b8 = _final(_heat(2)), _final(_heat(8))
    assert np.linalg.norm(b2 - b8) / np.linalg.norm(b8) > 5e-2  # backward-Euler depends strongly on the step


@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_beats_backward_euler():
    """At a fixed coarse step count the exponential integrator is much closer to the time-converged answer."""
    ref = _final(_heat(400))  # time-converged reference
    exp_err = np.linalg.norm(_final(_heat(4), time=jno.solve.exponential(order=40)) - ref) / np.linalg.norm(ref)
    be_err = np.linalg.norm(_final(_heat(4)) - ref) / np.linalg.norm(ref)
    assert exp_err < 0.5 * be_err  # exact-in-time wins at coarse steps


@pytest.mark.slow
@pytest.mark.skipif(not _HAS_MATFREE, reason="jno.solve.exponential needs the optional 'matfree' package")
def test_exponential_consistent_mass_is_more_accurate():
    """``mass='consistent'`` (full M, no lumping error, matrix-free M-inner-product Lanczos) is closer to
    the time-converged reference than ``mass='lumped'`` — and is still exact in time (step-independent)."""
    h = 0.16  # coarse: the consistent path runs a CG M-solve per Lanczos step
    ref = _final(_heat(300, h=h))
    lump = np.linalg.norm(_final(_heat(4, h=h), time=jno.solve.exponential(mass="lumped")) - ref) / np.linalg.norm(ref)
    cons = np.linalg.norm(
        _final(_heat(4, h=h), time=jno.solve.exponential(mass="consistent", order=30)) - ref
    ) / np.linalg.norm(ref)
    assert cons < lump  # consistent mass removes the lumping error

    c4 = _final(_heat(4, h=h), time=jno.solve.exponential(mass="consistent", order=30))
    c8 = _final(_heat(8, h=h), time=jno.solve.exponential(mass="consistent", order=30))
    assert np.linalg.norm(c4 - c8) / np.linalg.norm(c8) < 1e-5  # still exact in time


def test_m_inner_product_lanczos_matches_dense_and_differentiates():
    """The matrix-free M-inner-product Lanczos (the consistent-mass engine) computes ``f(M⁻¹A)·v``
    exactly and is differentiable — pure JAX, no host factorization, so consistent mass is scalable
    *and* autodiff-friendly (unlike a dense Cholesky with a concrete interior extraction)."""
    from jno.utils.solver.mass import m_inner_funm

    rng = np.random.default_rng(0)
    n = 50
    Bm = rng.standard_normal((n, n))
    M = jnp.asarray(Bm @ Bm.T + n * np.eye(n))
    Ba = rng.standard_normal((n, n))
    A = jnp.asarray(Ba @ Ba.T + np.eye(n))
    Minv = jnp.linalg.inv(M)
    L = Minv @ A
    m_inner = lambda a, b: a @ (M @ b)
    ones = jnp.ones(n)
    e0 = ones / jnp.sqrt(m_inner(ones, ones))
    v = jnp.asarray(rng.standard_normal(n))
    t = 0.05

    fv = m_inner_funm(lambda x: Minv @ (A @ x), m_inner, e0, v, lambda lam: jnp.exp(-t * lam), order=40)
    true = jax.scipy.linalg.expm(-t * L) @ v
    assert float(jnp.linalg.norm(fv - true) / jnp.linalg.norm(true)) < 1e-8  # exact vs dense expm

    def loss(scale):
        fw = m_inner_funm(lambda x: scale * (Minv @ (A @ x)), m_inner, e0, v, lambda lam: jnp.exp(-t * lam), order=40)
        return jnp.sum(fw**2)

    g = float(jax.grad(loss)(1.0))
    fd = float((loss(1.0 + 1e-5) - loss(1.0 - 1e-5)) / 2e-5)
    assert abs(g - fd) / abs(fd) < 1e-3  # gradient flows through the matrix-free Lanczos


def test_time_scheme_rejects_a_steady_problem():
    """``time=`` selects a TIME integrator, so it is an error on a non-transient problem."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi])
    with pytest.raises(ValueError, match="transient"):
        fem.solve(time=jno.solve.theta(0.5))
