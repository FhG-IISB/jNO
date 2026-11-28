"""The Krylov size of ``jno.solve.exponential`` is adaptive (#327): it grows until the a-posteriori error
estimate (Saad 1992) meets the tolerance, up to a cap, and warns at the cap instead of returning a step that
is silently off. Pure-JAX loop: jit, vmap and reverse mode must keep working."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver import mass as mass_mod
from jno.utils.solver.mass import m_inner_funm

TOL = 1e-10


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    mass_mod._WARNED.clear()
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _problem(n=120, lam_max=1e3, seed=0):
    """``L = Q diag(λ) Qᵀ`` with ``λ`` from 1 to ``lam_max`` (so the Krylov size needed grows with ``t·lam_max``)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    lam = np.linspace(1.0, lam_max, n)
    L = jnp.asarray((Q * lam) @ Q.T)
    v = jnp.asarray(rng.standard_normal(n))
    e0 = jnp.ones(n) / jnp.sqrt(n)
    return L, v, e0, Q, lam


def _exp_pair(t):
    def err(lam):  # tau * phi_1(-tau lam), the estimate's weight for exp
        z = t * lam
        return t * jnp.where(z < 1e-7, 1.0 - 0.5 * z, -jnp.expm1(-z) / jnp.where(z < 1e-7, 1.0, z))

    return (lambda lam: jnp.exp(-t * lam)), err


def _run(L, v, e0, t, cap, **kw):
    fun, err = _exp_pair(t)
    return m_inner_funm(lambda x: L @ x, lambda a, b: a @ b, e0, v, fun, cap, err_fun=err, tol=TOL, m_mv=lambda b: b, **kw)


def _true(Q, lam, v, t):
    return Q @ (np.exp(-t * lam) * (Q.T @ np.asarray(v)))


def test_a_mild_step_stays_at_the_first_stage_and_a_stiff_one_grows():
    L, v, e0, Q, lam = _problem()
    out, used, est = _run(L, v, e0, 1e-3, 100, return_info=True)  # t*lam_max = 1
    assert int(used) == 16, "a mild step must not pay for more than the first stage"
    assert np.linalg.norm(out - _true(Q, lam, v, 1e-3)) < 1e-9 * np.linalg.norm(v)
    out, used, est = _run(L, v, e0, 0.2, 100, return_info=True)  # t*lam_max = 200
    assert 16 < int(used) <= 100 and float(est) <= TOL
    assert np.linalg.norm(out - _true(Q, lam, v, 0.2)) < 1e-8 * np.linalg.norm(v)


def test_the_cap_warns_with_the_order_and_the_estimate_even_under_jit():
    L, v, e0, *_ = _problem()
    with pytest.warns(RuntimeWarning, match=r"cap of 8 vectors.*estimated relative error.*smaller dt"):
        jax.block_until_ready(jax.jit(lambda v: _run(L, v, e0, 0.2, 8))(v))
    with warnings.catch_warnings():  # and a converged step is silent
        warnings.simplefilter("error")
        jax.block_until_ready(jax.jit(lambda v: _run(L, v, e0, 1e-3, 40))(v))


def test_the_adaptive_krylov_step_vmaps_and_matches_a_finite_difference_gradient():
    L, v, e0, Q, lam = _problem(n=60, lam_max=300.0)

    def loss(s, t):
        return jnp.sum(_run(s * L, v, e0, t, 64) ** 2)

    for t in (0.003, 0.4):  # the first stage / the later stages
        g = float(jax.jit(jax.grad(loss))(1.0, t))
        fd = float((loss(1.0 + 1e-6, t) - loss(1.0 - 1e-6, t)) / 2e-6)
        assert abs(g - fd) <= 1e-5 * max(abs(fd), 1e-12), (t, g, fd)
    vs = jnp.stack([v, 2.0 * v])
    out = jax.vmap(lambda w: _run(L, w, e0, 0.01, 64))(vs)
    assert np.allclose(out[1], 2.0 * out[0], atol=1e-9 * float(jnp.linalg.norm(out[0])))


def test_a_fixed_order_call_without_an_estimate_is_unchanged():
    L, v, e0, Q, lam = _problem(n=60, lam_max=300.0)
    fun, _ = _exp_pair(0.01)
    out = m_inner_funm(lambda x: L @ x, lambda a, b: a @ b, e0, v, fun, 40)
    assert np.linalg.norm(out - _true(Q, lam, v, 0.01)) < 1e-9 * np.linalg.norm(v)


def test_an_argyris_plate_reaches_the_tight_reference_and_the_old_cap_warns():
    """The consistent-mass Argyris march was 6.8e-4 off an rtol-1e-8 SDIRK3 reference on the old fixed size 40."""
    from .test_fem_time_schemes_every_element import FAMILIES, NSTEPS, S, _march

    fem = FAMILIES["argyris"]()
    save = jnp.linspace(float(fem.t0), float(fem.t1), NSTEPS)[::4]
    ref = _march(fem, S.sdirk(3).adaptive(rtol=1e-8, atol=1e-11, max_steps=4000), save)
    scale = np.max(np.abs(ref))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # the default cap is enough: no warning
        got = _march(fem, S.exponential(mass="consistent"), save)
    assert np.max(np.abs(got[1:] - ref[1:])) / scale < 1e-6
    with pytest.warns(RuntimeWarning, match="cap of 40 vectors"):
        capped = _march(fem, S.exponential(mass="consistent", order=40), save)
    assert np.max(np.abs(capped[1:] - ref[1:])) / scale > 1e-5  # what the warning is about
