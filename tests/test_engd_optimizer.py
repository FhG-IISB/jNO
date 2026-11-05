"""Tests for jno.optimizers.engd — the ENGDOptimizer sentinel API."""

from __future__ import annotations

import math
import sys

import equinox as eqx
import jax
import numpy as _np
import optax
import pytest

import jno
from jno.optimizers import ENGDOptimizer, engd
from jno.optimizers.engd import ENGDOptimizer as _ENGDOptimizerDirect


@pytest.fixture(autouse=True)
def _x64():
    """ENGD's Gram solve needs float64. The session default is x64-off (see tests/conftest.py), and
    the flag is process-wide -- save/restore keeps it from leaking to whatever module runs next."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


# ---------------------------------------------------------------------------
# Inline Poisson2D fixture (jno.baseline lives on a separate branch)
# -Δu = 2π²sin(πx)sin(πy) on [0,1]², u=0 on ∂Ω, exact u*=sin(πx)sin(πy).
# ---------------------------------------------------------------------------

_N_INT = 30
_N_BDY = 30


def _int_pts():
    k = _np.linspace(1 / (_N_INT + 1), _N_INT / (_N_INT + 1), _N_INT)
    X, Y = _np.meshgrid(k, k, indexing="ij")
    return _np.stack([X.ravel(), Y.ravel()], axis=1).astype(_np.float64)


def _bdy_pts():
    t = _np.linspace(0, 1, _N_BDY + 1)[:-1]
    return _np.concatenate(
        [
            _np.stack([t, _np.zeros(_N_BDY)], 1),
            _np.stack([t, _np.ones(_N_BDY)], 1),
            _np.stack([_np.zeros(_N_BDY), t], 1),
            _np.stack([_np.ones(_N_BDY), t], 1),
        ]
    ).astype(_np.float64)


def _build_poisson2d(seed: int = 0, hidden_dims: int = 32, num_layers: int = 1):
    """Return (net, losses, gram_terms, eval_error) for Poisson2D on [0,1]²."""
    import foundax

    dom = jno.shape.rect(0, 0, 1, 1, size=0.05).domain()
    x, y, _ = dom.variable("interior")
    xb, yb, _ = dom.variable("boundary")
    dom.context["interior"] = _int_pts()[None, None]
    dom.context["boundary"] = _bdy_pts()[None, None]

    base = foundax.mlp(
        in_features=2,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(seed),
    )
    scaled = jax.tree_util.tree_map(lambda w: w * 0.1 if eqx.is_array(w) else w, base)
    net = jno.nn.wrap(scaled)

    pi = jno.np.pi
    u = net(x, y)
    pde_res = -u.laplacian(x, y) - 2 * pi**2 * jno.np.sin(pi * x) * jno.np.sin(pi * y)
    bc_res = net(xb, yb)
    pred = u
    exact = jno.np.sin(pi * x) * jno.np.sin(pi * y)
    gram_terms = [(pde_res.grad(net), 1.0), (bc_res.grad(net), 1.0)]
    losses = [pde_res.mse, bc_res.mse]

    def eval_error(crux):
        p, r = crux.eval([pred, exact])
        p_arr, r_arr = _np.asarray(p).ravel(), _np.asarray(r).ravel()
        return float(_np.linalg.norm(p_arr - r_arr) / (_np.linalg.norm(r_arr) + 1e-30))

    return net, losses, gram_terms, eval_error


# ---------------------------------------------------------------------------
# Sentinel construction
# ---------------------------------------------------------------------------


def test_engd_returns_sentinel():
    opt = engd()
    assert isinstance(opt, ENGDOptimizer)
    assert isinstance(opt, _ENGDOptimizerDirect)


def test_engd_defaults():
    opt = engd()
    assert opt._gram_terms is None
    assert opt._gram_interval == 1
    assert opt._rcond is None
    assert opt._line_search is True


def test_engd_explicit_args():
    gt = [("placeholder", 0.5)]
    opt = engd(gram_terms=gt, gram_interval=5, rcond=1e-10, line_search=False)
    assert opt._gram_terms is gt
    assert opt._gram_interval == 5
    assert opt._rcond == 1e-10
    assert opt._line_search is False


def test_engd_repr():
    r = repr(engd())
    assert "ENGDOptimizer" in r
    assert "gram_interval=1" in r


# ---------------------------------------------------------------------------
# Auto-detection — smoke: gram_terms built from losses at solve time
# ---------------------------------------------------------------------------


def test_engd_optimizer_auto_detect_smoke():
    """net.optimizer(engd()) resolves gram_terms automatically during solve."""

    net, losses, _, eval_error = _build_poisson2d(seed=0)
    net.optimizer(engd())  # no optax.sgd, no gram_terms
    crux = jno.core(losses)
    crux.solve(5)  # just check it doesn't crash

    err = eval_error(crux)
    assert math.isfinite(err), f"non-finite error after 5 steps: {err}"
    assert err >= 0.0


# ---------------------------------------------------------------------------
# Auto-detection — explicit gram_terms passed to engd()
# ---------------------------------------------------------------------------


def test_engd_optimizer_explicit_gram_terms():
    """Explicit gram_terms= bypasses auto-detection and is forwarded."""

    net, losses, gram_terms, eval_error = _build_poisson2d(seed=0)
    net.optimizer(engd(gram_terms=gram_terms))
    crux = jno.core(losses)
    crux.solve(5)

    err = eval_error(crux)
    assert math.isfinite(err)


# ---------------------------------------------------------------------------
# Convergence: matches jno.callbacks.engd on Poisson2D
# ---------------------------------------------------------------------------


def test_engd_optimizer_convergence():
    """jno.optimizers.engd() must reach rel-L² < 1e-3 in 200 epochs (=callbacks form)."""

    net, losses, _, eval_error = _build_poisson2d(seed=0)
    net.optimizer(engd(line_search=True))
    crux = jno.core(losses)
    crux.solve(200)

    err = eval_error(crux)
    assert math.isfinite(err), f"ENGD produced non-finite error: {err}"
    assert err < 1e-3, f"ENGD did not converge: rel-L² = {err:.3e} (expected < 1e-3)"


# ---------------------------------------------------------------------------
# Model not in any constraint — sentinel is silently skipped
# ---------------------------------------------------------------------------


def test_engd_optimizer_orphan_model_is_ignored():
    """A model with ENGDOptimizer that isn't in any constraint is not detected
    (it never appears in all_ops), so no error is raised and no ENGDCallback
    is injected.  The primary model still trains normally."""

    import foundax

    net, losses, _, eval_error = _build_poisson2d(seed=0)
    orphan_net = jno.nn.wrap(
        foundax.mlp(
            in_features=2,
            hidden_dims=8,
            num_layers=1,
            activation=jax.nn.tanh,
            key=jax.random.PRNGKey(99),
        )
    )
    # orphan_net is not in any of `losses` — _collect_flax_modules never sees it
    orphan_net.optimizer(engd())
    net.optimizer(optax.adam(1e-3))

    crux = jno.core(losses)
    crux.solve(5)  # must not raise

    err = eval_error(crux)
    assert math.isfinite(err)


# ---------------------------------------------------------------------------
# Idempotent: calling solve() twice doesn't corrupt fm._opt_fn
# ---------------------------------------------------------------------------


def test_engd_optimizer_idempotent_solve():
    """fm._opt_fn stays an ENGDOptimizer across multiple solve() calls."""

    net, losses, _, eval_error = _build_poisson2d(seed=0)
    net.optimizer(engd(gram_interval=2))
    crux = jno.core(losses)

    crux.solve(5)
    # After solve, the sentinel must still be on the model (not replaced)
    fm = list(crux._collect_flax_modules().values())[0]
    assert isinstance(fm._opt_fn, ENGDOptimizer)

    # A second solve should still work
    crux.solve(5)
    err = eval_error(crux)
    assert math.isfinite(err)


# ---------------------------------------------------------------------------
# Auto-detect uses reduces_axis, not name — handles custom reductions
# ---------------------------------------------------------------------------


def test_engd_optimizer_custom_reduction_unwrapped_via_reduces_axis():
    """Auto-detect strips reduction wrappers via FunctionCall.reduces_axis,
    not by name — so a custom reduction function still gets unwrapped
    correctly and the raw residual is passed to .grad()."""

    from jno.trace import FunctionCall

    net, losses, _, eval_error = _build_poisson2d(seed=0)

    # Verify that the losses use FunctionCall with reduces_axis set (the
    # property our code relies on — not _name).
    for loss in losses:
        assert isinstance(loss, FunctionCall), "expected FunctionCall loss"
        # `.reduces`, not `bool(reduces_axis)`: reduces_axis doubles as the axis VALUE, so a
        # falsy-but-real axis (`axis=0`) would fail a plain truth test. See FunctionCall.reduces.
        assert loss.reduces, "expected the loss to register as a reduction"

    net.optimizer(engd())
    crux = jno.core(losses)
    crux.solve(5)
    assert math.isfinite(eval_error(crux))


# ---------------------------------------------------------------------------
# compare() with new API style
# ---------------------------------------------------------------------------


def test_engd_optimizer_in_compare():
    """engd() dict style works end-to-end when run against Adam."""

    results = {}
    for name, opt in [("Adam", optax.adam(1e-3)), ("ENGD-new", engd(line_search=True))]:
        net, losses, _, eval_error = _build_poisson2d(seed=0)
        net.optimizer(opt)
        crux = jno.core(losses)
        crux.solve(10)
        results[name] = eval_error(crux)

    assert "ENGD-new" in results
    assert math.isfinite(results["ENGD-new"])


# ---------------------------------------------------------------------------
# SPRING momentum + Levenberg-Marquardt damping (Goldshlager, Abrahamsen & Lin 2024,
# arXiv:2401.10190; Jnini et al. 2026, arXiv:2604.05230, Sec. 2.3, eq. 26)
# ---------------------------------------------------------------------------


def _spring_kernel_oracle(J, r, phi_prev, mu, lam):
    """Eq. 26 of arXiv:2604.05230 in the paper's own N x N (kernel) form."""
    import jax.numpy as jnp

    N = J.shape[0]
    M = J @ J.T + lam * jnp.eye(N)
    return mu * phi_prev + J.T @ jnp.linalg.solve(M, r - mu * J @ phi_prev)


@pytest.mark.parametrize("mu,lam", [(0.0, 1e-3), (0.9, 1e-3), (0.99, 1e-6), (0.5, 10.0)])
def test_spring_direction_matches_kernel_form(mu, lam):
    """The P x P step with g = J^T r, G = J^T J equals the paper's N x N SPRING step (N < P)."""
    import jax.numpy as jnp

    from jno.utils.explainability import engd_direction

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    N, P = 12, 40
    J = jax.random.normal(k1, (N, P))
    r = jax.random.normal(k2, (N,))
    phi_prev = jax.random.normal(k3, (P,))
    got = engd_direction(J.T @ J, J.T @ r, phi_prev, momentum=mu, damping=lam, rcond=None)
    want = _spring_kernel_oracle(J, r, phi_prev, mu, lam)
    # P > N: J^T J + lam I has P - N eigenvalues equal to lam, so the P x P solve carries
    # cond = (sigma_max^2 + lam) / lam more rounding than the N x N one. Bound by that, not a guess.
    cond = (jnp.linalg.norm(J, 2) ** 2 + lam) / lam
    tol = 100 * jnp.finfo(J.dtype).eps * cond
    assert float(jnp.max(jnp.abs(got - want)) / jnp.max(jnp.abs(want))) < tol


def test_spring_undamped_is_pseudo_inverse_limit():
    """damping=0 (lstsq): phi = mu phi_prev + J^+ (r - mu J phi_prev), momentum only in null(J)."""
    import jax.numpy as jnp

    from jno.utils.explainability import engd_direction

    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
    N, P, mu = 10, 30, 0.8
    J = jax.random.normal(k1, (N, P))
    r = jax.random.normal(k2, (N,))
    phi_prev = jax.random.normal(k3, (P,))
    got = engd_direction(J.T @ J, J.T @ r, phi_prev, momentum=mu, damping=0.0, rcond=None)
    want = mu * phi_prev + jnp.linalg.pinv(J) @ (r - mu * J @ phi_prev)
    assert jnp.allclose(got, want, rtol=1e-7, atol=1e-9)


def test_engd_direction_plain_is_unchanged():
    """momentum=0, damping=0 is exactly the old lstsq(G, g)."""
    import jax.numpy as jnp

    from jno.utils.explainability import engd_direction

    J = jax.random.normal(jax.random.PRNGKey(2), (50, 20))
    g = jax.random.normal(jax.random.PRNGKey(3), (20,))
    G = J.T @ J
    got = engd_direction(G, g, jnp.zeros(20), momentum=0.0, damping=0.0, rcond=None)
    assert jnp.array_equal(got, jnp.linalg.lstsq(G, g)[0])


@pytest.mark.parametrize(
    "kw,match",
    [({"momentum": -0.1}, "momentum"), ({"momentum": 1.0}, "momentum"), ({"damping": -1e-3}, "damping")],
)
def test_engd_rejects_bad_momentum_and_damping(kw, match):
    """Both spellings (the sentinel and jno.callbacks.engd) reject momentum outside [0, 1) and damping < 0."""
    with pytest.raises(ValueError, match=match):
        engd(**kw)
    net, _, gram_terms, _ = _build_poisson2d(seed=0)
    with pytest.raises(ValueError, match=match):
        jno.callbacks.engd(gram_terms, **kw)


def test_engd_spring_convergence():
    """engd(momentum=, damping=) (SPRING) reaches rel-L2 < 1e-3 in 200 epochs on Poisson2D."""
    net, losses, _, eval_error = _build_poisson2d(seed=0)
    net.optimizer(engd(momentum=0.9, damping=1e-8, line_search=True))
    crux = jno.core(losses)
    crux.solve(200)
    err = eval_error(crux)
    assert math.isfinite(err), f"SPRING produced non-finite error: {err}"
    assert err < 1e-3, f"SPRING did not converge: rel-L2 = {err:.3e} (expected < 1e-3)"


def test_engd_callback_carries_spring_state():
    """The callback keeps phi between steps (momentum is applied, not silently dropped)."""
    net, losses, gram_terms, _ = _build_poisson2d(seed=0)
    cb = jno.callbacks.engd(gram_terms, momentum=0.9, damping=1e-6, line_search=False)
    net.optimizer(optax.sgd(1e-3))
    jno.core(losses).solve(3, callbacks=[cb])
    assert cb._phi is not None and float(jax.numpy.linalg.norm(cb._phi)) > 0.0

    net0, losses0, gram_terms0, _ = _build_poisson2d(seed=0)
    net1, losses1, gram_terms1, _ = _build_poisson2d(seed=0)
    c0 = jno.callbacks.engd(gram_terms0, momentum=0.0, damping=1e-6, line_search=False)
    c1 = jno.callbacks.engd(gram_terms1, momentum=0.9, damping=1e-6, line_search=False)
    for n in (net0, net1):
        n.optimizer(optax.sgd(1e-3))
    k0, k1 = jno.core(losses0), jno.core(losses1)
    k0.solve(3, callbacks=[c0])
    k1.solve(3, callbacks=[c1])
    p0 = jax.flatten_util.ravel_pytree(eqx.filter(k0.models, eqx.is_array))[0]
    p1 = jax.flatten_util.ravel_pytree(eqx.filter(k1.models, eqx.is_array))[0]
    assert not jax.numpy.allclose(p0, p1)


def test_engd_spring_line_search_carries_taken_step(monkeypatch):
    """P > N with the line search, momentum 0.9: carrying alpha_k * phi_k converges (measured 1.7e-6);
    eq. 26's unscaled phi_k stalled at 2.3e-3 on this exact setup."""
    monkeypatch.setattr(sys.modules[__name__], "_N_INT", 12)
    monkeypatch.setattr(sys.modules[__name__], "_N_BDY", 12)
    net, losses, _, eval_error = _build_poisson2d(seed=0, hidden_dims=20, num_layers=2)  # P = 501 > N = 192
    net.optimizer(engd(momentum=0.9, damping=1e-8, line_search=True))
    crux = jno.core(losses)
    crux.solve(200)
    err = eval_error(crux)
    assert err < 1e-4, f"SPRING (P > N, line search) stalled: rel-L2 = {err:.3e}"
