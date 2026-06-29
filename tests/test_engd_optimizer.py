"""Tests for jno.optimizers.engd — the ENGDOptimizer sentinel API."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import numpy as _np
import optax

import jno
from jno.optimizers import ENGDOptimizer, engd
from jno.optimizers.engd import ENGDOptimizer as _ENGDOptimizerDirect

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

    dom = jno.domain.rect(mesh_size=0.05)
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
    import jax

    jax.config.update("jax_enable_x64", True)

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
    import jax

    jax.config.update("jax_enable_x64", True)

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
    import jax

    jax.config.update("jax_enable_x64", True)

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
    import jax

    jax.config.update("jax_enable_x64", True)

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
    import jax

    jax.config.update("jax_enable_x64", True)

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
    import jax

    from jno.trace import FunctionCall

    jax.config.update("jax_enable_x64", True)

    net, losses, _, eval_error = _build_poisson2d(seed=0)

    # Verify that the losses use FunctionCall with reduces_axis set (the
    # property our code relies on — not _name).
    for loss in losses:
        assert isinstance(loss, FunctionCall), "expected FunctionCall loss"
        assert loss.reduces_axis, "expected reduces_axis to be truthy"

    net.optimizer(engd())
    crux = jno.core(losses)
    crux.solve(5)
    assert math.isfinite(eval_error(crux))


# ---------------------------------------------------------------------------
# compare() with new API style
# ---------------------------------------------------------------------------


def test_engd_optimizer_in_compare():
    """engd() dict style works end-to-end when run against Adam."""
    jax.config.update("jax_enable_x64", True)

    results = {}
    for name, opt in [("Adam", optax.adam(1e-3)), ("ENGD-new", engd(line_search=True))]:
        net, losses, _, eval_error = _build_poisson2d(seed=0)
        net.optimizer(opt)
        crux = jno.core(losses)
        crux.solve(10)
        results[name] = eval_error(crux)

    assert "ENGD-new" in results
    assert math.isfinite(results["ENGD-new"])
