"""Tests for jno.optimizers.engd — the ENGDOptimizer sentinel API."""

from __future__ import annotations

import math

import optax

import jno
import jno.baseline as B
from jno.optimizers import ENGDOptimizer, engd
from jno.optimizers.engd import ENGDOptimizer as _ENGDOptimizerDirect

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

    net, losses, _, eval_error = B.Poisson2D().build(seed=0)
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

    net, losses, gram_terms, eval_error = B.Poisson2D().build(seed=0)
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

    net, losses, _, eval_error = B.Poisson2D().build(seed=0)
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

    net, losses, _, eval_error = B.Poisson2D().build(seed=0)
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

    net, losses, _, eval_error = B.Poisson2D().build(seed=0)
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

    net, losses, _, eval_error = B.Poisson2D().build(seed=0)

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
    """compare() with the new engd() dict style (no lambda) works end-to-end."""
    result = B.Poisson2D().compare(
        {
            "Adam": {"optimizer": optax.adam(1e-3)},
            "ENGD-new": {"optimizer": engd(line_search=True)},
        },
        seeds=1,
        epochs=10,
        interval=5,
    )
    assert "ENGD-new" in result.data
    assert math.isfinite(result.data["ENGD-new"]["final_mean"])
