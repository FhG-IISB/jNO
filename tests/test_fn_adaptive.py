import jax.numpy as jnp

import jno
from jno.trace import OperationDef
from jno.trace_compiler import TraceCompiler
from tests.conftest import make_var


def test_fn_adaptive_relobralo_syntax_and_compilation():
    x = make_var("x")
    pde = (x - 1.0).mse
    bcs = (x + 1.0).mse

    w0, w1 = jno.fn.adaptive.ReLoBRaLo([pde, bcs], x, mode="minmax")
    expr = w0 * pde + w1 * bcs

    compiled = TraceCompiler.compile_traced_expression(expr, [OperationDef(expr)])
    points = {"x": jnp.linspace(-1.0, 1.0, 11).reshape(-1, 1)}

    out1 = compiled({}, points)
    out2 = compiled({}, points)

    assert bool(jnp.all(jnp.isfinite(out1)))
    assert bool(jnp.all(jnp.isfinite(out2)))


def test_fn_adaptive_lbpinns_returns_one_weight_per_loss():
    x = make_var("x")
    pde = (x - 1.0).mse
    bcs = (x + 1.0).mse

    w0, w1 = jno.fn.adaptive.LbPINNsLossBalancing([pde, bcs], x, mode="raw")
    expr = w0 * pde + w1 * bcs

    compiled = TraceCompiler.compile_traced_expression(expr, [OperationDef(expr)])
    points = {"x": jnp.linspace(-0.5, 0.5, 9).reshape(-1, 1)}
    out = compiled({}, points)

    assert bool(jnp.all(jnp.isfinite(out)))


def test_fn_adaptive_auto_exports_modules_via_dunder_all():
    exported = dir(jno.fn.adaptive)
    assert "ReLoBRaLo" in exported
    assert "relobralo" in exported
    assert "DLRS" in exported
    assert "dlrs" in exported
    assert "SoftAdapt" in exported
    assert "softadapt" in exported
    assert "DWA" in exported
    assert "dwa" in exported
    assert "RLW" in exported
    assert "rlw" in exported


def test_fn_adaptive_factory_symbol_works_for_traced_loss_balancing():
    x = make_var("x")
    pde = (x - 1.0).mse
    bcs = (x + 1.0).mse

    w0, w1 = jno.fn.adaptive.relobralo([pde, bcs], x, mode="minmax")
    expr = w0 * pde + w1 * bcs

    compiled = TraceCompiler.compile_traced_expression(expr, [OperationDef(expr)])
    points = {"x": jnp.linspace(-1.0, 1.0, 11).reshape(-1, 1)}
    out = compiled({}, points)

    assert bool(jnp.all(jnp.isfinite(out)))


def test_fn_adaptive_lrscheduler_symbols_are_passthrough_constructors():
    sched = jno.fn.adaptive.dlrs(lr0=1e-3, window=3)
    assert sched.__class__.__name__ == "DLRS"


def test_fn_adaptive_softadapt_traced_compilation():
    x = make_var("x")
    pde = (x - 1.0).mse
    bcs = (x + 1.0).mse

    w0, w1 = jno.fn.adaptive.SoftAdapt([pde, bcs], x, mode="raw")
    expr = w0 * pde + w1 * bcs

    compiled = TraceCompiler.compile_traced_expression(expr, [OperationDef(expr)])
    points = {"x": jnp.linspace(-1.0, 1.0, 11).reshape(-1, 1)}
    out = compiled({}, points)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_fn_adaptive_dwa_traced_compilation():
    x = make_var("x")
    pde = (x - 1.0).mse
    bcs = (x + 1.0).mse

    w0, w1 = jno.fn.adaptive.dwa([pde, bcs], x, mode="minmax")
    expr = w0 * pde + w1 * bcs

    compiled = TraceCompiler.compile_traced_expression(expr, [OperationDef(expr)])
    points = {"x": jnp.linspace(-1.0, 1.0, 11).reshape(-1, 1)}

    # DWA needs 3 calls to produce non-uniform weights; verify all are finite
    for _ in range(3):
        out = compiled({}, points)
        assert bool(jnp.all(jnp.isfinite(out)))


def test_fn_adaptive_rlw_traced_compilation():
    x = make_var("x")
    pde = (x - 1.0).mse
    bcs = (x + 1.0).mse

    w0, w1 = jno.fn.adaptive.rlw([pde, bcs], x, mode="raw")
    expr = w0 * pde + w1 * bcs

    compiled = TraceCompiler.compile_traced_expression(expr, [OperationDef(expr)])
    points = {"x": jnp.linspace(-1.0, 1.0, 11).reshape(-1, 1)}
    out = compiled({}, points)
    assert bool(jnp.all(jnp.isfinite(out)))


def test_fn_adaptive_single_call_no_mode():
    """Single-call syntax without mode/context."""
    x = make_var("x")
    pde = (x - 1.0).mse
    bcs = (x + 1.0).mse

    w0, w1 = jno.fn.adaptive.dwa([pde, bcs])
    expr = w0 * pde + w1 * bcs

    compiled = TraceCompiler.compile_traced_expression(expr, [OperationDef(expr)])
    points = {"x": jnp.linspace(-1.0, 1.0, 11).reshape(-1, 1)}
    out = compiled({}, points)
    assert bool(jnp.all(jnp.isfinite(out)))
