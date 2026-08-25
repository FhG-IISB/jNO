"""Dtype-correctness tests.

jNO's dtype contract has two halves:

* **Data precision** follows JAX's ``jax_enable_x64`` flag (the user's concern,
  not a jNO knob).  jNO must not *leak* float32 where the JAX default is float64.
* **Model precision** is the per-model ``Model.dtype()`` knob (covered in the
  Part-B tests added on ``feature/model-dtype-real-compute``).

This module pins the data half: enabling x64 must propagate to float64
end-to-end (sampled points, attached arrays/parameters, adaptive-weight and
LR-scheduler callbacks), with no silent float32 island.

Run on GPU with ``JAX_PLATFORMS=cuda,cpu`` per the project convention; the dtype
checks themselves are platform-independent.
"""

from __future__ import annotations

import contextlib

import foundax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


@contextlib.contextmanager
def x64_enabled():
    """Enable ``jax_enable_x64`` for the duration of the block, then restore.

    x64 is a process-global JAX flag; toggling it affects *newly created* arrays,
    so every domain/array in the block is built inside the context.
    """
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _line_domain():
    return jno.domain(constructor=jno.domain.line(mesh_size=0.1))


def _attach_coeff(dom):
    """Attach a coefficient field + scalar parameter (the user-data paths)."""
    return dom < ("kfield", np.linspace(0.0, 1.0, 11).reshape(-1, 1))


# ---------------------------------------------------------------------------
# Default (no x64) — data is float32
# ---------------------------------------------------------------------------


def test_default_attached_array_is_f32():
    # User-attached arrays follow the JAX default: float32 here, float64 under
    # x64 (see test_x64_attached_array_is_f64). Mesh-derived sampled points are
    # np.float64 in storage and normalized to the default at the callback seam,
    # so we assert the contract on the user-data path we control.
    dom = _attach_coeff(_line_domain())
    assert np.asarray(dom.arrays["kfield"]).dtype == np.float32


# ---------------------------------------------------------------------------
# x64 — data is float64 end-to-end (the leak-fix guard)
# ---------------------------------------------------------------------------


def test_x64_sampled_points_are_f64():
    with x64_enabled():
        dom = _line_domain()
        x, *_ = dom.variable("interior")
        assert np.asarray(dom.context[x.tag]).dtype == np.float64


def test_x64_attached_array_is_f64():
    # Was the live leak: domain.__lt__ hardcoded np.float32 (domain_class ~737).
    with x64_enabled():
        dom = _attach_coeff(_line_domain())
        assert np.asarray(dom.arrays["kfield"]).dtype == np.float64


def test_x64_batched_parameters_are_f64():
    # The self.parameters cast (domain_class ~902) under domain batching.
    with x64_enabled():
        dom = (_line_domain() < ("c", 1.5)) + (_line_domain() < ("c", 2.5))
        assert np.asarray(dom.parameters["c"]).dtype == np.float64


def test_x64_adaptive_weight_callback_is_f64():
    # ShapeDtypeStruct + host return were hardcoded f32 (weights.py).
    from jno.utils.adaptive.weights import ReLoBRaLo

    with x64_enabled():
        balancer = ReLoBRaLo()
        weights = balancer(jnp.asarray([1.0, 2.0]))
        assert np.asarray(weights[0]).dtype == np.float64


def test_x64_lr_scheduler_callback_is_f64():
    from jno.utils.adaptive.lrscheduler import DLRS

    with x64_enabled():
        sched = DLRS(lr0=1e-3)
        lr = sched(0, jnp.asarray([1.0, 2.0]))
        assert np.asarray(lr).dtype == np.float64


# ---------------------------------------------------------------------------
# No-leak sweep — under x64, no context/array is float32
# ---------------------------------------------------------------------------


def test_x64_parameter_and_unknown_are_f64():
    """``jno.np.parameter`` hardcoded ``float32``, and it is the storage behind BOTH a trainable
    coefficient and ``domain.unknown()``. ``jno.fdm._pde_residual_fn`` injects the DOF vector into
    that module and casts it to the module's dtype, so under x64 every strong-form residual
    evaluation silently rounded the unknown to single precision -- a NON-LINEAR operator (6e-08
    relative), which capped the forward Krylov solve near 1e-05 and broke the adjoint solve outright
    (a measured gradient wrong by twenty orders). Data precision is JAX's ``jax_enable_x64``; jNO's
    only precision knob is the per-model ``.dtype()``."""
    with x64_enabled():
        assert np.asarray(jno.np.parameter((3,)).model.module.value).dtype == np.float64
        dom = _line_domain()
        assert np.asarray(dom.unknown().model.module.value).dtype == np.float64

    # ...and it must still follow JAX the other way, rather than pinning float64 of its own accord.
    assert np.asarray(jno.np.parameter((3,)).model.module.value).dtype == np.float32


def test_x64_no_float32_leak_in_context():
    with x64_enabled():
        dom = _line_domain()
        dom.variable("interior")
        dom = _attach_coeff(dom)
        leaks = {
            key: np.asarray(val).dtype
            for key, val in {**dom.context, **dom.arrays, **dom.parameters}.items()
            if np.asarray(val).dtype == np.float32
        }
        assert not leaks, f"float32 leak under jax_enable_x64: {leaks}"


# ---------------------------------------------------------------------------
# Model precision — Model.dtype() makes real compute (Part B)
# ---------------------------------------------------------------------------


def _tiny_net(key_seed=0):
    return jno.nn(foundax.mlp(1, output_dim=1, hidden_dims=16, num_layers=3, key=jax.random.PRNGKey(key_seed)))


def test_dtype_bf16_gives_real_bf16_output():
    # .dtype(bf16) casts params AND (at the seam) inputs, so the forward truly
    # computes in bf16 rather than promoting back to f32.
    dom = _line_domain()
    x, *_ = dom.variable("interior")
    net = _tiny_net()
    net.dtype(jnp.bfloat16)
    net.optimizer(optax.adam(1e-3))
    u = net(x)
    crux = jno.core([u.mse])
    crux.solve(2)
    assert np.asarray(crux.eval(u)).dtype == jnp.bfloat16


def test_dtype_default_output_is_f32():
    # Control: without .dtype(), the forward stays at the default float (f32).
    dom = _line_domain()
    x, *_ = dom.variable("interior")
    net = _tiny_net(key_seed=1)
    net.optimizer(optax.adam(1e-3))
    u = net(x)
    crux = jno.core([u.mse])
    crux.solve(2)
    assert np.asarray(crux.eval(u)).dtype == jnp.float32


def test_dtype_bf16_pinn_derivative_trains():
    # The primary jNO path: a bf16 model behind a second-derivative PINN residual
    # must run through jacfwd/jacrev and the loss must decrease (bf16 *accuracy*
    # is a documented caveat; here we only require it trains, not its precision).
    dom = jno.domain(constructor=jno.domain.line(mesh_size=0.05))
    x, *_ = dom.variable("interior")
    net = _tiny_net()
    net.dtype(jnp.bfloat16)
    net.optimizer(optax.adam(1e-3))
    u = net(x)
    pde = (u.d(x).d(x) + 1.0).mse
    crux = jno.core([pde])
    stats = crux.solve(30)
    losses = stats.training_logs[-1]["total_loss"]
    assert np.isfinite(float(losses[-1]))
    assert float(losses[-1]) < float(losses[0]), "bf16 PINN residual did not train"


def test_dtype_f64_data_f32_model_downcasts_at_seam():
    # f64 data (x64) + a deliberately f32 network: the seam casts the f64 input
    # down to f32 so the forward runs in f32 without crashing.
    with x64_enabled():
        dom = _line_domain()
        x, *_ = dom.variable("interior")
        net = _tiny_net(key_seed=2)
        net.dtype(jnp.float32)
        net.optimizer(optax.adam(1e-3))
        u = net(x)
        crux = jno.core([u.mse])
        crux.solve(2)
        assert np.asarray(crux.eval(u)).dtype == jnp.float32


def test_dtype_rejects_string_with_jax_flag_hint():
    net = _tiny_net(key_seed=3)
    with pytest.raises(ValueError, match="jax_enable_x64"):
        net.dtype("float64")


def test_x64_f32_params_not_downcast_without_explicit_dtype():
    # Regression: the seam fires only on an EXPLICIT .dtype() opt-in. A plain
    # f32-param model (e.g. a loaded f32 checkpoint) under x64 must still compute
    # in f64 by promotion — the seam must NOT silently downcast the f64 data,
    # which would partly undo the Part-A x64 propagation.
    net = _tiny_net(key_seed=5)  # params created at the default → float32
    with x64_enabled():
        net.optimizer(optax.adam(1e-3))
        dom = _line_domain()
        x, *_ = dom.variable("interior")
        u = net(x)
        crux = jno.core([u.mse])
        assert np.asarray(crux.eval(u)).dtype == np.float64


@pytest.mark.slow
def test_dtype_bf16_vs_f32_speed_informational(capsys):
    # Best-effort speed comparison. Real bf16 speedups need bf16-capable
    # hardware (GPU); on CPU bf16 is emulated and may be slower. This logs the
    # ratio and only asserts both runs produce finite losses — it is NOT a gate.
    import time

    def _run(dtype):
        dom = jno.domain(constructor=jno.domain.line(mesh_size=0.02))
        x, *_ = dom.variable("interior")
        net = _tiny_net(key_seed=7)
        if dtype is not None:
            net.dtype(dtype)
        net.optimizer(optax.adam(1e-3))
        crux = jno.core([(net(x).d(x).d(x) + 1.0).mse])
        crux.solve(3)  # warm up / compile
        t0 = time.perf_counter()
        stats = crux.solve(20)
        return time.perf_counter() - t0, float(stats.training_logs[-1]["total_loss"][-1])

    t_f32, l_f32 = _run(None)
    t_bf16, l_bf16 = _run(jnp.bfloat16)
    with capsys.disabled():
        print(f"\n[dtype speed] f32={t_f32:.3f}s  bf16={t_bf16:.3f}s  ratio={t_bf16 / t_f32:.2f}x")
    assert np.isfinite(l_f32) and np.isfinite(l_bf16)
