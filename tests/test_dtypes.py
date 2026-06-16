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

import jax
import jax.numpy as jnp
import numpy as np

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
