"""Ramping a projection sharpness, chunk-independently.

The headline assertion is ``test_a_chunked_run_reaches_the_same_sharpness``. ``epoch`` is
per-``solve()``-call, so a schedule keyed on it ramps once per CHUNK and the physics then depends on
how the driver loop happened to be written -- measured on a 3-D bracket, the same 250-iteration run
reached ``beta`` 10.8 at ``CHUNK=10`` and 1.6 at ``CHUNK=50``, with ``M_nd`` 0.008 against 0.078.
Two different optimisations wearing the same configuration. Counting the callback's own invocations
is chunk-independent by construction, and that is what this pins.

What is deliberately NOT asserted: that any particular ramp length produces a good design. ``over``
is the caller's declaration and this only checks it is honoured.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _ramped(**kw):
    """A scalar parameter driven only by the ramp, so its value IS the schedule."""
    d = jno.domain.from_array({"_": np.zeros((1, 1))})
    beta = jno.np.parameter((1,), name="beta")
    beta.dtype(jnp.float64)
    beta.initialize(lambda k, sh, dtype=None, _s=kw.get("start", 1.0): jnp.full(sh, _s))
    beta.optimizer(optax.sgd(1.0))
    hv = jno.optimizers.heaviside_continuation(beta, **kw)
    crux = jno.core([(beta[0] * 0.0).name("noop")], domain=d)
    return crux, beta, hv


def _value(crux, beta):
    return float(np.asarray(crux.eval([beta])).reshape(-1)[0])


class TestChunkIndependence:
    """The property the whole design exists for."""

    def test_a_chunked_run_reaches_the_same_sharpness(self):
        c1, b1, h1 = _ramped(maximum=16.0, over=10)
        for _ in range(10):
            c1.solve(1, callbacks=[h1])
        c2, b2, h2 = _ramped(maximum=16.0, over=10)
        c2.solve(10, callbacks=[h2])
        assert h1.value == pytest.approx(h2.value, rel=1e-12), (
            f"ten solve(1) calls reached beta {h1.value}, one solve(10) reached {h2.value}; the "
            f"ramp must not depend on how the driver chunks its loop"
        )
        assert _value(c1, b1) == pytest.approx(_value(c2, b2), rel=1e-9), (
            "the parameter itself must agree, not merely the callback's bookkeeping"
        )

    def test_the_parameter_follows_the_callback(self):
        """The hook writes the parameter, so the two must not drift apart."""
        crux, beta, hv = _ramped(maximum=8.0, over=4)
        for _ in range(4):
            crux.solve(1, callbacks=[hv])
        assert _value(crux, beta) == pytest.approx(hv.value, rel=1e-9), (
            f"callback says {hv.value}, parameter reads {_value(crux, beta)}"
        )


class TestTheSchedule:
    """``over`` is a declaration and must be honoured exactly."""

    def test_it_reaches_the_maximum_at_the_declared_length(self):
        crux, _b, hv = _ramped(maximum=16.0, over=8)
        for _ in range(8):
            crux.solve(1, callbacks=[hv])
        assert hv.value == pytest.approx(16.0, rel=1e-12), f"after over=8 steps beta is {hv.value}"
        assert hv.saturated, "progress must report saturation once the ramp is done"

    def test_it_holds_at_the_maximum_afterwards(self):
        crux, _b, hv = _ramped(maximum=16.0, over=4)
        for _ in range(12):
            crux.solve(1, callbacks=[hv])
        assert hv.value == pytest.approx(16.0, rel=1e-12), "beta must clamp, not keep climbing"

    def test_hold_delays_the_ramp(self):
        """The topology should be allowed to form before the projection bites."""
        crux, _b, hv = _ramped(maximum=16.0, over=4, hold=3)
        for _ in range(3):
            crux.solve(1, callbacks=[hv])
        assert hv.value == pytest.approx(1.0, rel=1e-12), f"held steps must not ramp; got {hv.value}"
        for _ in range(4):
            crux.solve(1, callbacks=[hv])
        assert hv.value == pytest.approx(16.0, rel=1e-12), f"after hold+over: {hv.value}"

    def test_a_linear_schedule_is_linear(self):
        crux, _b, hv = _ramped(start=0.0, maximum=8.0, over=4, schedule="linear")
        seen = []
        for _ in range(4):
            crux.solve(1, callbacks=[hv])
            seen.append(hv.value)
        assert seen == pytest.approx([2.0, 4.0, 6.0, 8.0], rel=1e-12), seen

    def test_history_records_one_entry_per_invocation(self):
        crux, _b, hv = _ramped(maximum=4.0, over=5)
        for _ in range(5):
            crux.solve(1, callbacks=[hv])
        assert len(hv.history) == 5, f"expected 5 entries, got {len(hv.history)}"


class TestItRefusesBadSchedules:
    """Fail loudly on a schedule that cannot mean what it says."""

    def test_a_missing_length_is_refused(self):
        beta = jno.np.parameter((1,), name="beta")
        with pytest.raises(TypeError):
            jno.optimizers.heaviside_continuation(beta, maximum=16.0)

    def test_a_non_positive_length_is_refused(self):
        beta = jno.np.parameter((1,), name="beta")
        with pytest.raises(ValueError, match="over"):
            jno.optimizers.heaviside_continuation(beta, maximum=16.0, over=0)

    def test_a_maximum_below_the_start_is_refused(self):
        beta = jno.np.parameter((1,), name="beta")
        with pytest.raises(ValueError, match="below start"):
            jno.optimizers.heaviside_continuation(beta, start=8.0, maximum=2.0, over=10)

    def test_a_geometric_ramp_from_zero_is_refused(self):
        """It multiplies, so it can never leave zero -- silently flat rather than wrong."""
        beta = jno.np.parameter((1,), name="beta")
        with pytest.raises(ValueError, match="start > 0"):
            jno.optimizers.heaviside_continuation(beta, start=0.0, maximum=16.0, over=10)
