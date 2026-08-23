"""Smoothed-Heaviside projection — Wang, Lazarov & Sigmund, *Struct. Multidisc. Optim.* **43**(6)
2011, 767-784, eq. (1).

The oracles here are algebraic identities of the map, not tolerances: the denominator is the
numerator at ``rho = 1``, so ``H(0) = 0`` and ``H(1) = 1`` hold for EVERY beta and the volume
constraint keeps its units; ``eta`` is the exact fixed point; and the map is monotone onto ``[0, 1]``.

The test that matters most is none of those. It is
``test_the_sharpness_reaches_a_solve_through_the_trace``: the reason this is a node rather than a
``constrain()`` transform is that a transform's ``beta`` is a trace-time constant, so ramping it
changes nothing the step-program key can see and the gradient keeps being taken at the original
value while the log reports the new one. That failure is silent, and this asserts it cannot recur at
the API level.

What is deliberately NOT asserted: which designs a projected run converges to, and any grey-level
value. Those belong to the runs.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.trace import heaviside


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _oracle(rho, beta, eta=0.5):
    """eq. (1), transcribed one scalar at a time with the stdlib.

    Deliberately slow and deliberately separate: a shared helper would let one bug hide in both, and
    a vectorised oracle would share exactly the broadcasting the implementation could get wrong.
    """
    tb = math.tanh(beta * eta)
    den = tb + math.tanh(beta * (1.0 - eta))
    return [(tb + math.tanh(beta * (r - eta))) / den for r in rho]


class TestTheMap:
    """Identities that hold for every beta, so they are assertions and not tolerances."""

    @pytest.mark.parametrize("beta", [0.5, 1.0, 4.0, 16.0, 64.0])
    def test_it_matches_a_scalar_transcription(self, beta):
        rho = [0.0, 0.1, 0.37, 0.5, 0.63, 0.9, 1.0]
        got = np.asarray(heaviside(jnp.asarray(rho), beta))
        want = np.asarray(_oracle(rho, beta))
        assert np.allclose(got, want, rtol=0, atol=1e-14), f"beta={beta}: {got} vs {want}"

    @pytest.mark.parametrize("beta", [0.5, 1.0, 4.0, 16.0, 64.0])
    def test_the_endpoints_are_exact(self, beta):
        """Not approximately: the volume constraint's units depend on it."""
        assert float(heaviside(jnp.asarray(0.0), beta)) == 0.0, "H(0) must be exactly 0"
        assert float(heaviside(jnp.asarray(1.0), beta)) == 1.0, "H(1) must be exactly 1"

    def test_the_symmetric_threshold_is_a_fixed_point(self):
        """Only at eta = 0.5, and the general case is asserted separately below.

        H(eta) = tanh(b*eta) / (tanh(b*eta) + tanh(b*(1-eta))), which collapses to 1/2 exactly when
        eta = 1/2 and does NOT equal eta otherwise -- an asymmetric threshold moves its own image.
        Worth pinning, because "eta is the fixed point" is the natural thing to assume and it is only
        true for the default.
        """
        for beta in (1.0, 8.0, 32.0):
            got = float(heaviside(jnp.asarray(0.5), beta, 0.5))
            assert abs(got - 0.5) < 1e-15, f"beta={beta} moved the symmetric threshold to {got}"

    @pytest.mark.parametrize("eta", [0.3, 0.7])
    def test_an_asymmetric_threshold_maps_to_the_closed_form(self, eta):
        """The general statement, so the limit above is recorded rather than merely avoided."""
        for beta in (1.0, 8.0):
            got = float(heaviside(jnp.asarray(eta), beta, eta))
            want = math.tanh(beta * eta) / (math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta)))
            assert abs(got - want) < 1e-14, f"eta={eta}, beta={beta}: {got} vs {want}"
            assert abs(got - eta) > 1e-6, (
                f"eta={eta} came back as its own image, which only eta=0.5 should do"
            )

    @pytest.mark.parametrize("beta", [1.0, 4.0, 16.0])
    def test_it_is_monotone_and_stays_in_the_unit_interval(self, beta):
        r = np.linspace(0.0, 1.0, 101)
        v = np.asarray(heaviside(jnp.asarray(r), beta))
        assert np.all(np.diff(v) >= -1e-15), "the projection must not fold"
        assert v.min() >= 0.0 and v.max() <= 1.0, f"left [0,1]: [{v.min()}, {v.max()}]"

    def test_it_sharpens_with_beta(self):
        """The whole reason for the ramp: more beta, less grey."""
        r = np.linspace(0.0, 1.0, 201)
        m_nd = [float(np.mean(4.0 * (v := np.asarray(heaviside(jnp.asarray(r), b))) * (1.0 - v)))
                for b in (1.0, 2.0, 4.0, 8.0, 16.0)]
        assert all(a > b for a, b in zip(m_nd, m_nd[1:])), (
            f"the grey-level indicator must fall monotonically in beta; got {m_nd}"
        )

    def test_the_gradient_survives_a_sharp_projection(self):
        """A naive step would give zero gradient away from eta and the optimiser would stall."""
        g = float(jax.grad(lambda r: heaviside(r, 16.0))(jnp.asarray(0.2)))
        assert math.isfinite(g) and g > 0.0, f"d/drho must be finite and positive, got {g}"

    def test_a_threshold_outside_the_unit_interval_is_refused(self):
        d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.5).domain()
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name="rho")
        with pytest.raises(ValueError, match="eta"):
            rho.project(4.0, eta=0.0)


class TestItComposes:
    """The composition rule: non-local maps reparameterise, pointwise maps compose."""

    def test_it_chains_after_the_filter(self):
        d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.4).domain()
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name="rho")
        rho.dtype(jnp.float64)
        rho.initialize(lambda k, sh, dtype=None: jnp.full(sh, 0.5))
        node = rho.patch().project(8.0)
        crux = jno.core([node.sum.name("S")], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
        v = float(np.asarray(crux.eval([node.sum])).reshape(-1)[0])
        assert math.isfinite(v) and v > 0.0, f"filter then projection must evaluate; got {v}"

    def test_the_sharpness_reaches_a_solve_through_the_trace(self):
        """The one that would have caught the staleness bug at the API level.

        `beta` is a parameter NODE, so it is a traced argument of the compiled program. Changing it
        must change what a solve computes. A `constrain()` transform reading a Python float cannot
        do this -- that is the whole reason `project` is a node.
        """
        d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.4).domain()
        _r, s = d.fem_symbols(space="P0", names=("r", "s"))
        rho = jno.np.parameter(s, name="rho")
        rho.dtype(jnp.float64)
        # a deliberately GREY field: projection is the identity on 0 and 1, so a binary design
        # would move under no beta at all and the test would pass while asserting nothing.
        rho.initialize(lambda k, sh, dtype=None: jnp.full(sh, 0.3))

        beta = jno.np.parameter((1,), name="beta")
        beta.dtype(jnp.float64)
        beta.initialize(lambda k, sh, dtype=None: jnp.full(sh, 1.0))

        total = rho.project(beta[0]).sum.name("S")
        crux = jno.core([total], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
        at_1 = float(np.asarray(crux.eval([total])).reshape(-1)[0])

        beta.initialize(lambda k, sh, dtype=None: jnp.full(sh, 16.0))
        crux2 = jno.core([total], domain=jno.domain.from_array({"_": np.zeros((1, 1))}))
        at_16 = float(np.asarray(crux2.eval([total])).reshape(-1)[0])

        assert at_16 < at_1, (
            f"projecting rho = 0.3 harder must push it further towards 0, so the total must fall: "
            f"beta=1 gave {at_1}, beta=16 gave {at_16}. Equal values mean beta never reached the "
            f"computation."
        )
