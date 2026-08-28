"""``peec.build()``: freeze the discretisation once, and the solve becomes jittable.

Everything structural -- which cells are metal, which nodes a pad owns, which filaments weld to
which -- is decided on the host from concrete geometry, and none of it survives a trace. So
``.solve()`` was differentiable but not jittable, and it redid that whole pass on every design
iteration. ``build()`` is the same split ``jno.precond.ams().build(fem)`` and ``FemLinearSystem``
already use: host once, then pure jax.

What has to hold is that the split changes nothing but the cost -- the built solve must agree with
the eager one to the last bit, under jit and under a gradient, and a conductivity handed in at solve
time must agree with the same value attached to the geometry.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7
LX, LY, TZ, H = 0.040, 0.020, 0.001, 0.002


def network(sigma=SIG, freq=0.0):
    d = jno.Shape.box(0, 0, 0, LX, LY, TZ, size=(H, H, TZ)).attach(sigma=sigma).name("plate").domain()
    d.tag("A", lambda x, y, z: x < 1.1 * H)
    d.tag("B", lambda x, y, z: x > LX - 1.1 * H)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq)


def wire_network(sigma=SIG):
    """Lines AND a solid, welded -- the path where the resolver order has to survive `_weld`."""
    trace = jno.Shape.box(0, 0, 0, 0.02, 0.004, 0.001, size=0.001).attach(sigma=SIG).name("trace")
    wire = (
        jno.Shape.line([(0.019, 0.002, 0.0005), (0.019, 0.002, 0.006), (0.030, 0.002, 0.0005)], r=1.9e-4, size=0.001)
        .attach(sigma=sigma)
        .name("wire")
    )
    d = (trace + wire).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: (x > 0.0295) & (z < 0.0011))
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6)


# --- the split changes nothing ------------------------------------------------------------------


@pytest.mark.parametrize("freq", [0.0, 1e6])
def test_building_first_changes_no_digit(freq):
    a, b = network(freq=freq).solve(), network(freq=freq).build().solve()
    for k in ("R", "L", "Z"):
        assert complex(getattr(a, k)) == complex(getattr(b, k))


def test_a_welded_network_builds_the_same_too():
    """Lines then solids: the resolvers must concatenate in exactly `_weld`'s order."""
    a, b = wire_network().solve(), wire_network().build().solve()
    assert complex(a.Z) == complex(b.Z)
    assert np.array_equal(np.asarray(a.i), np.asarray(b.i))


def test_the_resolver_reproduces_the_eager_conductivity():
    built = wire_network().build()
    assert np.allclose(np.asarray(built._resolve(None)), SIG, rtol=0, atol=0)


def test_solve_is_still_the_one_call_path():
    """`peec.solve()` must BE `build().solve()`, not a second implementation drifting beside it."""
    assert np.asarray(network().solve().i).shape == np.asarray(network().build().solve().i).shape


# --- and now it jits ------------------------------------------------------------------------------


def test_the_built_solve_jits():
    built = network().build()
    f = lambda s: built.solve(sigma={"plate": SIG * s}).R  # noqa: E731
    assert float(jax.jit(f)(1.0)) == pytest.approx(float(f(1.0)), rel=1e-12)


def test_the_built_solve_differentiates_under_jit():
    """R = rho*L/A is exactly inverse in sigma, so dR/ds at s=1 is -R. An oracle, not a re-run."""
    built = network().build()
    f = lambda s: built.solve(sigma={"plate": SIG * s}).R  # noqa: E731
    r = float(f(1.0))
    for g in (float(jax.grad(f)(1.0)), float(jax.jit(jax.grad(f))(1.0))):
        assert g == pytest.approx(-r, rel=1e-10)


def test_a_field_conductivity_jits_too():
    built = network().build()

    def port(t):
        fld = lambda x, y, z: SIG * (0.2 + t * x / LX)  # noqa: E731
        return built.solve(sigma={"plate": fld}).R

    assert float(jax.jit(port)(0.8)) == pytest.approx(float(port(0.8)), rel=1e-12)
    g, h = float(jax.jit(jax.grad(port))(0.8)), 1e-6
    assert g == pytest.approx(float((port(0.8 + h) - port(0.8 - h)) / (2 * h)), rel=1e-6)
    assert g < 0  # more conductivity is less resistance


def test_a_welded_network_jits():
    """The welded path carries the near-field preconditioner, which is where the cache lives."""
    built = wire_network().build()
    f = lambda s: jnp.real(built.solve(sigma={"wire": SIG * s}).Z)  # noqa: E731
    assert float(jax.jit(f)(1.0)) == pytest.approx(float(f(1.0)), rel=1e-10)


def test_two_separate_traces_do_not_leak(capfd):
    """Regression: the Krylov cache stores a CLOSURE, and one built inside a jit belongs to that jit.

    While every call rediscretised, a fresh `fil` missed the cache every time and this never fired.
    Freezing the network made the identity stable, the cache finally hit, and the second trace
    raised UnexpectedTracerError -- a bug that `build()` did not introduce so much as expose.
    """
    built = network().build()
    f = lambda s: built.solve(sigma={"plate": SIG * s}).R  # noqa: E731
    one = float(jax.jit(f)(1.0))
    two = float(jax.jit(lambda s: f(s) + 0.0)(1.0))  # a DIFFERENT trace of the same network
    assert one == pytest.approx(two, rel=1e-12)
    assert float(jax.jit(jax.grad(f))(1.0)) < 0  # and a third, under a gradient


def test_the_eager_cache_still_serves_a_frequency_sweep():
    """Skipping the cache under a trace must not disable it where it was earning its keep."""
    from jno.utils.solver.peec import _KRYLOV_CACHE

    _KRYLOV_CACHE.clear()
    wire_network().build().solve()
    assert len(_KRYLOV_CACHE) >= 1


def test_the_dissipation_readout_jits():
    """The electro-thermal handoff has to cross into a jit, or no coupled objective can be built.

    `dissipation()` decided which elements a conductor owns by summing their volumes in jnp and
    reading the total back with `float()`. That is a STRUCTURAL question -- it is answered by the
    provenance array, on the host -- and asking it in jnp made the whole readout unjittable, which
    put `jno.core` out of reach for anything thermal.
    """
    built = network().build()
    f = lambda s: built.solve(sigma={"plate": SIG * s}).dissipation()["plate"]  # noqa: E731
    r = float(f(1.0))
    assert r > 0
    assert float(jax.jit(f)(1.0)) == pytest.approx(r, rel=1e-12)
    # An exact oracle, and one worth stating because the sign is the opposite of the intuition that
    # "more conductive dissipates less". The drive is a VOLTAGE: at DC, R goes as 1/s while the
    # current it drives goes as s, so the loss I^2 R goes as s exactly -- d(loss)/ds is +loss at
    # s = 1. (At 1 MHz the same readout falls with s instead: the loop is inductance-limited, the
    # current no longer follows sigma, and the surface resistance goes as 1/sqrt(sigma).)
    for g in (float(jax.grad(f)(1.0)), float(jax.jit(jax.grad(f))(1.0))):
        assert g == pytest.approx(r, rel=1e-9)


def test_dissipation_still_reconciles_with_the_total():
    """The host-side ownership must select the same elements the jnp mask did: sum(q_r V_r) = joule."""
    sol = wire_network().build().solve()
    q, vol, own = sol.dissipation(), np.asarray(sol._vol), np.asarray(sol._owner)
    total = sum(float(jnp.real(v)) * vol[own == k].sum() for k, (_n, v) in enumerate(q.items()))
    assert total == pytest.approx(float(sol.joule), rel=1e-10)


# --- the conductivity override --------------------------------------------------------------------


def test_an_override_matches_attaching_the_same_value():
    got = network().build().solve(sigma={"plate": SIG / 3})
    ref = network(sigma=SIG / 3).solve()
    assert complex(got.R) == pytest.approx(complex(ref.R), rel=1e-12)


def test_overriding_only_one_conductor_leaves_the_other_alone():
    a = wire_network().build().solve(sigma={"wire": SIG})
    assert complex(a.Z) == pytest.approx(complex(wire_network().solve().Z), rel=1e-12)
    b = wire_network().build().solve(sigma={"wire": SIG / 10})
    assert jnp.real(b.Z) > jnp.real(a.Z)  # a worse wire, in series, is more resistance


def test_an_unknown_conductor_in_the_override_is_refused():
    built = network().build()
    with pytest.raises(ValueError, match="names no conductor"):
        built.solve(sigma={"Plate": SIG})  # a terminal, a typo -- either way, loudly
    with pytest.raises(ValueError, match="names no conductor"):
        built.solve(sigma={"A": SIG})  # 'A' is a TERMINAL, not a conductor


# --- the reason it exists -------------------------------------------------------------------------


def test_building_once_beats_rediscretising(monkeypatch):
    """The host pass must actually be skipped -- counted, not timed, so it cannot be flaky."""
    import sys

    mod = sys.modules["jno.peec"]  # `jno.peec` the NAME is the function; the module is behind it

    calls = []
    orig = mod.PEEC._discretise
    monkeypatch.setattr(mod.PEEC, "_discretise", lambda self: (calls.append(1), orig(self))[1])

    built = network().build()
    for k in range(5):
        built.solve(sigma={"plate": SIG * (1.0 + 0.01 * k)})
    assert len(calls) == 1  # five solves, ONE discretisation

    calls.clear()
    for _ in range(3):
        network().solve()
    assert len(calls) == 3  # the un-built path still pays it every time
