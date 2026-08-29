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


def device_network(rdev=5e-3):
    """Two collinear wires with a gap: nothing conducts across it but the device."""
    ell, rad, gap = 0.05, 5e-4, 0.004
    lo = jno.Shape.line([(0, 0, 0), (0, 0, ell)], r=rad, size=ell / 10).attach(sigma=SIG).name("lo")
    hi = jno.Shape.line([(0, 0, ell + gap), (0, 0, 2 * ell + gap)], r=rad, size=ell / 10).attach(sigma=SIG).name("hi")
    pads = (
        jno.Shape.sphere(0, 0, 0.0, 2 * rad).name("A")
        + jno.Shape.sphere(0, 0, ell, 2 * rad).name("M")
        + jno.Shape.sphere(0, 0, ell + gap, 2 * rad).name("N")
        + jno.Shape.sphere(0, 0, 2 * ell + gap, 2 * rad).name("B")
    )
    d = (lo + hi + pads).domain()
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0, v(*at("M")) - v(*at("N")) - rdev * i(*at("M"))], freq=0.0)


# --- a device impedance that depends on the solved state -------------------------------------------


def test_a_device_impedance_can_be_overridden_at_solve():
    """A SiC die's R_ds(on) rises ~0.5 %/K, so an electro-thermal loop re-impresses it every pass.

    It cannot be a constant in the constraint list, and rebuilding the network per pass throws away
    the discretisation. So it is handed in at solve, exactly like `sigma`.
    """
    built = device_network().build()
    ref = complex(built.solve().Z)
    assert complex(built.solve(devices={}).Z) == ref  # an empty override changes nothing
    for spelling in (5e-3, jnp.asarray(5e-3), complex(5e-3)):  # scalar, array, complex all land
        assert complex(built.solve(devices={"M": spelling}).Z) == pytest.approx(ref, rel=1e-12)


def test_a_bigger_device_impedance_shows_up_at_the_port():
    """Series: the device's own resistance adds to the loop, so the port must see it."""
    built = device_network().build()
    lo = jnp.real(built.solve(devices={"M": 1e-3}).Z)
    hi = jnp.real(built.solve(devices={"M": 50e-3}).Z)
    assert float(hi - lo) == pytest.approx(49e-3, rel=1e-6)  # exactly the difference, nothing else


def test_re_solving_with_a_different_device_value_is_not_the_cached_operator():
    """Regression: the compiled Krylov solve captures the CONSTRAINT MATRIX, and a device impedance
    lives in it. Keyed only on the network, a second solve at a different Z reused the first call's
    operator -- surfacing as a solve that would not converge rather than as a wrong number, and only
    because the residual check was there. Invisible until `.build()` made the network identity stable.
    """
    built = device_network().build()
    a = float(jnp.real(built.solve(devices={"M": 1e-3}).Z))
    b = float(jnp.real(built.solve(devices={"M": 50e-3}).Z))
    c = float(jnp.real(built.solve(devices={"M": 1e-3}).Z))  # back again: must not be b's operator
    assert a == pytest.approx(c, rel=1e-12)
    assert b - a == pytest.approx(49e-3, rel=1e-6)


def test_an_unknown_device_terminal_is_refused():
    built = device_network().build()
    with pytest.raises(ValueError, match="names no device of this network"):
        built.solve(devices={"A": 1e-3})


def test_a_gradient_does_not_poison_the_cache_for_later_calls():
    """Regression: the compiled-solve cache must not keep a closure built under ANY transform.

    The first guard probed a bare `jnp.zeros(())`, which answers only half the question. Under jit
    everything stages and the probe is a tracer, so the guard fired. Under `jax.grad` nothing stages
    except what DEPENDS on the differentiated input -- the probe stays concrete while the element
    impedance does not -- so the guard missed, a closure holding LinearizeTracers was cached, and the
    next EAGER call reusing it raised UnexpectedTracerError from somewhere else entirely.

    Worse than the crash: when it did not crash it returned a stale operator, and the gradient came
    back 14 % wrong while still looking perfectly reasonable.
    """
    built = device_network().build()
    f = lambda z: jnp.real(built.solve(devices={"M": z}).Z)  # noqa: E731
    before = float(f(5e-3))
    g = float(jax.grad(f)(5e-3))
    after = float(f(5e-3))  # the same eager call, AFTER a gradient has run through the same network
    assert after == pytest.approx(before, rel=1e-12)
    # a series device: dR_port/dZ is exactly 1, which is an oracle rather than a re-run
    assert g == pytest.approx(1.0, rel=1e-6)


def test_the_cache_still_serves_repeated_identical_solves():
    """The guard must not disable the cache where it earns its keep: same network, same values."""
    from jno.utils.solver.peec import _KRYLOV_CACHE

    built = wire_network().build()
    _KRYLOV_CACHE.clear()
    a = complex(built.solve().Z)
    n_after_first = len(_KRYLOV_CACHE)
    b = complex(built.solve().Z)
    assert a == b
    assert n_after_first >= 1 and len(_KRYLOV_CACHE) == n_after_first  # hit, not a second entry


# --- the dense partial-inductance matrix is never formed to read an inductance ----------------------


def test_L_agrees_with_the_matrix_it_no_longer_forms():
    """The quadratic form and the matrix must give the same number, or the saving is a lie."""
    sol = network(freq=1e6).build().solve()
    cur = jnp.atleast_2d(sol.i)[0]
    ref = jnp.real(jnp.conj(cur) @ sol.partial.astype(cur.dtype) @ cur) / abs(complex(sol._port_current)) ** 2
    assert float(sol.L) == pytest.approx(float(ref), rel=1e-10)


def test_reading_L_does_not_build_an_n_by_n_array(monkeypatch):
    """Counted, not timed. `pair_matrix` is the one object a partial-element solve exists to avoid:
    it is quadratic in the SUB-POINT count, so three Gauss points make it nine times worse than the
    element count suggests. On the example module at a 2 mm pitch it was 3.7 GB of a 5.3 GB peak,
    against 0.84 GB for the solve itself.
    """
    import jno.utils.solver.kernel as K

    calls = []
    orig = K.pair_matrix
    monkeypatch.setattr(K, "pair_matrix", lambda *a, **k: (calls.append(1), orig(*a, **k))[1])

    # a PLAIN lattice: the welded path legitimately forms this for its thin wire sub-block, which is
    # a few hundred filaments against the lattice's thousands and is the point of the near/far split.
    sol = network().build().solve()
    assert len(calls) == 0, "solving formed the dense matrix"
    _ = float(sol.L)
    assert len(calls) == 0, "reading L formed the dense matrix"
    _ = sol.partial  # asking for the matrix itself is the one thing that may build it
    assert len(calls) == 1


def test_L_is_still_the_hermitian_form_at_frequency():
    """Re(I)'Lp Re(I) + Im(I)'Lp Im(I) is I^H Lp I exactly, and must stay positive once phases spread."""
    for freq in (0.0, 1e6, 1e7):
        assert float(network(freq=freq).build().solve().L) > 0
    assert float(wire_network().build().solve().L) > 0  # and on a WELDED network too


# --- the solver knobs the failure message names --------------------------------------------------


def test_the_dense_path_and_the_matrix_free_one_agree():
    """`matrix_free=False` is the escape hatch, so it has to give the same answer, not merely run."""
    built = wire_network().build()
    auto = complex(built.solve().Z)
    dense = complex(built.solve(matrix_free=False).Z)
    assert dense == pytest.approx(auto, rel=1e-9)


def test_a_deeper_restart_does_not_change_a_converged_answer():
    """It buys convergence at scale; where the default already converges it must be a no-op."""
    built = wire_network().build()
    base = complex(built.solve().Z)
    for r in (24, 48):
        assert complex(built.solve(restart=r).Z) == pytest.approx(base, rel=1e-8)


def test_the_non_convergence_message_names_a_reachable_knob():
    """It used to point at `matrix_free=False`, which no caller could reach from the front door.

    Measured on the example module at 21,980 bars: the default restart of 16 leaves 9.3e-06 where
    1e-6 is wanted, and 48 converges in 176 s against the 290 s the shallower one spends failing.
    """
    from jno.utils.solver.peec import _converged

    with pytest.raises(ValueError) as e:
        _converged(1e-3, 1e-6)
    msg = str(e.value)
    assert "restart=48" in msg  # the knob, and a value that is known to work
    assert "matrix_free=False" in msg
