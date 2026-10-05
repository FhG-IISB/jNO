"""Electro-thermal coupling written with jno.peec, jno.fem and jno.core alone.

    d.attach(sigma=sigma20 / (1 + alpha (T - T0)), k=...)     # sigma depends on the FEM field T
    em = jno.peec([...]);  heat = jno.fem([... - em.loss * s, ...])
    jno.core([em, heat]).solve()

Oracles: with alpha = 0 the coupling is one-way, so it must reproduce the network solved alone and the
heat problem solved with that network's loss attached by hand; with alpha > 0 the converged state is a
fixed point (one more pass changes nothing) and the resistance rises; with a trainable parameter the
gradient through the converged fixed point matches a central difference.
"""

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

SIG20, K_CU = 5.8e7, 400.0


def _domain():
    trace = lambda x0, x1: jno.shape.box(x0, 0, 1e-3, x1, 4e-3, 1.5e-3, size=1e-3)
    d = (trace(0, 8e-3).name("A") + trace(8e-3, 16e-3).name("B")).domain()
    d.tag("P", lambda x, y, z: x < 1.1e-3)
    d.tag("N", lambda x, y, z: x > 14.9e-3)
    return d


def _pad(d, t):
    return d.variable(t, split=True, sample=(4, None))[:3]


def _problems(d, sigma):
    T, s = d.fem_symbols()
    d.attach(sigma=sigma(T), k=K_CU)
    i, v = d.peec_symbols()
    em = jno.peec([v(*_pad(d, "P")) - v(*_pad(d, "N")) - 0.05], freq=1e3)
    x = d.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=x[0], y=x[1], z=x[2]), s.bind(x=x[0], y=x[1], z=x[2])
    heat = jno.fem([d.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - em.loss * si, T(*_pad(d, "N")) - 300.0])
    return em, heat, T


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn()


def test_one_way_coupling_reproduces_the_two_solves_done_separately():
    """alpha = 0: the network does not see T, so the coupled R is the network's own, and T is the heat
    problem with that network's loss attached by hand."""
    d = _domain()
    em, heat, _T = _problems(d, lambda T: SIG20 + 0.0 * T)
    sol = _quiet(lambda: jno.core([em, heat]).solve())

    d2 = _domain()
    d2.attach(sigma=SIG20, k=K_CU)
    i2, v2 = d2.peec_symbols()
    alone = _quiet(lambda: jno.peec([v2(*_pad(d2, "P")) - v2(*_pad(d2, "N")) - 0.05], freq=1e3).solve())
    assert float(sol.em.R) == pytest.approx(float(alone.R), rel=1e-10)

    for name, q in alone.dissipation().items():
        d2.attach(name, Q=float(np.real(q)))
    T2, s2 = d2.fem_symbols()
    x = d2.variable("interior", split=True)[:3]
    Ti, si = T2.bind(x=x[0], y=x[1], z=x[2]), s2.bind(x=x[0], y=x[1], z=x[2])
    by_hand = np.asarray(
        _quiet(
            lambda: jno.fem(
                [d2.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - d2.Q * si, T2(*_pad(d2, "N")) - 300.0]
            ).solve()
        )
    ).reshape(-1)
    # the hand-written solve runs the default iterative solver (relative tolerance 1e-8 on the residual),
    # the coupled one a direct solve, so they agree to the iterative solver's accuracy, not to round-off
    assert np.max(np.abs(sol.field - by_hand)) < 1e-6 * np.max(np.abs(by_hand))
    assert sol.field.max() > 300.0 + 1e-3, "the loss must actually heat the conductor"


def test_a_temperature_dependent_conductivity_converges_to_a_fixed_point_and_raises_R():
    alpha = 3.93e-3
    d = _domain()
    em, heat, _T = _problems(d, lambda T: SIG20 / (1 + alpha * (T - 293.15)))
    sol = _quiet(lambda: jno.core([em, heat]).solve())
    assert sol.change < 1e-8
    d0 = _domain()
    em0, heat0, _ = _problems(d0, lambda T: SIG20 / (1 + alpha * (300.0 - 293.15)) + 0.0 * T)
    cold = _quiet(lambda: jno.core([em0, heat0]).solve())
    # `cold` holds every element at the sink temperature; the coupled conductor is hotter than that
    # everywhere except at the sink itself, so it is strictly more resistive
    assert float(sol.em.R) > float(cold.em.R), "a hotter conductor must be more resistive"
    # and the fixed point is self-consistent: one more pass from it changes nothing
    assert sol.iterations < 50


def test_the_gradient_through_the_coupled_fixed_point_matches_a_central_difference():
    alpha = 3.93e-3
    d = _domain()
    scale = jno.np.parameter((1,), name="scale")
    scale.initialize(jax.nn.initializers.constant(1.0))
    em, heat, _T = _problems(d, lambda T: scale * SIG20 / (1 + alpha * (T - 293.15)))
    node = _quiet(lambda: jno.core([em, heat]).solve())
    f = lambda a: jnp.max(node.fn(jnp.asarray([a])))  # noqa: E731
    g = float(jax.grad(f)(1.0))
    h = 1e-4
    fd = float((f(1.0 + h) - f(1.0 - h)) / (2 * h))
    assert g > 0, "at a fixed voltage a more conductive conductor carries more current and runs hotter"
    assert g == pytest.approx(fd, rel=1e-5)


def test_a_field_dependent_network_alone_says_what_it_needs():
    d = _domain()
    em, _heat, _T = _problems(d, lambda T: SIG20 / (1 + 3.93e-3 * (T - 293.15)))
    with pytest.raises(ValueError, match=r"jno.core\(\[em, heat\]\)"):
        _quiet(lambda: em.solve())


def test_a_heat_problem_without_the_loss_is_not_coupled():
    d = _domain()
    T, s = d.fem_symbols()
    d.attach(sigma=SIG20, k=K_CU)
    i, v = d.peec_symbols()
    em = jno.peec([v(*_pad(d, "P")) - v(*_pad(d, "N")) - 0.05], freq=1e3)
    x = d.variable("interior", split=True)[:3]
    Ti, si = T.bind(x=x[0], y=x[1], z=x[2]), s.bind(x=x[0], y=x[1], z=x[2])
    heat = jno.fem([d.k * (Ti.x * si.x + Ti.y * si.y + Ti.z * si.z) - 1.0 * si, T(*_pad(d, "N")) - 300.0])
    with pytest.raises(ValueError, match="em.loss"):
        jno.core([em, heat])


def test_a_default_only_attachment_reads_back_as_a_coefficient():
    d = _domain()
    d.attach(k=K_CU)
    assert float(np.asarray(d.k)) == K_CU


def test_a_one_element_parameter_is_one_conductivity_for_the_whole_conductor():
    from jno.utils.solver.peec import resolve_sigma

    out = resolve_sigma(jnp.asarray([2.0]), np.zeros((5, 3)), "c")
    assert np.allclose(np.asarray(out), 2.0) and out.shape == (5,)
