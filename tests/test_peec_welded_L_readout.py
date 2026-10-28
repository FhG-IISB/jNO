"""`sol.L` of a WELDED network reuses the operator the solve built, rather than building it again.

The energy form `I^H Lp I / |I_port|^2` needs `Lp` applied once more, and on a welded network that
apply holds the dense wire-lattice cross block, which the readout used to BUILD AGAIN after the solve had
just built it. On a 54,866-element power module that second build was 13-14 s of a 48 s forward
solve -- paid again by every reverse-mode pass, since the readout sits in the differentiated graph.

The solve now hands its apply to the solution (`last_lp_apply`). These tests pin that the readout is
still the energy form and that the operator is built once.
"""

import jax
import numpy as np
import pytest

import jno
import jno.utils.solver.peec as PS
from jno.utils.solver.peec import welded_apply


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


MM, SIG, MU0 = 1e-3, 5.8e7, 4e-7 * np.pi


def _welded(freq=1e6):
    trace = jno.shape.box(0, 0, 0, 20 * MM, 4 * MM, 0.5 * MM, size=(1 * MM, 1 * MM, 0.5 * MM)).attach(sigma=SIG).name("t")
    arch = [
        (4 * MM, 2 * MM, 0.5 * MM),
        (4 * MM, 2 * MM, 1.5 * MM),
        (16 * MM, 2 * MM, 1.5 * MM),
        (16 * MM, 2 * MM, 0.5 * MM),
    ]
    wire = jno.shape.line(arch, r=0.15 * MM, size=1 * MM).attach(sigma=SIG).name("w")
    d = (trace + wire).domain()
    d.tag("A", lambda x, y, z: x < 1.01 * MM)
    d.tag("B", lambda x, y, z: x > 18.99 * MM)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=freq)


def test_L_is_the_energy_form_on_a_fresh_operator():
    b = _welded()
    sol = b.solve()
    cur = np.asarray(sol.i)
    ap = welded_apply(sol._fil, lambda r: 1.0 / r, mu_scale=MU0 / (4 * np.pi))
    energy = (np.vdot(cur.real, np.asarray(ap(cur.real))) + np.vdot(cur.imag, np.asarray(ap(cur.imag)))) / abs(
        complex(sol._port_current)
    ) ** 2
    assert float(sol.L) == pytest.approx(float(energy), rel=1e-6)


def test_reading_L_does_not_rebuild_the_welded_operator(monkeypatch):
    b = _welded()
    calls = []
    real = PS.welded_apply

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(PS, "welded_apply", counting)
    sol = b.solve()
    built_by_solve = len(calls)
    _ = float(sol.L)
    assert built_by_solve >= 1
    assert len(calls) == built_by_solve  # the readout added none
