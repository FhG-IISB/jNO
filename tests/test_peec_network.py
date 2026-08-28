"""The PEEC circuit solve: Kirchhoff over partial elements.

Every case here has an oracle that does not go through :func:`network_impedance`:

  series      the topology forces one current, so Z = sum(R) + j w sum(Lp) in closed form
  parallel    at DC the split is the conductance ratio, exactly
  crossover   at DC the conductance ratio again; at high frequency the split that minimises the
              magnetic energy, from the reduced 2x2 branch inductance
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import pair_matrix
from jno.utils.solver.peec import line_filaments, network_impedance

jax.config.update("jax_enable_x64", True)

MU0 = 4e-7 * np.pi
SIG = 5.8e7  # copper


def partials(f):
    return np.asarray(pair_matrix(f.pos, f.mom, lambda r: 1.0 / r, f.self_g, group=f.group)) * MU0 / (4 * np.pi)


@pytest.mark.parametrize("freq", [0.0, 1e6])
def test_a_series_chain_is_exactly_R_plus_jwL(freq):
    """The topology forces one current, so Z is the element impedance plus the loop inductance.

    The element impedance is the SURFACE one, which is the DC value at zero frequency and larger
    above it -- the current retreats towards the surface, and that is the whole reason a conductor
    need not be split across its section.
    """
    from jno.utils.solver.kernel import internal_impedance

    ell, a = 0.050, 5e-4
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, ell)], r=a), size=ell / 10, quad=3)
    w = 2 * np.pi * freq
    z, _ = network_impedance(f, SIG, ((0, 0, 0), (0, 0, ell)), omega=w)
    zint = complex(np.sum(np.asarray(internal_impedance(f.length, f.area, f.skin, f.round_, w, SIG))))
    ref = zint + 1j * w * partials(f).sum()
    assert abs(complex(z) - ref) / abs(ref) < 1e-12
    if freq == 0.0:
        assert abs(zint.real / (ell / (SIG * np.pi * a**2)) - 1) < 1e-12  # and at DC it IS rho l / A


def test_a_series_chain_carries_one_current():
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0.01, 0), (0.02, 0.01, 0)], r=3e-4), size=0.002)
    _, cur = network_impedance(f, SIG, ((0, 0, 0), (0.02, 0.01, 0)), omega=2 * np.pi * 1e6)
    cur = np.asarray(cur)
    assert np.allclose(cur, cur[0])  # a bend is not a branch


def test_parallel_branches_split_by_conductance_at_dc():
    r1, r2 = 4e-4, 2e-4
    path = lambda s, r: jno.Shape.line([(0, 0, 0), (0, s, 0), (0.05, s, 0), (0.05, 0, 0)], r=r)
    f = line_filaments([path(0.02, r1), path(-0.02, r2)], size=0.004)
    z, cur = network_impedance(f, SIG, ((0, 0, 0), (0.05, 0, 0)), omega=0.0)

    ell = 0.02 + 0.05 + 0.02
    r_of = lambda r: ell / (SIG * np.pi * r**2)
    assert abs(complex(z).real / (1 / (1 / r_of(r1) + 1 / r_of(r2))) - 1) < 1e-12

    thick = np.asarray(f.area) > np.pi * 3e-4**2
    i1, i2 = abs(np.asarray(cur)[thick][0]), abs(np.asarray(cur)[~thick][0])
    assert abs(i1 / (i1 + i2) - r1**2 / (r1**2 + r2**2)) < 1e-12  # conductance ~ area


def test_current_abandons_the_low_resistance_path_for_the_low_inductance_one():
    """The reason to run PEEC at all: the DC answer is the wrong answer at switching frequencies."""
    direct = jno.Shape.line([(0, 0, 0), (0.05, 0, 0)], r=1.5e-4)  # short, thin  -> high R, low L
    detour = jno.Shape.line([(0, 0, 0), (0, -0.04, 0), (0.05, -0.04, 0), (0.05, 0, 0)], r=6e-4)
    f = line_filaments([direct, detour], size=0.004)
    thin = np.asarray(f.area) < np.pi * 3e-4**2
    share = lambda c: abs(np.asarray(c)[thin][0]) / (abs(np.asarray(c)[thin][0]) + abs(np.asarray(c)[~thin][0]))

    ell = np.asarray(f.length)
    rb = np.array([ell[thin].sum() / (SIG * np.pi * 1.5e-4**2), ell[~thin].sum() / (SIG * np.pi * 6e-4**2)])
    assert rb[0] / rb[1] > 5  # the direct path really is the resistive one

    _, dc = network_impedance(f, SIG, ((0, 0, 0), (0.05, 0, 0)), omega=0.0)
    assert abs(share(dc) - (1 / rb[0]) / (1 / rb).sum()) < 1e-12

    # at high frequency the split is the one that minimises I' Lb I subject to sum(I) = 1
    lp = partials(f)
    lb = np.array([[lp[np.ix_(m, n)].sum() for n in (thin, ~thin)] for m in (thin, ~thin)])
    x = np.linalg.solve(lb, np.ones(2))
    _, hf = network_impedance(f, SIG, ((0, 0, 0), (0.05, 0, 0)), omega=2 * np.pi * 1e8)
    assert abs(share(hf) - x[0] / x.sum()) < 2e-3
    assert share(hf) > 4 * share(dc)  # and it is a reversal, not a nudge


def test_lines_that_meet_share_a_node():
    tee = [
        jno.Shape.line([(0, 0, 0), (0.01, 0, 0)], r=1e-4),
        jno.Shape.line([(0.01, 0, 0), (0.02, 0, 0)], r=1e-4),
        jno.Shape.line([(0.01, 0, 0), (0.01, 0.01, 0)], r=1e-4),
    ]
    f = line_filaments(tee, size=0.01)
    assert f.incidence.shape == (4, 3)  # 3 free ends + the junction, not 6 endpoints
    assert int((np.abs(f.incidence).sum(1) == 3).sum()) == 1


def test_a_closed_loop_has_no_port():
    ang = np.linspace(0, 2 * np.pi, 41)
    ring = jno.Shape.line([(0.02 * np.cos(t), 0.02 * np.sin(t), 0.0) for t in ang], r=2e-4)
    f = line_filaments(ring, size=0.005)
    with pytest.raises(ValueError, match="opened where its source sits"):
        network_impedance(f, SIG, ((0.02, 0, 0), (0.02, 0, 0)))


def test_the_impedance_is_differentiable_in_the_conductivity():
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0.03, 0, 0)], r=3e-4), size=0.005)
    r_of = lambda s: jnp.real(network_impedance(f, s, ((0, 0, 0), (0.03, 0, 0)), omega=0.0)[0])
    g = float(jax.grad(r_of)(SIG))
    fd = float((r_of(SIG * 1.001) - r_of(SIG * 0.999)) / (0.002 * SIG))
    assert abs(g / fd - 1) < 1e-6
    assert g < 0  # more conductive, less resistive
