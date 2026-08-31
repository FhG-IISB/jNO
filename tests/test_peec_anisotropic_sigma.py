"""Anisotropic conductivity: ``attach(sigma=(sx, sy, sz))``.

A rolled or laminated conductor does not conduct the same way along and across its grain, and a
sintered or plated layer often does not either. The lattice already has one current direction per
bar family, so the material only has to say which component belongs to which family -- there is no
new argument, only a fourth spelling of the one that already takes a scalar, a field and a vector,
and each of the three components may itself be any of those.

Scope, up front: **diagonal** anisotropy only. The bars are axis-aligned, so an off-diagonal
conductivity has nowhere to live in this discretisation. A wire is a one-dimensional conductor, so
what reaches it is the component along its own tangent, ``t . sigma . t``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import bar_filaments, line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7
LX, WY, TZ = 0.040, 0.004, 0.002


def _bar(sigma, pitch=0.002):
    f = bar_filaments(jno.Shape.box(0, 0, 0, LX, WY, TZ), size=pitch, sigma=[sigma])
    p = np.asarray(f.nodes)
    term = {
        "A": terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9),
        "B": terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9),
    }
    _c, _p, inj = solve_network(f, f.lattice["sigma"], term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0)
    return 1.0 / complex(inj["A"]).real


def test_an_isotropic_triple_is_the_scalar_it_repeats():
    """The new spelling has to reduce to the old one exactly, or it is a second code path."""
    assert abs(_bar((SIG, SIG, SIG)) / _bar(SIG) - 1) < 1e-12


def test_conduction_follows_the_component_along_the_CURRENT():
    """Current runs along x, so only sigma_x sets the resistance -- exactly, not approximately.

    This is the whole physical content: halving sigma_x doubles R, while halving sigma_y or sigma_z
    (which carry no net current on a straight bar) leaves it alone.
    """
    ref = _bar(SIG)
    assert abs(_bar((0.5 * SIG, SIG, SIG)) / ref - 2.0) < 1e-9
    assert abs(_bar((SIG, 0.5 * SIG, SIG)) / ref - 1.0) < 1e-9
    assert abs(_bar((SIG, SIG, 0.5 * SIG)) / ref - 1.0) < 1e-9


def test_each_component_may_itself_be_a_FIELD():
    """The three spellings compose: a component may be a callable of position, as a scalar may."""
    got = _bar((lambda x, y, z: jnp.full_like(x, 0.5 * SIG), SIG, SIG))
    assert abs(got / _bar((0.5 * SIG, SIG, SIG)) - 1) < 1e-12


def test_the_gradient_reaches_one_component_alone():
    """R = rho_x L / A is exactly inverse in sigma_x, so dR/ds at s=1 is -R. An oracle, not a re-run."""

    def loss(s):
        f = bar_filaments(jno.Shape.box(0, 0, 0, LX, WY, TZ), size=0.002, sigma=[(s * SIG, SIG, SIG)])
        p = np.asarray(f.nodes)
        term = {
            "A": terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9),
            "B": terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9),
        }
        _c, _p, inj = solve_network(f, f.lattice["resolve"]([(s * SIG, SIG, SIG)]), term,
                                    [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0)
        return jnp.real(1.0 / inj["A"])

    g = float(jax.grad(loss)(1.0))
    assert abs(g / -float(loss(1.0)) - 1) < 1e-8


def test_a_WIRE_takes_the_component_along_its_own_tangent():
    """A filament is one-dimensional, so what reaches it is ``t . sigma . t``, not a component.

    Checked on a 45-degree wire in the x-y plane, where t.sigma.t = (sx + sy)/2 exactly.
    """
    d = 1.0 / np.sqrt(2.0)
    sh = jno.Shape.line([(0, 0, 0), (0.01 * d, 0.01 * d, 0)], r=2e-4, size=0.002)
    f = line_filaments(sh)
    from jno.utils.solver.peec import element_centres, resolve_sigma

    got = np.asarray(resolve_sigma((SIG, 3.0 * SIG, 7.0 * SIG), np.asarray(element_centres(f)),
                                   "wire", tangent=np.asarray(f.mom)[:: f.mom.shape[0] // len(np.asarray(f.length))]))
    assert np.allclose(got, 2.0 * SIG, rtol=1e-9)  # (1 + 3)/2 = 2


def test_a_three_element_conductor_refuses_the_ambiguous_spelling():
    """``(3,)`` on a 3-element conductor is either three components or three elements. Ask, do not guess."""
    with pytest.raises(ValueError, match="ambiguous"):
        bar_filaments(jno.Shape.box(0, 0, 0, 0.003, 0.001, 0.001), size=0.001,  # exactly 3 cells
                      sigma=[np.array([SIG, SIG, SIG])])


def test_a_wrong_length_triple_is_refused():
    with pytest.raises(ValueError, match="conductivit"):
        _bar((SIG, SIG))
