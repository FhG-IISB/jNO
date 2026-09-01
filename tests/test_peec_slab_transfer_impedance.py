"""The 2-port slab impedance, and the identity that makes it safe to switch on.

A slab given ONE impedance has one current unknown, so its current can only be spread through the
whole cell -- which is why its inductance disagreed with its resistance. Giving each face a sheet
current lets the solve find the split. The 2x2 must not be a new model bolted alongside the old one:
in the symmetric mode it has to BE the old one, exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.kernel import internal_impedance, slab_transfer_impedance

jax.config.update("jax_enable_x64", True)

SIG, MU0 = 5.8e7, 4e-7 * np.pi
ELL, WID, THK = 0.010, 4e-3, 2e-3


def _one(omega, thick=THK, sigma=SIG):
    return slab_transfer_impedance(ELL, WID, thick, omega, sigma)


def _scalar(omega, thick=THK, sigma=SIG):
    """What the one-unknown model gives for the same slab."""
    return complex(internal_impedance(ELL, WID * thick, thick, False, omega, sigma, MU0, 1))


@pytest.mark.parametrize("hz", [1e2, 1e4, 1e5, 1e6, 1e7, 1e8])
def test_two_sheets_in_parallel_are_the_one_unknown_model(hz):
    """coth(x) + csch(x) = coth(x/2), so equal sheet currents reproduce it to the last bit.

    The sheets sit in parallel across one pair of nodes, so a total current I splits I/2 each and
    V = (z_self + z_mutual) I / 2. That must equal the slab branch of internal_impedance.
    """
    zs, zm = _one(2 * np.pi * hz)
    assert abs(complex(0.5 * (zs + zm)) / _scalar(2 * np.pi * hz) - 1) < 1e-12


@pytest.mark.parametrize("thick", [1e-4, 5e-4, 2e-3, 1e-2])
def test_the_identity_holds_across_thickness_too(thick):
    """Not just a coincidence at one aspect ratio -- gamma*t is what the identity is in."""
    zs, zm = _one(2 * np.pi * 1e6, thick=thick)
    assert abs(complex(0.5 * (zs + zm)) / _scalar(2 * np.pi * 1e6, thick=thick) - 1) < 1e-12


def test_at_dc_both_entries_are_the_dc_resistance():
    """No skin depth, so no face is preferred and the pair is degenerate -- which is exactly why a
    sheet pair is only emitted where the conductor is thick against the skin depth."""
    dc = ELL / (SIG * WID * THK)
    zs, zm = _one(0.0)
    assert abs(complex(zs).real / dc - 1) < 1e-12
    assert abs(complex(zm).real / dc - 1) < 1e-12


def test_the_faces_decouple_once_the_slab_is_many_skin_depths_thick():
    """csch(gamma t) -> 0: current on one face stops knowing about the other, which is the whole
    physical content of the fix -- copper below the skin layer carries nothing."""
    zs, zm = _one(2 * np.pi * 1e8)  # delta = 6.6 um against a 2 mm slab, 300 skin depths
    assert abs(complex(zm)) / abs(complex(zs)) < 1e-6
    # and the surviving self term is the surface impedance of one face, gamma rho l / w
    g = np.sqrt(1j * 2 * np.pi * 1e8 * MU0 * SIG)
    assert abs(complex(zs) / (g * ELL / (SIG * WID)) - 1) < 1e-6


def test_it_is_differentiable_including_at_dc():
    """The sqrt in gamma is zero at DC, and differentiating a masked branch through it gives NaN."""
    for hz in (0.0, 1e6):
        loss = lambda s, hz=hz: jnp.real(sum(slab_transfer_impedance(ELL, WID, THK, 2 * np.pi * hz, s)))
        g = float(jax.grad(loss)(SIG))
        fd = float((loss(SIG * 1.001) - loss(SIG * 0.999)) / (0.002 * SIG))
        assert np.isfinite(g) and abs(g / fd - 1) < 1e-4
