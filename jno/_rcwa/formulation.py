"""The Fourier factorizations a patterned layer can use.

``FFT`` is Laurent's rule for every component of the permittivity. Every other member is a vector formulation:
Laurent's rule for the field components tangent to the interfaces and the inverse rule for the normal one (Li,
JOSA A 13, 1870 (1996); for 2-D patterns through a tangent vector field, ``tangent.py``). The names are those of
``jno.rcwa``'s ``formulation=`` argument:

* ``NORMAL``: a pointwise-normalised tangent field (the normal-vector method; Schuster et al., JOSA A (2007);
  Götz et al., Opt. Express (2008));
* ``POL``: the field without normalisation (Liu & Fan, Comput. Phys. Commun. 183, 2233 (2012));
* ``JONES``: a complex elliptical (Jones) field, which has no zeros (Antos, Opt. Express (2009));
* ``JONES_DIRECT``: the Jones field sought directly by the field solve (Schubert & Hammond, arXiv:2308.08573);
* the ``_FOURIER`` variants smooth the field with a Fourier-domain penalty instead of a real-space one.
"""

from __future__ import annotations

import enum
from typing import Callable, Tuple

import jax.numpy as jnp

from . import basis as basis_mod


class Formulation(enum.Enum):
    """How the transverse permittivity matrix of a patterned layer is built (see the module docstring)."""

    FFT = "fft"
    JONES_DIRECT = "jones_direct"
    JONES = "jones"
    NORMAL = "normal"
    POL = "pol"
    JONES_DIRECT_FOURIER = "jones_direct_fourier"
    JONES_FOURIER = "jones_fourier"
    NORMAL_FOURIER = "normal_fourier"
    POL_FOURIER = "pol_fourier"


#: A custom formulation: maps ``(permittivity, expansion, lattice_vectors)`` to a tangent field ``(tx, ty)``.
VectorFn = Callable[
    [jnp.ndarray, basis_mod.Expansion, basis_mod.LatticeVectors],
    Tuple[jnp.ndarray, jnp.ndarray],
]
