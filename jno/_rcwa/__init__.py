"""Self-owned RCWA backend for ``jno.rcwa`` (private package).

Stage 1 ships the lattice/expansion and Fourier-convolution utilities; later
stages add the eig-free state-space layer solve, the chunked R-matrix
recursion, field reconstruction and the source terms. The public surface here
mirrors the subset of fmmax that ``jno.rcwa`` consumes, with identical names,
shapes and ordering conventions. fmmax must never be imported in this package
-- it appears only as a test-only oracle in the test suite.

Lattice/basis objects and helpers are re-exported flat (as in fmmax). The
field-transport functions live on the ``jno._rcwa.fft`` module, whose name is
kept from being shadowed by the like-named transform function (``fft.fft``).
"""

from jno._rcwa import basis, fft, fields, layer, matrices, sources, stack, utils  # noqa: E402,F401
from jno._rcwa.basis import (
    Expansion,
    LatticeVectors,
    Truncation,
    generate_expansion,
    min_array_shape_for_expansion,
    transverse_wavevectors,
    validate_shape_for_expansion,
)
from jno._rcwa.matrices import Formulation  # noqa: E402
from jno._rcwa.stack import (
    StateStack,
    count_patterned_solves,
    patterned_solve_count,
    reset_patterned_solve_count,
)
from jno._rcwa.utils import (
    absolute_axes,
    angular_frequency_for_wavelength,
    batch_diag,
    matrix_adjoint,
)

__all__ = [
    "Expansion",
    "Formulation",
    "LatticeVectors",
    "StateStack",
    "Truncation",
    "absolute_axes",
    "angular_frequency_for_wavelength",
    "basis",
    "batch_diag",
    "count_patterned_solves",
    "fft",
    "fields",
    "generate_expansion",
    "layer",
    "matrices",
    "matrix_adjoint",
    "min_array_shape_for_expansion",
    "patterned_solve_count",
    "reset_patterned_solve_count",
    "stack",
    "transverse_wavevectors",
    "utils",
    "validate_shape_for_expansion",
]
