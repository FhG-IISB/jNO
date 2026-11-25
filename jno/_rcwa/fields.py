"""Interior field reconstruction for the state-space RCWA backend.

The readouts that only touch the two ambients (efficiency, per-order fluxes,
order readouts, the incident/reflected states) are fully implemented in
``stack.py``. Interior field reconstruction -- propagating the surface state to
``z`` offsets inside a layer and inverse-Fourier-transforming to ``|E|^2``
volumes and z-profiles -- is stage-2 work. This module exists so that the seam
and callers get a loud, documented failure instead of a silent gap: the
signature of :func:`interior_field_amplitudes` is the one the future
implementation will keep.

Before that lands, ``field``/``field3d``-style readouts must raise rather than
return wrong values -- an interior probe on a layer with an evanescent-heavy
state would silently produce NaNs, never a plausible-but-wrong number, so the
risk of a silent gap is low; still, the seam task must wire what exists
(T/R/order/jones) and leave interior fields raising until stage 2 lands them.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax.numpy as jnp


def interior_field_amplitudes(
    stack,
    z_offsets: Sequence[float],
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Tangential ``(Ex, Ey)`` amplitudes inside the stack at `z_offsets`.

    Args:
        stack: A solved :class:`jno._rcwa.stack.StateStack` (its latest
            :meth:`~jno._rcwa.stack.StateStack.solve` supplies the surface
            state that is propagated inward).
        z_offsets: Distances below the superstrate interface at which the field
            is requested.

    Returns:
        ``(ex, ey, ez)`` Fourier amplitudes per layer/offset. Stage 2 will add
        the iFFT to real space and the ``|E|^2`` readouts.

    Raises:
        NotImplementedError: Always -- interior field reconstruction is stage-2
            work (see module docstring).
    """
    del stack, z_offsets
    raise NotImplementedError(
        "interior field reconstruction (field3d / |E|^2 volumes / z-profiles) is "
        "not implemented yet in jno/_rcwa -- stage 2 lands it. T/R/order/jones "
        "readouts from StateStack.solve() are fully functional; interior fields "
        "raise loudly rather than silently returning wrong values."
    )
