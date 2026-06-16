"""Default floating-point dtype helpers — follow JAX's ``jax_enable_x64`` setting.

jNO does not own *data* precision: float32 vs float64 is JAX's ``jax_enable_x64``
flag, set by the user.  Array-producing code (domain sampling, parameter/array
attachment, ``pure_callback`` result structs) should follow the JAX default so
that enabling x64 propagates end-to-end instead of silently downcasting to
float32.  Model *parameter* precision is a separate, per-model concern
(``Model.dtype()``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def default_float_dtype():
    """JAX's current default floating dtype (float32, or float64 under ``jax_enable_x64``)."""
    return jnp.asarray(0.0).dtype


def default_np_float_dtype() -> np.dtype:
    """NumPy equivalent of :func:`default_float_dtype` — for host-side arrays."""
    return np.dtype(default_float_dtype())
