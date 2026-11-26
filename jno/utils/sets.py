"""Sorted unique values of an array, by sorting.

``np.unique(a)`` with nothing else asked of it goes through a hash table since NumPy 2.3 and then sorts its result. On the
integer keys of a sparsity pattern, where most keys are distinct, that is 27-29x slower than sorting and dropping
repeats (3.6 s against 0.13 s for 11 M keys, measured with NumPy 2.5), and its table costs gigabytes at the size of a
large mesh's pattern. It was 44 of the 60 s of a 148 k-dof elasticity build and its three solves.
"""

import numpy as np

__all__ = ["unique_values"]


def unique_values(a) -> np.ndarray:
    """``np.unique(a)``, the flattened array's sorted distinct values; several NaNs count as one, as there."""
    s = np.asarray(a).ravel()
    if s.dtype.kind not in "biuf":
        return np.unique(s)
    if s.size < 2:
        return s.copy()
    s = np.sort(s)
    keep = np.empty(s.size, dtype=bool)
    keep[0] = True
    keep[1:] = s[1:] != s[:-1]
    if s.dtype.kind == "f" and np.isnan(s[-1]):  # NaNs sort last, and no NaN equals another
        keep[np.searchsorted(s, np.nan) + 1 :] = False
    return s[keep]
