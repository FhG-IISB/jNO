"""Shared helpers for the FDM benchmarks: x64, relative errors, observed rates."""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402


def nodes(d):
    """Node coordinates ``(N, dim)`` of a domain."""
    dim = int(getattr(d, "dimension", 2))
    return np.asarray(d.mesh_connectivity["points"])[:, :dim]


def rel(a, b):
    """Relative L2 error ``‖a − b‖ / ‖b‖``."""
    a, b = np.asarray(a).reshape(-1), np.asarray(b).reshape(-1)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def rates(errs):
    """Observed orders between successive halvings of h."""
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


def table(title, rows, cols):
    """Print ``rows`` (dicts) as an aligned table under ``title``."""
    print(f"\n{title}")
    print("  ".join(f"{c:>12s}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>12.3e}" if isinstance(r[c], float) else f"{str(r[c]):>12s}" for c in cols))
