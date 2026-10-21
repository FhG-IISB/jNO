"""The PARDISO / cuDSS factorization caches keep one plan per (sparsity pattern, matrix TYPE).

Both caches detect the type (symmetric or general) from the values, and the type is baked into the
symbolic analysis. Values that alternate between a symmetric and a general matrix on ONE pattern -- a
Navier-Stokes Newton solve whose first tangent is the Stokes matrix at u = 0, then the unsymmetric ones --
used to throw the plan away and redo the analysis on every switch.
"""

import numpy as np
import pytest

from jno.utils.solver import linear as L

BACKENDS = {
    "pardiso": (L._pardiso_available, L._pardiso_host_solve, L._PARDISO_CACHE),
    "cudss": (L._cudss_available, L._cudss_host_solve, L._CUDSS_CACHE),
}


def _pair(n=40, seed=0):
    """A symmetric and an unsymmetric matrix on the same pattern (a 1-D Laplacian band plus a diagonal)."""
    rng = np.random.default_rng(seed)
    rows = np.concatenate([np.arange(n), np.arange(n - 1), np.arange(1, n)])
    cols = np.concatenate([np.arange(n), np.arange(1, n), np.arange(n - 1)])
    off = rng.uniform(0.5, 1.0, n - 1)
    sym = np.concatenate([4.0 + rng.uniform(0, 1, n), -off, -off])
    gen = np.concatenate([4.0 + rng.uniform(0, 1, n), -off, -0.3 * off])
    return np.stack([rows, cols], 1), sym, gen, (n, n)


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_alternating_types_reuse_both_plans(backend):
    available, solve, cache = BACKENDS[backend]
    if not available():
        pytest.skip(f"{backend} is not available here")
    idx, sym, gen, shape = _pair()
    b = np.ones(shape[0])
    cache.clear()
    plans = []
    for k in range(6):  # S, G, S, G, ... with new values each time, so every call refactorizes
        data = (sym if k % 2 == 0 else gen) * (1.0 + 1e-3 * k)
        x = np.asarray(solve(data, idx, b, shape, False))
        A = np.zeros(shape)
        np.add.at(A, (idx[:, 0], idx[:, 1]), data)
        np.testing.assert_allclose(A @ x, b, rtol=0, atol=1e-10)  # a direct solve of a well-conditioned band
        plans.append({id(e["solver"]) for e in cache.values()})
    assert len(plans[-1]) == 2, "one plan per matrix type"
    assert plans[1] == plans[-1], "the plans of the first two calls were rebuilt"
