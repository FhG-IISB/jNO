"""RT–P0 mixed Poisson — end-to-end validation of the non-nodal assembly engine.

This is the convergence gate for the element zoo's first slice: it exercises edge
numbering + orientation (:mod:`fem_topology`), contravariant Piola + divergence
(:mod:`fem_elements`), and the saddle-block assembly (:mod:`fem_nonnodal`) together,
against the manufactured solution ``p = sin(πx) sin(πy)`` on the unit square
(``f = 2π² p``, ``p = 0`` on ∂Ω — natural in the mixed form, so no essential flux
BC). Correctness is the *convergence rate*, not a single-mesh error: a wrong Piola
or edge orientation still produces a small error but at the wrong rate.

Known rates for lowest-order RT–P0: flux ``||u-u_h||_{L²} = O(h)``; the P0 pressure
is ``O(h²)`` superconvergent at cell centroids. The dense solve runs on the host
(``np.linalg.solve``) so the test does not depend on GPU memory.
"""

from __future__ import annotations

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np
import pytest

from jno.utils.solver.fem_nonnodal import assemble_mixed_poisson_rt, rt_flux_at_centroids  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _unit_square_mesh(n: int):
    """``n x n`` unit square split into ``2 n^2`` triangles (point (c/n, r/n))."""
    pts = np.array([[c / n, r / n] for r in range(n + 1) for c in range(n + 1)], float)
    idx = lambda r, c: r * (n + 1) + c  # noqa: E731
    cells = []
    for r in range(n):
        for c in range(n):
            a, b, cc, d = idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)
            cells += [[a, b, cc], [b, d, cc]]
    return pts, np.asarray(cells)


def _p_exact(x, y):
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def _f(x, y):
    return 2 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def _u_exact(x, y):
    return jnp.stack(
        [-jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y), -jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)], -1
    )


def _solve_and_measure(n: int):
    """Assemble + solve RT–P0 mixed Poisson at resolution ``n``; return (err_p, err_u)."""
    pts, cells = _unit_square_mesh(n)
    A, b, top, _ = assemble_mixed_poisson_rt(pts, cells, _f, quad_degree=4)
    x = jnp.asarray(np.linalg.solve(np.asarray(A), np.asarray(b)))  # host solve (GPU-mem independent)
    ne = top.n_edges
    p_h = x[ne:]
    tris = jnp.asarray(pts[cells])
    centroid = tris.mean(1)
    area = jnp.abs(jax.vmap(lambda v: jnp.linalg.det(jnp.stack([v[1] - v[0], v[2] - v[0]], 1)))(tris)) / 2
    err_p = float(jnp.sqrt(jnp.sum(area * (p_h - _p_exact(centroid[:, 0], centroid[:, 1])) ** 2)))
    u_h = rt_flux_at_centroids(pts, cells, top, x[:ne])
    err_u = float(jnp.sqrt(jnp.sum(area * jnp.sum((u_h - _u_exact(centroid[:, 0], centroid[:, 1])) ** 2, 1))))
    return err_p, err_u


def test_assembled_system_shape_and_solvable():
    pts, cells = _unit_square_mesh(4)
    A, b, top, spec = assemble_mixed_poisson_rt(pts, cells, _f, quad_degree=4)
    n = top.n_edges + cells.shape[0]
    assert A.shape == (n, n) and b.shape == (n,)
    assert spec.family == "RT" and spec.n_dof == 3
    err_p, err_u = _solve_and_measure(4)
    assert np.isfinite(err_p) and np.isfinite(err_u) and err_p < 0.1 and err_u < 0.5


def test_rt_flux_converges_first_order():
    errs = [_solve_and_measure(n)[1] for n in (4, 8, 16)]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    # RT0 flux is O(h); allow a small margin below the asymptotic rate 1.
    assert all(r > 0.9 for r in rates), f"flux rates {rates} not ~O(h)"


def test_rt_pressure_superconverges_at_centroids():
    errs = [_solve_and_measure(n)[0] for n in (4, 8, 16)]
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    # P0 pressure is O(h²) superconvergent at cell centroids; require clearly > 1.
    assert all(r > 1.7 for r in rates), f"pressure rates {rates} not superconvergent"
