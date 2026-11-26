"""Stacked small blocks never reach batched LAPACK (issue #353).

On XLA:CPU a batched ``jnp.linalg.det`` / ``inv`` / ``solve`` becomes an LU (``lu``) and triangular-solve kernel
that splits its batch over the intra-op thread pool and waits for the pieces. Several of them side by side deadlock
a pool of four threads or fewer. The geometry of a mesh (cell Jacobians, edge matrices, the gradient of a nodal
field) is stacked dim x dim blocks with dim <= 3, so it is written closed-form (``small_det`` / ``small_inv``):
the jaxpr holds no ``lu`` and no ``triangular_solve``, and the values are those of ``jnp.linalg``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.fem_adapt import _dirichlet_energy_jax, _signed_simplex_measures_jax

LAPACK_PRIMITIVES = ("lu", "triangular_solve", "cholesky", "eigh", "svd", "lu_pivots_to_permutation")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _primitives(jaxpr) -> set[str]:
    out = set()
    for eqn in jaxpr.eqns:
        out.add(eqn.primitive.name)
        for v in eqn.params.values():
            for sub in v if isinstance(v, (list, tuple)) else [v]:
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    out |= _primitives(inner)
    return out


def _mesh(dim, n=40, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, dim))
    cells = np.stack([rng.permutation(n)[: dim + 1] for _ in range(3 * n)])
    return jnp.asarray(pts), jnp.asarray(cells)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_simplex_geometry_has_no_lapack_kernel(dim):
    pts, cells = _mesh(dim)
    u = jnp.sin(pts.sum(axis=1))
    for f, args in (
        (lambda p, c: _signed_simplex_measures_jax(p, c, dim), (pts, cells)),
        (lambda p, c, u: _dirichlet_energy_jax(p, u, c, dim), (pts, cells, u)),
    ):
        prims = _primitives(jax.make_jaxpr(f)(*args).jaxpr)
        assert not prims & set(LAPACK_PRIMITIVES), sorted(prims)


@pytest.mark.parametrize("dim", [2, 3])
def test_the_dirichlet_energy_equals_the_lapack_solve(dim):
    pts, cells = _mesh(dim)
    u = jnp.cos(pts @ jnp.arange(1.0, dim + 1))
    v = pts[cells]
    e = jnp.stack([v[:, i + 1] - v[:, 0] for i in range(dim)], axis=1)
    du = jnp.stack([u[cells][:, i + 1] - u[cells][:, 0] for i in range(dim)], axis=1)[..., None]
    grad = jnp.linalg.solve(e, du)
    vol = jnp.abs(jnp.linalg.det(e)) / (2.0 if dim == 2 else 6.0)
    ref = jnp.sum(jnp.sum(grad**2, axis=(1, 2)) * vol)
    got = _dirichlet_energy_jax(pts, u, cells, dim)
    np.testing.assert_allclose(got, ref, rtol=1e-10)
