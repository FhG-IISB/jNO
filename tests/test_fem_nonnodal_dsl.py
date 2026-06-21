"""Non-nodal (RT) elements through the ``jno.fem`` weak-form DSL.

`jno.fem([... space="RT" ...])` routes to the native push-forward assembler
(`fem_nonnodal.assemble_fem_nonnodal`), reusing the shared integrand evaluator's
space-guarded RT branches. #2a covers the single-field H(div) mass / L²-projection
system. These pin it against the proven direct assembler (the mass matrix) and an
exact case: a constant vector lies in RT0, so its projection is recovered exactly.
Dense solves run on host (GPU-memory independent).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("feax", reason="feax required for jno.fem")
pytest.importorskip("pygmsh", reason="pygmsh required for 2D meshing")
pytest.importorskip("basix", reason="basix required for RT tabulation")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_nonnodal import assemble_mixed_poisson_rt, rt_flux_at_centroids  # noqa: E402
from jno.utils.solver.fem_topology import build_edge_topology  # noqa: E402

inner, grad, trace, sin = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.sin


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _dense(A):
    return np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))


def _rt_domain(mesh_size=0.4):
    d = jno.domain(box(0, 0, 1, 1), mesh_size=mesh_size)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    xi, yi, _ = d.variable("interior", split=True)
    return d, u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)


def _mesh(d):
    return np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])


def test_rt_mass_matrix_via_dsl_matches_direct_assembler():
    d, ui, vi = _rt_domain()
    A = _dense(jno.fem([inner(ui, vi)]).A)  # residual inner(u,v) -> A = RT mass, b = 0
    pts, cells = _mesh(d)
    A_dir, _, top, _ = assemble_mixed_poisson_rt(pts, cells, lambda x, y: 0.0 * x)
    M = np.asarray(A_dir)[: top.n_edges, : top.n_edges]
    np.testing.assert_allclose(A, M, atol=1e-12)
    np.testing.assert_allclose(A, A.T, atol=1e-12)  # RT mass is symmetric


def test_rt_projection_of_constant_is_exact():
    # constant g=(1,0) lies in RT0 -> the L2 projection recovers it exactly.
    d, ui, vi = _rt_domain()
    fem = jno.fem([inner(ui, vi) - vi[0]])  # residual ∫u·v - ∫g·v, g=(1,0)
    A, b = _dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1)
    uu = np.linalg.solve(A, b)
    pts, cells = _mesh(d)
    flux = np.asarray(rt_flux_at_centroids(pts, cells, build_edge_topology(cells), jnp.asarray(uu)))
    np.testing.assert_allclose(flux, np.tile([1.0, 0.0], (flux.shape[0], 1)), atol=1e-10)


def test_mixed_poisson_rt_p0_via_dsl_matches_direct_assembler():
    # Full RT-P0 mixed Poisson written through jno.fem must assemble the SAME (A, b) as the proven
    # direct assembler (which is convergence-tested in test_fem_nonnodal). div = trace(grad(.)).
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.3)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")
    p, q = d.fem_symbols(names=("p", "q"), space="P0")
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
    f = 2 * jnp.pi**2 * sin(jnp.pi * xi) * sin(jnp.pi * yi)
    fem = jno.fem([inner(ui, vi) - pp * divv, qq * divu - f * qq], quad_degree=4)
    A, b = _dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1)

    pts, cells = _mesh(d)
    src = lambda x, y: 2 * jnp.pi**2 * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)  # noqa: E731
    A_dir, b_dir, _, _ = assemble_mixed_poisson_rt(pts, cells, src, quad_degree=4)
    np.testing.assert_allclose(A, np.asarray(A_dir), atol=1e-11)
    np.testing.assert_allclose(b, np.asarray(b_dir), atol=1e-11)
