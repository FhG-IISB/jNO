"""A P0 (per-cell) field parameter as a coefficient of a non-nodal (N1E) form.

The design density of topology optimisation is one value per element. The nodal assembler gathers such a
field by cell index; the non-nodal one used to gather every field parameter at VERTEX ids (a per-cell array
then read the wrong cells' values, out-of-range ids clamped silently), so it refused P0 outright. These
tests hold the wired path to an independent oracle: the per-region coefficient (`attach`), which reaches
the same per-cell constant through the region masks.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D box meshing")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
from jno.domain.mesh_utils import mesh_cell_region_membership  # noqa: E402

inner = jno.np.inner
B = jno.shape.box


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _domain():
    return (
        B(0, 0, 0, 0.5, 1, 1).name("a").attach(k=2.0)
        + B(0, 0, 0, 1, 1, 1).name("b").attach(k=5.0)
    ).sized(0.45).domain()


def _forms(d, coeff, bc=True):
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    ci = d.variable("interior", split=True)
    x, y, z = ci[0], ci[1], ci[2]
    cA, cV = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    terms = [coeff * inner(cA, cV) + inner(u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z))]
    if bc:
        terms.append(u.vector.cross(d.variable("boundary", normals=True)))
    return jno.fem(terms)


def _cell_density(d):
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    rho = jno.np.parameter(s, name="rho")
    rho.dtype(jnp.float64)
    return rho


def _dense(op, args=None):
    A = op.evaluate(args)[0] if hasattr(op, "evaluate") else op[0]
    return np.asarray(A.todense())


def test_p0_coefficient_equals_the_per_region_coefficient():
    d = _domain()
    in_a = np.asarray(mesh_cell_region_membership(d.mesh, 3)["a"], bool)
    assert 0 < in_a.sum() < len(in_a)
    K_region = _dense(_forms(d, d.k).operator)
    K_cell = _dense(_forms(d, _cell_density(d)).operator, {"rho": jnp.where(jnp.asarray(in_a), 2.0, 5.0)})
    np.testing.assert_allclose(K_cell, K_region, rtol=1e-12, atol=1e-12 * np.abs(K_region).max())


def test_p0_coefficient_is_linear_and_local_per_cell():
    """rho = one cell's indicator touches only that cell's six edges; the operator is linear in rho."""
    d = _domain()
    nc = len(d.mesh.cells_dict["tetra"])
    op = _forms(d, _cell_density(d), bc=False).operator
    K = jax.jit(lambda r: op.evaluate({"rho": r})[0].todense())
    K0 = np.asarray(K(jnp.zeros(nc)))
    for c in (0, nc // 2, nc - 1):
        dK = np.asarray(K(jnp.zeros(nc).at[c].set(1.0))) - K0
        rows = np.flatnonzero(np.abs(dK).max(1) > 1e-12 * np.abs(dK).max())
        assert 1 <= len(rows) <= 6, (c, len(rows))
    r1, r2 = jnp.asarray(np.random.default_rng(0).uniform(size=nc)), jnp.asarray(np.random.default_rng(1).uniform(size=nc))
    np.testing.assert_allclose(np.asarray(K(r1 + 2 * r2)) - K0, (np.asarray(K(r1)) - K0) + 2 * (np.asarray(K(r2)) - K0),
                               rtol=1e-10, atol=1e-10 * np.abs(K0).max())


def test_p0_gradient_matches_finite_differences():
    d = _domain()
    nc = len(d.mesh.cells_dict["tetra"])
    op = _forms(d, _cell_density(d) ** 3).operator          # a SIMP-like nonlinearity in rho
    n = op.evaluate({"rho": jnp.ones(nc)})[0].shape[0]
    w = jnp.asarray(np.random.default_rng(2).normal(size=n))
    f = jax.jit(lambda r: w @ (op.evaluate({"rho": r})[0] @ w))
    r0 = jnp.asarray(np.random.default_rng(3).uniform(0.3, 1.0, size=nc))
    g = np.asarray(jax.grad(f)(r0))
    for c in (1, nc // 3, nc - 2):
        h = 1e-6
        fd = (float(f(r0.at[c].add(h))) - float(f(r0.at[c].add(-h)))) / (2 * h)
        np.testing.assert_allclose(g[c], fd, rtol=1e-6, atol=1e-9 * np.abs(g).max())


def test_p0_array_of_the_wrong_length_is_refused():
    d = _domain()
    op = _forms(d, _cell_density(d)).operator
    with pytest.raises(ValueError, match="one value per cell"):
        op.evaluate({"rho": jnp.ones(len(d.mesh.points))})          # a vertex-length array for a P0 field
