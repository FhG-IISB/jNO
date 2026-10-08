"""A per-region coefficient as a COMPONENT of a vector coefficient assembles (it used to hang).

``vec(0*x, d.e, 0*x)`` with ``d.e`` from ``d.attach`` mixes a per-cell scalar with per-point
components, so ``concat`` takes its rank-alignment fallback. That fallback called ``max``, which in
``jno/jnp_ops.py`` is the module's own trace reduction, not the builtin: it built a trace node as a
reshape target, JAX formatted the shape error by iterating the node, and that never ended -- a silent
hang inside ``jno.fem`` instead of an assembled load. The workaround was ``d.e + 0*x``.
"""

import numpy as np
import pytest

import jax.numpy as jnp

import jno
from jno.jnp_ops import concat

gmsh = pytest.importorskip("gmsh")
inner, vec = jno.np.inner, jno.np.vector


def test_concat_fallback_aligns_ranks():
    """The rank-alignment path itself: a scalar next to per-point columns."""
    fc = concat([jnp.zeros((4, 1)), jnp.asarray(2.0), jnp.ones((4, 1))])
    out = np.asarray(fc.fn(jnp.zeros((4, 1)), jnp.asarray(2.0), jnp.ones((4, 1))))
    np.testing.assert_array_equal(out, np.stack([np.zeros(4), np.full(4, 2.0), np.ones(4)], axis=1))


def _two_region_box(path):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        outer = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        core = gmsh.model.occ.addBox(0.3, 0.3, 0.3, 0.4, 0.4, 0.4)
        gmsh.model.occ.fragment([(3, outer)], [(3, core)])
        gmsh.model.occ.synchronize()
        vols = [t for _, t in gmsh.model.getEntities(3)]
        cu = [t for t in vols if abs(gmsh.model.occ.getMass(3, t) - 0.064) < 1e-9]
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, cu), "cu")
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, [t for t in vols if t not in cu]), "air")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.3)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def test_region_coefficient_component_of_a_vector_load(tmp_path):
    _two_region_box(tmp_path / "box.msh")
    d = jno.domain(str(tmp_path / "box.msh"))
    d.attach("cu", e=2.5).attach("air", e=0.0)
    u, w = d.fem_symbols(value_shape=(3,), names=("u", "w"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    A, V = u.bind(x=x, y=y, z=z), w.bind(x=x, y=y, z=z)

    def load(E):
        return np.asarray(jno.fem([inner(A, V) - inner(E, V)]).b).reshape(-1)

    b_bare = load(vec(0.0 * x, d.e, 0.0 * x))                  # used to hang here
    b_broadcast = load(vec(0.0 * x, d.e + 0.0 * x, 0.0 * x))   # the old workaround
    np.testing.assert_allclose(b_bare, b_broadcast, rtol=0, atol=1e-14)
    assert np.abs(b_bare).max() > 0.0
