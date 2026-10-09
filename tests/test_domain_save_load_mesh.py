"""A mesh-file domain survives jno.save / jno.load with its regions and attached materials.

The round trip must not need the original mesh file, and the operator assembled from the loaded
domain must be bit-identical to the one assembled from the original -- for a domain saved both
before and after a weak form was built on it.
"""

import os

import numpy as np
import pytest

import jno

gmsh = pytest.importorskip("gmsh")
inner = jno.np.inner


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
        air = [t for t in vols if t not in cu]
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, cu), "cu")
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, air), "air")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.25)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def _operator(d):
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    x, y, z = c[0], c[1], c[2]
    A, V = u.bind(x=x, y=y, z=z), v.bind(x=x, y=y, z=z)
    curl_A, curl_V = u.vector.curl(x, y, z), v.vector.curl(x, y, z)
    n_x_A = u.vector.cross(d.variable("boundary", normals=True))
    K = jno.fem([d.nu * inner(curl_A, curl_V) + 1j * 1e3 * d.sigma * inner(A, V) + 1e-6 * inner(A, V), n_x_A]).operator[0]
    return np.asarray(K.todense()) if hasattr(K, "todense") else np.asarray(K)


@pytest.mark.parametrize("built_before_save", [False, True])
def test_mesh_domain_round_trip(tmp_path, built_before_save):
    msh = tmp_path / "box.msh"
    _two_region_box(msh)
    d = jno.domain(str(msh))
    d.attach("cu", sigma=5.8e7, nu=1.0).attach("air", sigma=0.0, nu=2.0)
    if built_before_save:
        K0 = _operator(d)
    jno.save(d, str(tmp_path / "box.dom"))
    if not built_before_save:
        K0 = _operator(d)
    os.remove(msh)  # the saved domain must carry the mesh itself

    d2 = jno.load(str(tmp_path / "box.dom"), expected_type=jno.domain)
    assert d2.attached("sigma") == {"cu": 5.8e7, "air": 0.0}
    np.testing.assert_array_equal(_operator(d2), K0)
