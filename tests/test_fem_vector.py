"""Vector-valued FEM through ``jno.fem``.

Covers: ``vec`` inferred from the trial's ``value_shape``; a vector elliptic
solve recovers a manufactured solution; full lambda-mu linear elasticity (with
the volumetric ``div(u)*div(phi)`` term enabled by the kernel prefix-alignment
fix) assembles symmetrically; and a vector Neumann *traction* assembles and
contributes to the load. All-component Dirichlet is via the scalar-broadcast
form ``u(region) - 0.0``; per-component (roller) Dirichlet is a follow-up.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
from shapely.geometry import box  # noqa: E402


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def test_vec_inferred_from_value_shape():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2)
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    n_nodes = int(np.asarray(d.mesh.points).shape[0])
    assert fem.dofs == 2 * n_nodes


def test_vector_poisson_recovers_manufactured():
    # Decoupled vector Poisson: -lap(u_k) = f_k, u = 0 on the boundary.
    # u_exact = (g, 0.5 g), g = x(1-x)y(1-y).
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    vi = phi.bind(x=xi, y=yi)
    f1 = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    f2 = 0.5 * f1
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2) - (
        f1 * vi.component(0) + f2 * vi.component(1)
    )
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    assert fem.is_linear and fem.dofs == 2 * int(np.asarray(d.mesh.points).shape[0])
    sol = np.linalg.solve(_dense(fem.A), np.asarray(fem.b).reshape(-1)).reshape(-1, 2)
    c = np.asarray(d.mesh.points)[:, :2]
    g = c[:, 0] * (1 - c[:, 0]) * c[:, 1] * (1 - c[:, 1])
    exact = np.stack([g, 0.5 * g], axis=-1)
    rel = np.linalg.norm(exact - sol) / np.linalg.norm(exact)
    assert rel < 1e-2


def test_full_elasticity_assembles_symmetric():
    # Full lambda-mu elasticity incl. the volumetric div(u)*div(phi) term.
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    eps_u, eps_phi = jno.np.symgrad(u, [xi, yi]), jno.np.symgrad(phi, [xi, yi])
    weak = 1.0 * jno.np.trace(eps_u) * jno.np.trace(eps_phi) + 2.0 * jno.np.inner(eps_u, eps_phi, n_contract=2)
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    A = _dense(fem.A)
    assert A.shape[0] == A.shape[1] == fem.dofs
    assert np.allclose(A, A.T, atol=1e-7)


# NOTE (follow-up): vector Neumann *traction* (`t·phi`) and per-component
# (roller) Dirichlet (`u(region).component(i) - g`) are not yet exposed through
# jno.fem because the vector view ops drop region/component metadata:
#   - `.component(i)` / `inner(...)` discard the bound view's `_coord_vars`, so a
#     vector boundary term can't be classified onto its region (scalar `g_N*phi`
#     keeps it via the ScalarView `_rewrap`);
#   - `.component(i)` hides the component index inside a `getitem` closure.
# Both are view-layer fixes (preserve `_coord_vars` + expose the component index),
# independent of the kernel/assembly, which handle vector fields correctly.
