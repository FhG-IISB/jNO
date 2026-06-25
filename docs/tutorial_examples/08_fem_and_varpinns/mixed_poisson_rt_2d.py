"""09 - Mixed Poisson via Raviart-Thomas H(div) flux + P0 pressure (``jno.fem``, ``space="RT"``).

Solves the Poisson problem ``-Δp = f`` in its first-order (dual) mixed form

    u = -grad p,    div u = f      on [0, 1]^2,    p = 0 on the boundary,

with an **H(div)-conforming** Raviart-Thomas flux ``u`` and a piecewise-constant ``P0`` pressure
``p`` -- the first non-nodal element family wired through ``jno.fem``. RT/P0 are assembled by jNO's
native push-forward engine, but the weak form is written exactly like any
other coupled problem: two ``fem_symbols`` field pairs, with the field's ``space`` the only new knob.
``div`` is ``trace(grad(.))`` (as in the Stokes tutorial). A Dirichlet condition on ``p`` is *natural*
in the mixed form, so ``p = 0`` needs no essential constraint on the flux.

Mixed weak form (``v`` in RT, ``q`` in P0):

    ∫ u·v - ∫ p div v = 0,        ∫ q div u = ∫ f q.

Manufactured solution ``p = sin(πx) sin(πy)`` gives ``f = -Δp = 2π² p`` and ``p = 0`` on ∂Ω. The
lowest-order RT-P0 pair recovers the flux at ``O(h)`` and the (centroid) pressure at ``O(h²)``.

Reference: P.-A. Raviart, J.-M. Thomas, *A mixed finite element method for 2nd order elliptic
problems*, Mathematical Aspects of FEM, Lecture Notes in Math. 606 (1977).
"""

import jax

jax.config.update("jax_enable_x64", True)  # the saddle system is assembled/solved in float64

import jax.numpy as jnp
import numpy as np
from shapely.geometry import box

import jno
from jno.utils.solver.fem_nonnodal import rt_flux_at_centroids
from jno.utils.solver.fem_topology import build_edge_topology

inner, grad, trace, sin = jno.np.inner, jno.np.grad, jno.np.trace, jno.np.sin
dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), space="RT")  # H(div) flux
p, q = d.fem_symbols(names=("p", "q"), space="P0")  # piecewise-constant pressure
xi, yi, _ = d.variable("interior", split=True)
ui, vi, pp, qq = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi), p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
divu, divv = trace(grad(ui, [xi, yi])), trace(grad(vi, [xi, yi]))
f = 2.0 * jnp.pi**2 * sin(jnp.pi * xi) * sin(jnp.pi * yi)

# momentum (∫u·v - ∫p div v) tested by v ; continuity (∫q div u - ∫f q) tested by q
fem = jno.fem([inner(ui, vi) - pp * divv, qq * divu - f * qq], quad_degree=4)

# Solve the saddle system on host; slice per field via fem.offsets ([0, n_edges, n_edges + n_cells]).
pts, cells = np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])
sol = np.linalg.solve(dense(fem.A), np.asarray(jnp.asarray(fem.b)).reshape(-1))
off = fem.offsets
u_dofs, p_cells = sol[off[0] : off[1]], sol[off[1] : off[2]]
top = build_edge_topology(cells)  # still needed to evaluate the RT flux field at the centroids

# Verify the computed fields against the exact solution (audit the prediction, not a hand-built field).
centroid = pts[cells].mean(1)
area = np.abs([np.linalg.det(np.column_stack([pts[c][1] - pts[c][0], pts[c][2] - pts[c][0]])) for c in cells]) / 2
p_exact = np.sin(np.pi * centroid[:, 0]) * np.sin(np.pi * centroid[:, 1])
u_exact = np.stack(
    [
        -np.pi * np.cos(np.pi * centroid[:, 0]) * np.sin(np.pi * centroid[:, 1]),
        -np.pi * np.sin(np.pi * centroid[:, 0]) * np.cos(np.pi * centroid[:, 1]),
    ],
    axis=-1,
)
flux = np.asarray(rt_flux_at_centroids(pts, cells, top, jnp.asarray(u_dofs)))
err_p = float(np.sqrt(np.sum(area * (p_cells - p_exact) ** 2)))
err_u = float(np.sqrt(np.sum(area * np.sum((flux - u_exact) ** 2, axis=1))))
print(f"\nMixed Poisson (RT-P0 via jno.fem): dofs={fem.dofs}  L2 err p={err_p:.3e}  flux={err_u:.3e}")
assert err_p < 5e-3 and err_u < 0.2  # lowest-order RT-P0 at this resolution
