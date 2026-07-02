"""13 - 3-D Maxwell cavity resonator via Nédélec H(curl) edge elements (``jno.fem``, ``space="N1E"``).

The resonant modes of a hollow perfect-electric-conductor (PEC) cavity solve the source-free Maxwell
eigenproblem ``curl curl E = k² E`` with ``n × E = 0`` on the walls. For a unit cube the spectrum is
**analytic** -- ``k²_{lmn} = π²(l² + m² + n²)`` -- so it is the textbook benchmark for a curl-curl
discretization, and the one that exposes *spurious modes*: a **nodal** (Lagrange) vector discretization
of ``curl curl`` produces extra non-physical eigenvalues in the gap below the first true mode, which is
why Maxwell eigenproblems are done with **edge** elements. Nédélec first-kind edge elements are
H(curl)-conforming (one tangential-moment DOF per mesh edge); the discrete gradient fields lie *exactly*
in the curl-curl kernel, so they collapse to ``λ = 0`` and are trivially separated -- **no spurious modes
appear**, and the physical modes converge (from below) to the analytic ``k²``.

This tutorial is the 3-D counterpart of ``maxwell_nedelec_2d.py``. jNO assembles N1E on a tetrahedral
cube with its native push-forward engine; the 3-D curl is the view sugar ``u.vector.curl(x, y, z)`` and
the PEC wall ``n × E = 0`` is written ``u.vector.cross(d.variable("boundary", normals=True))``. The
eigenproblem itself is assembled as the two matrices ``K`` (curl-curl) and ``M`` (mass) via ``jno.fem``,
reduced to the interior edges (PEC pins the boundary-face edge DOFs), and solved as a small dense
generalized eigenproblem ``K x = λ M x``.

Reference: J.-C. Nédélec, *Mixed finite elements in* R^3, Numer. Math. 35 (1980) 315-341; cavity
wavenumbers ``k²_{lmn} = π²(l²+m²+n²)`` (J.D. Jackson, *Classical Electrodynamics*, 3rd ed., §8.7-8.8).
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless figure rendering
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small dense eigenproblem -- CPU is plenty
from pathlib import Path  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # edge-element systems are assembled/solved in float64
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import jno  # noqa: E402
from jno.utils.solver.fem_facets import build_facet_connectivity  # noqa: E402
from jno.utils.solver.fem_topology import BASIX_TET_EDGES, build_edge_topology  # noqa: E402

inner = jno.np.inner
dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731
PI = float(np.pi)


def cavity_spectrum(mesh_size, n_modes=6):
    """Assemble the PEC-cube curl-curl eigenproblem and return its lowest ``n_modes`` non-kernel eigenvalues.

    ``K`` (curl-curl) and ``M`` (mass) come straight from ``jno.fem``; PEC (``n × E = 0``) is imposed by
    dropping the **boundary-face** edge DOFs (facet connectivity -- on a tet most boundary edges are shared
    by several cells, so the 2-D "edge used once" rule would miss them). The reduced ``K x = λ M x`` is
    solved as a Cholesky-reduced symmetric eigenproblem (``M = L Lᵀ`` ⇒ eig of ``L⁻¹ K L⁻ᵀ``); the
    gradient-field kernel collapses to ``λ ≈ 0`` and is filtered off."""
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    K = dense(jno.fem([inner(cu, cv)]).A)  # curl-curl stiffness
    M = dense(jno.fem([inner(ui, vi)]).A)  # H(curl) mass

    # PEC: interior edges = all edges except those on a boundary face
    cells = np.asarray(d.mesh.cells_dict["tetra"])
    top = build_edge_topology(cells, BASIX_TET_EDGES)
    edge_id = {(int(a), int(b)): i for i, (a, b) in enumerate(np.asarray(top.edge_vertices))}
    fc = build_facet_connectivity(cells, "tetrahedron")
    boundary_edges = set()
    for f in range(fc.n_bfaces):
        fn = [int(x) for x in fc.face_nodes[f]]
        for a, b in ((fn[0], fn[1]), (fn[1], fn[2]), (fn[0], fn[2])):
            boundary_edges.add(edge_id[(min(a, b), max(a, b))])
    interior = [i for i in range(K.shape[0]) if i not in boundary_edges]
    Ki, Mi = K[np.ix_(interior, interior)], M[np.ix_(interior, interior)]

    L = np.linalg.cholesky(Mi)  # Mi SPD
    Asym = np.linalg.solve(L, np.linalg.solve(L, Ki).T).T  # L⁻¹ K L⁻ᵀ
    w = np.sort(np.linalg.eigvalsh(Asym))
    w = w[w > 1.0]  # filter the curl-curl gradient kernel (λ ≈ 0), far below the first mode ~2π²
    return len(interior), w[:n_modes]


# --- analytic cavity ladder: k² = π²(l²+m²+n²) for modes with >= 2 nonzero indices ------------------
sums = sorted(
    {a * a + b * b + c * c for a in range(3) for b in range(3) for c in range(3) if (a > 0) + (b > 0) + (c > 0) >= 2}
)
analytic = [PI**2 * s for s in sums]  # 2π², 3π², 5π², 6π², 8π², ...

ndof_c, wc = cavity_spectrum(0.40)
ndof_f, wf = cavity_spectrum(0.28)
print(f"analytic lowest k² = 2π² = {2 * PI**2:.4f}")
print(f"coarse (interior dofs {ndof_c}): lowest 6 = {np.round(wc, 3)}")
print(f"fine   (interior dofs {ndof_f}): lowest 6 = {np.round(wf, 3)}")

# --- checks: the lowest triplet is the 3-fold-degenerate 2π² mode, spurious-free, converging from below --
k2 = 2.0 * PI**2
assert wc[0] > 0.6 * k2, f"a spurious sub-mode appeared: {wc[0]:.3f} (2π²={k2:.3f})"  # no fake low eigenvalue
assert abs(np.mean(wc[:3]) / k2 - 1.0) < 0.12, f"coarse triplet {np.round(wc[:3], 2)} not ≈ 2π²"
assert np.mean(wf[:3]) > np.mean(wc[:3]) and abs(np.mean(wf[:3]) / k2 - 1.0) < 0.08  # converges up toward 2π²
print("PASS: lowest cavity modes match the analytic k² ladder, spurious-free, converging from below.")

# --- figure: computed spectrum vs the analytic mode ladder (faithful -- only computed λ + analytic lines) --
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for a in analytic[:5]:
    ax.axhline(a, color="0.7", lw=1.0, ls="--", zorder=0)
ax.axhline(analytic[0], color="0.7", lw=1.0, ls="--", label="analytic $k^2=\\pi^2(l^2+m^2+n^2)$")
ax.plot(range(1, len(wc) + 1), wc, "o", ms=8, color="#c1121f", label=f"coarse ({ndof_c} dofs)")
ax.plot(range(1, len(wf) + 1), wf, "s", ms=7, color="#003049", label=f"fine ({ndof_f} dofs)")
ax.set_xlabel("mode index (nonzero eigenvalues, kernel filtered)")
ax.set_ylabel("$k^2$")
ax.set_title("PEC cube cavity: Nédélec H(curl) eigenvalues vs analytic modes")
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, analytic[4] * 1.1)
fig.tight_layout()
out = Path(__file__).resolve().parents[2] / "assets" / "maxwell_nedelec_3d.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=110)
print(f"figure -> {out}")
