"""Inverse plate stiffness — recover a spatially-varying k(x) from a deflection (4th-order field inverse).

A clamped Kirchhoff plate under a known load ``q`` bends according to ``k(x)·Δ²w = q``, where the flexural
rigidity ``k(x)`` varies in space (a stiffer or thinner region). The **inverse** problem: given the measured
deflection ``w``, recover the hidden stiffness field ``k(x)``. This is well posed here — under a uniform load
``Δ²w = q/k > 0`` everywhere, so the deflection sees ``k`` at every point.

This exercises the deepest new capabilities together: the **C¹ Argyris** biharmonic element, a spatially
varying **P1 field parameter** ``k(x) = jno.np.parameter(kf)`` (interpolated at the mesh vertices, independent
of the Argyris trial), and the **differentiable** ``fem.solve()`` — the whole solve is re-assembled at each
``k`` and reverse-mode-differentiated, so ``crux.solve`` drives ``k`` from a wrong initial guess to the truth.

We plant a stiffer central patch ``k*(x) = 1 + 0.6·sin(πx)sin(πy)``, generate the deflection, then recover
``k*`` end-to-end through ``crux`` from a flat ``k = 1`` start.

Reference: the differentiable-solve / inverse pattern (`docs/inverse-problems.md`). J.H. Argyris, I. Fried,
D.W. Scharpf (1968); R.C. Kirby, SMAI J. Comput. Math. 4 (2018) — the C¹ element.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # dense re-assembled solve per step — CPU is the right device

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import optax
from shapely.geometry import box

import jno

PI = np.pi
laplacian = jno.np.laplacian
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})
_dense = lambda A: A.todense() if hasattr(A, "todense") else A  # noqa: E731


def build(mesh_size):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    kf, _ = d.fem_symbols()  # P1 stiffness field, independent of the Argyris trial
    k = jno.np.parameter(kf, name="k")
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    q = 1.0 + 0.0 * xi  # uniform transverse load
    fem = jno.fem([k * (laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi])) - q * vi, u(xb, yb) - 0.0])
    return d, k, fem


def main():
    d, k, fem = build(0.3)
    assert fem.is_linear and list(fem.operator.runtime_parameter_exprs) == ["k"], "expected a parametric k-field solve"
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_true = jnp.asarray(1.0 + 0.6 * np.sin(PI * nodes[:, 0]) * np.sin(PI * nodes[:, 1]))  # stiffer central patch

    # "measured" deflection at the true stiffness (the data we invert)
    A_t, b = fem.operator.evaluate({"k": k_true})
    solver = lambda A, b: jnp.linalg.solve(jnp.asarray(_dense(A)), jnp.asarray(b).reshape(-1))  # noqa: E731
    w_obs = solver(A_t, b)

    # recover k(x) end-to-end through crux (differentiable re-assembled solve)
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))  # flat wrong guess
    k.optimizer(optax.adam(3e-2))
    crux = jno.core([(fem.solve(solver) - w_obs).mse], domain=_DUMMY)
    crux.solve(350)
    k_rec = np.asarray(crux.eval([k])).reshape(-1)

    kt = np.asarray(k_true)
    rel = float(np.linalg.norm(k_rec - kt) / np.linalg.norm(kt))
    print("\nInverse plate stiffness (Argyris C¹ + P1 field parameter):")
    print(f"  mesh nodes = {nodes.shape[0]}   recovered k(x) rel-L² error: {rel:.3e}")
    print(f"  k* range [{kt.min():.3f}, {kt.max():.3f}]   recovered [{k_rec.min():.3f}, {k_rec.max():.3f}]")

    assert rel < 0.05, f"the stiffness field k(x) was not recovered: rel {rel:.3e}"
    assert abs(k_rec.max() - kt.max()) < 0.1, "the stiff central patch must be recovered"

    # figure: true vs recovered stiffness field
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], np.asarray(d.mesh.cells_dict["triangle"]))
    lvl = np.linspace(min(kt.min(), k_rec.min()), max(kt.max(), k_rec.max()), 21)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, field, ttl in ((ax0, kt, "true k*(x)"), (ax1, k_rec, f"recovered k(x)  (rel {rel:.1e})")):
        tcf = ax.tricontourf(tri, field, levels=lvl, cmap="magma")
        ax.set_aspect("equal")
        ax.set_title(ttl)
        fig.colorbar(tcf, ax=ax, shrink=0.85)
    fig.suptitle("Recovering a plate's hidden stiffness field from its deflection")
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[2] / "assets" / "inverse_plate_stiffness_2d.png", dpi=130, bbox_inches="tight")

    print("\nOK: the hidden stiffness field k(x) is recovered from the deflection through the differentiable C¹ solve.")


if __name__ == "__main__":
    main()
