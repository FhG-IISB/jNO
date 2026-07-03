"""Inverse plate stiffness — image two hidden defects from a plate's deflection (4th-order field inverse).

A plate with hidden flaws has a spatially varying flexural rigidity ``k(x)`` — a stiff inclusion here, a soft
void there. Its deflection under a known load obeys ``k(x)·Δ²w = q``. The **inverse** problem: given only the
measured deflection ``w``, recover the field ``k(x)`` and so *locate the defects*. This is well posed under a
uniform load: ``Δ²w = q/k > 0`` everywhere, so the deflection is sensitive to ``k`` at every point.

It exercises the deepest capabilities together: the **Argyris** ``C¹`` element for the biharmonic forward
solve, a spatially varying **P1 field parameter** ``k = jno.np.parameter(kf)`` (its unknowns live at the mesh
vertices, interpolated with ``P1`` shape functions independently of the Argyris trial), and a **differentiable**
``fem.solve()`` — the whole system is re-assembled at each ``k`` and reverse-mode differentiated, so gradients
of the deflection w.r.t. ``k`` flow through the entire solve and ``crux`` drives ``k`` from a flat guess to the
truth.

We plant a **stiff inclusion** and a **soft void** (two Gaussian defects at different corners), generate the
clamped-plate deflection under a uniform load, then hand only that deflection to ``crux``, starting from a flat
``k = 1``.

Reference: the differentiable-solve / inverse pattern (`docs/inverse-problems.md`). J.H. Argyris, I. Fried,
D.W. Scharpf (1968); R.C. Kirby, SMAI J. Comput. Math. **4** (2018) — the ``C¹`` element.
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

laplacian = jno.np.laplacian
_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})
_dense = lambda A: A.todense() if hasattr(A, "todense") else A  # noqa: E731
solver = lambda A, b: jnp.linalg.solve(jnp.asarray(_dense(A)), jnp.asarray(b).reshape(-1))  # noqa: E731


def k_defects(xy):
    """True stiffness: a stiff inclusion (+) at one corner, a soft void (−) at the other."""
    x, y = xy[:, 0], xy[:, 1]
    stiff = 0.6 * np.exp(-((x - 0.32) ** 2 + (y - 0.36) ** 2) / (2 * 0.13**2))
    soft = -0.45 * np.exp(-((x - 0.68) ** 2 + (y - 0.64) ** 2) / (2 * 0.13**2))
    return 1.0 + stiff + soft


def build(mesh_size):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    kf, _ = d.fem_symbols()  # P1 stiffness field, independent of the Argyris trial
    k = jno.np.parameter(kf, name="k")
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    q = 1.0 + 0.0 * xi  # uniform transverse load
    fem = jno.fem([k * (laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi])) - q * vi, u(xb, yb) - 0.0, u.dn(xb, yb) - 0.0])
    return d, k, fem


def main():
    d, k, fem = build(0.12)
    assert fem.is_linear and list(fem.operator.runtime_parameter_exprs) == ["k"], "expected a parametric k-field solve"
    nodes = np.asarray(d.built_mesh.points)[:, :2]
    k_true = jnp.asarray(k_defects(nodes))

    # "measured" deflection at the true stiffness (the data we invert)
    A_t, b = fem.operator.evaluate({"k": k_true})
    w_obs = solver(A_t, b)

    # recover k(x) end-to-end through crux (differentiable re-assembled solve), from a flat wrong guess
    k.dtype(jnp.float64)
    k.initialize(jax.nn.initializers.constant(1.0))
    k.optimizer(optax.adam(3e-2))
    crux = jno.core([(fem.solve(linear=jno.solve.dense()) - w_obs).mse], domain=_DUMMY)
    crux.solve(200)  # loss plateaus early; 200 steps recovers both defects with budget to spare
    k_rec = np.asarray(crux.eval([k])).reshape(-1)

    kt = np.asarray(k_true)
    rel = float(np.linalg.norm(k_rec - kt) / np.linalg.norm(kt))
    print("\nInverse plate stiffness — two hidden defects (Argyris C¹ + P1 field parameter):")
    print(f"  mesh nodes = {nodes.shape[0]}   recovered k(x) rel-L² error: {rel:.3e}")
    print(f"  k* range [{kt.min():.3f}, {kt.max():.3f}]   recovered [{k_rec.min():.3f}, {k_rec.max():.3f}]")

    assert rel < 0.05, f"the stiffness field k(x) was not recovered: rel {rel:.3e}"
    assert abs(k_rec.max() - kt.max()) < 0.1, "the stiff inclusion must be recovered"
    assert abs(k_rec.min() - kt.min()) < 0.1, "the soft void must be recovered"

    # figure: true field | recovered field | pointwise recovery error
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], np.asarray(d.mesh.cells_dict["triangle"]))
    lvl = np.linspace(kt.min(), kt.max(), 25)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.0))
    for ax, field, ttl, cmap, levels in (
        (axes[0], kt, "true k*(x): stiff + soft defects", "RdBu_r", lvl),
        (axes[1], k_rec, f"recovered k(x)  (rel {rel:.1e})", "RdBu_r", lvl),
        (axes[2], np.abs(k_rec - kt), "recovery error |k − k*|", "magma", 20),
    ):
        tcf = ax.tricontourf(tri, field, levels=levels, cmap=cmap)
        ax.set_aspect("equal")
        ax.set_title(ttl)
        fig.colorbar(tcf, ax=ax, shrink=0.85)
    fig.suptitle("Locating hidden defects in a plate from its deflection (4th-order field inverse)")
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[2] / "assets" / "inverse_plate_stiffness_2d.png", dpi=130, bbox_inches="tight")

    print("\nOK: both the stiff inclusion and the soft void are recovered from the deflection through the C¹ solve.")


if __name__ == "__main__":
    main()
