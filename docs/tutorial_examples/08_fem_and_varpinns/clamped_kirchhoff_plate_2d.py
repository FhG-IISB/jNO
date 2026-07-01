"""Clamped Kirchhoff plate — the *proper* clamped BC (free boundary curvature) on the Argyris C¹ element.

A thin plate bends under a transverse load: ``D·Δ²w = q`` with ``D`` the flexural rigidity. Clamped edges
impose ``w = 0`` and ``∂w/∂n = 0`` on ``∂Ω``. Crucially, a clamped edge *resists rotation but the plate still
bends there*, so the boundary curvature ``∂²w/∂n²`` — proportional to the reaction bending moment
``Mₙ = −D·∂²w/∂n²`` the clamp exerts — is **nonzero**. It is a *natural* boundary condition, not something to
prescribe.

This is exactly what the Argyris **proper clamped BC** gets right: ``u(region) - g`` pins ``u`` and ``∂u/∂n``
but leaves ``∂²u/∂n²`` free, so the clamp reaction moment emerges from the solve. Pinning the *full* C¹ trace
(``∂²w/∂n² = 0``) would wrongly annihilate the clamp moment and stiffen the plate.

Manufactured check: ``w* = sin²(πx) sin²(πy)`` is clamped on the unit square (``w* = ∂w*/∂n = 0`` on ∂Ω) yet
has nonzero boundary curvature; with ``D = 1`` the load is ``q = Δ²w*``. We solve, verify convergence to
``w*``, and read the clamped-edge curvature ``∂²w/∂n²`` **directly off the Argyris DOF that proper-clamped
frees** — it recovers the exact ``∂²w*/∂x²|_{x=0} = 2π² sin²(πy)``, the reaction moment a real clamp carries.

Reference: S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells* (1959) — clamped-plate bending
and edge moments. J.H. Argyris, I. Fried, D.W. Scharpf (1968); R.C. Kirby, SMAI J. Comput. Math. 4 (2018).
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # stiff h⁻⁴ biharmonic direct solve — CPU dodges GPU cuSolver OOM

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely.geometry import box

import jno

PI = np.pi
laplacian = jno.np.laplacian


def solve_plate(mesh_size):
    """Clamped plate ``Δ²w = q`` with ``q = Δ²(sin²πx·sin²πy)``; proper clamped BC (free ∂²w/∂n²)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    a = 2 * PI
    c2x, c2y = jno.np.cos(a * xi), jno.np.cos(a * yi)
    px, py = (1 - c2x) / 2, (1 - c2y) / 2
    q = 8 * PI**4 * (c2x * (c2y - py) - px * c2y)  # Δ²(sin²πx · sin²πy)
    fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - q * vi, u(xb, yb) - 0.0])
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :2]
    return d, pts, sol


def main():
    # --- convergence of the deflection to the exact clamped solution ---
    errs, hs = [], [0.28, 0.18]
    for ms in hs:
        _d, pts, sol = solve_plate(ms)
        nv = pts.shape[0]
        w = sol[6 * np.arange(nv)]  # value DOFs = deflection at the vertices
        w_exact = (np.sin(PI * pts[:, 0]) ** 2) * (np.sin(PI * pts[:, 1]) ** 2)
        errs.append(float(np.linalg.norm(w - w_exact) / np.linalg.norm(w_exact)))

    d, pts, sol = solve_plate(0.14)
    nv = pts.shape[0]

    # --- the clamp reaction: boundary curvature ∂²w/∂n² read off the FREED Argyris DOF ---
    # On the x=0 edge the normal is x, so ∂²w/∂n² = ∂²w/∂x² = the ∂ₓₓ DOF (6v+3) at each edge vertex.
    on_x0 = np.where(pts[:, 0] < 1e-9)[0]
    yv = pts[on_x0, 1]
    kappa_nn = sol[6 * on_x0 + 3]  # the boundary curvature the proper clamped BC left free
    kappa_exact = 2 * PI**2 * np.sin(PI * yv) ** 2  # ∂²w*/∂x²|_{x=0}
    order = np.argsort(yv)
    curv_rel = float(np.linalg.norm(kappa_nn - kappa_exact) / np.linalg.norm(kappa_exact))

    print("\nClamped Kirchhoff plate (Argyris C¹, proper clamped BC):")
    print(f"  deflection rel-L² error:  h={hs[0]}: {errs[0]:.3e}   h={hs[1]}: {errs[1]:.3e}")
    print(
        f"  clamped-edge curvature ∂²w/∂n² (x=0): max |computed| = {np.max(np.abs(kappa_nn)):.3f}  (a real clamp: nonzero)"
    )
    print(f"  vs exact 2π²sin²(πy):     rel error {curv_rel:.3e}")

    # --- asserts: the plate bends correctly AND the clamp carries a (correct, nonzero) reaction moment ---
    assert errs[1] < errs[0] and errs[1] < 0.02, f"clamped-plate deflection must converge to w*: {errs}"
    assert np.max(np.abs(kappa_nn)) > 1.0, "the clamped-edge curvature (reaction moment) must be nonzero"
    assert curv_rel < 0.05, f"the freed boundary curvature must match the exact clamp moment: rel {curv_rel:.3e}"

    # --- figure: the deflected plate + the clamp-edge curvature (free ∂²w/∂n²) vs exact ---
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], np.asarray(d.mesh.cells_dict["triangle"]))
    w = sol[6 * np.arange(nv)]
    tcf = ax0.tricontourf(tri, w, levels=20, cmap="viridis")
    ax0.set_aspect("equal")
    ax0.set_title("clamped plate deflection w(x,y)")
    fig.colorbar(tcf, ax=ax0, shrink=0.85, label="w")
    ax1.plot(yv[order], kappa_exact[order], "-", color="k", lw=2, label="exact ∂²w/∂n²  (= 2π²sin²πy)")
    ax1.plot(yv[order], np.asarray(kappa_nn)[order], "o", color="#c0392b", ms=4, label="computed (freed DOF)")
    ax1.axhline(0.0, color="#2471a3", ls="--", lw=1, label="full-trace over-pin (= 0, wrong)")
    ax1.set_xlabel("y along the clamped edge x = 0")
    ax1.set_ylabel("boundary curvature ∂²w/∂n²")
    ax1.set_title("clamp reaction moment ∝ ∂²w/∂n²")
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[2] / "assets" / "clamped_kirchhoff_plate_2d.png", dpi=130, bbox_inches="tight")

    print("\nOK: the plate bends to w* and the clamp carries the correct nonzero reaction moment (free ∂²w/∂n²).")


if __name__ == "__main__":
    main()
