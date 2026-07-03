"""Clamped Kirchhoff plate under uniform pressure — the classic benchmark, with the *proper* clamped BC.

A thin plate under a uniform transverse pressure ``q``, clamped on all four edges, bends according to
``D·Δ²w = q`` (``D`` = flexural rigidity). A clamped edge imposes ``w = 0`` and ``∂w/∂n = 0`` — but it does
**not** flatten the plate: the boundary curvature ``∂²w/∂n²`` is nonzero and equals the reaction bending
moment ``Mₙ = −D·∂²w/∂n²`` the clamp carries. That curvature is a *natural* boundary condition — the solve
determines it, you do not prescribe it — and it is exactly what the Argyris **proper clamped BC** leaves free
(``u(region) - g`` pins ``w`` and ``∂w/∂n`` but not ``∂²w/∂n²``). Pinning the full ``C¹`` trace would wrongly
annihilate the clamp moment.

This is the textbook **clamped square plate under uniform load** (Timoshenko & Woinowsky-Krieger, *Theory of
Plates and Shells*, 2nd ed. 1959, Table 35). With ``D=1``, ``q=1`` on the unit square the tabulated results are

    center deflection   w_max = 0.00126 q a⁴ / D,
    edge-midpoint moment  M    = 0.0513  q a²      ( = D·∂²w/∂n²,  the freed boundary curvature).

We solve on a refined mesh, validate both coefficients, and — because the ``C¹`` element carries the full
Hessian as *degrees of freedom* — read the entire **bending-moment field** ``M(x,y)`` directly off the solution
(no post-hoc differentiation), which peaks at the clamped-edge midpoints where the plate is most stressed.

Reference: S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed. (1959), §30 & Table 35.
J.H. Argyris, I. Fried, D.W. Scharpf (1968); R.C. Kirby, SMAI J. Comput. Math. **4** (2018).
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # stiff h⁻⁴ biharmonic — CPU sparse-direct dodges GPU cuSolver OOM

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely.geometry import box

import jno

PI = np.pi
laplacian = jno.np.laplacian
NU = 0.3  # Poisson ratio (enters the moment tensor, not the clamped governing equation)


def solve_plate(mesh_size):
    """Clamped square plate ``Δ²w = q`` under uniform pressure ``q = 1`` (``D = 1``); proper clamped BC."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    q = 1.0 + 0.0 * xi  # uniform transverse pressure
    # clamped = deflection w=0 (u(reg)-g) AND rotation ∂w/∂n=0 (u.dn(reg)-h); the boundary curvature stays free
    fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - q * vi, u(xb, yb) - 0.0, u.dn(xb, yb) - 0.0])
    sol = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)  # sparse-direct: O(nnz) memory on the fine mesh
    pts = np.asarray(d.mesh.points)[:, :2]
    return d, pts, sol


def main():
    d, pts, sol = solve_plate(0.06)
    nv = pts.shape[0]

    # value + Hessian DOFs straight off the Argyris solution (6·v + {0:w, 3:w_xx, 4:w_xy, 5:w_yy})
    w = sol[6 * np.arange(nv)]
    wxx, wxy, wyy = sol[6 * np.arange(nv) + 3], sol[6 * np.arange(nv) + 4], sol[6 * np.arange(nv) + 5]

    # bending-moment field M(x,y) = −D(∂²w + ν·∂²w^T); D=1 (von Mises equivalent moment)
    Mx, My, Mxy = -(wxx + NU * wyy), -(wyy + NU * wxx), -(1 - NU) * wxy
    Mvm = np.sqrt(Mx**2 - Mx * My + My**2 + 3 * Mxy**2)

    # --- benchmark 1: center deflection vs Timoshenko 0.00126 q a⁴/D ---
    ic = np.argmin((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2)
    w_center = w[ic]

    # --- benchmark 2: freed edge curvature ∂²w/∂n² at the clamped-edge midpoint vs Timoshenko 0.0513 q a²/D ---
    on_x0 = np.where(pts[:, 0] < 1e-9)[0]
    yv = pts[on_x0, 1]
    kappa_nn = wxx[on_x0]  # on x=0 the normal is x ⇒ ∂²w/∂n² = ∂²w/∂x², the DOF proper-clamped left free
    order = np.argsort(yv)
    kappa_mid = kappa_nn[np.argmin(np.abs(yv - 0.5))]

    print("\nClamped square plate under uniform pressure (Argyris C¹, proper clamped BC):")
    print(f"  mesh nv={nv}  dofs={21 * nv}")
    print(f"  center deflection   w_max = {w_center:.4e}   (Timoshenko 1.26e-03,  ratio {w_center / 1.26e-3:.3f})")
    print(f"  edge-mid curvature ∂²w/∂n² = {kappa_mid:.4f}     (Timoshenko 0.0513,     ratio {kappa_mid / 0.0513:.3f})")
    print(f"  peak bending moment |M| = {Mvm.max():.4f} at the clamped edge (stress concentration)")

    # --- asserts: both classical coefficients recovered; the clamp carries a nonzero reaction moment ---
    assert abs(w_center / 1.26e-3 - 1.0) < 0.05, f"center deflection must match Timoshenko: {w_center:.4e}"
    assert abs(kappa_mid / 0.0513 - 1.0) < 0.05, f"edge moment (freed curvature) must match Timoshenko: {kappa_mid:.4f}"
    assert Mvm.max() > Mvm[ic] * 1.5, "bending moment must peak at the clamped edges, not the center"

    # --- figure: 3D deflected dome | bending-moment field (from the C¹ Hessian) | edge moment vs Timoshenko ---
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], np.asarray(d.mesh.cells_dict["triangle"]))
    fig = plt.figure(figsize=(14.0, 4.2))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.plot_trisurf(tri, w, cmap="viridis", linewidth=0.1, antialiased=True)
    ax0.set_title("deflected plate w(x,y)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.view_init(elev=28, azim=-125)
    ax1 = fig.add_subplot(1, 3, 2)
    tcf = ax1.tricontourf(tri, Mvm, levels=30, cmap="inferno")
    ax1.set_aspect("equal")
    ax1.set_title("bending moment |M| (from C¹ Hessian)")
    fig.colorbar(tcf, ax=ax1, shrink=0.85)
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(yv[order], kappa_nn[order], "-o", color="#c0392b", ms=3, label="computed ∂²w/∂n² (freed DOF)")
    ax2.axhline(0.0513, color="k", ls="--", lw=1.5, label="Timoshenko edge moment 0.0513")
    ax2.set_xlabel("y along the clamped edge x = 0")
    ax2.set_ylabel("boundary curvature ∂²w/∂n²")
    ax2.set_title("clamp reaction moment")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[2] / "assets" / "clamped_kirchhoff_plate_2d.png", dpi=130, bbox_inches="tight")

    print("\nOK: the uniform-load clamped plate matches Timoshenko's classical coefficients, and the freed")
    print("    boundary curvature carries the correct clamp reaction moment.")


if __name__ == "__main__":
    main()
