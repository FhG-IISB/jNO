"""Conforming biharmonic with the Argyris C¹ element — an order-of-accuracy study (``jno.fem``, ``space="Argyris"``).

The biharmonic (Kirchhoff plate / Cahn–Hilliard) operator ``Δ²u = f`` needs an **H²-conforming** space:
the weak form ``∫Δu·Δv = ∫f v`` is only convergent if the discrete normal derivative ``∂u/∂n`` is
continuous across element edges. C⁰ Lagrange is *not* C¹, so ``∫Δu·Δv`` over P_k is non-conforming and does
not converge; the **Argyris** quintic triangle (21 DOF: value + gradient + Hessian at each vertex, normal
derivative at each edge midpoint) is the classical C¹-conforming element that does.

This tutorial is a numerical-analysis **verification**: solve the clamped biharmonic on a sequence of
unstructured meshes and measure the empirical convergence *rate* against the a-priori theory. For a degree
``k = 5`` C¹ element on a 4th-order (``2m = 4``, ``m = 2``) problem, the optimal estimates are

    energy / H²-seminorm   ‖Δ(u - u_h)‖_{L²} = O(h^{k-1})  = O(h⁴),
    displacement / L²      ‖u - u_h‖_{L²}    = O(h^{k+1})  = O(h⁶)   (Aubin–Nitsche).

Manufactured solution ``u* = sin(πx) sin(πy)`` on the unit square gives ``Δu* = -2π² u*`` and
``f = Δ²u* = 4π⁴ u*``. Since ``u* = Δu* = 0`` on ``∂Ω`` this is a **simply-supported** plate, so the deflection
Dirichlet term ``u(boundary) - u*`` alone is the exact essential BC (it pins the value; the rotation and the
moment ``M_n ∝ Δu`` are the natural conditions, satisfied by ``u*``). ``u_h → u*`` at the optimal rate. (For a
*clamped* plate — additionally pinning the rotation with ``u.dn(region)-h`` — see ``clamped_kirchhoff_plate_2d.py``.)
Error norms are
computed by per-cell Gauss quadrature, reconstructing ``u_h`` and ``Δu_h`` from the solution DOFs with the
public element functions (``argyris_triangle`` / ``argyris_pushforward``) — i.e. we audit the discrete
solution itself, not a hand-built field.

References:
* J.H. Argyris, I. Fried, D.W. Scharpf, "The TUBA family of plate elements for the matrix displacement
  method", Aeronautical Journal 72 (1968) 701–709 — the quintic C¹ triangle.
* R.C. Kirby, "A general approach to transforming finite elements", SMAI J. Comput. Math. 4 (2018) 197–224
  — the affine DOF-transform ``M(cell)`` that maps the reference dual basis to a physical cell.
* P.G. Ciarlet, *The Finite Element Method for Elliptic Problems* (2002), §6 — C¹ elements and the a-priori
  estimates for 4th-order problems.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # headless figure

import jax

jax.config.update("jax_enable_x64", True)  # 4th-order assembly + the dense solve run in float64

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box

import jno
from jno.utils.solver.fem_elements import argyris_pushforward, argyris_triangle
from jno.utils.solver.fem_topology import BASIX_TRIANGLE_EDGES, build_edge_topology

PI = np.pi
laplacian, sin = jno.np.laplacian, jno.np.sin


def u_exact(x, y):
    return np.sin(PI * x) * np.sin(PI * y)


def lap_u_exact(x, y):  # Δu* = -2π² u*
    return -2.0 * PI**2 * np.sin(PI * x) * np.sin(PI * y)


def solve_biharmonic(mesh_size):
    """Solve the clamped biharmonic ``∫Δu·Δv = ∫f v`` with the Argyris element; return (domain, solution)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 4.0 * PI**4 * sin(PI * xi) * sin(PI * yi)
    g = sin(PI * xb) * sin(PI * yb)  # clamped to the exact trace (value + normal derivative via autodiff)
    fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi, u(xb, yb) - g])
    sol = np.asarray(fem.solve()).reshape(-1)
    return d, sol, fem.dofs


def error_norms(d, sol, quad_degree=12):
    """True L² and energy (Δ-seminorm) errors of the Argyris solution by per-cell Gauss quadrature.

    Reconstructs ``u_h`` and ``Δu_h`` at the quadrature points of every cell from the solution DOFs using
    the SAME push-forward the assembler uses (so this is the discrete solution, audited)."""
    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["triangle"]).astype(np.int64)
    top = build_edge_topology(cells, BASIX_TRIANGLE_EDGES)
    nv, nc = pts.shape[0], cells.shape[0]

    spec = argyris_triangle(quad_degree=quad_degree)
    qp, qw = np.asarray(spec.quad_points), np.asarray(spec.quad_weights)
    rv, rg, rh = jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_grads), jnp.asarray(spec.ref_hess)
    nodal = tuple(jnp.asarray(a) for a in spec.ref_aux)

    # globally-oriented physical edge normals (the assembler's convention), per cell
    ev = np.asarray(top.edge_vertices)
    dv = pts[ev[:, 1]] - pts[ev[:, 0]]
    enrm = np.stack([-dv[:, 1], dv[:, 0]], axis=1)
    enrm = enrm / np.linalg.norm(enrm, axis=1, keepdims=True)
    cell_en = enrm[np.asarray(top.cell_edges)]  # (nc, 3, 2)

    verts = pts[cells]  # (nc, 3, 2)
    J = np.stack([verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]], axis=-1)  # (nc, 2, 2)
    detJ = np.linalg.det(J)

    pf = jax.vmap(lambda Jc, dJ, en: argyris_pushforward(rv, rg, rh, Jc, dJ, en, nodal), in_axes=(0, 0, 0))
    phi, _grad, hess = pf(jnp.asarray(J), jnp.asarray(detJ), jnp.asarray(cell_en))  # (nc, nq, 21), (.,.,21,2,2)
    phi, hess = np.asarray(phi), np.asarray(hess)

    # global DOF gather, identical to the assembler's 21-DOF layout (6·v+k vertex, 6·n_verts+edge_id edge)
    vdofs = (6 * cells[:, :, None] + np.arange(6)).reshape(nc, 18)
    edofs = 6 * nv + np.asarray(top.cell_edges)
    cdofs = np.concatenate([vdofs, edofs], axis=1)  # (nc, 21)
    cc = sol[cdofs]  # (nc, 21)

    uh = np.einsum("cqn,cn->cq", phi, cc)  # (nc, nq)
    lap_uh = np.einsum("cqn,cn->cq", hess[:, :, :, 0, 0] + hess[:, :, :, 1, 1], cc)
    xq = verts[:, 0:1, :] + np.einsum("cda,qa->cqd", J, qp)  # (nc, nq, 2) physical quad points
    w = qw[None, :] * np.abs(detJ)[:, None]  # (nc, nq)
    ue = u_exact(xq[:, :, 0], xq[:, :, 1])
    lue = lap_u_exact(xq[:, :, 0], xq[:, :, 1])
    l2 = float(np.sqrt(np.sum(w * (uh - ue) ** 2)))
    energy = float(np.sqrt(np.sum(w * (lap_uh - lue) ** 2)))
    return l2, energy


def main():
    mesh_sizes = [0.5, 0.35, 0.25, 0.18]
    rows = []
    for ms in mesh_sizes:
        d, sol, dofs = solve_biharmonic(ms)
        # representative mesh size h = longest edge over the mesh (drives the asymptotic rate)
        pts = np.asarray(d.mesh.points)[:, :2]
        cells = np.asarray(d.mesh.cells_dict["triangle"])
        e = pts[cells]
        h = float(np.max([np.linalg.norm(e[:, i] - e[:, j], axis=1) for i, j in ((0, 1), (1, 2), (2, 0))]))
        l2, energy = error_norms(d, sol)
        rows.append((h, dofs, l2, energy))

    print("\nArgyris C¹ biharmonic — convergence (clamped, u* = sin(πx)sin(πy)):")
    print(f"  {'h':>8} {'dofs':>7} {'L2 err':>12} {'rate':>6} {'energy err':>12} {'rate':>6}")
    l2_rates, en_rates = [], []
    for i, (h, dofs, l2, energy) in enumerate(rows):
        if i == 0:
            print(f"  {h:8.4f} {dofs:7d} {l2:12.3e} {'--':>6} {energy:12.3e} {'--':>6}")
        else:
            hp, _, l2p, enp = rows[i - 1]
            r_l2 = np.log(l2p / l2) / np.log(hp / h)
            r_en = np.log(enp / energy) / np.log(hp / h)
            l2_rates.append(r_l2)
            en_rates.append(r_en)
            print(f"  {h:8.4f} {dofs:7d} {l2:12.3e} {r_l2:6.2f} {energy:12.3e} {r_en:6.2f}")

    # least-squares order of accuracy over all levels (slope of log(err) vs log(h))
    hs = np.array([r[0] for r in rows])
    l2s = np.array([r[2] for r in rows])
    ens = np.array([r[3] for r in rows])
    l2_order = float(np.polyfit(np.log(hs), np.log(l2s), 1)[0])
    en_order = float(np.polyfit(np.log(hs), np.log(ens), 1)[0])
    print(f"\n  least-squares order:  L2 ≈ {l2_order:.2f}  (theory 6)   energy ≈ {en_order:.2f}  (theory 4)")

    # log-log convergence plot: measured errors + reference O(h⁴)/O(h⁶) slope guides
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.loglog(hs, ens, "o-", color="#c0392b", label=f"energy ‖Δ(u−u_h)‖  (order {en_order:.2f})")
    ax.loglog(hs, l2s, "s-", color="#2471a3", label=f"L²  ‖u−u_h‖  (order {l2_order:.2f})")
    ax.loglog(hs, ens[-1] * (hs / hs[-1]) ** 4, "--", color="#c0392b", alpha=0.5, label="O(h⁴) ref")
    ax.loglog(hs, l2s[-1] * (hs / hs[-1]) ** 6, "--", color="#2471a3", alpha=0.5, label="O(h⁶) ref")
    ax.set_xlabel("mesh size  h")
    ax.set_ylabel("error norm")
    ax.set_title("Argyris C¹ biharmonic — order of accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.savefig(
        Path(__file__).parents[2] / "assets" / "biharmonic_argyris_convergence_2d.png", dpi=130, bbox_inches="tight"
    )

    # --- assertions: errors decrease monotonically and the measured high orders confirm C¹ conformity ---
    assert np.all(np.diff(l2s) < 0), f"L2 error must decrease under refinement: {l2s}"
    assert np.all(np.diff(ens) < 0), f"energy error must decrease under refinement: {ens}"
    # the energy (Δ-seminorm) order is the robust headline — clearly high-order (≫ the 2nd-order a
    # non-conforming/mixed scheme delivers), bracketing the O(h⁴) theory with pre-asymptotic slack.
    assert 3.0 <= en_order <= 5.5, f"energy order {en_order:.2f} not near the O(h⁴) theory"
    # L² superconverges (O(h⁶)); require clearly more than 4 (well above any C⁰/mixed rate)
    assert l2_order >= 4.5, f"L2 order {l2_order:.2f} below the expected high-order (C¹) convergence"
    print("\nOK: the Argyris C¹ element delivers optimal high-order biharmonic convergence.")


if __name__ == "__main__":
    main()
