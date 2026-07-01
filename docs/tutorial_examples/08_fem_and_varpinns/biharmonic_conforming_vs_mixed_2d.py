"""Why C¹? Conforming Argyris vs. the non-conforming mixed method on the biharmonic (``jno.fem``).

A 4th-order operator $\\Delta^2 u = f$ has two routes through `jno.fem`:

1. **Conforming** — the $C^1$ **Argyris** element ($H^2$-conforming): write $\\int\\Delta u\\,\\Delta v$
   directly. Optimal order $k+1 = 6$ in $L^2$ for the quintic.
2. **Mixed (Ciarlet–Raviart)** — split into two 2nd-order problems with an auxiliary $w = \\Delta u$ and
   coupled C⁰ Lagrange fields (here $P_2$). No $C^1$ element needed, but the auxiliary variable costs
   accuracy: the displacement converges at a *reduced* rate.

This tutorial measures both on the **same** manufactured problem. With $u^\\ast = \\sin(\\pi x)\\sin(\\pi y)$
on the unit square, $f = \\Delta^2 u^\\ast = 4\\pi^4 u^\\ast$, and $u^\\ast = \\Delta u^\\ast = 0$ on
$\\partial\\Omega$ (simply supported — natural for the mixed pair, and a valid known-trace datum for
Argyris). The robust, honest headline is **accuracy at equal cost**: for a comparable number of DOFs the
conforming $C^1$ element is two-to-three orders of magnitude more accurate, and the gap widens under
refinement (it converges faster too). That is exactly why the $C^1$ element earns its extra machinery. We
assert only these *relative* facts; the asymptotic order of Argyris is measured in the dedicated
convergence tutorial.

References:
* P.G. Ciarlet, P.-A. Raviart, "A mixed finite element method for the biharmonic equation", in *Mathematical
  Aspects of Finite Elements in PDE* (1974) 125–145 — the mixed method.
* J.H. Argyris, I. Fried, D.W. Scharpf (1968); R.C. Kirby, SMAI J. Comput. Math. 4 (2018) — the C¹ element.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from shapely.geometry import box

import jno

PI = np.pi
laplacian, sin = jno.np.laplacian, jno.np.sin


def dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def nodal_l2(uh, pts):
    ue = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    return float(np.sqrt(np.mean((uh - ue) ** 2)))


def solve_argyris(mesh_size):
    """C¹ conforming: ∫Δu·Δv = ∫f v, clamped to the known u*. Displacement = vertex value DOFs."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 4.0 * PI**4 * sin(PI * xi) * sin(PI * yi)
    g = sin(PI * xb) * sin(PI * yb)
    fem = jno.fem([laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi]) - f * vi, u(xb, yb) - g])
    sol = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(d.mesh.points)[:, :2]
    nv = pts.shape[0]
    uh = sol[6 * np.arange(nv)]  # value DOFs at the vertices
    return nodal_l2(uh, pts), int(fem.dofs)


def solve_mixed(mesh_size):
    """Ciarlet–Raviart mixed (coupled P2): w = Δu, Δw = f; simply supported u = w = 0 on ∂Ω."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    u, p = d.fem_symbols(order=2, names=("u", "p"))
    w, q = d.fem_symbols(order=2, names=("w", "q"))
    ui, pi = u.bind(x=xi, y=yi), p.bind(x=xi, y=yi)
    wi, qi = w.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    f = 4.0 * PI**4 * sin(PI * xi) * sin(PI * yi)
    fem = jno.fem(
        [
            (ui.x * pi.x + ui.y * pi.y) + wi * pi,  # ∫∇u·∇p + ∫w p = 0   (w = Δu, weak; u-trial first ⇒ u=field 0)
            wi.x * qi.x + wi.y * qi.y + f * qi,  # ∫∇w·∇q + ∫f q = 0    (Δw = f, weak)
            u(xb, yb) - 0.0,
            w(xb, yb) - 0.0,
        ]
    )
    sol = np.linalg.solve(dense(fem.A), np.asarray(fem.b).reshape(-1))
    off = fem.offsets
    uh = sol[off[0] : off[1]]
    pts_u = np.asarray(fem.field_points[0])
    return nodal_l2(uh, pts_u), int(off[-1])


def order(hs, errs):
    return float(np.polyfit(np.log(hs), np.log(errs), 1)[0])


def main():
    mesh_sizes = [0.42, 0.3, 0.21]
    hs, arg_err, arg_dof, mix_err, mix_dof = [], [], [], [], []
    for ms in mesh_sizes:
        # representative h (target size is monotone in the actual mesh; use it as the abscissa)
        hs.append(ms)
        e_a, n_a = solve_argyris(ms)
        e_m, n_m = solve_mixed(ms)
        arg_err.append(e_a)
        arg_dof.append(n_a)
        mix_err.append(e_m)
        mix_dof.append(n_m)

    hs = np.array(hs)
    arg_err = np.array(arg_err)
    mix_err = np.array(mix_err)
    arg_order = order(hs, arg_err)
    mix_order = order(hs, mix_err)

    ratios = mix_err / arg_err  # how many times more accurate Argyris is, at comparable cost
    print("\nBiharmonic Δ²u = f, u* = sin(πx)sin(πy) — displacement nodal-L² error at comparable cost:")
    print(f"  {'h':>6} | {'Argyris dofs':>12} {'L2 err':>11} | {'mixed dofs':>10} {'L2 err':>11} | {'Argyris/mixed':>13}")
    for i, h in enumerate(hs):
        print(
            f"  {h:6.3f} | {arg_dof[i]:12d} {arg_err[i]:11.3e} | {mix_dof[i]:10d} {mix_err[i]:11.3e} | {ratios[i]:11.0f}×"
        )
    print(
        f"\n  measured nodal order (pre-asymptotic, coarse): Argyris (C¹) ≈ {arg_order:.2f}   mixed (P2) ≈ {mix_order:.2f}"
    )
    print("  (the dedicated convergence tutorial measures Argyris's asymptotic L² order ≈ 6 on finer meshes.)")

    # The robust, honest headline is *accuracy at equal cost*: for a comparable DOF count the conforming
    # C¹ element is orders of magnitude more accurate, and it also converges faster. We assert only these
    # relative facts (no delicate absolute-rate claim on coarse, nodal-norm data).
    assert np.all(np.diff(arg_err) < 0), f"Argyris error must decrease: {arg_err}"
    assert np.all(np.diff(mix_err) < 0), f"mixed error must decrease: {mix_err}"
    assert np.all(arg_dof <= 1.3 * np.array(mix_dof)), "Argyris must not use materially more DOFs than the mixed pair"
    assert np.all(ratios > 20.0), f"conforming C¹ must be >20× more accurate at comparable cost, got {ratios}"
    assert arg_order > mix_order, (
        f"conforming C¹ must converge faster than the mixed method: {arg_order:.2f} vs {mix_order:.2f}"
    )
    print(
        f"\nOK: at comparable cost the conforming C¹ element is {ratios.min():.0f}–{ratios.max():.0f}× more accurate and converges faster."
    )


if __name__ == "__main__":
    main()
