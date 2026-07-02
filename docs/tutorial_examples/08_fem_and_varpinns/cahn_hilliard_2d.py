"""Cahn–Hilliard phase separation — transient, nonlinear, 4th-order — with the Argyris C¹ element.

The Cahn–Hilliard equation is the gradient flow of the Ginzburg–Landau free energy

    E[u] = ∫ [ ¼(u²−1)²  +  (κ/2)|∇u|² ] dx      (double-well bulk + interface penalty)

under the mass-conserving H⁻¹ metric: ``∂ₜu = Δμ``, ``μ = u³ − u − κΔu`` (mobility 1). Eliminating μ gives a
single **4th-order** PDE ``∂ₜu = Δ(u³−u) − κΔ²u`` whose primal weak form needs an **H²-conforming** space —
exactly what Argyris provides. With no-flux natural BCs the weak form is (trial/test u, v):

    ∫ ∂ₜu·v  +  ∫ (3u²−1)∇u·∇v  +  κ ∫ Δu·Δv  =  0 .

Three jno.fem features compose here at once: a **transient** time term, a genuinely **nonlinear** term
(``(3u²−1)∇u·∇v``, driving Newton), and the **biharmonic** ``∫Δu·Δv`` (the C¹ element). Because the
biharmonic is ``h⁻⁴``-conditioned, the default matrix-free Newton–Krylov step is slow; we pass a
**bring-your-own integrator** (``solve_fn``) that does backward Euler with a *dense direct* Newton solve per
step — a ~100× speed-up and the recommended pattern for a stiff 4th-order problem.

Cahn–Hilliard has two exact invariants that make it an ideal verification target — no manufactured solution
needed, the physics *is* the test:

1. **Mass conservation** — taking ``v ≡ 1`` kills every spatial term, so ``d/dt ∫u = 0``.
2. **Energy dissipation** — ``dE/dt = −∫|∇μ|² ≤ 0``: ``E`` is a strict Lyapunov functional.

We drive a ``+1`` droplet in a ``−1`` sea; curvature-driven coarsening shrinks it while the discrete solution
conserves mass to machine precision and dissipates the discrete free energy monotonically.

References:
* J.W. Cahn, J.E. Hilliard, "Free energy of a nonuniform system. I. Interfacial free energy",
  J. Chem. Phys. 28 (1958) 258–267.
* J.H. Argyris, I. Fried, D.W. Scharpf (1968); R.C. Kirby, SMAI J. Comput. Math. 4 (2018) — the C¹ element.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
# This transient solve repeatedly assembles a dense Jacobian and factorises it directly — memory-heavy on a
# GPU but fast on the CPU (~15 s), which is the right device for a dense-direct stepper. Default to CPU so it
# is robust on a small/contended GPU; an explicit JAX_PLATFORMS still wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely.geometry import box

import jno
from jno.utils.solver.fem_elements import argyris_pushforward, argyris_triangle
from jno.utils.solver.fem_topology import BASIX_TRIANGLE_EDGES, build_edge_topology

PI = np.pi
laplacian = jno.np.laplacian
KAPPA = 0.03  # interface energy; width ~ sqrt(2κ) ≈ 0.24, resolved by the quintic element on this mesh
T_END = 0.06  # final time (matches the `time=(0, T_END, ...)` window below)

_DUMMY = jno.domain.from_array({"_": np.zeros((1, 1))})


def _eval(out):
    if isinstance(out, jax.Array):
        return np.asarray(out)
    return np.asarray(jno.core([out.mean], domain=_DUMMY).eval([out]))


def direct_newton(block, args, save_ts):
    """A **bring-your-own** transient integrator: backward Euler with a *dense direct* Newton solve per
    step. The biharmonic operator is ``h⁻⁴``-conditioned, so jno.fem's default matrix-free Newton–Krylov
    needs a great many Krylov iterations; a direct solve of the (small, dense) step Jacobian is far faster
    and rock-solid. Built only from the block's own pieces — ``mass`` / ``residual`` / ``jacobian`` /
    ``state0`` — exactly as :meth:`SemidiscreteTimeBlock.solve` documents."""
    from jax import lax

    s0 = jnp.asarray(block.state0).reshape(-1)
    dt = float(block.dt)
    grid = jnp.asarray(save_ts, s0.dtype)

    def _dense(x):
        return jnp.asarray(x.todense() if hasattr(x, "todense") else x)

    def step(uprev, t_next):
        mass = _dense(block.mass(t_next, args))

        def newton(_i, un):  # G(uₙ) = M(uₙ − uₚ)/dt + R(uₙ) = 0
            g = mass @ (un - uprev) / dt + jnp.asarray(block.residual(un, t_next, args)).reshape(-1)
            jac = mass / dt + _dense(block.jacobian(un, t_next, args))
            return un + jnp.linalg.solve(jac, -g)

        un = lax.fori_loop(0, 4, newton, uprev)  # 4 Newton iterations converge this problem
        return un, un

    _, ys = lax.scan(step, s0, grid[1:])
    return ys


def solve_cahn_hilliard():
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.11, time=(0.0, T_END, 10))
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    u, phi = d.fem_symbols(space="Argyris")
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    r = jno.np.sqrt((ci[0] - 0.5) ** 2 + (ci[1] - 0.5) ** 2)
    u0 = jno.np.tanh((0.28 - r) / 0.15)  # a +1 droplet (radius 0.28) in a -1 sea
    form = (
        ui.t * vi  # transient
        + (3.0 * ui * ui - 1.0) * (ui.x * vi.x + ui.y * vi.y)  # nonlinear: ∇(u³-u)·∇v
        + KAPPA * laplacian(ui, [xi, yi]) * laplacian(vi, [xi, yi])  # biharmonic (C¹)
    )
    fem = jno.fem([form, u(ci[0], ci[1]) - u0])
    assert fem.is_transient and not fem.is_linear, "Cahn–Hilliard must be a nonlinear transient problem"
    traj = _eval(fem.solve(solve_fn=direct_newton))  # (n_steps, ndof)
    return d, traj


def reconstruct(d):
    """Per-cell shape data (value + gradient at quad points) and the mass matrix, for the invariants."""
    pts = np.asarray(d.mesh.points)[:, :2]
    cells = np.asarray(d.mesh.cells_dict["triangle"]).astype(np.int64)
    top = build_edge_topology(cells, BASIX_TRIANGLE_EDGES)
    nv, nc = pts.shape[0], cells.shape[0]
    spec = argyris_triangle(quad_degree=8)
    qw = np.asarray(spec.quad_weights)
    rv, rg, rh = jnp.asarray(spec.ref_values), jnp.asarray(spec.ref_grads), jnp.asarray(spec.ref_hess)
    nodal = tuple(jnp.asarray(a) for a in spec.ref_aux)
    ev = np.asarray(top.edge_vertices)
    dv = pts[ev[:, 1]] - pts[ev[:, 0]]
    enrm = np.stack([-dv[:, 1], dv[:, 0]], axis=1)
    enrm /= np.linalg.norm(enrm, axis=1, keepdims=True)
    cell_en = enrm[np.asarray(top.cell_edges)]
    verts = pts[cells]
    J = np.stack([verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0]], axis=-1)
    detJ = np.linalg.det(J)
    pf = jax.vmap(lambda Jc, dJ, en: argyris_pushforward(rv, rg, rh, Jc, dJ, en, nodal), in_axes=(0, 0, 0))
    phi, grad, _h = pf(jnp.asarray(J), jnp.asarray(detJ), jnp.asarray(cell_en))
    vdofs = (6 * cells[:, :, None] + np.arange(6)).reshape(nc, 18)
    cdofs = np.concatenate([vdofs, 6 * nv + np.asarray(top.cell_edges)], axis=1)
    wq = qw[None, :] * np.abs(detJ)[:, None]
    return dict(pts=pts, cells=cells, nv=nv, phi=np.asarray(phi), grad=np.asarray(grad), cdofs=cdofs, wq=wq)


def mass_and_energy(rec, u_flat):
    cc = u_flat[rec["cdofs"]]  # (nc, 21) local DOFs per cell
    uh = np.einsum("cqn,cn->cq", rec["phi"], cc)  # u at quad points
    gh = np.einsum("cqnd,cn->cqd", rec["grad"], cc)  # ∇u at quad points
    mass = float(np.sum(rec["wq"] * uh))  # ∫u dx
    energy = float(np.sum(rec["wq"] * (0.25 * (uh**2 - 1.0) ** 2 + 0.5 * KAPPA * np.sum(gh**2, axis=2))))
    return mass, energy


def main():
    d, traj = solve_cahn_hilliard()
    rec = reconstruct(d)
    mass = np.array([mass_and_energy(rec, traj[k])[0] for k in range(traj.shape[0])])
    energy = np.array([mass_and_energy(rec, traj[k])[1] for k in range(traj.shape[0])])
    vmax = np.array([float(np.abs(traj[k][6 * np.arange(rec["nv"])]).max()) for k in range(traj.shape[0])])

    print("\nCahn–Hilliard (Argyris C¹) — droplet coarsening:")
    print(f"  steps={traj.shape[0]}  dofs={traj.shape[1]}")
    print(f"  mass ∫u:  {mass[0]:.6f} → {mass[-1]:.6f}   (drift {np.max(np.abs(mass - mass[0])):.2e})")
    print(f"  energy E: {energy[0]:.5f} → {energy[-1]:.5f}   (monotone: {bool(np.all(np.diff(energy) < 1e-10))})")
    print(f"  max|u|:   {vmax.max():.3f}  (bounded — no overshoot past the ±1 wells)")

    # --- the two rigorous Cahn–Hilliard invariants + physicality ---
    assert np.all(np.isfinite(traj)), "trajectory must stay finite"
    assert np.max(np.abs(mass - mass[0])) < 1e-8, f"mass ∫u must be conserved, drift {np.max(np.abs(mass - mass[0])):.2e}"
    assert np.all(np.diff(energy) < 1e-10), f"free energy must dissipate monotonically, E={energy}"
    assert energy[-1] < 0.9 * energy[0], "the droplet must actually coarsen (energy drop)"
    assert vmax.max() < 1.1, f"u must stay near the ±1 wells (no numerical overshoot), got max|u|={vmax.max():.3f}"

    # --- figure: the two invariants + the coarsening field ---
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    # the backward-Euler scan returns states at t = dt, 2·dt, …, T_END (the IC at t=0 is not in `traj`)
    tt = np.linspace(0.0, T_END, traj.shape[0] + 1)[1:]
    ax0.plot(tt, energy, "o-", color="#c0392b", label="free energy E(t)")
    ax0.set_xlabel("time t")
    ax0.set_ylabel("free energy E", color="#c0392b")
    ax0.tick_params(axis="y", labelcolor="#c0392b")
    axm = ax0.twinx()
    axm.plot(tt, mass, "s--", color="#2471a3", label="mass ∫u")
    axm.set_ylabel("mass ∫u", color="#2471a3")
    axm.tick_params(axis="y", labelcolor="#2471a3")
    axm.set_ylim(mass[0] - 0.05, mass[0] + 0.05)
    ax0.set_title("E dissipates monotonically; mass is conserved")

    tri = mtri.Triangulation(rec["pts"][:, 0], rec["pts"][:, 1], rec["cells"])
    uf = traj[-1][6 * np.arange(rec["nv"])]
    tcf = ax1.tricontourf(tri, uf, levels=np.linspace(-1, 1, 21), cmap="coolwarm")
    ax1.tricontour(tri, uf, levels=[0.0], colors="k", linewidths=1.2)  # the interface u = 0
    ax1.set_aspect("equal")
    ax1.set_title(f"u at t = {T_END:.3f} (interface in black)")
    fig.colorbar(tcf, ax=ax1, shrink=0.85, label="u")
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[2] / "assets" / "cahn_hilliard_2d.png", dpi=130, bbox_inches="tight")

    print("\nOK: mass conserved to machine precision and the free energy is a monotone Lyapunov functional.")


if __name__ == "__main__":
    main()
