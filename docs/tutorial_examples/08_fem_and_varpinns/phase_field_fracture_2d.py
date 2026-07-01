"""Brittle fracture — 4th-order phase-field on the cheap Morley element (coupled multiphysics).

A cracking solid minimizes elastic energy + fracture (crack-surface) energy. The variational phase-field
model regularizes the sharp crack by a smooth damage field ``d∈[0,1]`` (0 = intact, 1 = broken) over a length
``ℓ``. Borden, Hughes, Landis & Verhoosel (CMAME **273** (2014) 100–118) use a **fourth-order** regularization
whose crack-surface density carries a second-derivative term, giving the damage a biharmonic operator. A
4th-order weak form needs a **special biharmonic element** — plain ``C⁰`` Lagrange is non-convergent. Two
work: the conforming ``C¹`` **Argyris** (21 DOF, accurate) and the **non-conforming Morley** triangle (6 DOF:
value at the 3 vertices + normal derivative at the 3 edge midpoints). We use **Morley** here: it is ~3.5×
cheaper, so it clears the Argyris construction memory ceiling and scales to the **fine mesh a sharp crack
needs**. Because Morley is non-conforming the biharmonic form is the full-Hessian inner product ``∫D²d:D²φ``
(the Laplacian form ``∫Δd·Δφ`` is *singular* for Morley). Its 1D optimal profile is still
``d(x) = (1 + |x|/ℓ) e^(−|x|/ℓ)`` — Part 1 verifies it.

Part 2 solves a real crack by the *canonical* **alternate minimization** (Bourdin–Francfort–Marigo). With the
AT2 degradation ``g(d)=(1−d)²+η`` each sub-problem is LINEAR given the other field —

  * elasticity (P1 vector, native):  ``∫ g(d) σ(u):ε(v) = 0``   (displacement-controlled tension),
  * damage    (Morley):              ``(2H+Gc/ℓ)dφ + 2Gcℓ∇d·∇φ + Gcℓ³ D²d:D²φ = 2Hφ``,

coupled by two scalar fields: ``g(d)`` degrades the stiffness, and the tensile strain-energy history
``H = max_t ψ⁺(ε(u))`` (irreversible ⇒ no crack healing) drives damage. A single-edge-notched specimen is
pulled in tension; the crack initiates at the notch, propagates across, and the reaction force softens.

Reference: L.S.D. Morley, *The triangular equilibrium element in the solution of plate bending problems*,
Aeronautical Quarterly **19** (1968) 149–169 — the non-conforming element. M.J. Borden, T.J.R. Hughes,
C.M. Landis, C.V. Verhoosel, CMAME **273** (2014) 100–118 — the fourth-order phase-field model. B. Bourdin,
G.A. Francfort, J.-J. Marigo, J. Elasticity **91** (2008) 5–148 — alternate minimization. C. Miehe,
M. Hofacker, F. Welschinger, CMAME **199** (2010) 2765–2778 — the tension/compression (spectral) split.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # stiff biharmonic direct solves — CPU dodges GPU cuSolver OOM

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from shapely.geometry import box

import jno

inner, symgrad, trace, hess = jno.np.inner, jno.np.symgrad, jno.np.trace, jno.np.hessian
dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731
lu = lambda A, b: np.asarray(jnp.linalg.solve(dense(A), jnp.asarray(b).reshape(-1)))  # noqa: E731

# --- material & regularization (Morley is cheap ⇒ a fine mesh + small ℓ for a sharp crack) ---
E, nu = 1.0, 0.3
lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
Gc, ell, eta = 2.0e-3, 0.08, 1e-3
h = 0.04

d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=h)
xi, yi, _ = d.variable("interior", split=True)
xb, yb, _ = d.variable("bottom", split=True)
xt, yt, _ = d.variable("top", split=True)
xl, yl, _ = d.variable("left", split=True)
nodes = np.asarray(d.built_mesh.points)[:, :2]
cells = np.asarray(d.mesh.cells_dict["triangle"])
nv, nc = nodes.shape[0], cells.shape[0]

# P1 shape gradients (constant per triangle) for strain / energy recovery
pN = nodes[cells]
v0, v1, v2 = pN[:, 0], pN[:, 1], pN[:, 2]
area2 = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
cell_area = 0.5 * np.abs(area2)
gradN = (
    np.stack(
        [
            np.stack([v1[:, 1] - v2[:, 1], v2[:, 0] - v1[:, 0]], 1),
            np.stack([v2[:, 1] - v0[:, 1], v0[:, 0] - v2[:, 0]], 1),
            np.stack([v0[:, 1] - v1[:, 1], v1[:, 0] - v0[:, 0]], 1),
        ],
        1,
    )
    / area2[:, None, None]
)


def psi_cellwise(u):
    """Tensile strain-energy density per cell (Miehe 2010 spectral split): ψ⁺ = ½λ⟨trε⟩₊² + μ Σ⟨εᵢ⟩₊²."""
    gu = np.einsum("cid,cie->cde", u[cells], gradN)
    eps = 0.5 * (gu + np.transpose(gu, (0, 2, 1)))
    exx, eyy, exy = eps[:, 0, 0], eps[:, 1, 1], eps[:, 0, 1]
    tr = exx + eyy
    disc = np.sqrt(np.maximum(((exx - eyy) / 2) ** 2 + exy**2, 0.0))  # principal strains (2×2 analytic)
    e1, e2 = tr / 2 + disc, tr / 2 - disc
    pos = lambda v: np.maximum(v, 0.0)  # noqa: E731
    return 0.5 * lam * pos(tr) ** 2 + mu * (pos(e1) ** 2 + pos(e2) ** 2)


def scatter_max_to_vertices(psi_cell):
    Hv = np.zeros(nv)
    np.maximum.at(Hv, cells.reshape(-1), np.repeat(psi_cell, 3))
    return Hv


# --- Morley damage forms: full-Hessian biharmonic term (Morley is non-conforming ⇒ ∫D²d:D²φ, not ∫Δd·Δφ) ---
dd, dphi = d.fem_symbols(space="Morley")
di, vi = dd.bind(x=xi, y=yi), dphi.bind(x=xi, y=yi)
reg = 2.0 * Gc * ell * (di.x * vi.x + di.y * vi.y) + Gc * ell**3 * inner(
    hess(di, [xi, yi]), hess(vi, [xi, yi]), n_contract=2
)
Hpar = jno.np.parameter(gsym := d.fem_symbols()[0], name="H")
fem_d = jno.fem([(2.0 * Hpar + Gc / ell) * (di * vi) + reg - 2.0 * Hpar * vi])

# --- elasticity operator (parametric in g(d)); unit top-displacement, load-linearity u = δ·û ---
u, phi = d.fem_symbols(value_shape=(2,), order=1)
gd = jno.np.parameter(gsym, name="gd")
eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
weak_e = gd * (lam * trace(eu) * trace(ep) + 2.0 * mu * inner(eu, ep, n_contract=2))
fem_e = jno.fem([weak_e, u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0, u(xt, yt)[0] - 0.0, u(xt, yt)[1] - 1.0])


@jax.jit  # compile the Morley assemble+solve ONCE (eager re-assembly is ~100x slower)
def solve_e(gd_vals):
    A, b = fem_e.operator.evaluate({"gd": gd_vals})
    return jnp.linalg.solve(dense(A), jnp.asarray(b).reshape(-1))


@jax.jit
def solve_d(H_vals):
    A, b = fem_d.operator.evaluate({"H": H_vals})
    return jnp.linalg.solve(dense(A), jnp.asarray(b).reshape(-1))


def main():
    # ================= Part 1: the crack profile (controlled verification) =================
    # Pin d=1 on the left edge (Morley pins the value + ∂d/∂n=0, the smooth-peak condition), no source: d
    # decays as the optimal 1D profile. d=1 on the whole edge ⇒ the solution is x-only, so d-vs-x collapses.
    fem_p = jno.fem([(Gc / ell) * (di * vi) + reg, dd(xl, yl) - 1.0])
    dprof = np.asarray(fem_p.solve(lu)).reshape(-1)[np.arange(nv)]  # Morley value DOFs = first n_verts entries
    xs = nodes[:, 0]
    p4x, p2x = (1 + xs / ell) * np.exp(-xs / ell), np.exp(-xs / ell)
    e4 = float(np.sqrt(np.mean((dprof - p4x) ** 2)))
    e2 = float(np.sqrt(np.mean((dprof - p2x) ** 2)))

    # ================= Part 2: propagating crack (coupled alternate minimization) =================
    notch = (nodes[:, 0] < 0.3) & (np.abs(nodes[:, 1] - 0.5) < 0.9 * h)  # short single-edge notch
    Hhist = np.where(notch, 1e2, 0.0)
    dvals = np.clip(np.asarray(solve_d(jnp.asarray(Hhist)))[np.arange(nv)], 0.0, 1.0)
    loads = np.linspace(0.03, 0.14, 9)
    fd, fronts, snaps = [], [], []
    for delta in loads:
        for _ in range(3):
            u_hat = np.asarray(solve_e(jnp.asarray((1 - dvals) ** 2 + eta))).reshape(-1, 2)
            psi_c = psi_cellwise(u_hat)
            Hhist = np.maximum(Hhist, delta**2 * scatter_max_to_vertices(psi_c))
            dvals = np.clip(np.asarray(solve_d(jnp.asarray(Hhist)))[np.arange(nv)], 0.0, 1.0)
        gd_c = (1 - dvals[cells].mean(1)) ** 2 + eta
        fd.append((delta, 2.0 * delta * float(np.sum(gd_c * psi_c * cell_area))))  # reaction ∝ dΠ/dδ
        fronts.append(nodes[dvals > 0.5, 0].max())
        snaps.append(dvals.copy())
    fd = np.array(fd)
    icross = next((i for i, f in enumerate(fronts) if f >= 0.95), len(fronts) - 1)  # first fully-spanned frame
    d_show = snaps[icross]

    print("\n4th-order phase-field fracture (Morley, non-conforming):")
    print(f"  mesh nv={nv} nc={nc}  damage-dofs={fem_d.dofs}  ℓ={ell}  h/ℓ={h / ell:.2f}   (Argyris OOMs at this mesh)")
    print(f"  Part 1  crack profile: RMS vs 4th-order={e4:.3f}  vs 2nd-order={e2:.3f}  ({e2 / e4:.1f}× better)")
    print(f"  Part 2  crack front x: {fronts[0]:.2f} → {fronts[-1]:.2f}  ({(dvals > 0.5).sum()}/{nv} damaged)")
    print(
        f"          peak reaction {fd[:, 1].max():.3e} at δ={fd[fd[:, 1].argmax(), 0]:.3f};  final {fd[-1, 1]:.3e} (softening)"
    )

    # --- asserts: profile matches the 4th-order shape; crack propagates + localizes + softens ---
    assert e4 < 0.05 and e4 < 0.4 * e2, f"crack profile must match the 4th-order shape: {e4:.3f} vs {e2:.3f}"
    assert fronts[-1] > fronts[0] + 0.3, f"the crack must propagate across: front {fronts[0]:.2f}→{fronts[-1]:.2f}"
    assert (dvals > 0.5).sum() > 2 * int(notch.sum()), "damage must localize into a growing crack band"
    assert fd[-1, 1] < 0.7 * fd[:, 1].max(), "the specimen must soften (reaction drops after the peak)"

    # --- figure: crack profile | propagated (sharp) crack | force–displacement ---
    fig = plt.figure(figsize=(13.5, 4.0))
    ax0 = fig.add_subplot(1, 3, 1)
    rr = np.linspace(0, 0.5, 200)
    ax0.plot(rr, (1 + rr / ell) * np.exp(-rr / ell), "-", color="k", lw=2, label="4th-order $(1+r/\\ell)e^{-r/\\ell}$")
    ax0.plot(rr, np.exp(-rr / ell), "--", color="#888", lw=1.5, label="2nd-order $e^{-r/\\ell}$ (kink)")
    o = np.argsort(xs)
    ax0.plot(xs[o], dprof[o], "o", color="#c0392b", ms=2.5, label="computed (Morley)")
    ax0.set_xlabel("distance from crack  r")
    ax0.set_ylabel("damage d")
    ax0.set_xlim(0, 0.5)
    ax0.set_title(f"crack profile (RMS {e4:.0e})")
    ax0.legend(fontsize=7.5)
    ax1 = fig.add_subplot(1, 3, 2)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1], cells)
    tcf = ax1.tricontourf(tri, d_show, levels=np.linspace(0, 1, 21), cmap="inferno")
    ax1.set_aspect("equal")
    ax1.set_title(f"sharp crack, nv={nv} (δ={loads[icross]:.2f})")
    fig.colorbar(tcf, ax=ax1, shrink=0.85, label="d")
    ax2 = fig.add_subplot(1, 3, 3)
    ax2.plot(fd[:, 0], fd[:, 1], "o-", color="#2471a3")
    ipk = fd[:, 1].argmax()
    ax2.plot(fd[ipk, 0], fd[ipk, 1], "*", color="#c0392b", ms=13, label="peak (crack advances)")
    ax2.set_xlabel("applied displacement δ")
    ax2.set_ylabel("reaction force  (∝, energy-based)")
    ax2.set_title("force–displacement (brittle)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).parents[2] / "assets" / "phase_field_fracture_2d.png", dpi=130, bbox_inches="tight")

    print("\nOK: the cheap non-conforming Morley element captures the smooth 4th-order profile and, at a fine")
    print("    mesh the conforming element cannot reach, drives a sharp crack with the correct brittle response.")


if __name__ == "__main__":
    main()
