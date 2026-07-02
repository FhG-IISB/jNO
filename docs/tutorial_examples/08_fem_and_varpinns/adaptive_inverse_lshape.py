r"""Adaptive mesh refinement wrapped around an *inverse* solve -- the minimal mesh for the
recovered design.

The committed adaptive loop (:mod:`adaptive_l_shape`) refines the mesh for a *forward*
solution.  Here we go one step further: at every round we **recover an unknown parameter**
on the current mesh (a differentiable inverse solve through ``jno.core``), and only then
estimate/mark/refine.  The mesh therefore ends up minimal for the *recovered design*, and
-- crucially -- the recovered parameter de-biases toward the truth as the corner is
resolved.

Model (reaction--diffusion on the L-shape)::

    -lap u + kappa*u = f,   f = kappa_true * u_singular,   u = u_singular on the boundary

``u_singular = r^(2/3) sin(2 phi/3)`` about the reentrant corner (0.5, 0.5) is harmonic, so
at ``kappa = kappa_true`` the exact solution is exactly ``u_singular`` -- a *value*-singular
state.  ``kappa`` multiplies the mass (reaction) term, so it enters through the operator and
the inverse is differentiable via implicit diff (a parameter in the Dirichlet data would
not be).  Observations are the closed-form ``u_singular`` at mesh nodes, weighted to the
corner (``r < 0.2``) so the under-resolved singularity actually biases the fit -- that bias
is what adaptation removes.

The loop is::

    recover kappa (crux.solve)  ->  estimate error (ZZ)  ->  mark (Dörfler)  ->  domain.refine

driven entirely by the public :func:`jno.utils.solver.fem_adapt.run_adaptive_inverse`.

Run::

    JAX_PLATFORMS=cpu pixi run -e fem python docs/tutorial_examples/08_fem_and_varpinns/adaptive_inverse_lshape.py

It writes ``adaptive_inverse_lshape.png`` (adapted meshes + the kappa-vs-DOF convergence of
adaptive against uniform refinement) next to this file.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small FEM solves; keep off the GPU

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import optax

import jno
import jno.jnp_ops as J
from jno.utils.solver.fem_adapt import AdaptSpec, run_adaptive_inverse

KAPPA_TRUE = 5.0  # the parameter we pretend not to know
KAPPA_INIT = 2.0  # deliberately far-off starting guess


# --- the harmonic singular corner mode (the exact state at kappa = KAPPA_TRUE) -------
def _mod(a, m):
    # jno.np has no `mod`; build it from `floor` (works on trace symbols and numpy)
    return a - m * J.floor(a / m)


def u_singular(x, y, xp, mod):
    X, Y = x - 0.5, y - 0.5
    r = xp.sqrt(X * X + Y * Y)
    th = mod(xp.arctan2(Y, X), 2.0 * np.pi)
    phi = mod(th - np.pi / 2.0, 2.0 * np.pi)  # material wedge is 3*pi/2 wide
    return (r ** (2.0 / 3.0)) * xp.sin(2.0 / 3.0 * phi)


def reaction_diffusion_fem(d, kappa):
    """``-lap u + kappa*u = kappa_true*u_singular`` with Dirichlet ``g = u_singular``."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = KAPPA_TRUE * u_singular(xi, yi, J, _mod)
    return jno.fem(
        [ui.x * vi.x + ui.y * vi.y + kappa * (ui * vi) - f * vi, u(xb, yb) - u_singular(xb, yb, J, _mod)],
        quad_degree=4,
    )


def corner_observations(d):
    """Closed-form ``u_singular`` at nodes, weighted to the corner patch (r < 0.2)."""
    nodes = np.asarray(d.mesh.points)[:, :2]
    s = jnp.asarray(u_singular(nodes[:, 0], nodes[:, 1], np, np.mod))
    r = np.linalg.norm(nodes - (0.5, 0.5), axis=1)
    return s, jnp.asarray((r < 0.2).astype(np.float64))


def fresh_kappa(seed):
    k = jno.np.parameter((1,), key=jax.random.PRNGKey(seed), name=f"kappa{seed}")
    k.initialize(jax.nn.initializers.constant(KAPPA_INIT))
    k.dtype(jnp.float64)
    k.optimizer(optax.adam(1e-1))
    return k


# --- the adaptive-inverse run --------------------------------------------------------
def run_adaptive(mesh_size=0.2, n_rounds=10, n_opt=220):
    """Alternate recover-kappa / estimate / refine; snapshot mesh + state per round."""
    d = jno.domain(jno.domain.l_shape(size=1.0, mesh_size=mesh_size))
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    kappa = fresh_kappa(0)
    best: dict = {}
    snaps: list[dict] = []

    def build_inverse(dd):
        if "k" in best:  # warm-start the next (finer) round from the current estimate
            kappa.initialize(jax.nn.initializers.constant(best["k"]))
        s, w = corner_observations(dd)
        fem = reaction_diffusion_fem(dd, kappa)
        return jno.core([(w * (fem.solve() - s)).mse], domain=dummy), fem.solve()

    def readout(crux):
        v = float(np.asarray(crux.eval([kappa])).reshape(-1)[0])
        best["k"] = v
        # snapshot the recovered state + mesh for rendering (real computed values only)
        u = np.asarray(crux.eval([reaction_diffusion_fem(d, kappa).solve()])).reshape(-1)
        snaps.append(
            {
                "pts": np.asarray(d.mesh.points)[:, :2].copy(),
                "tris": np.asarray(d.mesh.cells_dict["triangle"]).copy(),
                "u": u,
                "n_dofs": len(u),
                "kappa": v,
            }
        )
        return v

    # eps stops the loop once the recovered kappa stops moving between rounds (with a
    # patience of 2, so a single flat step does not stop it early); max_iters / max_dofs are
    # the budget caps. Whichever fires first ends the "refine until good enough" loop. With a
    # low eps and a generous budget the recovered kappa marches all the way to the truth (5).
    hist = run_adaptive_inverse(
        d,
        build_inverse,
        AdaptSpec(theta=0.7, max_iters=n_rounds, refine_factor=1.6, eps=0.005, max_dofs=1500),
        n_opt=n_opt,
        readout=readout,
    )
    return snaps, hist


def run_uniform(mesh_sizes, n_opt=250):
    """Recover kappa once on each of a sequence of uniform meshes (the baseline)."""
    dummy = jno.domain.from_array({"_": np.zeros((1, 1))})
    out = []
    for i, ms in enumerate(mesh_sizes):
        d = jno.domain(jno.domain.l_shape(size=1.0, mesh_size=ms))
        s, w = corner_observations(d)
        k = fresh_kappa(100 + i)
        fem = reaction_diffusion_fem(d, k)
        crux = jno.core([(w * (fem.solve() - s)).mse], domain=dummy)
        crux.solve(n_opt)
        out.append({"n_dofs": len(d.mesh.points), "kappa": float(np.asarray(crux.eval([k])).reshape(-1)[0])})
    return out


# --- rendering -----------------------------------------------------------------------
def render(snaps, adapt_hist, uniform, outdir):
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Frutiger 45 Light", "Frutiger", "FreeSans", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.titlesize": 9,
        }
    )

    # show at most 4 mesh snapshots (first, two middle, last) so the strip stays legible
    idx = sorted({0, len(snaps) // 3, 2 * len(snaps) // 3, len(snaps) - 1})
    show = [snaps[i] for i in idx]
    umax = max(float(np.abs(s["u"]).max()) for s in snaps)

    fig = plt.figure(figsize=(2.0 * len(show), 4.6))
    gs = fig.add_gridspec(2, len(show), height_ratios=[1.0, 1.15], hspace=0.28, wspace=0.08)

    for j, s in enumerate(show):
        ax = fig.add_subplot(gs[0, j])
        tri = Triangulation(s["pts"][:, 0], s["pts"][:, 1], s["tris"])
        ax.tripcolor(tri, np.abs(s["u"]), shading="gouraud", cmap="cividis", vmin=0.0, vmax=umax)
        ax.triplot(tri, color="1.0", lw=0.2, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(rf"{s['n_dofs']} dofs · $\kappa$={s['kappa']:.2f}")

    # convergence: |recovered kappa - truth| vs DOFs, adaptive vs uniform
    ax = fig.add_subplot(gs[1, :])
    a_dofs = [s["n_dofs"] for s in snaps]
    a_err = [abs(s["kappa"] - KAPPA_TRUE) for s in snaps]
    u_dofs = [u["n_dofs"] for u in uniform]
    u_err = [abs(u["kappa"] - KAPPA_TRUE) for u in uniform]
    ax.plot(a_dofs, a_err, "o-", color="#b8322c", lw=1.6, ms=5, label="adaptive (recover→refine)")
    ax.plot(u_dofs, u_err, "s--", color="0.45", lw=1.4, ms=5, label="uniform")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mesh DOFs")
    ax.set_ylabel(r"$|\kappa_{\mathrm{recovered}} - \kappa_{\mathrm{true}}|$")
    ax.set_title(r"Recovered-parameter error per DOF — adaptive reaches uniform's accuracy at far fewer DOFs")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Adaptive mesh refinement on top of the inverse design (L-shape)", fontweight="bold")
    out = os.path.join(outdir, "adaptive_inverse_lshape.png")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    snaps, hist = run_adaptive()
    dof_adapt = snaps[-1]["n_dofs"]
    print("adaptive rounds (dofs, kappa):", [(s["n_dofs"], round(s["kappa"], 4)) for s in snaps])

    # a uniform sweep spanning the adaptive DOF range for the comparison
    uniform = run_uniform([0.12, 0.09, 0.07, 0.05, 0.04, 0.03])
    print("uniform (dofs, kappa):", [(u["n_dofs"], round(u["kappa"], 4)) for u in uniform])

    k_adapt = snaps[-1]["kappa"]
    matched = [u for u in uniform if u["n_dofs"] >= dof_adapt]
    if matched:
        best_u = min(matched, key=lambda u: abs(u["kappa"] - KAPPA_TRUE))
        print(
            f"adaptive |k-{KAPPA_TRUE}|={abs(k_adapt - KAPPA_TRUE):.3f} @ {dof_adapt} dofs  vs  "
            f"uniform {abs(best_u['kappa'] - KAPPA_TRUE):.3f} @ {best_u['n_dofs']} dofs"
        )

    out = render(snaps, hist, uniform, here)
    print("wrote", out)
