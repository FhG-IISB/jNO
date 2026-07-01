r"""Adaptive mesh refinement on the L-shape, driven by the simple ``jno`` API.

Starting from a *coarse* uniform mesh, we repeatedly

    solve  ->  estimate error (Zienkiewicz-Zhu)  ->  mark (Dörfler)  ->  ``domain.refine``

so the mesh automatically concentrates elements at the reentrant corner, where the
solution of the Laplace problem has the classic ``r^(2/3)`` singularity.

The model problem is ``-lap u = 0`` on the L-shape with Dirichlet data equal to the
exact singular mode ``u = r^(2/3) sin(2*phi/3)`` about the corner (0.5, 0.5); ``u`` is
harmonic, so the only error comes from resolving the corner.

Run::

    JAX_PLATFORMS=cpu pixi run -e fem python docs/tutorial_examples/08_fem_and_varpinns/adaptive_l_shape.py

It writes ``adaptive_l_shape_panel.png`` (a filmstrip of mesh + solution per round) and
``adaptive_l_shape.gif`` (the same, animated) next to this file.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small FEM solves; keep off the GPU

import jax

jax.config.update("jax_enable_x64", True)

import jno
import jno.jnp_ops as J
from jno.utils.solver.fem_adapt import dorfler_mark, size_field_from_marks, zz_error_indicators


# --- the exact singular corner mode (harmonic) --------------------------------------
def _mod(a, m):
    # jno.np has no `mod`; build it from `floor` (works on trace symbols and numpy)
    return a - m * J.floor(a / m)


def u_singular(x, y, xp, mod):
    X, Y = x - 0.5, y - 0.5
    r = xp.sqrt(X * X + Y * Y)
    th = mod(xp.arctan2(Y, X), 2.0 * np.pi)
    phi = mod(th - np.pi / 2.0, 2.0 * np.pi)  # material wedge is 3*pi/2 wide
    return (r ** (2.0 / 3.0)) * xp.sin(2.0 / 3.0 * phi)


def build_constraints(d):
    """``-lap u = 0`` with the singular mode as Dirichlet data on the whole boundary."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    return [ui.x * vi.x + ui.y * vi.y, u(xb, yb) - u_singular(xb, yb, J, _mod)]


# --- the adaptive loop, using only the public building blocks ------------------------
def run_adaptive(mesh_size=0.3, n_rounds=6, theta=0.6, refine_factor=1.7):
    d = jno.domain(jno.domain.l_shape(size=1.0, mesh_size=mesh_size))
    snapshots = []
    for _ in range(n_rounds):
        fem = jno.fem(build_constraints(d))
        u = np.asarray(fem.solve()).reshape(-1)  # scalar P1: one value per vertex
        eta, est = zz_error_indicators(d, u)

        pts = np.asarray(d.mesh.points)[:, :2].copy()
        tris = np.asarray(d.mesh.cells_dict["triangle"]).copy()
        snapshots.append({"pts": pts, "tris": tris, "u": u.copy(), "n_dofs": len(u), "estimate": est})

        marked = dorfler_mark(eta, theta)
        if marked.size == 0:
            break
        d.refine(size_field_from_marks(d, marked, refine_factor=refine_factor))
    return snapshots


# --- rendering -----------------------------------------------------------------------
def render(snapshots, outdir):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.tri import Triangulation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # House style via rcParams (seaborn-free so the tutorial needs no extra dependency).
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Frutiger 45 Light", "Frutiger", "FreeSans", "DejaVu Sans"],
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
        }
    )

    # the singular mode u = r^(2/3) sin(2*phi/3) is non-negative, so use a 0->vmax
    # sequential scale (shared across rounds) rather than a washed-out symmetric one
    umax = max(float(s["u"].max()) for s in snapshots)
    vlim = 0.75 if umax <= 0.75 else 1.0

    def tri_of(s):
        return Triangulation(s["pts"][:, 0], s["pts"][:, 1], s["tris"])

    def draw_mesh(ax, s):
        ax.triplot(tri_of(s), color="0.2", lw=0.4)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"mesh · {s['n_dofs']} dofs")

    def draw_sol(ax, s, add_cbar=False):
        tpc = ax.tripcolor(tri_of(s), s["u"], shading="gouraud", cmap="cividis", vmin=0.0, vmax=vlim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(rf"$u$ · est {s['estimate']:.2e}")
        if add_cbar:
            cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
            cbar = plt.colorbar(tpc, cax=cax)
            cbar.outline.set_visible(False)
            cbar.ax.tick_params(length=0)
            for tl in cbar.ax.yaxis.get_ticklabels():
                tl.set_fontstyle("italic")
                tl.set_fontweight("light")
        return tpc

    # (1) filmstrip panel: row 0 = meshes, row 1 = solutions, one column per round
    n = len(snapshots)
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.4))
    axes = np.atleast_2d(axes)
    for j, s in enumerate(snapshots):
        draw_mesh(axes[0, j], s)
        draw_sol(axes[1, j], s, add_cbar=(j == n - 1))
    fig.suptitle("Adaptive refinement at the L-shape reentrant corner", fontweight="bold")
    fig.tight_layout()
    panel = os.path.join(outdir, "adaptive_l_shape_panel.png")
    fig.savefig(panel)
    plt.close(fig)

    # (2) side-by-side animation: mesh | solution, one frame per round
    figa, (axm, axs) = plt.subplots(1, 2, figsize=(8.2, 4.2))
    draw_sol(axs, snapshots[0], add_cbar=True)

    def frame(i):
        s = snapshots[i]
        axm.clear()
        axs.clear()
        draw_mesh(axm, s)
        draw_sol(axs, s)
        figa.suptitle(f"Adaptive refinement — round {i + 1}/{n}", fontweight="bold")

    anim = FuncAnimation(figa, frame, frames=n, interval=900)
    gif = os.path.join(outdir, "adaptive_l_shape.gif")
    anim.save(gif, writer=PillowWriter(fps=1.4))
    plt.close(figa)
    return panel, gif


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    snaps = run_adaptive()
    print("rounds:", [(s["n_dofs"], round(s["estimate"], 5)) for s in snaps])
    panel, gif = render(snaps, here)
    print("wrote", panel)
    print("wrote", gif)
