r"""Anisotropic (metric-based) mesh refinement — stretched elements for a thin oblique layer.

Isotropic refinement (:mod:`adaptive_l_shape`) makes triangles *smaller* where the error is
large. For a thin **directional** feature that wastes elements: to resolve a layer of width
``eps`` isotropically you need ``~eps``-sized triangles all along it. **Anisotropic** adaptation
instead uses *stretched* triangles — thin across the layer, long along it — resolving the same
feature with far fewer degrees of freedom.

The metric is built from the solution's recovered **Hessian**: its eigenvectors orient the
triangles with the solution's curvature and its eigenvalues set the size along each direction
(:func:`jno.utils.solver.fem_adapt.hessian_metric`). ``AdaptSpec(anisotropic=True)`` switches the
adaptive loop from isotropic ZZ + Dörfler marking to this metric.

Model problem (an **oblique** internal layer, the case isotropic handles worst)::

    -lap u = f  on the unit square,   u = tanh((x + y - 1)/eps)   (thin layer along x+y=1)

Run::

    JAX_PLATFORMS=cpu pixi run -e fem python docs/tutorial_examples/08_fem_and_varpinns/anisotropic_layer.py

It writes ``anisotropic_layer.png`` (isotropic vs anisotropic meshes + the error-estimate
convergence per DOF) next to this file.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

from shapely.geometry import box

import jno
import jno.jnp_ops as J
from jno.utils.solver.fem_adapt import AdaptSpec, _solve_vertex_values, zz_error_indicators

EPS = 0.03


def u_exact(x, y, xp):
    return xp.tanh((x + y - 1.0) / EPS)


def build(d):
    """``-lap u = f`` with the oblique-layer exact solution as Dirichlet data."""
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    t = J.tanh((xi + yi - 1.0) / EPS)
    f = (4.0 / EPS**2) * (1.0 - t * t) * t  # -lap(tanh) for the diagonal layer
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - u_exact(xb, yb, J)])


def run(spec):
    """Adaptively solve, tracking (dofs, estimate) per round; return the final domain + history."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    fem = build(d)
    fem.solve(adapt=spec)
    return d, fem.adapt_history


def snapshot(d):
    return np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])


def render(iso, aniso, outdir):
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
            "axes.titlesize": 10,
        }
    )
    (di, hi), (da, ha) = iso, aniso
    fig = plt.figure(figsize=(9.2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.25)

    for ax, (d, h), title in [
        (fig.add_subplot(gs[0, 0]), (di, hi), f"isotropic · {len(di.mesh.points)} dofs"),
        (fig.add_subplot(gs[0, 1]), (da, ha), f"anisotropic · {len(da.mesh.points)} dofs"),
    ]:
        pts, tris = snapshot(d)
        ax.triplot(Triangulation(pts[:, 0], pts[:, 1], tris), color="0.2", lw=0.3)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot([r["n_dofs"] for r in hi], [r["estimate"] for r in hi], "s--", color="0.45", lw=1.4, ms=4, label="isotropic")
    ax.plot(
        [r["n_dofs"] for r in ha], [r["estimate"] for r in ha], "o-", color="#b8322c", lw=1.6, ms=4, label="anisotropic"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mesh DOFs")
    ax.set_ylabel("ZZ error estimate")
    ax.set_title("error estimate per DOF")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8)
    # clean, non-overlapping x ticks (log minor labels are cluttered on this narrow range)
    from matplotlib.ticker import FuncFormatter, NullFormatter

    ax.set_xticks([200, 500, 1000, 2000, 5000])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))

    fig.suptitle("Anisotropic vs isotropic refinement — thin oblique layer", fontweight="bold")
    out = os.path.join(outdir, "anisotropic_layer.png")
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    iso = run(AdaptSpec(theta=0.7, max_iters=9, refine_factor=1.7, max_dofs=3000))
    aniso = run(AdaptSpec(anisotropic=True, max_iters=8, refine_factor=1.6, max_dofs=2500))

    def report(name, d, h):
        fem = build(d)
        _, est = zz_error_indicators(d, _solve_vertex_values(fem))
        print(f"{name:>12}: {len(d.mesh.points):>5} dofs, ZZ estimate {est:.3e}")

    report("isotropic", *iso)
    report("anisotropic", *aniso)
    print("wrote", render(iso, aniso, here))
