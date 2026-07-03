"""08 - SUPG-stabilized high-Peclet advection-diffusion (uses ``domain.cell_size``).

    beta . grad u - nu Lap u = 0     on the unit square,  beta = (1, 0)
    u = 0 at x=0 (inflow),  u = 1 at x=1 (outflow),  natural (du/dn=0) on y=0, y=1.

At high Peclet (``Pe_cell = |beta| h / (2 nu) >> 1``) the layer at the outflow is far thinner than
the mesh, so the standard Galerkin solution develops spurious node-to-node oscillations (it
over/undershoots the physical [0,1] range). Streamline-upwind Petrov-Galerkin (SUPG) adds a
mesh-dependent stabilization ``tau (beta.grad u)(beta.grad v)`` with ``tau = h / (2 |beta|)`` --
which needs the element size ``h``. jNO exposes it as the symbol ``domain.cell_size``, so the whole
stabilized form is written declaratively in the same ``jno.fem([...])`` list; nothing else changes.

This tutorial solves the SAME problem with and without the stabilization and shows that SUPG removes
the oscillations (bounded range, lower total variation) while Galerkin overshoots.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jno

N, NU = 20, 1e-2  # Pe_cell = (1)*(1/N)/(2 NU) = 2.5 >> 1  -> Galerkin oscillates
dom = jno.domain(constructor=jno.domain.equi_distant_rect(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=N, ny=N))
dom.tag("inflow", lambda x, y: x < 1e-6)
dom.tag("outflow", lambda x, y: x > 1 - 1e-6)

u, phi = dom.fem_symbols()
xi, yi, _ = dom.variable("interior", split=True)
xin, yin, _ = dom.variable("inflow", split=True)
xout, yout, _ = dom.variable("outflow", split=True)
ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

bx, by = 1.0, 0.0  # advection velocity beta
adv_u = bx * ui.x + by * ui.y  # beta . grad u
adv_v = bx * vi.x + by * vi.y  # beta . grad v
galerkin = adv_u * vi + NU * (ui.x * vi.x + ui.y * vi.y)  # no source; pure transport

# --- the stabilization, written declaratively via the new `cell_size` symbol ---
h = dom.cell_size
tau = h / (2.0 * np.hypot(bx, by))  # high-Pe SUPG parameter h/(2|beta|)
supg = tau * adv_u * adv_v

bcs = [u(xin, yin) - 0.0, u(xout, yout) - 1.0]


def solve(weak):
    fem = jno.fem([weak, *bcs])
    # non-symmetric advection operator -> GMRES
    return jnp.asarray(fem.solve(linear=jno.solve.gmres(), precond=jno.precond.jacobi())), fem


u_galerkin, fem_g = solve(galerkin)
u_supg, _ = solve(galerkin + supg)

# --- reduce to an x-profile (average over y) on the structured grid ---
pts = np.asarray(fem_g.points)
ix = np.round(pts[:, 0] * N).astype(int)
iy = np.round(pts[:, 1] * N).astype(int)


def x_profile(uvals):
    g = np.full((N + 1, N + 1), np.nan)
    g[iy, ix] = np.asarray(uvals)
    return np.nanmean(g, axis=0)


xs = np.linspace(0.0, 1.0, N + 1)
pg, ps = x_profile(u_galerkin), x_profile(u_supg)
tv = lambda p: float(np.sum(np.abs(np.diff(p))))  # total variation along x  # noqa: E731

print(f"\nSUPG advection-diffusion (N={N}, nu={NU}, Pe_cell={1.0 / N / (2 * NU):.1f})")
print(f"  Galerkin : range [{pg.min():+.3f}, {pg.max():+.3f}]  total-variation={tv(pg):.3f}")
print(f"  SUPG     : range [{ps.min():+.3f}, {ps.max():+.3f}]  total-variation={tv(ps):.3f}")

# --- figure ---
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhspan(0.0, 1.0, color="0.93", zorder=0, label="physical range [0,1]")
    ax.plot(xs, pg, "o-", lw=1.5, ms=3, label=f"Galerkin (TV={tv(pg):.2f})")
    ax.plot(xs, ps, "s-", lw=1.5, ms=3, label=f"SUPG (TV={tv(ps):.2f})")
    ax.set(
        xlabel="x", ylabel="u (averaged over y)", title=f"High-Pe advection-diffusion · Pe_cell={1.0 / N / (2 * NU):.1f}"
    )
    ax.legend(loc="upper left", frameon=False)
    out = Path(__file__).parents[2] / "assets" / "supg_advection_diffusion_2d.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"  figure -> {out}")
except Exception as e:  # pragma: no cover - plotting is optional in CI
    print(f"  (plot skipped: {e})")

# --- verification: SUPG removes the oscillation that Galerkin produces ---
assert ps.min() > -0.02 and ps.max() < 1.02, "SUPG solution should stay within the physical [0,1] range"
assert pg.min() < -0.05 or pg.max() > 1.05, "Galerkin should overshoot at this Peclet number"
assert tv(ps) < 0.6 * tv(pg), "SUPG must have substantially lower total variation (less oscillation)"
print("  OK: SUPG bounded & monotone; Galerkin oscillates.")
