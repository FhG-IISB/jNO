"""Reanalysis: does the optimised design report its own stiffness honestly?

The optimiser moves nodes AND densities, so it can lower compliance either by improving the
structure or by distorting elements until they under-integrate strain energy and report a
stiffness they do not have. Those are indistinguishable from inside the run. The test that
separates them (Jung, Yun & Kim 2026, Fig. 7c-d) is to take the converged design, freeze it as a
binary layout, and re-solve it on a FRESH, UNDISTORTED mesh:

    C_design      the design's own number, on its own deformed mesh
    C_reanalysis  the same layout on a clean fine mesh

If they agree the reported stiffness is real. If C_reanalysis is much higher, the run was farming
discretisation error. The paper measures +17.6 % for the conventional element and -0.5 % with
E-FEM; that gap is what the enrichment exists to close.
"""

import argparse
import logging

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import TrapezoidMapTriFinder, Triangulation  # noqa: E402

import jno  # noqa: E402

inner, symgrad, trace = jno.np.inner, jno.np.symgrad, jno.np.trace

p = argparse.ArgumentParser()
p.add_argument("--design", required=True, help="path to the .npz design field to re-analyse")
p.add_argument("--h-fine", type=float, default=0.5, help="reanalysis mesh size")
p.add_argument("--threshold", type=float, default=0.5)
args = p.parse_args()
logging.disable(logging.INFO)

L, H = 60.0, 30.0
E0, EMIN, NU, VOLFRAC = 1.0, 1e-9, 0.3, 0.4
LAM, MU = E0 * NU / (1 - NU**2), E0 / (2 * (1 + NU))
TOL = 1e-6 * L


def build(size, trainable=False):
    """The SAME weak form as the optimisation run: clamped left, unit traction on the right."""
    d = jno.Shape.rect(0, 0, L, H, size=size).domain()
    if trainable:
        xm, ym, _ = d.variable("mv", where=lambda x, y: (x > TOL) & (x < L - TOL) & (y > TOL) & (y < H - TOL), split=True)
        xm.trainable(name="mesh_x"), ym.trainable(name="mesh_y")
    u, phi = d.fem_symbols(value_shape=(2,))
    _r, s = d.fem_symbols(space="P0", names=("r", "s"))
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    rho = jno.np.parameter(s, name="rho")
    rho.dtype(jnp.float64)
    eu, ep = symgrad(u, [xi, yi]), symgrad(phi, [xi, yi])
    fem = jno.fem(
        [
            (EMIN + rho**3.0 * (E0 - EMIN)) * (LAM * trace(eu) * trace(ep) + 2 * MU * inner(eu, ep, n_contract=2)),
            u(xl, yl) - (0.0, 0.0),
            -1.0 * inner(jnp.array([0.0, -1.0 / H]), phi.bind(x=xr, y=yr), n_contract=1),
        ],
        quad_degree=2,
    )
    return d, fem


def compliance(fem, args_dict):
    """C = f . u, from the assembled system — no optimiser, no trace."""
    a, b = fem.operator.evaluate(args_dict)
    a_d = np.asarray(jnp.asarray(a.todense()), dtype=np.float64)
    f = np.asarray(jnp.asarray(b).reshape(-1), dtype=np.float64)
    u = np.linalg.solve(a_d, f)
    return float(f @ u), u


z = np.load(args.design)
pts_def, cells_def, rho_e = z["pts"], z["cells"], z["rho_e"]
solid = rho_e > args.threshold
binary = np.where(solid, 1.0, 1e-3)
print(f"design: {cells_def.shape[0]} elements, {int(solid.sum())} solid ({solid.mean():.3f} by count), reported C = 77.974")

# --- 1. the design's own number, on its own deformed mesh, made BINARY --------------------------
d_def, fem_def = build(1.0, trainable=True)
specs = d_def._trainable_coords
ids = np.asarray(specs[0]["ids"], dtype=int)
coord_args = {sp["name"]: jnp.asarray(pts_def[ids, int(sp["axis"])]) for sp in specs}
assert np.asarray(d_def._cells_p1()).shape == cells_def.shape, "mesh mismatch — regenerate"

c_grey, _ = compliance(fem_def, {"rho": jnp.asarray(rho_e), **coord_args})
c_bin_def, _ = compliance(fem_def, {"rho": jnp.asarray(binary), **coord_args})
print(f"on the deformed mesh:   grey C = {c_grey:.4f}   binary C = {c_bin_def:.4f}")

# --- 2. the same layout on a fresh undistorted mesh ---------------------------------------------
d_new, fem_new = build(args.h_fine)
cells_new = np.asarray(d_new._cells_p1())
pts_new = np.asarray(d_new.mesh.points)[:, :2]
cen_new = pts_new[cells_new].mean(axis=1)

# Which deformed element does each new element's centroid fall in? Exact point-in-triangle via
# matplotlib's trapezoid map; -1 means outside the deformed mesh (impossible here, same domain).
finder = TrapezoidMapTriFinder(Triangulation(pts_def[:, 0], pts_def[:, 1], cells_def))
owner = finder(cen_new[:, 0], cen_new[:, 1])
outside = int((owner < 0).sum())
rho_new = np.where(owner >= 0, rho_e[np.maximum(owner, 0)], 1e-3)
print(
    f"reanalysis mesh: {cells_new.shape[0]} elements ({outside} centroids outside), "
    f"solid fraction {float((rho_new > 0.5).mean()):.3f}"
)

c_re, _ = compliance(fem_new, {"rho": jnp.asarray(rho_new)})
rho_new_bin = np.where(owner >= 0, binary[np.maximum(owner, 0)], 1e-3)
c_re_bin, _ = compliance(fem_new, {"rho": jnp.asarray(rho_new_bin)})

# --- verdict ------------------------------------------------------------------------------------
gap = 100.0 * (c_re - c_grey) / c_grey
gap_bin = 100.0 * (c_re_bin - c_bin_def) / c_bin_def
print("\n" + "=" * 68)
print("  DENSITY TRANSFERRED AS-IS -- isolates discretisation from thresholding")
print(f"  C reported by the design, on its deformed mesh   : {c_grey:10.4f}")
print(f"  C of the same density field on a clean fine mesh : {c_re:10.4f}")
print(f"  the design over-reports its stiffness by         : {gap:+9.2f} %")
print("  --")
print(f"  thresholded at {args.threshold}: deformed {c_bin_def:.3e} -> clean {c_re_bin:.3e} ({gap_bin:+.2f} %)")
print("  (paper: +17.6 % conventional, -0.5 % with E-FEM)")
print("=" * 68)
