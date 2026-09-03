"""pypeec on the SAME power-module layout jNO and Ansys Q3D were run on.

The point is to split the remaining Q3D gap in two. If pypeec reproduces Q3D, the fault is in jNO's
model of this layout and can be bisected by stripping structure. If pypeec is also high, then both
codes are reading the geometry or the port the same wrong way, which is a different question.

HONEST LIMITS of the voxel model, stated up front because they bound what the answer can mean:

* pypeec voxelises, so the stack is snapped to a uniform dz. The real layers are 0.49 / 0.37 / 0.51
  mm (plane / gap / trace) and become 3 / 2 / 3 cells of dz, so each is distorted by up to ~8 %.
* A bond wire is 0.25 x 0.08 mm, far below any affordable voxel. Represented as a one-voxel path,
  its cross-section is ~10x too large, which shortens its own inductance by roughly a quarter. Bond
  wires were measured earlier to be worth <2 % of the loop, so call this a couple of percent.
* At 1 MHz the skin depth is 66 um and NO affordable voxel resolves it -- neither code does here.
  That mostly costs internal inductance and resistance, not the external inductance the port sees.

So this comparison is good to a few percent, which is ample against a ~20 % discrepancy, but it is
not a convergence study.

SCOPE LIMIT -- the z-stack is hard-coded to LAYOUT1's proportions. `build` lays the plane, gap,
trace and wire loop down as 3 / 2 / 3 / 2 cells of a single dz taken from the trace thickness, which
is layout1's stack and nobody else's. Layouts 2 and 3 run and return a number, and that number is
for the wrong stack. Only `c2352b86` (layout1, plate-less) and `19fa1112` (layout1) are valid here.

This file has ALREADY ROTTED ONCE. `rect` had silently become an overlap test while its own comment
documented a centre test, and it reproduced the exact failure value that comment records -- 12.674
against a 20.641 reference. That is why it is in the repo under review rather than in a scratch
directory, and why `tests/test_peec_pypeec_crosscheck.py` pins the geometry rule it turns on.
"""

import json
import logging
import os
import sys
import time

import numpy as np



def _paths():
    """The two inputs, NAMED rather than hard-coded -- neither lives in this repo.

    The layouts are the collaborator's and are not ours to redistribute; the tolerance config ships
    with pypeec's own examples. A missing one is an immediate, named failure, never a wrong number.
    """
    u, ppex = os.environ.get("JNO_LAYOUT_DIR"), os.environ.get("PYPEEC_EXAMPLES")
    if not u or not ppex:
        raise SystemExit(
            "benchmarks/pypeec_layout.py needs two paths:\n"
            "  JNO_LAYOUT_DIR   the directory holding <tag>-layout.json\n"
            "  PYPEEC_EXAMPLES  a checkout of pypeec's examples (for config/tolerance.yaml)\n"
            "Neither is in this repo -- the layouts are not ours to redistribute."
        )
    return u, ppex


def cells_in(r, nx, ny, pitch, ks, lin):
    """Cells whose CENTRE falls inside rectangle ``r``, on every z index in ``ks``.

    THE rule this harness turns on, lifted out of `build` so it can be tested without a pypeec
    install. Rounding OUTWARD instead -- `(i+1)p > x0 and i*p < x1`, an overlap test -- grows every
    trace by up to a voxel per side, and this module has 40+ traces with sub-millimetre gaps: they
    merge, the port finds a shortcut that is not in the layout, and the loop inductance collapses.
    Measured 12.69 nH that way against 20.64.

    This file carried the overlap test while documenting the centre test, and reproduced the
    documented failure value exactly -- 12.674 nH. That is what `test_peec_pypeec_crosscheck.py`
    pins, and why the rule is a named function rather than a closure nobody can reach.
    """
    cxs, cys = (np.arange(nx) + 0.5) * pitch, (np.arange(ny) + 0.5) * pitch
    ix = np.flatnonzero((cxs > r["x"][0]) & (cxs < r["x"][1]))
    iy = np.flatnonzero((cys > r["y"][0]) & (cys < r["y"][1]))
    return [lin(a, b, k) for k in ks for a in ix.tolist() for b in iy.tolist()]


CASES = {
    "19fa1112": 20.641,
    "4bcf216a": 16.768,
    "82819b08": 23.189,
    # layout1 with the Baseplate and Bottom Metal DELETED -- the unamplified case.
    # 20.6 = 54.5 - 33.9, so the with-plane number magnifies every error by 2.64x;
    # this one does not, which is why it is the comparison that means something.
    "c2352b86": 54.505,
}
RHO_CU, RHO_AL = 1.0 / 5.8e7, 1.0 / 3.8e7
FREQ = float(os.environ.get("PPFREQ", "1e6"))
LOOP = 0.35  # bond-wire loop height above the trace top, mm -- the reference's own model


def build(tag, pitch, wires=True, with_plane=True):
    u, _ppex = _paths()
    raw = json.load(open(f"{u}/{tag}-layout.json"))
    comp = [L for L in raw["power_module"]["layers"] if "traces" in L][0]
    # The plate-less file has no Bottom Metal at all, so the footprint comes from whatever the
    # bottom-most layer is. B0/B1 are only meaningful when there IS a plane.
    BM = ([L for L in raw["power_module"]["layers"] if L["name"] == "Bottom Metal"] or raw["power_module"]["layers"])[0]
    T0, T1 = comp["z"]
    B0, B1 = BM["z"]
    X1, Y1 = BM["x"][1], BM["y"][1]

    # z: plane (3 cells) | gap (2) | trace (3) | wire loop (2), one uniform dz. ZREF refines all of
    # it together -- at 1 MHz the default dz is 0.17 mm against a 66 um skin depth (2.6 delta), so
    # whether pypeec's own answer is converged there is a question, not an assumption.
    k = int(os.environ.get("ZREF", "1"))
    dz = (T1 - T0) / (3.0 * k)
    if with_plane:
        nz = 10 * k
        PLANE_K, TRACE_K, WIRE_K = range(0, 3 * k), range(5 * k, 8 * k), range(8 * k, 10 * k)
    else:  # no plane, so its layers and the gap are empty grid -- drop them and halve the problem
        nz = 5 * k
        PLANE_K, TRACE_K, WIRE_K = range(0, 0), range(0, 3 * k), range(3 * k, 5 * k)

    nx, ny = int(np.ceil(X1 / pitch)), int(np.ceil(Y1 / pitch))
    lin = lambda ix, iy, iz: int(ix + nx * iy + nx * ny * iz)
    cx = lambda v: int(np.clip(np.floor(v / pitch), 0, nx - 1))
    cy = lambda v: int(np.clip(np.floor(v / pitch), 0, ny - 1))

    rect = lambda r, ks: cells_in(r, nx, ny, pitch, ks, lin)

    trace, src, sink = [], [], []
    for t in comp["traces"]:
        cells = rect(t, TRACE_K)
        port = None
        for c in t.get("connectors", []):
            nm = c["name"].replace("+", "P").replace("-", "N")
            if nm in ("DCP", "DCN"):
                port = (nm, set(rect(c, TRACE_K)))
        if port is None:
            trace += cells
        else:
            nm, pc = port
            (src if nm == "DCP" else sink).extend(sorted(pc))
            trace += [c for c in cells if c not in pc]

    wire = []
    if wires:
        for w in comp["bondwires"]:
            (ax, ay), (bx, by) = w["endA"]["point"], w["endB"]["point"]
            for k in WIRE_K:  # the two posts
                wire += [lin(cx(ax), cy(ay), k), lin(cx(bx), cy(by), k)]
            n = max(2, int(np.hypot(bx - ax, by - ay) / pitch) * 3)
            for s in np.linspace(0.0, 1.0, n):  # and the bridge, rasterised
                wire.append(lin(cx(ax + s * (bx - ax)), cy(ay + s * (by - ay)), WIRE_K[-1]))

    plane = [lin(ix, iy, k) for k in PLANE_K for ix in range(nx) for iy in range(ny)] if with_plane else []
    seen, doms = set(), {}
    for name, cells in (("src", src), ("sink", sink), ("wire", wire), ("trace", trace), ("plane", plane)):
        keep = sorted({c for c in cells if c not in seen})
        seen |= set(keep)
        if keep:
            doms[name] = keep

    # pypeec REQUIRES every galvanically connected component to carry a source; jNO instead lets a
    # conductor float and pins it. A module is full of islands -- the ground plane, every gate trace
    # -- so each one that the port does not reach gets a V = 0 pin. It cannot pass net current (there
    # is nowhere for it to go), so it fixes a potential and nothing else, which is what jNO's
    # auto-pin does. Reported, because the count is a fact about the model worth seeing.
    from scipy import ndimage

    occ = np.zeros(nx * ny * nz, dtype=bool)
    occ[sorted(seen)] = True
    # lin = ix + nx*iy + nx*ny*iz, so the C-order (nz, ny, nx) view IS the right one. The transposed
    # F-order version here before was garbage: it reported 58 components where there are 2, and every
    # spurious one got a zero-current pin dropped into the MIDDLE of a conductor, puncturing it.
    lab = ndimage.label(occ.reshape((nz, ny, nx)), structure=ndimage.generate_binary_structure(3, 1))
    lab, ncomp = lab[0].reshape(-1), lab[1]
    live = {int(lab[c]) for c in doms["src"]} | {int(lab[c]) for c in doms["sink"]}
    pins = {}
    for cid in range(1, ncomp + 1):
        if cid in live:
            continue
        cells = np.flatnonzero(lab == cid)
        pins[f"pin{cid}"] = [int(cells[0])]
    for k, v in pins.items():
        for name in list(doms):
            doms[name] = [c for c in doms[name] if c not in set(v)]
        doms[k] = v
    doms = {k: v for k, v in doms.items() if v}
    print(f"   metal in {ncomp} connected components; port reaches {len(live)}, pinning {len(pins)}", flush=True)

    geo = {
        "mesh_type": "voxel",
        "data_voxelize": {
            "param": {"n": [nx, ny, nz], "d": [pitch * 1e-3, pitch * 1e-3, dz * 1e-3], "c": [0.0, 0.0, 0.0]},
            "domain_index": doms,
        },
        "data_point": {"check_cloud": False, "filter_cloud": False, "pts_cloud": []},
        "data_resampling": {"use_reduce": False, "use_resample": False, "resampling_factor": [1, 1, 1]},
        "data_conflict": {"resolve_rules": False, "resolve_random": False, "conflict_rules": []},
        "data_integrity": {"check_integrity": False, "domain_connected": {}, "domain_adjacent": {}},
    }
    cu = [d for d in doms if d != "wire"]
    pin_names = [d for d in doms if d.startswith("pin")]
    mat = {"copper": {"rho_re": RHO_CU, "rho_im": 0.0}}
    mdef = {
        "copper": {"domain_list": cu, "material_type": "electric", "orientation_type": "isotropic", "var_type": "lumped"}
    }
    if "wire" in doms:
        mat["alu"] = {"rho_re": RHO_AL, "rho_im": 0.0}
        mdef["alu"] = {
            "domain_list": ["wire"],
            "material_type": "electric",
            "orientation_type": "isotropic",
            "var_type": "lumped",
        }
    pro = {
        "material_def": mdef,
        "source_def": dict(
            {
                "src": {"domain_list": ["src"], "source_type": "current", "var_type": "lumped"},
                "sink": {"domain_list": ["sink"], "source_type": "voltage", "var_type": "lumped"},
            },
            # a ZERO-CURRENT source, not a voltage one: it satisfies pypeec's "every component needs
            # a source" without providing a path. A V = 0 pin looks harmless and is not -- it sits at
            # the sink's potential, so the return current escapes through the plane and L collapses
            # (measured 11.19 nH, -46 %, against 20.64).
            **{p: {"domain_list": [p], "source_type": "current", "var_type": "lumped"} for p in pin_names},
        ),
        "sweep_solver": {
            "sim": {
                "init": None,
                "param": {
                    "freq": FREQ,
                    "material_val": mat,
                    "source_val": dict(
                        {
                            "src": {"I_re": 1.0, "I_im": 0.0, "Y_re": 0.0, "Y_im": 0.0},
                            "sink": {"V_re": 0.0, "V_im": 0.0, "Z_re": 0.0, "Z_im": 0.0},
                        },
                        **{p: {"I_re": 0.0, "I_im": 0.0, "Y_re": 0.0, "Y_im": 0.0} for p in pin_names},
                    ),
                },
            }
        },
    }
    return geo, pro, sum(len(v) for v in doms.values()), nx * ny * nz


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "19fa1112"
    pitch = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    wires = (sys.argv[3] != "0") if len(sys.argv) > 3 else True
    with_plane = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
    # Quieting the log is a SCRIPT concern and lives here rather than at module scope. At module
    # scope it silenced the whole process on import, and this file is imported by a test -- which
    # took out every other test in the suite that asserts on a warning. Three of them, silently.
    logging.disable(logging.CRITICAL)

    # imported HERE, not at module scope: the geometry rule above is what regressed and what a test
    # has to reach, and it must not need a pypeec install to do it
    import pypeec
    import scisave

    _u, ppex = _paths()
    tol = scisave.load_config(f"{ppex}/config/tolerance.yaml")
    tol["condition_options"]["check"] = False
    geo, pro, nmet, ntot = build(tag, pitch, wires, with_plane)
    print(
        f"{tag}  pitch {pitch} mm  wires={'on' if wires else 'OFF'}  plane={'on' if with_plane else 'OFF'}  {nmet} metal of {ntot} voxels",
        flush=True,
    )
    t0 = time.perf_counter()
    vox = pypeec.run_mesher_data(geo)
    t1 = time.perf_counter()
    sol = pypeec.run_solver_data(vox, pro, tol)
    t2 = time.perf_counter()
    s = sol["data_sweep"]["sim"]
    sv = s["source_values"]
    Z = (complex(sv["src"]["V"]) - complex(sv["sink"]["V"])) / complex(sv["src"]["I"])
    L = Z.imag / (2 * np.pi * FREQ)
    ref = CASES[tag]
    print(
        f"  mesh {t1 - t0:.1f}s  solve {t2 - t1:.1f}s  dof {int(s['solver_status']['n_dof_total'])}  "
        f"R = {Z.real * 1e6:.1f} uOhm   L = {L * 1e9:.3f} nH   vs Ansys {ref:.3f}: {100 * (L * 1e9 / ref - 1):+.2f} %",
        flush=True,
    )
