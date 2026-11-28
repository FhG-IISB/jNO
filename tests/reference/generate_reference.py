#!/usr/bin/env python
"""Reference T and R from independent RCWA libraries, for the cross-checks in ``tests/test_rcwa_reference.py``.

Runs in the separate ``rcwa-ref`` pixi environment (fmmax, torcwa, Meent; see ``README.md`` here), never in
jNO's own: it imports no ``jno`` and writes plain numbers to ``rcwa_reference.json``. Every library solves the same
truncated problem as jNO's ``formulation="FFT"`` (Laurent's rule) with the parallelogramic truncation
``|m|, |n| <= M``:

* fmmax 1.7.1: ``Formulation.FFT``, ``Truncation.PARALLELOGRAMIC``, pixel-centred Fourier coefficients;
* torcwa: Laurent's rule (``_material_conv``), ``order=[M, M]``; it samples the grid at pixel corners, which only
  shifts the structure by half a pixel and leaves every diffraction efficiency unchanged.

Usage (in the rcwa-ref env): ``python tests/reference/generate_reference.py [OUT.json]``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cases import CASES, grid_of  # noqa: E402


def fmmax_tr(case, m):
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from fmmax import basis, fields, fmm, scattering, sources

    lv = basis.LatticeVectors(jnp.array([case["period"][0], 0.0]), jnp.array([0.0, case["period"][1]]))
    ex = basis.generate_expansion(lv, (2 * m + 1) ** 2, basis.Truncation.PARALLELOGRAMIC)
    wl = case["wavelength"]
    k0 = 2 * math.pi / wl
    kin = jnp.asarray([case["k_in"][0] * k0, case["k_in"][1] * k0])

    def solve(grid):
        return fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wl),
            in_plane_wavevector=kin,
            primitive_lattice_vectors=lv,
            permittivity=jnp.asarray(grid, jnp.complex128),
            expansion=ex,
            formulation=fmm.Formulation.FFT,
        )

    layers = [solve(np.asarray([[case["eps_top"]]]))]
    thick = [1.0]
    for d, spec in case["layers"]:
        layers.append(solve(grid_of(spec)))
        thick.append(d)
    layers.append(solve(np.asarray([[case["eps_bottom"]]])))
    thick.append(1.0)
    s = scattering.stack_s_matrix(layers, thick)
    g = 2 * m + 1
    px, py = case["polarization"]
    fwd = sources.amplitudes_for_fields(
        jnp.full((g, g, 1), px, jnp.complex128),
        jnp.full((g, g, 1), py, jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        layers[0],
    )[0]
    # keep only the specular order: the uniform field above is the zeroth order alone
    pin = float(jnp.sum(jnp.real(fields.directional_poynting_flux(fwd, jnp.zeros_like(fwd), layers[0])[0])))
    t = float(jnp.sum(jnp.real(fields.directional_poynting_flux(s.s11 @ fwd, jnp.zeros_like(fwd), layers[-1])[0])))
    r = -float(jnp.sum(jnp.real(fields.directional_poynting_flux(jnp.zeros_like(fwd), s.s21 @ fwd, layers[0])[1])))
    return t / pin, r / pin


def torcwa_tr(case, m):
    import torch
    import torcwa

    torch.set_default_dtype(torch.float64)
    wl = case["wavelength"]
    n_top = math.sqrt(case["eps_top"])
    kx, ky = case["k_in"]
    sin_t = math.hypot(kx, ky) / n_top
    inc = math.asin(sin_t)
    azi = math.atan2(ky, kx) if sin_t > 0 else 0.0
    sim = torcwa.rcwa(freq=1.0 / wl, order=[m, m], L=list(case["period"]), dtype=torch.complex128, device="cpu")
    sim.add_input_layer(eps=case["eps_top"])
    sim.add_output_layer(eps=case["eps_bottom"])
    sim.set_incident_angle(inc_ang=inc, azi_ang=azi)
    for d, spec in case["layers"]:
        sim.add_layer(thickness=d, eps=torch.as_tensor(grid_of(spec), dtype=torch.complex128))
    sim.solve_global_smatrix()
    orders = [[i, j] for i in range(-m, m + 1) for j in range(-m, m + 1)]
    pin = case["pol"]  # a pure s or p input, so no s/p sign convention can enter
    out = []
    for port in ("transmission", "reflection"):
        total = 0.0
        for pout in ("p", "s"):
            s = sim.S_parameters(
                orders=orders, direction="forward", port=port, polarization=pout + pin, ref_order=[0, 0], power_norm=True
            )
            total += float(torch.sum(torch.abs(s) ** 2))
        out.append(total)
    return out[0], out[1]


LIBRARIES = {"fmmax": fmmax_tr, "torcwa": torcwa_tr}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "rcwa_reference.json")
    results, versions = {}, {}
    import importlib.metadata as md

    for lib in LIBRARIES:
        versions[lib] = md.version(lib)
    jobs = [(name, m, lib) for name, case in CASES.items() for m in case["truncations"] for lib in LIBRARIES]
    t0 = time.perf_counter()
    for i, (name, m, lib) in enumerate(jobs):
        t, r = LIBRARIES[lib](CASES[name], m)
        results.setdefault(name, {}).setdefault(str(m), {})[lib] = {"T": t, "R": r}
        el = time.perf_counter() - t0
        print(
            f"case {i + 1}/{len(jobs)} done, {el:.0f} s elapsed, ~{el / (i + 1) * (len(jobs) - i - 1):.0f} s left: "
            f"{name} M={m} {lib} T={t:.12f} R={r:.12f}",
            flush=True,
        )
    with open(out_path, "w") as fh:
        json.dump({"versions": versions, "results": results}, fh, indent=1, sort_keys=True)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
