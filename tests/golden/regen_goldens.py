#!/usr/bin/env python
"""Regenerate the committed RCWA golden references.

This script is the ORACLE side of the parity contract. For every canonical case
in ``tests/golden/_cases.py`` it builds the equivalent problem with **fmmax
1.7.1** (the reference RCWA implementation) in complex128 and freezes the
resulting T/R (for the uniform slab also a small interior |E|^2 sample; for the
emitter case the radiated up/down powers instead of T/R) into
``tests/golden/<name>.npz``. It then writes ``tests/golden/manifest.json`` with
a sha256 of every ``.npz`` so a regression is caught at commit time.

Usage (needs the ``rcwa-oracle`` extra, which carries fmmax):

    python tests/golden/regen_goldens.py        # regen every case
    python tests/golden/regen_goldens.py slab   # regen one case by name

The companion ``tests/test_rcwa_golden.py`` runs WITHOUT fmmax and checks that
the self-owned eig-free backend reproduces these frozen references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # tests/ on path

from fmmax import basis as fmmax_basis  # noqa: E402
from fmmax import fields as fmmax_fields  # noqa: E402
from fmmax import fmm, scattering, sources  # noqa: E402
from golden import _cases  # noqa: E402


# ---------------------------------------------------------------------------
# fmmax reference builders (all run in complex128 / x64).
# ---------------------------------------------------------------------------
def _expansion(period, orders):
    lv = fmmax_basis.LatticeVectors(jnp.array([period[0], 0.0]), jnp.array([0.0, period[1]]))
    ex = fmmax_basis.generate_expansion(lv, orders)
    return lv, ex


def _formulation(name):
    if name is None or name == "JONES_DIRECT_FOURIER":
        return fmm.Formulation.JONES_DIRECT_FOURIER
    if name == "FFT":
        return fmm.Formulation.FFT
    raise ValueError(f"unknown formulation {name!r}")


def _iso_layer(wavelength, kin, lv, mat, ex, formulation):
    # mat: float (uniform) or (g,g) array (patterned) -> fmmax needs a (g,g) grid
    grid = mat if isinstance(mat, np.ndarray) else np.asarray([[mat]])
    return fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=jnp.asarray(kin),
        primitive_lattice_vectors=lv,
        permittivity=jnp.asarray(grid),
        expansion=ex,
        formulation=formulation,
    )


def _layer_solve(wavelength, kin, lv, mat, ex, formulation):
    if isinstance(mat, tuple) and len(mat) == 10:
        eps = tuple(mat[:5])
        mu = tuple(mat[5:])
        return fmm.eigensolve_general_anisotropic_media(
            jnp.asarray(wavelength),
            jnp.asarray(kin),
            lv,
            *[jnp.asarray([[v]]) for v in eps],
            *[jnp.asarray([[v]]) for v in mu],
            ex,
            formulation=formulation,
        )
    if isinstance(mat, tuple):  # 5-comp anisotropic
        eps = tuple(mat)
        mu = (1.0, 0.0, 0.0, 1.0, 1.0)
        return fmm.eigensolve_general_anisotropic_media(
            jnp.asarray(wavelength),
            jnp.asarray(kin),
            lv,
            *[jnp.asarray([[v]]) for v in eps],
            *[jnp.asarray([[v]]) for v in mu],
            ex,
            formulation=formulation,
        )
    return _iso_layer(wavelength, kin, lv, mat, ex, formulation)


def _ambient_layer(wavelength, kin, lv, eps, ex, formulation):
    return fmm.eigensolve_isotropic_media(
        wavelength=jnp.asarray(wavelength),
        in_plane_wavevector=jnp.asarray(kin),
        primitive_lattice_vectors=lv,
        permittivity=jnp.asarray([[eps]]),
        expansion=ex,
        formulation=formulation,
    )


def _plane_wave_tref(spec, formulation):
    """fmmax T/R for a plane-wave stack, and (for the slab) interior |E|^2."""
    wavelength = spec["wavelength"]
    period = spec["period"]
    kin = spec["k_in"]
    lv, ex = _expansion(period, spec["orders"])
    form = _formulation(formulation)
    layers = [
        _ambient_layer(wavelength, kin, lv, spec["amb_sup"], ex, form),
    ]
    thicknesses = []
    for thick, mat in spec["layers"]:
        layers.append(_layer_solve(wavelength, kin, lv, mat, ex, form))
        thicknesses.append(thick)
    layers.append(_ambient_layer(wavelength, kin, lv, spec["amb_sub"], ex, form))

    s = scattering.stack_s_matrix(layers, [1.0] + thicknesses + [1.0])
    g = 2 * int(np.amax(np.abs(ex.basis_coefficients))) + 1
    fwd = sources.amplitudes_for_fields(
        jnp.ones((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        jnp.zeros((g, g, 1), jnp.complex128),
        layers[0],
    )[0]
    pin = float(jnp.sum(jnp.real(fmmax_fields.directional_poynting_flux(fwd, jnp.zeros_like(fwd), layers[0])[0])))
    ft = fmmax_fields.directional_poynting_flux(s.s11 @ fwd, jnp.zeros_like(fwd), layers[-1])[0]
    fbw = fmmax_fields.directional_poynting_flux(jnp.zeros_like(fwd), s.s21 @ fwd, layers[0])[1]
    t = float(jnp.sum(jnp.real(ft))) / pin
    r = -float(jnp.sum(jnp.real(fbw))) / pin
    return {"t_ref": t, "r_ref": r, "_expansion": ex, "_layers": layers}


def _e2_slab(spec, expansion, layers, nx=8):
    """Analytic mid-film |E|^2 ratio for the uniform slab (normal incidence, TE).

    fmmax's field reconstruction uses mode-normalised amplitudes whose ABSOLUTE
    field magnitude is engine-dependent, so an absolute |E|^2 cannot be frozen
    portably. For the uniform slab this is the closed-form Fresnel/Airy field:
    the exact mid-film / incident-superstrate |E|^2 ratio, which both engines
    must reproduce (it is the same physics the Airy T/R contract already pins).
    """
    n1, n2 = spec["amb_sup"], spec["layers"][0][1]
    n3 = spec["amb_sub"]
    d = float(spec["layers"][0][0])
    k0 = 2 * np.pi / float(spec["wavelength"])
    n1, n2, n3 = float(n1), np.sqrt(float(n2)), float(n3)
    kz2 = n2 * k0
    r12 = (n1 - n2) / (n1 + n2)
    r23 = (n2 - n3) / (n2 + n3)
    r21 = -r12
    t12 = 2 * n1 / (n1 + n2)
    ph = np.exp(2j * kz2 * d)
    # incident superstrate field |E_inc|^2 = |t01|^2 = 1 (unit plane wave in vacuum)
    z = d / 2.0
    # forward/backward inside film at z, relative to a unit incident field
    Ex = t12 * (np.exp(1j * kz2 * z) + r23 * np.exp(1j * kz2 * (2 * d - z))) / (1 - r21 * r23 * ph)
    return np.asarray(float(np.real(Ex * np.conj(Ex))), dtype=np.float64)


def regen_one(name: str):
    spec = _cases.CASES[name]
    formulation = spec.get("formulation")
    out: dict = {}

    if "source" in spec:
        # ---- internal emitter: fmmax amplitudes_for_source up/down powers ----
        wavelength = spec["wavelength"]
        period = spec["period"]
        kin = spec["k_in"]
        lv, ex = _expansion(period, spec["orders"])
        layers = [
            _ambient_layer(wavelength, kin, lv, spec["amb_sup"], ex, _formulation(formulation)),
            _layer_solve(wavelength, kin, lv, spec["layers"][0][1], ex, _formulation(formulation)),
            _ambient_layer(wavelength, kin, lv, spec["amb_sub"], ex, _formulation(formulation)),
        ]
        src = spec["source"]
        before, before_t = layers[:2], [1.0, src["t_upper"]]
        after, after_t = layers[1:], [src["t_lower"], 1.0]
        s_before = scattering.stack_s_matrix(before, before_t)
        s_after = scattering.stack_s_matrix(after, after_t)
        loc = jnp.asarray(src["loc"])
        base = (
            sources.gaussian_source(jnp.asarray(src["fwhm"]), loc, jnp.asarray(kin), lv, ex)
            if src["kind"] == "gaussian"
            else sources.dirac_delta_source(loc, jnp.asarray(kin), lv, ex)
        )
        ox, oy, oz = src["orient"]
        amp = jnp.asarray(src["amp"]) + 0j
        jx, jy, jz = amp * ox * base, amp * oy * base, amp * oz * base
        bwd0, _, _, _, _, fN = sources.amplitudes_for_source(jx, jy, jz, s_before, s_after)
        _, ub = fmmax_fields.directional_poynting_flux(jnp.zeros_like(bwd0), bwd0, layers[0])
        df, _ = fmmax_fields.directional_poynting_flux(fN, jnp.zeros_like(fN), layers[-1])
        out["up_ref"] = float(jnp.abs(jnp.sum(jnp.real(ub))))
        out["down_ref"] = float(jnp.abs(jnp.sum(jnp.real(df))))
    else:
        # ---- plane-wave stack: T/R (+ slab |E|^2) ----
        ref = _plane_wave_tref(spec, formulation)
        out["t_ref"] = ref["t_ref"]
        out["r_ref"] = ref["r_ref"]
        out["period"] = np.asarray(spec["period"], dtype=np.float64)
        if name == "slab_airy" and spec.get("e2_planes"):
            out["e2_ref"] = _e2_slab(spec, ref["_expansion"], ref["_layers"], nx=spec.get("e2_nx", 8))

    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="case names; default all")
    args = ap.parse_args()

    names = args.names or list(_cases.CASES)
    manifest = {}
    for name in names:
        print(f"[golden] {name} ...")
        data = regen_one(name)
        npz_path = os.path.join(HERE, f"{name}.npz")
        np.savez(npz_path, **data)
        manifest[name] = {
            "sha256": _sha256(npz_path),
            "description": _cases.CASES[name]["description"],
            "dtype": _cases.CASES[name]["dtype"],
        }
        print(f"  wrote {name}.npz  keys={sorted(data)}")

    manifest_path = os.path.join(HERE, "manifest.json")
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"[golden] manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
